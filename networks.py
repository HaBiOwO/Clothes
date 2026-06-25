#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import Flatten
import torch.nn.functional as F
import os

import numpy as np

WIDTH = 192
HEIGHT = 256

def weights_init_normal(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.normal_(module.weight, mean=0.0, std=0.02)

        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight, mean=1.0, std=0.02)
        init.constant_(module.bias, 0)


def weights_init_xavier(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.xavier_normal_(module.weight, gain=0.02)

        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight, mean=1.0, std=0.02)
        init.constant_(module.bias, 0)


def weights_init_kaiming(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(module.weight, a=0, mode="fan_in")

        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight, mean=1.0, std=0.02)
        init.constant_(module.bias, 0)


def init_weights(model, init_type="normal"):
    initializers = {
        "normal": weights_init_normal,
        "xavier": weights_init_xavier,
        "kaiming": weights_init_kaiming,
    }

    if init_type not in initializers:
        raise ValueError(f"Unsupported initialization: {init_type}")

    model.apply(initializers[init_type])

class Extraction(nn.Module):
    def __init__(self, input_channel, output_channel=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()

        MAX_CHANNEL = 512
        layers = []
        in_channels = input_channel
        out_channels = output_channel
        layers += self._conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm_layer=norm_layer)
        current_channels = output_channel

        for _ in range(n_layers):
            next_channels = min(current_channels * 2, MAX_CHANNEL)
            layers += self._conv(current_channels, next_channels, kernel_size=4, stride=2, padding=1, norm_layer=norm_layer)
            current_channels = next_channels

        layers += [
            nn.Conv2d(MAX_CHANNEL, MAX_CHANNEL, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            norm_layer(MAX_CHANNEL),

            nn.Conv2d(MAX_CHANNEL, MAX_CHANNEL, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]

        self.model = nn.Sequential(*layers)

        init_weights(self.model, init_type="normal")

    def _conv(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            norm_layer(out_channels),
        ]

    def forward(self, x):
        return self.model(x)

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1, eps=1e-6)
    
class Correlation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_A, feature_B):
        batch_size, channels, height, width = feature_A.size()
        num_locations = height * width

        feature_A = (
            feature_A
            .transpose(2, 3)
            .contiguous()
            .view(batch_size, channels, num_locations)
        )

        feature_B = (
            feature_B
            .view(batch_size, channels, num_locations)
            .transpose(1, 2)
        )

        correlation = torch.bmm(feature_B, feature_A)

        correlation = (
            correlation
            .view(batch_size, height, width, num_locations)
            .transpose(2, 3)
            .transpose(1, 2)
        )

        return correlation
    
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class Regression(nn.Module):
    def __init__(self, input_nc=512, output_dim=6):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNReLU(input_nc, 512, kernel_size=4, stride=2, padding=1),
            ConvBNReLU(512, 256, kernel_size=4, stride=2, padding=1),
            ConvBNReLU(256, 128, kernel_size=3, padding=1),
            ConvBNReLU(128, 64, kernel_size=3, padding=1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 3, output_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

class AffineGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, out_ch=3, align_corners=False):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        self.align_corners = align_corners

    def forward(self, theta):
        batch_size = theta.size(0)
        output_size = (batch_size, self.out_ch, self.out_h, self.out_w)
        return F.affine_grid(theta.contiguous(), size=output_size, align_corners=self.align_corners)
        
class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=5, reg_factor=0, use_cuda=True):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._create_sampling_grid()

        if use_regular_grid:
            self._create_control_points(grid_size)

    def _create_sampling_grid(self):
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1, 1, self.out_w),
            np.linspace(-1, 1, self.out_h),
        )

        self.grid_X = (
            torch.FloatTensor(grid_x)
            .unsqueeze(0)
            .unsqueeze(3)
            .to(self.device)
        )

        self.grid_Y = (
            torch.FloatTensor(grid_y)
            .unsqueeze(0)
            .unsqueeze(3)
            .to(self.device)
        )

    def _create_control_points(self, grid_size):
        axis_coords = np.linspace(-1, 1, grid_size)

        self.N = grid_size * grid_size

        control_y, control_x = np.meshgrid(axis_coords, axis_coords)

        control_x = torch.FloatTensor(control_x.reshape(-1, 1)).to(self.device)
        control_y = torch.FloatTensor(control_y.reshape(-1, 1)).to(self.device)

        self.P_X_base = control_x.clone()
        self.P_Y_base = control_y.clone()

        self.Li = self.compute_L_inverse(control_x, control_y).unsqueeze(0)

        self.P_X = (
            control_x
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .transpose(0, 4)
        )

        self.P_Y = (
            control_y
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .transpose(0, 4)
        )

    def forward(self, theta):
        points = torch.cat((self.grid_X, self.grid_Y), dim=3)
        return self.apply_transformation(theta, points)

    def compute_L_inverse(self, X, Y):
        num_points = X.size(0)
        X_matrix = X.expand(num_points, num_points)
        Y_matrix = Y.expand(num_points, num_points)
        dist_squared = (
            (X_matrix - X_matrix.transpose(0, 1)) ** 2
            + (Y_matrix - Y_matrix.transpose(0, 1)) ** 2
        )

        dist_squared[dist_squared == 0] = 1

        K = dist_squared * torch.log(dist_squared)

        ones = torch.ones(num_points, 1).to(self.device)
        zeros = torch.zeros(3, 3).to(self.device)

        P = torch.cat((ones, X, Y), dim=1)
        upper = torch.cat((K, P), dim=1)
        lower = torch.cat((P.transpose(0, 1), zeros), dim=1)
        L = torch.cat((upper, lower), dim=0)
        return torch.inverse(L)

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        batch_size = theta.size(0)

        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        points_b, points_h, points_w = points.size()[:3]

        P_X = self.P_X.expand(1, points_h, points_w, 1, self.N)
        P_Y = self.P_Y.expand(1, points_h, points_w, 1, self.N)

        Li_kernel = self.Li[:, :self.N, :self.N]
        Li_affine = self.Li[:, self.N:, :self.N]

        W_X = torch.bmm(
            Li_kernel.expand(batch_size, self.N, self.N),
            Q_X,
        )

        W_Y = torch.bmm(
            Li_kernel.expand(batch_size, self.N, self.N),
            Q_Y,
        )

        W_X = (
            W_X
            .unsqueeze(3)
            .unsqueeze(4)
            .transpose(1, 4)
            .repeat(1, points_h, points_w, 1, 1)
        )

        W_Y = (
            W_Y
            .unsqueeze(3)
            .unsqueeze(4)
            .transpose(1, 4)
            .repeat(1, points_h, points_w, 1, 1)
        )

        A_X = torch.bmm(
            Li_affine.expand(batch_size, 3, self.N),
            Q_X,
        )

        A_Y = torch.bmm(
            Li_affine.expand(batch_size, 3, self.N),
            Q_Y,
        )

        A_X = (
            A_X
            .unsqueeze(3)
            .unsqueeze(4)
            .transpose(1, 4)
            .repeat(1, points_h, points_w, 1, 1)
        )

        A_Y = (
            A_Y
            .unsqueeze(3)
            .unsqueeze(4)
            .transpose(1, 4)
            .repeat(1, points_h, points_w, 1, 1)
        )

        points_X = points[:, :, :, 0]
        points_Y = points[:, :, :, 1]

        points_X_for_sum = (
            points_X
            .unsqueeze(3)
            .unsqueeze(4)
            .expand(points_X.size() + (1, self.N))
        )

        points_Y_for_sum = (
            points_Y
            .unsqueeze(3)
            .unsqueeze(4)
            .expand(points_Y.size() + (1, self.N))
        )

        if points_b == 1:
            delta_X = points_X_for_sum - P_X
            delta_Y = points_Y_for_sum - P_Y
        else:
            delta_X = points_X_for_sum - P_X.expand_as(points_X_for_sum)
            delta_Y = points_Y_for_sum - P_Y.expand_as(points_Y_for_sum)

        dist_squared = delta_X ** 2 + delta_Y ** 2
        dist_squared[dist_squared == 0] = 1

        U = dist_squared * torch.log(dist_squared)

        points_X_batch = points_X.unsqueeze(3)
        points_Y_batch = points_Y.unsqueeze(3)

        if points_b == 1:
            points_X_batch = points_X_batch.expand(
                (batch_size,) + points_X_batch.size()[1:]
            )
            points_Y_batch = points_Y_batch.expand(
                (batch_size,) + points_Y_batch.size()[1:]
            )

        points_X_prime = (
            A_X[:, :, :, :, 0]
            + A_X[:, :, :, :, 1] * points_X_batch
            + A_X[:, :, :, :, 2] * points_Y_batch
            + torch.sum(W_X * U.expand_as(W_X), dim=4)
        )

        points_Y_prime = (
            A_Y[:, :, :, :, 0]
            + A_Y[:, :, :, :, 1] * points_X_batch
            + A_Y[:, :, :, :, 2] * points_Y_batch
            + torch.sum(W_Y * U.expand_as(W_Y), dim=4)
        )

        return torch.cat((points_X_prime, points_Y_prime), dim=3)
        
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__( self, input_nc, output_nc, num_downs, feature_num=64, norm_layer=nn.BatchNorm2d, use_dropout=False,):
        super().__init__()

        # Innermost block
        unet_block = UnetSkipConnectionBlock(
            outer_nc=feature_num * 8,
            inner_nc=feature_num * 8,
            norm_layer=norm_layer,
            innermost=True,
        )

        # Middle blocks
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                outer_nc=feature_num * 8,
                inner_nc=feature_num * 8,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )

        # Decoder / encoder transition blocks
        channel_pairs = [
            (feature_num * 4, feature_num * 8),
            (feature_num * 2, feature_num * 4),
            (feature_num, feature_num * 2),
        ]

        for outer_nc, inner_nc in channel_pairs:
            unet_block = UnetSkipConnectionBlock(outer_nc=outer_nc, inner_nc=inner_nc, submodule=unet_block, norm_layer=norm_layer,)

        # Outermost block
        self.model = UnetSkipConnectionBlock(outer_nc=output_nc, inner_nc=feature_num, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer,)

    def forward(self, x):
        return self.model(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__( self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        use_bias = norm_layer == nn.InstanceNorm2d

        downconv = self._down_conv(input_nc, inner_nc, use_bias)
        upsample = self._upsample()

        downrelu = nn.LeakyReLU(0.2, inplace=True)
        uprelu = nn.ReLU(inplace=True)

        downnorm = norm_layer(inner_nc)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = self._up_conv(inner_nc * 2, outer_nc, use_bias)
            model = [
                downconv,
                submodule,
                uprelu,
                upsample,
                upconv,
                upnorm,
            ]

        elif innermost:
            upconv = self._up_conv(inner_nc, outer_nc, use_bias)
            model = [
                downrelu,
                downconv,
                uprelu,
                upsample,
                upconv,
                upnorm,
            ]

        else:
            upconv = self._up_conv(inner_nc * 2, outer_nc, use_bias)
            model = [
                downrelu,
                downconv,
                downnorm,
                submodule,
                uprelu,
                upsample,
                upconv,
                upnorm,
            ]
            if use_dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*model)

    @staticmethod
    def _down_conv(in_channels, out_channels, use_bias):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias,
        )

    @staticmethod
    def _up_conv(in_channels, out_channels, use_bias):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
        )

    @staticmethod
    def _upsample():
        return nn.Upsample(
            scale_factor=2,
            mode="bilinear",
        )

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        vgg_features = models.vgg19(pretrained=True).features
        slice_ranges = [
            (0, 2),
            (2, 7),
            (7, 12),
            (12, 21),
            (21, 30),
        ]
        self.slices = nn.ModuleList([
            self._make_slice(vgg_features, start, end)
            for start, end in slice_ranges
        ])

        if not requires_grad:
            self._freeze_parameters()

    @staticmethod
    def _make_slice(features, start, end):
        return nn.Sequential(*[
            features[i]
            for i in range(start, end)
        ])

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = []
        for slice_module in self.slices:
            x = slice_module(x)
            outputs.append(x)
        return outputs

class VGGLoss(nn.Module):
    def __init__(self, layids=None):
        super().__init__()

        self.vgg = Vgg19()
        self.vgg.cuda()

        self.criterion = nn.L1Loss()
        self.weights = [
            1.0 / 32,
            1.0 / 16,
            1.0 / 8,
            1.0 / 4,
            1.0,
        ]

        self.layids = layids

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        if self.layids is None:
            layer_ids = range(len(x_features))
        else:
            layer_ids = self.layids
        loss = 0
        for layer_id in layer_ids:
            loss += (self.weights[layer_id] * self.criterion(x_features[layer_id], y_features[layer_id].detach(),))
        return loss

class GMM(nn.Module):
    """ Geometric Matching Module
    """
    def __init__(self, opt):
        super(GMM, self).__init__()
        self.extractionA = Extraction(24, 64, n_layers=3, norm_layer=nn.BatchNorm2d) 
        self.extractionB = Extraction(3, 64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.l2norm = L2Norm()
        self.correlation = Correlation()
        self.regression = Regression(input_nc=192, output_dim=2*5**2)
        self.GridGen = TpsGridGen(HEIGHT, WIDTH, use_cuda=True, grid_size=5)
        
    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)

        theta = self.regression(correlation)
        grid = self.GridGen(theta)
        return grid, theta

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()

