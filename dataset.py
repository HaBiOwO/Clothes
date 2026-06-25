# coding=utf-8
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from image import save_images


BATCH_SIZE = 4
WORKER = 16
WIDTH = 192
HEIGHT = 256

DATA_ROOT = "./data"
GRID_IMAGE_PATH = "grid.png"


class Dataset(data.Dataset):
    """Dataset for CP-VTON."""

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.root = DATA_ROOT
        self.datamode = opt.datamode
        self.stage = opt.stage
        self.data_list = opt.data_list

        self.fine_height = HEIGHT
        self.fine_width = WIDTH
        self.data_path = osp.join(self.root, self.datamode)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        self.im_names, self.c_names = self._load_pairs()

    def name(self):
        return "Dataset"

    def _load_pairs(self):
        im_names = []
        c_names = []

        pair_path = osp.join(self.root, self.data_list)

        with open(pair_path, "r") as file:
            for line in file:
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        return im_names, c_names

    def _load_rgb_image(self, *paths):
        image_path = osp.join(*paths)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def _load_cloth(self, cloth_name):
        if self.stage == "GMM":
            cloth_dir = "cloth"
            mask_dir = "cloth-mask"
        else:
            cloth_dir = "warp-cloth"
            mask_dir = "warp-mask"

        cloth = self._load_rgb_image(
            self.data_path,
            cloth_dir,
            cloth_name,
        )

        mask = Image.open(
            osp.join(self.data_path, mask_dir, cloth_name)
        )

        mask = self._binarize_mask(mask)

        return cloth, mask

    @staticmethod
    def _binarize_mask(mask):
        mask_array = np.array(mask)
        mask_array = (mask_array >= 128).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_array)
        return mask_tensor.unsqueeze(0)

    def _load_parse_maps(self, image_name):
        parse_name = image_name.replace(".jpg", ".png")
        parse_path = osp.join(
            self.data_path,
            "image-parse",
            parse_name,
        )

        parse_image = Image.open(parse_path)
        parse_array = np.array(parse_image)

        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (
            (parse_array == 1).astype(np.float32)
            + (parse_array == 2).astype(np.float32)
            + (parse_array == 4).astype(np.float32)
            + (parse_array == 13).astype(np.float32)
        )

        parse_cloth = (
            (parse_array == 5).astype(np.float32)
            + (parse_array == 6).astype(np.float32)
            + (parse_array == 7).astype(np.float32)
        )

        return parse_shape, parse_head, parse_cloth

    def _make_shape_tensor(self, parse_shape):
        shape_image = Image.fromarray(
            (parse_shape * 255).astype(np.uint8)
        )

        shape_image = shape_image.resize(
            (
                self.fine_width // 16,
                self.fine_height // 16,
            ),
            Image.BILINEAR,
        )

        shape_image = shape_image.resize(
            (
                self.fine_width,
                self.fine_height,
            ),
            Image.BILINEAR,
        )

        shape_image = shape_image.convert("RGB")

        return self.transform(shape_image)

    def _load_pose_data(self, image_name):
        pose_name = image_name.replace(".jpg", "_keypoints.json")
        pose_path = osp.join(
            self.data_path,
            "pose",
            pose_name,
        )

        with open(pose_path, "r") as file:
            pose_label = json.load(file)

        pose_data = pose_label["people"][0]["pose_keypoints"]
        pose_data = np.array(pose_data)

        return pose_data.reshape((-1, 3))

    def _make_pose_map(self, pose_data):
        point_num = pose_data.shape[0]

        pose_map = torch.zeros(
            point_num,
            self.fine_height,
            self.fine_width,
        )

        pose_image = Image.new(
            "L",
            (
                self.fine_width,
                self.fine_height,
            ),
        )

        pose_draw = ImageDraw.Draw(pose_image)
        radius = 5

        for point_id in range(point_num):
            point_x = pose_data[point_id, 0]
            point_y = pose_data[point_id, 1]

            single_pose = Image.new(
                "L",
                (
                    self.fine_width,
                    self.fine_height,
                ),
            )
            draw = ImageDraw.Draw(single_pose)
            if point_x > 1 and point_y > 1:
                box = (
                    point_x - radius,
                    point_y - radius,
                    point_x + radius,
                    point_y + radius,
                )
                draw.rectangle(box, fill="white", outline="white")
                pose_draw.rectangle(box, fill="white", outline="white")
            single_pose = single_pose.convert("RGB")
            single_pose = self.transform(single_pose)
            pose_map[point_id] = single_pose[0]

        pose_image = pose_image.convert("RGB")
        pose_image = self.transform(pose_image)

        return pose_map, pose_image

    def _load_grid_image(self):
        if self.stage != "GMM":
            return ""

        grid_image = Image.open(GRID_IMAGE_PATH).convert("RGB")
        return self.transform(grid_image)

    def __getitem__(self, index):
        cloth_name = self.c_names[index]
        image_name = self.im_names[index]
        cloth, cloth_mask = self._load_cloth(cloth_name)
        image = self._load_rgb_image(
            self.data_path,
            "image",
            image_name,
        )
        parse_shape, parse_head, parse_cloth = self._load_parse_maps(
            image_name
        )
        shape = self._make_shape_tensor(parse_shape)
        head_mask = torch.from_numpy(parse_head)
        cloth_parse_mask = torch.from_numpy(parse_cloth)
        parse_cloth_image = image * cloth_parse_mask + (1 - cloth_parse_mask)
        head_image = image * head_mask - (1 - head_mask)

        pose_data = self._load_pose_data(image_name)
        pose_map, pose_image = self._make_pose_map(pose_data)
        save_images(pose_image, "test.png", "./result/pose")
        agnostic = torch.cat(
            [
                shape,
                head_image,
                pose_map,
            ],
            dim=0,
        )
        grid_image = self._load_grid_image()
        return {
            "c_name": cloth_name,
            "im_name": image_name,
            "cloth": cloth,
            "cloth_mask": cloth_mask,
            "image": image,
            "agnostic": agnostic,
            "parse_cloth": parse_cloth_image,
            "shape": shape,
            "head": head_image,
            "pose_image": pose_image,
            "grid_image": grid_image,
        }
    def __len__(self):
        return len(self.im_names)


class DataLoader:
    def __init__(self, opt, dataset):
        super().__init__()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(sampler is None),
            num_workers=WORKER,
            pin_memory=True,
            sampler=sampler,
        )
        self.data_iter = iter(self.data_loader)

    def next_batch(self):
        try:
            return next(self.data_iter)

        except StopIteration:
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    return parser.parse_args()

if __name__ == "__main__":
    opt = get_opt()

    dataset = Dataset(opt)
    data_loader = DataLoader(opt, dataset)

    first_item = dataset[0]
    first_batch = data_loader.next_batch()

    print("First item keys:", first_item.keys())
    print("Dataset size:", len(dataset))