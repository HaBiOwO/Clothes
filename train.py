# coding=utf-8
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import Dataset, DataLoader
from networks import GMM, UnetGenerator, VGGLoss, save_checkpoint
from image import board_add_images


DISPLAY_COUNT = 10000
SAVE_COUNT = 10000
LR = 1e-4
KEEP_STEP = 100000
DECAY_STEP = 100000
TOTAL_STEPS = KEEP_STEP + DECAY_STEP

TENSOR_DIR = "tensorboard"
CHECKPOINT_DIR = "checkpoints"


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    return parser.parse_args()


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0
        - max(0, step - KEEP_STEP) / float(DECAY_STEP + 1),
    )


def to_cuda(inputs, keys):
    return {
        key: inputs[key].cuda()
        for key in keys
    }


def save_stage_checkpoint(model, stage, step):
    save_path = os.path.join(
        CHECKPOINT_DIR,
        stage,
        f"step_{step:06d}.pth",
    )
    save_checkpoint(model, save_path)


def log_training(board, visuals, losses, step):
    board_add_images(board, "combine", visuals, step)

    for name, value in losses.items():
        board.add_scalar(name, value.item(), step)


def train_gmm(train_loader, model, board):
    model.cuda()
    model.train()

    criterion_l1 = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.5, 0.999),
    )

    scheduler = get_scheduler(optimizer)

    pbar = tqdm(range(TOTAL_STEPS), desc="Training GMM")

    for step in pbar:
        current_step = step + 1
        start_time = time.time()
        inputs = train_loader.next_batch()
        data = to_cuda(
            inputs,
            [
                "image",
                "pose_image",
                "head",
                "shape",
                "agnostic",
                "cloth",
                "cloth_mask",
                "parse_cloth",
                "grid_image",
            ],
        )

        grid, theta = model(data["agnostic"], data["cloth"],)
        warped_cloth = F.grid_sample(data["cloth"], grid, padding_mode="border",)
        warped_mask = F.grid_sample(data["cloth_mask"], grid, padding_mode="zeros",)
        warped_grid = F.grid_sample(data["grid_image"], grid, padding_mode="zeros",)
        loss = criterion_l1(warped_cloth, data["parse_cloth"],)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        visuals = [
            [
                data["head"],
                data["shape"],
                data["pose_image"],
            ],
            [
                data["cloth"],
                warped_cloth,
                data["parse_cloth"],
            ],
            [
                warped_grid,
                (warped_cloth + data["image"]) * 0.5,
                data["image"],
            ],
        ]

        if current_step % DISPLAY_COUNT == 0:
            log_training(
                board,
                visuals,
                {"metric": loss},
                current_step,
            )

            elapsed_time = time.time() - start_time
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Time/step": f"{elapsed_time:.3f}s",
                }
            )

        if current_step % SAVE_COUNT == 0:
            save_stage_checkpoint(
                model,
                "GMM",
                current_step,
            )


def train_tom(train_loader, model, board):
    model.cuda()
    model.train()

    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGLoss()
    criterion_mask = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.5, 0.999),
    )

    scheduler = get_scheduler(optimizer)

    pbar = tqdm(range(TOTAL_STEPS), desc="Training TOM")

    for step in pbar:
        current_step = step + 1
        start_time = time.time()

        inputs = train_loader.next_batch()

        data = to_cuda(
            inputs,
            [
                "image",
                "pose_image",
                "head",
                "shape",
                "agnostic",
                "cloth",
                "cloth_mask",
            ],
        )

        model_input = torch.cat(
            [
                data["agnostic"],
                data["cloth"],
            ],
            dim=1,
        )

        outputs = model(model_input)

        p_rendered, m_composite = torch.split(
            outputs,
            3,
            dim=1,
        )

        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)

        p_tryon = (
            data["cloth"] * m_composite
            + p_rendered * (1 - m_composite)
        )

        loss_l1 = criterion_l1(
            p_tryon,
            data["image"],
        )

        loss_vgg = criterion_vgg(
            p_tryon,
            data["image"],
        )

        loss_mask = criterion_mask(
            m_composite,
            data["cloth_mask"],
        )

        loss = loss_l1 + loss_vgg + loss_mask

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        visuals = [
            [
                data["head"],
                data["shape"],
                data["pose_image"],
            ],
            [
                data["cloth"],
                data["cloth_mask"] * 2 - 1,
                m_composite * 2 - 1,
            ],
            [
                p_rendered,
                p_tryon,
                data["image"],
            ],
        ]

        if current_step % DISPLAY_COUNT == 0:
            log_training(
                board,
                visuals,
                {
                    "metric": loss,
                    "L1": loss_l1,
                    "VGG": loss_vgg,
                    "MaskL1": loss_mask,
                },
                current_step,
            )

            elapsed_time = time.time() - start_time
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Time/step": f"{elapsed_time:.3f}s",
                }
            )

        if current_step % SAVE_COUNT == 0:
            save_stage_checkpoint(
                model,
                "TOM",
                current_step,
            )


def create_tensorboard_writer():
    os.makedirs(TENSOR_DIR, exist_ok=True)

    return SummaryWriter(
        log_dir=os.path.join(
            TENSOR_DIR,
            "tensorboard",
        )
    )


def create_model(stage, opt):
    if stage == "GMM":
        return GMM(opt)

    if stage == "TOM":
        return UnetGenerator(
            input_nc=27,
            output_nc=4,
            num_downs=6,
            feature_num=64,
            norm_layer=nn.InstanceNorm2d,
        )

    raise NotImplementedError(
        f"Model [{stage}] is not implemented"
    )


def train_stage(stage, train_loader, model, board):
    if stage == "GMM":
        train_gmm(train_loader, model, board)
        final_name = "gmm_final.pth"

    elif stage == "TOM":
        train_tom(train_loader, model, board)
        final_name = "tom_final.pth"

    else:
        raise NotImplementedError(
            f"Model [{stage}] is not implemented"
        )

    final_path = os.path.join(
        CHECKPOINT_DIR,
        stage,
        final_name,
    )

    save_checkpoint(model, final_path)


def main():
    opt = get_opt()

    print(opt)
    print(f"Start to train stage: {opt.stage}!")

    train_dataset = Dataset(opt)
    train_loader = DataLoader(opt, train_dataset)

    board = create_tensorboard_writer()

    model = create_model(opt.stage, opt)

    train_stage(
        stage=opt.stage,
        train_loader=train_loader,
        model=model,
        board=board,
    )

    board.close()

    print(f"Finished training {opt.stage}!")


if __name__ == "__main__":
    main()