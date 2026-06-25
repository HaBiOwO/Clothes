# coding=utf-8
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataset import Dataset, DataLoader
from networks import GMM, UnetGenerator, load_checkpoint
from image import board_add_images, save_images


DISPLAY_COUNT = 100

TENSOR_DIR = "tensorboard"
RESULT_DIR = "result"


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--checkpoint", type=str, default="", help="model for test")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_cuda(inputs, keys):
    return {
        key: inputs[key].cuda()
        for key in keys
    }


def create_tensorboard_writer():
    ensure_dir(TENSOR_DIR)
    return SummaryWriter(
        log_dir=os.path.join(TENSOR_DIR, "tensorboard")
    )


def get_result_dir(opt):
    checkpoint_name = os.path.basename(opt.checkpoint)
    return ensure_dir(
        os.path.join(
            RESULT_DIR,
            checkpoint_name,
            opt.datamode,
        )
    )


def create_model(opt):
    if opt.stage == "GMM":
        return GMM(opt)
    if opt.stage == "TOM":
        return UnetGenerator(
            input_nc=27,
            output_nc=4,
            num_downs=6,
            feature_num=64,
            norm_layer=nn.InstanceNorm2d,
        )

    raise NotImplementedError(
        f"Model [{opt.stage}] is not implemented"
    )


def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    save_dir = get_result_dir(opt)
    warp_cloth_dir = ensure_dir(os.path.join(save_dir, "warp-cloth"))
    warp_mask_dir = ensure_dir(os.path.join(save_dir, "warp-mask"))
    grid_dir = ensure_dir(os.path.join(save_dir, "grid"))

    print(f"Dataset size: {len(test_loader.dataset):05d}!", flush=True)

    for step, inputs in enumerate(test_loader.data_loader):
        current_step = step + 1
        start_time = time.time()
        cloth_names = inputs["c_name"]
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
        grid, theta = model(
            data["agnostic"],
            data["cloth"],
        )
        warped_cloth = F.grid_sample(
            data["cloth"],
            grid,
            padding_mode="border",
        )
        warped_mask = F.grid_sample(
            data["cloth_mask"],
            grid,
            padding_mode="zeros",
        )
        warped_grid = F.grid_sample(
            data["grid_image"],
            grid,
            padding_mode="zeros",
        )
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
        save_images(
            warped_cloth,
            cloth_names,
            warp_cloth_dir,
        )
        save_images(
            warped_mask * 2 - 1,
            cloth_names,
            warp_mask_dir,
        )
        if current_step % DISPLAY_COUNT == 0:
            board_add_images(
                board,
                "combine",
                visuals,
                current_step,
            )
            save_images(
                warped_grid,
                cloth_names,
                grid_dir,
            )
            elapsed_time = time.time() - start_time
            print(
                f"step: {current_step:8d}, time: {elapsed_time:.3f}",
                flush=True,
            )


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    save_dir = get_result_dir(opt)
    try_on_dir = ensure_dir(os.path.join(save_dir, "try-on"))
    composite_dir = ensure_dir(os.path.join(save_dir, "composite"))
    print(f"Dataset size: {len(test_loader.dataset):05d}!", flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        current_step = step + 1
        start_time = time.time()
        image_names = inputs["im_name"]
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
        visuals = [
            [
                data["head"],
                data["shape"],
                data["pose_image"],
            ],
            [
                data["cloth"],
                data["cloth_mask"] * 2 - 1,
                m_composite,
            ],
            [
                p_rendered,
                p_tryon,
                data["image"],
            ],
        ]
        save_images(
            m_composite,
            image_names,
            composite_dir,
        )
        save_images(
            p_tryon,
            image_names,
            try_on_dir,
        )
        if current_step % DISPLAY_COUNT == 0:
            board_add_images(
                board,
                "combine",
                visuals,
                current_step,
            )
            elapsed_time = time.time() - start_time
            print(
                f"step: {current_step:8d}, time: {elapsed_time:.3f}",
                flush=True,
            )

def run_test(opt, test_loader, model, board):
    if opt.stage == "GMM":
        test_gmm(opt, test_loader, model, board)
        return

    if opt.stage == "TOM":
        test_tom(opt, test_loader, model, board)
        return

    raise NotImplementedError(
        f"Model [{opt.stage}] is not implemented"
    )


def main():
    opt = get_opt()

    print(opt)
    print(f"Start to test stage: {opt.stage}!")

    test_dataset = Dataset(opt)
    test_loader = DataLoader(opt, test_dataset)

    board = create_tensorboard_writer()

    model = create_model(opt)
    load_checkpoint(model, opt.checkpoint)

    with torch.no_grad():
        run_test(
            opt=opt,
            test_loader=test_loader,
            model=model,
            board=board,
        )

    board.close()

    print(f"Finished test {opt.stage}!")
if __name__ == "__main__":
    main()