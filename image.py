import os

import torch
from PIL import Image


def tensor_for_board(img_tensor):
    tensor = (img_tensor.clone() + 1) * 0.5
    tensor = tensor.cpu().clamp(0, 1)

    if tensor.size(1) == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    return tensor


def tensor_list_for_board(img_tensors_list):
    grid_rows = len(img_tensors_list)
    grid_cols = max(
        len(img_tensors)
        for img_tensors in img_tensors_list
    )

    batch_size, channels, height, width = tensor_for_board(
        img_tensors_list[0][0]
    ).size()

    canvas = torch.full(
        (
            batch_size,
            channels,
            grid_rows * height,
            grid_cols * width,
        ),
        fill_value=0.5,
    )

    for row_idx, img_tensors in enumerate(img_tensors_list):
        for col_idx, img_tensor in enumerate(img_tensors):
            row_start = row_idx * height
            col_start = col_idx * width

            tensor = tensor_for_board(img_tensor)

            canvas[
                :,
                :,
                row_start : row_start + height,
                col_start : col_start + width,
            ].copy_(tensor)

    return canvas


def board_add_image(board, tag_name, img_tensor, step_count):
    tensor = tensor_for_board(img_tensor)

    for idx, image in enumerate(tensor):
        board.add_image(
            f"{tag_name}/{idx:03d}",
            image,
            step_count,
        )


def board_add_images(board, tag_name, img_tensors_list, step_count):
    tensor = tensor_list_for_board(img_tensors_list)

    for idx, image in enumerate(tensor):
        board.add_image(
            f"{tag_name}/{idx:03d}",
            image,
            step_count,
        )


def save_images(img_tensors, img_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        array = tensor.numpy().astype("uint8")

        if array.shape[0] == 1:
            array = array.squeeze(0)

        elif array.shape[0] == 3:
            array = array.transpose(1, 2, 0)

        save_path = os.path.join(save_dir, img_name)

        Image.fromarray(array).save(
            save_path,
            format="PNG",
        )