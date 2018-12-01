import os
from typing import Tuple

import torch
import torch.utils.data as data
# import matplotlib.pyplot as plt
import cv2
import numpy as np

from data import ICDAR15Dataset
import net
import postprocess


def resize_image(image: np.ndarray, size=(512, 512)) -> Tuple[np.ndarray, float, float]:
    height, width, _ = image.shape
    image = cv2.resize(image, size)
    height_ratio = height / float(size[1])
    width_ratio = width / float(size[0])
    return (image, height_ratio, width_ratio)


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = np.transpose(image, (2, 0, 1))
    return torch.Tensor(image.astype(np.float32) / 255.)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    scale = state_dict['scale']
    pixellink = net.MobileNetV2PixelLink(scale=scale)
    pixellink.load_state_dict(state_dict['pixellink'])
    pixellink.eval()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    resized_image, height_ratio, width_ratio = resize_image(image)
    image_tensor = preprocess_image(resized_image).unsqueeze(0)

    with torch.no_grad():
        pixel_pred, link_pred = pixellink(image_tensor)
        instance_map = postprocess.mask_to_instance_map(pixel_pred, link_pred)
        bounding_boxes = postprocess.instance_map_to_bboxes(instance_map, scale=scale)

    instance_map = instance_map[0]
    bounding_boxes = bounding_boxes[0]

    # colorize instance map.
    colored_instance_map = np.zeros(instance_map.shape + (3,), np.uint8)
    n_instances = np.max(instance_map)
    for n in range(1, n_instances + 1):
        colors = (np.random.random(3) * 255).astype(np.uint8)
        for y, x in zip(*np.where(instance_map == n)):
            colored_instance_map[y, x, :] = colors
    colored_instance_map = cv2.resize(colored_instance_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("instances.png", colored_instance_map)

    # draw boxes
    for box in bounding_boxes:
        box[:, 0] *= height_ratio
        box[:, 1] *= width_ratio
        xy_box = np.zeros_like(box)
        xy_box[:, 0] = box[:, 1]
        xy_box[:, 1] = box[:, 0]
        xy_box = xy_box.astype(np.int32)
        image = cv2.drawContours(image, [xy_box], 0, (0, 0, 255), thickness=2)
    cv2.imwrite("predict.png", image)


if __name__ == '__main__':
    main()
