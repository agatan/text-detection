import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from icecream import ic

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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tensor = torch.Tensor(image.astype(np.float32) / 255.)
    return normalize(tensor)


def softmax_links(link_pred: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(link_pred)
    for i in range(8):
        softmaxed = link_pred[2 * i:2 * (i + 1), :, :].softmax(dim=0)
        out[2 * i:2 * (i + 1), :, :] = softmaxed
    return out


def predict(pixellink: net.MobileNetV2PixelLink, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pixel_preds, link_preds, _ = pixellink(image.unsqueeze(0))
    pixel_pred = pixel_preds.squeeze(0)
    link_pred = link_preds.squeeze(0)
    pixel_pred = pixel_pred.softmax(dim=0)
    for i in range(8):
        link_pred[2 * i:2 * (i + 1)] = link_pred[2 * i:2 * (i + 1)].softmax(0)
    return pixel_pred, link_pred


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mask-thres", type=float, default=0.5)
    parser.add_argument("--link-thres", type=float, default=0.5)
    args = parser.parse_args()

    pixellink = torch.load(args.model, map_location=lambda storage, loc: storage)
    pixellink.eval()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    resized_image, height_ratio, width_ratio = resize_image(image)
    image_tensor = preprocess_image(resized_image)

    with torch.no_grad():
        start = time.time()
        pixel_pred, link_pred = predict(pixellink, image_tensor)
        pixel_pred = pixel_pred.transpose(0, 1).transpose(1, 2).cpu().numpy()
        link_pred = link_pred.transpose(0, 1).transpose(1, 2).cpu().numpy()
        shrink_ratio = 2
        pixel_pred = cv2.resize(pixel_pred, (image.shape[1] // shrink_ratio, image.shape[0] // shrink_ratio))
        link_pred = cv2.resize(link_pred, (image.shape[1] // shrink_ratio, image.shape[0] // shrink_ratio))
        instance_map = postprocess.mask_to_instance_map(pixel_pred, link_pred, args.mask_thres, args.link_thres)
        instance_map = cv2.resize(instance_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        bounding_boxes = postprocess.instance_map_to_bboxes(instance_map)
        print("Inference + Postprocess: {:.4f}s".format(time.time() - start))

        pixel_pred = cv2.resize(pixel_pred, (image.shape[1], image.shape[0]))
        link_pred = cv2.resize(link_pred, (image.shape[1], image.shape[0]))
        pixel_mask = np.argmax(pixel_pred, axis=2)
        link_probabilities = []
        for i in range(8):
            link_probability = link_pred[:, :, 2 * i + 1]
            link_probability = link_probability * pixel_mask.astype(np.float32)
            link_probability = (link_probability * 255).astype(np.uint8)
            link_probabilities.append(link_probability)
        pixel_probability = (pixel_pred[:, :, 1] * 255).astype(np.uint8)

    images = [
        np.concatenate([link_probabilities[0], link_probabilities[1], link_probabilities[2]], axis=1),
        np.concatenate([link_probabilities[3], pixel_probability, link_probabilities[4]], axis=1),
        np.concatenate([link_probabilities[5], link_probabilities[6], link_probabilities[7]], axis=1),
    ]
    concat_image = np.concatenate(images, axis=0)
    cv2.imwrite("map.png", cv2.applyColorMap(concat_image, cv2.COLORMAP_JET))
    binarized_image = (concat_image > 255 / 2).astype(np.uint8) * 255
    cv2.imwrite("binmap.png", cv2.applyColorMap(binarized_image, cv2.COLORMAP_JET))

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
        xy_box = np.zeros_like(box)
        xy_box[:, 0] = box[:, 1]
        xy_box[:, 1] = box[:, 0]
        xy_box = xy_box.astype(np.int32)
        image = cv2.drawContours(image, [xy_box], 0, (0, 0, 255), thickness=2)
    cv2.imwrite("predict.png", image)


if __name__ == '__main__':
    main()
