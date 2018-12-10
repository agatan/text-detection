import os
import time

import torch
import cv2
import numpy as np

import util


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image_height, image_width, _ = image.shape
    if image_height > image_width and height > width:
        return cv2.resize(image, (width, height))
    if image_width > image_height and width > height:
        return cv2.resize(image, (width, height))
    image = util.rotate(image, 90)
    return cv2.resize(image, (width, height))


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    t = np.transpose(image, (2, 0, 1))
    t = t.astype(np.float32) / 255.
    return torch.Tensor(t)


def construct_boxes(distance_map: np.ndarray) -> np.ndarray:
    _, height, width = distance_map.shape
    out = np.zeros_like(distance_map)
    # top
    out[0, :, :] = np.expand_dims(np.arange(0, height), axis=1) - distance_map[0, :, :]
    # left
    out[1, :, :] = np.expand_dims(np.arange(0, width), axis=0) - distance_map[1, :, :]
    # bottom
    out[2, :, :] = np.expand_dims(np.arange(0, height), axis=1) + distance_map[2, :, :]
    # right
    out[3, :, :] = np.expand_dims(np.arange(0, width), axis=0) + distance_map[3, :, :]
    return out


def nms(scores: np.ndarray, boxes: np.ndarray, thres: float = 0.5) -> np.ndarray:
    """Non maximum suppression.

    Args:
        scores: confidences scores for each box. [N]
        boxes: coordinates of boxes. [4 (xmin, ymin, xmax, ymax), N]
        thres: threshold of IoU to determine whether boxes will be merged.
    Returns:
        keep: indices to keep.
    """
    xmin = boxes[0, :]
    ymin = boxes[1, :]
    xmax = boxes[2, :]
    ymax = boxes[3, :]
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(xmin[i], xmin[order[1:]])
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thres)[0]
        order = order[inds + 1]
    return np.array(keep)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("image")
    args = parser.parse_args()

    model = torch.load(args.model, map_location='cpu')
    model.eval()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    image = resize_image(image, 640, 384)
    image_tensor = image_to_tensor(image).unsqueeze_(0)

    with torch.no_grad():
        start = time.time()
        mask_map, distance_map = model(image_tensor)
        mask_map = mask_map.squeeze(0).softmax(dim=0).cpu().detach().numpy()
        distance_map = distance_map.squeeze(0).cpu().detach().numpy()
        boxes = construct_boxes(distance_map) * model.scale
        print("Inference: ", time.time() - start, "[s]")
        argmax = np.argmax(mask_map, axis=0)
        scores = np.reshape(mask_map[1, :, :], -1)
        boxes = np.reshape(boxes, (4, -1))
        confident_indices = scores > 0.5
        scores = scores[confident_indices]
        boxes = boxes[:, confident_indices]
        keep = nms(scores, boxes)
        print("Inference + NMS: ", time.time() - start, "[s]")
        for k in keep:
            top, left, bottom, right = boxes[:, k]
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), thickness=1)
        cv2.imwrite("out.png", image)


main()
