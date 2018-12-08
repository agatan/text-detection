from typing import Union, Tuple
from pathlib import Path
import copy

import torch
import torch.utils.data as data
import cv2
import numpy as np


def _compute_min_rect_and_angle(contours: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returns:
        min_rect_contours: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        angle: radian
    """
    print(contours)
    print(cv2.minAreaRect(contours))
    (left, top), (width, height), angle = cv2.minAreaRect(contours)
    box = cv2.boxPoints(((left, top), (width, height), angle))
    print(box)
    return box


class Dataset(data.Dataset):
    def __init__(self, image_dir: Union[str, Path], label_dir: Union[str, Path],
                 image_size: Tuple[int, int] = (512, 512),
                 scale: int = 4):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.labels = self._read_labels()
        self.image_size = image_size
        self.scale = scale

    def _read_labels(self):
        labels = []
        index = 1
        while True:
            filename = self.label_dir / "gt_img_{}.txt".format(index)
            if not filename.exists():
                break
            label = {
                "text": [],
                "ignored": [],
                "points": [],
            }
            with open(str(filename), "r", encoding="utf-8_sig") as fp:
                for line in fp:
                    fields = line.split(",")
                    fields = fields[1:]
                    points = [[int(fields[2 * i]), int(fields[2 * i + 1])] for i in range(0, 4)]
                    label["points"].append(points)
                    text = fields[8].strip()
                    label["text"].append(text)
                    label["ignored"].append(text == "###")
            labels.append(label)
            index += 1
        return labels

    def __len__(self):
        return len(self.labels)

    def _read_image(self, index: int):
        filename = str(self.image_dir / "img_{}.jpg".format(index + 1))
        return cv2.imread(filename, cv2.IMREAD_COLOR)

    def __getitem__(self, index: int):
        image = self._read_image(index)
        label = self.labels[index]
        image, label = self._resize_image_with_labels(image, label)
        image = np.transpose(image, (2, 0, 1)).astype(np.float) / 255.
        mask_map, distance_map = self._mask_and_distances(label)
        return torch.FloatTensor(image), torch.LongTensor(mask_map), torch.FloatTensor(distance_map)

    def _resize_image_with_labels(self, image, labels):
        labels = copy.deepcopy(labels)
        height, width = self.image_size
        original_height, original_width, _ = image.shape
        resized_image = cv2.resize(image, (width, height))
        points = np.array(labels["points"])
        points[:, :, 0] = points[:, :, 0] * width / original_width
        points[:, :, 1] = points[:, :, 1] * height / original_height
        labels["points"] = points.tolist()
        return resized_image, labels

    def _mask_and_distances(self, label):
        map_size = (self.image_size[0] // self.scale, self.image_size[1] // self.scale)
        mask_map = np.zeros(map_size, np.uint8)
        # top, left, bottom, right
        distance_map = np.zeros((4,) + map_size, np.float)
        for points in label["points"]:
            xmin = min((p[0] for p in points))
            ymin = min((p[1] for p in points))
            xmax = max((p[0] for p in points))
            ymax = max((p[1] for p in points))
            x_r = (xmax - xmin) * 0.3
            x_from = round((xmin + x_r) / self.scale)
            x_to = round((xmax - x_r) / self.scale)
            y_r = (ymax - ymin) * 0.3
            y_from = round((ymin + y_r) / self.scale)
            y_to = round((ymax - y_r) / self.scale)
            mask_map[y_from:y_to + 1, x_from:x_to + 1] = 1
            # top
            distance_map[0, y_from:y_to + 1, x_from:x_to + 1] = np.expand_dims(np.arange(y_from, y_to + 1).astype(np.float32) - ymin / self.scale, axis=-1)
            # bottom
            distance_map[2, y_from:y_to + 1, x_from:x_to + 1] = np.expand_dims(ymax / self.scale - np.arange(y_from, y_to + 1).astype(np.float32), axis=-1)
            # left
            distance_map[1, y_from:y_to + 1, x_from:x_to + 1] = np.expand_dims(np.arange(x_from, x_to + 1).astype(np.float32) - xmin / self.scale, axis=0)
            # right
            distance_map[3, y_from:y_to + 1, x_from:x_to + 1] = np.expand_dims(xmax / self.scale - np.arange(x_from, x_to + 1).astype(np.float32), axis=-0)
        return mask_map, distance_map

