from typing import Union, Tuple
from pathlib import Path
import copy
import math

import torch
import torch.utils.data as data
import cv2
import numpy as np

import util


class Dataset(data.Dataset):
    def __init__(self, image_dir: Union[str, Path], label_dir: Union[str, Path],
                 image_size: Tuple[int, int] = (512, 512),
                 scale: int = 4,
                 training: bool = False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.labels = self._read_labels()
        self.image_size = image_size
        self.scale = scale
        self.training = training

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
        if self.training:
            image, label = self._train_transform(image, label)
        else:
            image, label = self._test_transform(image, label)
        image = np.transpose(image, (2, 0, 1)).astype(np.float) / 255.
        mask_map, distance_map = self._mask_and_distances(label)
        return torch.FloatTensor(image), torch.LongTensor(mask_map), torch.FloatTensor(distance_map)

    def _train_transform(self, image, label):
        image, label = self._random_rotate_with_labels(image, label)
        image, label = self._rotate_image_to_wide_with_label(image, label)
        image, label = self._resize_image_with_labels(image, label)
        label = self._filter_small_labels(label)
        return image, label

    def _test_transform(self, image, label):
        image, label = self._resize_image_with_labels(image, label)
        image, label = self._rotate_image_to_wide_with_label(image, label)
        label = self._filter_small_labels(label)
        return image, label

    def _rotate_image_to_wide_with_label(self, image: np.ndarray, label: dict) -> Tuple[np.ndarray, dict]:
        height, width, _ = image.shape
        if height < width:
            return image, label
        return self._rotate_with_labels(image, label, 90)

    def _random_rotate_with_labels(self, image: np.ndarray, labels: dict) -> Tuple[np.ndarray, dict]:
        angle = np.random.randint(0, 4) * 90
        if angle == 0:
            return image, labels
        return self._rotate_with_labels(image, labels, angle)

    def _rotate_with_labels(self, image: np.ndarray, labels: dict, angle: int) -> Tuple[np.ndarray, dict]:
        new_labels = copy.deepcopy(labels)
        original_height, original_width, _ = image.shape
        image = util.rotate(image, angle)
        points = np.array(labels["points"])
        new_points = np.array(new_labels["points"])
        if angle == 90:
            new_points[:, :, 0] = original_height - points[:, :, 1]
            new_points[:, :, 1] = points[:, :, 0]
        elif angle == 180:
            new_points[:, :, 0] = original_width - points[:, :, 0]
            new_points[:, :, 1] = original_height - points[:, :, 1]
        elif angle == 270:
            new_points[:, :, 0] = points[:, :, 1]
            new_points[:, :, 1] = original_width - points[:, :, 0]
        new_labels["points"] = new_points.tolist()
        return image, new_labels

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

    def _filter_small_labels(self, labels: dict) -> dict:
        new_labels = copy.deepcopy(labels)
        for i in range(len(labels["points"])):
            points = new_labels["points"][i]
            for j in range(len(points)):
                y1, x1 = points[j]
                y2, x2 = points[(j + 1) % 4]
                distance = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if distance < 10:
                    new_labels["ignored"][i] = True
                    break
        return new_labels

    def _mask_and_distances(self, label):
        map_size = (self.image_size[0] // self.scale, self.image_size[1] // self.scale)
        height, width = map_size
        mask_map = np.zeros(map_size, np.uint8)
        # top, left, bottom, right
        distance_map = np.zeros((4,) + map_size, np.float)
        for points in label["points"]:
            xmin = min((p[0] for p in points))
            ymin = min((p[1] for p in points))
            xmax = max((p[0] for p in points))
            ymax = max((p[1] for p in points))
            x_r = (xmax - xmin) * 0.3
            x_from = int(round((xmin + x_r) / self.scale))
            x_to = int(round((xmax - x_r) / self.scale))
            x_to = min(x_to, width - 1)
            y_r = (ymax - ymin) * 0.3
            y_from = int(round((ymin + y_r) / self.scale))
            y_to = int(round((ymax - y_r) / self.scale))
            y_to = min(y_to, height - 1)
            if x_from == x_to or y_from == y_to:
                continue
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

