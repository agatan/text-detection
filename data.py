import math
from typing import Tuple
import copy
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
import cv2

import util


class ICDAR15Dataset(data.Dataset):
    def __init__(self, image_dir, labels_dir, image_size=(512, 512), scale=4, training=True):
        self.image_dir = Path(image_dir)
        self.labels_dir = Path(labels_dir)
        self.labels = self._read_labels()
        self.image_size = image_size
        self.scale = scale
        self.training = training

    def _read_labels(self):
        labels = []
        index = 1
        while True:
            filename = self.labels_dir / "gt_img_{}.txt".format(index)
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

    def _read_image(self, index):
        filename = str(self.image_dir / "img_{}.jpg".format(index + 1))
        return cv2.imread(filename, cv2.IMREAD_COLOR)

    def __getitem__(self, index):
        image = self._read_image(index)
        if self.training:
            image, labels = self._train_transform(image, self.labels[index])
        else:
            image, labels = self._test_transform(image, self.labels[index])
        pos_pixel_mask, neg_pixel_mask, pixel_weight, link_mask = self._mask_and_pixel_weights(labels)
        image = torch.Tensor(image.transpose(2, 0, 1)) / 255.
        return image, pos_pixel_mask, neg_pixel_mask, pixel_weight, link_mask

    def _train_transform(self, image, labels):
        if np.random.random() < 0.2:
            image, labels = self._random_rotate_with_labels(image, labels)
        image, labels = self._random_crop_with_labels(image, labels)
        image, labels = self._resize_image_with_labels(image, labels)
        labels = self._filter_small_labels(labels)
        return image, labels

    def _test_transform(self, image, labels):
        image, labels = self._resize_image_with_labels(image, labels)
        labels = self._filter_small_labels(labels)
        return image, labels

    def _random_rotate_with_labels(self, image: np.ndarray, labels: dict) -> Tuple[np.ndarray, dict]:
        angle = np.random.randint(0, 4) * 90
        if angle == 0:
            return image, labels
        return self._rotate_with_labels(image, labels, angle)

    def _rotate_with_labels(self, image: np.ndarray, labels: dict, angle: int) -> Tuple[np.ndarray, dict]:
        new_labels = copy.deepcopy(labels)
        original_height, original_width, _ = image.shape
        image = util.rotate(image, angle)
        for i in range(len(labels["points"])):
            points = np.array(labels["points"][i])
            new_points = np.array(new_labels["points"][i])
            if angle == 90:
                new_points[:, 0] = points[:, 1]
                new_points[:, 1] = original_height - points[:, 0]
            elif angle == 180:
                new_points[:, 0] = original_height - points[:, 0]
                new_points[:, 1] = original_width - points[:, 1]
            elif angle == 270:
                new_points[:, 0] = original_width - points[:, 1]
                new_points[:, 1] = points[:, 0]
            new_labels["points"][i] = new_points.tolist()
        return image, labels

    def _random_crop_with_labels(self, image: np.ndarray, labels: dict) -> Tuple[np.ndarray, dict]:
        new_labels = copy.deepcopy(labels)
        scale = 0.1 + np.random.random() * 0.9
        image, (offset_y, offset_x) = util.random_crop(image, scale)
        height, width, _ = image.shape
        for i in range(len(labels["points"])):
            points = np.array(labels["points"][i])
            original_area = cv2.contourArea(points)
            new_points = np.array(new_labels["points"][i])
            new_points[:, 0] = np.clip(points[:, 0] + offset_x, 0, width)
            new_points[:, 1] = np.clip(points[:, 1] + offset_y, 0, height)
            new_labels["points"][i] = new_points.tolist()
            new_area = cv2.contourArea(new_points)
            if original_area * 0.2 > new_area:
                new_labels["ignored"][i] = True
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

    def _mask_and_pixel_weights(self, label):
        def is_valid_coor(y, x, height, width):
            return 0 <= x < width and 0 <= y < height

        def neighbors(y, x):
            return [
                [y - 1, x - 1],
                [y - 1, x],
                [y - 1, x + 1],
                [y, x - 1],
                [y, x + 1],
                [y + 1, x - 1],
                [y + 1, x + 1],
            ]

        label_points = np.array(label["points"]) // self.scale
        pixel_mask_size = [size // self.scale for size in self.image_size]
        link_mask_size = [8,] + pixel_mask_size
        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_weight = np.zeros(pixel_mask_size, dtype=np.float32)
        link_mask = np.zeros(link_mask_size, dtype=np.uint8)

        bbox_masks = []
        n_positive_bboxes = 0
        for i, ps in enumerate(label_points):
            pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
            cv2.drawContours(pixel_mask_tmp, [ps], -1, color=1, thickness=-1)
            bbox_masks.append(pixel_mask_tmp)
            if not label["ignored"][i]:
                pixel_mask += pixel_mask_tmp
                n_positive_bboxes += 1

        pos_pixel_mask = (pixel_mask == 1).astype(np.int)
        n_pos_pixels = np.sum(pos_pixel_mask)
        sum_mask = np.sum(bbox_masks, axis=0)
        neg_pixel_mask = (sum_mask != 1).astype(np.int)
        not_overlapped_mask = sum_mask == 1
        for bbox_index, bbox_mask in enumerate(bbox_masks):
            bbox_positive_pixel_mask = bbox_mask * pos_pixel_mask
            n_pos_pixel = np.sum(bbox_positive_pixel_mask)
            if n_pos_pixel:
                per_bbox_weight = n_pos_pixels / float(n_positive_bboxes)
                per_pixel_weight = per_bbox_weight / n_pos_pixel
                pixel_weight += bbox_positive_pixel_mask * per_pixel_weight
            for link_index in range(8):
                link_mask[link_index][np.where(bbox_positive_pixel_mask)] = 1

            def in_bbox(y, x):
                return bbox_positive_pixel_mask[y, x]
            for y, x in zip(*np.where(bbox_positive_pixel_mask)):
                for n_index, (y_, x_) in enumerate(neighbors(y, x)):
                    if is_valid_coor(y_, x_, self.image_size[0] // self.scale, self.image_size[1] // self.scale) and not in_bbox(y_, x_):
                        link_mask[n_index][y, x] = 0
        return torch.LongTensor(pos_pixel_mask), torch.LongTensor(neg_pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)
