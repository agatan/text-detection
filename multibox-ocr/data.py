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
                 pool_height: int = 8,
                 training: bool = False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.labels = self._read_labels()
        # H, W
        self.image_size = image_size
        self.scale = scale
        self.pool_height = pool_height
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
        grids = self._generate_grids(label)
        return torch.FloatTensor(image), torch.FloatTensor(grids)

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

    def _generate_grids(self, label):
        n_boxes = len(label["text"])
        w_per_h = []
        boxes = []
        for points in label["points"]:
            xmin = min((p[0] for p in points))
            ymin = min((p[1] for p in points))
            xmax = max((p[0] for p in points))
            ymax = max((p[1] for p in points))
            box_height = ymax - ymin
            box_width = xmax - xmin
            boxes.append((xmin, ymin, xmax, ymax))
            if box_width > box_height:
                w_per_h.append(box_width / float(box_height))
            else:
                w_per_h.append(box_height / float(box_width))
        max_width = int(math.ceil(max(w_per_h) * self.pool_height))
        grids_size = (n_boxes, 2, self.pool_height, max_width, 2)
        grids = torch.full(grids_size, -2.0)
        image_height, image_width = self.image_size
        feature_map_height = image_height // self.scale
        feature_map_width = image_width // self.scale
        for box_id, box in enumerate(boxes[:1]):
            xmin, ymin, xmax, ymax = box
            grid, width = self._make_grid(xmin, ymin, xmax, ymax)
            grids[box_id, 0, :, :width, :] = grid
            grids[box_id, 1, :, :width, :] = grid.flip((0, 1))
        return grids

    def _make_grid(self, xmin, ymin, xmax, ymax):
        if xmax - xmin > ymax - ymin:
            return self._make_grid_wide(xmin, ymin, xmax, ymax)
        else:
            return self._make_grid_tall(xmin, ymin, xmax, ymax)

    def _make_grid_wide(self, xmin, ymin, xmax, ymax):
        image_height, image_width = self.image_size
        width = int(math.ceil((xmax - xmin) / (ymax - ymin) * self.pool_height))
        each_w = (xmax - xmin) / (width - 1)
        each_h = (ymax - ymin) / (self.pool_height - 1)
        xx = torch.arange(0, width, dtype=torch.float32) * each_w + xmin
        xx = xx.view(1, -1).repeat(self.pool_height, 1)
        xx = (xx - image_width / 2) / (image_width / 2)
        yy = torch.arange(0, self.pool_height, dtype=torch.float32) * each_h + ymin
        yy = yy.view(-1, 1).repeat(1, width)
        yy = (yy - image_height / 2) / (image_height / 2)
        return torch.stack([xx, yy], -1), width

    def _make_grid_tall(self, xmin, ymin, xmax, ymax):
        image_height, image_width = self.image_size
        height = int(math.ceil((ymax - ymin) / (xmax - xmin) * self.pool_height))
        each_w = (ymax - ymin) / (height - 1)
        each_h = (xmax - xmin) / (self.pool_height - 1)
        xx = torch.arange(height, 0, step=-1, dtype=torch.float32) * each_w + ymin
        xx = xx.view(1, -1).repeat(self.pool_height, 1)
        xx = (xx - image_height / 2) / (image_height / 2)
        yy = torch.arange(0, self.pool_height, dtype=torch.float32) * each_h + xmin
        yy = yy.view(-1, 1).repeat(1, height)
        yy = (yy - image_width / 2) / (image_width / 2)
        return torch.stack([yy, xx], -1), height


def test():
    dataset = Dataset("./dataset/cards-all/images", "./dataset/cards-all/labels", image_size=(384, 640), pool_height=32)
    image, grids = dataset[4]
    import torch.nn.functional as F
    sample = F.grid_sample(torch.stack([image], 0), grids[:1, 1])
    import torchvision.utils as utils
    utils.save_image(image[:, 16:338, 83:107].unsqueeze(0), "out-orig-box.png", normalize=True)
    utils.save_image(image.unsqueeze(0), "out-orig.png", normalize=True)
    utils.save_image(sample, "out.png", normalize=True)
