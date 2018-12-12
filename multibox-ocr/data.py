from typing import Union, Tuple, List, Dict, Optional
from pathlib import Path
import copy
import math

import torch
import torch.utils.data as data
import cv2
import numpy as np

import util


class CharSet:
    def __init__(self, chars: List[str]) -> None:
        self.chars = chars
        self._idx2char = ["<PAD>", "<UNK>"] + chars
        self._char2idx: Dict[str, int] = {}
        for i, c in enumerate(self._idx2char):
            self._char2idx[c] = i

    def __len__(self) -> int:
        return len(self._idx2char)

    def char2idx(self, char: str) -> int:
        if char in self._char2idx:
            return self._char2idx[char]
        return self._char2idx["<UNK>"]

    def idx2char(self, idx: int) -> Optional[str]:
        if 2 <= idx and idx < len(self._idx2char):
            return self._idx2char[idx]
        return None


class Dataset(data.Dataset):
    def __init__(self, image_dir: Union[str, Path], label_dir: Union[str, Path],
                 charset: CharSet,
                 image_size: Tuple[int, int] = (512, 512),
                 training: bool = False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.charset = charset
        self.labels = self._read_labels()
        # H, W
        self.image_size = image_size
        self.training = training

    def _read_labels(self):
        labels = []
        index = 1
        while True:
            filename = self.label_dir / "gt_img_{}.txt".format(index)
            if not filename.exists():
                break
            label: dict = {
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
            break
        return labels

    def __len__(self):
        return len(self.labels)

    def _read_image(self, index: int):
        filename = str(self.image_dir / "img_{}.jpg".format(index + 1))
        return cv2.imread(filename, cv2.IMREAD_COLOR)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        image = self._read_image(index)
        label = self.labels[index]
        if self.training:
            image, label = self._train_transform(image, label)
        else:
            image, label = self._test_transform(image, label)
        image = np.transpose(image, (2, 0, 1)).astype(np.float) / 255.
        n_boxes = len(label["points"])
        selected_box = np.random.randint(0, n_boxes)
        box = self._generate_boxes(label["points"][selected_box])
        text_target, text_lengths = self._text_target(label["text"][selected_box])
        return torch.FloatTensor(image), box, text_target, torch.LongTensor(text_lengths)

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

    def _text_target(self, text: str) -> Tuple[torch.Tensor, int]:
        length = len(text)
        text_target = torch.tensor([self.charset.char2idx(c) for c in text])
        return text_target, length

    def _generate_boxes(self, points: list) -> torch.Tensor:
        xmin = min((p[0] for p in points))
        ymin = min((p[1] for p in points))
        xmax = max((p[0] for p in points))
        ymax = max((p[1] for p in points))
        return torch.Tensor([xmin, ymin, xmax, ymax])


def collate_fn(samples: list) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    images, boxes, text_targets, text_lengths = zip(*samples)
    batch_size = len(samples)
    max_length = max((txt.size(0) for txt in text_targets))
    padded_text_targets = torch.zeros(batch_size, max_length, dtype=torch.int32)
    for i, txt in enumerate(text_targets):
        length = txt.size(0)
        padded_text_targets[i, :length] = txt
    return torch.stack(list(images)), torch.stack(list(boxes)), padded_text_targets, torch.stack(list(text_lengths))

