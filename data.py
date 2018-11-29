from pathlib import Path

import torch.utils.data as data
from PIL import Image


class ICDAR15Dataset(data.Dataset):
    def __init__(self, image_dir, labels_dir):
        self.image_dir = Path(image_dir)
        self.labels_dir = Path(labels_dir)
        self.labels = self._read_labels()

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
        return Image.open(filename).convert("RGB")

    def __getitem__(self, index):
        return self._read_image(index), self.labels[index]


print(ICDAR15Dataset("./dataset/icdar2015/images", "./dataset/icdar2015/localization")[0])
