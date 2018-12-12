from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def _conv1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class _InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(_InvertedResidual, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                # pointwise
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            hidden_dim = round(in_channels * expand_ratio)
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class CSE(nn.Module):
    def __init__(self, channels, ratio=16):
        super(CSE, self).__init__()
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = x.mean(dim=3).mean(dim=2)
        e = self.excitation(z).unsqueeze(2).unsqueeze(3)
        return x * e


class SSE(nn.Module):
    def __init__(self, channels):
        super(SSE, self).__init__()
        self.se = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e = self.se(x)
        return x * e


class SCSE(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SCSE, self).__init__()
        self.cse = CSE(channels, ratio)
        self.sse = SSE(channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


class Backbone(nn.Module):
    def __init__(self, scale, out_channels, excitation_cls=None):
        super(Backbone, self).__init__()
        assert scale in [2, 4]
        self.scale = scale
        first_channels = 32
        inverted_residual_config = [
            # t (expand ratio), channels, n (layers), stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = [_conv_bn(3, first_channels, stride=2)]
        input_channels = first_channels
        for i, (t, c, n, s) in enumerate(inverted_residual_config):
            output_channels = c
            layers = []
            for j in range(n):
                if j == 0:
                    layers.append(_InvertedResidual(input_channels, output_channels, stride=s, expand_ratio=t))
                else:
                    layers.append(_InvertedResidual(input_channels, output_channels, stride=1, expand_ratio=t))
                input_channels = output_channels
            features.append(nn.Sequential(*layers))
        last_channels = 256
        features.append(_conv1x1_bn(input_channels, last_channels))
        self.excitation_cls = excitation_cls
        self.block1 = self._layers_with_excitation_as_need(features[:2], 16)
        self.block2 = self._layers_with_excitation_as_need(features[2:3], 24)
        self.block3 = self._layers_with_excitation_as_need(features[3:4], 32)
        self.block4 = self._layers_with_excitation_as_need(features[4:6], 96)
        self.block5 = self._layers_with_excitation_as_need(features[6:], 256)
        self.out_conv1 = self._out_conv_with_excitation_as_need(256 + 96, 128)
        self.out_conv2 = self._out_conv_with_excitation_as_need(128 + 32, 64)
        self.out_conv3 = self._out_conv_with_excitation_as_need(64 + 24, 32)
        if self.scale == 2:
            self.out_conv4 = self._out_conv_with_excitation_as_need(32 + 16, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _layers_with_excitation_as_need(self, layers, channels):
        if self.excitation_cls is not None:
            layers.append(self.excitation_cls(channels))
        return nn.Sequential(*layers)

    def _out_conv_with_excitation_as_need(self, channels, out_channels):
        layers = [
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ]
        if self.excitation_cls is not None:
            layers.append(self.excitation_cls(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        o5 = torch.cat([x4, F.interpolate(x5, scale_factor=2)], dim=1)
        o4 = self.out_conv1(o5)
        o4 = torch.cat([x3, F.interpolate(o4, scale_factor=2)], dim=1)
        o3 = self.out_conv2(o4)
        o3 = torch.cat([x2, F.interpolate(o3, scale_factor=2)], dim=1)
        o2 = self.out_conv3(o3)
        if self.scale == 2:
            o1 = torch.cat([x1, F.interpolate(o2, scale_factor=2)], dim=1)
            o1 = self.out_conv4(o1)
        else:
            o1 = o2
        return self.final_conv(o1)


class BidirectionalBoxPool(nn.Module):
    def __init__(self, pool_height: int) -> None:
        super().__init__()
        self.pool_height = pool_height

    def forward(self, x: torch.Tensor, boxes: torch.Tensor):
        """
        Args:
            x: (N, C, H, W)
            boxes: (N, #boxes, 4 (xmin, ymin, xmax, ymax))
        Returns:
            features: (N, #boxes, 2, C, PH, PW)
            widths: (N, #boxes, 2)
        """
        batch_size = x.size(0)
        max_box = boxes.size(1)
        channels = x.size(1)
        base_height = x.size(2)
        base_width = x.size(3)
        max_width = self._max_width(boxes)
        device = x.device
        features = torch.zeros(batch_size, max_box, 2, channels, self.pool_height, max_width).to(device)
        widths = torch.zeros(batch_size, max_box, 2).to(device)
        for box_id in range(0, max_box):
            grids = torch.full((batch_size, 2, self.pool_height, max_width, 2), -2.0).to(device)
            for batch_id, box in enumerate(boxes[:, box_id]):
                xmin, ymin, xmax, ymax = box
                if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                    continue
                grid, width = self._make_grid(xmin, ymin, xmax, ymax, base_height, base_width)
                widths[batch_id, box_id, :] = width
                grids[batch_id, 0, :, :width, :] = grid
                grids[batch_id, 1, :, :width, :] = grid.flip((0, 1))
            sampled_features = F.grid_sample(x, grids[:, 0, :, :, :])
            inverted_sampled_features = F.grid_sample(x, grids[:, 1, :, :, :])
            features[:, box_id, 0, :, :, :] = sampled_features
            features[:, box_id, 1, :, :, :] = inverted_sampled_features
        return features, widths

    def _max_width(self, boxes: torch.Tensor) -> int:
        max_w_per_h = 0
        for batch in boxes:
            for box in batch:
                xmin, ymin, xmax, ymax = box
                if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                    continue
                box_height = ymax - ymin
                box_width = xmax - xmin
                if box_width > box_height:
                    w_per_h = box_width / float(box_height)
                else:
                    w_per_h = box_height / float(box_width)
                if max_w_per_h < w_per_h:
                    max_w_per_h = w_per_h
        return int(math.ceil(max_w_per_h * self.pool_height))

    def _make_grid(self, xmin, ymin, xmax, ymax, base_height, base_width):
        if xmax - xmin > ymax - ymin:
            return self._make_grid_wide(xmin, ymin, xmax, ymax, base_height, base_width)
        else:
            return self._make_grid_tall(xmin, ymin, xmax, ymax, base_height, base_width)

    def _make_grid_wide(self, xmin, ymin, xmax, ymax, base_height, base_width):
        width = int(math.ceil((xmax - xmin) / (ymax - ymin) * self.pool_height))
        device = xmin.device
        each_w = (xmax - xmin) / (width - 1)
        each_h = (ymax - ymin) / (self.pool_height - 1)
        xx = torch.arange(0, width, dtype=torch.float32).to(device) * each_w + xmin
        xx = xx.view(1, -1).repeat(self.pool_height, 1)
        xx = (xx - base_width / 2) / (base_width / 2)
        yy = torch.arange(0, self.pool_height, dtype=torch.float32).to(device) * each_h + ymin
        yy = yy.view(-1, 1).repeat(1, width)
        yy = (yy - base_height / 2) / (base_height / 2)
        return torch.stack([xx, yy], -1), width

    def _make_grid_tall(self, xmin, ymin, xmax, ymax, base_height, base_width):
        height = int(math.ceil((ymax - ymin) / (xmax - xmin) * self.pool_height))
        device = xmin.device
        each_w = (ymax - ymin) / (height - 1)
        each_h = (xmax - xmin) / (self.pool_height - 1)
        xx = torch.arange(height, 0, step=-1, dtype=torch.float32).to(device) * each_w + ymin
        xx = xx.view(1, -1).repeat(self.pool_height, 1)
        xx = (xx - base_height / 2) / (base_height / 2)
        yy = torch.arange(0, self.pool_height, dtype=torch.float32).to(device) * each_h + xmin
        yy = yy.view(-1, 1).repeat(1, height)
        yy = (yy - base_width / 2) / (base_width / 2)
        return torch.stack([yy, xx], -1), height


class RecognitionModule(nn.Module):
    def __init__(self, n_vocab: int, channels: int):
        super().__init__()
        self.n_vocab = n_vocab
        self.layer = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, self.n_vocab, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (N, C, H, W)
        Returns:
            tensor: (N, n_vocab, W)
        """
        return self.layer(x)


class Net(nn.Module):
    def __init__(self, n_vocab: int, excitation_cls: Optional[type] = None) -> None:
        super().__init__()
        self.feature_map_scale = 4
        self.backbone = Backbone(scale=self.feature_map_scale, out_channels=24, excitation_cls=excitation_cls)
        self.pool_height = 8
        self.box_pool = BidirectionalBoxPool(self.pool_height)
        self.recognition = RecognitionModule(n_vocab, 24)
        self.n_vocab = n_vocab

    def forward(self, images: torch.Tensor, boxes: torch.Tensor):
        """
        Args:
            images: (N, C, H, W)
            boxes: (N, #boxes, 4 (xmin, ymin, xmax, ymax))
        """
        feature_map = self.backbone(images)
        box_feature_map, widths = self.box_pool(feature_map, boxes / self.feature_map_scale)
        batch_size, max_box, _, channels, height, width = box_feature_map.size()
        flatten_feature = box_feature_map.view(-1, channels, height, width)
        flatten_recognition = self.recognition(flatten_feature)
        recognition = flatten_recognition.view(batch_size, max_box, 2, -1, width)
        return recognition, widths


def compute_loss(recognition: torch.Tensor, width: torch.Tensor,
                 text_target: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
    text_target = text_target.unsqueeze(2).repeat(1, 1, 2, 1)
    text_lengths = text_lengths.unsqueeze(2).repeat(1, 1, 2)
    batch_size, max_box, _bidi, vocab, max_width = recognition.size()
    assert _bidi == 2
    recognition = recognition.view(batch_size * max_box * 2, vocab, max_width)
    width = width.view(batch_size * max_box * 2)
    text_target = text_target.view(batch_size * max_box * 2, -1)
    text_lengths = text_lengths.view(batch_size * max_box * 2)

    # (boxes, channels, length)
    log_probs = F.log_softmax(recognition, dim=1)
    # (length, boxes, channels)
    log_probs = log_probs.transpose(1, 2).transpose(0, 1)

    bidirectional_loss = F.ctc_loss(log_probs, text_target, width, text_lengths, reduction='none')
    bidirectional_loss = bidirectional_loss.view(batch_size * max_box, 2)
    loss, _ = torch.min(bidirectional_loss, dim=1)

    # filter 0 length boxes
    indices = width.view(batch_size * max_box, 2)[:, 0] != 0
    return loss[indices].mean()



def test():
    import data
    charset = data.CharSet(list("0123456789"))
    dataset = data.Dataset("./dataset/cards-all/images", "./dataset/cards-all/labels", charset=charset, image_size=(384, 640))
    image, boxes, text_target, text_lengths = dataset[5]
    recognition, widths = Net(12)(image.unsqueeze(0), boxes.unsqueeze(0))
    loss = compute_loss(recognition, widths, text_target.unsqueeze(0), text_lengths.unsqueeze(0))
    print(loss)
    print(recognition.size(), widths.size())
    features, widths = BidirectionalBoxPool(pool_height=64)(image.unsqueeze(0), boxes.unsqueeze(0))
    import torchvision.utils as utils
    utils.save_image(features.view(-1, 3, 64, features.size(5)), "out.png", normalize=True)


if __name__ == "__main__":
    test()
