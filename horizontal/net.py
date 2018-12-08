import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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
            nn.ReLU(),
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


class Net(nn.Module):
    def __init__(self, scale, excitation_cls=None):
        super(Net, self).__init__()
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
        self.block5 = self._layers_with_excitation_as_need(features[6:], 1280)
        self.out_conv1 = self._out_conv_with_excitation_as_need(256 + 96, 128)
        self.out_conv2 = self._out_conv_with_excitation_as_need(128 + 32, 64)
        self.out_conv3 = self._out_conv_with_excitation_as_need(64 + 24, 32)
        if self.scale == 2:
            self.out_conv4 = self._out_conv_with_excitation_as_need(32 + 16, 32)
        self.final_conv = nn.Conv2d(32, 6, kernel_size=1)

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
        o = self.final_conv(o1)
        # Returns text/non-text, distances
        return o[:, :2, :, :], o[:, 2:, :, :]
