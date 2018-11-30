import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PixelLink(nn.Module):
    def __init__(self, scale, pretrained=False):
        super(PixelLink, self).__init__()
        assert scale in [2, 4]
        self.scale = scale
        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        self.block1 = nn.Sequential(*vgg16.features[:9])
        self.block2 = nn.Sequential(*vgg16.features[9:16])
        self.block3 = nn.Sequential(*vgg16.features[16:23])
        self.block4 = nn.Sequential(*vgg16.features[23:29])
        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        out_channels = 2 + 8 * 2  # text/non-text, 8 neighbors link
        self.out_conv1 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.out_conv2 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.out_conv3 = nn.Conv2d(256, out_channels, kernel_size=1)
        if self.scale == 2:
            self.out_conv4 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        o4 = self.out_conv1(x5) + self.out_conv2(x4)
        o3 = self.out_conv2(x3) + F.interpolate(o4, scale_factor=2)
        o2 = self.out_conv3(x2) + F.interpolate(o3, scale_factor=2)
        if self.scale == 2:
            o = self.out_conv4(x1) + F.interpolate(o2, scale_factor=2)
        else:
            o = o2
        return o
