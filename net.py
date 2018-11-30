import torch
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
        self.last_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

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
        o = self.last_conv(o)
        # Returns text/non-text, 8 neighbors logits
        return o[:, :2, :, :], o[:, 2:, :, :]


class PixelLinkLoss:
    def __init__(self, pixel_input, pixel_target, neg_pixel_masks, pixel_weight, link_input, link_target, r=3):
        self._set_pixel_weight_and_loss(pixel_input, pixel_target, neg_pixel_masks, pixel_weight, r=r)
        self._set_link_loss(link_input, link_target)
        self.loss = self.pixel_loss + self.link_loss

    def _set_pixel_weight_and_loss(self, input, target, neg_pixel_masks, pixel_weight, r):
        batch_size = input.size(0)
        softmax_input = F.softmax(input, dim=1)
        self.pixel_cross_entropy = F.cross_entropy(softmax_input, target, reduction='none')
        self.area_per_image = torch.sum(target.view(batch_size, -1), dim=1)
        int_area_per_image = self.area_per_image.type(torch.int).detach().tolist()
        self.area_per_image = self.area_per_image.type(torch.float32)
        print(int_area_per_image)
        self.pos_pixel_weight = pixel_weight
        self.neg_pixel_weight = torch.zeros_like(pixel_weight, dtype=torch.uint8)
        self.neg_area_per_image = torch.zeros_like(self.area_per_image, dtype=torch.int)
        for i in range(batch_size):
            wrong = softmax_input[i, 0][neg_pixel_masks[i] == 1].view(-1)
            self.neg_area_per_image[i] = min(int_area_per_image[i] * r, wrong.size(0))
            topk, _ = torch.topk(-wrong, self.neg_area_per_image[i].item())
            self.neg_pixel_weight[i][softmax_input[i, 0] <= -topk[-1]] = 1
            self.neg_pixel_weight[i] = self.neg_pixel_weight[i] & (neg_pixel_masks[i] == 1)
        self.pixel_weight = self.pos_pixel_weight + self.neg_pixel_weight.type(torch.float32)
        weighted_pixel_cross_entropy_pos = (self.pos_pixel_weight * self.pixel_cross_entropy).view(batch_size, -1)
        weighted_pixel_cross_entropy_neg = (self.neg_pixel_weight.type(torch.float32) * self.pixel_cross_entropy).view(batch_size, -1)
        weighted_pixel_cross_entropy = weighted_pixel_cross_entropy_pos + weighted_pixel_cross_entropy_neg
        self.pixel_loss = weighted_pixel_cross_entropy.sum() / (self.area_per_image + self.neg_area_per_image.type(torch.float32)).sum()

    def _set_link_loss(self, link_input, link_target):
        batch_size = link_input.size(0)
        positive_pixels = self.pos_pixel_weight.unsqueeze(1).expand(-1, 8, -1, -1)
        self.pos_link_weight = (link_target == 1).type(torch.float32) * positive_pixels
        self.neg_link_weight = (link_target == 0).type(torch.float32) * positive_pixels
        link_cross_entropies = torch.Tensor.new_empty(self.pos_link_weight, self.pos_link_weight.size())
        for i in range(8):
            input = link_input[:, 2 * i:2 * (i + 1)]
            softmax_input = F.softmax(input, dim=1)
            target = link_target[:, i]
            link_cross_entropies[:, i] = F.cross_entropy(softmax_input, target, reduction='none')
        sum_pos_link_weight = torch.sum(self.pos_link_weight.view(batch_size, -1), dim=1).clamp(min=1e-8)
        sum_neg_link_weight = torch.sum(self.neg_link_weight.view(batch_size, -1), dim=1).clamp(min=1e-8)

        loss_link_pos = link_cross_entropies * self.pos_link_weight / sum_pos_link_weight
        loss_link_neg = link_cross_entropies * self.neg_link_weight / sum_neg_link_weight
        loss_link_pos = loss_link_pos.view(batch_size, -1).sum(dim=1)
        loss_link_neg = loss_link_neg.view(batch_size, -1).sum(dim=1)
        self.link_loss = torch.mean(loss_link_pos) + torch.mean(loss_link_neg)


def test():
    from torch.utils.data import DataLoader
    from data import ICDAR15Dataset
    dataset = ICDAR15Dataset("./dataset/icdar2015/images", "./dataset/icdar2015/labels")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    images, pos_pixel_masks, neg_pixel_masks, pixel_weight, link_mask = next(iter(loader))
    pixellink = PixelLink(scale=4)
    pixel_input, link_input = pixellink(images)
    loss_object = PixelLinkLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weight, link_input, link_mask)
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    image = (images[0].transpose(0, 1).transpose(1, 2).numpy() * 255).astype(np.uint8)
    weight = cv2.resize(cv2.applyColorMap((loss_object.pixel_weight[0].numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET), (image.shape[1], image.shape[0]))
    out = cv2.addWeighted(image, 0.5, weight, 0.5, 0)
    plt.imshow(out)
    plt.show()
    print(loss_object.pixel_loss)
    print(loss_object.link_loss)
    print(loss_object.loss)
