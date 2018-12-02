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


class MobileNetV2PixelLink(nn.Module):
    def __init__(self, scale):
        super(MobileNetV2PixelLink, self).__init__()
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
        last_channels = 1280
        features.append(_conv1x1_bn(input_channels, last_channels))
        self.block1 = nn.Sequential(*features[:2])
        self.block2 = nn.Sequential(*features[2:3])
        self.block3 = nn.Sequential(*features[3:4])
        self.block4 = nn.Sequential(*features[4:6])
        self.block5 = nn.Sequential(*features[6:])
        out_channels = 2 + 8 * 2  # text/non-text, 8 neighbors
        self.out_conv1 = nn.Conv2d(1280, out_channels, kernel_size=1)
        self.out_conv2 = nn.Conv2d(96, out_channels, kernel_size=1)
        self.out_conv3 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.out_conv4 = nn.Conv2d(24, out_channels, kernel_size=1)
        if self.scale == 2:
            self.out_conv5 = nn.Conv2d(16, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        o5 = self.out_conv1(x5)
        o4 = self.out_conv2(x4) + F.interpolate(o5, scale_factor=2)
        o3 = self.out_conv3(x3) + F.interpolate(o4, scale_factor=2)
        o2 = self.out_conv4(x2) + F.interpolate(o3, scale_factor=2)
        if self.scale == 2:
            o1 = self.out_conv5(x1) + F.interpolate(o2, scale_factor=2)
        else:
            o1 = o2
        o = self.final_conv(o1)
        # Returns text/non-text, 8 neighbors logits
        return o[:, :2, :, :], o[:, 2:, :, :]

class PixelLinkLoss:
    def __init__(self, pixel_input, pixel_target, neg_pixel_masks, pixel_weight, link_input, link_target, r=3, l=2):
        self._set_pixel_weight_and_loss(pixel_input, pixel_target, neg_pixel_masks, pixel_weight, r=r)
        self._set_link_loss(link_input, link_target)
        self._set_pixel_accuracy(pixel_input, pixel_target, neg_pixel_masks)
        self._set_link_accuracy(link_input, link_target, pixel_target)
        self.loss = l * self.pixel_loss + self.link_loss

    def _set_pixel_weight_and_loss(self, input, target, neg_pixel_masks, pixel_weight, r):
        batch_size = input.size(0)
        softmax_input = F.softmax(input, dim=1)
        self.pixel_cross_entropy = F.cross_entropy(input, target, reduction='none')
        self.area_per_image = torch.sum(target.view(batch_size, -1), dim=1)
        int_area_per_image = self.area_per_image.type(torch.int).detach().tolist()
        self.area_per_image = self.area_per_image.type(torch.float32)
        self.pos_pixel_weight = pixel_weight
        self.neg_pixel_weight = torch.zeros_like(pixel_weight, dtype=torch.uint8)
        self.neg_area_per_image = torch.zeros_like(self.area_per_image, dtype=torch.int)

        for i in range(batch_size):
            wrong = softmax_input[i, 0][neg_pixel_masks[i] == 1].view(-1)
            self.neg_area_per_image[i] = min(int_area_per_image[i] * r, wrong.size(0))
            topk, _ = torch.topk(-wrong, self.neg_area_per_image[i].item())
            if topk.size(0) != 0:
                self.neg_pixel_weight[i][softmax_input[i, 0] <= -topk[-1]] = 1
                self.neg_pixel_weight[i] = self.neg_pixel_weight[i] & (neg_pixel_masks[i] == 1)

        self.pixel_weight = self.pos_pixel_weight + self.neg_pixel_weight.type(torch.float32)
        weighted_pixel_cross_entropy_pos = (self.pos_pixel_weight * self.pixel_cross_entropy).view(batch_size, -1)
        weighted_pixel_cross_entropy_neg = (self.neg_pixel_weight.type(torch.float32) * self.pixel_cross_entropy).view(batch_size, -1)
        weighted_pixel_cross_entropy = weighted_pixel_cross_entropy_pos + weighted_pixel_cross_entropy_neg
        self.pixel_loss = weighted_pixel_cross_entropy.sum() / (self.area_per_image + self.neg_area_per_image.type(torch.float32)).sum()

    def _set_pixel_accuracy(self, input, target, neg_pixel_mask):
        input = input.detach()
        _, argmax = torch.max(input, dim=1)
        positive_count = torch.sum((argmax == 1) * (target == 1)).item()
        negative_count = torch.sum((argmax == 0) * (neg_pixel_mask == 1)).item()
        elt_count = torch.sum(target == 1).item() + torch.sum(neg_pixel_mask == 1).item()
        self.pixel_accuracy = (positive_count + negative_count) / float(elt_count)

    def _set_link_loss(self, link_input, link_target):
        batch_size = link_input.size(0)
        positive_pixels = self.pos_pixel_weight.unsqueeze(1).expand(-1, 8, -1, -1)
        self.pos_link_weight = (link_target == 1).type(torch.float32) * positive_pixels
        self.neg_link_weight = (link_target == 0).type(torch.float32) * positive_pixels
        link_cross_entropies = torch.Tensor.new_empty(self.pos_link_weight, self.pos_link_weight.size())
        for i in range(8):
            input = link_input[:, 2 * i:2 * (i + 1)]
            target = link_target[:, i]
            link_cross_entropies[:, i] = F.cross_entropy(input, target, reduction='none')
        sum_pos_link_weight = torch.sum(self.pos_link_weight.view(batch_size, -1), dim=1).clamp(min=1e-8)
        sum_neg_link_weight = torch.sum(self.neg_link_weight.view(batch_size, -1), dim=1).clamp(min=1e-8)

        loss_link_pos = link_cross_entropies * self.pos_link_weight
        loss_link_neg = link_cross_entropies * self.neg_link_weight
        loss_link_pos = loss_link_pos.view(batch_size, -1).sum(dim=1) / sum_pos_link_weight
        loss_link_neg = loss_link_neg.view(batch_size, -1).sum(dim=1) / sum_neg_link_weight
        self.link_loss = torch.mean(loss_link_pos) + torch.mean(loss_link_neg)

    def _set_link_accuracy(self, input, target, pixel_mask):
        input = input.detach()
        elt_count = float(pixel_mask.sum().item())
        if elt_count == 0.0:
            self.link_accuracy = [0] * 8
            return
        pixel_mask = pixel_mask.byte()
        accuracies = []
        for i in range(8):
            _, argmax = torch.max(input[:, 2*i:2*(i+1)], dim=1)
            match_count = torch.sum((argmax == target[:, i]) * pixel_mask).item()
            accuracies.append(match_count / elt_count)
        self.link_accuracy = accuracies


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, float):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        else:
            self.alpha = alpha

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        original_size = target.size()
        target = target.contiguous().view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(dim=1, index=target).view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            self.alpha = self.alpha.type(logpt.dtype).to(logpt.device)
            at = self.alpha.gather(dim=0, index=target.detach().view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.view(original_size[0], original_size[1], original_size[2])


class PixelLinkFocalLoss(PixelLinkLoss):
    def __init__(self, pixel_input, pixel_target, neg_pixel_masks, pixel_weight, link_input, link_target, r=3, l=2):
        self._set_pixel_weight_and_loss(pixel_input, pixel_target, neg_pixel_masks, pixel_weight, r=r)
        self._set_link_loss(link_input, link_target, pixel_target)
        self._set_pixel_accuracy(pixel_input, pixel_target, neg_pixel_masks)
        self._set_link_accuracy(link_input, link_target, pixel_target)
        self.loss = l * self.pixel_loss + self.link_loss

    def _set_pixel_weight_and_loss(self, input, target, neg_pixel_mask, pixel_weight, r):
        batch_size = input.size(0)
        self.pixel_cross_entropy = FocalLoss(alpha=0.25)(input, target)
        non_ignored = (target == 1) + (neg_pixel_mask == 1)
        self.pixel_cross_entropy *= non_ignored.float()
        loss = torch.sum(self.pixel_cross_entropy, dim=1) / torch.sum(non_ignored, dim=1).float().clamp(min=1e-8)
        self.pixel_loss = torch.mean(loss)

    def _set_link_loss(self, input, target, positive_mask):
        batch_size = input.size(0)
        positive_mask = positive_mask.float().unsqueeze(1).expand(-1, 8, -1, -1)
        link_cross_entropies = torch.zeros_like(positive_mask)
        focal = FocalLoss(alpha=0.25)
        for i in range(8):
            inp = input[:, 2 * i:2 * (i + 1)]
            tar = target[:, i]
            link_cross_entropies[:, i] = focal(inp, tar)
        link_cross_entropies *= positive_mask
        pixels_per_image = torch.sum(positive_mask.contiguous().view(batch_size, -1), dim=1)
        loss = torch.sum(link_cross_entropies.view(batch_size, -1), dim=1) / pixels_per_image.float().clamp(min=1e-8)
        self.link_loss = torch.mean(loss)


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
