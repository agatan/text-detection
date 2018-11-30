import os

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2
import numpy as np

from data import ICDAR15Dataset
import net

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./dataset/icdar2015")
    parser.add_argument("--restore")
    args = parser.parse_args()
    dataset = ICDAR15Dataset(os.path.join(args.train, "images"), os.path.join(args.train, "labels"), image_size=(512, 512), scale=4)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pixellink = net.PixelLink(args.scale, pretrained=False).to(device)
    pixellink = net.MobileNetV2PixelLink(4).to(device)
    pixellink.load_state_dict(torch.load(args.restore))
    pixellink.eval()

    with torch.no_grad():
        for images, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_masks in dataloader:
            images = images.to(device)
            pos_pixel_masks = pos_pixel_masks.to(device)
            neg_pixel_masks = neg_pixel_masks.to(device)
            pixel_weights = pixel_weights.to(device)
            link_masks = link_masks.to(device)
            pixel_prediction, link_prediction = pixellink(images)
            pixel_prediction = torch.softmax(pixel_prediction, dim=1)
            pixel_map = pixel_prediction[0, 1, :, :].numpy()
            # pixel_map += 0.3
            neighbor = 2
            link_prediction = torch.softmax(link_prediction[:, 2*neighbor: 2*(neighbor+1)], dim=1)
            link_map = link_prediction[0, 1, :, :].numpy() * pos_pixel_masks[0].numpy()
            link_map /= link_map.max()
            # link_map += 0.3
            m = link_map
            # m = pixel_map
            image = images.transpose(1, 2).transpose(2, 3).numpy()[0]
            out = image * np.expand_dims(cv2.resize(m, image.shape[:2]), -1)
            plt.imshow(out)
            plt.show()

main()
