import os

import torch
import torch.utils.data as data

from data import ICDAR15Dataset
import net

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./dataset/icdar2015")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--checkpoint", default="checkpoint")
    parser.add_argument("--restore")
    args = parser.parse_args()
    dataset = ICDAR15Dataset(os.path.join(args.train, "images"), os.path.join(args.train, "labels"), image_size=(512, 512), scale=args.scale)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pixellink = net.PixelLink(args.scale, pretrained=False).to(device)
    pixellink = net.MobileNetV2PixelLink(args.scale).to(device)
    if args.restore:
        pixellink.load_state_dict(torch.load(args.restore))
    optimizer = torch.optim.Adam(pixellink.parameters())

    steps = 0
    os.makedirs(args.checkpoint, exist_ok=True)
    for epoch in range(args.epochs):
        print(epoch)
        for images, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_masks in dataloader:
            pixellink.train()
            optimizer.zero_grad()
            images = images.to(device)
            pos_pixel_masks = pos_pixel_masks.to(device)
            neg_pixel_masks = neg_pixel_masks.to(device)
            pixel_weights = pixel_weights.to(device)
            link_masks = link_masks.to(device)
            pixel_input, link_input = pixellink(images)
            loss_object = net.PixelLinkLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
            loss_object.loss.backward()
            optimizer.step()
            steps += 1
            if steps % 1 == 0:
                print("Loss: {} (Pixel: {}, Link: {})".format(loss_object.loss.item(), loss_object.pixel_loss.item(), loss_object.link_loss.item()))
                torch.save(pixellink.state_dict(), os.path.join(args.checkpoint, "best.pth"))


main()
