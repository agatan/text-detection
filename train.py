import os

import torch
import torch.utils.data as data
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

from data import ICDAR15Dataset
import net

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./dataset/icdar2015/train")
    parser.add_argument("--test", default="./dataset/icdar2015/test")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--checkpoint", default="checkpoint")
    parser.add_argument("--restore")
    args = parser.parse_args()
    image_size = (512, 512)
    dataset = ICDAR15Dataset(os.path.join(args.train, "images"), os.path.join(args.train, "labels"), image_size=image_size, scale=args.scale)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_dataset = ICDAR15Dataset(os.path.join(args.test, "images"), os.path.join(args.test, "labels"), image_size=image_size, scale=args.scale, training=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pixellink = net.PixelLink(args.scale, pretrained=False).to(device)
    pixellink = net.MobileNetV2PixelLink(args.scale).to(device)
    optimizer = torch.optim.Adam(pixellink.parameters(), lr=1e-3)
    steps = 0
    start_epoch = 0
    best_loss = None
    if args.restore:
        state_dict = torch.load(args.restore)
        pixellink.load_state_dict(state_dict['pixellink'])
        optimizer.load_state_dict(state_dict['optimizer'])
        steps = state_dict["steps"]
        start_epoch = state_dict["epoch"]
        args.scale = state_dict["scale"]
        best_loss = state_dict["best_loss"]

    os.makedirs(args.checkpoint, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.logdir, "train"))
    test_writer = SummaryWriter(os.path.join(args.logdir, "test"))
    for epoch in range(start_epoch, args.epochs):
        print("[Epoch {}]".format(epoch))
        pixel_losses = []
        pixel_accuracies = []
        link_losses = []
        link_accuracies = []
        losses = []
        steps_per_epoch = (len(dataset) - 1) // args.batch_size + 1
        for images, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_masks in tqdm(dataloader, total=steps_per_epoch):
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
            pixel_losses.append(loss_object.pixel_loss.item())
            pixel_accuracies.append(loss_object.pixel_accuracy)
            link_losses.append(loss_object.link_loss.item())
            link_accuracies.append(loss_object.link_accuracy)
            losses.append(loss_object.loss.item())
        print("Loss: {} (Pixel: {}, Link: {})".format(np.mean(losses), np.mean(pixel_losses), np.mean(link_losses)))
        print("Pixel Accuracy: {:.4f}, Link Accuracy: {:.4f}".format(np.mean(pixel_accuracies), np.mean(link_accuracies)))
        writer.add_scalar("loss", np.mean(losses), steps)
        writer.add_scalar("loss/pixel", np.mean(pixel_losses), steps)
        writer.add_scalar("loss/link", np.mean(link_losses), steps)
        writer.add_scalar("accuracy/pixel", np.mean(pixel_accuracies), steps)
        writer.add_scalar("accuracy/link", np.mean(link_accuracies), steps)

        pixellink.eval()
        test_losses = []
        test_pixel_losses = []
        test_link_losses = []
        test_pixel_accuracies = []
        test_link_accuracies = []
        test_steps_per_epoch = (len(test_dataset) - 1) // args.batch_size + 1
        with torch.no_grad():
            for images, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_masks in tqdm(test_dataloader, total=test_steps_per_epoch):
                images = images.to(device)
                pos_pixel_masks = pos_pixel_masks.to(device)
                neg_pixel_masks = neg_pixel_masks.to(device)
                pixel_weights = pixel_weights.to(device)
                link_masks = link_masks.to(device)
                pixel_input, link_input = pixellink(images)
                loss_object = net.PixelLinkLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
                test_pixel_losses.append(loss_object.pixel_loss.item())
                test_pixel_accuracies.append(loss_object.pixel_accuracy)
                test_link_losses.append(loss_object.link_loss.item())
                test_link_accuracies.append(loss_object.link_accuracy)
                test_losses.append(loss_object.loss.item())
        print("Test Loss: {} (Pixel: {}, Link: {})".format(np.mean(test_losses), np.mean(test_pixel_losses), np.mean(test_link_losses)))
        print("Pixel Accuracy: {:.4f}, Link Accuracy: {:.4f}".format(np.mean(test_pixel_accuracies), np.mean(test_link_accuracies)))
        current_loss = np.mean(losses)
        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            state_dict = {
                "pixellink": pixellink.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": steps,
                "epoch": epoch,
                "scale": args.scale,
                "best_loss": best_loss,
            }
            checkpoint_path = os.path.join(args.checkpoint, "best.pth.tar")
            print("[Epoch {} Step {}] Save checkpoint {}".format(epoch, steps, checkpoint_path))
            torch.save(state_dict, checkpoint_path)
        test_writer.add_scalar("loss", np.mean(test_losses), steps)
        test_writer.add_scalar("loss/pixel", np.mean(test_pixel_losses), steps)
        test_writer.add_scalar("loss/link", np.mean(test_link_losses), steps)
        test_writer.add_scalar("accuracy/pixel", np.mean(test_pixel_accuracies), steps)
        test_writer.add_scalar("accuracy/link", np.mean(test_link_accuracies), steps)

    writer.close()

main()
