import os

import torch
import torch.utils.data as data
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import itertools

from data import ICDAR15Dataset
import net


class Reporter:
    def __init__(self, name, writer, frequency=None):
        self.name = name
        self.losses = []
        self.pixel_losses = []
        self.pixel_accuracies = []
        self.link_losses = []
        self.link_accuracies = []
        self.writer = writer
        self.frequency = frequency

    def step(self, loss_object, steps):
        self.losses.append(loss_object.loss.item())
        self.pixel_losses.append(loss_object.pixel_loss.item())
        self.pixel_accuracies.append(loss_object.pixel_accuracy)
        self.link_losses.append(loss_object.link_loss.item())
        self.link_accuracies.append(loss_object.link_accuracy)
        if self.frequency is not None and len(self.losses) == self.frequency:
            self.flush(steps)

    def flush(self, steps):
        print("Report {}: Step {}".format(self.name, steps))
        print("Loss: {} (Pixel: {}, Link: {})".format(np.mean(self.losses), np.mean(self.pixel_losses), np.mean(self.link_losses)))
        print("Pixel Accuracy: {:.4f}, Link Accuracy: {:.4f}".format(np.mean(self.pixel_accuracies), np.mean(self.link_accuracies)))
        self._write("loss", self.losses, steps)
        self._write("loss/pixel", self.pixel_losses, steps)
        self._write("loss/link", self.link_losses, steps)
        self._write("accuracy/pixel", self.pixel_accuracies, steps)
        self._write("accuracy/link", self.link_accuracies, steps)
        self._reset()

    def _reset(self):
        self.losses = []
        self.pixel_losses = []
        self.pixel_accuracies = []
        self.link_losses = []
        self.link_accuracies = []

    def _write(self, name, value, steps):
        self.writer.add_scalar(name, np.mean(value), steps)



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
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_size = (512, 512)
    dataset = ICDAR15Dataset(os.path.join(args.train, "images"), os.path.join(args.train, "labels"), image_size=image_size, scale=args.scale, training=True)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    if args.test is not None:
        test_dataset = ICDAR15Dataset(os.path.join(args.test, "images"), os.path.join(args.test, "labels"), image_size=image_size, scale=args.scale, training=False)
    else:
        n_train = int(len(dataset) * 0.95)
        dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    test_dataloader = data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pixellink = net.PixelLink(args.scale, pretrained=False).to(device)
    pixellink = net.MobileNetV2PixelLink(args.scale).to(device)
    optimizer = torch.optim.Adam(pixellink.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    steps = 0
    start_epoch = 0
    best_score = None
    if args.restore:
        state_dict = torch.load(args.restore)
        pixellink.load_state_dict(state_dict['pixellink'])
        optimizer.load_state_dict(state_dict['optimizer'])
        steps = state_dict["steps"]
        start_epoch = state_dict["epoch"]
        args.scale = state_dict["scale"]
        best_score = state_dict["best_score"]

    os.makedirs(args.checkpoint, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.logdir, "train"))
    reporter = Reporter("train", writer, 200)
    test_writer = SummaryWriter(os.path.join(args.logdir, "test"))
    test_reporter = Reporter("test", test_writer)
    for epoch in range(start_epoch, args.epochs):
        print("[Epoch {}]".format(epoch))
        steps_per_epoch = (len(dataset) - 1) // args.batch_size + 1
        for images, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_masks in tqdm(itertools.islice(dataloader, steps_per_epoch), total=steps_per_epoch):
            pixellink.train()
            optimizer.zero_grad()
            images = images.to(device)
            pos_pixel_masks = pos_pixel_masks.to(device)
            neg_pixel_masks = neg_pixel_masks.to(device)
            pixel_weights = pixel_weights.to(device)
            link_masks = link_masks.to(device)
            pixel_input, link_input = pixellink(images)
            # loss_object = net.PixelLinkLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
            loss_object = net.PixelLinkFocalLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
            loss_object.loss.backward()
            optimizer.step()
            steps += 1
            reporter.step(loss_object, steps)
        reporter.flush(steps)

        pixellink.eval()
        test_steps_per_epoch = (len(test_dataset) - 1) // 8 + 1
        with torch.no_grad():
            for images, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_masks in tqdm(test_dataloader, total=test_steps_per_epoch):
                images = images.to(device)
                pos_pixel_masks = pos_pixel_masks.to(device)
                neg_pixel_masks = neg_pixel_masks.to(device)
                pixel_weights = pixel_weights.to(device)
                link_masks = link_masks.to(device)
                pixel_input, link_input = pixellink(images)
                # loss_object = net.PixelLinkLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
                loss_object = net.PixelLinkFocalLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
                test_reporter.step(loss_object, steps)

        current_score = np.mean(test_reporter.pixel_accuracies) * np.mean(test_reporter.link_accuracies)
        # scheduler.step(current_loss)
        best_score = current_score
        state_dict = {
            "pixellink": pixellink.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": steps,
            "epoch": epoch,
            "scale": args.scale,
            "best_score": best_score,
        }
        checkpoint_path = os.path.join(args.checkpoint, "epoch-{}.pth.tar".format(epoch))
        print("[Epoch {} Step {}] Save checkpoint {}".format(epoch, steps, checkpoint_path))
        torch.save(state_dict, checkpoint_path)
        test_reporter.flush(steps)

    writer.close()
    test_writer.close()

main()
