import os
import warnings

import torch
import torch.utils.data as data
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import itertools

from data import ICDAR15Dataset
import net


LOG_FREQ = 200


def create_summary_writer(model, dummy, logdir):
    writer = SummaryWriter(logdir)
    writer.add_graph(model, dummy)
    return writer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./dataset/icdar2015/train")
    parser.add_argument("--test")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--logdir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--restore")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--excitation", choices=["cse", "sse", "scse"], default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_size = (512, 512)
    dataset = ICDAR15Dataset(os.path.join(args.train, "images"), os.path.join(args.train, "labels"), image_size=image_size, scale=args.scale, training=True)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    if args.test is not None:
        test_dataset = ICDAR15Dataset(os.path.join(args.test, "images"), os.path.join(args.test, "labels"), image_size=image_size, scale=args.scale, training=False)
    else:
        n_test = min(1000, (len(dataset) * 0.05))
        indices = np.arange(len(dataset))
        dataset = torch.utils.data.Subset(dataset, indices[n_test:])
        test_dataset = torch.utils.data.Subset(dataset, indices[:n_test])
        print(len(dataset), len(test_dataset))
    test_dataloader = data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pixellink = net.PixelLink(args.scale, pretrained=False).to(device)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    if args.restore:
        if torch.cuda.is_available():
            map_location = None
        else:
            def map_location(storage, loc):
                return storage
        pixellink = torch.load(args.restore, map_location=map_location).to(device)
    else:
        excitation_cls = {"cse": net.CSE, "sse": net.SSE, "scse": net.SCSE}.get(args.excitation, None)
        print(excitation_cls)
        pixellink = net.MobileNetV2PixelLink(args.scale, excitation_cls=excitation_cls).to(device)
    optimizer = torch.optim.Adam(pixellink.parameters(), lr=1e-3)

    def step_fn(training):
        def fn(engine, batch):
            if training:
                pixellink.train()
            else:
                pixellink.eval()
            with torch.set_grad_enabled(training):
                images, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_masks = batch
                if training:
                    optimizer.zero_grad()
                images = images.to(device)
                pos_pixel_masks = pos_pixel_masks.to(device)
                neg_pixel_masks = neg_pixel_masks.to(device)
                pixel_weights = pixel_weights.to(device)
                link_masks = link_masks.to(device)
                pixel_input, link_input = pixellink(images)
                # loss_object = net.PixelLinkLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
                loss_object = net.PixelLinkFocalLoss(pixel_input, pos_pixel_masks, neg_pixel_masks, pixel_weights, link_input, link_masks)
                if training:
                    loss_object.loss.backward()
                    optimizer.step()
                return {
                    "loss": loss_object.loss.item(),
                    "loss/pixel": loss_object.pixel_loss.item(),
                    "loss/link": loss_object.link_loss.item(),
                    "accuracy/pixel": loss_object.pixel_accuracy,
                    "accuracy/link": np.mean(loss_object.link_accuracy),
                }
        return fn

    dummy = torch.randn(1, 3, image_size[0], image_size[1], dtype=torch.float).to(device)
    writer = create_summary_writer(pixellink, dummy, os.path.join(args.logdir, "train"))
    test_writer = create_summary_writer(pixellink, dummy, os.path.join(args.logdir, "test"))

    trainer = Engine(step_fn(training=True))
    evaluator = Engine(step_fn(training=False))

    checkpoint_handler = ModelCheckpoint(
        args.checkpoint,
        "networks",
        n_saved=5,
        require_empty=False,
        score_function=lambda engine: -engine.state.metrics["loss"],
        score_name="loss")
    evaluator.add_event_handler(Events.COMPLETED, handler=checkpoint_handler,
                                to_save={"net": pixellink})
    timer = Timer(average=True)

    monitoring_metrics = ["loss", "loss/pixel", "loss/link", "accuracy/pixel", "accuracy/link"]
    for metric in monitoring_metrics:
        def output_transform(m):
            def fn(x):
                return x[m]
            return fn
        RunningAverage(output_transform=output_transform(metric)).attach(trainer, metric)
        RunningAverage(output_transform=output_transform(metric)).attach(evaluator, metric)

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % LOG_FREQ != 0:
            return
        for key, value in engine.state.metrics.items():
            writer.add_scalar(key, value, engine.state.iteration)

        message = "[{epoch}/{max_epoch}][{i}/{max_i}] (train)\t".format(
            epoch=engine.state.epoch,
            max_epoch=args.epochs,
            i=(engine.state.iteration % len(dataloader)),
            max_i=len(dataloader),
        )
        for key, value in engine.state.metrics.items():
            message += ' | {key}: {value}'.format(key=key, value=str(round(value, 5)))
        pbar.log_message(message)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_validation_results(engine):
        if (engine.state.iteration - 1) % LOG_FREQ != 0:
            return
        evaluator.run(test_dataloader)
        for key, value in evaluator.state.metrics.items():
            test_writer.add_scalar(key, value, engine.state.iteration)

        message = "[{epoch}/{max_epoch}][{i}/{max_i}] (test) \t".format(
            epoch=engine.state.epoch,
            max_epoch=args.epochs,
            i=(engine.state.iteration % len(dataloader)),
            max_i=len(dataloader),
        )
        for key, value in evaluator.state.metrics.items():
            message += ' | {key}: {value}'.format(key=key, value=str(round(value, 5)))
        pbar.log_message(message)

    timer.attach(trainer, start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_time(engine):
        pbar.log_message("Epoch {} done. Time per batch: {:.3f}[s]".format(
            engine.state.epoch,
            timer.value(),
        ))
        timer.reset()

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")
            checkpoint_handler(engine, {"net": pixellink})
        else:
            raise e

    trainer.run(dataloader, args.epochs)
    writer.close()
    test_writer.close()

main()
