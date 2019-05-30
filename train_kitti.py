import os
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, IoU
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from lilanet.datasets import KITTI, Normalize, Compose, RandomHorizontalFlip
from lilanet.datasets.transforms import ToTensor
from lilanet.model import LiLaNet


def get_data_loaders(data_dir, batch_size, val_batch_size, num_workers):
    normalize = Normalize(mean=KITTI.mean(), std=KITTI.std())
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    val_transforms = Compose([
        ToTensor(),
        normalize
    ])

    train_loader = DataLoader(KITTI(root=data_dir, split='train', transform=transforms),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(KITTI(root=data_dir, split='val', transform=val_transforms),
                            batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def run(args):
    train_loader, val_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.val_batch_size,
                                                args.num_workers)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = KITTI.num_classes()
    model = LiLaNet(num_classes)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Using %d GPU(s)" % device_count)
        model = nn.DataParallel(model)
        args.batch_size = device_count * args.batch_size
        args.val_batch_size = device_count * args.val_batch_size

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=KITTI.class_weights(), reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (Epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    def _prepare_batch(batch, non_blocking=True):
        distance, reflectivity, target = batch

        return (convert_tensor(distance, device=device, non_blocking=non_blocking),
                convert_tensor(reflectivity, device=device, non_blocking=non_blocking),
                convert_tensor(target, device=device, non_blocking=non_blocking))

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        distance, reflectivity, target = _prepare_batch(batch)
        pred = model(distance, reflectivity)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        return loss.item()

    trainer = Engine(_update)

    checkpoint_handler = ModelCheckpoint(args.output_dir, 'checkpoint', save_interval=1, n_saved=10,
                                         require_empty=False, create_dir=True, save_as_state_dict=False)
    # attach running average metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    # attach progress bar
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        checkpoint = {'model': model.state_dict(), 'epoch': trainer.state.epoch,
                      'optimizer': optimizer.state_dict()}
        key = 'epoch{}'.format(engine.state.epoch)
        checkpoint_handler(engine, {key: checkpoint})

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            distance, reflectivity, target = _prepare_batch(batch)
            pred = model(distance, reflectivity)

            return pred, target

    evaluator = Engine(_inference)
    cm = ConfusionMatrix(num_classes)
    IoU(cm, ignore_index=0).attach(evaluator, 'IoU')
    Loss(criterion).attach(evaluator, 'loss')

    def _global_step_transform(engine, event_name):
        return trainer.state.iteration

    tb_logger = TensorboardLogger(args.log_dir)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag='training',
                                               metric_names=['loss']),
                     event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='validation',
                                               metric_names=['loss', 'IoU'],
                                               global_step_transform=_global_step_transform),
                     event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        engine.state.exception_raised = False
        if args.resume:
            engine.state.epoch = args.start_epoch

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        pbar.log_message('Start Validation - Epoch: [{}/{}]'.format(engine.state.epoch, engine.state.max_epochs))
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        iou = metrics['IoU']
        mean_iou = iou.mean()

        pbar.log_message('Validation results - Epoch: [{}/{}]: Loss: {:.2e}, mIoU: {:.1f}'
                         .format(engine.state.epoch, engine.state.max_epochs, loss, mean_iou * 100.0))

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        engine.state.exception_raised = True
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")

            checkpoint = {'model': model.state_dict(), 'epoch': trainer.state.epoch,
                          'optimizer': optimizer.state_dict()}
            checkpoint_handler(engine, {'checkpoint_exception': checkpoint})
        else:
            raise e

    @trainer.on(Events.COMPLETED)
    def save_final_model(engine):
        if not engine.state.exception_raised:
            checkpoint_handler(engine, {'final': model, 'final_state_dict': model.state_dict()})

    print("Start training")
    trainer.run(train_loader, max_epochs=args.epochs)
    tb_logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('LiLaNet with PyTorch')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=10,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='manual seed')
    parser.add_argument('--output-dir', default='./checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log-dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--dataset-dir", type=str, default="../data/kitti",
                        help="location of the dataset")

    run(parser.parse_args())
