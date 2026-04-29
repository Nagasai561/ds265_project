# simclr(args=, model=, optimizer=, scheduler=, criterion=)
# args should contain, epochs, device, fp16_precision, batch_size

import argparse
import os
import torch
from dataset import ContrastiveLearningDataset
from dataloader import ClusteredSimCLRLoader
from resnet import ResNetSimCLR
from simclr import SimCLR
import warnings
import time

warnings.filterwarnings("ignore")

model_names = ["resnet18", "resnet50"]

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--out-dim', default=128, type=int,
                    help='feature dimension of projection head (default: 128)')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='softmax temperature for InfoNCE loss (default: 0.5)')
parser.add_argument('--fp16-precision', action='store_true',
                    help='enable mixed precision training when running on CUDA')
parser.add_argument('--disable-cuda', action='store_true',
                    help='force training on CPU even if CUDA is available')
parser.add_argument('--kmeans-epochs', default='1', type=str,
                    help='KMeans schedule: int cadence (e.g. 5), comma list (e.g. 0,3,7), or off')
parser.add_argument('--kmeans-iters', default=10, type=int,
                    help='number of KMeans refinement iterations')
parser.add_argument('--num-clusters', default=None, type=int,
                    help='optional fixed cluster count; if omitted, auto-derived from dataset size and batch size')
parser.add_argument('--file-name', default='run', type=str,
                    help='base name for outputs: metrics saved to ./metrics/<name>.csv and weights to ./weights/<name>.pth')


def parse_kmeans_epochs(value: str):
    value = value.strip()
    if value.lower() in {'off', 'none', 'false', 'disable', 'disabled'}:
        return None
    if ',' in value:
        return [int(v.strip()) for v in value.split(',') if v.strip()]
    return int(value)


def main():
    args = parser.parse_args()

    file_name = args.file_name.strip()
    if not file_name:
        raise ValueError('--file-name must be a non-empty string')

    os.makedirs('./metrics', exist_ok=True)
    os.makedirs('./weights', exist_ok=True)
    args.metrics_csv_path = os.path.join('./metrics', f'{file_name}.csv')
    args.weights_path = os.path.join('./weights', f'{file_name}.pth')

    if (not args.disable_cuda) and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.device.type != 'cuda':
        args.fp16_precision = False

    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name)

    kmeans_epochs = parse_kmeans_epochs(args.kmeans_epochs)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    train_loader = ClusteredSimCLRLoader(
        train_dataset,
        model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        num_clusters=args.num_clusters,
        kmeans_epochs=kmeans_epochs,
        kmeans_iters=args.kmeans_iters,
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)

    print(f"Starting training with step: {args.kmeans_epochs}")
    start_time = time.time()
    simclr.train(train_loader)
    end_time = time.time()

    with open("timing.log", "a") as file:
        file.write(f"{args.kmeans_epochs},{end_time-start_time}\n")


if __name__ == "__main__":
    main()

