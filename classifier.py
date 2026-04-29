import argparse
import csv
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from resnet import ResNetSimCLR


def parse_args():
	parser = argparse.ArgumentParser(description="Linear evaluation on top of pretrained SimCLR encoder")
	parser.add_argument("--data", metavar="DIR", default="./datasets", help="path to dataset")
	parser.add_argument(
		"--dataset-name",
		default="cifar10",
		choices=["stl10", "cifar10"],
		help="dataset name for linear evaluation",
	)
	parser.add_argument(
		"-a",
		"--arch",
		metavar="ARCH",
		default="resnet18",
		choices=["resnet18", "resnet50"],
		help="encoder architecture",
	)
	parser.add_argument("--out-dim", default=128, type=int, help="SimCLR projection output dimension")
	parser.add_argument("--weights-path", required=True, type=str, help="path to pretrained SimCLR .pth")
	parser.add_argument("--epochs", default=100, type=int, help="number of linear-eval epochs")
	parser.add_argument("-b", "--batch-size", default=256, type=int, help="mini-batch size")
	parser.add_argument("-j", "--workers", default=8, type=int, help="number of data loading workers")
	parser.add_argument("--lr", default=0.1, type=float, help="classifier learning rate")
	parser.add_argument("--wd", "--weight-decay", default=0.0, type=float, dest="weight_decay")
	parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
	parser.add_argument("--disable-cuda", action="store_true", help="force training on CPU")
	parser.add_argument("--fp16-precision", action="store_true", help="enable mixed precision on CUDA")
	parser.add_argument("--seed", default=0, type=int, help="random seed")
	parser.add_argument(
		"--file-name",
		default="",
		type=str,
		help="file path to save metrics",
	)
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def build_transforms(dataset_name: str):
	if dataset_name == "cifar10":
		train_transform = transforms.Compose(
			[
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
			]
		)
		test_transform = transforms.Compose([transforms.ToTensor()])
		return train_transform, test_transform

	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(96, padding=12),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		]
	)
	test_transform = transforms.Compose([transforms.ToTensor()])
	return train_transform, test_transform


def build_dataloaders(args):
	train_transform, test_transform = build_transforms(args.dataset_name)

	if args.dataset_name == "cifar10":
		train_dataset = datasets.CIFAR10(args.data, train=True, transform=train_transform, download=False)
		test_dataset = datasets.CIFAR10(args.data, train=False, transform=test_transform, download=False)
		num_classes = 10
	else:
		train_dataset = datasets.STL10(args.data, split="train", transform=train_transform, download=False)
		test_dataset = datasets.STL10(args.data, split="test", transform=test_transform, download=False)
		num_classes = 10

	pin_memory = args.device.type == "cuda"
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers,
		pin_memory=pin_memory,
		drop_last=False,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=pin_memory,
		drop_last=False,
	)
	return train_loader, test_loader, num_classes


def clean_state_dict_keys(state_dict):
	cleaned = {}
	for key, value in state_dict.items():
		new_key = key[7:] if key.startswith("module.") else key
		cleaned[new_key] = value
	return cleaned


def load_simclr_encoder(args):
	model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
	ckpt = torch.load(args.weights_path, map_location="cpu")

	if isinstance(ckpt, dict) and "state_dict" in ckpt:
		state_dict = ckpt["state_dict"]
	else:
		state_dict = ckpt

	state_dict = clean_state_dict_keys(state_dict)
	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"Warning: missing keys when loading pretrained weights: {missing}")
	if unexpected:
		print(f"Warning: unexpected keys when loading pretrained weights: {unexpected}")

	return model.backbone


class LinearEvalModel(nn.Module):
	def __init__(self, backbone: nn.Module, num_classes: int):
		super().__init__()
		if isinstance(backbone.fc, nn.Sequential):
			feature_dim = backbone.fc[0].in_features
		elif isinstance(backbone.fc, nn.Linear):
			feature_dim = backbone.fc.in_features
		else:
			raise ValueError("Unsupported backbone.fc type for linear evaluation")

		backbone.fc = nn.Identity()
		self.backbone = backbone
		for param in self.backbone.parameters():
			param.requires_grad = False

		self.classifier = nn.Linear(feature_dim, num_classes)

	def forward(self, x):
		with torch.no_grad():
			features = self.backbone(x)
		logits = self.classifier(features)
		return logits


def topk_accuracy(logits, labels, topk=(1, 5)):
	with torch.no_grad():
		maxk = max(topk)
		batch_size = labels.size(0)
		_, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
		pred = pred.t()
		correct = pred.eq(labels.view(1, -1).expand_as(pred))

		result = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			result.append(float(correct_k.mul_(100.0 / batch_size).item()))
		return result


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, fp16_precision):
	model.train()
	model.backbone.eval()

	total_loss = 0.0
	total_top1 = 0.0
	total_top5 = 0.0
	total_samples = 0

	for images, labels in tqdm(loader, leave=False, desc="Train", unit="batch"):
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(device_type=device.type, enabled=fp16_precision):
			logits = model(images)
			loss = criterion(logits, labels)

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		batch_size = labels.size(0)
		top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
		total_loss += float(loss.item()) * batch_size
		total_top1 += top1 * batch_size
		total_top5 += top5 * batch_size
		total_samples += batch_size

	return {
		"loss": total_loss / total_samples,
		"top1": total_top1 / total_samples,
		"top5": total_top5 / total_samples,
	}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
	model.eval()

	total_loss = 0.0
	total_top1 = 0.0
	total_top5 = 0.0
	total_samples = 0

	for images, labels in tqdm(loader, leave=False, desc="Eval", unit="batch"):
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, labels)

		batch_size = labels.size(0)
		top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
		total_loss += float(loss.item()) * batch_size
		total_top1 += top1 * batch_size
		total_top5 += top5 * batch_size
		total_samples += batch_size

	return {
		"loss": total_loss / total_samples,
		"top1": total_top1 / total_samples,
		"top5": total_top5 / total_samples,
	}


def main():
	args = parse_args()
	set_seed(args.seed)

	if (not args.disable_cuda) and torch.cuda.is_available():
		args.device = torch.device("cuda")
	else:
		args.device = torch.device("cpu")

	if args.device.type != "cuda":
		args.fp16_precision = False

	os.makedirs("./metrics", exist_ok=True)
	os.makedirs("./weights", exist_ok=True)

	metrics_path = args.file_name

	train_loader, test_loader, num_classes = build_dataloaders(args)
	backbone = load_simclr_encoder(args)
	model = LinearEvalModel(backbone=backbone, num_classes=num_classes).to(args.device)

	criterion = nn.CrossEntropyLoss().to(args.device)
	optimizer = torch.optim.SGD(
		model.classifier.parameters(),
		lr=args.lr,
		momentum=args.momentum,
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	scaler = torch.amp.GradScaler(device=args.device.type)


	with open(metrics_path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow([
			"epoch",
			"train_loss",
			"train_top1",
			"train_top5",
			"test_loss",
			"test_top1",
			"test_top5",
		])

		for epoch in tqdm(range(args.epochs), desc="Epochs", unit="epoch"):
			train_metrics = train_one_epoch(
				model,
				train_loader,
				optimizer,
				criterion,
				scaler,
				args.device,
				args.fp16_precision,
			)
			test_metrics = evaluate(model, test_loader, criterion, args.device)

			writer.writerow(
				[
					epoch,
					train_metrics["loss"],
					train_metrics["top1"],
					train_metrics["top5"],
					test_metrics["loss"],
					test_metrics["top1"],
					test_metrics["top5"],
				]
			)
			f.flush()
			scheduler.step()




if __name__ == "__main__":
	main()
