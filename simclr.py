import csv
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

torch.manual_seed(0)

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self, train_loader):

        scaler = torch.amp.GradScaler(self.args.device)
        # self.model = torch.compile(self.model)

        metrics_file = None
        metrics_writer = None

        if self.args.metrics_csv_path:
            metrics_dir = os.path.dirname(self.args.metrics_csv_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            metrics_file = open(self.args.metrics_csv_path, "w", newline="")
            metrics_writer = csv.writer(metrics_file)
            metrics_writer.writerow(["epoch", "top1_accuracy", "top5_accuracy"])
            metrics_file.flush()

        try:
            for epoch_counter in tqdm(range(self.args.epochs), desc="Epochs", unit="epoch"):
                if hasattr(train_loader, "set_epoch"):
                    train_loader.set_epoch(epoch_counter)

                last_top1 = None
                last_top5 = None

                for images, _ in tqdm(train_loader, leave=False, desc="Batches", unit="batch"):
                    with torch.autocast(device_type=self.args.device.type, enabled=self.args.fp16_precision):
                        features = self.model(images)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    top1, top5 = self.accuracy(logits, labels, topk=(1, 5))
                    last_top1 = float(top1[0].item())
                    last_top5 = float(top5[0].item())

                if metrics_writer is not None and ((epoch_counter + 1) % 5 == 0):
                    metrics_writer.writerow([
                        epoch_counter,
                        last_top1,
                        last_top5,
                    ])
                    metrics_file.flush()

                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
        finally:
            if metrics_file is not None:
                metrics_file.close()

        if getattr(self.args, "weights_path", None):
            torch.save(self.model.state_dict(), self.args.weights_path)
