from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def simclr_collate(batch: Sequence[Tuple[Sequence[torch.Tensor], int]]) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Collate SimCLR samples into shape [2B, C, H, W]."""
	views_a = torch.stack([sample[0][0] for sample in batch], dim=0)
	views_b = torch.stack([sample[0][1] for sample in batch], dim=0)
	images = torch.cat([views_a, views_b], dim=0)
	labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
	return images, labels


class ClusteredSimCLRLoader:
	"""Simple SimCLR loader that reclusters data every N epochs."""

	def __init__(
		self,
		dataset,
		encoder: nn.Module,
		*,
		device: torch.device,
		batch_size: int,
		num_workers: int = 4,
		move_to_device: bool = True,
		num_clusters: Optional[int] = None,
		kmeans_epochs: Optional[int | Sequence[int]] = 1,
		kmeans_iters: int = 10,
		embedding_batch_size: int = 256,
		seed: int = 0,
	) -> None:
		self.dataset = dataset
		self.encoder = encoder.to(device)
		self.device = device
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.pin_memory = self.device.type == "cuda"
		self.drop_last = True
		self.move_to_device = move_to_device
		self.kmeans_iters = kmeans_iters
		self.embedding_batch_size = embedding_batch_size
		self.seed = seed
		self.kmeans_enabled = kmeans_epochs is not None

		if kmeans_epochs is None:
			self.kmeans_every_n_epochs = None
			self.kmeans_epoch_set = None
		elif isinstance(kmeans_epochs, int):
			if kmeans_epochs <= 0:
				raise ValueError("kmeans_epochs as int must be > 0")
			self.kmeans_every_n_epochs: Optional[int] = kmeans_epochs
			self.kmeans_epoch_set: Optional[set[int]] = None
		else:
			epoch_list = sorted({int(ep) for ep in kmeans_epochs if int(ep) >= 0})
			if not epoch_list:
				raise ValueError("kmeans_epochs list must contain at least one non-negative epoch")
			self.kmeans_every_n_epochs = None
			self.kmeans_epoch_set = set(epoch_list)

		n = len(self.dataset)
		default_clusters = max(2, math.ceil(n / max(1, batch_size)))
		self.num_clusters = min(n, max(2, default_clusters if num_clusters is None else int(num_clusters)))

		self._assignments = [i for i in range(n)]
		self._batches: List[List[int]] = []
		self._loader: Optional[DataLoader] = None

		self.set_epoch(0)

	@torch.no_grad()
	def _extract_embeddings(self) -> torch.Tensor:
		was_training = self.encoder.training
		self.encoder.eval()

		embeddings: List[torch.Tensor] = []
		total = len(self.dataset)
		for start in range(0, total, self.embedding_batch_size):
			end = min(start + self.embedding_batch_size, total)
			x = torch.stack([self.dataset[i][0][0] for i in range(start, end)], dim=0)
			x = x.to(self.device, non_blocking=True)
			z = F.normalize(self.encoder(x), dim=1)
			embeddings.append(z.detach().cpu())

		if was_training:
			self.encoder.train()

		return torch.cat(embeddings, dim=0)

	def _kmeans(self, x: torch.Tensor, epoch: int) -> List[int]:
		n = x.shape[0]
		k = max(1, min(self.num_clusters, n))
		g = torch.Generator(device=x.device)
		g.manual_seed(self.seed + epoch)

		centroids = x[torch.randperm(n, generator=g, device=x.device)[:k]].clone()
		assignments = torch.zeros(n, dtype=torch.long, device=x.device)

		for _ in range(self.kmeans_iters):
			distances = torch.cdist(x, centroids)
			new_assignments = distances.argmin(dim=1)
			if torch.equal(assignments, new_assignments):
				break
			assignments = new_assignments
			for cluster_id in range(k):
				mask = assignments == cluster_id
				if mask.any():
					centroids[cluster_id] = x[mask].mean(dim=0)

		return assignments.cpu().tolist()

	def _build_diverse_batches(self, epoch: int) -> List[List[int]]:
		rng = random.Random(self.seed + epoch)
		buckets: Dict[int, Deque[int]] = defaultdict(deque)
		for idx, cluster_id in enumerate(self._assignments):
			buckets[int(cluster_id)].append(idx)

		for cluster_id in buckets:
			indices = list(buckets[cluster_id])
			rng.shuffle(indices)
			buckets[cluster_id] = deque(indices)

		batches: List[List[int]] = []
		while True:
			active = [cid for cid in buckets if buckets[cid]]
			if not active:
				break

			batch: List[int] = []
			while len(batch) < self.batch_size and active:
				rng.shuffle(active)
				next_active: List[int] = []
				for cid in active:
					if len(batch) >= self.batch_size:
						break
					if buckets[cid]:
						batch.append(buckets[cid].popleft())
					if buckets[cid]:
						next_active.append(cid)
				active = next_active

			if len(batch) == self.batch_size:
				batches.append(batch)
			elif batch and not self.drop_last:
				batches.append(batch)

		rng.shuffle(batches)
		return batches

	def _rebuild_loader(self, epoch: int) -> None:
		self._batches = self._build_diverse_batches(epoch)

		self._loader = DataLoader(
			self.dataset,
			batch_sampler=self._batches,
			num_workers=self.num_workers,
			pin_memory=self.pin_memory,
			collate_fn=simclr_collate,
		)

	def set_epoch(self, epoch: int) -> None:
		do_kmeans = False
		if self.kmeans_enabled:
			if self.kmeans_every_n_epochs is not None:
				do_kmeans = (epoch % self.kmeans_every_n_epochs == 0)
			else:
				do_kmeans = epoch in self.kmeans_epoch_set

		if do_kmeans:
			embeddings = self._extract_embeddings()
			self._assignments = self._kmeans(embeddings, epoch)
		self._rebuild_loader(epoch)

	def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
		if self._loader is None:
			self.set_epoch(0)
		assert self._loader is not None
		for images, labels in self._loader:
			if self.move_to_device:
				images = images.to(self.device, non_blocking=True)
				labels = labels.to(self.device, non_blocking=True)
			yield images, labels

	def __len__(self) -> int:
		return len(self._batches)
