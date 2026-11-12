"""How to feed SynthGenerator into training. Three setups:

A) One or more GPUs (recommended). Each process / DDP rank owns a generator on its own GPU
   with its own seed. Volumes are made inside the training loop, so there's no DataLoader
   and no host->device copy of volumes:

       rank = int(os.environ.get("LOCAL_RANK", 0))
       gen = SynthGenerator(size=160, device=f"cuda:{rank}", seed=1000 + rank)
       for step in range(n_steps):
           image, labels, semantic = gen.sample(batch_size)   # already on cuda:rank
           loss = criterion(model(image), semantic)          # or `labels` for instances
           ...

   Launch: torchrun --nproc_per_node=<n_gpus> train.py

B) GPU + CPU workers. Only the (tiny) tree skeletons are grown in DataLoader workers,
   overlapping with GPU work; the GPU does everything voxel-sized. Rarely needed: the
   skeletons cost about 2 ms/volume.

C) CPU-only nodes. Each DataLoader worker runs a full generator on the CPU with 1-2
   threads. Throughput scales with cores.

Run this file to benchmark C locally:  python train_example.py <num_workers>
"""
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from synthvol.synth_gpu import SynthGenerator


class TreeSkeletons(IterableDataset):
    """(B) CPU workers grow tree skeletons; pass the result to gen.sample(B, tree_pts=...)."""

    def __init__(self, batch_size, size=160, n_trees=8, seed=0):
        self.B, self.size, self.T, self.seed = batch_size, size, n_trees, seed

    def __iter__(self):
        from synthvol.synth_gpu import grow_trees_points
        wi = get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0])
        while True:
            parts = []
            for b in range(self.B):
                P, R, t = grow_trees_points(rng, self.size, self.T)
                parts.append((P, R, t, np.full(len(P), b, np.int64)))
            yield [np.concatenate(x) for x in zip(*parts)]


class CPUVolumes(IterableDataset):
    """(C) each worker generates whole volumes on the CPU."""

    def __init__(self, size=160, seed=0, threads_per_worker=1):
        self.size, self.seed, self.threads = size, seed, threads_per_worker

    def __iter__(self):
        torch.set_num_threads(self.threads)
        wi = get_worker_info()
        gen = SynthGenerator(size=self.size, device="cpu",
                             seed=self.seed * 1000 + (wi.id if wi else 0))
        while True:
            image, labels, semantic = gen.sample(1)
            yield image[0], semantic[0]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dl = DataLoader(CPUVolumes(), batch_size=4, num_workers=nw, persistent_workers=True,
                    prefetch_factor=2)
    it = iter(dl)
    for _ in range(2 * nw + 4):  # drain the prefetch queue so we time steady state
        next(it)
    t = time.perf_counter()
    n = 0
    for _ in range(12):
        img, sem = next(it)
        n += img.shape[0]
    dt = time.perf_counter() - t
    print(f"CPU workers={nw}: {n / dt:.1f} volumes/s ({dt / n:.2f} s/volume), batch {tuple(img.shape)}")
