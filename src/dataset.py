"""On-the-fly training data: a PyTorch IterableDataset around SynthGenerator.

    from src.dataset import SynthDataset

    ds = SynthDataset(batch_size=4, size=160, device="cuda", seed=0)
    for image, target in ds:                 # endless stream of fresh batches
        loss = criterion(model(image), target)

Every item is a whole batch, already on `device`, so iterate the dataset directly or wrap
it with `DataLoader(ds, batch_size=None, num_workers=0)` (see `synth_loader`) if your
framework (e.g. Lightning) wants a DataLoader.

Seeding. The stream for each (seed, distributed rank, DataLoader worker, epoch) is
independent, so DDP ranks never see the same volumes. Call `ds.set_epoch(e)` at the start
of each epoch (like DistributedSampler) to get a fresh, reproducible stream per epoch;
without it, a dataset iterated in the main process just keeps drawing from where it left
off. `seed=None` draws fresh OS entropy (non-reproducible, never repeats).

Devices. With device="cuda" (the default when available) generate in the main process:
CUDA cannot be used in forked DataLoader workers, and the GPU generator is far faster than
any number of CPU workers anyway. With device="cpu", `num_workers > 0` runs one full
generator per worker process.
"""
import inspect
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .synth_gpu import SynthGenerator, count_range

TARGETS = ("semantic", "labels", "both")
# Output ids put the structures to segment first and body / background last.
SEMANTIC_CLASSES = ("blob", "tree", "body", "background")
_GEN_DEFAULTS = inspect.signature(SynthGenerator.__init__).parameters


def _dist_rank():
    """(rank, local_rank) from torch.distributed if initialised, else torchrun env vars."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = int(os.environ.get("RANK", 0))
    return rank, int(os.environ.get("LOCAL_RANK", 0))


class SynthDataset(IterableDataset):
    """Endless (or `steps`-long) stream of synthetic (image, target) batches.

    Args:
        batch_size: volumes per yielded batch.
        steps: batches per pass over the dataset (one "epoch"). None = infinite. Also
            sets len(ds), which some training loops / progress bars need.
        target: "semantic" -> (image, semantic)
                "labels"   -> (image, labels)
                "both"     -> (image, labels, semantic)
            Ids (body and background always last; differs from SynthGenerator's raw ids):
                semantic: 0 blob, 1 tree, 2 body, 3 background
                labels:   per example, with n, t = that example's blob / tree counts:
                          0 .. n-1 blobs, n .. n+t-1 trees, n+t body, n+t+1 background
            With fixed int counts n, t are the same for every example. With
            n_blobs=(lo, hi) / n_trees=(lo, hi) each example draws its own counts, so the
            body / background ids differ between examples (see `last_counts`, `label_type`).
        device: "cuda", "cuda:1", "mps", "cpu" or None. None picks cuda:LOCAL_RANK under
            torchrun, else cuda -> mps -> cpu.
        seed: base seed (int) or None for fresh entropy.
        image_dtype: dtype of the yielded image, e.g. torch.float16 / bfloat16 for AMP.
        threads_per_worker: torch CPU threads in each DataLoader worker (CPU device only).
        **gen_kwargs: forwarded to SynthGenerator (n_blobs, n_trees, blob_radius,
            roughness, compile, bright_foreground, ...).

    Shapes: image (B, 1, S, S, S); labels / semantic (B, S, S, S) int64 ids.
    """

    def __init__(self, batch_size=2, size=160, steps=None, target="semantic", device=None, seed=None, image_dtype=torch.float32, threads_per_worker=1,
                 **gen_kwargs):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        self.batch_size = batch_size
        self.size = size
        self.steps = steps
        self.target = target
        self.device = device
        self.seed = seed
        self.image_dtype = image_dtype
        self.threads = threads_per_worker
        self.gen_kwargs = gen_kwargs
        self.epoch = 0
        self._gen = None          # built lazily, inside the process that iterates
        self._gen_key = None      # (rank, worker, epoch) the generator was last seeded for
        self._base_entropy = (np.random.SeedSequence().entropy if seed is None else seed)
        get = lambda k: gen_kwargs.get(k, _GEN_DEFAULTS[k].default)
        # n_blobs / n_trees may be (lo, hi) ranges; self.n_blobs / n_trees are the maxima.
        blob_rng = count_range(get("n_blobs"), "n_blobs")
        tree_rng = count_range(get("n_trees"), "n_trees")
        self.n_blobs, self.n_trees = blob_rng[1], tree_rng[1]
        self.variable_counts = blob_rng[0] < blob_rng[1] or tree_rng[0] < tree_rng[1]
        self.last_counts = None   # (n_blobs (B,), n_trees (B,)) of the last yielded batch
        # Generator semantic: 0 bg, 1 body, 2 blob, 3 tree.
        self._semantic_lut = torch.tensor([3, 2, 0, 1])

    # ------------------------------------------------------------------ helpers
    @property
    def n_label_classes(self):
        """Maximum number of instance ids: max n_blobs + max n_trees + body + background.
        With variable counts an example has n_b + t_b + 2 (see `last_counts`)."""
        return self.n_blobs + self.n_trees + 2

    @property
    def n_semantic_classes(self):
        return len(SEMANTIC_CLASSES)

    @property
    def n_classes(self):
        """Output channels a segmentation net needs ("both": the semantic count)."""
        return self.n_label_classes if self.target == "labels" else self.n_semantic_classes

    def label_type(self, k, b=None):
        """'blob' | 'tree' | 'body' | 'background' for output instance id `k`.

        b: index of the example in the last yielded batch; needed with variable counts,
        where ids depend on that example's counts. None uses the maximum counts.
        """
        if b is None:
            if self.variable_counts:
                raise ValueError("variable counts: pass b, the example index in the batch")
            n, t = self.n_blobs, self.n_trees
        else:
            n, t = (int(c[b]) for c in self.last_counts)
        if not 0 <= k < n + t + 2:
            raise ValueError(f"label id {k} out of range [0, {n + t + 2})")
        if k < n:
            return "blob"
        if k < n + t:
            return "tree"
        return "body" if k == n + t else "background"

    def set_epoch(self, epoch):
        """Switch to the (reproducible) stream for `epoch`. Call before iterating."""
        self.epoch = int(epoch)

    def __len__(self):
        if self.steps is None:
            raise TypeError("SynthDataset is infinite; pass steps=... to give it a length")
        return self.steps

    def _resolve_device(self, local_rank):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank % torch.cuda.device_count())
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def generator(self):
        """The underlying SynthGenerator for this process (built on first use)."""
        rank, local_rank = _dist_rank()
        wi = get_worker_info()
        worker = wi.id if wi is not None else 0
        if self._gen is None:
            dev = self._resolve_device(local_rank)
            if wi is not None and dev.type != "cpu":
                raise RuntimeError(
                    f"SynthDataset on {dev} cannot run inside DataLoader workers. Use "
                    "num_workers=0 (generation is already asynchronous on the GPU), or "
                    "device='cpu' for CPU worker processes.")
            if wi is not None:
                torch.set_num_threads(self.threads)
            self._gen = SynthGenerator(size=self.size, device=dev, **self.gen_kwargs)
            self._semantic_lut = self._semantic_lut.to(dev)
        key = (rank, worker, self.epoch)
        if key != self._gen_key:
            self._gen.reseed(np.random.SeedSequence([self._base_entropy, rank, worker, self.epoch]))
            self._gen_key = key
        return self._gen

    def _split_steps(self):
        """Steps this worker produces, so all workers together yield exactly `steps`."""
        if self.steps is None:
            return None
        wi = get_worker_info()
        if wi is None:
            return self.steps
        return self.steps // wi.num_workers + (wi.id < self.steps % wi.num_workers)

    # ------------------------------------------------------------------ iteration
    def _semantic(self, semantic):
        return self._semantic_lut[semantic]

    def _labels(self, labels, nb, nt):
        """Generator ids -> per-example compact ids (blobs, trees, body, background)."""
        gen = self._gen
        B, N, T = len(nb), gen.N, gen.T
        # Per-example lookup table indexed by generator id: 0 bg, 1 body, 2..N+1 blob slots,
        # N+2..N+T+1 tree slots. Unused slots never occur; they map to background.
        lut = np.empty((B, N + T + 2), np.int64)
        lut[:] = (nb + nt + 1)[:, None]                               # bg (+ unused slots)
        lut[:, 1] = nb + nt                                           # body
        slot_b, slot_t = np.arange(N), np.arange(T)
        lut[:, 2:N + 2] = np.where(slot_b < nb[:, None], slot_b, lut[:, :1])
        lut[:, N + 2:] = np.where(slot_t < nt[:, None], nb[:, None] + slot_t, lut[:, :1])
        lut = torch.from_numpy(lut).to(labels.device)
        return lut.gather(1, labels.view(B, -1)).view_as(labels)

    def __iter__(self):
        gen = self.generator()
        n = self._split_steps()
        i = 0
        while n is None or i < n:
            image, labels, semantic = gen.sample(self.batch_size)
            image = image.to(self.image_dtype)
            nb, nt = gen.last_n_blobs, gen.last_n_trees
            self.last_counts = (nb, nt)
            if self.target == "semantic":
                yield image, self._semantic(semantic)
            elif self.target == "labels":
                yield image, self._labels(labels, nb, nt)
            else:
                yield image, self._labels(labels, nb, nt), self._semantic(semantic)
            i += 1


def synth_loader(num_workers=0, pin_memory=None, prefetch_factor=2, **dataset_kwargs):
    """DataLoader over a SynthDataset (for frameworks that require a DataLoader).

    GPU/MPS: keep num_workers=0 (batches come straight from the device). CPU: use
    num_workers > 0 to generate in parallel processes; pin_memory defaults to True then if
    CUDA is available, so `.to("cuda", non_blocking=True)` is fast.
    """
    ds = SynthDataset(**dataset_kwargs)
    cpu = ds.device is not None and torch.device(ds.device).type == "cpu"
    if pin_memory is None:
        pin_memory = cpu and torch.cuda.is_available()
    extra = dict(persistent_workers=True, prefetch_factor=prefetch_factor) if num_workers else {}
    return DataLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=pin_memory,
                      **extra)
