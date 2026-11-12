"""Fast "smash" generator: low-res noise blobs + trees painted on top.

1. Blobs: K independent low-res Gaussian noise fields (noise_res^3) are upsampled
   trilinearly to full size. Each voxel goes to the field that is largest there (argmax
   binning). This partitions the whole volume into smooth, irregular blob regions.
   Binning a *single* field by quantiles makes nested onion rings instead of blobs.
   With split_blobs=True, every connected region gets its own label. A random
   `drop_frac` of the blobs is then sent to background (label 0).
2. Trees: IFS trees are rasterized and painted in order. Each new tree overwrites
   whatever is underneath, including blobs and earlier trees, so earlier trees can be
   cut into pieces.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .trees import grow_tree, rasterize

CONN26 = np.ones((3, 3, 3), bool)


def noise_blobs(size, n_classes, noise_res, rng, split_blobs=True, min_voxels=1000):
    shape = (size,) * 3
    low = rng.standard_normal((n_classes,) + (noise_res,) * 3, dtype=np.float32)
    # Two-stage upsample: cubic to an intermediate grid (cheap, gives smooth curved
    # boundaries), then trilinear to full size. Corners are aligned so the volume edges
    # aren't clamped into flat slabs.
    mid = min(size, max(4 * noise_res, 32))
    z = (mid - 1) / (noise_res - 1)
    mid_f = np.stack([ndimage.zoom(f, z, order=3) for f in low])  # per field: 3D cubic only
    fields = F.interpolate(torch.from_numpy(mid_f)[None], size=shape, mode="trilinear",
                           align_corners=True)[0].numpy()
    # Running argmax over the K fields (much faster than torch argmax over the slow axis).
    best = fields[0].copy()
    cls = np.zeros(shape, np.uint16)
    for k in range(1, n_classes):
        win = fields[k] > best
        cls[win] = k
        np.maximum(best, fields[k], out=best)
    if not split_blobs:
        return cls + 1, n_classes

    labels = np.zeros(shape, np.uint16)
    nxt = 1
    for k in range(n_classes):
        lab, n = ndimage.label(cls == k, structure=CONN26)
        if n == 0:
            continue
        fg = lab > 0
        labels[fg] = lab[fg] + (nxt - 1)
        nxt += n
    # Merge tiny slivers (e.g. corners clipped by the volume edge) into the neighbour
    # they touch most. Each check runs only inside that sliver's bounding box.
    sizes = np.bincount(labels.ravel(), minlength=nxt)
    objs = ndimage.find_objects(labels)
    for k in np.flatnonzero(sizes < min_voxels):
        if k == 0 or objs[k - 1] is None:
            continue
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in objs[k - 1])
        crop = labels[sl]
        m = crop == k
        ring = ndimage.binary_dilation(m, CONN26) & ~m
        crop[m] = np.bincount(crop[ring]).argmax()
    # Relabel to consecutive ids 1..n with a lookup table (no full sort).
    present = np.bincount(labels.ravel(), minlength=nxt) > 0
    present[0] = False
    lut = np.zeros(len(present), np.uint16)
    lut[present] = np.arange(1, present.sum() + 1)
    return lut[labels], int(present.sum())


def drop_blobs(labels, n, drop_frac, rng):
    """Randomly send a fraction of blob labels to background (0), relabel the rest 1..m."""
    keep = rng.random(n) >= drop_frac
    lut = np.zeros(n + 1, np.uint16)
    lut[1:][keep] = np.arange(1, keep.sum() + 1)
    return lut[labels], int(keep.sum())


def make_volume_smash(size=160, n_classes=6, noise_res=4, n_trees=8, seed=0,
                      split_blobs=True, drop_frac=0.5):
    rng = np.random.default_rng(seed)
    shape = (size,) * 3
    labels, n_blobs = noise_blobs(size, n_classes, noise_res, rng, split_blobs)
    if drop_frac > 0:
        labels, n_blobs = drop_blobs(labels, n_blobs, drop_frac, rng)
    meta = {0: {"type": "background"}}
    meta.update({i: {"type": "blob"} for i in range(1, n_blobs + 1)})

    nxt = n_blobs + 1
    for _ in range(n_trees):
        root = rng.uniform(0.15, 0.85, size=3) * size
        d = rng.normal(size=3)
        depth = int(rng.integers(4, 7))
        L = rng.uniform(0.09, 0.14) * size
        r0 = rng.uniform(2.0, 3.5)
        segs = grow_tree(root, d, L, r0, rng, max_depth=depth)
        mask = rasterize(segs, shape)
        labels[mask] = nxt  # newest tree wins over blobs and earlier trees
        meta[nxt] = {"type": "tree", "root": root.round(1).tolist(), "depth": depth,
                     "trunk_length": round(L, 1), "trunk_radius": round(r0, 2),
                     "n_segments": len(segs)}
        nxt += 1

    counts = np.bincount(labels.ravel(), minlength=nxt)
    for k in meta:
        meta[k]["voxels"] = int(counts[k])
    return labels, meta
