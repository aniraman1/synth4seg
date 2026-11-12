"""Compose blobs and IFS trees into a single integer label volume."""
import numpy as np
from scipy import ndimage

from .blobs import make_blob
from .trees import grow_tree, rasterize

CONN26 = np.ones((3, 3, 3), bool)


def _largest_component(mask):
    lab, n = ndimage.label(mask, structure=CONN26)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def _stays_connected(labels, ids, sl):
    """True if every label in `ids` is still a single 26-connected piece (checked inside slice sl)."""
    for i in ids:
        if ndimage.label(labels[sl] == i, structure=CONN26)[1] > 1:
            return False
    return True


def make_volume(size=160, n_blobs=10, n_trees=6, seed=0, max_tries=60,
                blob_overlap=0.35):
    """Dense packing. Blobs may overlap each other (later blob wins) by up to `blob_overlap`
    of either blob. Trees are painted over blobs and always take precedence. Trees never
    cross other trees, and no placement may split an existing structure into pieces."""
    rng = np.random.default_rng(seed)
    shape = (size,) * 3
    c = np.array(shape) / 2
    labels = np.zeros(shape, np.uint8)
    meta = {0: {"type": "background"}}

    # Label 1: enclosing "body" so the background tissue is non-trivial.
    body = make_blob(shape, c, np.array([0.44, 0.40, 0.42]) * size, rng, roughness=0.08)
    labels[body] = 1
    meta[1] = {"type": "body"}
    tree_zone = np.zeros(shape, bool)  # existing trees dilated by 1 voxel: no-go for new trees
    next_id = 2

    def place(make_fn, kind):
        nonlocal next_id
        for _ in range(max_tries):
            mask, params = make_fn()
            raw = mask.sum()
            mask &= body
            mask = _largest_component(mask)
            n = mask.sum()
            # Reject if clipping to the body cut most of it away (e.g. a tree reduced to a stump).
            if n < 0.7 * raw or n < 50:
                continue
            if (mask & tree_zone).any():  # nothing may cut through an existing tree
                continue
            hit_ids, hit_counts = np.unique(labels[mask], return_counts=True)
            keep = hit_ids >= 2
            hit_ids, hit_counts = hit_ids[keep], hit_counts[keep]
            if kind == "blob" and hit_ids.size:
                if hit_counts.sum() / n > blob_overlap or any(
                        cnt / (labels == i).sum() > blob_overlap
                        for i, cnt in zip(hit_ids, hit_counts)):
                    continue
            idx = np.argwhere(mask)
            sl = tuple(slice(max(a - 1, 0), b + 2) for a, b in zip(idx.min(0), idx.max(0)))
            before = labels[sl].copy()
            labels[mask] = next_id
            if hit_ids.size and not _stays_connected(labels, hit_ids, sl):
                labels[sl] = before  # undo: it would fragment something already placed
                continue
            if kind == "tree":
                tree_zone[ndimage.binary_dilation(mask, CONN26)] = True
            meta[next_id] = {"type": kind, **params}
            next_id += 1
            return True
        print(f"warning: could not place a {kind} after {max_tries} tries")
        return False

    def blob_fn():
        r = rng.uniform(0.05, 0.11) * size * rng.uniform(0.7, 1.3, size=3)
        ctr = c + rng.uniform(-0.3, 0.3, size=3) * size
        rough = rng.uniform(0.25, 0.5)
        return make_blob(shape, ctr, r, rng, roughness=rough), {
            "center": ctr.round(1).tolist(), "radii": r.round(1).tolist(), "roughness": round(rough, 3)}

    def tree_fn():
        root = c + rng.uniform(-0.3, 0.3, size=3) * size
        # Grow roughly toward the volume centre so the tree stays inside the body.
        d = (c - root) / (np.linalg.norm(c - root) + 1e-6) + rng.normal(0, 0.5, size=3)
        depth = int(rng.integers(4, 7))
        L = rng.uniform(0.09, 0.14) * size
        r0 = rng.uniform(2.0, 3.5)
        segs = grow_tree(root, d, L, r0, rng, max_depth=depth)
        return rasterize(segs, shape), {
            "root": root.round(1).tolist(), "depth": depth, "trunk_length": round(L, 1),
            "trunk_radius": round(r0, 2), "n_segments": len(segs)}

    # Paint order: blobs first, then trees on top so trees take precedence.
    for _ in range(n_blobs):
        place(blob_fn, "blob")
    for _ in range(n_trees):
        place(tree_fn, "tree")
    counts = np.bincount(labels.ravel(), minlength=next_id)
    for k in meta:
        meta[k]["voxels"] = int(counts[k])
    return labels, meta
