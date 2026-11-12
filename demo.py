"""Generate a synthetic 3D volume of blobs + IFS branching trees.

Usage:
    python demo.py [--method smash|place] --size 160 --seed 0 --out outputs [--show]
      smash (default): --n-classes 6 --noise-res 4 --drop 0.5 --n-trees 8 [--no-split]
      place:           --n-blobs 10 --n-trees 6
      gpu:             --device cuda|mps|cpu --n-blobs 10 --n-trees 6
                       (labels + image together; see train_example.py for training use)
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from src.compose import CONN26, make_volume
from src.intensity import synth_image
from src.smash import make_volume_smash
from src.synth_gpu import SynthGenerator
from src.viz import render_3d, save_slice_gif, save_slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--method", choices=["smash", "place", "gpu"], default="smash")
    ap.add_argument("--device", default=None, help="gpu method: cuda / mps / cpu (default: auto)")
    ap.add_argument("--n-blobs", type=int, default=10, help="place and gpu")
    ap.add_argument("--n-trees", type=int, default=None, help="default: 8 smash, 6 place")
    ap.add_argument("--n-classes", type=int, default=6, help="smash: number of noise fields")
    ap.add_argument("--noise-res", type=int, default=4,
                    help="smash: low-res noise grid size (smaller = bigger blobs)")
    ap.add_argument("--drop", type=float, default=0.5,
                    help="smash: fraction of blobs randomly dropped to background")
    ap.add_argument("--no-split", action="store_true",
                    help="smash: one label per noise class instead of per connected blob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--show", action="store_true", help="open interactive 3D window")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    image = None
    if args.method == "gpu":
        gen = SynthGenerator(size=args.size, n_blobs=args.n_blobs,
                             n_trees=args.n_trees if args.n_trees is not None else 6,
                             device=args.device, seed=args.seed)
        gen.sample(1)  # warm-up (kernel compilation), excluded from timing
        t0 = time.time()
        img_t, lab_t, _ = gen.sample(1)
        labels = lab_t[0].cpu().numpy().astype(np.uint16)
        image = img_t[0, 0].cpu().numpy()
        present = np.bincount(labels.ravel())
        meta = {int(k): {"type": gen.label_type(int(k))} for k in np.flatnonzero(present)}
        meta.setdefault(0, {"type": "background"})
        meta = dict(sorted(meta.items()))
        for k in meta:
            meta[k]["voxels"] = int(present[k])
        print(f"gpu generator on {gen.device}")
    elif args.method == "smash":
        labels, meta = make_volume_smash(args.size, args.n_classes, args.noise_res,
                                         args.n_trees if args.n_trees is not None else 8,
                                         args.seed, split_blobs=not args.no_split,
                                         drop_frac=args.drop)
    else:
        labels, meta = make_volume(args.size, args.n_blobs,
                                   args.n_trees if args.n_trees is not None else 6, args.seed)
    t1 = time.time()
    if image is None:
        surround = [k for k, m in meta.items() if m["type"] in ("background", "body")]
        image = synth_image(labels, np.random.default_rng(args.seed + 1), surround_ids=surround)
    t2 = time.time()
    if args.method == "gpu":
        print(f"generated {labels.shape}: labels + image {t1 - t0:.2f}s (incl. copy to host)")
    else:
        print(f"generated {labels.shape}: labels {t1 - t0:.2f}s, image {t2 - t1:.2f}s")

    affine = np.eye(4)
    nib.save(nib.Nifti1Image(labels, affine), out / "labels.nii.gz")
    nib.save(nib.Nifti1Image(image, affine), out / "image.nii.gz")
    (out / "labels.json").write_text(json.dumps({str(k): v for k, v in meta.items()}, indent=2))

    kinds = {}
    for m in meta.values():
        kinds[m["type"]] = kinds.get(m["type"], 0) + 1
    print("labels:", ", ".join(f"{v} {k}" for k, v in kinds.items()))
    if len(meta) <= 30:
        print(f"{'id':>3} {'type':<10} {'voxels':>9} {'components':>10}")
    for k, m in meta.items():
        if len(meta) > 30 and m["type"] != "tree":
            continue
        mask = labels == k
        _, ncomp = ndimage.label(mask, structure=CONN26)
        print(f"{k:>3} {m['type']:<10} {int(mask.sum()):>9} {ncomp:>10}")
    print(f"image range [{image.min():.3f}, {image.max():.3f}]")

    save_slices(labels, image, meta, out / "slices.png")
    save_slice_gif(labels, image, out / "sweep.gif", meta)
    render_3d(labels, meta, str(out / "render3d.png"), interactive=args.show)
    print(f"wrote outputs to {out.resolve()} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
