"""Benchmark SynthGenerator.sample() per stage.  python bench.py [--device cuda] [--size 160] [--batch 1 2 4 8]"""
import argparse
import time

import torch

from src.synth_gpu import SynthGenerator


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def timed(dev, fn, *a):
    sync(dev)
    t = time.perf_counter()
    out = fn(*a)
    sync(dev)
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    gen = SynthGenerator(size=args.size, device=args.device, seed=0)
    dev = gen.device
    print(f"device={dev} ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else dev.type}), "
          f"size={args.size}, torch {torch.__version__}")
    print(f"{'B':>3} {'total ms':>9} {'ms/vol':>8} {'vol/s':>7} | {'blobs':>7} {'skel':>7} {'trees':>7} "
          f"{'image':>7} | peak GB")
    for B in args.batch:
        for _ in range(args.warmup):
            gen.sample(B)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        tot = {k: 0.0 for k in ("all", "blobs", "skel", "trees", "image")}
        for _ in range(args.iters):
            _, t_all = timed(dev, gen.sample, B)
            tot["all"] += t_all
            labels, t = timed(dev, gen.blobs, B); tot["blobs"] += t
            pts, t = timed(dev, gen.tree_points, B); tot["skel"] += t
            labels, t = timed(dev, gen.paint_trees, labels, pts); tot["trees"] += t
            labels = torch.where(gen.body, labels, 0)
            _, t = timed(dev, gen.synth_image, labels); tot["image"] += t
        ms = {k: 1e3 * v / args.iters for k, v in tot.items()}
        peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else float("nan")
        print(f"{B:>3} {ms['all']:>9.1f} {ms['all'] / B:>8.1f} {1e3 * B / ms['all']:>7.1f} | "
              f"{ms['blobs']:>7.1f} {ms['skel']:>7.1f} {ms['trees']:>7.1f} {ms['image']:>7.1f} | {peak:.2f}")


if __name__ == "__main__":
    main()
