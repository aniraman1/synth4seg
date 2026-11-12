"""Branching structures grown as a 3D tree IFS.

Each branch is the image of its parent under one of k contractive affine maps
    w_i(x) = s_i * R_i * x + t
where t moves the origin to the parent's tip, R_i tilts the direction by a branch
angle theta_i about an axis perpendicular to the parent (with a phyllotaxis roll),
and s_i < 1 shrinks the length. Radii follow Murray's law: r_parent^3 = sum r_child^3.
The attractor (iterated to finite depth) is rasterized as tapered capsules.
"""
from dataclasses import dataclass

import numpy as np

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))


@dataclass
class Segment:
    a: np.ndarray  # start point (z, y, x)
    b: np.ndarray  # end point
    ra: float      # radius at a
    rb: float      # radius at b
    depth: int


def _perp_basis(d):
    """Two unit vectors orthogonal to unit vector d."""
    helper = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def grow_tree(root, direction, length, radius, rng, max_depth=5,
              n_children=(2, 3), angle_deg=(30, 50), scale=(0.62, 0.8),
              min_radius=0.6, tortuosity=0.08, n_sub=3):
    """Return a list of Segments forming one connected branching tree."""
    segments = []
    d0 = np.asarray(direction, float)
    d0 /= np.linalg.norm(d0)

    def branch(p, d, L, r, depth, roll):
        k = rng.integers(n_children[0], n_children[1] + 1) if depth < max_depth else 0
        r_child = r * k ** (-1 / 3) if k else r * 0.7  # Murray's law (equal split)

        # Draw this branch as a slightly wiggly polyline; interior points jittered only,
        # so the end point (where children attach) is exact.
        end = p + d * L
        u, v = _perp_basis(d)
        pts = [p]
        for j in range(1, n_sub):
            t = j / n_sub
            off = rng.normal(size=2) * tortuosity * L
            pts.append(p + d * L * t + off[0] * u + off[1] * v)
        pts.append(end)
        radii = np.linspace(r, r_child, n_sub + 1)
        for j in range(n_sub):
            segments.append(Segment(pts[j], pts[j + 1], radii[j], radii[j + 1], depth))

        if k == 0 or r_child < min_radius:
            return
        # Tip direction follows the last sub-segment so branches look continuous.
        d_tip = pts[-1] - pts[-2]
        d_tip /= np.linalg.norm(d_tip)
        u, v = _perp_basis(d_tip)
        for i in range(k):
            theta = np.deg2rad(rng.uniform(*angle_deg))
            phi = roll + 2 * np.pi * i / k + rng.normal(0, 0.2)
            axis_dir = np.cos(phi) * u + np.sin(phi) * v
            d_new = np.cos(theta) * d_tip + np.sin(theta) * axis_dir  # R_i applied to d
            s = rng.uniform(*scale)                                     # s_i
            r_i = r_child * rng.uniform(0.9, 1.1)
            branch(end, d_new / np.linalg.norm(d_new), L * s, r_i, depth + 1,
                   roll + GOLDEN_ANGLE)

    branch(np.asarray(root, float), d0, length, radius, 0, rng.uniform(0, 2 * np.pi))
    return segments


def rasterize(segments, shape, min_draw_radius=0.9):
    """Rasterize segments as tapered capsules into a bool mask."""
    mask = np.zeros(shape, bool)
    shape_arr = np.array(shape)
    for s in segments:
        ra, rb = max(s.ra, min_draw_radius), max(s.rb, min_draw_radius)
        rmax = max(ra, rb) + 1
        lo = np.maximum(np.floor(np.minimum(s.a, s.b) - rmax).astype(int), 0)
        hi = np.minimum(np.ceil(np.maximum(s.a, s.b) + rmax).astype(int) + 1, shape_arr)
        if np.any(hi <= lo):
            continue
        g = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)],
                                 indexing="ij"), -1).astype(float)
        ab = s.b - s.a
        t = np.clip(((g - s.a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        closest = s.a + t[..., None] * ab
        dist = np.linalg.norm(g - closest, axis=-1)
        rad = ra + t * (rb - ra)
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= dist <= rad
    return mask
