"""Batched, device-agnostic (CUDA / MPS / CPU) on-the-fly generator for training.

GPU port of the original body + blobs + trees method:
  * body (label 1): a slightly rough ellipsoid, varied per sample.
  * blobs (labels 2..n_blobs+1): rough, randomly rotated ellipsoids placed inside the body.
    Each blob is star-shaped: it is inside where |q| < 1 + roughness * h(q/|q|), with q in
    the blob's rotated, radius-scaled frame and h a smooth random function of direction
    (random Fourier features). That keeps every blob connected and hole-free by
    construction. Each blob is evaluated only inside a fixed-size crop around it (static
    shapes), and later blobs overwrite earlier ones.
  * trees: IFS skeletons are grown on the CPU (vectorized numpy, about 1 ms per volume),
    starting inside the body and growing inwards. They are splatted on the device as
    balls. Trees are painted last, in order, so a later tree overwrites blobs and earlier
    trees.
  Blobs and trees are clipped to the body.
  * image: SynthSeg-style GMM -> per-sample Gaussian blur -> smooth bias field -> noise.
Two code paths:
  * CUDA / CPU (fast): all blobs evaluated in one batched call and written with a single
    scatter_reduce(amax); trees with one amax-scatter per radius bucket; blur as banded-matrix
    matmuls; no host syncs in sample(). On CUDA the star-shape test is fused with torch.compile.
  * MPS (compat): only basic ops (MPS lacks 3D interpolate, scatter_reduce, max_pool3d and
    cummin); padded flat buffers with dummy slots, power-of-two point groups.
Benchmark: python bench.py --batch 1 4 8 16
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))


# --------------------------------------------------------------------------- utils
def cubic_interp_matrix(n_in, n_out):
    """(n_out, n_in) Catmull-Rom interpolation matrix, corners aligned."""
    x = np.linspace(0, n_in - 1, n_out)
    i0 = np.floor(x).astype(int)
    t = x - i0
    w = np.stack([
        (-t ** 3 + 2 * t ** 2 - t) / 2,
        (3 * t ** 3 - 5 * t ** 2 + 2) / 2,
        (-3 * t ** 3 + 4 * t ** 2 + t) / 2,
        (t ** 3 - t ** 2) / 2,
    ], 1)
    W = np.zeros((n_out, n_in))
    for j, off in enumerate((-1, 0, 1, 2)):
        np.add.at(W, (np.arange(n_out), np.clip(i0 + off, 0, n_in - 1)), w[:, j])
    return W.astype(np.float32)


def upsample3d(low, W):
    """low: (..., r, r, r), W: (S, r) -> (..., S, S, S) via three separable matmuls."""
    x = torch.einsum("...ijk,zk->...ijz", low, W)
    x = torch.einsum("...ijz,yj->...iyz", x, W)
    return torch.einsum("...iyz,xi->...xyz", x, W).contiguous()


# --------------------------------------------------------------------------- fused (CUDA)
def star_field_ew(G, off, A, f, ph, amp, rough):
    """Elementwise form of SynthGenerator._star_field (same signature, (K,V) bool result).
    Written so torch.compile fuses it into a single kernel: no (K,V,M) intermediate."""
    g = [G[None, :, j] + off[:, j, None] for j in range(3)]               # (K,V) each
    q = [g[0] * A[:, 0, j, None] + g[1] * A[:, 1, j, None] + g[2] * A[:, 2, j, None]
         for j in range(3)]
    rho = torch.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) + 1e-6
    inv = 1 / rho
    h = torch.zeros_like(rho)
    for k in range(f.shape[1]):
        h = h + amp[:, k, None] * torch.cos(
            (q[0] * f[:, k, 0, None] + q[1] * f[:, k, 1, None] + q[2] * f[:, k, 2, None]) * inv
            + ph[:, k, None])
    return rho < 1 + rough[:, None] * h.clamp(-1.5, 1.5)


def crop_scatter_args(inside, G, lo, bi, ids, S: int):
    """Flat target index and amax value for every crop voxel (out-of-volume -> value 0)."""
    vox = lo[:, None, :] + G[None].long()                                 # (K,V,3)
    valid = ((vox >= 0) & (vox < S)).all(-1)
    vox = vox.clamp(0, S - 1)
    idx = ((bi[:, None] * S + vox[..., 0]) * S + vox[..., 1]) * S + vox[..., 2]
    return idx, torch.where(inside & valid, ids[:, None], 0)


# --------------------------------------------------------------------------- trees (CPU)
def _perp_basis(d):
    helper = np.where(np.abs(d[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, np.cross(d, u)


def grow_trees_points(rng, size, n_trees, step=0.5, n_sub=3, tortuosity=0.08,
                      min_radius=0.6, angle_deg=(30, 50), scale=(0.62, 0.8),
                      length=(0.09, 0.14), radius=(2.0, 3.5), depth=(4, 6), inward=False):
    """Grow n_trees IFS trees level by level (vectorized over all branches).

    Returns float32 arrays (pts (N,3), radii (N,), tree_idx (N,)) of dense points along
    every branch, `step` voxels apart. With inward=True, roots sit within +-0.3*size of the
    centre and trunks point roughly towards it (so trees stay inside a body).
    """
    T = n_trees
    if inward:
        c = np.full(3, size / 2)
        p = c + rng.uniform(-0.3, 0.3, (T, 3)) * size
        d = (c - p) / (np.linalg.norm(c - p, axis=1, keepdims=True) + 1e-6) + rng.normal(0, 0.5, (T, 3))
    else:
        p = rng.uniform(0.15, 0.85, (T, 3)) * size
        d = rng.normal(size=(T, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    L = rng.uniform(*length, T) * size
    r = rng.uniform(*radius, T)
    roll = rng.uniform(0, 2 * np.pi, T)
    tid = np.arange(T)
    max_depth = rng.integers(depth[0], depth[1] + 1, T)

    seg_a, seg_b, seg_ra, seg_rb, seg_t = [], [], [], [], []
    level = 0
    while len(p):
        last = level >= max_depth[tid]
        k = np.where(last, 0, rng.integers(2, 4, len(p)))
        r_child = np.where(k > 0, r * np.maximum(k, 1) ** (-1 / 3), r * 0.7)

        # Wiggly polyline: jitter interior points perpendicular to d; endpoints exact.
        u, v = _perp_basis(d)
        ts = np.linspace(0, 1, n_sub + 1)
        pts = p[:, None] + d[:, None] * (L[:, None, None] * ts[None, :, None])
        off = rng.normal(size=(len(p), n_sub + 1, 2)) * tortuosity * L[:, None, None]
        off[:, [0, -1]] = 0
        pts += off[..., :1] * u[:, None] + off[..., 1:] * v[:, None]
        rr = r[:, None] + (r_child - r)[:, None] * ts[None]
        seg_a.append(pts[:, :-1].reshape(-1, 3)); seg_b.append(pts[:, 1:].reshape(-1, 3))
        seg_ra.append(rr[:, :-1].ravel()); seg_rb.append(rr[:, 1:].ravel())
        seg_t.append(np.repeat(tid, n_sub))

        # Children: apply the k contractive maps w_i to every growing branch.
        grow = (k > 0) & (r_child >= min_radius)
        if not grow.any():
            break
        kk = k[grow]
        tip = pts[grow, -1]
        dt = pts[grow, -1] - pts[grow, -2]
        dt /= np.linalg.norm(dt, axis=1, keepdims=True)
        rep = np.repeat(np.arange(len(kk)), kk)
        child_i = np.arange(len(rep)) - np.repeat(np.cumsum(kk) - kk, kk)  # 0..k-1 per parent
        u, v = _perp_basis(dt)
        n = len(rep)
        theta = np.deg2rad(rng.uniform(*angle_deg, n))
        phi = roll[grow][rep] + 2 * np.pi * child_i / kk[rep] + rng.normal(0, 0.2, n)
        axis = np.cos(phi)[:, None] * u[rep] + np.sin(phi)[:, None] * v[rep]
        d = np.cos(theta)[:, None] * dt[rep] + np.sin(theta)[:, None] * axis
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = tip[rep]
        L = L[grow][rep] * rng.uniform(*scale, n)
        r = r_child[grow][rep] * rng.uniform(0.9, 1.1, n)
        roll = roll[grow][rep] + GOLDEN_ANGLE
        tid = tid[grow][rep]
        level += 1

    a, b = np.concatenate(seg_a), np.concatenate(seg_b)
    ra, rb, t = np.concatenate(seg_ra), np.concatenate(seg_rb), np.concatenate(seg_t)
    # Dense points along each segment.
    seg_len = np.linalg.norm(b - a, axis=1)
    sp = np.maximum(step, 0.5 * np.minimum(ra, rb))  # thick parts: fewer, bigger balls
    cnt = np.maximum(np.ceil(seg_len / sp).astype(int), 1) + 1
    sid = np.repeat(np.arange(len(a)), cnt)
    start = np.repeat(np.cumsum(cnt) - cnt, cnt)
    f = (np.arange(cnt.sum()) - start) / (cnt[sid] - 1)
    P = a[sid] + (b - a)[sid] * f[:, None]
    R = ra[sid] + (rb - ra)[sid] * f
    return P.astype(np.float32), R.astype(np.float32), t[sid].astype(np.int64)


# --------------------------------------------------------------------------- generator
def count_range(n, name="count"):
    """int n -> (n, n); (lo, hi) -> (lo, hi), inclusive. Requires 0 <= lo <= hi, hi >= 1."""
    lo, hi = (n, n) if np.ndim(n) == 0 else tuple(n)
    lo, hi = int(lo), int(hi)
    if not 0 <= lo <= hi or hi < 1:
        raise ValueError(f"{name} must be an int >= 1 or a (lo, hi) range with "
                         f"0 <= lo <= hi and hi >= 1, got {n!r}")
    return lo, hi


class SynthGenerator:
    """gen = SynthGenerator(device="cuda"); image, labels, semantic = gen.sample(4)

    image    (B,1,S,S,S) float32 in [0,1]
    labels   (B,S,S,S)   int64: 0 background, 1 body, 2..N+1 blobs,
                         N+2 .. N+T+1 trees (gen.tree_base = N+1)
    semantic (B,S,S,S)   int64: 0 background, 1 body, 2 blob, 3 tree

    n_blobs / n_trees: an int, or an inclusive (lo, hi) range drawn independently for each
    example. N and T above are the maxima (gen.N, gen.T): an example with k blobs uses
    ids 2..k+1, and its other blob and tree slots are simply absent. The counts drawn for
    the last batch are in gen.last_n_blobs / gen.last_n_trees, each an int array of shape (B,).
    """

    def __init__(self, size=160, n_blobs=10, n_trees=6, blob_radius=(0.05, 0.11),
                 roughness=(0.25, 0.5), body_radii=(0.44, 0.40, 0.42), body_roughness=0.08,
                 n_freq=12, freq=(1.5, 3.5), min_draw_radius=0.9, device=None, seed=None,
                 compile=None, bright_foreground=True, bg_mean_range=(0, 110),
                 fg_mean_range=(140, 255)):
        self.S = size
        # Shapes are sized for the maximum counts; unused slots are switched off per example.
        self.n_blobs_range = count_range(n_blobs, "n_blobs")
        self.n_trees_range = count_range(n_trees, "n_trees")
        self.N = self.n_blobs_range[1]
        self.T = self.n_trees_range[1]
        self.last_n_blobs = self.last_n_trees = None
        self.blob_radius = blob_radius
        self.rough = roughness
        self.body_radii = np.asarray(body_radii)
        self.body_rough = body_roughness
        self.M = n_freq
        self.freq = freq
        self.min_r = min_draw_radius
        self.tree_base = self.N + 1
        # Intensity priors. bright_foreground: background/body means are drawn from
        # bg_mean_range and blob/tree means from fg_mean_range, so every structure to be
        # segmented is brighter than its surroundings. Otherwise all means are U(0, 255).
        self.bright_fg = bright_foreground
        self.bg_mean_range = bg_mean_range
        self.fg_mean_range = fg_mean_range
        # Fixed crop for every blob: covers radius * 1.3 (axis jitter) * (1 + 1.5*max rough).
        rmax = blob_radius[1] * 1.3 * size * (1 + 1.5 * roughness[1])
        self.crop = int(2 * np.ceil(rmax) + 3)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                              "mps" if torch.backends.mps.is_available() else "cpu"))
        dev = self.device
        # MPS lacks scatter_reduce, so it keeps the padded/dummy-slot path. CUDA and CPU use
        # batched blobs and amax-scatters (blob ids increase with paint order and tree ids
        # exceed blob ids, so "later wins" is exactly "largest id wins").
        self.fast = dev.type != "mps"
        self.chunk_elems = 1 << 26  # cap on (voxels x Fourier terms) per field evaluation
        # torch.compile fuses the blob/body field into one kernel (needs Triton + a C
        # compiler). compile=None: on for CUDA, off elsewhere. Falls back to eager on failure.
        self.compiled = (dev.type == "cuda") if compile is None else bool(compile)
        if self.compiled:
            self._field_c = torch.compile(star_field_ew, dynamic=True)
            self._scatter_args_c = torch.compile(crop_scatter_args, dynamic=True)
        # Random numbers are drawn directly on the device (no host->device copies).
        try:
            self.g = torch.Generator(device=dev)
        except RuntimeError:  # backends without device generators
            self.g = None
        self.reseed(seed)
        self.W_bias = torch.from_numpy(cubic_interp_matrix(4, size)).to(dev)
        ax = torch.arange(size, dtype=torch.float32, device=dev)
        self.ax = ax
        if self.fast:
            grid = lambda n: torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32,
                                                                     device=dev)] * 3, indexing="ij"), -1).view(-1, 3)
            self.grid_full, self.grid_crop = grid(size), grid(self.crop)
        # Ball offsets for splatting, one table per integer radius bucket.
        self.offsets = {}
        for R in range(1, 6):
            o = torch.stack(torch.meshgrid(*[torch.arange(-R, R + 1)] * 3, indexing="ij"), -1)
            o = o.reshape(-1, 3)
            self.offsets[R] = o[(o.float().norm(dim=1) <= R + 0.87)].int().to(dev)

    def reseed(self, seed=None):
        """Restart both random streams (CPU numpy + device torch) from `seed`.

        `seed` may be anything np.random.default_rng accepts (int, list of ints,
        SeedSequence, None for fresh OS entropy). Cheap: compiled kernels are kept.
        """
        self.np_rng = np.random.default_rng(seed)
        torch_seed = int(self.np_rng.integers(2**62))
        if self.g is not None:
            self.g.manual_seed(torch_seed)
        else:
            torch.manual_seed(torch_seed)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.g, device=self.device)

    # ---------------------------------------------------------------- blobs
    def _shape_params(self, n):
        t = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        return tuple(t(x) for x in self._shape_params_np(n))

    def _shape_params_np(self, n):
        """CPU: random rotation (n,3,3), Fourier directions*freq (n,M,3), phases, amplitudes."""
        rng = self.np_rng
        q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
        q *= np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
        q[np.linalg.det(q) < 0, :, 0] *= -1
        f = rng.normal(size=(n, self.M, 3))
        f *= (rng.uniform(*self.freq, (n, self.M)) / np.linalg.norm(f, axis=2))[..., None]
        ph = rng.uniform(0, 2 * np.pi, (n, self.M))
        amp = rng.normal(size=(n, self.M)) * np.sqrt(2 / self.M)  # h has unit variance
        return q, f, ph, amp

    def _star_inside(self, z, y, x, c, rot, radii, f, ph, amp, rough):
        """Star-shaped blob test. z,y,x: (B,Z),(B,Y),(B,X) coordinates. Returns (B,Z,Y,X) bool."""
        dz = (z - c[:, 0:1])[:, :, None, None]
        dy = (y - c[:, 1:2])[:, None, :, None]
        dx = (x - c[:, 2:3])[:, None, None, :]
        q = [(dz * rot[:, 0, j, None, None, None] + dy * rot[:, 1, j, None, None, None] +
              dx * rot[:, 2, j, None, None, None]) / radii[:, j, None, None, None]
             for j in range(3)]
        rho = torch.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2) + 1e-6
        h = torch.zeros_like(rho)
        for k in range(self.M):  # h(u) = sum_k a_k cos(f_k . u + phase_k), u = q/|q|
            fk = f[:, k, :, None, None, None]
            h = h + amp[:, k, None, None, None] * torch.cos(
                (q[0] * fk[:, 0] + q[1] * fk[:, 1] + q[2] * fk[:, 2]) / rho + ph[:, k, None, None, None])
        return rho < 1 + rough[:, None, None, None] * h.clamp(-1.5, 1.5)

    def _counts(self, B, n, rng_range):
        """Per-example counts (B,) int64: given `n` (int or (B,)), else drawn from rng_range.
        A fixed count draws no random numbers, so fixed-count streams are unchanged."""
        lo, hi = rng_range
        if n is None:
            n = hi if lo == hi else self.np_rng.integers(lo, hi + 1, B)
        n = np.broadcast_to(np.asarray(n, np.int64), (B,)).copy()
        if n.min() < 0 or n.max() > hi:
            raise ValueError(f"counts must be in [0, {hi}], got {n}")
        return n

    @torch.no_grad()
    def blobs(self, B, n_blobs=None):
        """Body + blobs label volume (B,S,S,S). n_blobs: int or (B,) counts; None = draw."""
        nb = self.last_n_blobs = self._counts(B, n_blobs, self.n_blobs_range)
        return self._blobs_fast(B, nb) if self.fast else self._blobs_compat(B, nb)

    @torch.no_grad()
    def paint_trees(self, labels, pts):
        return self._paint_trees_fast(labels, pts) if self.fast else self._paint_trees_compat(labels, pts)

    def _field(self, *args):
        if self.compiled:
            try:
                return self._field_c(*args)
            except Exception as e:  # no Triton / C compiler: stay eager from now on
                import warnings
                warnings.warn(f"torch.compile unavailable ({type(e).__name__}); using eager path")
                self.compiled = False
        return self._star_field(*args)

    def _star_field(self, G, off, A, f, ph, amp, rough):
        """Batched star-shaped test, chunked to bound memory.
        G (V,3) integer grid offsets; off (K,3) grid origin minus centre; A (K,3,3) = rot / radii
        (so q = (G + off) @ A); f (K,M,3); ph, amp (K,M); rough (K,). Returns (K,V) bool."""
        K, V = A.shape[0], G.shape[0]
        out = torch.empty(K, V, dtype=torch.bool, device=G.device)
        n = max(1, self.chunk_elems // (V * self.M))
        for s0 in range(0, K, n):
            sl = slice(s0, s0 + n)
            q = torch.bmm(G[None] + off[sl, None], A[sl])                    # (k,V,3)
            rho = q.norm(dim=-1, keepdim=True) + 1e-6
            ang = torch.baddbmm(ph[sl, None], q / rho, f[sl].transpose(1, 2)).cos_()  # (k,V,M)
            h = torch.bmm(ang, amp[sl, :, None]).squeeze(-1).clamp_(-1.5, 1.5)
            out[sl] = rho.squeeze(-1) < 1 + rough[sl, None] * h
        return out

    def _blobs_fast(self, B, nb):
        S, dev, rng, N, Cs = self.S, self.device, self.np_rng, self.N, self.crop
        # Draw everything on the CPU in the same order as the compat path, then do a single
        # host->device copy so the kernels below run without host syncs (the CPU can grow
        # tree skeletons meanwhile).
        r_body = self.body_radii * S * rng.uniform(0.9, 1.1, (B, 3))
        _, fb, phb, ampb = self._shape_params_np(B)
        P = []
        for i in range(N):
            ctr = S / 2 + rng.uniform(-0.3, 0.3, (B, 3)) * S
            radii = rng.uniform(*self.blob_radius, (B, 1)) * S * rng.uniform(0.7, 1.3, (B, 3))
            rough = rng.uniform(*self.rough, B)
            lo = np.round(ctr).astype(np.int64) - Cs // 2
            rot, f, ph, amp = self._shape_params_np(B)
            P.append((ctr, radii, rough, lo, rot, f, ph, amp))
        ctr, radii, rough, lo, rot, f, ph, amp = (np.stack(x, 1).reshape(B * N, *x[0].shape[1:])
                                                  for x in zip(*P))      # blob-major per sample
        A = rot / radii[:, None, :]
        flat = np.concatenate([a.reshape(-1) for a in (
            np.eye(3) / r_body[:, None, :], np.full((B, 3), -S / 2), fb, phb, ampb,
            A, lo - ctr, f, ph, amp, rough)]).astype(np.float32)
        # Blob i of example b has id i+2 if i < nb[b], else 0 (an amax no-op: slot unused).
        slot = np.arange(N)
        ids = np.where(slot[None] < nb[:, None], slot[None] + 2, 0).reshape(-1)
        ints = torch.from_numpy(np.concatenate([lo.reshape(-1), ids])).to(dev)
        lo_t, ids = ints[:B * N * 3].view(B * N, 3), ints[B * N * 3:]
        buf = torch.from_numpy(flat).to(dev)
        sizes = [B * 9, B * 3, B * self.M * 3, B * self.M, B * self.M,
                 B * N * 9, B * N * 3, B * N * self.M * 3, B * N * self.M, B * N * self.M, B * N]
        (Ab, offb, fb, phb, ampb, A, off, f, ph, amp, rough) = torch.split(buf, sizes)
        M = self.M
        # Body: full volume, axis-aligned.
        body = self._field(self.grid_full, offb.view(B, 3), Ab.view(B, 3, 3), fb.view(B, M, 3),
                                phb.view(B, M), ampb.view(B, M),
                                torch.full((B,), self.body_rough, device=dev)).view(B, S, S, S)
        # Blobs: fixed-size crops, scattered with amax (blob i has id i+2).
        inside = self._field(self.grid_crop, off.view(-1, 3), A.view(-1, 3, 3),
                                  f.view(-1, M, 3), ph.view(-1, M), amp.view(-1, M), rough)
        bi = torch.arange(B, device=dev).repeat_interleave(N)
        fn = self._scatter_args_c if self.compiled else crop_scatter_args
        idx, val = fn(inside, self.grid_crop, lo_t, bi, ids, S)
        labels = body.view(-1).long()
        labels.scatter_reduce_(0, idx.view(-1), val.view(-1), "amax")
        labels = labels.view(B, S, S, S) * body                           # clip blobs to body
        self.body = body
        return labels

    def _paint_trees_fast(self, labels, pts):
        S, dev = self.S, self.device
        B = labels.shape[0]
        P, R, t, b = pts
        R = np.maximum(R, self.min_r)
        bucket = np.minimum(np.ceil(R).astype(np.int64), max(self.offsets))
        order = np.argsort(bucket, kind="stable")
        counts = np.bincount(bucket, minlength=max(self.offsets) + 1)
        lab = (self.tree_base + 1 + t).astype(np.int64)
        data = np.concatenate([P[order].ravel(), R[order]]).astype(np.float32)
        ints = np.concatenate([b[order], lab[order]]).astype(np.int64)
        pin = dev.type == "cuda"
        data, ints = (torch.from_numpy(x) for x in (data, ints))
        if pin:
            data, ints = data.pin_memory(), ints.pin_memory()
        data, ints = data.to(dev, non_blocking=pin), ints.to(dev, non_blocking=pin)
        n = len(order)
        Pt, Rt, bt, lt = data[:3 * n].view(n, 3), data[3 * n:], ints[:n], ints[n:]
        out = labels.reshape(-1).clone()
        s0 = 0
        for Rb in range(1, max(self.offsets) + 1):  # one amax-scatter per radius bucket
            e0 = s0 + int(counts[Rb])
            if e0 > s0:
                p, r = Pt[s0:e0], Rt[s0:e0]
                vox = p.round().int()[:, None, :] + self.offsets[Rb][None]
                ok = ((vox.float() - p[:, None]).norm(dim=-1) <= r[:, None]) & \
                     ((vox >= 0) & (vox < S)).all(-1)
                vox = vox.long().clamp_(0, S - 1)
                idx = ((bt[s0:e0, None] * S + vox[..., 0]) * S + vox[..., 1]) * S + vox[..., 2]
                out.scatter_reduce_(0, idx.view(-1), torch.where(ok, lt[s0:e0, None], 0).view(-1), "amax")
            s0 = e0
        return out.view(B, S, S, S)

    @torch.no_grad()
    def _blobs_compat(self, B, nb):
        S, dev, rng = self.S, self.device, self.np_rng
        t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(dev)
        # Body: full volume, axis-aligned, radii jittered +-10% per sample.
        c_body = np.full((B, 3), S / 2)
        r_body = self.body_radii * S * rng.uniform(0.9, 1.1, (B, 3))
        rot, f, ph, amp = self._shape_params(B)
        rot = torch.eye(3, device=dev).expand(B, 3, 3).contiguous()
        ax = self.ax.expand(B, S)
        body = self._star_inside(ax, ax, ax, t(c_body), rot, t(r_body), f, ph, amp,
                                 t(np.full(B, self.body_rough)))
        # Blobs: one fixed-size crop per blob, painted in order (later wins), clipped to body.
        # Crop voxels outside the volume write to unique dummy slots past the end of the
        # buffer, so every write in a crop goes to a distinct address (no races).
        Cs = self.crop
        n_vox = B * S ** 3
        buf = torch.zeros(n_vox + B * Cs ** 3, dtype=torch.int64, device=dev)
        buf[:n_vox] = body.view(-1).long()                          # 1 = body
        labels = buf[:n_vox].view(B, S, S, S)
        body_flat = torch.cat([body.view(-1), torch.zeros(B * Cs ** 3, dtype=torch.bool, device=dev)])
        dummy = (n_vox + torch.arange(B * Cs ** 3, device=dev)).view(B, Cs, Cs, Cs)
        ar = torch.arange(Cs, device=dev, dtype=torch.float32)
        bidx = torch.arange(B, device=dev, dtype=torch.int64)[:, None, None, None]
        active = torch.from_numpy(np.arange(self.N)[None] < nb[:, None]).to(dev)  # (B,N)
        for i in range(self.N):
            ctr = S / 2 + rng.uniform(-0.3, 0.3, (B, 3)) * S
            radii = rng.uniform(*self.blob_radius, (B, 1)) * S * rng.uniform(0.7, 1.3, (B, 3))
            rough = rng.uniform(*self.rough, B)
            lo = np.round(ctr).astype(np.int64) - Cs // 2               # crop origin (B,3)
            rot, f, ph, amp = self._shape_params(B)
            zz, yy, xx = (t(lo[:, j:j + 1]) + ar[None] for j in range(3))  # (B,Cs) each
            inside = self._star_inside(zz, yy, xx, t(ctr), rot, t(radii), f, ph, amp, t(rough))
            zi, yi, xi = (v.long() for v in (zz, yy, xx))
            valid = ((zi >= 0) & (zi < S))[:, :, None, None] & ((yi >= 0) & (yi < S))[:, None, :, None] \
                & ((xi >= 0) & (xi < S))[:, None, None, :]
            idx = ((bidx * S + zi.clamp(0, S - 1)[:, :, None, None]) * S +
                   yi.clamp(0, S - 1)[:, None, :, None]) * S + xi.clamp(0, S - 1)[:, None, None, :]
            idx = torch.where(valid, idx, dummy)
            ok = inside & body_flat[idx] & active[:, i, None, None, None]
            # Write the blob id where ok; elsewhere rewrite the current value (a no-op).
            buf[idx.view(-1)] = torch.where(ok, i + 2, buf[idx]).view(-1)
        self.body = body
        return labels

    # ---------------------------------------------------------------- trees
    def tree_points(self, B, n_trees=None):
        """CPU: dense skeleton points for B volumes (cheap; can live in DataLoader workers).
        n_trees: int or (B,) counts; None = draw. Returns [P, R, tree slot, batch index]."""
        nt = self.last_n_trees = self._counts(B, n_trees, self.n_trees_range)
        empty = [np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                 np.zeros(0, np.int64), np.zeros(0, np.int64)]
        if self.fast:  # grow all trees in one vectorized call (per-level cost is mostly overhead)
            total = int(nt.sum())
            if total == 0:
                return empty
            P, R, t = grow_trees_points(self.np_rng, self.S, total, inward=True)
            b_of = np.repeat(np.arange(B), nt)                        # tree -> example
            slot = np.arange(total) - np.repeat(np.cumsum(nt) - nt, nt)  # tree -> slot in example
            return [P, R, slot[t], b_of[t]]
        out = []
        for b in range(B):
            if nt[b] == 0:
                continue
            P, R, t = grow_trees_points(self.np_rng, self.S, int(nt[b]), inward=True)
            out.append((P, R, t, np.full(len(P), b, np.int64)))
        return [np.concatenate(x) for x in zip(*out)] if out else empty

    @torch.no_grad()
    def _paint_trees_compat(self, labels, pts):
        """Splat tree points as balls. No data-dependent shapes, so there are no host syncs:
        points are grouped on the CPU by (tree, radius bucket), and voxels outside the ball or
        the volume are written to a dummy slot at the end of a padded flat buffer."""
        S, dev = self.S, self.device
        B = labels.shape[0]
        P, R, t, b = pts
        if len(P) == 0:
            return labels
        R = np.maximum(R, self.min_r)
        bucket = np.minimum(np.ceil(R).astype(np.int64), max(self.offsets))
        order = np.lexsort((bucket, t))  # sort by tree, then bucket
        key = t[order] * 16 + bucket[order]
        cuts = np.flatnonzero(np.diff(key)) + 1
        starts = np.r_[0, cuts]; ends = np.r_[cuts, len(key)]
        # Pad every group to a power-of-two length so only a few tensor shapes ever occur
        # (MPS compiles a graph per new shape). Pad points sit far outside the volume.
        P, R, b = P[order], R[order], b[order].astype(np.int32)
        Pp, Rp, bp, new_s, new_e = [], [], [], [], []
        pos = 0
        for s0, e0 in zip(starts, ends):
            n = e0 - s0
            m = 1 << int(np.ceil(np.log2(max(n, 64))))
            Pp.append(np.concatenate([P[s0:e0], np.full((m - n, 3), -1e4, np.float32)]))
            Rp.append(np.concatenate([R[s0:e0], np.ones(m - n, np.float32)]))
            bp.append(np.concatenate([b[s0:e0], np.zeros(m - n, np.int32)]))
            new_s.append(pos); new_e.append(pos + m); pos += m
        keys = key[starts]
        starts, ends = new_s, new_e
        P, R, b = (torch.from_numpy(np.concatenate(x)).to(dev) for x in (Pp, Rp, bp))
        n_vox = B * S ** 3
        D = 1 << 16  # dummy slots: rejected writes spread out (one shared slot = write contention)
        buf = torch.empty(n_vox + D, dtype=labels.dtype, device=dev)
        buf[:n_vox] = labels.view(-1)
        spread = torch.arange(D, dtype=torch.int32, device=dev) + n_vox
        for g, (s0, e0) in enumerate(zip(starts, ends)):  # tree-major: later trees overwrite
            ti, Rb = divmod(int(keys[g]), 16)
            off = self.offsets[Rb]
            p, r, bb = P[s0:e0], R[s0:e0], b[s0:e0]
            vox = p.round().int()[:, None, :] + off[None]               # (n, m, 3) int32
            ok = ((vox.float() - p[:, None]).norm(dim=-1) <= r[:, None]) & \
                 ((vox >= 0) & (vox < S)).all(-1)
            idx = ((bb[:, None] * S + vox[..., 0]) * S + vox[..., 1]) * S + vox[..., 2]  # int32: B*S^3 < 2^31
            idx = torch.where(ok, idx, spread.repeat(-(-idx.numel() // D))[:idx.numel()].view_as(idx))
            buf.index_put_((idx.view(-1).long(),), torch.as_tensor(self.tree_base + 1 + ti, device=dev))
        return buf[:n_vox].view(B, S, S, S)

    # ---------------------------------------------------------------- image
    @torch.no_grad()
    def synth_image(self, labels):
        B, S, dev = labels.shape[0], self.S, self.device
        L = self.tree_base + self.T + 1  # static: avoids a host sync on labels.max()
        if self.bright_fg:
            fg = torch.arange(L, device=dev) > 1                     # blobs + trees
            lo = torch.where(fg, self.fg_mean_range[0], self.bg_mean_range[0]).float()
            hi = torch.where(fg, self.fg_mean_range[1], self.bg_mean_range[1]).float()
            means = lo + (hi - lo) * self._rand(B, L)
        else:
            means = self._rand(B, L) * 255
        stds = self._rand(B, L) * 25
        flat = labels.view(B, -1)
        img = means.gather(1, flat) + stds.gather(1, flat) * self._randn(B, S ** 3)
        img = img.view(1, B, S, S, S)
        # Per-sample Gaussian blur (sigma 0.5-1.5) as a grouped separable conv.
        sig = 0.5 + self._rand(B)
        rad = 5
        x = torch.arange(-rad, rad + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-x[None] ** 2 / (2 * sig[:, None] ** 2))
        k = (k / k.sum(1, keepdim=True)).view(B, 1, -1)
        if self.fast:
            # Same blur as three batched matmuls with a banded (S,S) matrix per sample;
            # replicate padding is folded in by clamping the column index.
            src = (torch.arange(S, device=dev)[:, None] + x.long()[None]).clamp_(0, S - 1)  # (S,2r+1)
            Wb = torch.zeros(B, S, S, device=dev)
            Wb.scatter_add_(2, src.expand(B, S, -1), k.view(B, 1, -1).expand(B, S, -1).contiguous())
            img = img.view(B, S, S, S)
            img = torch.matmul(img, Wb.transpose(1, 2)[:, None])                          # x
            img = torch.matmul(Wb[:, None], img)                                          # y
            img = torch.matmul(Wb, img.view(B, S, S * S)).view(B, S, S, S)                # z
            return self._finish_image(img)
        # Replicate padding (zero padding would darken the volume borders).
        img = F.conv3d(F.pad(img, (0, 0, 0, 0, rad, rad), mode="replicate"),
                       k.view(B, 1, -1, 1, 1), groups=B)
        img = F.conv3d(F.pad(img, (0, 0, rad, rad, 0, 0), mode="replicate"),
                       k.view(B, 1, 1, -1, 1), groups=B)
        img = F.conv3d(F.pad(img, (rad, rad, 0, 0, 0, 0), mode="replicate"),
                       k.view(B, 1, 1, 1, -1), groups=B)
        return self._finish_image(img.view(B, S, S, S))

    def _finish_image(self, img):
        B, S = img.shape[0], self.S
        bias = torch.exp(upsample3d(self._randn(B, 4, 4, 4) * 0.3, self.W_bias))
        img = img * bias + self._randn(B, S, S, S) * (2 + 6 * self._rand(B, 1, 1, 1))
        lo = img.view(B, -1).amin(1).view(B, 1, 1, 1)
        hi = img.view(B, -1).amax(1).view(B, 1, 1, 1)
        return ((img - lo) / (hi - lo + 1e-8))[:, None]

    # ---------------------------------------------------------------- all
    @torch.no_grad()
    def sample(self, B, tree_pts=None, n_blobs=None, n_trees=None):
        """n_blobs / n_trees: optional int or (B,) per-example counts overriding the
        configured range (ignored for trees when tree_pts is given)."""
        labels = self.blobs(B, n_blobs)
        if tree_pts is not None:
            self.last_n_trees = None  # unknown: the caller grew the skeletons
        else:
            tree_pts = self.tree_points(B, n_trees)
        labels = self.paint_trees(labels, tree_pts)
        labels = torch.where(self.body, labels, 0)                 # clip trees to the body
        semantic = torch.where(labels > self.tree_base, 3,
                               torch.where(labels > 1, 2, labels))  # 0 bg, 1 body, 2 blob, 3 tree
        return self.synth_image(labels), labels, semantic

    def label_type(self, k):
        """'background' | 'body' | 'blob' | 'tree' for a label id."""
        if k == 0:
            return "background"
        if k == 1:
            return "body"
        return "tree" if k > self.tree_base else "blob"
