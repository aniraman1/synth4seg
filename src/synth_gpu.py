"""Batched, device-agnostic (CUDA / MPS / CPU) on-the-fly generator for training.

GPU port of the original body + blobs + trees method:
  * body (label 1): a slightly rough ellipsoid, varied per sample.
  * blobs (labels 2..n_blobs+1): rough, randomly rotated ellipsoids placed inside the body.
    Each blob is star-shaped: it is inside where |q| < 1 + roughness * h(q/|q|), with q in
    the blob's rotated, radius-scaled frame and h a smooth random function of direction
    (random Fourier features). That keeps every blob connected and hole-free by
    construction. Each blob is evaluated only inside a fixed-size crop around it (static
    shapes), and later blobs overwrite earlier ones.
  * trees: IFS skeletons are grown on the CPU (vectorized numpy, about 2 ms per volume),
    starting inside the body and growing inwards. They are splatted on the device as
    balls. Trees are painted last, in order, so a later tree overwrites blobs and earlier
    trees.
  Blobs and trees are clipped to the body.
  * image: SynthSeg-style GMM -> per-sample Gaussian blur -> smooth bias field -> noise.
Only basic ops are used (MPS lacks 3D interpolate, scatter_reduce, max_pool3d and cummin).
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
        child_i = np.concatenate([np.arange(n) for n in kk])
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
class SynthGenerator:
    """gen = SynthGenerator(device="cuda"); image, labels, semantic = gen.sample(4)

    image    (B,1,S,S,S) float32 in [0,1]
    labels   (B,S,S,S)   int64: 0 background, 1 body, 2..n_blobs+1 blobs,
                         n_blobs+2 .. n_blobs+n_trees+1 trees (gen.tree_base = n_blobs+1)
    semantic (B,S,S,S)   int64: 0 background, 1 body, 2 blob, 3 tree
    """

    def __init__(self, size=160, n_blobs=10, n_trees=6, blob_radius=(0.05, 0.11),
                 roughness=(0.25, 0.5), body_radii=(0.44, 0.40, 0.42), body_roughness=0.08,
                 n_freq=12, freq=(1.5, 3.5), min_draw_radius=0.9, device=None, seed=None):
        self.S = size
        self.N = n_blobs
        self.T = n_trees
        self.blob_radius = blob_radius
        self.rough = roughness
        self.body_radii = np.asarray(body_radii)
        self.body_rough = body_roughness
        self.M = n_freq
        self.freq = freq
        self.min_r = min_draw_radius
        self.tree_base = n_blobs + 1
        # Fixed crop for every blob: covers radius * 1.3 (axis jitter) * (1 + 1.5*max rough).
        rmax = blob_radius[1] * 1.3 * size * (1 + 1.5 * roughness[1])
        self.crop = int(2 * np.ceil(rmax) + 3)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                              "mps" if torch.backends.mps.is_available() else "cpu"))
        self.np_rng = np.random.default_rng(seed)
        dev = self.device
        # Random numbers are drawn directly on the device (no host->device copies).
        try:
            self.g = torch.Generator(device=dev)
        except RuntimeError:  # backends without device generators
            self.g = None
            torch.manual_seed(int(self.np_rng.integers(2**62)))
        if self.g is not None:
            self.g.manual_seed(int(self.np_rng.integers(2**62)))
        self.W_bias = torch.from_numpy(cubic_interp_matrix(4, size)).to(dev)
        ax = torch.arange(size, dtype=torch.float32, device=dev)
        self.ax = ax
        # Ball offsets for splatting, one table per integer radius bucket.
        self.offsets = {}
        for R in range(1, 6):
            o = torch.stack(torch.meshgrid(*[torch.arange(-R, R + 1)] * 3, indexing="ij"), -1)
            o = o.reshape(-1, 3)
            self.offsets[R] = o[(o.float().norm(dim=1) <= R + 0.87)].int().to(dev)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.g, device=self.device)

    # ---------------------------------------------------------------- blobs
    def _shape_params(self, n):
        """CPU: random rotation (n,3,3), Fourier directions*freq (n,M,3), phases, amplitudes."""
        rng = self.np_rng
        q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
        q *= np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
        q[np.linalg.det(q) < 0, :, 0] *= -1
        f = rng.normal(size=(n, self.M, 3))
        f *= (rng.uniform(*self.freq, (n, self.M)) / np.linalg.norm(f, axis=2))[..., None]
        ph = rng.uniform(0, 2 * np.pi, (n, self.M))
        amp = rng.normal(size=(n, self.M)) * np.sqrt(2 / self.M)  # h has unit variance
        t = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        return t(q), t(f), t(ph), t(amp)

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

    @torch.no_grad()
    def blobs(self, B):
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
            ok = inside & body_flat[idx]
            # Write the blob id where ok; elsewhere rewrite the current value (a no-op).
            buf[idx.view(-1)] = torch.where(ok, i + 2, buf[idx]).view(-1)
        self.body = body
        return labels

    # ---------------------------------------------------------------- trees
    def tree_points(self, B):
        """CPU: dense skeleton points for B volumes (cheap; can live in DataLoader workers)."""
        out = []
        for b in range(B):
            P, R, t = grow_trees_points(self.np_rng, self.S, self.T, inward=True)
            out.append((P, R, t, np.full(len(P), b, np.int64)))
        return [np.concatenate(x) for x in zip(*out)]

    @torch.no_grad()
    def paint_trees(self, labels, pts):
        """Splat tree points as balls. No data-dependent shapes, so there are no host syncs:
        points are grouped on the CPU by (tree, radius bucket), and voxels outside the ball or
        the volume are written to a dummy slot at the end of a padded flat buffer."""
        S, dev = self.S, self.device
        B = labels.shape[0]
        P, R, t, b = pts
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
        L = int(labels.max()) + 1
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
        # Replicate padding (zero padding would darken the volume borders).
        img = F.conv3d(F.pad(img, (0, 0, 0, 0, rad, rad), mode="replicate"),
                       k.view(B, 1, -1, 1, 1), groups=B)
        img = F.conv3d(F.pad(img, (0, 0, rad, rad, 0, 0), mode="replicate"),
                       k.view(B, 1, 1, -1, 1), groups=B)
        img = F.conv3d(F.pad(img, (rad, rad, 0, 0, 0, 0), mode="replicate"),
                       k.view(B, 1, 1, 1, -1), groups=B)
        img = img.view(B, S, S, S)
        bias = torch.exp(upsample3d(self._randn(B, 4, 4, 4) * 0.3, self.W_bias))
        img = img * bias + self._randn(B, S, S, S) * (2 + 6 * self._rand(B, 1, 1, 1))
        lo = img.view(B, -1).amin(1).view(B, 1, 1, 1)
        hi = img.view(B, -1).amax(1).view(B, 1, 1, 1)
        return ((img - lo) / (hi - lo + 1e-8))[:, None]

    # ---------------------------------------------------------------- all
    @torch.no_grad()
    def sample(self, B, tree_pts=None):
        labels = self.blobs(B)
        labels = self.paint_trees(labels, tree_pts if tree_pts is not None else self.tree_points(B))
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
