"""Organic blob structures: noise-perturbed, randomly rotated ellipsoids."""
import numpy as np
from scipy import ndimage


def random_rotation(rng):
    """Uniform random 3x3 rotation matrix (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _bbox(shape, center, half):
    lo = np.maximum(np.floor(center - half).astype(int), 0)
    hi = np.minimum(np.ceil(center + half).astype(int) + 1, shape)
    return lo, hi


def make_blob(shape, center, radii, rng, roughness=0.35):
    """Return a full-size bool mask containing one connected, hole-free blob.

    Field f(x) = 1 - ||R^T (x - c) / radii|| + roughness * smooth_noise(x); blob = f > 0.
    Work is done only inside the blob's bounding box.
    """
    center = np.asarray(center, float)
    radii = np.asarray(radii, float)
    half = radii.max() * (1.0 + 2 * roughness) + 2
    lo, hi = _bbox(shape, center, half)

    zz, yy, xx = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
    p = np.stack([zz, yy, xx], -1) - center
    q = p @ random_rotation(rng)  # coordinates in the blob's own frame
    ell = 1.0 - np.sqrt(((q / radii) ** 2).sum(-1))

    noise = ndimage.gaussian_filter(rng.normal(size=ell.shape), sigma=max(radii.mean() / 3, 1.0))
    noise /= noise.std() + 1e-8
    sub = ell + roughness * noise > 0

    lab, n = ndimage.label(sub)
    if n > 1:
        sizes = ndimage.sum(sub, lab, range(1, n + 1))
        sub = lab == (np.argmax(sizes) + 1)
    sub = ndimage.binary_fill_holes(sub)

    mask = np.zeros(shape, bool)
    mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sub
    return mask
