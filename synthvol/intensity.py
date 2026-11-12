"""SynthSeg-style intensity synthesis from a label map."""
import numpy as np
from scipy import ndimage


def synth_image(labels, rng, mean_range=(0, 255), std_range=(0, 25),
                blur_range=(0.5, 1.5), bias_std=0.3, noise_std=(2, 8)):
    ids = np.unique(labels)
    means = rng.uniform(*mean_range, size=ids.max() + 1)
    stds = rng.uniform(*std_range, size=ids.max() + 1)

    # Per-label Gaussian mixture sample.
    img = means[labels] + stds[labels] * rng.standard_normal(labels.shape)

    # Partial-volume style blur.
    img = ndimage.gaussian_filter(img, sigma=rng.uniform(*blur_range))

    # Smooth multiplicative bias field.
    coarse = rng.normal(0, bias_std, size=(4, 4, 4))
    zoom = [s / 4 for s in labels.shape]
    bias = np.exp(ndimage.zoom(coarse, zoom, order=1))
    img = img * bias[: labels.shape[0], : labels.shape[1], : labels.shape[2]]

    img += rng.normal(0, rng.uniform(*noise_std), size=img.shape)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)
