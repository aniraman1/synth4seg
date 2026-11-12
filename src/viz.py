"""Visualization: slice montage (matplotlib) and 3D surface render (pyvista)."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def label_colors(n, meta=None):
    """RGB per label id. Background black, body grey; with `meta`, blobs get muted and
    trees saturated golden-ratio hues so trees stand out. Without meta: the old tab10 scheme."""
    if meta is None:
        base = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)
        cols = [(0, 0, 0), (0.35, 0.35, 0.35)]
        cols += [base[i % len(base)] for i in range(max(n - 2, 0))]
        return cols
    from matplotlib.colors import hsv_to_rgb
    cols = [(0.0, 0.0, 0.0)] * n
    for k in range(1, n):
        kind = meta.get(k, {}).get("type", "blob")
        h = (k * 0.61803398875) % 1.0
        if kind == "body":
            cols[k] = (0.35, 0.35, 0.35)
        elif kind == "tree":
            cols[k] = tuple(hsv_to_rgb((h, 0.95, 1.0)))
        else:
            cols[k] = tuple(hsv_to_rgb((h, 0.45, 0.75)))
    return cols


def _structure_mask(labels, meta):
    """Voxels that belong to a blob or tree (not background or body)."""
    body = [k for k, m in meta.items() if m["type"] in ("background", "body")]
    return ~np.isin(labels, body + [0])


def save_slices(labels, image, meta, path):
    n = int(labels.max()) + 1
    cmap = ListedColormap(label_colors(n, meta))
    kw = dict(cmap=cmap, vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
    c = [s // 2 for s in labels.shape]
    sl = [lambda v: v[c[0]], lambda v: v[:, c[1]], lambda v: v[:, :, c[2]]]
    names = ["axial", "coronal", "sagittal"]

    tree_ids = [k for k, m in meta.items() if m["type"] == "tree"]
    trees_only = np.where(np.isin(labels, tree_ids), labels, 0)

    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4), facecolor="white")
    for j in range(3):
        ax[0, j].imshow(sl[j](labels), **kw)
        ax[0, j].set_title(f"labels — {names[j]}")
        ax[1, j].imshow(sl[j](image), cmap="gray")
        ax[1, j].set_title(f"synthetic image — {names[j]}")
    ax[0, 3].imshow(trees_only.max(0), **kw)
    ax[0, 3].set_title("trees only — max projection (axial)")
    ax[1, 3].imshow(trees_only.max(1), **kw)
    ax[1, 3].set_title("trees only — max projection (coronal)")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _surface(mask):
    import pyvista as pv
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, mask.shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    grid = pv.ImageData(dimensions=sub.shape, origin=tuple(lo.astype(float)))
    grid.point_data["m"] = sub.ravel(order="F")
    return grid.contour([0.5], scalars="m").smooth_taubin(n_iter=30, pass_band=0.1)


def render_3d(labels, meta, path, interactive=False, max_blob_surfaces=80):
    """Blob surfaces (semi-transparent) + tree surfaces (opaque). With very many blobs,
    blobs are shown on three orthogonal cut planes instead of as surfaces."""
    import pyvista as pv
    n = int(labels.max()) + 1
    cols = label_colors(n, meta)
    pl = pv.Plotter(off_screen=not interactive, window_size=(1200, 1000))
    pl.set_background("white")
    n_blobs = sum(m["type"] == "blob" for m in meta.values())
    many = n_blobs > max_blob_surfaces
    if many:
        lut = (np.array(cols) * 255).astype(np.uint8)
        grid = pv.ImageData(dimensions=labels.shape)
        grid.point_data["rgb"] = lut[labels.ravel(order="F")]
        c = [s // 2 for s in labels.shape]
        for normal, origin in (("x", (c[0], 0, 0)), ("y", (0, c[1], 0)), ("z", (0, 0, c[2]))):
            pl.add_mesh(grid.slice(normal=normal, origin=origin), scalars="rgb", rgb=True,
                        opacity=0.55)
    # Volume outline for context.
    pl.add_mesh(pv.Box(bounds=(0, labels.shape[0] - 1, 0, labels.shape[1] - 1,
                               0, labels.shape[2] - 1)), style="wireframe", color="grey")
    n_trees = sum(m["type"] == "tree" for m in meta.values())
    for k, m in meta.items():
        if m["type"] == "background" or (many and m["type"] == "blob"):
            continue
        mask = labels == k if m["type"] != "body" else labels > 0
        if not mask.any():
            continue
        surf = _surface(mask)
        if m["type"] == "body":
            pl.add_mesh(surf, color=cols[1], opacity=0.08)
        elif m["type"] == "blob":
            pl.add_mesh(surf, color=cols[k], opacity=0.4, smooth_shading=True)
        else:
            pl.add_mesh(surf, color=cols[k], smooth_shading=True, specular=0.3,
                        label=f"{k}: {m['type']}" if n_trees <= 25 else None)
    if 0 < n_trees <= 25:
        pl.add_legend(bcolor="white", face="circle", size=(0.16, 0.25))
    pl.camera_position = "iso"
    pl.camera.zoom(1.1)
    if interactive:
        pl.show()
    else:
        pl.screenshot(path)
        pl.close()


def save_slice_gif(labels, image, path, meta=None, axis=0, scale=2, fps=20):
    """Animated sweep through every slice: image | labels | overlay."""
    from PIL import Image, ImageDraw

    n = int(labels.max()) + 1
    lut = (np.array(label_colors(n, meta)) * 255).astype(np.uint8)
    smask = np.moveaxis(_structure_mask(labels, meta) if meta else labels >= 2, axis, 0)
    lab = np.moveaxis(labels, axis, 0)
    img = np.moveaxis(image, axis, 0)
    frames = []
    for i in range(lab.shape[0]):
        g = (img[i] * 255).astype(np.uint8)
        gray = np.repeat(g[..., None], 3, -1)
        col = lut[lab[i]]
        struct = smask[i][..., None]
        over = np.where(struct, (0.45 * gray + 0.55 * col).astype(np.uint8), gray)
        sep = np.full((g.shape[0], 4, 3), 255, np.uint8)
        row = np.concatenate([gray, sep, col, sep, over], 1)
        im = Image.fromarray(row).resize((row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST)
        ImageDraw.Draw(im).text((8, 6), f"slice {i + 1}/{lab.shape[0]}", fill=(255, 255, 0))
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=256))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
