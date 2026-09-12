"""Fully synthetic, in-memory dataset of small grayscale shape images.

No external downloads (no MNIST, no internet dependency) -- generated
on the fly, same philosophy as the AtharX synthetic-data pipeline: define an
explicit generative process so every sample is reproducible from its seed.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def _draw_shape(size: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    kind = rng.integers(0, 3)
    cy, cx = rng.uniform(0.3, 0.7, size=2) * size
    r = rng.uniform(0.15, 0.3) * size
    if kind == 0:  # filled circle
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r**2
    elif kind == 1:  # filled square
        mask = (np.abs(yy - cy) < r) & (np.abs(xx - cx) < r)
    else:  # ring
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 < r**2) & (
            (yy - cy) ** 2 + (xx - cx) ** 2 > (r * 0.6) ** 2
        )
    img[mask] = rng.uniform(0.6, 1.0)
    img += rng.normal(0, 0.03, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


class ShapesDataset(Dataset):
    def __init__(self, n_samples: int = 512, size: int = 28, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.data = np.stack([_draw_shape(size, rng) for _ in range(n_samples)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).unsqueeze(0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = ShapesDataset(16)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(ds[i][0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("dataset_sample.png", dpi=150)
    print("Saved dataset_sample.png")
