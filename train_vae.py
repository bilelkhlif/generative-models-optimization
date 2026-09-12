"""Trains the small ConvVAE on the synthetic shapes dataset and saves
reconstructions + freshly sampled generations.
"""
import torch
from torch.utils.data import DataLoader

from synthetic_dataset import ShapesDataset
from vae import ConvVAE, vae_loss


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ShapesDataset(n_samples=512, size=28, seed=0)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = ConvVAE(latent_dim=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 25
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(ds)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:02d}  avg ELBO loss/sample: {avg:.2f}")

    torch.save(model.state_dict(), "vae.pt")

    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        # reconstructions
        sample_batch = next(iter(loader))[:8].to(device)
        recon, _, _ = model(sample_batch)
        # fresh generations from the prior
        z = torch.randn(8, model.latent_dim, device=device)
        gen = model.decode(z)

    fig, axes = plt.subplots(3, 8, figsize=(12, 4.5))
    for i in range(8):
        axes[0, i].imshow(sample_batch[i, 0].cpu(), cmap="gray")
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="gray")
        axes[2, i].imshow(gen[i, 0].cpu(), cmap="gray")
    for ax, label in zip(axes[:, 0], ["input", "reconstruction", "generated (prior)"]):
        ax.set_ylabel(label, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("vae_results.png", dpi=150)
    print("Saved vae_results.png and vae.pt")


if __name__ == "__main__":
    main()
