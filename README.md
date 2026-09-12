# Generative Models + Combinatorial Optimization

Two small, independent demos, built out of interest in two areas adjacent to
my existing computer-vision/ML work that I hadn't gotten properly hands-on
with yet: generative modelling and combinatorial optimization.

## 1. Generative model — a small convolutional VAE

`synthetic_dataset.py` generates a fully synthetic, in-memory dataset of
28x28 shapes (circles / squares / rings) — no external downloads, same
philosophy as the AtharX degradation pipeline: an explicit, seeded generative
process so every sample is reproducible. `vae.py` / `train_vae.py` train a
small convolutional VAE (encoder → latent → decoder, reparameterization
trick, ELBO loss) on it in ~25 epochs / well under a minute on CPU.

```
python synthetic_dataset.py   # -> dataset_sample.png
python train_vae.py           # -> vae_results.png, vae.pt
```

**Honest result**: reconstructions are blurry and generated samples (drawn
fresh from the prior) are recognizable as blob-like shapes but lose sharp
detail (e.g. rings often collapse toward solid blobs) — this is the
well-known VAE failure mode (the Gaussian likelihood assumption penalizes
pixel-wise error, which pushes the decoder toward "safe," blurry averages).
It's exactly the reason diffusion models and GANs have displaced VAEs for
image fidelity in practice, while VAEs remain useful where a well-behaved,
smooth latent space matters more than sharp samples. I picked a VAE over a
diffusion model here specifically because it trains in seconds on CPU and
still exercises the full generative-modelling pipeline (latent variable,
sampling, reconstruction vs. generation) that the project asks about.

## 2. Combinatorial matching — UAV-to-rescue-site assignment

`optimization_matching.py` poses a static assignment problem: given `N` UAV
positions and `N` rescue-site positions, assign each UAV to exactly one site
to minimize total travel distance. Solved two ways:

- **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) — optimal
  in polynomial time.
- **Greedy baseline** — repeatedly assigns the UAV with the fewest good
  options left to its nearest still-available site.

```
python optimization_matching.py   # -> assignment_comparison.png
```

On the included scenario (10 UAVs, 10 sites, seed 2): Hungarian finds a
total distance of **222.5**, the greedy heuristic gets **317.4** — the
optimal solver is **~30% shorter**, and the plotted paths make the reason
visible (greedy produces crossing, backtracking routes that an optimal
matching avoids entirely).

**Scope note**: this is the *static* sub-problem — a fully dynamic version
(agents and targets coordinating over time, under uncertainty) would bring
reinforcement learning into the mix on top of this. This demo covers the
assignment/matching foundation that a dynamic system would still need to
solve repeatedly at each decision point.

## Setup

```bash
pip install -r requirements.txt
```
