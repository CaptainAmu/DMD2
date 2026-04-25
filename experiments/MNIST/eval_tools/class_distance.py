#!/usr/bin/env python3
"""Per-class mean images from the teacher EDM, pairwise L2 between means, heatmap + stats."""
from __future__ import annotations

import argparse
import os
import sys

import dnnlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from tqdm import tqdm

_DMD2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _DMD2_ROOT not in sys.path:
    sys.path.insert(0, _DMD2_ROOT)


def edm_sampler(
    net,
    residual_net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if num_steps == 1:
        t_steps = torch.tensor([sigma_max, 0.0], dtype=torch.float64, device=latents.device)
        t_steps[0] = net.round_sigma(t_steps[0])
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        if residual_net is None:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        else:
            original_denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            denoised = residual_net(x_hat, original_denoised, t_hat, class_labels).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def load_teacher(teacher_pkl: str, device: torch.device):
    with dnnlib.util.open_url(teacher_pkl, verbose=False) as f:
        teacher = pickle.load(f)["ema"]
    teacher.eval()
    teacher = teacher.to(device)
    if hasattr(teacher, "model") and hasattr(teacher.model, "map_augment"):
        del teacher.model.map_augment
        teacher.model.map_augment = None
    return teacher


@torch.no_grad()
def run_teacher_batches(
    teacher,
    latents: torch.Tensor,
    one_hot: torch.Tensor,
    num_steps: int,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    n = latents.shape[0]
    outs = []
    for start in tqdm(range(0, n, batch_size), desc=desc, unit="batch"):
        end = min(start + batch_size, n)
        out = edm_sampler(
            teacher,
            None,
            latents[start:end].to(device),
            class_labels=one_hot[start:end].to(device),
            num_steps=num_steps,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            S_churn=0,
        )
        outs.append(out.float())
    return torch.cat(outs, dim=0)


def pairwise_l2_matrix(means: torch.Tensor) -> torch.Tensor:
    """means: (K, C, H, W) -> (K, K) L2 between flattened rows."""
    k = means.shape[0]
    flat = means.reshape(k, -1).double()
    d = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        for j in range(i + 1, k):
            t = torch.norm(flat[i] - flat[j], p=2)
            d[i, j] = t
            d[j, i] = t
    return d


def offdiag_mean_std(d: torch.Tensor) -> tuple[float, float]:
    k = d.shape[0]
    tri = d[torch.triu(torch.ones(k, k, dtype=torch.bool), diagonal=1)]
    m = tri.mean().item()
    s = tri.std(unbiased=False).item()
    return m, s


def full_matrix_mean_std(d: torch.Tensor) -> tuple[float, float]:
    t = d.reshape(-1)
    return t.mean().item(), t.std(unbiased=False).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_pkl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_per_class", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class_indices = list(range(args.label_dim))
    k = len(class_indices)
    c_ch = args.img_channels
    h = args.resolution
    w = args.resolution
    n_pc = args.num_per_class

    teacher = load_teacher(args.teacher_pkl, device)
    means_list: list[torch.Tensor] = []

    for ci, c in enumerate(class_indices):
        z = torch.randn(n_pc, c_ch, h, w)
        one_hot = torch.zeros(n_pc, args.label_dim)
        one_hot[:, c] = 1.0
        imgs = run_teacher_batches(
            teacher,
            z,
            one_hot,
            args.num_steps,
            device,
            args.batch_size,
            desc=f"Class {c} ({ci + 1}/{k})",
        )
        means_list.append(imgs.mean(dim=0).cpu())

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    means = torch.stack(means_list, dim=0)
    d = pairwise_l2_matrix(means)
    d_np = d.numpy()

    off_m, off_s = offdiag_mean_std(d)
    full_m, full_s = full_matrix_mean_std(d)

    summary_path = os.path.join(args.out_dir, "class_distance_summary.txt")
    summary_lines = [
        f"K={k}  num_per_class={n_pc}  num_steps={args.num_steps}  seed={args.seed}",
        f"Pairwise L2 (unique off-diagonal i<j): mean={off_m:.6f}  std={off_s:.6f}",
        f"Full matrix entries (incl. diagonal): mean={full_m:.6f}  std={full_s:.6f}",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(d_np, cmap="viridis", aspect="equal")
    ax.set_title("L2 distance between per-class mean images (teacher)")
    ax.set_xlabel("class index (row in means)")
    ax.set_ylabel("class index (row in means)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if k <= 20:
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
    heat_path = os.path.join(args.out_dir, "heatmap.png")
    fig.savefig(heat_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {heat_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
