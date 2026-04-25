#!/usr/bin/env python3
"""Pairwise pixel-L2 and LPIPS between MNIST student (1-step) and teacher (1-step, 8-step)."""
from __future__ import annotations

DEFAULT_NUM_IMAGES = 100

import argparse
import gc
import os
import sys

import dnnlib
import lpips as lpips_lib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import torch
from tqdm import tqdm

_DMD2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _DMD2_ROOT not in sys.path:
    sys.path.insert(0, _DMD2_ROOT)

from main.edm.test_folder_edm import create_generator


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


@torch.no_grad()
def run_student_batches(
    student,
    latents: torch.Tensor,
    one_hot: torch.Tensor,
    conditioning_sigma: float,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    n = latents.shape[0]
    outs = []
    for start in tqdm(range(0, n, batch_size), desc="Student 1-step", unit="batch"):
        end = min(start + batch_size, n)
        z_b = latents[start:end].to(device)
        oh_b = one_hot[start:end].to(device)
        ts_b = torch.ones(end - start, device=device, dtype=torch.long)
        out = student(z_b * conditioning_sigma, ts_b * conditioning_sigma, oh_b)
        outs.append(out.float())
    return torch.cat(outs, dim=0)


def pixel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image L2 in pixel space; CPU 1D float tensor (no ``.numpy()`` — broken NumPy builds)."""
    return (a - b).pow(2).mean(dim=[1, 2, 3]).sqrt().cpu().float().reshape(-1)


def to_rgb3(x: torch.Tensor) -> torch.Tensor:
    """(N,C,H,W) with C=1 -> repeat to 3ch for LPIPS."""
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


@torch.no_grad()
def lpips_batched(
    loss_fn,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    n = a.shape[0]
    a3 = to_rgb3(a)
    b3 = to_rgb3(b)
    scores = []
    for start in tqdm(range(0, n, batch_size), desc="LPIPS", unit="batch"):
        end = min(start + batch_size, n)
        d = loss_fn(a3[start:end].to(device), b3[start:end].to(device))
        scores.append(d.squeeze().detach().cpu().float().reshape(-1))
    return torch.cat(scores, dim=0)


def _tvec(a: np.ndarray | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float64).reshape(-1)


def _hist_bar_vertical(ax, a: np.ndarray | torch.Tensor, n_bins: int, **bar_kw) -> None:
    t = _tvec(a).float()
    lo, hi = t.min().item(), t.max().item()
    if hi <= lo:
        hi = lo + 1e-6
    counts, edges = torch.histogram(t, bins=n_bins, range=(lo, hi))
    w = (hi - lo) / n_bins
    centers = ((edges[:-1] + edges[1:]) / 2).cpu().tolist()
    ax.bar(centers, counts.cpu().tolist(), width=w * 0.95, **bar_kw)


def _hist_bar_horizontal(ax, a: np.ndarray | torch.Tensor, n_bins: int, **bar_kw) -> None:
    t = _tvec(a).float()
    lo, hi = t.min().item(), t.max().item()
    if hi <= lo:
        hi = lo + 1e-6
    counts, edges = torch.histogram(t, bins=n_bins, range=(lo, hi))
    h = (hi - lo) / n_bins
    centers = ((edges[:-1] + edges[1:]) / 2).cpu().tolist()
    ax.barh(centers, counts.cpu().tolist(), height=h * 0.95, **bar_kw)


def joint_scatter_hist(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    n_bins: int = 25,
) -> None:
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], hspace=0.05, wspace=0.05)
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_hist_y = fig.add_subplot(gs[0, 0], sharey=ax_scatter)
    ax_hist_x = fig.add_subplot(gs[1, 1], sharex=ax_scatter)
    ax_blank = fig.add_subplot(gs[1, 0])
    ax_blank.set_visible(False)

    tx, ty = _tvec(x), _tvec(y)
    ax_scatter.scatter(tx.cpu().tolist(), ty.cpu().tolist(), s=12, alpha=0.55, edgecolors="none", color="steelblue")
    ax_scatter.set_title(title, pad=8)
    ax_scatter.tick_params(labelbottom=False, labelleft=False)

    lo = min(tx.min().item(), ty.min().item())
    hi = max(tx.max().item(), ty.max().item())
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.plot([lo, hi], [lo, hi], color="tomato", lw=1.0, ls="--", label="y = x")
    ax_scatter.legend(fontsize=8, loc="upper left")

    _hist_bar_vertical(ax_hist_x, x, n_bins, color="steelblue", alpha=0.75)
    ax_hist_x.set_xlabel(xlabel)
    ax_hist_x.invert_yaxis()
    ax_hist_x.tick_params(labelleft=False)

    _hist_bar_horizontal(ax_hist_y, y, n_bins, color="steelblue", alpha=0.75)
    ax_hist_y.set_ylabel(ylabel)
    ax_hist_y.invert_xaxis()
    ax_hist_y.tick_params(labelbottom=False)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def joint_scatter_hist_ratio(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    n_bins: int = 25,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], hspace=0.05, wspace=0.05)
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_hist_y = fig.add_subplot(gs[0, 0], sharey=ax_scatter)
    ax_hist_x = fig.add_subplot(gs[1, 1], sharex=ax_scatter)
    ax_blank = fig.add_subplot(gs[1, 0])
    ax_blank.set_visible(False)

    tx, ty = _tvec(x), _tvec(y)
    ax_scatter.scatter(tx.cpu().tolist(), ty.cpu().tolist(), s=12, alpha=0.55, edgecolors="none", color="steelblue")
    ax_scatter.set_title(title, pad=8)
    ax_scatter.tick_params(labelbottom=False, labelleft=False)

    lo_x, hi_x = tx.min().item(), tx.max().item()
    lo_y = y_min if y_min is not None else ty.min().item()
    hi_y = y_max if y_max is not None else ty.max().item()
    ax_scatter.set_xlim(lo_x, hi_x)
    ax_scatter.set_ylim(lo_y, hi_y)
    ax_scatter.axhline(1.0, color="tomato", lw=1.0, ls="--", label="y = 1")
    ax_scatter.legend(fontsize=8, loc="upper left")

    _hist_bar_vertical(ax_hist_x, x, n_bins, color="steelblue", alpha=0.75)
    ax_hist_x.set_xlabel(xlabel)
    ax_hist_x.invert_yaxis()
    ax_hist_x.tick_params(labelleft=False)

    _hist_bar_horizontal(ax_hist_y, y, n_bins, color="steelblue", alpha=0.75)
    ax_hist_y.set_ylabel(ylabel)
    ax_hist_y.invert_xaxis()
    ax_hist_y.tick_params(labelbottom=False)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt_dir", type=str, required=True)
    parser.add_argument("--teacher_pkl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--lpips_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="VGG/LPIPS device. Default cpu avoids segfaults after GPU LPIPS (CUDA teardown + NumPy/Matplotlib).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.student_ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_dev = torch.device(
        "cuda"
        if (args.lpips_device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    K = args.num_images
    c = args.img_channels
    z = torch.randn(K, c, args.resolution, args.resolution, device=device)
    class_idx = torch.randint(0, args.label_dim, (K,), device=device)
    one_hot = torch.zeros(K, args.label_dim, device=device)
    one_hot[torch.arange(K, device=device), class_idx] = 1.0
    latents_cpu = z.cpu()
    one_hot_cpu = one_hot.cpu()

    student_path = os.path.join(args.student_ckpt_dir, "pytorch_model.bin")
    student = create_generator(student_path, base_model=None, dataset_name="mnist").to(device)
    student.eval()
    s_1 = run_student_batches(
        student, latents_cpu, one_hot_cpu,
        args.conditioning_sigma, device, args.batch_size,
    )
    del student
    torch.cuda.empty_cache()

    teacher = load_teacher(args.teacher_pkl, device)
    t_1 = run_teacher_batches(teacher, latents_cpu, one_hot_cpu, 1, device, args.batch_size, "Teacher 1-step")
    t_8 = run_teacher_batches(teacher, latents_cpu, one_hot_cpu, 8, device, args.batch_size, "Teacher 8-step")
    del teacher
    torch.cuda.empty_cache()

    print("Computing pixel L2 distances...", flush=True)
    d2_s1t1 = pixel_l2(s_1, t_1)
    d2_s1t8 = pixel_l2(s_1, t_8)
    d2_t1t8 = pixel_l2(t_1, t_8)

    print(f"Computing LPIPS distances on {lpips_dev}...", flush=True)
    loss_fn = lpips_lib.LPIPS(net="vgg").to(lpips_dev)
    dl_s1t1 = lpips_batched(loss_fn, s_1, t_1, args.batch_size, lpips_dev)
    dl_s1t8 = lpips_batched(loss_fn, s_1, t_8, args.batch_size, lpips_dev)
    dl_t1t8 = lpips_batched(loss_fn, t_1, t_8, args.batch_size, lpips_dev)
    del loss_fn
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    eps = 1e-8
    den_l2 = torch.clamp(d2_t1t8.double(), min=eps)
    trig_l2 = (d2_s1t1.double() + d2_s1t8.double()) / den_l2
    den_lp = torch.clamp(dl_t1t8.double(), min=eps)
    trig_lpips = (dl_s1t1.double() + dl_s1t8.double()) / den_lp

    def _summ(name: str, arr: np.ndarray | torch.Tensor) -> None:
        t = torch.as_tensor(arr, dtype=torch.float64)
        print(
            f"{name}: mean={t.mean().item():.4f}  std={t.std(unbiased=False).item():.4f}  "
            f"min={t.min().item():.4f}  max={t.max().item():.4f}",
            flush=True,
        )

    for name, arr in [
        ("d2_s1t1", d2_s1t1), ("d2_s1t8", d2_s1t8), ("d2_t1t8", d2_t1t8),
        ("dl_s1t1", dl_s1t1), ("dl_s1t8", dl_s1t8), ("dl_t1t8", dl_t1t8),
        ("trig_l2", trig_l2), ("trig_lpips", trig_lpips),
    ]:
        _summ(name, arr)

    joint_scatter_hist(
        x=d2_s1t1, y=d2_s1t8,
        xlabel="d2_s1t1  (pixel L2: s1 vs t1)",
        ylabel="d2_s1t8  (pixel L2: s1 vs t8)",
        title="L2: s1t1 (x) / s1t8 (y)",
        out_path=os.path.join(out_dir, "s1t1-s1t8_l2.png"),
    )
    joint_scatter_hist(
        x=dl_s1t1, y=dl_s1t8,
        xlabel="dl_s1t1  (LPIPS: s1 vs t1)",
        ylabel="dl_s1t8  (LPIPS: s1 vs t8)",
        title="LPIPS: s1t1 (x) / s1t8 (y)",
        out_path=os.path.join(out_dir, "s1t1-s1t8_lpips.png"),
    )
    joint_scatter_hist(
        x=d2_t1t8, y=d2_s1t8,
        xlabel="d2_t1t8  (pixel L2: t1 vs t8)",
        ylabel="d2_s1t8  (pixel L2: s1 vs t8)",
        title="L2: t1t8 (x) / s1t8 (y)",
        out_path=os.path.join(out_dir, "t1t8-s1t8_l2.png"),
    )
    joint_scatter_hist(
        x=dl_t1t8, y=dl_s1t8,
        xlabel="dl_t1t8  (LPIPS: t1 vs t8)",
        ylabel="dl_s1t8  (LPIPS: s1 vs t8)",
        title="LPIPS: t1t8 (x) / s1t8 (y)",
        out_path=os.path.join(out_dir, "t1t8-s1t8_lpips.png"),
    )
    joint_scatter_hist_ratio(
        x=d2_t1t8, y=trig_l2,
        xlabel="d2_t1t8  (pixel L2: t1 vs t8)",
        ylabel="(d2_s1t1 + d2_s1t8) / d2_t1t8  (pixel L2 ratio)",
        title="Trig L2 ratio: t1t8 (x) / ((s1t1 + s1t8) / t1t8) (y)",
        out_path=os.path.join(out_dir, "s1t1t8_trig_l2.png"),
        y_min=0.0,
        y_max=trig_l2.max().item() + 1.0,
    )
    joint_scatter_hist_ratio(
        x=dl_t1t8, y=trig_lpips,
        xlabel="dl_t1t8  (LPIPS: t1 vs t8)",
        ylabel="(dl_s1t1 + dl_s1t8) / dl_t1t8  (LPIPS ratio)",
        title="Trig LPIPS ratio: t1t8 (x) / ((s1t1 + s1t8) / t1t8) (y)",
        out_path=os.path.join(out_dir, "s1t1t8_trig_lpips.png"),
        y_min=0.0,
        y_max=trig_lpips.max().item() + 1.0,
    )


if __name__ == "__main__":
    main()
