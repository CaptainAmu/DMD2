#!/usr/bin/env python3
"""Nearest-train memorization eval for MNIST teacher/student.

Given shared Gaussian latents e_i:
  - teacher t_i: 8-step EDM sample
  - student s_i: 1-step distilled sample

For each t_i, find two nearest train images y_{i,1}, y_{i,2} in flattened
Euclidean L2 on the model space [-1, 1] (LMDB [0,1] is mapped with y -> 2y-1).
Then compute:
  r_i = ||t_i - y_{i,1}||_2 / ||t_i - y_{i,2}||_2
  d_i = ||s_i - t_i||_2

Outputs:
  - density plot of r_i with multiple threshold vlines
  - scatter plot of d_i vs r_i
  - summary txt with memorized percentage per threshold and mean d_i
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time

import dnnlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

_DMD2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _DMD2_ROOT not in sys.path:
    sys.path.insert(0, _DMD2_ROOT)

from main.data.lmdb_dataset import LMDBDataset


def _parse_thresholds(values: list[str]) -> list[float]:
    out: list[float] = []
    for raw in values:
        for token in raw.split(","):
            token = token.strip()
            if token == "":
                continue
            out.append(float(token))
    if not out:
        raise ValueError("No valid thresholds parsed from --r_thres")
    for t in out:
        if not (0.0 < t < 1.0):
            raise ValueError(f"Threshold must be in (0,1), got {t}")
    return out


def _label_font(size: int = 13):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tensor_to_uint8_nhwc(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.permute(0, 2, 3, 1).numpy()


def load_train_lmdb_vectors(train_lmdb: str) -> tuple[torch.Tensor, LMDBDataset]:
    ds = LMDBDataset(train_lmdb)
    n = len(ds)
    sample0 = ds[0]["images"]  # [0,1], CHW
    flat_dim = sample0.numel()
    out = torch.empty((n, flat_dim), dtype=torch.float32)
    for i in tqdm(range(n), desc="Load train LMDB", unit="img"):
        x = ds[i]["images"]  # [0,1]
        out[i] = (x * 2.0 - 1.0).reshape(-1)  # map to [-1,1]
    return out, ds


def gather_train_images_uint8_nhwc(ds: LMDBDataset, indices: torch.Tensor) -> np.ndarray:
    rows = []
    for idx in indices.cpu().tolist():
        x = ds[idx]["images"]  # [0,1], CHW
        x_u8 = (x * 255.0).clamp(0, 255).to(torch.uint8)
        if x_u8.shape[0] == 1:
            x_u8 = x_u8.repeat(3, 1, 1)
        rows.append(x_u8.permute(1, 2, 0).numpy())
    return np.stack(rows, axis=0)


def build_grid_rows(row_tensors_uint8: list[np.ndarray]) -> np.ndarray:
    assert len(row_tensors_uint8) >= 1
    n = row_tensors_uint8[0].shape[0]
    strips = []
    for row in row_tensors_uint8:
        assert row.shape[0] == n, "all rows must have the same number of images"
        strips.append(np.concatenate([row[i] for i in range(n)], axis=1))
    return np.concatenate(strips, axis=0)


def add_left_row_labels(grid: np.ndarray, row_labels: list[str], gutter_px: int = 220) -> np.ndarray:
    h, w, _ = grid.shape
    n_rows = len(row_labels)
    row_h = h // n_rows
    img = Image.new("RGB", (w + gutter_px, h), (248, 248, 248))
    img.paste(Image.fromarray(grid), (gutter_px, 0))
    draw = ImageDraw.Draw(img)
    font = _label_font(13)
    for i, label in enumerate(row_labels):
        y0 = i * row_h
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(6, (gutter_px - tw) // 2)
        y = y0 + (row_h - th) // 2
        draw.text((x, y), label, fill=(25, 25, 25), font=font)
    return np.asarray(img)


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
    one_hot: torch.Tensor | None,
    num_steps: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    n = latents.shape[0]
    outs = []
    for start in tqdm(range(0, n, batch_size), desc=f"Teacher {num_steps}-step", unit="batch"):
        end = min(start + batch_size, n)
        out = edm_sampler(
            teacher,
            None,
            latents[start:end].to(device),
            class_labels=None if one_hot is None else one_hot[start:end].to(device),
            num_steps=num_steps,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            S_churn=0,
        )
        outs.append(out.float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def run_student_batches(
    student,
    latents: torch.Tensor,
    one_hot: torch.Tensor | None,
    conditioning_sigma: float,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    n = latents.shape[0]
    outs = []
    for start in tqdm(range(0, n, batch_size), desc="Student 1-step", unit="batch"):
        end = min(start + batch_size, n)
        z_b = latents[start:end].to(device)
        oh_b = None if one_hot is None else one_hot[start:end].to(device)
        ts_b = torch.ones(end - start, device=device, dtype=torch.long)
        if oh_b is None:
            out = student(z_b * conditioning_sigma, ts_b * conditioning_sigma)
        else:
            out = student(z_b * conditioning_sigma, ts_b * conditioning_sigma, oh_b)
        outs.append(out.float().cpu())
    return torch.cat(outs, dim=0)


def top2_train_l2(
    queries_flat: torch.Tensor, train_flat: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = queries_flat.shape[0]
    d1_all = torch.empty(n, dtype=torch.float32)
    d2_all = torch.empty(n, dtype=torch.float32)
    idx1_all = torch.empty(n, dtype=torch.long)
    idx2_all = torch.empty(n, dtype=torch.long)
    for start in tqdm(range(0, n, chunk_size), desc="2-NN over train set", unit="chunk"):
        end = min(start + chunk_size, n)
        dmat = torch.cdist(queries_flat[start:end], train_flat, p=2.0)
        vals, inds = torch.topk(dmat, k=2, largest=False, dim=1)
        d1_all[start:end] = vals[:, 0]
        d2_all[start:end] = vals[:, 1]
        idx1_all[start:end] = inds[:, 0]
        idx2_all[start:end] = inds[:, 1]
    return d1_all, d2_all, idx1_all, idx2_all


def mean_train_l2(queries_flat: torch.Tensor, train_flat: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n = queries_flat.shape[0]
    out = torch.empty(n, dtype=torch.float32)
    for start in tqdm(range(0, n, chunk_size), desc="Mean dist over train set", unit="chunk"):
        end = min(start + chunk_size, n)
        dmat = torch.cdist(queries_flat[start:end], train_flat, p=2.0)
        out[start:end] = dmat.mean(dim=1)
    return out


def plot_ratio_density(r: torch.Tensor, thresholds: list[float], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(r.cpu().numpy(), bins=50, density=True, color="steelblue", alpha=0.35, edgecolor="none")
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    for i, th in enumerate(thresholds):
        ls = linestyles[i % len(linestyles)]
        ax.axvline(th, linestyle=ls, linewidth=1.8, label=f"r_thres={th:g}")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("r_i = ||t_i - y_{i,1}|| / ||t_i - y_{i,2}||")
    ax.set_ylabel("Density")
    ax.set_title("Distribution density of nearest-neighbor ratio r_i")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_scatter_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    hline_y: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(x.cpu().numpy(), y.cpu().numpy(), s=11, alpha=0.55, edgecolors="none", color="steelblue")
    if hline_y is not None:
        ax.axhline(hline_y, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary(
    out_path: str,
    thresholds: list[float],
    r: torch.Tensor,
    d: torch.Tensor,
    d_t1t8: torch.Tensor,
    delta: torch.Tensor,
    d_t8_tc1: torch.Tensor,
    mean_train_dist: torch.Tensor | None,
    args: argparse.Namespace,
    train_count: int,
) -> None:
    def _stats_line(name: str, arr: torch.Tensor) -> str:
        x = arr.float()
        return (
            f"{name}_mean={x.mean().item():.8f}\n"
            f"{name}_std={x.std(unbiased=False).item():.8f}\n"
            f"{name}_min={x.min().item():.8f}\n"
            f"{name}_max={x.max().item():.8f}\n"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("memorization_nn_eval summary\n")
        f.write(f"teacher_pkl={args.teacher_pkl}\n")
        f.write(f"student_ckpt_dir={args.student_ckpt_dir}\n")
        f.write(f"train_lmdb={args.train_lmdb}\n")
        f.write(f"N={args.num_samples}\n")
        f.write(f"num_train_images={train_count}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"teacher_num_steps={args.teacher_num_steps}\n")
        f.write(f"conditioning_sigma={args.conditioning_sigma}\n")
        f.write(_stats_line("d_t8s", d))
        f.write(_stats_line("d_t1t8", d_t1t8))
        f.write(_stats_line("delta", delta))
        f.write(_stats_line("d_t8_tc1", d_t8_tc1))
        if mean_train_dist is not None:
            f.write(_stats_line("mean_train_dist", mean_train_dist))
        f.write("\n")
        f.write("threshold\tmemorized_percent\n")
        for th in thresholds:
            pct = (r < th).float().mean().item() * 100.0
            f.write(f"{th:g}\t{pct:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST nearest-train memorization evaluation (teacher8 vs student1).")
    parser.add_argument("--teacher_pkl", type=str, required=True)
    parser.add_argument("--student_ckpt_dir", type=str, required=True)
    parser.add_argument("--train_lmdb", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--r_thres", type=str, nargs="+", default=["0.25"])
    parser.add_argument("--teacher_num_steps", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=0)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--nn_chunk", type=int, default=200)
    parser.add_argument("--viz_cols", type=int, default=30)
    parser.add_argument("--compute_mean_train_dist", action="store_true")
    parser.add_argument("--mean_chunk", type=int, default=0)
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.r_thres)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.num_samples
    z = torch.randn(n, args.img_channels, args.resolution, args.resolution, device=device)
    if args.label_dim > 0:
        cls = torch.randint(0, args.label_dim, (n,), device=device)
        one_hot = torch.zeros(n, args.label_dim, device=device)
        one_hot[torch.arange(n, device=device), cls] = 1.0
        one_hot_cpu = one_hot.cpu()
    else:
        one_hot_cpu = None
    latents_cpu = z.cpu()

    student_path = os.path.join(args.student_ckpt_dir, "pytorch_model.bin")
    if not os.path.isfile(student_path):
        raise FileNotFoundError(f"Missing student checkpoint: {student_path}")

    with dnnlib.util.open_url(args.teacher_pkl, verbose=False) as f:
        student = pickle.load(f)["ema"]
    if hasattr(student, "model") and hasattr(student.model, "map_augment"):
        del student.model.map_augment
        student.model.map_augment = None
    student = copy.deepcopy(student)
    while True:
        try:
            state_dict = torch.load(student_path, map_location="cpu")
            break
        except Exception:
            print(f"fail to load checkpoint {student_path}", flush=True)
            time.sleep(1)
    try:
        print(student.load_state_dict(state_dict, strict=True))
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load student checkpoint with strict=True due to teacher/student architecture mismatch.\n"
            f"student_path={student_path}\n"
            f"teacher_pkl={args.teacher_pkl}\n"
            "Hint: teacher and student checkpoints are likely from different runs. "
            "Set --teacher_pkl to the teacher checkpoint used for this student's distillation."
        ) from e
    student = student.to(device)
    student.eval()
    s = run_student_batches(student, latents_cpu, one_hot_cpu, args.conditioning_sigma, device, args.batch_size)
    del student
    torch.cuda.empty_cache()

    teacher = load_teacher(args.teacher_pkl, device)
    t1 = run_teacher_batches(teacher, latents_cpu, one_hot_cpu, 1, device, args.batch_size)
    t = run_teacher_batches(teacher, latents_cpu, one_hot_cpu, args.teacher_num_steps, device, args.batch_size)
    del teacher
    torch.cuda.empty_cache()

    train_flat, train_ds = load_train_lmdb_vectors(args.train_lmdb)
    train_flat = train_flat.float()
    t_flat = t.reshape(n, -1).float().cpu()
    t1_flat = t1.reshape(n, -1).float().cpu()
    s_flat = s.reshape(n, -1).float().cpu()

    d1, d2, idx1, idx2 = top2_train_l2(t_flat, train_flat, args.nn_chunk)
    mean_chunk = args.mean_chunk if args.mean_chunk > 0 else args.nn_chunk
    mean_train_dist = mean_train_l2(t_flat, train_flat, mean_chunk) if args.compute_mean_train_dist else None
    eps = 1e-12
    r = d1 / torch.clamp(d2, min=eps)
    d = torch.linalg.vector_norm(s_flat - t_flat, ord=2, dim=1)  # ||T8 - S||
    d_t1t8 = torch.linalg.vector_norm(t1_flat - t_flat, ord=2, dim=1)  # ||T1 - T8||
    delta = d - d_t1t8

    ratio_plot = os.path.join(args.out_dir, "ratio_density.png")
    scatter_plot = os.path.join(args.out_dir, "d_vs_r_scatter.png")
    delta_vs_t8tc1_plot = os.path.join(args.out_dir, "delta_vs_t8_tc1.png")
    delta_vs_d_plot = os.path.join(args.out_dir, "delta_vs_mean_train_dist.png")
    grid_png = os.path.join(args.out_dir, "memorization_triplet_with_neighbors.png")
    summary_txt = os.path.join(args.out_dir, "memorization_summary.txt")

    plot_ratio_density(r, thresholds, ratio_plot)
    plot_scatter_xy(
        r, d,
        "r_i = ||T8_i - TC1_i|| / ||T8_i - TC2_i||",
        "||T8_i - S_i||_2",
        "Student-teacher distance vs memorization ratio",
        scatter_plot,
    )
    plot_scatter_xy(
        d1, delta,
        "||T8_i - TC1_i||_2",
        "||T8_i - S_i||_2 - ||T1_i - T8_i||_2",
        "Delta vs nearest-train distance",
        delta_vs_t8tc1_plot,
    )
    if mean_train_dist is not None:
        plot_scatter_xy(
            mean_train_dist, delta,
            "D_i = mean_j ||T8_i - train_j||_2",
            "||T8_i - S_i||_2 - ||T1_i - T8_i||_2",
            "Delta vs average train distance",
            delta_vs_d_plot,
            hline_y=0.0,
        )

    k = min(args.viz_cols, n)
    viz = torch.arange(k, dtype=torch.long)
    t1_u8 = tensor_to_uint8_nhwc(t1[viz])
    t8_u8 = tensor_to_uint8_nhwc(t[viz])
    s_u8 = tensor_to_uint8_nhwc(s[viz])
    tc1_u8 = gather_train_images_uint8_nhwc(train_ds, idx1[viz])
    tc2_u8 = gather_train_images_uint8_nhwc(train_ds, idx2[viz])
    grid = build_grid_rows([t1_u8, t8_u8, s_u8, tc1_u8, tc2_u8])
    grid = add_left_row_labels(
        grid,
        [
            "Teacher 1-step (T1)",
            f"Teacher {args.teacher_num_steps}-step (T8)",
            "Student 1-step (S)",
            "Closest train to T8 (TC1)",
            "2nd closest train to T8 (TC2)",
        ],
    )
    Image.fromarray(grid).save(grid_png)

    write_summary(
        summary_txt,
        thresholds,
        r,
        d,
        d_t1t8,
        delta,
        d1,
        mean_train_dist,
        args,
        train_count=train_flat.shape[0],
    )

    print(f"Saved {ratio_plot}", flush=True)
    print(f"Saved {scatter_plot}", flush=True)
    print(f"Saved {delta_vs_t8tc1_plot}", flush=True)
    if mean_train_dist is not None:
        print(f"Saved {delta_vs_d_plot}", flush=True)
    print(f"Saved {grid_png}", flush=True)
    print(f"Saved {summary_txt}", flush=True)


if __name__ == "__main__":
    main()
