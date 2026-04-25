#!/usr/bin/env python3
"""PCA triplet plot for MNIST: Teacher 8-step / 1-step / Student 1-step (shared noise + labels)."""
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


def pca_project_from_t8(t8: torch.Tensor, t1: torch.Tensor, s1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_t8 = t8.detach().float().cpu().reshape(t8.shape[0], -1)
    x_t1 = t1.detach().float().cpu().reshape(t1.shape[0], -1)
    x_s1 = s1.detach().float().cpu().reshape(s1.shape[0], -1)

    mu = x_t8.mean(dim=0, keepdim=True)
    x_t8c = x_t8 - mu
    x_t1c = x_t1 - mu
    x_s1c = x_s1 - mu

    print("Fitting PCA on teacher-8 images...", flush=True)
    try:
        q = min(16, x_t8c.shape[1], x_t8c.shape[0])
        _, _, v = torch.pca_lowrank(x_t8c, q=q, center=False)
        comps = v[:, :2]
        z_t8 = x_t8c @ comps
        z_t1 = x_t1c @ comps
        z_s1 = x_s1c @ comps
        return z_t8, z_t1, z_s1
    except RuntimeError:
        _, _, vh = torch.linalg.svd(x_t8c, full_matrices=False)
        comps = vh[:2].T
        z_t8 = x_t8c @ comps
        z_t1 = x_t1c @ comps
        z_s1 = x_s1c @ comps
        return z_t8, z_t1, z_s1


def _hex_rgb_torch(h: str) -> torch.Tensor:
    return torch.tensor(
        [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)], dtype=torch.float32
    ) / 255.0


def rainbow_colors_from_z(z_t8: torch.Tensor) -> torch.Tensor:
    z = torch.as_tensor(z_t8, dtype=torch.float32)
    x = z[:, 0]
    y = z[:, 1]

    xmin, xmax = x.min().item(), x.max().item()
    ymin, ymax = y.min().item(), y.max().item()
    xspan = max(xmax - xmin, 1e-12)
    yspan = max(ymax - ymin, 1e-12)
    x_n = torch.clamp((x - xmin) / xspan, 0.0, 1.0)
    y_n = torch.clamp((y - ymin) / yspan, 0.0, 1.0)

    c_tl = _hex_rgb_torch("#00CED1")
    c_tr = _hex_rgb_torch("#FFD700")
    c_bl = _hex_rgb_torch("#8A2BE2")
    c_br = _hex_rgb_torch("#FF7F50")

    x_n = x_n[:, None]
    y_n = y_n[:, None]
    colors = (
        (1 - x_n) * (1 - y_n) * c_bl
        + x_n * (1 - y_n) * c_br
        + (1 - x_n) * y_n * c_tl
        + x_n * y_n * c_tr
    )
    return torch.clamp(colors, 0.0, 1.0)


def plot_triplet_pca(
    z_t8: torch.Tensor,
    z_t1: torch.Tensor,
    z_s1: torch.Tensor,
    colors: torch.Tensor,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    panels = [
        (z_t8, "Teacher 8-step"),
        (z_t1, "Teacher 1-step"),
        (z_s1, "Student 1-step"),
    ]

    x_all = torch.cat([z_t8[:, 0], z_t1[:, 0], z_s1[:, 0]], dim=0)
    y_all = torch.cat([z_t8[:, 1], z_t1[:, 1], z_s1[:, 1]], dim=0)
    x_pad = 0.04 * max(float(x_all.max().item() - x_all.min().item()), 1e-9)
    y_pad = 0.04 * max(float(y_all.max().item() - y_all.min().item()), 1e-9)
    xlim = (float(x_all.min().item() - x_pad), float(x_all.max().item() + x_pad))
    ylim = (float(y_all.min().item() - y_pad), float(y_all.max().item() + y_pad))

    c_list = colors.detach().cpu().tolist()

    for ax, (pts, title) in zip(axes, panels):
        px = pts[:, 0].detach().cpu().tolist()
        py = pts[:, 1].detach().cpu().tolist()
        ax.scatter(px, py, s=5, c=c_list, alpha=0.55, edgecolors="none", rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt_dir", type=str, required=True)
    parser.add_argument("--teacher_pkl", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out_path or os.path.join(args.student_ckpt_dir, "teacher8_teacher1_student1.png")
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.num_images
    c = args.img_channels
    print(f"Generating shared inputs: N={n}, C={c}", flush=True)
    z = torch.randn(n, c, args.resolution, args.resolution, device=device)
    class_idx = torch.randint(0, args.label_dim, (n,), device=device)
    one_hot = torch.zeros(n, args.label_dim, device=device)
    one_hot[torch.arange(n, device=device), class_idx] = 1.0

    print("Loading student model...", flush=True)
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

    print("Generating student 1-step...", flush=True)
    s1 = run_student_batches(
        student=student,
        latents=z.detach().cpu(),
        one_hot=one_hot.detach().cpu(),
        conditioning_sigma=args.conditioning_sigma,
        device=device,
        batch_size=args.batch_size,
    )
    del student
    torch.cuda.empty_cache()

    print("Loading teacher model...", flush=True)
    teacher = load_teacher(args.teacher_pkl, device)
    latents_cpu = z.detach().cpu()
    one_hot_cpu = one_hot.detach().cpu()

    print("Generating teacher 1-step...", flush=True)
    t1 = run_teacher_batches(
        teacher=teacher,
        latents=latents_cpu,
        one_hot=one_hot_cpu,
        num_steps=1,
        device=device,
        batch_size=args.batch_size,
        desc="Teacher 1-step",
    )

    print("Generating teacher 8-step...", flush=True)
    t8 = run_teacher_batches(
        teacher=teacher,
        latents=latents_cpu,
        one_hot=one_hot_cpu,
        num_steps=8,
        device=device,
        batch_size=args.batch_size,
        desc="Teacher 8-step",
    )
    del teacher
    torch.cuda.empty_cache()

    z_t8, z_t1, z_s1 = pca_project_from_t8(t8=t8, t1=t1, s1=s1)
    colors = rainbow_colors_from_z(z_t8)
    plot_triplet_pca(z_t8=z_t8, z_t1=z_t1, z_s1=z_s1, colors=colors, out_path=out_path)


if __name__ == "__main__":
    main()
