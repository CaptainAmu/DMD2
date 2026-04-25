#!/usr/bin/env python3
"""4-row grid: Teacher 1-step | 8-step | 32-step | Student 1-step (MNIST 32x32, shared noise and classes)."""
from __future__ import annotations

DEFAULT_NUM_IMAGES = 30

import argparse
import copy
import os
import sys
import time

import dnnlib
import numpy as np
import pickle
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


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


LABEL_GUTTER_PX = 220


def add_left_row_labels(grid: np.ndarray, row_labels: list[str], gutter_px: int = LABEL_GUTTER_PX) -> np.ndarray:
    h, w, _ = grid.shape
    n_rows = len(row_labels)
    assert n_rows >= 1
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


def tensor_to_uint8_nhwc(x: torch.Tensor) -> np.ndarray:
    """x: (N,C,H,W) float ~[-1,1]; C=1 or 3. Output (N,H,W,3) uint8."""
    x = x.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = x.permute(0, 2, 3, 1).numpy()
    return x


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
):
    n = latents.shape[0]
    outs = []
    for start in tqdm(range(0, n, batch_size), desc=desc, unit="batch"):
        end = min(start + batch_size, n)
        lat_b = latents[start:end].to(device)
        oh_b = one_hot[start:end].to(device)
        out = edm_sampler(
            teacher,
            None,
            lat_b,
            class_labels=oh_b,
            num_steps=num_steps,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            S_churn=0,
        )
        outs.append(out.float())
    return torch.cat(outs, dim=0)


def build_grid_rows(row_tensors_uint8: list[np.ndarray]) -> np.ndarray:
    assert len(row_tensors_uint8) >= 1
    n = row_tensors_uint8[0].shape[0]
    strips = []
    for row in row_tensors_uint8:
        assert row.shape[0] == n, "all rows must have the same number of images"
        strips.append(np.concatenate([row[i] for i in range(n)], axis=1))
    return np.concatenate(strips, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt_dir", type=str, required=True)
    parser.add_argument("--teacher_pkl", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_path", type=str, default="teacher_student_comparison_4rows.png")
    parser.add_argument("--teacher_batch_size", type=int, default=20)
    parser.add_argument("--student_batch_size", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.num_images
    c = args.img_channels
    z = torch.randn(n, c, args.resolution, args.resolution, device=device)

    class_idx = torch.randint(0, args.label_dim, (n,), device=device)
    one_hot = torch.zeros(n, args.label_dim, device=device)
    one_hot[torch.arange(n, device=device), class_idx] = 1.0

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
    print(student.load_state_dict(state_dict, strict=True))
    student = student.to(device)
    student.eval()

    student_outs = []
    for start in tqdm(range(0, n, args.student_batch_size), desc="Student 1-step", unit="batch"):
        end = min(start + args.student_batch_size, n)
        z_b = z[start:end]
        oh_b = one_hot[start:end]
        ts_b = torch.ones(end - start, device=device, dtype=torch.long)
        with torch.no_grad():
            out_b = student(z_b * args.conditioning_sigma, ts_b * args.conditioning_sigma, oh_b)
        student_outs.append(out_b)
    student_u8 = tensor_to_uint8_nhwc(torch.cat(student_outs, dim=0))

    teacher = load_teacher(args.teacher_pkl, device)
    latents_cpu = z.detach().cpu()

    t1 = run_teacher_batches(
        teacher,
        latents_cpu,
        one_hot.cpu(),
        num_steps=1,
        device=device,
        batch_size=args.teacher_batch_size,
        desc="Teacher 1-step",
    )
    t8 = run_teacher_batches(
        teacher,
        latents_cpu,
        one_hot.cpu(),
        num_steps=8,
        device=device,
        batch_size=args.teacher_batch_size,
        desc="Teacher 8-step",
    )
    t32 = run_teacher_batches(
        teacher,
        latents_cpu,
        one_hot.cpu(),
        num_steps=32,
        device=device,
        batch_size=args.teacher_batch_size,
        desc="Teacher 32-step",
    )

    t1_u8 = tensor_to_uint8_nhwc(t1)
    t8_u8 = tensor_to_uint8_nhwc(t8)
    t32_u8 = tensor_to_uint8_nhwc(t32)

    grid = build_grid_rows([t1_u8, t8_u8, t32_u8, student_u8])
    labels = ["Teacher 1-step", "Teacher 8-step", "Teacher 32-step", "Student 1-step"]
    grid_labeled = add_left_row_labels(grid, labels)
    Image.fromarray(grid_labeled).save(args.out_path)
    print(f"Saved {args.out_path} shape={grid_labeled.shape}", flush=True)


if __name__ == "__main__":
    main()
