#!/usr/bin/env python3
"""Top-k closest/farthest aligned teacher-student pairs for ImageNet-64.

Workflow:
1) Sample K shared Gaussian noises z for one fixed class index.
2) Generate teacher T8 (8-step EDM) and student S1 (single-step) outputs.
3) Rank aligned pair distances d_i = ||T8_i - S1_i||_2.
4) For each selected teacher sample, fetch nearest and second-nearest train images.
5) Save 4-row grids:
   row1: Teacher 8-step
   row2: Student 1-step
   row3: Closest train to teacher 8-step
   row4: 2nd closest train to teacher 8-step
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time

import dnnlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

_DMD2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _DMD2_ROOT not in sys.path:
    sys.path.insert(0, _DMD2_ROOT)

from main.data.lmdb_dataset import LMDBDataset


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


def add_left_row_labels(grid: np.ndarray, row_labels: list[str], gutter_px: int = 260) -> np.ndarray:
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


def tensor_to_uint8_nhwc(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.permute(0, 2, 3, 1).numpy()


def build_grid_rows(row_tensors_uint8: list[np.ndarray]) -> np.ndarray:
    n = row_tensors_uint8[0].shape[0]
    strips = []
    for row in row_tensors_uint8:
        assert row.shape[0] == n, "all rows must have the same number of images"
        strips.append(np.concatenate([row[i] for i in range(n)], axis=1))
    return np.concatenate(strips, axis=0)


def load_train_lmdb_vectors_for_class(
    train_lmdb: str, class_idx: int
) -> tuple[torch.Tensor, LMDBDataset, torch.Tensor]:
    ds = LMDBDataset(train_lmdb)
    n = len(ds)
    class_indices: list[int] = []
    for i in tqdm(range(n), desc=f"Scan labels (class={class_idx})", unit="img"):
        y = int(ds[i]["class_labels"].item())
        if y == class_idx:
            class_indices.append(i)

    if len(class_indices) < 2:
        raise RuntimeError(
            f"Not enough train images for class_idx={class_idx}. "
            f"Found {len(class_indices)} image(s), need at least 2."
        )

    class_indices_t = torch.tensor(class_indices, dtype=torch.long)
    flat_dim = ds[class_indices[0]]["images"].numel()
    out = torch.empty((len(class_indices), flat_dim), dtype=torch.float32)
    for row, i in enumerate(tqdm(class_indices, desc=f"Load class {class_idx} LMDB", unit="img")):
        x = ds[i]["images"]  # [0,1]
        out[row] = (x * 2.0 - 1.0).reshape(-1)  # map to [-1,1]
    return out, ds, class_indices_t


def gather_train_images_uint8_nhwc(ds: LMDBDataset, global_indices: torch.Tensor) -> np.ndarray:
    rows = []
    for idx in global_indices.cpu().tolist():
        x = ds[idx]["images"]  # [0,1], CHW
        x_u8 = (x * 255.0).clamp(0, 255).to(torch.uint8)
        rows.append(x_u8.permute(1, 2, 0).numpy())
    return np.stack(rows, axis=0)


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
) -> torch.Tensor:
    n = latents.shape[0]
    outs = []
    for start in tqdm(range(0, n, batch_size), desc=f"Teacher {num_steps}-step", unit="batch"):
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
        outs.append(out.float().cpu())
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
        outs.append(out.float().cpu())
    return torch.cat(outs, dim=0)


def save_rank_grid(
    out_path: str,
    rank_indices: torch.Tensor,
    teacher_imgs: torch.Tensor,
    student_imgs: torch.Tensor,
    train_ds: LMDBDataset,
    idx1_global: torch.Tensor,
    idx2_global: torch.Tensor,
    teacher_steps: int,
    class_idx: int,
) -> None:
    t_u8 = tensor_to_uint8_nhwc(teacher_imgs[rank_indices])
    s_u8 = tensor_to_uint8_nhwc(student_imgs[rank_indices])
    tc1_u8 = gather_train_images_uint8_nhwc(train_ds, idx1_global[rank_indices])
    tc2_u8 = gather_train_images_uint8_nhwc(train_ds, idx2_global[rank_indices])
    grid = build_grid_rows([t_u8, s_u8, tc1_u8, tc2_u8])
    grid = add_left_row_labels(
        grid,
        [
            f"Teacher {teacher_steps}-step (class {class_idx})",
            f"Student 1-step (class {class_idx})",
            f"Closest train to Teacher {teacher_steps}-step",
            f"2nd closest train to Teacher {teacher_steps}-step",
        ],
    )
    Image.fromarray(grid).save(out_path)
    print(f"Saved {out_path}", flush=True)


def write_rank_summary(
    out_path: str,
    rank_name: str,
    indices: torch.Tensor,
    pair_distances: torch.Tensor,
    idx1_global: torch.Tensor,
    idx2_global: torch.Tensor,
    d1_all: torch.Tensor,
    d2_all: torch.Tensor,
) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"[{rank_name}]\n")
        f.write("rank\tpair_idx\tpair_dist\ttrain1_idx\ttrain1_dist\ttrain2_idx\ttrain2_dist\n")
        for rank, i in enumerate(indices.cpu().tolist(), start=1):
            f.write(
                f"{rank}\t{i}\t{pair_distances[i].item():.8f}\t"
                f"{idx1_global[i].item()}\t{d1_all[i].item():.8f}\t"
                f"{idx2_global[i].item()}\t{d2_all[i].item():.8f}\n"
            )
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet top-k closest/farthest teacher-student aligned pairs.")
    parser.add_argument("--teacher_pkl", type=str, required=True)
    parser.add_argument("--student_ckpt_dir", type=str, required=True)
    parser.add_argument("--train_lmdb", type=str, required=True)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/slurm-storage/shucli/PROJECT_FOLDER/DMD2/experiments/imagenet/TopKs",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--teacher_num_steps", type=int, default=8)
    parser.add_argument("--class_idx", type=int, default=207, help="ImageNet class index (default 207 = golden retriever).")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--label_dim", type=int, default=1000)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--nn_chunk", type=int, default=100)
    args = parser.parse_args()

    if not (0 <= args.class_idx < args.label_dim):
        raise ValueError(f"class_idx must be in [0, {args.label_dim - 1}], got {args.class_idx}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_txt = os.path.join(args.out_dir, "top10_pair_summary.txt")
    if os.path.exists(summary_txt):
        os.remove(summary_txt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.num_samples
    z = torch.randn(n, args.img_channels, args.resolution, args.resolution, device=device)
    one_hot = torch.zeros(n, args.label_dim, device=device)
    one_hot[:, args.class_idx] = 1.0
    latents_cpu = z.cpu()
    one_hot_cpu = one_hot.cpu()

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
    s = run_student_batches(student, latents_cpu, one_hot_cpu, args.conditioning_sigma, device, args.batch_size)
    del student
    torch.cuda.empty_cache()

    teacher = load_teacher(args.teacher_pkl, device)
    t = run_teacher_batches(teacher, latents_cpu, one_hot_cpu, args.teacher_num_steps, device, args.batch_size)
    del teacher
    torch.cuda.empty_cache()

    t_flat = t.reshape(n, -1).float()
    s_flat = s.reshape(n, -1).float()
    pair_dist = torch.linalg.vector_norm(t_flat - s_flat, ord=2, dim=1)

    train_flat, train_ds, class_global_indices = load_train_lmdb_vectors_for_class(args.train_lmdb, args.class_idx)
    d1_all, d2_all, idx1_local, idx2_local = top2_train_l2(t_flat, train_flat.float(), args.nn_chunk)
    idx1_global = class_global_indices[idx1_local]
    idx2_global = class_global_indices[idx2_local]

    top_k = min(args.top_k, n)
    sorted_idx = torch.argsort(pair_dist)
    closest_idx = sorted_idx[:top_k]
    farthest_idx = sorted_idx[-top_k:].flip(0)

    save_rank_grid(
        os.path.join(args.out_dir, "closest_top10_pairs.png"),
        closest_idx,
        t,
        s,
        train_ds,
        idx1_global,
        idx2_global,
        teacher_steps=args.teacher_num_steps,
        class_idx=args.class_idx,
    )
    save_rank_grid(
        os.path.join(args.out_dir, "farthest_top10_pairs.png"),
        farthest_idx,
        t,
        s,
        train_ds,
        idx1_global,
        idx2_global,
        teacher_steps=args.teacher_num_steps,
        class_idx=args.class_idx,
    )

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("top10_teacher_student_pairs summary\n")
        f.write(f"teacher_pkl={args.teacher_pkl}\n")
        f.write(f"student_ckpt_dir={args.student_ckpt_dir}\n")
        f.write(f"train_lmdb={args.train_lmdb}\n")
        f.write(f"num_samples={args.num_samples}\n")
        f.write(f"top_k={top_k}\n")
        f.write(f"teacher_num_steps={args.teacher_num_steps}\n")
        f.write(f"class_idx={args.class_idx}\n")
        f.write(f"class_subset_size={train_flat.shape[0]}\n")
        f.write(f"seed={args.seed}\n\n")

    write_rank_summary(summary_txt, "closest", closest_idx, pair_dist, idx1_global, idx2_global, d1_all, d2_all)
    write_rank_summary(summary_txt, "farthest", farthest_idx, pair_dist, idx1_global, idx2_global, d1_all, d2_all)

    print(f"Saved {summary_txt}", flush=True)


if __name__ == "__main__":
    main()
