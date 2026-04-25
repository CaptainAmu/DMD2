#!/usr/bin/env python3
"""MNIST: noise–image L2 distance matrix, heatmap, discrete OT (POT), and 3×K comparison grid.

Requires: pip install POT  (import name: ot)
          scipy (cdist)

Defaults: 32×32, 10 classes, 1 channel. Teacher helpers are inlined (no main.edm import at load).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import dnnlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist
from tqdm import tqdm

_EVAL = os.path.dirname(os.path.abspath(__file__))
_DMD2_ROOT = os.path.abspath(os.path.join(_EVAL, "..", "..", ".."))
if _DMD2_ROOT not in sys.path:
    sys.path.insert(0, _DMD2_ROOT)


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


def _parse_viz_indices(s: str | None, n: int, default_cols: int) -> np.ndarray:
    if s is None or s.strip() == "":
        k = min(default_cols, n)
        return np.arange(k, dtype=np.int64)
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    idx = np.array([int(p) for p in parts], dtype=np.int64)
    if np.any((idx < 0) | (idx >= n)):
        raise ValueError(f"viz_indices must be in [0, {n})")
    return idx


def _flatten_for_cdist(t: torch.Tensor, n: int) -> np.ndarray:
    t = t.detach().float().cpu().contiguous().reshape(n, -1)
    return np.ascontiguousarray(t.numpy(), dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="MNIST: Gaussian noise vs teacher images, L2 matrix, OT, 3-row viz.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Dir containing mnist_checkpoint_model/network-snapshot-*.pkl (see --teacher_pkl).",
    )
    parser.add_argument(
        "--teacher_pkl",
        type=str,
        default=None,
        help="Full path to teacher pickle (default: CHECKPOINT_PATH/mnist_checkpoint_model/network-snapshot-004659.pkl).",
    )
    parser.add_argument("--class_idx", type=int, required=True, help="MNIST digit class 0..9 (fixed conditioning).")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--img_channels", type=int, default=1, help="Latent channels (typically 1 for MNIST).")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_steps", type=int, default=8, help="Teacher EDM sampler steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--heatmap_size", type=int, default=3000, help="Per-panel heatmap size in pixels (width of combined = 2x).")
    parser.add_argument("--viz_cols", type=int, default=50)
    parser.add_argument("--viz_indices", type=str, default=None)
    parser.add_argument(
        "--cost",
        type=str,
        choices=("sqeuclidean", "euclidean"),
        default="sqeuclidean",
        help="OT ground cost: squared L2 (default) or L2 (same as M).",
    )
    parser.add_argument("--cdist_chunk", type=int, default=0)
    args = parser.parse_args()

    if args.teacher_pkl:
        teacher_pkl = args.teacher_pkl
    elif args.checkpoint_path:
        teacher_pkl = os.path.join(args.checkpoint_path, "mnist_checkpoint_model", "network-snapshot-004659.pkl")
    else:
        parser.error("Provide --teacher_pkl or --checkpoint_path")

    if not (0 <= args.class_idx < args.label_dim):
        raise ValueError("class_idx must be in [0, label_dim)")

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import ot  # noqa: F401
    except ImportError as e:
        raise ImportError("Install POT: pip install POT  (provides import ot)") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.num_samples
    c = args.img_channels
    z = torch.randn(n, c, args.resolution, args.resolution, device=device)
    one_hot = torch.zeros(n, args.label_dim, device=device)
    one_hot[:, args.class_idx] = 1.0

    teacher = load_teacher(teacher_pkl, device)
    latents_cpu = z.detach().cpu()
    x = run_teacher_batches(
        teacher,
        latents_cpu,
        one_hot.cpu(),
        num_steps=args.num_steps,
        device=device,
        batch_size=args.batch_size,
        desc="Teacher EDM",
    )

    Z = _flatten_for_cdist(z, n)
    X = _flatten_for_cdist(x, n)

    if args.cdist_chunk and args.cdist_chunk > 0:
        rows = []
        for s in tqdm(range(0, n, args.cdist_chunk), desc="cdist chunks", unit="chunk"):
            e = min(s + args.cdist_chunk, n)
            rows.append(cdist(Z[s:e], X, metric="euclidean").astype(np.float32))
        M = np.vstack(rows)
    else:
        M = cdist(Z, X, metric="euclidean").astype(np.float32)

    np.save(os.path.join(args.out_dir, "distance_matrix.npy"), M)

    if args.cost == "sqeuclidean":
        C = (M**2).astype(np.float64)
    else:
        C = M.astype(np.float64)

    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    T = ot.emd(a, b, C)
    total_cost = float(np.sum(T * C))

    np.save(os.path.join(args.out_dir, "ot_plan.npy"), T.astype(np.float32))
    cost_txt = os.path.join(args.out_dir, "ot_total_cost.txt")
    with open(cost_txt, "w", encoding="utf-8") as f:
        f.write(f"{total_cost}\n")
        f.write(f"cost_metric={args.cost}\n")
    print(f"OT total cost = {total_cost} ({args.cost})", flush=True)

    support = T > 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    vmin = float(np.quantile(M, 0.01))
    vmax = float(np.quantile(M, 0.99))
    w_in = args.heatmap_size / 100.0

    fig = plt.figure(figsize=(w_in, w_in), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("L2 distance ||Z[i] - X[j]||_2 (MNIST)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    heatmap_path = os.path.join(args.out_dir, "distance_heatmap.png")
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {heatmap_path}", flush=True)

    cmap_support = ListedColormap(["white", "red"])
    fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(2 * w_in, w_in), dpi=100)
    im0 = ax0.imshow(M, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax0.set_title("L2 distance ||Z[i] - X[j]||_2 (MNIST)")
    fig2.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    ax1.imshow(support.astype(np.float32), cmap=cmap_support, aspect="auto", vmin=0, vmax=1)
    ax1.set_title("OT plan support (T[i,j] > 0)")
    combined_path = os.path.join(args.out_dir, "distance_and_ot_plan.png")
    fig2.savefig(combined_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {combined_path}", flush=True)

    viz_idx = _parse_viz_indices(args.viz_indices, n, args.viz_cols)
    idx_t = torch.tensor(viz_idx, device=z.device, dtype=torch.long)
    z_sel = z[idx_t]
    x_paired = x[idx_t]
    j_ot = np.argmax(T[viz_idx, :], axis=1)
    jg = torch.tensor(j_ot, device=x.device, dtype=torch.long)
    x_ot = x[jg]

    row_z = tensor_to_uint8_nhwc(z_sel)
    row_paired = tensor_to_uint8_nhwc(x_paired)
    row_ot = tensor_to_uint8_nhwc(x_ot)
    grid = build_grid_rows([row_z, row_paired, row_ot])
    grid = add_left_row_labels(
        grid,
        [
            "MNIST noise z (clip [-1,1] → RGB)",
            f"teacher x from same z (digit {args.class_idx}, {args.num_steps}-step)",
            "OT-matched x (argmax_j T[i,j])",
        ],
    )
    out_png = os.path.join(args.out_dir, "ot_noise_compare_grid.png")
    Image.fromarray(grid).save(out_png)
    print(f"Saved {out_png} shape={grid.shape}", flush=True)


if __name__ == "__main__":
    main()
