#!/usr/bin/env python3
"""One-shot FID for MNIST student checkpoint (Inception features on grayscale repeated to RGB)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm

_DMD2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _DMD2_ROOT not in sys.path:
    sys.path.insert(0, _DMD2_ROOT)

from main.edm.test_folder_edm import (
    calculate_fid_from_inception_stats,
    calculate_inception_stats,
    create_evaluator,
    create_generator,
)


@torch.no_grad()
def sample_with_tqdm(accelerator, current_model, args):
    timesteps = torch.ones(args.eval_batch_size, device=accelerator.device, dtype=torch.long)
    current_model.eval()
    all_images_tensor = []

    current_index = 0
    all_labels = torch.arange(0, args.total_eval_samples * 2, device=accelerator.device, dtype=torch.long) % args.label_dim

    set_seed(args.seed + accelerator.process_index)

    total_batches = (args.total_eval_samples + args.eval_batch_size - 1) // args.eval_batch_size
    if accelerator.is_main_process:
        print(
            "[FID 1/2] Student generator: sampling "
            f"({args.total_eval_samples} total, batch_size={args.eval_batch_size}).",
            flush=True,
        )
    pbar = tqdm(
        range(total_batches),
        desc="[1/2] Student sampling",
        disable=not accelerator.is_main_process,
        unit="batch",
    )

    c = args.img_channels
    for _ in pbar:
        noise = torch.randn(
            args.eval_batch_size,
            c,
            args.resolution,
            args.resolution,
            device=accelerator.device,
        )
        random_labels = all_labels[current_index : current_index + args.eval_batch_size]
        one_hot_labels = torch.eye(args.label_dim, device=accelerator.device)[random_labels]
        current_index += args.eval_batch_size

        eval_images = current_model(
            noise * args.conditioning_sigma,
            timesteps * args.conditioning_sigma,
            one_hot_labels,
        )
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        if c == 1:
            eval_images = eval_images.repeat(1, 1, 1, 3)
        eval_images = eval_images.contiguous()
        gathered_images = accelerator.gather(eval_images)
        all_images_tensor.append(gathered_images.cpu())

    all_images_tensor = torch.cat(all_images_tensor, dim=0)[: args.total_eval_samples]
    if accelerator.is_main_process:
        print(f"Collected {len(all_images_tensor)} images for FID.")
    accelerator.wait_for_everyone()
    return all_images_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--detector_url", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--total_eval_samples", type=int, default=50000)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    args = parser.parse_args()

    ckpt_bin = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
    if not os.path.isfile(ckpt_bin):
        raise FileNotFoundError(f"Missing {ckpt_bin}")

    accelerator_project_config = ProjectConfiguration(logging_dir=args.checkpoint_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        project_config=accelerator_project_config,
    )
    print(accelerator.state, flush=True)

    evaluator, evaluator_kwargs, feature_dim = create_evaluator(args.detector_url)
    evaluator = accelerator.prepare(evaluator)

    ref_dict = dict(np.load(args.ref_path))

    generator = create_generator(ckpt_bin, base_model=None, dataset_name="mnist")
    generator = generator.to(accelerator.device)

    all_images_tensor = sample_with_tqdm(accelerator, generator, args)

    if accelerator.is_main_process:
        print(
            "[FID 2/2] Inception: extract features (μ,Σ) "
            f"({len(all_images_tensor)} images, max_batch_size={args.max_batch_size}).",
            flush=True,
        )
    pred_mu, pred_sigma = calculate_inception_stats(
        all_images_tensor,
        evaluator,
        accelerator,
        evaluator_kwargs,
        feature_dim,
        args.max_batch_size,
        tqdm_desc="[2/2] Inception (μ,Σ)",
    )

    if accelerator.is_main_process:
        fid = calculate_fid_from_inception_stats(
            pred_mu, pred_sigma, ref_dict["mu"], ref_dict["sigma"]
        )
        print(f"FID = {fid}", flush=True)
        out_path = os.path.join(args.checkpoint_dir, "fid_once.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"fid={fid}\n")
            f.write(f"total_eval_samples={args.total_eval_samples}\n")
        print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
