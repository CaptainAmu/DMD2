#!/usr/bin/env python3
"""Compute Inception μ,Σ on real MNIST (LMDB) for FID reference (1ch -> repeat RGB uint8)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_DMD2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _DMD2_ROOT not in sys.path:
    sys.path.insert(0, _DMD2_ROOT)

from main.data.lmdb_dataset import LMDBDataset
from main.edm.test_folder_edm import calculate_inception_stats, create_evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True, help="MNIST LMDB (same format as train_edm)")
    parser.add_argument("--out_path", type=str, required=True, help="Output .npz with keys mu, sigma")
    parser.add_argument("--detector_url", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = LMDBDataset(args.lmdb_path)
    n = min(len(ds), args.num_samples)
    perm = torch.randperm(len(ds))[:n].tolist()
    subset = Subset(ds, perm)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    chunks = []
    for batch in tqdm(loader, desc="Load LMDB"):
        im = batch["images"]
        if im.dim() == 3:
            im = im.unsqueeze(1)
        u8 = (im * 255.0).clamp(0, 255).byte()
        u8 = u8.permute(0, 2, 3, 1)
        u8 = u8.repeat(1, 1, 1, 3)
        chunks.append(u8.cpu())

    all_images_tensor = torch.cat(chunks, dim=0)
    print(f"Stacked {len(all_images_tensor)} reference images (NHWC uint8 RGB).", flush=True)

    accelerator_project_config = ProjectConfiguration(logging_dir=os.path.dirname(args.out_path) or ".")
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        project_config=accelerator_project_config,
    )
    evaluator, evaluator_kwargs, feature_dim = create_evaluator(args.detector_url)
    evaluator = accelerator.prepare(evaluator)

    pred_mu, pred_sigma = calculate_inception_stats(
        all_images_tensor,
        evaluator,
        accelerator,
        evaluator_kwargs,
        feature_dim,
        args.max_batch_size,
        tqdm_desc="Inception ref (μ,Σ)",
    )

    if accelerator.is_main_process:
        np.savez(args.out_path, mu=pred_mu, sigma=pred_sigma)
        print(f"Wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
