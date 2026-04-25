#!/usr/bin/env python3
"""Random sample of LMDB rows -> one PNG grid (MNIST-style uint8 CHW)."""
import argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lmdb
from main.utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lmdb_path")
    ap.add_argument("-o", "--out", default="lmdb_preview.png")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    ish = get_array_shape_from_lmdb(env, "images")
    lsh = get_array_shape_from_lmdb(env, "labels")
    N, r, c = ish[0], 5, 6
    idxs = random.Random(args.seed).sample(range(N), min(args.n, N))
    fig, axs = plt.subplots(r, c, figsize=(c * 1.1, r * 1.1))
    for ax, i in zip(axs.flat, idxs):
        img = retrieve_row_from_lmdb(env, "images", np.uint8, ish[1:], i)
        x = img[0] if img.ndim == 3 else img
        lb = retrieve_row_from_lmdb(env, "labels", np.int64, lsh[1:], i)
        ax.imshow(x, cmap="gray", vmin=0, vmax=255)
        ax.set_title(str(int(lb.reshape(-1)[0])))
        ax.axis("off")
    for ax in axs.flat[len(idxs):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print("wrote", args.out)

if __name__ == "__main__":
    main()