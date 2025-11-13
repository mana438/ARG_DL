#!/usr/bin/env python3
"""Precompute ProteinBERT embeddings and append to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from proteinbert_adapter import ProteinBertEmbedder


def serialize_vector(vec: np.ndarray) -> str:
    return " ".join(f"{x:.6f}" for x in vec.tolist())


def compute_embeddings(
    sequences: List[str],
    embedder: ProteinBertEmbedder,
) -> np.ndarray:
    embeddings = embedder.encode(sequences)
    return embeddings.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ProteinBERT embeddings for a CSV file")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV (must contain 'sequence' column)")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Path to output CSV. If omitted, input CSV will be overwritten in-place.",
    )
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length fed to ProteinBERT")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding computation")
    parser.add_argument("--tf_gpu", action="store_true", help="Use GPU for TensorFlow/ProteinBERT")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rows that already have a non-empty 'embedding' column",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else input_path

    df = pd.read_csv(input_path)
    if "sequence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column")

    if "embedding" not in df.columns or not args.skip_existing:
        df["embedding"] = ""

    existing = df["embedding"].fillna("").astype(str)
    if args.skip_existing:
        mask = existing.str.strip() == ""
    else:
        mask = np.ones(len(df), dtype=bool)

    indices = np.where(mask)[0]
    if len(indices) == 0:
        print("No rows require embedding computation.")
        if output_path != input_path:
            df.to_csv(output_path, index=False)
        return

    embedder = ProteinBertEmbedder(seq_len=args.seq_len, use_tf_gpu=args.tf_gpu)
    sequences = df["sequence"].fillna("").astype(str)

    for start in tqdm(range(0, len(indices), args.batch_size), desc="embedding"):
        slice_idx = indices[start : start + args.batch_size]
        batch_sequences = sequences.iloc[slice_idx].tolist()
        vectors = compute_embeddings(batch_sequences, embedder)
        for row_idx, vec in zip(slice_idx, vectors):
            df.at[row_idx, "embedding"] = serialize_vector(vec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
