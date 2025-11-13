from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import MetadataIndex, ProteinBatchCollator, ProteinSequenceDataset, load_split_paths
from models import ModelConfig, build_model
from utils import evaluate, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved ARG models")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--data_split", default="random")
    parser.add_argument("--subset", choices=["train", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_path", type=str, default=None, help="Optional custom CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    metadata = MetadataIndex(
        mechanism2id=checkpoint["metadata"]["mechanism2id"],
        drug2id=checkpoint["metadata"]["drug2id"],
        taxonomy2id=checkpoint["metadata"]["taxonomy2id"],
    )
    model_type = ckpt_args.get("model_type", "metadata")
    config = ModelConfig(
        seq_len=ckpt_args.get("seq_len", 2048),
        dropout=ckpt_args.get("dropout", 0.1),
        metadata_embedding_dim=ckpt_args.get("metadata_dim", 128),
        hidden_layers=tuple(ckpt_args.get("hidden_layers", [512, 256])),
        use_drug=ckpt_args.get("use_drug", True),
        use_taxonomy=ckpt_args.get("use_taxonomy", True),
    )

    model = build_model(model_type, metadata, config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    collator = ProteinBatchCollator()

    if args.csv_path:
        target_path = Path(args.csv_path)
    else:
        train_path, test_path = load_split_paths(Path(args.data_root), args.data_split)
        target_path = train_path if args.subset == "train" else test_path
    df = pd.read_csv(target_path)
    dataset = ProteinSequenceDataset(df, metadata)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    label_names = [label for label, _ in sorted(metadata.mechanism2id.items(), key=lambda kv: kv[1])]
    negative_index = metadata.mechanism2id["negative"]
    loss_fn = nn.CrossEntropyLoss()
    result = evaluate(model, dataloader, loss_fn, device, label_names, negative_index)

    print(json.dumps({"loss": result.loss, **result.metrics}, indent=2))


if __name__ == "__main__":
    main()
