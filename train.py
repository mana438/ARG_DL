from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data_utils import (
    MetadataIndex,
    ProteinBatchCollator,
    ProteinSequenceDataset,
    build_metadata_index,
    load_split_paths,
)
from models import MODEL_REGISTRY, ModelConfig, build_model
from utils import evaluate, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARG mechanism classifiers")
    parser.add_argument("--model_type", choices=MODEL_REGISTRY.keys(), default="metadata")
    parser.add_argument("--data_split", default="random", help="random or lhd0.4/0.6/0.8")
    parser.add_argument("--data_root", default="data", help="Path to data directory")
    parser.add_argument("--output_dir", default="outputs", help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--metadata_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, nargs="*", default=[512, 256])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--tf_gpu",
        action="store_true",
        help="Allow TensorFlow (ProteinBERT) to use GPU when computing embeddings on-the-fly.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sorted_label_names(metadata: MetadataIndex) -> Sequence[str]:
    return [label for label, _ in sorted(metadata.mechanism2id.items(), key=lambda item: item[1])]


def create_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return max(
            0.0,
            (num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps),
        )

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metadata: MetadataIndex,
    args: argparse.Namespace,
    epoch: int,
    best_metric: float,
) -> None:
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metadata": {
            "mechanism2id": metadata.mechanism2id,
            "drug2id": metadata.drug2id,
            "taxonomy2id": metadata.taxonomy2id,
        },
        "args": vars(args),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path, test_path = load_split_paths(Path(args.data_root), args.data_split)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    metadata = build_metadata_index([train_df, test_df])

    collator = ProteinBatchCollator()

    # NOTE: 論文の設定では train/test のみを使用するため、デフォルトでは train 全量を学習に回す。
    train_dataset = ProteinSequenceDataset(train_df, metadata)
    test_dataset = ProteinSequenceDataset(test_df, metadata)

    if args.val_ratio > 0:
        # オプションで validation を作る場合のみ stratify 付きで分割する。
        train_subset, valid_subset = train_test_split(
            train_df,
            test_size=args.val_ratio,
            stratify=train_df["mechanism"],
            random_state=args.seed,
        )
        train_dataset = ProteinSequenceDataset(train_subset.reset_index(drop=True), metadata)
        valid_dataset = ProteinSequenceDataset(valid_subset.reset_index(drop=True), metadata)
        valid_loader: Optional[DataLoader] = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
    else:
        valid_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    config = ModelConfig(
        seq_len=args.seq_len,
        dropout=args.dropout,
        metadata_embedding_dim=args.metadata_dim,
        hidden_layers=tuple(args.hidden_layers),
        use_tf_gpu=args.tf_gpu,
    )
    model = build_model(args.model_type, metadata, config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = create_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    label_counts = train_df["mechanism"].value_counts().to_dict()
    total_labels = sum(label_counts.values())
    num_labels = metadata.num_labels
    class_weight_list = []
    for label in sorted_label_names(metadata):
        count = label_counts.get(label, 0)
        if count == 0:
            weight = 0.0
        else:
            weight = total_labels / (num_labels * count)
        class_weight_list.append(weight)
    class_weights = torch.tensor(class_weight_list, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    label_names = list(sorted_label_names(metadata))
    negative_index = metadata.mechanism2id["negative"]

    best_val = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            scheduler=scheduler,
        )
        val_result = (
            evaluate(model, valid_loader, loss_fn, device, label_names, negative_index)
            if valid_loader is not None
            else None
        )
        test_result = evaluate(model, test_loader, loss_fn, device, label_names, negative_index)

        # 学習ログを JSON 形式でそのまま標準出力に流す（後で jq などで解析しやすくするため）
        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_result.metrics if val_result else {},
            "test": test_result.metrics,
        }
        print(json.dumps(log_payload, indent=2))
        history.append(log_payload)

        # validation が無い場合はテスト指標をベスト判定に使う
        reference_metric = (
            val_result.metrics["macro_f1"] if val_result else test_result.metrics["macro_f1"]
        )
        is_best = reference_metric > best_val
        if is_best:
            best_val = reference_metric
    final_ckpt_path = output_dir / "model_final.pt"
    save_checkpoint(final_ckpt_path, model, optimizer, scheduler, metadata, args, args.epochs, best_val)

    history_path = output_dir / "results.json"
    payload = {
        "history": history,
        "best_tracked_macro_f1": best_val,
        "arguments": vars(args),
    }
    history_path.write_text(json.dumps(payload, indent=2))

    print(f"Best tracked macro F1: {best_val:.4f}")
    print(f"Saved final checkpoint to {final_ckpt_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
