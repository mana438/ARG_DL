from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    # DataLoader から得た dict を GPU/CPU に移す共通処理
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    arg_labels: np.ndarray,
    label_names: Sequence[str],
    negative_class_index: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(labels, predictions)
    metrics["macro_f1"] = f1_score(labels, predictions, average="macro", zero_division=0)
    per_class = f1_score(
        labels,
        predictions,
        average=None,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    for idx, score in enumerate(per_class):
        metrics[f"f1_{label_names[idx]}"] = float(score)
    arg_preds = (predictions != negative_class_index).astype(int)
    metrics["arg_f1"] = f1_score(arg_labels, arg_preds, average="binary", zero_division=0)
    return metrics


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = 1.0,
    scheduler=None,
) -> float:
    # AMP 対応の共通学習ループ。TensorFlow 側は feature extractor 内で完結しているためここでは PyTorch のみを扱う。
    model.train()
    total_loss = 0.0
    num_steps = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        with autocast(enabled=scaler is not None):
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        num_steps += 1
    return total_loss / max(1, num_steps)


def evaluate(
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device,
    label_names: Sequence[str],
    negative_class_index: int,
) -> EpochResult:
    # Macro-F1 や ARG/非ARG の F1 をまとめて算出する評価ルーチン
    model.eval()
    total_loss = 0.0
    num_steps = 0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_args: List[int] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            total_loss += loss.item()
            num_steps += 1
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_args.extend(batch["arg_labels"].cpu().numpy().tolist())
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_args),
        label_names,
        negative_class_index,
    )
    return EpochResult(loss=total_loss / max(1, num_steps), metrics=metrics)
