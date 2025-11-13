from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LEVEL_NAMES = [
    "Superkingdom",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
    "Species",
]

MECHANISM_ORDER = [
    "antibiotic target alteration",
    "antibiotic target replacement",
    "antibiotic target protection",
    "antibiotic inactivation",
    "antibiotic efflux",
    "others",
    "negative",
]


@dataclass
class MetadataIndex:
    mechanism2id: Dict[str, int]
    drug2id: Dict[str, int]
    taxonomy2id: List[Dict[str, int]]

    def to_json(self, path: Path) -> None:
        payload = {
            "mechanism2id": self.mechanism2id,
            "drug2id": self.drug2id,
            "taxonomy2id": self.taxonomy2id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "MetadataIndex":
        payload = json.loads(path.read_text())
        return cls(
            mechanism2id=payload["mechanism2id"],
            drug2id=payload["drug2id"],
            taxonomy2id=payload["taxonomy2id"],
        )

    @property
    def num_labels(self) -> int:
        return len(self.mechanism2id)

    @property
    def num_drugs(self) -> int:
        return len(self.drug2id)

    @property
    def taxonomy_vocab_sizes(self) -> List[int]:
        return [len(v) for v in self.taxonomy2id]

    def mechanism_id(self, label: str) -> int:
        if label not in self.mechanism2id:
            raise KeyError(f"Unknown mechanism label: {label}")
        return self.mechanism2id[label]

    def drug_id(self, target: str) -> int:
        return self.drug2id.get(target, self.drug2id["<unk-drug>"])

    def taxonomy_ids(self, taxonomy_strings: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for i, (token, vocab) in enumerate(zip(taxonomy_strings, self.taxonomy2id)):
            ids.append(vocab.get(token, vocab["<unk-" + LEVEL_NAMES[i] + ">"]))
        return ids


def load_split_paths(data_root: Path, split: str) -> Tuple[Path, Path]:
    split = split.lower()
    if split == "random":
        base = data_root / "HMDARG-DB"
        return base / "fold_5.train.csv", base / "fold_5.test.csv"
    if split.startswith("lhd"):
        suffix = split.replace("lhd", "")
        base = data_root / "LHD" / f"c{suffix}"
        return base / f"fold_5_{suffix}.train.csv", base / f"fold_5_{suffix}.test.csv"
    raise ValueError(f"Unsupported split '{split}'. Use 'random' or 'lhd0.4/0.6/0.8'.")


def parse_taxonomy(raw: str) -> List[str]:
    tokens = []
    raw = raw if isinstance(raw, str) else ""
    parts = [p.strip() for p in raw.split(";")]
    for level_idx in range(len(LEVEL_NAMES)):
        if level_idx < len(parts) and parts[level_idx]:
            tokens.append(parts[level_idx])
        else:
            tokens.append(f"Unknown_{LEVEL_NAMES[level_idx]}")
    return tokens[: len(LEVEL_NAMES)]


def build_metadata_index(frames: Iterable[pd.DataFrame]) -> MetadataIndex:
    drug_values = set()
    tax_vocab: List[Dict[str, int]] = [dict() for _ in LEVEL_NAMES]
    for i, level in enumerate(LEVEL_NAMES):
        tax_vocab[i][f"Unknown_{level}"] = 0

    mech2id = {label: idx for idx, label in enumerate(MECHANISM_ORDER)}

    for df in frames:
        for value in df["target"].fillna("Unknown").tolist():
            value = value.strip() if isinstance(value, str) else "Unknown"
            if value:
                drug_values.add(value)
        for species in df["species"].fillna("").tolist():
            levels = parse_taxonomy(species)
            for i, token in enumerate(levels):
                tax_vocab[i][token] = 0

    drug_list = sorted(drug_values)
    drug2id = {name: idx for idx, name in enumerate(drug_list, start=0)}
    if "Unknown" not in drug2id:
        drug2id["Unknown"] = len(drug2id)
    drug2id["<unk-drug>"] = len(drug2id)

    taxonomy2id: List[Dict[str, int]] = []
    for i, vocab in enumerate(tax_vocab):
        entries = sorted(vocab.keys())
        mapping = {token: idx for idx, token in enumerate(entries)}
        unk_token = f"<unk-{LEVEL_NAMES[i]}>"
        if unk_token not in mapping:
            mapping[unk_token] = len(mapping)
        taxonomy2id.append(mapping)

    return MetadataIndex(mechanism2id=mech2id, drug2id=drug2id, taxonomy2id=taxonomy2id)


class ProteinSequenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, metadata: MetadataIndex):
        self.metadata = metadata
        self.df = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        taxonomy_tokens = parse_taxonomy(row.get("species", ""))
        sample = {
            "record_id": row.get("ID", f"sample_{idx}"),
            "sequence": str(row.get("sequence", "")),
            "drug_id": self.metadata.drug_id(str(row.get("target", "Unknown"))),
            "taxonomy_ids": self.metadata.taxonomy_ids(taxonomy_tokens),
            "label_id": self.metadata.mechanism_id(str(row.get("mechanism"))),
        }
        sample["arg_label"] = 0 if row.get("mechanism") == "negative" else 1
        return sample


class ProteinBatchCollator:
    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        drug_ids = torch.tensor([sample["drug_id"] for sample in batch], dtype=torch.long)
        taxonomy_ids = torch.tensor([sample["taxonomy_ids"] for sample in batch], dtype=torch.long)
        labels = torch.tensor([sample["label_id"] for sample in batch], dtype=torch.long)
        arg_labels = torch.tensor([sample["arg_label"] for sample in batch], dtype=torch.long)
        return {
            "sequences": [sample["sequence"] for sample in batch],
            "drug_ids": drug_ids,
            "taxonomy_ids": taxonomy_ids,
            "labels": labels,
            "arg_labels": arg_labels,
            "record_ids": [sample["record_id"] for sample in batch],
        }
