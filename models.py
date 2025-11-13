from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn

from proteinbert_adapter import ProteinBertEmbedder

if TYPE_CHECKING:
    from data_utils import MetadataIndex


@dataclass
class ModelConfig:
    seq_len: int = 2048
    dropout: float = 0.1
    metadata_embedding_dim: int = 128
    hidden_layers: List[int] | tuple[int, ...] = (512, 256)


class ProteinBertSequenceEncoder(nn.Module):
    """ProteinBERT のグローバル表現を 512 次元に射影して返す共通エンコーダ。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedder = ProteinBertEmbedder(seq_len=config.seq_len)
        self.projection = nn.Linear(self.embedder.global_dim, 512)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            seq_features = self.embedder.encode(sequences)
        seq_features = seq_features.to(device)
        seq_features = self.projection(seq_features)
        return self.dropout(seq_features)


class SequenceOnlyModel(nn.Module):
    """配列情報のみで予測するベースラインモデル (512次元→MLP)。"""

    def __init__(self, metadata: "MetadataIndex", config: ModelConfig) -> None:
        super().__init__()
        self.encoder = ProteinBertSequenceEncoder(config)
        self.classifier = self._build_classifier(512, metadata.num_labels, config)

    def _build_classifier(self, input_dim: int, num_labels: int, config: ModelConfig) -> nn.Module:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden in config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_labels))
        return nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch["labels"].device
        seq_features = self.encoder(batch["sequences"], device)
        return self.classifier(seq_features)


class MetadataEnhancedModel(nn.Module):
    """配列 + 薬剤 + 生物種情報を結合した提案モデル。"""

    def __init__(self, metadata: "MetadataIndex", config: ModelConfig) -> None:
        super().__init__()
        self.encoder = ProteinBertSequenceEncoder(config)
        embed_dim = config.metadata_embedding_dim
        self.drug_embedding = nn.Embedding(metadata.num_drugs, embed_dim)
        self.taxonomy_embeddings = nn.ModuleList(
            [nn.Embedding(size, embed_dim) for size in metadata.taxonomy_vocab_sizes]
        )
        feature_dim = 512 + embed_dim * 2  # 512 (seq) + 128 (drug) + 128 (species)
        self.classifier = self._build_classifier(feature_dim, metadata.num_labels, config)

    def _build_classifier(self, input_dim: int, num_labels: int, config: ModelConfig) -> nn.Module:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden in config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_labels))
        return nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch["labels"].device
        seq_features = self.encoder(batch["sequences"], device)
        drug_embed = self.drug_embedding(batch["drug_ids"])
        tax_embeds = [emb(batch["taxonomy_ids"][:, idx]) for idx, emb in enumerate(self.taxonomy_embeddings)]
        tax_embed = torch.stack(tax_embeds, dim=1).mean(dim=1)
        fused = torch.cat([seq_features, drug_embed, tax_embed], dim=1)
        return self.classifier(fused)


MODEL_REGISTRY = {
    "sequence_only": SequenceOnlyModel,
    "metadata": MetadataEnhancedModel,
}


def build_model(model_type: str, metadata: "MetadataIndex", config: ModelConfig) -> nn.Module:
    return MODEL_REGISTRY[model_type](metadata, config)
