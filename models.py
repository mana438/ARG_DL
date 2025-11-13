from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence

import torch
import torch.nn as nn

from proteinbert_adapter import ProteinBertEmbedder

if TYPE_CHECKING:
    from data_utils import MetadataIndex


GLOBAL_PROTEINBERT_DIM = 512


@dataclass
class ModelConfig:
    seq_len: int = 2048
    dropout: float = 0.1
    metadata_embedding_dim: int = 128
    hidden_layers: Sequence[int] = (512, 128, 32)
    use_tf_gpu: bool = False


def _resolve_hidden_layers(config: ModelConfig) -> Sequence[int]:
    if len(config.hidden_layers) == 3:
        return (512, 128, 32)
    return config.hidden_layers


class ProteinBertSequenceEncoder(nn.Module):
    """ProteinBERT のグローバル表現を 512 次元に射影して返す共通エンコーダ。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedder: ProteinBertEmbedder | None = None
        self.projection = nn.Linear(GLOBAL_PROTEINBERT_DIM, 512)
        self.dropout = nn.Dropout(config.dropout)

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            self.embedder = ProteinBertEmbedder(
                seq_len=self.config.seq_len,
                use_tf_gpu=self.config.use_tf_gpu,
            )

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        self._ensure_embedder()
        assert self.embedder is not None
        with torch.no_grad():
            seq_features = self.embedder.encode(sequences)
        return self._project(seq_features.to(device))

    def project_precomputed(self, embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self._project(embeddings.to(device))

    def _project(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(tensor))


class SequenceOnlyModel(nn.Module):
    """配列情報のみで予測するベースラインモデル (512次元→MLP)。"""

    def __init__(self, metadata: "MetadataIndex", config: ModelConfig) -> None:
        super().__init__()
        self.encoder = ProteinBertSequenceEncoder(config)
        self.classifier = self._build_classifier(512, metadata.num_labels, config)

    def _build_classifier(self, input_dim: int, num_labels: int, config: ModelConfig) -> nn.Module:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        hidden_dims = _resolve_hidden_layers(config)
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_labels))
        return nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch["labels"].device
        precomputed = batch.get("precomputed_embeddings")
        if precomputed is not None:
            seq_features = self.encoder.project_precomputed(precomputed, device)
        else:
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
        hidden_dims = _resolve_hidden_layers(config)
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_labels))
        return nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch["labels"].device
        precomputed = batch.get("precomputed_embeddings")
        if precomputed is not None:
            seq_features = self.encoder.project_precomputed(precomputed, device)
        else:
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
