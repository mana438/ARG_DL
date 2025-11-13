from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List

import numpy as np
import torch

# TensorFlow/Keras を使うが、遅延読み込みしないと初期化時に GPU メモリを専有するため
# import はクラス初期化のタイミングで行うようにした。


class ProteinBertEmbedder:
    """GitHub版 ProteinBERT から配列全体の 512 次元グローバル表現を抽出するラッパー。"""

    def __init__(
        self,
        seq_len: int = 2048,
        cache_dir: Path | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.cache_dir = cache_dir or (Path(__file__).resolve().parent / "proteinbert_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._tf_predict_lock = threading.Lock()
        self._load_proteinbert()

    def _load_proteinbert(self) -> None:
        repo_path = Path(__file__).resolve().parent / "protein_bert"
        if repo_path.exists():
            sys.path.insert(0, str(repo_path))

        from proteinbert import load_pretrained_model  # type: ignore
        from tensorflow import keras  # type: ignore

        cache_dir = str(self.cache_dir.resolve())
        pretrained_model_generator, input_encoder = load_pretrained_model(
            local_model_dump_dir=cache_dir,
            download_model_dump_if_not_exists=True,
            validate_downloading=False,
        )
        self.input_encoder = input_encoder
        base_model = pretrained_model_generator.create_model(self.seq_len)
        base_model.trainable = False
        self.embedding_model = self._build_global_model(base_model, keras)

    def _build_global_model(self, base_model, keras_module):
        """global-merge2-norm-block6 (512 dim) を出力するサブモデルを生成。"""
        global_layers = [layer for layer in base_model.layers if layer.name.startswith("global-merge2-norm")]
        if not global_layers:
            raise RuntimeError("global-merge2-norm-* layers not found in ProteinBERT.")
        final_global_layer = global_layers[-1]
        self.global_dim = int(final_global_layer.output.shape[-1])
        return keras_module.Model(inputs=base_model.inputs, outputs=final_global_layer.output)

    def _normalize_sequence(self, sequence: str) -> str:
        max_core_len = self.seq_len - 2  # <START>,<END> を考慮
        seq = (sequence or "").replace(" ", "").upper()
        return seq[:max_core_len]

    def encode(self, sequences: List[str]) -> torch.Tensor:
        cleaned = [self._normalize_sequence(seq) for seq in sequences]
        encoded_x = self.input_encoder.encode_X(cleaned, self.seq_len)
        batch_size = max(1, len(cleaned))
        with self._tf_predict_lock:
            global_repr = self.embedding_model.predict(encoded_x, batch_size=batch_size, verbose=0)
        return torch.from_numpy(np.asarray(global_repr, dtype=np.float32))
