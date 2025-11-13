# ARG_DL 再現実装

森平健太郎「機械学習を用いた抗生物質耐性遺伝子の耐性メカニズム分類」(2024年度学士論文) の実験を手元で再現するための実装です。`protein_bert/` ディレクトリに格納された [nadavbra/protein_bert](https://github.com/nadavbra/protein_bert) をそのまま呼び出し、GitHub 版 ProteinBERT の 512 次元グローバル表現を TensorFlow で生成しつつ、PyTorch 側で (1) 配列のみ、(2) 配列 + メタデータ の 2 モデルを学習します。

## フォルダ / ファイル構成
- `data/` : 既存の HMD-ARG / LHD データ (提供済み)
- `train.py` : 学習・検証・評価を一括実行する CLI スクリプト
- `test.py` : 保存済みチェックポイントを読み込み、任意の CSV に対して評価
- `models.py` : ProteinBERT 埋め込み + 2 モデル (sequence_only / metadata) の実装
- `data_utils.py` : データ読み込み、タクソノミー/薬剤エンコーディング、バッチ整形 (シーケンスは生で渡し、TensorFlow 側でトークナイズ)
- `utils.py` : 学習ループ、メトリクス計算、AMP 対応ユーティリティ
- `precompute_embeddings.py` : ProteinBERT 埋め込みを事前計算し、CSV に `embedding` 列を追加するスクリプト
- `requirements.txt` : 必要なパッケージ一覧

## セットアップ手順
```bash
git submodule update --init --recursive        # protein_bert/shared_utils を取得
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
初回実行時に ProteinBERT の学習済み重み (~200MB) を `proteinbert_cache/` に自動ダウンロードします。TensorFlow・PyTorch を併用するため GPU 実行時はメモリに余裕を持ってください。

## ProteinBERT 埋め込みの事前計算（推奨）
TensorFlow を学習ジョブ中に動かさないようにするため、あらかじめ CSV に 512 次元の埋め込みを追記しておくことを推奨します。

```bash
python precompute_embeddings.py \
  --input_csv data/HMDARG-DB/fold_5.train.csv

python precompute_embeddings.py \
  --input_csv data/HMDARG-DB/fold_5.test.csv
```

`precompute_embeddings.py` は `embedding` 列に「スペース区切りの 512 個の浮動小数」を書き込みます。学習／評価時にこの列が存在すると、TensorFlow を呼び出さずに PyTorch 側でそのまま使います。追加で CSV を生成したい場合は `--output_csv` を指定してください。

（どうしてもオンザフライで ProteinBERT を走らせたい場合は `train.py --tf_gpu` を指定すれば TensorFlow が GPU を利用します。デフォルトでは CPU のみを使用します。）

## 学習の実行例
### 1. ランダム分割 (fold_5)
```bash
python train.py \
  --model_type metadata \
  --data_split random \
  --batch_size 4 \
  --epochs 5 \
  --learning_rate 3e-5 \
  --output_dir outputs/random_metadata
```

### 2. 低相同性データ (例: LHD0.4)
```bash
python train.py \
  --model_type metadata \
  --data_split lhd0.4 \
  --batch_size 4 \
  --epochs 5 \
  --learning_rate 3e-5 \
  --output_dir outputs/lhd04_metadata
```

### 3. Sequence-only (配列のみ)
```bash
python train.py \
  --model_type sequence_only \
  --data_split random \
  --batch_size 6 \
  --epochs 5 \
  --output_dir outputs/baseline
```

### 4. Metadata モデル (配列 + 薬剤 + 生物種)
```bash
python train.py \
  --model_type metadata \
  --data_split random \
  --batch_size 4 \
  --epochs 5 \
  --output_dir outputs/metadata
```

#### 主なオプション
- `--data_split` : `random`, `lhd0.4`, `lhd0.6`, `lhd0.8`
- `--seq_len` : ProteinBERT に渡す最大長 (default 2048。<START>/<END>を含むため実際の配列は 2046 文字まで)
- `--use_amp` : GPU + AMP (mixed precision) を有効化
- `--val_ratio` : 追加で validation を作成したい場合のみ指定 (デフォルト 0.0 で train/test のみ)

`outputs/` 以下に各エポックのチェックポイント (`*.pt`) とログが保存されます。ベストモデルは検証 Macro-F1 が最大となったエポックで上書きされます。

## 評価 (再計算)
保存済みモデルをテスト CSV に適用するには `test.py` を使用します。
```bash
python test.py \
  --checkpoint outputs/lhd04_metadata/metadata_epoch5.pt \
  --data_split lhd0.4 \
  --subset test \
  --batch_size 8
```
任意の CSV を直接与える場合は `--csv_path path/to/file.csv` を指定してください。データは `train.py` と同じフォーマット (ID, target, mechanism, species, sequence など) を想定しています。

## 実装メモ
- **ProteinBERT 部分**: `proteinbert_adapter.py` が GitHub 版 ProteinBERT (TensorFlow/Keras) を直接呼び出し、最終グローバル層 (`global-merge2-norm-block6`) の 512 次元表現を抽出します。
- **Sequence-only モデル**: ProteinBERT の 512 次元表現のみを入力し、MLP (既定 512→256) で 7 クラス分類。
- **Metadata モデル**: 512 次元 + 薬剤 (128 次元) + 生物種 (128 次元平均) = 768 次元を MLP に入力します。
- **クラス重み**: ラベル順 (`alteration`, `replacement`, `protection`, `inactivation`, `efflux`, `others`, `negative`) に対し `[10.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0]` を固定で適用。
- **評価指標**: メカニズム 7 クラスの Macro-F1 + 各クラス F1、ARG/非ARG の 2 値 F1 を算出。
- **データ加工**: 種情報はドメイン〜種 (7 階層) を整数 ID 化。未知カテゴリは `<unk-*>`、薬剤は `<unk-drug>` を用意。

## 参考
- 論文: FINAL_thesis_morihira.pdf (リポジトリ同梱)
- 使用データ: HMD-ARG / LHD (付属 CSV)

不明点や追加の再現実験が必要な場合は `train.py --help` を参照し、ハイパーパラメータを調整してください。
