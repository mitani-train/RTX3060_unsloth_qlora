# Unsloth QLoRA Fine-tuning

[Unsloth](https://github.com/unslothai/unsloth) を使用した QLoRA ファインチューニング環境です。

## 動作環境

| 項目 | 内容 |
|------|------|
| Python | 3.13 |
| パッケージ管理 | uv |
| GPU | NVIDIA GeForce RTX 3060 (VRAM 12.0 GB) |
| CUDA | 8.6 / CUDA Toolkit 13.0 |
| PyTorch | 2.10.0+cu130 |
| Unsloth | 2026.4.2 |
| Transformers | 5.5.0 |
| Xformers | 0.0.35（Flash Attention 2 の代替） |
| Triton | 3.6.0 |

> **Note:** Flash Attention 2 は Windows 環境では利用不可のため、Xformers にフォールバックします。パフォーマンスへの影響はありません。

---

## セットアップ

### 1. 仮想環境の作成

```bash
uv venv --python 3.13
```

### 2. 依存パッケージのインストール

```bash
uv pip install unsloth --torch-backend=auto
```

`--torch-backend=auto` により、環境に応じた PyTorch バックエンド（CUDA 等）が自動選択されます。

### 3. CUDA 動作確認

```python
python
>>> import torch
>>> print(torch.cuda.is_available())
True
>>> print(torch.cuda.get_device_name(0))
NVIDIA GeForce RTX 3060
>>> exit()
```

---

## 使い方

### モデルの読み込み確認

```bash
python test_load.py
```

### チュートリアルの実行

基本的な動作確認として、`tutorial/` 内のスクリプトを `01` ～ `06` の順に実施してください。

| スクリプト | 内容 |
|-----------|------|
| `01_model_load.py` | モデルの読み込み確認 |
| `02_dataset_check.py` | データセットの確認 |
| `03_format_check.py` | フォーマットの確認 |
| `04_lola_check.py` | LoRA 設定の確認 |
| `05_train_check.py` | 学習の動作確認 |
| `06_inference_check.py` | 推論の動作確認 |

```bash
python tutorial/01_model_load.py
python tutorial/02_dataset_check.py
python tutorial/03_format_check.py
python tutorial/04_lola_check.py
python tutorial/05_train_check.py
python tutorial/06_inference_check.py
```

---

## 学習

学習は YAML 形式の設定ファイルを指定して実行します。

```bash
python train.py configs/<CONFIG_NAME>.yaml
```

### 実行例

```bash
python train.py configs/APTO-001.yaml
```

### 設定ファイル

`configs/` ディレクトリに YAML ファイルを配置します。

```yaml
# configs/APTO-001.yaml（例）
QLoRA: True

model:
  name: unsloth/Llama-3.1-8B-Instruct-bnb-4bit
  max_seq_length: 1024

dataset:
  name: APTO-001/japanese-reasoning-dataset-sample
  split: train[:100]

lora:
  r: 8
  alpha: 16
  dropout: 0
 # dropout: 0.05


training:
  batch_size: 1
  grad_accum: 8
  lr: 0.0002
  epochs: 1

output:
  dir: outputs/APTO-001
  ```

### 主なパラメータの説明

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `model.name` | `unsloth/Llama-3.1-8B-Instruct-bnb-4bit` | ベースモデル（4bit量子化済み） |
| `model.max_seq_length` | `1024` | 最大シーケンス長 |
| `dataset.split` | `train[:100]` | 使用するデータ件数（先頭100件） |
| `lora.r` | `8` | LoRA のランク |
| `lora.alpha` | `16` | LoRA のスケーリング係数（通常 r の 2 倍推奨） |
| `lora.dropout` | `0` | LoRA の Dropout 率 |
| `training.batch_size` | `1` | デバイスあたりのバッチサイズ |
| `training.grad_accum` | `8` | 勾配累積ステップ数（実効バッチサイズ = 1×8 = **8**） |
| `training.lr` | `0.0002` | 学習率 |
| `training.epochs` | `1` | エポック数 |

### 学習ログ（実行例）

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 100 | Num Epochs = 1 | Total steps = 13
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 8
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 8 x 1) = 8
 "-____-"     Trainable parameters = 20,971,520 of 8,051,232,768 (0.26% trained)
```

### 学習結果サマリ

| 項目 | 値 |
|------|-----|
| 総ステップ数 | 13 |
| 学習時間 | 170.1 秒（約 2 分 50 秒） |
| サンプル / 秒 | 0.588 |
| ステップ / 秒 | 0.076 |
| 最終 Train Loss | **1.5527** |
| 学習済みパラメータ数 | 20,971,520 / 8,051,232,768（**0.26%**） |

### 出力先

学習結果は以下のディレクトリに保存されます。

```
outputs/
└── <CONFIG_NAME>/
    └── final/        ← 最終モデルの保存先
```

---

## 推論・評価

学習済みモデルを使って推論を実行し、ベースモデルとの比較を行います。

```bash
python compare_inference.py configs/<CONFIG_NAME>.yaml
```

### 実行例

```bash
python .\compare_inference.py .\configs\APTO-001.yaml
```

`compare_inference.py` はベースモデルとファインチューニング済みモデルの出力を並べて比較します。

---

## ディレクトリ構成

```
.
├── configs/               # 学習設定ファイル (YAML)
│   └── APTO001.yaml
├── outputs/               # 学習結果の出力先
│   └── APTO-001/
│       └── final/
├── tutorial/              # チュートリアルスクリプト (01〜06)
│   ├── 01_model_load.py
│   ├── 02_dataset_check.py
│   ├── 03_format_check.py
│   ├── 04_lola_check.py
│   ├── 05_train_check.py
│   └── 06_inference_check.py
├── test_load.py           # モデル読み込み確認スクリプト
├── train.py               # 学習スクリプト
└── compare_inference.py   # 推論比較スクリプト
```

---

## 参考

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [uv - Python package manager](https://github.com/astral-sh/uv)