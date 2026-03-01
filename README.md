# 🛡️ Multimodal Network Anomaly Detection

> **Late-Fusion deep learning model fusing structured network telemetry with LLM-encoded security logs for real-time threat classification.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project implements a **multimodal anomaly detection system** that simultaneously processes:

- **Structured network telemetry** (UNSW-NB15 CSV features) through a 3-layer MLP
- **Unstructured security log text** (HDFS/BGL-style log strings) through `distilbert-base-uncased`

Both branches are fused via late concatenation into a unified threat classifier capable of distinguishing **8 attack categories** with high precision and low false-positive rates.

---

## 🏗️ Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐
│  Network Telemetry  │     │     System Log Text       │
│  (UNSW-NB15 CSV)    │     │  (HDFS/BGL log strings)   │
└────────┬────────────┘     └────────────┬─────────────┘
         │                               │
   ┌─────▼──────┐               ┌────────▼────────┐
   │  MLP (3L)  │               │   DistilBERT    │
   │ ReLU+Drop  │               │  [CLS] → 768d   │
   └─────┬──────┘               └────────┬────────┘
         │                               │
         └──────────────┬────────────────┘
                        │  CONCAT
                  ┌─────▼──────┐
                  │ Classifier  │
                  │  Softmax    │
                  └─────────────┘
```

| Component | Details |
|---|---|
| **Numeric branch** | 3-layer MLP (128 → 64 hidden), ReLU + Dropout(0.2) |
| **Text branch** | `distilbert-base-uncased`, frozen for first 5 epochs then fine-tuned |
| **Fusion** | Concatenation of MLP output + BERT `[CLS]` (768-d) |
| **Classifier** | Dense(256) → ReLU → Dropout → Dense(n_classes) |
| **Loss** | Weighted CrossEntropy (inverse class frequency) |
| **Optimizer** | AdamW, lr=2e-5 (BERT branch ×0.1 after unfreeze) |
| **Regularisation** | Gradient clipping (norm=1.0), Cosine LR schedule |

---

## 🗂️ Dataset

| Dataset | Description |
|---|---|
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Network traffic features (41 attributes) |
| HDFS/BGL-style logs | Synthetic log strings generated per attack class |

**Attack Classes:** `Normal`, `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Reconnaissance`, `Worms`

> A high-fidelity **synthetic dataset** is generated automatically when `USE_SYNTHETIC = True`, allowing the notebook to run end-to-end without manual data uploads.

---

## 📁 Project Structure

```
multimodal-anomaly-detection/
│
├── multimodal_anomaly_detection.ipynb   # Main notebook (all phases)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── outputs/                             # Generated after training
│   ├── multimodal_anomaly_detector.pt   # Model weights
│   ├── preprocessors.pkl                # Scaler, OHE, LabelEncoder
│   ├── model_config.json                # Hyperparameters & class names
│   └── training_curves.png             # Loss/F1 plots
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-anomaly-detection.git
cd multimodal-anomaly-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

#### Option A — Google Colab (Recommended)

1. Open the notebook in **Google Colab**
2. Set Runtime → Change runtime type → **T4 GPU**
3. Run all cells (`Runtime → Run all`)
4. The notebook runs out-of-the-box with synthetic data (`USE_SYNTHETIC = True`)

#### Option B — Real UNSW-NB15 Data

1. Download the dataset from the [UNSW-NB15 project page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
2. Upload `UNSW_NB15_training-set.csv` to `/content/`
3. Set `USE_SYNTHETIC = False` in the data acquisition cell
4. Run all cells

#### Option C — Local Jupyter

```bash
jupyter notebook multimodal_anomaly_detection.ipynb
```

> **Note:** GPU is strongly recommended. CPU-only training will be significantly slower due to DistilBERT inference.

---

## 📊 Target Metrics

| Metric | Description |
|---|---|
| **F1-Score (macro)** | Primary performance metric across all 8 classes |
| **PR-AUC** | Precision-Recall Area Under Curve per class |
| **False Positive Rate** | FP / (FP + TN) per attack category |

---

## 🔍 Explainability

SHAP values are computed for the **tabular (MLP) branch**, providing per-feature attribution scores across all attack classes. The text branch uses zeroed embeddings during SHAP analysis to isolate tabular contributions.

---

## 💾 Saved Artifacts

After training, the following files are saved:

- `multimodal_anomaly_detector.pt` — PyTorch model state dict
- `preprocessors.pkl` — Fitted `StandardScaler`, `OneHotEncoder`, `LabelEncoder`
- `model_config.json` — Full configuration including class names, feature columns, and best validation F1

---

## 🔧 Configuration

Key hyperparameters (editable at the top of the notebook):

```python
USE_SYNTHETIC = True       # Use synthetic data (False for real UNSW-NB15)
SEED          = 42         # Random seed for reproducibility
MAX_LEN       = 128        # DistilBERT max token length
BERT_NAME     = 'distilbert-base-uncased'
```

---

## 📋 Requirements

- Python 3.9+
- CUDA-capable GPU recommended (NVIDIA T4 or better)
- ~4 GB GPU VRAM minimum

See [`requirements.txt`](requirements.txt) for the full dependency list.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) — Moustafa & Slay, UNSW Canberra
- [HuggingFace Transformers](https://huggingface.co/transformers/) — DistilBERT implementation
- [SHAP](https://github.com/slundberg/shap) — Model explainability
