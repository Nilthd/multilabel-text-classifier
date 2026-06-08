# Multi-Label Text Classifier using PyTorch

A PyTorch implementation of multi-label text classification using a pretrained encoder (BERT) and a custom classification head. Built as a solution to an ML engineering interview problem.

**Real-world use case:** Classifying engineering design requirement sentences into one or more verification categories simultaneously.

> *"The system shall respond within 2ms under full load."*
> → ✓ `performance_test`  ✓ `timing_requirement`  ✗ `safety_test`

## What makes this multi-label?

One sentence can belong to **multiple categories at once.**

| | Multi-class (softmax) | Multi-label (sigmoid) |
|---|---|---|
| Output | ONE winner | ANY number of winners |
| Scores sum to | 1.0 | Independent |
| Loss function | CrossEntropyLoss | BCEWithLogitsLoss |
| Example | Cat vs Dog | Topic tagging |

## Architecture

```
Input sentence
      ↓
Pretrained encoder (BERT)     ← reads sentence, produces a 768-dim fingerprint
      ↓
CLS token  (batch, 768)       ← BERT's summary of the whole sentence
      ↓
Linear(768 → 384) + ReLU      ← compress and find patterns
      ↓
Linear(384 → num_classes)     ← one raw score per category
      ↓
sigmoid()                     ← convert to probability [0, 1]
      ↓
threshold filter (≥ 0.5)      ← only predict confident classes
```

## Project Structure

```
multilabel-text-classifier/
├── multilabel_classifier.py  
├── requirements.txt
└── README.md
```

## Components

### `MultiLabelClassificationHead`
A 2-layer MLP that sits on top of BERT:
- Takes BERT's CLS token as input
- Outputs raw logits — one per class
- Sigmoid + threshold applied at inference time

### `train_one_epoch`
One full training pass:
- Uses `BCEWithLogitsLoss` (sigmoid + binary cross-entropy, numerically stable)
- Clears gradients → forward pass → compute loss → backprop → update weights

### `predict_with_threshold`
Inference with confidence filtering:
- Applies sigmoid to get probabilities
- Returns only classes where `prob >= confidence_threshold`
- Higher threshold = more precise, lower threshold = more recall

## How to Run

```bash
pip install -r requirements.txt
python multilabel_classifier.py
```

## Key Design Decisions

**Why BCEWithLogitsLoss?**
Each class is treated as an independent binary question. "With Logits" means sigmoid is applied internally, more numerically stable than applying it manually before the loss.

**Why the CLS token?**
BERT trains its position-0 output to summarize the entire input sequence, making it the natural choice for sentence-level classification.

**Why a threshold instead of argmax?**
Argmax always picks exactly one winner. A threshold lets multiple classes win when the model is confident about several, which is the whole point of multi-label classification.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+

## Author

Niloofar Tavahoodi — M.A.Sc. Candidate, Electrical & Computer Engineering, University of Victoria
