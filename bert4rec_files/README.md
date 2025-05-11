# 📊 Sequential Recommendation with BERT4Rec

This report explores the **BERT4Rec** model—a transformer-based architecture—for multi-item **sequential recommendation** using the **MovieLens 1M** dataset.

---

## 🔧 Data Preprocessing

- Ratings ≥ 4 are converted to **binary implicit feedback**.
- Chronological sequences are generated per user.
- Users with <5 interactions are filtered out (final count: **5,955 users**).
- Sequences truncated to a **maximum length of 200**.
- Splits:
  - **70%** training
  - **15%** validation
  - **15%** test
- The **last 5 interactions** of each user are reserved as targets.

---

## 🧠 Model: BERT4Rec

BERT4Rec uses a **bidirectional transformer encoder** to model sequences of item indices. Key features:

- Embedding sizes: **64, 128, 256**
- Layers: **2 or 4**
- Attention heads: **2 or 4**
- Masking probability: **0.2 or 0.4**
- Regularization:
  - **Dropout:** 0.2
  - **Weight decay:** 0.01
  - **Early stopping:** based on NDCG@10

---

## ⚙️ Training Setup

- Framework: **PyTorch**
- Optimizer: **AdamW** (`lr=0.0003`)
- Scheduler: **LambdaLR** for learning rate decay
- **Batch size:** 256 (train), 128 (eval)
- **Training duration:** up to 200 epochs (early stopping with patience=10)

---

## 📈 Evaluation

Metrics used:
- **Recall@10**
- **NDCG@10**

| Model                 | Recall@10 | NDCG@10 |
|----------------------|-----------|---------|
| BERT4Rec-L-Multi 5   | **0.6369**   | **0.5712**  |
| BERT4Rec-XL-Multi 5  | 0.6386   | 0.5618  |
| BERT4Rec-M-Multi 5   | 0.5978   | 0.5378  |
| BERT4Rec-S-Multi 5   | 0.4102   | 0.3531  |

> Larger models generally improved pattern capture but introduced risk of overfitting.

---

## ⚠️ Challenges

- Data sparsity after filtering users
- Sensitive hyperparameter tuning
- Overfitting in deeper models
- Inconsistent evaluation setup may skew metrics

---

## 🌟 Recommendations for Improvement

- Advanced regularization (e.g., **layer-wise dropout**)
- **Position-aware masking** for better context
- Incorporate **side information** (e.g., genres, timestamps)
- Use of **contrastive learning** and **knowledge graphs**
- Aligning validation/test evaluation methods

---

## 🚀 Run It Yourself

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Model:

```bash
python bert_multi.py
```
