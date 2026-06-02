# ML Coding Interview Questions (Python)

**Document type:** Interview question bank  
**Format:** MSA — **M**etadata, **S**pecification, **A**nswer (reference code)  
**Language:** Python 3.10+ (NumPy unless noted; no PyTorch/TensorFlow unless stated)  
**Audience:** ML engineer / applied scientist coding rounds (45–60 min)

---

## How to use this doc

- **Interviewer:** Pick 1 easy + 1 medium (+ optional hard). Ask candidate to implement from spec only; compare against reference after.
- **Candidate:** Cover the spec with tests, then peek at answers only after attempting.
- **Allowed in most rounds:** `numpy`, standard library. **Disallowed unless specified:** sklearn, torch, scipy.

---

## Table of contents

| # | Topic | Difficulty |
|---|-------|------------|
| 1 | Softmax + cross-entropy | Easy |
| 2 | Train/validation split (stratified) | Easy |
| 3 | Precision, recall, F1 | Easy |
| 4 | Mini-batch iterator | Easy |
| 5 | K-means clustering | Medium |
| 6 | Logistic regression (GD) | Medium |
| 7 | Batch normalization (forward) | Medium |
| 8 | Confusion matrix + macro-F1 | Medium |
| 9 | PCA (via SVD) | Medium |
| 10 | Scaled dot-product attention | Medium |
| 11 | Learning-rate warmup + cosine decay | Medium |
| 12 | Early stopping tracker | Easy |
| 13 | Pairwise AUC-ROC | Hard |
| 14 | Decision tree split (Gini) | Hard |
| 15 | Layer normalization | Medium |

---

## 1. Softmax + cross-entropy

### Metadata

| Field | Value |
|-------|-------|
| Topic | Classification losses |
| Difficulty | Easy |
| Time | 10–15 min |
| Libraries | NumPy only |

### Specification

Implement two functions:

1. `softmax(logits: np.ndarray) -> np.ndarray`  
   - Input shape `(N, C)` or `(C,)`.  
   - Output same shape, rows sum to 1 (for 2D).  
   - Must be numerically stable (no overflow in `exp`).

2. `cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float`  
   - `logits`: `(N, C)`, `targets`: `(N,)` integer class indices.  
   - Return **mean** cross-entropy over the batch.  
   - Use log-softmax internally; do not compute `log(softmax)` from unstable softmax.

### Answer

```python
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 1:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n, c = logits.shape
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
    log_probs = shifted - log_sum_exp[:, None]
    nll = -log_probs[np.arange(n), targets]
    return float(nll.mean())
```

---

## 2. Stratified train/validation split

### Metadata

| Field | Value |
|-------|-------|
| Topic | Data splitting |
| Difficulty | Easy |
| Time | 10 min |
| Libraries | NumPy only |

### Specification

Implement:

```python
def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
```

- Return `(X_train, X_val, y_train, y_val)`.
- Preserve approximate class proportions in both splits.
- Every sample appears in exactly one split.
- Shuffle within each class using `seed` (reproducible).

### Answer

```python
import numpy as np


def stratified_split(X, y, val_fraction=0.2, seed=0):
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
```

---

## 3. Precision, recall, F1 (multiclass)

### Metadata

| Field | Value |
|-------|-------|
| Topic | Classification metrics |
| Difficulty | Easy |
| Time | 10–15 min |
| Libraries | NumPy only |

### Specification

Implement:

```python
def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> dict[str, float]:
    ...
```

- Return dict with keys: `precision`, `recall`, `f1` (single floats for the chosen average).
- Support `average="macro"` (unweighted mean over classes present in `y_true`).
- Handle zero division: if denominator is 0, that class contributes 0 to the average.

### Answer

```python
import numpy as np


def classification_report(y_true, y_pred, average="macro"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    precisions, recalls, f1s = [], [], []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    if average == "macro":
        return {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
        }
    raise ValueError(f"Unsupported average: {average}")
```

---

## 4. Mini-batch data iterator

### Metadata

| Field | Value |
|-------|-------|
| Topic | Training infrastructure |
| Difficulty | Easy |
| Time | 10 min |
| Libraries | NumPy only |

### Specification

Implement a generator:

```python
def batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
):
    """Yields (X_batch, y_batch) covering all samples once per call."""
    ...
```

- Last batch may be smaller if `N % batch_size != 0`.
- When `shuffle=True`, order changes each full pass but is deterministic for a given `seed` per epoch if you increment seed externally (document: shuffle once at start of each epoch).

### Answer

```python
import numpy as np


def batch_iterator(X, y, batch_size, shuffle=True, seed=0):
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]
```

---

## 5. K-means clustering

### Metadata

| Field | Value |
|-------|-------|
| Topic | Unsupervised learning |
| Difficulty | Medium |
| Time | 20–25 min |
| Libraries | NumPy only |

### Specification

Implement Lloyd's algorithm:

```python
def kmeans(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids, labels). centroids shape (k, d), labels shape (n,)."""
    ...
```

- Initialize centroids by sampling `k` distinct points from `X`.
- Stop when labels unchanged or `max_iters` reached.
- Use squared Euclidean distance.

### Answer

```python
import numpy as np


def kmeans(X, k, max_iters=100, seed=0):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = X[init_idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
    return centroids, labels
```

---

## 6. Logistic regression (batch gradient descent)

### Metadata

| Field | Value |
|-------|-------|
| Topic | Linear models |
| Difficulty | Medium |
| Time | 20–25 min |
| Libraries | NumPy only |

### Specification

Implement binary logistic regression training:

```python
def train_logistic_regression(
    X: np.ndarray,          # (n, d)
    y: np.ndarray,          # (n,) in {0, 1}
    lr: float = 0.1,
    n_epochs: int = 1000,
    l2: float = 0.0,
) -> np.ndarray:            # weights w shape (d,)
    ...
```

- Model: `P(y=1|x) = sigmoid(w @ x)` (no explicit bias column; assume `X` already includes bias if needed).
- Loss: mean binary cross-entropy + `(l2/2) * ||w||^2`.
- Return learned `w`.

### Answer

```python
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.1, n_epochs=1000, l2=0.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)

    for _ in range(n_epochs):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / n + l2 * w
        w -= lr * grad
    return w
```

---

## 7. Batch normalization (forward pass)

### Metadata

| Field | Value |
|-------|-------|
| Topic | Normalization layers |
| Difficulty | Medium |
| Time | 15–20 min |
| Libraries | NumPy only |

### Specification

Implement forward batch norm for **training mode** on a 2D activation tensor `(N, D)`:

```python
def batch_norm_forward(
    x: np.ndarray,
    gamma: np.ndarray,   # (D,)
    beta: np.ndarray,    # (D,)
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict]:
    ...
```

- Normalize each feature dimension across the batch.
- Apply affine transform: `y = gamma * x_hat + beta`.
- Return `(y, cache)` where `cache` holds values needed for backward (mean, var, x, x_hat, gamma) — backward not required in interview, but cache shows understanding.

### Answer

```python
import numpy as np


def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    cache = {"mean": mean, "var": var, "x": x, "x_hat": x_hat, "gamma": gamma, "eps": eps}
    return y, cache
```

---

## 8. Confusion matrix + macro-F1 from scratch

### Metadata

| Field | Value |
|-------|-------|
| Topic | Evaluation |
| Difficulty | Medium |
| Time | 15 min |
| Libraries | NumPy only |

### Specification

Implement:

```python
def confusion_matrix(y_true, y_pred, num_classes: int | None = None) -> np.ndarray:
    """Return (C, C) matrix where entry [i, j] = count(true=i, pred=j)."""

def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Compute macro-F1 directly from confusion matrix."""
```

### Answer

```python
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_from_confusion(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s))
```

---

## 9. PCA via SVD

### Metadata

| Field | Value |
|-------|-------|
| Topic | Dimensionality reduction |
| Difficulty | Medium |
| Time | 15–20 min |
| Libraries | NumPy only |

### Specification

Implement:

```python
def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X_proj, components).
    X_proj: (n, n_components) projected data
    components: (n_components, d) principal axes (unit vectors)
    """
```

- Center `X` column-wise.
- Use SVD; do not form covariance matrix explicitly (numerical stability + efficiency follow-up question).

### Answer

```python
import numpy as np


def pca(X, n_components):
    X = np.asarray(X, dtype=np.float64)
    X_centered = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    components = vt[:n_components]
    X_proj = X_centered @ components.T
    return X_proj, components
```

---

## 10. Scaled dot-product attention

### Metadata

| Field | Value |
|-------|-------|
| Topic | Transformers |
| Difficulty | Medium |
| Time | 15–20 min |
| Libraries | NumPy only |

### Specification

Implement single-head attention:

```python
def scaled_dot_product_attention(
    Q: np.ndarray,  # (..., seq_q, d_k)
    K: np.ndarray,  # (..., seq_k, d_k)
    V: np.ndarray,  # (..., seq_k, d_v)
    mask: np.ndarray | None = None,  # broadcastable; True = keep, False = mask out
) -> tuple[np.ndarray, np.ndarray]:
    """Return (output, attn_weights)."""
```

- `attn_weights = softmax(Q @ K.T / sqrt(d_k))` along last axis of keys.
- If `mask` provided, set masked positions to `-inf` before softmax.
- Output: `attn_weights @ V`.

### Answer

```python
import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    output = attn_weights @ V
    return output, attn_weights
```

---

## 11. Learning-rate warmup + cosine decay

### Metadata

| Field | Value |
|-------|-------|
| Topic | Optimization schedules |
| Difficulty | Medium |
| Time | 10–15 min |
| Libraries | Standard library + NumPy optional |

### Specification

Implement:

```python
def lr_schedule(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """
    - Linear warmup from 0 -> base_lr over [0, warmup_steps)
    - Cosine decay from base_lr -> 0 over [warmup_steps, total_steps]
    - step >= total_steps returns 0
    """
```

### Answer

```python
import math


def lr_schedule(step, warmup_steps, total_steps, base_lr):
    if step >= total_steps:
        return 0.0
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
```

---

## 12. Early stopping tracker

### Metadata

| Field | Value |
|-------|-------|
| Topic | Training loops |
| Difficulty | Easy |
| Time | 10 min |
| Libraries | Standard library |

### Specification

Implement a small class:

```python
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        ...

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
```

- Lower metric is better (e.g., validation loss).
- Stop when metric has not improved by at least `min_delta` for `patience` consecutive calls.
- Track best metric and restore hint via `best` property.

### Answer

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
```

---

## 13. Pairwise AUC-ROC

### Metadata

| Field | Value |
|-------|-------|
| Topic | Ranking metrics |
| Difficulty | Hard |
| Time | 20–25 min |
| Libraries | NumPy only |

### Specification

Implement AUC for binary labels using the **Mann–Whitney U** / pairwise ranking formula:

```python
def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    ...
```

- `y_true` in `{0, 1}`, `y_score` continuous scores (higher = more likely positive).
- Handle ties with average ranking (midrank).
- Return value in `[0, 1]`.

### Answer

```python
import numpy as np


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires both classes present")

    # Count concordant pairs; average ties
    auc_num = 0.0
    for p in pos:
        auc_num += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(auc_num / (len(pos) * len(neg)))
```

**Follow-up:** Discuss O(n log n) rank-based implementation for large data.

---

## 14. Decision tree split (Gini impurity)

### Metadata

| Field | Value |
|-------|-------|
| Topic | Tree models |
| Difficulty | Hard |
| Time | 25–30 min |
| Libraries | NumPy only |

### Specification

For a **binary classification** dataset, find the best axis-aligned split on one feature:

```python
def best_gini_split(
    X: np.ndarray,  # (n, d) all numeric
    y: np.ndarray,  # (n,) in {0, 1}
    feature_idx: int,
) -> tuple[float, float]:
    """
    Return (threshold, gini_gain).
    Split rule: x[:, feature_idx] <= threshold -> left, else right.
    gini_gain = parent_gini - weighted_child_gini.
    If no valid split, return (np.nan, 0.0).
    """
```

### Answer

```python
import numpy as np


def gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 2 * p * (1 - p)


def best_gini_split(X, y, feature_idx):
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return np.nan, 0.0

    parent = gini(y)
    best_gain, best_thr = 0.0, np.nan
    thresholds = (values[:-1] + values[1:]) / 2.0

    for thr in thresholds:
        left = y[X[:, feature_idx] <= thr]
        right = y[X[:, feature_idx] > thr]
        if len(left) == 0 or len(right) == 0:
            continue
        weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
        gain = parent - weighted
        if gain > best_gain:
            best_gain, best_thr = gain, thr
    return best_thr, best_gain
```

---

## 15. Layer normalization

### Metadata

| Field | Value |
|-------|-------|
| Topic | Normalization layers |
| Difficulty | Medium |
| Time | 15 min |
| Libraries | NumPy only |

### Specification

Implement layer norm over the **last dimension** for arbitrary batch shapes:

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x: (..., d)
    gamma, beta: (d,)
    Normalize each sample vector across d.
    """
```

### Answer

```python
import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta
```

---

## Suggested interview bundles

| Round length | Questions |
|--------------|-----------|
| 30 min | #1 + #4 |
| 45 min | #1 + #6 or #5 |
| 60 min | #10 + #7 or #9 |
| Senior signal | #13 + #14 + discuss complexity / edge cases |

---

## Common follow-up prompts

1. **Complexity:** What is time/memory for your implementation? Can you improve it?
2. **Numerical stability:** Where can this blow up? How did you guard against it?
3. **Vectorization:** Can you remove Python loops?
4. **Tests:** What unit tests would you write? (empty batch, single class, ties in AUC, etc.)
5. **Production:** How would this differ with PyTorch / batched GPU tensors?

---

## Quick self-test harness (optional)

```python
import numpy as np

# Smoke test softmax + CE
logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 0.5]])
targets = np.array([0, 2])
probs = softmax(logits)
assert np.allclose(probs.sum(axis=1), 1.0)
loss = cross_entropy(logits, targets)
assert loss > 0
print("OK", loss)
```

---

## Part 2 — LLM tuning & RL / RLHF

| # | Topic | Difficulty | Tags |
|---|-------|------------|------|
| 23 | SFT loss with response-only mask | Medium | tuning |
| 24 | Bradley–Terry reward model loss | Medium | tuning, RLHF |
| 25 | Token-level KL to reference | Medium | tuning, RLHF, RL |
| 26 | Linear LR scaling for fine-tuning | Easy | tuning |
| 27 | Packed-sequence block attention mask | Hard | tuning |
| 28 | Sequence log-prob (teacher forcing) | Easy | tuning, RL |
| 29 | PPO clipped surrogate | Hard | RL, RLHF |
| 30 | GAE | Hard | RL, RLHF |
| 31 | GRPO group-relative advantages | Medium | RL, RLHF |
| 32 | REINFORCE with baseline | Medium | RL |
| 33 | Running reward normalization | Easy | RL, RLHF |

> **Interactive version:** see `ml-coding-interview-questions.html` for filters (LLM tuning, RL, RLHF) and collapsible answers.

---

## 23. SFT loss with response-only mask

### Metadata

| Field | Value |
|-------|-------|
| Topic | Supervised fine-tuning |
| Difficulty | Medium |
| Time | 15 min |
| Libraries | NumPy |

### Specification

` sft_loss(log_probs, targets, loss_mask) ` — mean NLL only on tokens where `loss_mask == 1` (assistant completion). Normalize by count of supervised tokens.

### Answer

```python
import numpy as np

def sft_loss(log_probs, targets, loss_mask):
    loss_mask = np.asarray(loss_mask, dtype=bool)
    if loss_mask.sum() == 0:
        raise ValueError("No supervised tokens")
    return float(-np.asarray(log_probs)[loss_mask].mean())
```

---

## 24. Bradley–Terry reward model loss

### Metadata

| Field | Value |
|-------|-------|
| Topic | RLHF reward modeling |
| Difficulty | Medium |
| Time | 15 min |

### Specification

Pairwise preference loss: `-mean(log σ(r_chosen - r_rejected))`.

### Answer

```python
import numpy as np

def reward_model_loss(r_chosen, r_rejected):
    diff = np.asarray(r_chosen) - np.asarray(r_rejected)
    return float(-np.log(1.0 / (1.0 + np.exp(-diff))).mean())
```

---

## 25. Token-level KL to reference model

### Metadata

| Field | Value |
|-------|-------|
| Topic | PPO / RLHF KL penalty |
| Difficulty | Medium |
| Time | 18 min |

### Specification

k1 estimator: `KL ≈ exp(log_ref - log_pol) - (log_ref - log_pol) - 1`. Masked mean × `beta`.

### Answer

```python
import numpy as np

def kl_divergence(logp_policy, logp_ref):
    log_ratio = np.asarray(logp_ref) - np.asarray(logp_policy)
    return np.exp(log_ratio) - log_ratio - 1.0

def kl_penalty_loss(logp_policy, logp_ref, mask, beta=0.02):
    mask = np.asarray(mask, dtype=bool)
    return float(beta * kl_divergence(logp_policy, logp_ref)[mask].mean())
```

---

## 29. PPO clipped surrogate objective

### Metadata

| Field | Value |
|-------|-------|
| Topic | RLHF policy optimization |
| Difficulty | Hard |
| Time | 20 min |

### Specification

`L = -mean(min(r·A, clip(r, 1-ε, 1+ε)·A))` where `r = exp(logp_new - logp_old)`.

### Answer

```python
import numpy as np

def ppo_loss(logp_new, logp_old, advantages, clip_eps=0.2):
    ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    adv = np.asarray(advantages)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return float(-np.minimum(unclipped, clipped).mean())
```

---

## 30. GAE (generalized advantage estimation)

### Metadata

| Field | Value |
|-------|-------|
| Topic | PPO advantage estimation |
| Difficulty | Hard |
| Time | 25 min |

### Answer

```python
import numpy as np

def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T)
    gae_acc = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae_acc = delta + gamma * lam * (1 - dones[t]) * gae_acc
        advantages[t] = gae_acc
    return advantages, advantages + np.asarray(values)
```

---

## 31. GRPO group-relative advantages

### Metadata

| Field | Value |
|-------|-------|
| Topic | Group RL for LLMs |
| Difficulty | Medium |
| Time | 18 min |

### Answer

```python
import numpy as np

def grpo_advantages(rewards, group_ids, eps=1e-8):
    rewards = np.asarray(rewards, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    adv = np.zeros_like(rewards)
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        r = rewards[mask]
        adv[mask] = 0.0 if r.size == 1 else (r - r.mean()) / (r.std() + eps)
    return adv
```

---

## Suggested bundles (LLM tuning + RL)

| Round | Questions |
|-------|-----------|
| LLM fine-tuning (60 min) | #23 → #26 → #15 LoRA → #27 |
| RLHF / PPO (60 min) | #24 → #25 → #30 → #29 → #17 DPO |
| Modern RL (45 min) | #32 → #31 → #33 → #29 |

---

*End of document.*
