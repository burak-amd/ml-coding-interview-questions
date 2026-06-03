#!/usr/bin/env python3
"""Rebuild QUESTIONS array: merge current HTML + missing basics from full history."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "ml-coding-interview-questions-share.html"

ORDER = [
    "Softmax + cross-entropy",
    "Stratified train/val split",
    "Macro precision / recall / F1",
    "Mini-batch iterator",
    "K-means (Lloyd's algorithm)",
    "Logistic regression (batch GD)",
    "Early stopping tracker",
    "Confusion matrix + macro-F1",
    "PCA via SVD",
    "Batch normalization (forward)",
    "Layer normalization",
    "Label-smoothing cross-entropy",
    "Pairwise AUC-ROC",
    "Decision tree split (Gini)",
    "Cosine LR warmup schedule",
    "Causal attention mask",
    "Scaled dot-product attention",
    "Rotary position encoding",
    "Multi-head self-attention",
    "Grouped-query attention",
    "Incremental KV cache",
    "Streaming softmax for attention",
    "Block-diagonal causal mask",
    "Top-k expert routing",
    "InfoNCE contrastive loss",
    "Gradient accumulation wrapper",
    "Nucleus (top-p) sampling",
    "Temperature + top-k sampling",
    "Masked supervised loss",
    "Low-rank adapter layer",
    "KL penalty vs reference policy",
    "Pairwise ranking loss",
    "GAE (generalized advantage estimation)",
    "REINFORCE with baseline",
    "Clipped policy surrogate loss",
    "GRPO objective",
    "DPO objective",
    "Running reward normalization",
]

NEW = [
    {
        "title": "Mini-batch iterator",
        "difficulty": "easy", "era": "classic",
        "tags": ["classic", "training"],
        "time": "10 min", "libs": "NumPy",
        "topic": "Training infrastructure",
        "spec": """<p><code>batch_iterator(X, y, batch_size, shuffle=True, seed=0)</code> — yields <code>(X_batch, y_batch)</code>.</p>
<ul>
<li>Cover all samples once per generator call; last batch may be smaller.</li>
<li>Deterministic shuffle when <code>shuffle=True</code> and <code>seed</code> fixed.</li>
</ul>""",
        "answer": """import numpy as np

def batch_iterator(X, y, batch_size, shuffle=True, seed=0):
    X = np.asarray(X)
    y = np.asarray(y)
    idx = np.arange(len(X))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(X), batch_size):
        b = idx[start : start + batch_size]
        yield X[b], y[b]""",
    },
    {
        "title": "Early stopping tracker",
        "difficulty": "easy", "era": "classic",
        "tags": ["classic", "training"],
        "time": "10 min", "libs": "stdlib",
        "topic": "Training loops",
        "spec": """<p>Class <code>EarlyStopping(patience=5, min_delta=0.0)</code> with <code>step(metric) -> bool</code>.</p>
<ul>
<li>Lower metric is better. Stop after <code>patience</code> steps without improvement ≥ <code>min_delta</code>.</li>
<li>Track <code>best</code> metric seen.</li>
</ul>""",
        "answer": """class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, metric):
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience""",
    },
    {
        "title": "Confusion matrix + macro-F1",
        "difficulty": "medium", "era": "classic",
        "tags": ["classic"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Evaluation",
        "spec": """<p>Implement <code>confusion_matrix(y_true, y_pred, num_classes=None)</code> and <code>macro_f1_from_confusion(cm)</code>.</p>
<ul>
<li>Matrix entry <code>[i,j]</code> = count(true=i, pred=j).</li>
<li>Macro-F1 averaged over classes from the matrix alone.</li>
</ul>""",
        "answer": """import numpy as np

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
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))""",
    },
    {
        "title": "PCA via SVD",
        "difficulty": "medium", "era": "classic",
        "tags": ["classic"],
        "time": "15–20 min", "libs": "NumPy",
        "topic": "Dimensionality reduction",
        "spec": """<p><code>pca(X, n_components)</code> → <code>(X_proj, components)</code>.</p>
<ul>
<li>Center columns of <code>X</code>; use SVD (not explicit covariance matrix).</li>
<li><code>components</code> shape <code>(n_components, d)</code>.</li>
</ul>""",
        "answer": """import numpy as np

def pca(X, n_components):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    components = vt[:n_components]
    return Xc @ components.T, components""",
    },
    {
        "title": "Batch normalization (forward)",
        "difficulty": "medium", "era": "classic",
        "tags": ["classic", "training", "transformer"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Normalization layers",
        "spec": """<p><code>batch_norm_forward(x, gamma, beta, eps=1e-5)</code> → <code>(y, cache)</code>.</p>
<ul>
<li><code>x</code> shape <code>(N, D)</code>; normalize each feature across batch.</li>
<li>Affine: <code>y = gamma * x_hat + beta</code>. Cache mean, var, x_hat for backward discussion.</li>
</ul>""",
        "answer": """import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    cache = {"mean": mean, "var": var, "x_hat": x_hat, "gamma": gamma, "eps": eps}
    return y, cache""",
    },
    {
        "title": "Layer normalization",
        "difficulty": "medium", "era": "classic",
        "tags": ["classic", "transformer"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Normalization layers",
        "spec": """<p><code>layer_norm(x, gamma, beta, eps=1e-5)</code> — normalize over last dim of <code>x</code>.</p>
<ul><li><code>gamma</code>, <code>beta</code> shape <code>(d,)</code>.</li></ul>""",
        "answer": """import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta""",
    },
    {
        "title": "Pairwise AUC-ROC",
        "difficulty": "hard", "era": "classic",
        "tags": ["classic"],
        "time": "20 min", "libs": "NumPy",
        "topic": "Ranking metrics",
        "spec": """<p><code>roc_auc_score(y_true, y_score)</code> via pairwise concordance.</p>
<ul>
<li><code>y_true</code> in {0,1}; higher score = more likely positive.</li>
<li>Average ties with factor 0.5. Raise if only one class present.</li>
</ul>""",
        "answer": """import numpy as np

def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires both classes")
    auc_num = 0.0
    for p in pos:
        auc_num += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(auc_num / (len(pos) * len(neg)))""",
    },
    {
        "title": "Decision tree split (Gini)",
        "difficulty": "hard", "era": "classic",
        "tags": ["classic"],
        "time": "25 min", "libs": "NumPy",
        "topic": "Tree models",
        "spec": """<p><code>best_gini_split(X, y, feature_idx)</code> → <code>(threshold, gini_gain)</code> for binary labels.</p>
<ul>
<li>Try midpoints between sorted unique feature values.</li>
<li>Return best gain; <code>(nan, 0)</code> if no valid split.</li>
</ul>""",
        "answer": """import numpy as np

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
    for thr in (values[:-1] + values[1:]) / 2.0:
        left = y[X[:, feature_idx] <= thr]
        right = y[X[:, feature_idx] > thr]
        if len(left) == 0 or len(right) == 0:
            continue
        weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
        gain = parent - weighted
        if gain > best_gain:
            best_gain, best_thr = gain, thr
    return best_thr, best_gain""",
    },
    {
        "title": "Causal attention mask",
        "difficulty": "easy", "era": "modern",
        "tags": ["transformer", "llm"],
        "time": "8 min", "libs": "NumPy",
        "topic": "Autoregressive models",
        "spec": """<p><code>causal_mask(seq_len)</code> → bool <code>(seq_len, seq_len)</code>; allow attend to j iff <code>j &lt;= i</code>.</p>
<p><code>merge_masks(a, b)</code> — element-wise AND.</p>""",
        "answer": """import numpy as np

def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

def merge_masks(a, b):
    return np.asarray(a) & np.asarray(b)""",
    },
    {
        "title": "Scaled dot-product attention",
        "difficulty": "medium", "era": "modern",
        "tags": ["transformer"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Attention core",
        "spec": """<p><code>scaled_dot_product_attention(Q, K, V, mask=None)</code> → <code>(output, weights)</code>.</p>
<ul>
<li>Scale by <code>sqrt(d_k)</code>; bool mask True = keep.</li>
<li>Stable softmax on key axis.</li>
</ul>""",
        "answer": """import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V, w""",
    },
    {
        "title": "Masked supervised loss",
        "difficulty": "medium", "era": "modern",
        "tags": ["tuning", "training", "llm"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Fine-tuning with selective labels",
        "spec": """<p><code>masked_nll_loss(log_probs, loss_mask)</code> — mean NLL on supervised tokens only.</p>
<ul><li>Normalize by count of masked positions; error if none.</li></ul>""",
        "answer": """import numpy as np

def masked_nll_loss(log_probs, loss_mask):
    log_probs = np.asarray(log_probs)
    loss_mask = np.asarray(loss_mask, dtype=bool)
    if loss_mask.sum() == 0:
        raise ValueError("No supervised tokens")
    return float(-log_probs[loss_mask].mean())""",
    },
    {
        "title": "Low-rank adapter layer",
        "difficulty": "medium", "era": "modern",
        "tags": ["tuning", "training", "llm"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Parameter-efficient fine-tuning",
        "spec": """<p><code>lora_linear(x, W, A, B, alpha=16, rank=8)</code> — base plus scaled low-rank delta.</p>""",
        "answer": """import numpy as np

def lora_linear(x, W, A, B, alpha=16, rank=8):
    base = x @ W
    delta = (x @ A @ B) * (alpha / rank)
    return base + delta""",
    },
    {
        "title": "Pairwise ranking loss",
        "difficulty": "medium", "era": "modern",
        "tags": ["tuning", "rlhf"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Preference / reward modeling",
        "spec": """<p><code>pairwise_ranking_loss(score_preferred, score_other)</code> = <code>-mean(log σ(s_pref - s_other))</code>.</p>""",
        "answer": """import numpy as np

def pairwise_ranking_loss(score_preferred, score_other):
    diff = np.asarray(score_preferred) - np.asarray(score_other)
    return float(-np.log(1.0 / (1.0 + np.exp(-diff))).mean())""",
    },
    {
        "title": "REINFORCE with baseline",
        "difficulty": "medium", "era": "modern",
        "tags": ["rl"],
        "time": "15 min", "libs": "NumPy",
        "topic": "Policy gradients",
        "spec": """<p><code>reinforce_loss(log_probs, rewards, baseline=None)</code> = <code>-mean(log_prob * (reward - baseline))</code>.</p>
<ul><li>Default baseline: batch mean reward.</li></ul>""",
        "answer": """import numpy as np

def reinforce_loss(log_probs, rewards, baseline=None):
    log_probs = np.asarray(log_probs)
    rewards = np.asarray(rewards)
    baseline = rewards.mean() if baseline is None else baseline
    advantage = rewards - baseline
    return float(-(log_probs * advantage).mean())""",
    },
    {
        "title": "Clipped policy surrogate loss",
        "difficulty": "hard", "era": "modern",
        "tags": ["rl", "rlhf"],
        "time": "20 min", "libs": "NumPy",
        "topic": "PPO-style optimization",
        "spec": """<p><code>clipped_surrogate_loss(logp_new, logp_old, advantages, clip_eps=0.2)</code>.</p>
<ul><li><code>L = -mean(min(r·A, clip(r)·A))</code> with <code>r = exp(logp_new - logp_old)</code>.</li></ul>""",
        "answer": """import numpy as np

def clipped_surrogate_loss(logp_new, logp_old, advantages, clip_eps=0.2):
    ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    adv = np.asarray(advantages)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return float(-np.minimum(unclipped, clipped).mean())""",
    },
]


def parse_existing(text):
    blocks = re.findall(r'\{\s*id: \d+,.*?answer: `[^`]*`\s*\}', text, re.S)
    by_title = {}
    for b in blocks:
        title = re.search(r'title: "([^"]+)"', b).group(1)
        by_title[title] = b
    return by_title


def block_from_dict(q, qid):
    tags = ", ".join(f'"{t}"' for t in q["tags"])
    return f"""      {{
        id: {qid}, title: "{q["title"]}", difficulty: "{q["difficulty"]}", era: "{q["era"]}",
        tags: [{tags}],
        time: "{q["time"]}", libs: "{q["libs"]}",
        topic: "{q["topic"]}",
        spec: `{q["spec"]}`,
        answer: `{q["answer"]}`
      }}"""


def block_from_existing(b, qid):
    b = re.sub(r'id: \d+,', f'id: {qid},', b, count=1)
    return "      " + b.strip()


def main():
    text = HTML.read_text()
    existing = parse_existing(text)
    for q in NEW:
        existing[q["title"]] = None  # mark as new dict

    merged = {}
    for title in ORDER:
        if title not in existing and title not in {q["title"] for q in NEW}:
            raise SystemExit(f"Missing question: {title}")
        new_q = next((q for q in NEW if q["title"] == title), None)
        if new_q:
            merged[title] = ("new", new_q)
        else:
            merged[title] = ("old", existing[title])

    blocks = []
    for i, title in enumerate(ORDER, 1):
        kind, data = merged[title]
        if kind == "new":
            blocks.append(block_from_dict(data, i))
        else:
            blocks.append(block_from_existing(data, i))

    new_arr = "const QUESTIONS = [\n" + ",\n".join(blocks) + "\n    ];"
    start = text.index("const QUESTIONS = [")
    end = text.index("\n    ];", start)
    text = text[:start] + new_arr + text[end + len("\n    ];") :]

    n = len(ORDER)
    tuning = sum(1 for t in ORDER if any(
        q["title"] == t for q in NEW if "tuning" in q["tags"]
    )) + 3  # kl, block mask from old
    # recount from merged tags roughly
    rl = 8

    text = re.sub(
        r'(<div class="stat"><strong>)\d+(</strong><span>Questions</span></div>)',
        rf"\g<1>{n}\2",
        text,
    )
    text = re.sub(
        r'(<div class="stat"><strong>)\d+(</strong><span>Fine-tuning</span></div>)',
        r"\g<1>5\2",
        text,
    )
    text = re.sub(
        r'(<div class="stat"><strong>)\d+(</strong><span>RL & alignment</span></div>)',
        rf"\g<1>{rl}\2",
        text,
    )
    text = re.sub(r"\d+ Python coding interview questions", f"{n} Python coding interview questions", text)
    text = re.sub(r"\d+ Python MSA interview questions", f"{n} Python MSA interview questions", text)

    bundles = """    <section class="bundles">
      <h2>Suggested interview bundles</h2>
      <div class="bundle-grid">
        <div class="bundle-card">
          <h3>ML foundations (45 min)</h3>
          <p>#1 Softmax + CE → #4 Mini-batch → #5 K-means → #8 Confusion matrix</p>
        </div>
        <div class="bundle-card">
          <h3>Classic depth (60 min)</h3>
          <p>#6 Logistic regression → #9 PCA → #10 Batch norm → #11 Layer norm → #13 AUC</p>
        </div>
        <div class="bundle-card">
          <h3>Transformer stack (60 min)</h3>
          <p>#16 Causal mask → #17 Scaled attention → #18 Rotary → #19 MHA → #21 KV cache</p>
        </div>
        <div class="bundle-card">
          <h3>Fine-tuning (45 min)</h3>
          <p>#29 Masked loss → #30 Low-rank adapter → #31 KL penalty → #15 Cosine LR</p>
        </div>
        <div class="bundle-card">
          <h3>RL & alignment (60 min)</h3>
          <p>#32 Pairwise ranking → #36 DPO → #35 GRPO → #34 PPO clip → #33 GAE</p>
        </div>
        <div class="bundle-card">
          <h3>Hard signal (60 min)</h3>
          <p>#14 Decision tree → #22 Streaming softmax → #24 Expert routing → #13 AUC</p>
        </div>
      </div>
    </section>"""
    text = re.sub(r'<section class="bundles">.*?</section>', bundles, text, flags=re.S)

    HTML.write_text(text)
    print(f"Wrote {n} questions to {HTML}")


if __name__ == "__main__":
    main()
