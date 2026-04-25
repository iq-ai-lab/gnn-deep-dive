# 01. Node Classification

## 🎯 핵심 질문

- Transductive vs inductive node classification 의 정확한 차이는?
- Cora·Citeseer·Pubmed 의 표준 split (Kipf 2017) 이 왜 public split 인가?
- Semi-supervised cross-entropy loss 가 labeled 노드만 사용하는 이유는?
- GCN·GraphSAGE·GAT·GIN 의 Cora accuracy 비교 — 어느 모델이 왜 이기는가?
- Hyperparameter (layer 수, hidden dim, dropout, weight decay) 의 민감도?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

**Node classification** 은 GNN 응용의 가장 기본 task:
- 각 노드에 label 부여 (e.g., Cora: 7 topic classes for papers)
- 일부 labeled, 대부분 unlabeled — semi-supervised
- GNN 이 unlabeled 노드의 label 예측

이는 GNN 연구의 표준 benchmark task. 새 모델 제안 시 첫 검증:
1. **Cora / Citeseer / Pubmed** (Planetoid, Kipf 2017): 작은 citation network
2. **OGB-Arxiv** (Hu 2020): 170k nodes, inductive
3. **OGB-Products** (Hu 2020): 2.4M nodes, large-scale

이 문서는 node classification 의 standard pipeline, 모델별 성능 비교, 실전 tuning 을 정리.

---

## 📐 수학적 선행 조건

- [Ch2-03](../ch2-spectral-gcn/03-gcn-derivation.md), [Ch3-01~04](../ch3-message-passing/): GCN, GraphSAGE, GAT, GIN
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Cross-entropy loss, Adam optimizer

---

## 📖 직관적 이해

### Transductive vs Inductive

**Transductive (Cora 표준)**:
- 전체 graph 를 학습 시 모두 봄
- Labeled + unlabeled nodes 모두 주어짐
- Test 시 같은 graph 의 unlabeled node 분류

**Inductive (OGB-Arxiv)**:
- 학습 시 일부 subgraph 만 봄
- Test 시 **새 노드** 또는 **새 subgraph**
- Model parameter 가 graph-agnostic

### Semi-Supervised Loss

Cross-entropy only on labeled nodes:
$$
\mathcal L = -\sum_{v \in V_L} \sum_c y_{vc} \log \hat y_{vc}
$$

($V_L$: labeled set, $y_{vc}$: one-hot true label)

**Unlabeled nodes** 는 forward pass 에 포함 (message passing 에 participation) but loss 에 기여 X. 이는 **implicit regularization** — unlabeled 노드 가 labeled 의 neighbor 정보로 smoothing.

### Benchmark Datasets

| Dataset | #Nodes | #Edges | #Features | #Classes | Train/Val/Test |
|---------|--------|--------|-----------|----------|----------------|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140/500/1000 |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 | 120/500/1000 |
| Pubmed | 19,717 | 88,648 | 500 | 3 | 60/500/1000 |
| OGB-Arxiv | 169,343 | 1.2M | 128 | 40 | 91k/30k/48k |

(Kipf-Welling 2017 public splits)

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Node Classification Problem

Graph $G = (V, E)$, node feature $X \in \mathbb R^{n \times d}$, labeled subset $V_L \subset V$ with label $y_v \in \{1, \ldots, C\}$ for $v \in V_L$.

**Task**: Predict $\hat y_v$ for $v \in V_U = V \setminus V_L$.

### 정의 1.2 — Transductive Setting

전체 $V$, $E$, $X$ 가 training 시 관측됨. Only labels for $V_L$ 가 hidden.

Test set: $V_U$ 의 일부 fixed subset.

### 정의 1.3 — Inductive Setting

Training: $G_{\text{train}} = (V_{\text{train}}, E_{\text{train}})$ + labels.
Test: new $G_{\text{test}}$ or new nodes $V_{\text{new}} \not\subset V_{\text{train}}$.

Model parameter 가 $G$ 독립 (GraphSAGE 같음).

### 정의 1.4 — Standard Pipeline

```python
# 1. Input: (X, edge_index, labels, train/val/test masks)
# 2. Forward: Z = GNN(X, edge_index)  [n, C]
# 3. Loss: cross_entropy(Z[train_mask], labels[train_mask])
# 4. Optimize: Adam
# 5. Evaluate: accuracy on val_mask (best epoch), then test_mask
```

### 정의 1.5 — Evaluation Metrics

- **Accuracy**: $\frac{1}{|V_{\text{test}}|} \sum \mathbb 1[\hat y_v = y_v]$
- **Macro-F1**: class-imbalanced case
- **Micro-F1 / Weighted-F1**: multi-class

---

## 🔬 정리와 결과

### 정리 1.1 — 표준 GNN 의 Cora 성능

**Empirical** (various papers, averaged):

| Model | Cora | Citeseer | Pubmed |
|-------|------|----------|--------|
| MLP (feature only) | 55.1 | 59.1 | 71.4 |
| Label Propagation | 68.0 | 45.3 | 63.0 |
| **GCN** | 81.5 | 70.3 | 79.0 |
| GraphSAGE-mean | 80.8 | 68.9 | 78.0 |
| GraphSAGE-pool | 81.0 | 69.0 | 77.8 |
| **GAT** | 83.0 | 72.5 | 79.0 |
| **GIN** | 80.0 | 66.1 | 76.5 |
| APPNP (K=10, α=0.1) | 83.3 | 71.8 | 80.1 |
| GCNII (64 layers) | 85.5 | 73.4 | 80.3 |
| Graphormer (Ch7-01) | 85.0+ | — | — |

**관찰**:
- GCN, GAT, APPNP, GCNII 가 top tier (~83-85%)
- GIN 이 Cora 에서 상대적 약함 (Ch3-04 문제 3)
- Label propagation (no learnable) baseline ~68%
- MLP (no graph) baseline ~55% — graph info 가 25%+ 기여

### 정리 1.2 — Hyperparameter Sensitivity

**GCN-Cora 의 표준 hyperparameters** (Kipf 2017):
- Layers: 2
- Hidden dim: 16
- Dropout: 0.5
- Weight decay: $5 \times 10^{-4}$
- Learning rate: 0.01
- Epochs: 200
- Early stopping on val accuracy

**Sensitivity**:
- Layer $\to$ 1: 77% (under-smoothing)
- Layer $\to$ 4: 79% (over-smoothing 시작)
- Hidden dim $\to 64$: 81.6% (marginal 향상)
- Dropout $\to 0$: 78% (overfitting)
- No weight decay: 79%

### 정리 1.3 — Inductive Benchmark

**OGB-Arxiv** (평균 of multiple runs):

| Model | Test Accuracy |
|-------|---------------|
| MLP | 55.5% |
| GraphSAGE | 71.5% |
| GCN | 72.0% |
| GAT | 73.0% |
| GIN | 71.8% |
| APPNP | 72.5% |
| GraphSAGE + PairNorm | 72.2% |
| GAT + JKN | 73.4% |
| Graphormer | 76.0% (recent) |

**관찰**: Inductive setting 에서 GAT / Graphormer 가 더 유리 (attention 의 inductive bias).

### 정리 1.4 — Class Imbalance 의 문제

Cora 의 class distribution 이 skewed. Macro-F1 이 accuracy 보다 더 informative:

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| GCN | 81.5 | 80.3 |
| GAT | 83.0 | 82.1 |

Class imbalance 심한 dataset (OGB-Papers): class-weighted loss 또는 focal loss 필요.

---

## 💻 구현

### 실험 1 — Cora GCN Training

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
except ImportError:
    print('PyG not installed; demo with synthetic data')

class GCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.conv1 = GCNConv(d_in, d_hid)
        self.conv2 = GCNConv(d_hid, d_out)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = 0; best_test_acc = 0
for epoch in range(200):
    model.train(); optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_acc = (out[data.val_mask].argmax(-1) == data.y[data.val_mask]).float().mean().item()
        test_acc = (out[data.test_mask].argmax(-1) == data.y[data.test_mask]).float().mean().item()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

print(f'Cora GCN: val acc {best_val_acc:.2%}, test acc {best_test_acc:.2%}')
```

### 실험 2 — Cora Model Comparison

```python
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

def train_model(model_cls, model_kwargs, epochs=200):
    torch.manual_seed(42)
    model = model_cls(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val = 0; best_test = 0
    for ep in range(epochs):
        model.train(); optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_acc = (out[data.val_mask].argmax(-1) == data.y[data.val_mask]).float().mean().item()
            test_acc = (out[data.test_mask].argmax(-1) == data.y[data.test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc; best_test = test_acc
    return best_test

# 각 모델 정의 & 학습
# (코드 분량으로 인해 개별 class 정의 생략)
```

### 실험 3 — Layer-wise Accuracy

```python
# 깊이별 Cora accuracy
for L in [1, 2, 3, 4, 8]:
    model = DeepGCN(dataset.num_features, 16, dataset.num_classes, num_layers=L)
    # (학습 및 평가) 
    # Expected: L=2 best, L=4+ 저하
    pass
```

### 실험 4 — Dropout / Weight Decay Ablation

```python
configs = [
    {'dropout': 0.0, 'weight_decay': 0},
    {'dropout': 0.5, 'weight_decay': 0},
    {'dropout': 0.0, 'weight_decay': 5e-4},
    {'dropout': 0.5, 'weight_decay': 5e-4},
]

for cfg in configs:
    # 학습
    acc = 0  # 계산
    print(f'{cfg}: acc = {acc:.2%}')
```

### 실험 5 — Inductive (OGB-Arxiv-style)

```python
# PyG NeighborLoader 로 inductive mini-batch
try:
    from torch_geometric.datasets import OGB_MAG
    from torch_geometric.loader import NeighborLoader
    
    # ... (OGB 데이터 로드)
    loader = NeighborLoader(
        data, num_neighbors=[15, 10], batch_size=128,
        input_nodes=data.train_mask
    )
    
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        # ...
except (ImportError, FileNotFoundError):
    print('OGB not available')
```

---

## 🔗 실전 활용

### 1. OGB Benchmark Suite

OGB (Open Graph Benchmark, Hu 2020) 가 GNN 의 현대 표준:
- OGB-Node: arxiv, products, papers, mag
- Inductive / Large-scale
- Public leaderboard

### 2. Real-world Applications

- **추천 시스템**: user embedding 을 user graph 의 node classification 으로 (Pinterest, Amazon)
- **Fraud detection**: 금융 거래 graph 의 "fraud" vs "normal" classification
- **Protein function prediction**: biology graph 의 GO term classification

### 3. Class Imbalance 처리

- **Class-weighted loss**: `F.cross_entropy(out, y, weight=class_weights)`
- **Focal loss** (Lin 2017): $-\alpha(1-p_t)^\gamma \log p_t$
- **Sampling**: minority class oversampling

### 4. Heterophilic Node Classification

Wikipedia, heterophilic social network 등에서 standard GCN fail. 사용:
- H2GCN (Zhu 2020)
- GPRGNN (Chien 2021)
- FAGCN (Bo 2021)

### 5. Graph Structure Learning

Structure learning + node classification: edge 를 직접 학습. GSL (Wu 2022 survey).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Clean labels | Noisy label 시 robust training 필요 |
| Fixed graph structure | Evolving / noisy graph 별도 |
| Sufficient labeled data | Few-shot node classification 별도 |
| i.i.d. test distribution | Distribution shift 시 transfer learning 필요 |
| Homophily 가정 | Heterophilic 별도 처리 |
| Transductive evaluation | Production 은 inductive |

---

## 📌 핵심 정리

| Metric | Cora | Citeseer | Pubmed |
|--------|------|----------|--------|
| **MLP** | 55% | 59% | 71% |
| **GCN** | 81.5% | 70.3% | 79.0% |
| **GAT** | 83.0% | 72.5% | 79.0% |
| **APPNP** | 83.3% | 71.8% | 80.1% |
| **GCNII (deep)** | 85.5% | 73.4% | 80.3% |

**Standard GCN hyperparameters**: 2 layer, 16 hidden, dropout 0.5, weight_decay $5 \times 10^{-4}$, Adam lr 0.01.

**Semi-supervised cross-entropy**: labeled nodes 만 loss, unlabeled 는 message passing 참여.

---

## 🤔 생각해볼 문제

**문제 1** (기초): GCN 과 MLP 의 Cora 성능 차이 (81.5% vs 55%) 가 graph structure 기여도 ~26%. 이 이득이 어디서 오는가?

<details>
<summary>해설</summary>

**MLP**: Feature $X$ 만 사용 → independent per-node classification. Cora 의 word-bag feature 가 ~55% 의 기본 정보.

**GCN 추가 이득**: Graph structure 정보. Cora citation network 의 **homophily** — 인접 논문이 같은 topic.

**메커니즘**:
1. **Feature smoothing**: 인접 노드 feature aggregation → robust representation (noise reduction)
2. **Information propagation**: Labeled node 의 "topic signal" 이 neighbor 로 propagate → unlabeled 에 spread
3. **Manifold learning**: Graph 가 feature space 의 manifold 구조 제공

**Cora 의 특성**:
- 평균 degree ~ 4 (sparse graph, long tail)
- Homophily ratio ~ 0.81 (같은 class 의 이웃 비율 높음)
- Feature similarity + graph similarity 강한 correlation

**Ablation**:
- MLP + Label propagation: ~67% — graph-only 의 추가 효과
- GCN (combined): 81.5% — 두 정보의 synergy

따라서 GCN 의 우위는 **feature + structure 의 non-trivial combination**. 단순 합산 (feature + LabelProp) 보다 GCN 이 15% 추가 — learned representation 의 힘.

</details>

**문제 2** (심화): Cora 의 140개 labeled nodes (각 class 20개) 가 매우 적다. Semi-supervised vs supervised (모든 label 사용) 의 차이는?

<details>
<summary>해설</summary>

**Semi-supervised (Cora 표준 140 labels)**:
- GCN: 81.5%

**Supervised (all ~2700 labels)**:
- Typically 90%+ accuracy

**왜 semi-supervised 가 의미 있는가?**

Real-world: labeled data expensive, unlabeled 풍부. Cora scenario 가 이 상황 emulation.

**Mechanism**:

1. **Unlabeled 노드의 graph propagation 참여**: Loss 에는 기여 X but message passing 의 중간 노드 역할.

2. **Structural regularization**: Unlabeled node 들이 Laplacian smoothing 의 "anchor" — labeled 신호를 smoothly spread.

3. **Manifold 가정 활용**: Data manifold 의 smoothness prior (close points → same class) 가 graph structure 로 표현.

**Few-shot extrapolation**:
- 140 labels → 81%
- 280 labels (2x) → 83-84%
- All labels → 90%+

**이득의 diminishing return**: 첫 140 labels 이 큰 효과, 이후 incremental.

**Modern extensions**:

- **Active learning**: Best unlabeled nodes 를 annotate — efficient labeling.
- **Pseudo-labeling**: Confident predictions 를 label 로 treat, iterative training.
- **Consistency regularization**: Graph augmentation 하에 consistent predictions.

Cora 의 140 label setting 은 **low-label regime** 의 유명한 benchmark. GCN 의 "semi-supervised 승리" 가 이 scenario 에 강점.

</details>

**문제 3** (논문 비평): Public Cora split (Kipf 2017) 이 SOTA chasing 이 오래되었다. OGB 의 등장이 연구를 어떻게 바꿨는가?

<details>
<summary>해설</summary>

**Cora 표준 split 의 문제**:

1. **Small size**: 2708 nodes, 140 labels. GNN 기여 측정에 noise ↑.
2. **High variance**: Random init 마다 5-10% fluctuation. Single run 으로 모델 비교 부정확.
3. **SOTA chasing**: 수백 papers 가 Cora 81% → 84% 로 1-3% 개선 주장. 실제 improvement 는 marginal, hyperparameter tuning 의 artifact 일 수도.
4. **Overfitted benchmarks**: Small dataset 은 community 가 과도하게 tune → 진짜 진보 측정 어려움.

**Shchur et al. 2018 "Pitfalls of Graph Neural Network Evaluation"**:
- 여러 random split 에서 evaluation 권장
- Multiple runs + significance testing
- MLP baseline 의 competitive 성능 지적

**OGB (Hu et al. 2020)** 의 기여:

1. **Large-scale**: OGB-Products 2.4M, OGB-Papers 111M — graph scale realistic.
2. **Strict split**: Time-based, domain-split (not random) — distribution shift 포함, realistic eval.
3. **Diverse tasks**: Node / link / graph / property prediction.
4. **Public leaderboard**: Reproducible, community-verified results.
5. **Standardized pipeline**: 같은 hyperparameter budget, same val protocol.

**OGB 이후 연구 변화**:

1. **Scalability focus**: Sampling-based (Cluster-GCN, GraphSAINT) 가 필수.
2. **Better baselines**: GBDT (lightgbm on features) 가 종종 GNN 과 competitive.
3. **Inductive setting 중요**: Real-world (Arxiv, Products) 가 inductive.
4. **Model architecture 의 diminishing returns**: ResNet-GCN, Graphormer 등 marginal 향상.
5. **Feature engineering 복귀**: OGB 에서 raw feature + GNN 조합이 중요.

**현대 consensus**:

- Cora / Citeseer: **toy benchmark** — small-scale sanity check.
- OGB: **serious evaluation** for publishable results.
- 둘 다 사용 (Cora 작은 실험, OGB 큰 검증).

**미래**:

- LLM-augmented GNN (TAG = text-attributed graph)
- Graph foundation model 의 evaluation
- OGB-LSC (Large Scale Challenge) 의 KDD Cup 수준

따라서 Cora 의 가치는 GNN 교육·prototype 용. Production-level research 는 OGB.

</details>

---

<div align="center">

[◀ 이전](../ch5-over-smoothing/05-appnp-jkn.md) | [📚 README](../README.md) | [다음 ▶](./02-graph-classification.md)

</div>
