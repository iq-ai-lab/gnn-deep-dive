# 02. Graph Classification

## 🎯 핵심 질문

- Graph-level representation $h_G = \text{READOUT}(\{h_v : v \in V\})$ 에서 READOUT 의 선택이 어떻게 WL 표현력을 결정하는가?
- Sum / Mean / Max / Attention Pool / Set2Set 의 trade-off?
- TU dataset (MUTAG, PROTEINS, IMDB-B) 의 특성과 적합한 모델?
- OGB-molhiv / OGB-molpcba 같은 large-scale graph classification 의 도전?
- GIN + sum readout 이 이론적으로 가장 강한 graph classifier 인 이유?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

**Graph classification** 은 전체 graph 에 label 부여 — 분자 이 toxic 한가, social network 가 어떤 community 에 속하는가 등. Node classification 과 달리:

1. **Multi-graph dataset**: 각 instance 가 별개 graph
2. **Variable graph size**: 분자마다 원자 수 다름
3. **Graph-level readout 필요**: 노드 feature → single vector
4. **WL 표현력이 critical**: 1-WL 가 구분 못하는 graph 쌍은 graph classification 에서 fail

이 문서는 graph classification pipeline, READOUT 의 이론적 분석, TU / OGB benchmark 을 정리.

---

## 📐 수학적 선행 조건

- [Ch3-01~04](../ch3-message-passing/): MPNN, aggregator
- [Ch4-03](../ch4-expressive-power/03-gin-optimality.md): GIN 의 multiset injectivity
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): MLP, UAT

---

## 📖 직관적 이해

### Graph-level Representation

Node embedding $\{h_v\}_{v \in V}$ 을 single vector $h_G$ 로 압축:

$$
h_G = \text{READOUT}(\{h_v : v \in V\})
$$

**Requirements**:
1. **Permutation invariance**: $h_G$ 가 노드 순서 무관
2. **Variable-size**: 다른 $|V|$ 에서도 같은 $d$ output
3. **Multiset injectivity** (이상적): 다른 multiset of $h_v$ → 다른 $h_G$

### READOUT 선택의 표현력

- **Sum**: Multiset injective (Ch4-03), **WL optimal**
- **Mean**: size 정보 손실 (normalized)
- **Max**: multiset-injective 부족
- **Attention pool**: self-attention 으로 set encoding
- **Set2Set** (Vinyals 2015): LSTM-like iterative attention

### 분자 예시

MUTAG: 188 분자, 2 class (mutagenic vs non-mutagenic).
- 각 분자: ~17 atom, 이산 atom type + bond type
- Graph-level classification: toxic 여부

**GIN + sum readout** 이 이론적 최강 (1-WL 도달 + multiset injective).

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Graph Classification Problem

Dataset $\{(G_i, y_i)\}_{i=1}^N$, $G_i$ graph, $y_i \in \{1, \ldots, C\}$.

**Task**: Learn $f: \mathcal G \to \{1, \ldots, C\}$.

### 정의 2.2 — Graph-level Readout

함수 $R: \mathbb R^{n \times d} \to \mathbb R^{d'}$ — permutation invariant, variable-size input.

### 정의 2.3 — Standard Readouts

- **Sum**: $h_G = \sum_v h_v$
- **Mean**: $h_G = \frac{1}{|V|} \sum_v h_v$
- **Max**: $h_G = \max_v h_v$ (element-wise)
- **Sort**: Top-$k$ selection after sorting (SortPool, Zhang 2018)
- **Attention pool**: $h_G = \sum_v \alpha_v h_v$, $\alpha_v = \text{softmax}(a^T h_v)$
- **Set2Set** (Vinyals 2015): LSTM + attention iteratively
- **DiffPool** (Ying 2018): learnable hierarchical clustering

### 정의 2.4 — GIN + All-layer Sum Readout

Xu 2019 권장:
$$
h_G = \big\|_{l=0}^L \left( \sum_{v \in V} h_v^{(l)} \right)
$$

(모든 layer 의 sum-readout 을 concat)

### 정의 2.5 — Standard Pipeline

```python
# 1. For each graph G_i:
#    - node features X_i
#    - edge_index_i
# 2. Batched GNN forward (PyG batch 처리)
# 3. Graph-level readout per batch
# 4. Classifier on h_G
# 5. Cross-entropy loss on batch
```

---

## 🔬 정리와 결과

### 정리 2.1 — Readout 의 WL 표현력

**Theorem**: Graph classification 의 표현력 $\leq$ node-level 표현력 + readout injectivity.

**Sum readout**: Multiset injective → WL 표현력 그대로.
**Mean readout**: Multiset size 정보 손실 → Strict less.
**Max readout**: 단일 최댓값만 → Strict less.

### 정리 2.2 — GIN + Sum Readout = Maximum Expressive

**Theorem (Xu 2019)**: GIN (message passing 의 1-WL 도달) + sum readout 이 **graph-level 1-WL** 의 모든 구분 가능 graph 를 구분.

### 정리 2.3 — TU Dataset Benchmark 성능

| Dataset | # Graphs | Avg # Nodes | # Classes |
|---------|----------|-------------|-----------|
| MUTAG | 188 | 17.9 | 2 |
| PROTEINS | 1113 | 39.1 | 2 |
| IMDB-B | 1000 | 19.8 | 2 |
| COLLAB | 5000 | 74.5 | 3 |
| NCI1 | 4110 | 29.8 | 2 |
| PTC | 344 | 14.3 | 2 |

**Standard results** (10-fold CV, Xu 2019):

| Model | MUTAG | PROTEINS | IMDB-B | COLLAB |
|-------|-------|----------|--------|--------|
| GCN | 85.6 | 76.0 | 74.0 | 79.0 |
| GraphSAGE | 85.1 | 75.9 | 72.3 | 79.7 |
| **GIN** | **89.4** | **76.2** | **75.1** | **80.2** |
| WL kernel | 90.4 | 75.0 | 73.8 | 78.9 |

GIN 이 most benchmark 에서 best.

### 정리 2.4 — OGB-molhiv Benchmark

**Molecular property prediction**:
- OGB-molhiv: 40k 분자, binary classification (HIV inhibit)
- OGB-molpcba: 128 biological activities (multi-label)

| Model | OGB-molhiv (ROC-AUC) |
|-------|----------------------|
| GIN | 75.8 |
| GIN + VN (virtual node) | 77.1 |
| DeeperGCN | 78.6 |
| PNA (multi-aggregator) | 79.0 |
| Graphormer | 80.5 |

### 정리 2.5 — Readout 별 Empirical 비교

**GIN + various readouts on MUTAG**:

| Readout | Accuracy |
|---------|----------|
| Sum | 89.4% |
| Mean | 86.5% |
| Max | 84.2% |
| Attention pool | 88.9% |
| Set2Set | 89.0% |
| DiffPool | 85.0% |

Sum 이 이론 + 실증 모두 우위.

---

## 💻 구현

### 실험 1 — GIN for Graph Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class GINLayer(nn.Module):
    def __init__(self, d_in, d_out, eps=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
            nn.ReLU(), nn.Linear(d_out, d_out)
        )
        self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float))
    
    def forward(self, x, edge_index):
        src, dst = edge_index
        agg = scatter_add(x[src], dst, dim=0, dim_size=x.size(0))
        return self.mlp((1 + self.eps) * x + agg)

class GIN_GraphClassifier(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers=3, readout='sum'):
        super().__init__()
        self.readout = readout
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = d_in if i == 0 else d_hid
            self.layers.append(GINLayer(in_d, d_hid))
        self.cls = nn.Linear(d_hid * num_layers, d_out)
    
    def forward(self, x, edge_index, batch):
        layer_readouts = []
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
            if self.readout == 'sum':
                r = scatter_add(h, batch, dim=0)
            elif self.readout == 'mean':
                from torch_scatter import scatter_mean
                r = scatter_mean(h, batch, dim=0)
            elif self.readout == 'max':
                from torch_scatter import scatter_max
                r, _ = scatter_max(h, batch, dim=0)
            layer_readouts.append(r)
        h_G = torch.cat(layer_readouts, dim=-1)
        return self.cls(h_G)
```

### 실험 2 — MUTAG Training

```python
try:
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    
    dataset = TUDataset(root='./data/MUTAG', name='MUTAG').shuffle()
    train_ds = dataset[:150]
    test_ds = dataset[150:]
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    model = GIN_GraphClassifier(dataset.num_features, 32, dataset.num_classes, num_layers=3, readout='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward(); optimizer.step()
        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch.x, batch.edge_index, batch.batch).argmax(-1)
                correct += (pred == batch.y).sum().item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: test acc {correct / len(test_ds):.2%}')
except (ImportError, FileNotFoundError):
    print('PyG / TUDataset not available')
```

### 실험 3 — Readout 비교

```python
# 같은 GIN, 다른 readout
readouts = ['sum', 'mean', 'max']
results = {}
for ro in readouts:
    model = GIN_GraphClassifier(
        dataset.num_features, 32, dataset.num_classes,
        num_layers=3, readout=ro
    )
    # 학습 & 평가 (생략)
    results[ro] = 0  # placeholder

for ro, acc in results.items():
    print(f'{ro}: {acc:.2%}')
```

### 실험 4 — Attention Pool

```python
class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)
    def forward(self, x, batch):
        alpha = self.attn(x).squeeze(-1)
        # Softmax per graph
        from torch_scatter import scatter_softmax, scatter_add
        alpha = scatter_softmax(alpha, batch)
        return scatter_add(alpha.unsqueeze(-1) * x, batch, dim=0)
```

### 실험 5 — DiffPool (Hierarchical)

```python
# DiffPool 은 learnable clustering
# 간략: 매 layer 후 cluster assignment 학습 → coarser graph
# 너무 복잡하므로 interface 만
class DiffPoolLayer(nn.Module):
    """Simplified DiffPool."""
    def __init__(self, d_in, d_out, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.gnn_embed = GINLayer(d_in, d_out)
        self.gnn_assign = GINLayer(d_in, n_clusters)
    
    def forward(self, x, adj):
        z = self.gnn_embed(x, None)  # simplified
        s = F.softmax(self.gnn_assign(x, None), dim=-1)   # [n, n_clusters]
        # Coarser node features + coarser adjacency
        x_next = s.T @ z
        adj_next = s.T @ adj @ s
        # Auxiliary losses (link prediction, entropy) 포함
        return x_next, adj_next
```

---

## 🔗 실전 활용

### 1. 분자 Chemistry

QM9, OGB-molhiv/molpcba 가 주요 benchmark. GNN (GIN, PNA) + chemistry knowledge (bond type, valence) 가 standard.

**주요 모델**:
- GIN, GIN-edge
- DGN (Beaini 2021, Directional Graph Network)
- GPS (Rampášek 2022, graph transformer for chemistry)
- Graphormer (Ch7-01)

### 2. Social Network Analysis

IMDB, COLLAB 의 community pattern 분류. GIN 이 표준.

### 3. Code Analysis

Abstract Syntax Tree (AST) 의 graph representation. Graph classification 으로 code type, vulnerability detection.

### 4. PyG Implementation

```python
from torch_geometric.nn import GINConv, global_add_pool

conv1 = GINConv(mlp1)
conv2 = GINConv(mlp2)

x = conv2(conv1(x, edge_index), edge_index)
h_G = global_add_pool(x, batch)   # sum readout
```

### 5. 10-fold Cross-Validation

TU datasets 표준:
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), [d.y for d in dataset])):
    # train on train_idx, eval on test_idx
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Permutation invariant readout | Order 정보 (SMILES string) 시 lose |
| Fixed $L$ layer | Adaptive per-graph depth 어려움 |
| Sum readout | Large graph 시 magnitude 폭발 |
| Multiset independence | Node correlation 복잡 시 제한 |
| TU dataset 소수 split | High variance 10-fold CV |
| Molecular 3D 정보 X | EGNN 등 equivariant GNN 필요 (Ch7-03) |

---

## 📌 핵심 정리

$$\boxed{h_G = \text{READOUT}(\{h_v\}), \quad \text{(permutation invariant)}}$$

| Readout | Multiset Injective | 실전 |
|---------|---------------------|------|
| **Sum (GIN)** | ✓ | Best for graph classification |
| **Mean** | ✗ | OK for size-normalized |
| **Max** | ✗ | OK for salient features |
| **Attention** | ≈ (depending on design) | Flexible, extra params |
| **DiffPool** | Hierarchical | Large graph |
| **Set2Set** | ≈ (LSTM) | Slow but expressive |

**Standard practice**:
- GIN + sum readout + all-layer concat
- 3-5 layer, hidden 32-64
- BatchNorm between GIN layers
- Adam lr 0.01, 100-200 epochs

---

## 🤔 생각해볼 문제

**문제 1** (기초): 크기가 다른 두 graph (5 nodes vs 50 nodes) 의 sum vs mean readout 의 output magnitude 차이는?

<details>
<summary>해설</summary>

**Sum**: $h_G = \sum_v h_v$. $|V|$ 에 비례.
- 5-node: $|h_G| \approx 5 \cdot |h|$
- 50-node: $|h_G| \approx 50 \cdot |h|$
- 10배 차이 → downstream classifier 가 size-bias.

**Mean**: $h_G = \frac{1}{|V|} \sum_v h_v$. $|V|$ 무관.
- 5-node, 50-node 둘 다 $|h_G| \approx |h|$

**Trade-off**:

- **Sum의 장점**: Multiset injective (WL optimal). 같은 composition 이라도 size 다르면 다른 output.
- **Sum의 단점**: Magnitude 차이 → unstable training, normalization 필요.
- **Mean의 장점**: Scale-invariant, stable.
- **Mean의 단점**: Size 정보 손실, WL 상한 도달 못함.

**Solution**:
1. **BatchNorm on h_G**: Sum 후 정규화 — GIN-edge 표준.
2. **Concat sum + mean + max**: PNA 스타일, 여러 정보 결합.
3. **Adaptive magnitude**: LayerNorm per-graph.

**권장**: BatchNorm 적용한 sum readout — GIN 표준. $\square$

</details>

**문제 2** (심화): DiffPool 의 hierarchical clustering 이 flat readout 보다 어떤 graph 에서 우월한가? 한계는?

<details>
<summary>해설</summary>

**DiffPool (Ying 2018)** 의 hierarchical clustering:

각 layer 에서 soft cluster assignment $S \in [0, 1]^{n \times K}$, coarser graph 로 전환:
$$
X_{\text{next}} = S^T X, \quad A_{\text{next}} = S^T A S
$$

**DiffPool 이 flat 보다 유리**:

1. **Multi-scale structure**: Protein 의 secondary structure, social network 의 community hierarchy.

2. **Long-range dependency**: Coarser graph 에서는 long-range edge 가 short-range — GNN 이 효과적으로 처리.

3. **Large graph size**: 100+ node 의 graph 에서 flat readout 의 magnitude 문제 회피.

**예**: PROTEINS (proteins secondary structure): DiffPool 이 GIN 과 비슷 but 일부 graph 에서 +1%.

**한계**:

1. **Learnable cluster 의 instability**: Cluster assignment $S$ 의 학습이 unstable. Auxiliary loss (link prediction, entropy regularization) 필요.

2. **Computational cost**: $S^T A S$ 의 $O(n K)$ multiplication. Large graph 시 expensive.

3. **Fixed $K$ (cluster 수)**: Graph-specific $K$ 학습 어려움. Fixed hyperparameter 가 모든 graph 에 맞지 않음.

4. **Rigid hierarchy**: 실제 community 가 soft/overlapping 이면 $S$ 학습 어려움.

5. **Empirical results**: TU benchmark 에서 GIN 과 대체로 비슷. 큰 우위 dataset-specific.

**Modern 대체**:

- **Set Transformer** (Lee 2019): Attention-based set encoding, permutation invariant.
- **SortPool** (Zhang 2018): Top-k sort 후 CNN — 간단.
- **Global attention pool**: 간단 + 효과적 — many papers 의 표준.

**결론**:

DiffPool 이 이론적 우아하고 hierarchical structure 를 내포하는 graph 에 적합. 단 실전 TU benchmark 에서는 GIN + sum 이 더 stable 하고 학습 쉬움.

Real-world hierarchical graph (protein, program structure) 에서 DiffPool 의 가치 — niche but important use case.

</details>

**문제 3** (논문 비평): 분자 graph classification 에서 왜 PNA, Graphormer 가 GIN 을 outperform 하는가? "1-WL 표현력" 의 실전 한계?

<details>
<summary>해설</summary>

**GIN 의 1-WL 표현력 한계 (in practice)**:

1. **Substructure counting**: 분자 의 functional group (ring, benzene, etc.) 가 substructure counting 에 의존. GIN 이 triangle/cycle counting 못함 (Chen 2020).

2. **Heterogeneous chemistry**: Atom type + bond type + 3D position 의 복잡한 상호작용. GIN 의 homogeneous message passing 한계.

3. **Long-range dependency**: 분자 의 "intramolecular" effect (예: resonance structure) 가 long-range. 1-WL 은 local only.

4. **Real-world TU 에서 1-WL 충분**: TU MUTAG 의 간단한 분자 는 1-WL 충분. OGB-molhiv 의 복잡한 분자 는 부족.

**PNA (Principal Neighborhood Aggregation, Corso 2020)**:

Multi-aggregator: sum + mean + max + min + std 등을 concat + degree scaling:
$$
h_i' = \text{MLP}([\text{sum} \| \text{mean} \| \text{max} \| \ldots] \odot d_{\text{scaler}})
$$

**이득**:
- 같은 1-WL 표현력 (sum 포함)
- But degree-aware, variance-aware — practical 더 많은 정보
- Empirical: OGB-molhiv 에서 GIN 75.8 → PNA 79.0

**Graphormer (Ch7-01)**:

- Dense attention (모든 pair)
- Spatial encoding (SPD bias)
- Centrality encoding
- Edge encoding

**이득**:
- 1-WL 상한 우회 (dense attention = k-hop)
- Structural encoding 으로 substructure 감지
- Empirical: OGB-molhiv 80.5%

**핵심 통찰**:

- **1-WL 표현력은 이론 ceiling but not empirical one**: Real-world performance 는 inductive bias + 정보 풍부성이 결정.
- **Aggregation 의 다양성** (PNA): 같은 표현력 내에서 더 많은 signal extraction.
- **Architecture 의 보강** (Graphormer): 표현력 자체 넘어 PE + attention 으로 추가 정보.

**결론**:

GIN 이 이론적 최적이지만 실전 chemistry 에서는:
- Chemistry-specific features (bond type, valence) 가 중요
- Structural encoding (PE, substructure) 이 1-WL 보강
- Attention-based model 이 더 flexible

따라서 chemistry 분야 modern GNN 은 "GIN + PE + attention" hybrid — Graphormer 의 철학. Ch7 의 modern graph transformer 가 이 trend.

**미래**: "GNN foundation model for chemistry" — pre-trained on massive molecular dataset, fine-tune on downstream. Graph Transformer 가 이 direction.

</details>

---

<div align="center">

[◀ 이전](./01-node-classification.md) | [📚 README](../README.md) | [다음 ▶](./03-link-prediction.md)

</div>
