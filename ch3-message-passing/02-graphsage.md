# 02. GraphSAGE — Sampling과 Inductive Learning (Hamilton 2017)

## 🎯 핵심 질문

- GraphSAGE 의 "SAGE" (SAmple and aggreGatE) 가 의미하는 두 단계는 무엇인가?
- Mean / Pool / LSTM aggregator 의 차이와 trade-off 는?
- Transductive (GCN) vs inductive (GraphSAGE) learning 의 본질적 차이는?
- Fixed-size neighbor sampling $|N_s(v)| = S$ 가 어떻게 mini-batch 학습을 가능하게 하는가?
- 대규모 그래프 (Reddit, PPI) 에서 GraphSAGE 의 scalability 가 GCN 대비 어떻게 우월한가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GCN 은 **transductive** — 학습 시 전체 그래프 (인접 행렬 + 모든 노드 feature) 를 알아야 함. 이는 두 한계로 이어집니다:
1. **새 노드 처리 불가**: 학습 후 새 노드가 추가되면 전체 재학습
2. **Mini-batch 불가**: 단일 그래프 → 단일 forward pass, GPU 메모리 한계

GraphSAGE (Hamilton et al. 2017 — **"Inductive Representation Learning on Large Graphs"**) 가 두 문제를 해결:
1. **Inductive**: 학습 후 새 노드에 적용 가능 (이웃 sampling 만 필요)
2. **Mini-batch**: 노드별 fixed-size neighborhood 만 사용 → mini-batch 학습 가능

이는 Pinterest (Ying 2018, PinSage) 에서 30억 노드급 추천 시스템 등 대규모 산업 응용의 첫 길.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-mpnn-framework.md](./01-mpnn-framework.md) — MPNN framework
- [Ch1-03](../ch1-graph-laplacian/03-normalized-laplacian.md): Mean aggregator 와 random walk Laplacian 의 관계
- 확률론: Sampling, variance

---

## 📖 직관적 이해

### Sample → Aggregate 의 두 단계

**Step 1 (Sample)**: 각 노드 $v$ 의 이웃 $N(v)$ 에서 fixed size $S$ 만큼 랜덤 sample → $N_s(v)$.

**Step 2 (Aggregate)**: $N_s(v)$ 의 feature 를 aggregator (mean / pool / LSTM) 로 모음.

이 두 단계가 **transductive 한계 우회** + **scalability** 를 동시 제공.

### Inductive vs Transductive

- **Transductive (GCN)**: 학습 시 모든 노드 보임. Test 시 같은 그래프의 unlabeled 노드 분류. 단, 새 노드 (training 시 안 봄) 처리 불가.

- **Inductive (GraphSAGE)**: 학습 시 일부 그래프만 보임. Test 시 **새 그래프** 또는 **새 노드** 에 적용. Real-world (계속 변하는 social network 등) 에 적합.

### Aggregator 들

**Mean**: 가장 단순. $h_{N(v)} = \text{mean}(\{h_u : u \in N_s(v)\})$. GCN-like.

**Pool**: 각 이웃 feature 에 MLP + element-wise max:
$$
h_{N(v)} = \max(\{ \sigma(W_{\text{pool}} h_u + b) : u \in N_s(v)\})
$$

표현력 ↑ (max 가 multiset 정보 일부 보존), 학습 어려움 ↑.

**LSTM**: 이웃을 sequence 로 보고 LSTM 적용:
$$
h_{N(v)} = \text{LSTM}(\{h_u : u \in N_s(v)\})
$$

순서 의존 → permutation invariance 깨짐. 실전에서 random shuffle 평균.

### Concat-then-Update 패턴

GraphSAGE 의 update:
$$
h_v^{(l+1)} = \sigma(W^{(l)} \cdot [h_v^{(l)} \| h_{N(v)}^{(l)}])
$$

(concat → linear → activation)

이는 GCN 의 self-loop 통합과 다른 형태 — **자기 정보와 이웃 정보를 separate 한 뒤 결합** → 표현력 ↑ (GCN 의 tied weight 한계 완화).

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Neighbor Sampling

각 layer $l$ 에서 sample size $S_l$. 노드 $v$ 의 $l$-th layer neighborhood:
$$
N_s^{(l)}(v) = \text{Uniform sample of size } S_l \text{ from } N(v)
$$

$|N(v)| < S_l$ 이면 with replacement 또는 $N(v)$ 전체.

### 정의 2.2 — GraphSAGE Aggregator

세 종류:

**Mean aggregator**:
$$
h_v^{(l+1)} = \sigma\left( W^{(l)} \cdot \text{mean}\{h_v^{(l)}\} \cup \{h_u^{(l)} : u \in N_s(v)\} \right)
$$

**Pool aggregator**:
$$
h_{N(v)} = \max\{\sigma(W_{\text{pool}} h_u + b) : u \in N_s(v)\}, \quad h_v^{(l+1)} = \sigma(W^{(l)} [h_v^{(l)} \| h_{N(v)}])
$$

**LSTM aggregator**:
$$
h_{N(v)} = \text{LSTM}(\text{shuffle}([h_u : u \in N_s(v)])), \quad h_v^{(l+1)} = \sigma(W^{(l)} [h_v^{(l)} \| h_{N(v)}])
$$

### 정의 2.3 — Multi-Layer GraphSAGE

$L$-layer GraphSAGE: layer $l$ 마다 $N_s^{(l)}$ 새로 sample. Receptive field $\prod_l S_l$.

### 정의 2.4 — Unsupervised GraphSAGE Loss

Random walk-based positive pair, negative sampling:
$$
\mathcal L_{\text{unsup}} = -\log \sigma(z_v^T z_u) - Q \cdot \mathbb{E}_{u_n \sim P_n} [\log \sigma(-z_v^T z_{u_n})]
$$

(positive: walk 동시 등장, negative: 랜덤)

---

## 🔬 정리와 결과

### 정리 2.1 — Mean Aggregator ≈ Random Walk Laplacian

**Theorem**: GraphSAGE-mean (no self in message, no concat) 은 정확히:
$$
h_v^{(l+1)} = \sigma\left( W^{(l)} \cdot \frac{1}{|N_s(v)|} \sum_{u \in N_s(v)} h_u^{(l)} \right)
$$

이는 random walk transition $P = D^{-1} A$ 적용과 같음 (sample 이 충분히 크면 expectation).

**증명**: $\mathbb{E}_{N_s} [\frac{1}{|N_s(v)|} \sum h_u] = \frac{1}{|N(v)|} \sum_{u \in N(v)} h_u = (P h)_v$. $\square$

### 정리 2.2 — Sampling Variance

**Theorem**: $|N_s(v)| = S$ uniform sampling 에서 mean estimate variance:
$$
\text{Var}(\hat h_{N(v)}) = \frac{\sigma^2}{S} \cdot (1 - S/|N(v)|)
$$

(finite population correction)

작은 $S$ → 큰 variance → noisy gradient. 이는 over-smoothing 완화에 기여 (Ch5-04).

### 정리 2.3 — Inductive Generalization

**Theorem (informal)**: GraphSAGE 의 학습 파라미터 $W^{(l)}, W_{\text{pool}}^{(l)}$ 는 graph 와 독립. 따라서 새 graph $G'$ 에서 같은 $W$ 적용 가능 (단, $X'$ feature space 가 동일).

**증명**: Forward pass 는 $X, A$ 만 사용. $W$ 는 input/output dimension 에만 의존. 별도 graph-specific 파라미터 없음. $\square$

이는 GCN 의 $\tilde A$ 자체가 graph-specific 인 것과 대비.

### 정리 2.4 — Mini-batch Computational Cost

**Theorem**: $L$-layer GraphSAGE 의 mini-batch (batch size $B$) cost:
$$
O(B \cdot \prod_{l=1}^L S_l \cdot d^2)
$$

GCN 의 full-batch $O(n d^2)$ 와 비교:
- GraphSAGE: $B \cdot S^L \cdot d^2$ — graph-size $n$ 과 무관
- GCN: $n \cdot d^2$ — 전체 그래프 의존

대규모 ($n \gg B \cdot S^L$) 에서 GraphSAGE 압도적 우위.

### 정리 2.5 — Pool vs Mean 표현력

**Pool > Mean 표현력**: Pool 의 max 가 multiset 의 일부 정보 보존 (특히 sparse one-hot feature).

단, Pool 은 sum 보다 약함 (Ch4 GIN). 따라서:
$$
\text{Sum} > \text{Pool} > \text{Mean}
$$

GraphSAGE-pool 이 mean 보다 평균 +1~2% accuracy.

---

## 💻 구현

### 실험 1 — GraphSAGE Mean Aggregator

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max

class SAGEConv(nn.Module):
    def __init__(self, d_in, d_out, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        if aggr == 'pool':
            self.W_pool = nn.Linear(d_in, d_in)
        # concat self + agg → linear
        self.W = nn.Linear(2 * d_in, d_out)
    
    def forward(self, x, edge_index):
        src, dst = edge_index
        if self.aggr == 'mean':
            agg = scatter_mean(x[src], dst, dim=0, dim_size=x.size(0))
        elif self.aggr == 'pool':
            transformed = F.relu(self.W_pool(x))
            agg, _ = scatter_max(transformed[src], dst, dim=0, dim_size=x.size(0))
        h = torch.cat([x, agg], dim=-1)
        return F.relu(self.W(h))
```

### 실험 2 — Neighbor Sampling 구현

```python
import numpy as np

def sample_neighbors(adj_dict, node, S, replace=False):
    """Sample S neighbors of node from adj_dict."""
    neighbors = adj_dict[node]
    if len(neighbors) == 0:
        return [node]  # self-fallback
    if len(neighbors) <= S:
        return list(neighbors)
    if replace:
        return list(np.random.choice(list(neighbors), S, replace=True))
    return list(np.random.choice(list(neighbors), S, replace=False))

def collect_subgraph(adj_dict, batch_nodes, layer_sizes):
    """L-layer ego-subgraph for batch."""
    layers = [set(batch_nodes)]
    for S in layer_sizes:
        next_layer = set()
        for v in layers[-1]:
            sampled = sample_neighbors(adj_dict, v, S)
            next_layer.update(sampled)
        layers.append(layers[-1] | next_layer)
    return layers
```

### 실험 3 — Karate Club 에서 GraphSAGE

```python
import networkx as nx

G = nx.karate_club_graph()
n = G.number_of_nodes()
edges = np.array(list(G.edges())).T
edge_index = torch.tensor(np.concatenate([edges, edges[::-1]], axis=1), dtype=torch.long)

X = torch.eye(n)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True

class GraphSAGE(nn.Module):
    def __init__(self, d_in, d_hid, d_out, aggr='mean'):
        super().__init__()
        self.sage1 = SAGEConv(d_in, d_hid, aggr)
        self.sage2 = SAGEConv(d_hid, d_out, aggr)
    
    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        return self.sage2(h, edge_index)

model = GraphSAGE(d_in=n, d_hid=8, d_out=2, aggr='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

for epoch in range(200):
    model.train(); optimizer.zero_grad()
    out = model(X, edge_index)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward(); optimizer.step()

acc = (model(X, edge_index).argmax(-1) == labels).float().mean().item()
print(f'GraphSAGE-mean accuracy: {acc:.2%}')
```

### 실험 4 — Mean vs Pool vs Sum 비교

```python
results = {}
for aggr in ['mean', 'pool']:
    model = GraphSAGE(d_in=n, d_hid=8, d_out=2, aggr=aggr)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    for epoch in range(200):
        model.train(); optimizer.zero_grad()
        out = model(X, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
    acc = (model(X, edge_index).argmax(-1) == labels).float().mean().item()
    results[aggr] = acc

print('Aggregator comparison on Karate Club:')
for k, v in results.items():
    print(f'  {k}: {v:.2%}')
```

### 실험 5 — Inductive Test (Train on subgraph, test on full)

```python
# Train: nodes 0~20 만 사용
train_nodes = list(range(20))
G_train = G.subgraph(train_nodes).copy()
edges_tr = np.array(list(G_train.edges())).T
edge_index_tr = torch.tensor(np.concatenate([edges_tr, edges_tr[::-1]], axis=1), dtype=torch.long)
X_tr = torch.eye(len(train_nodes))
labels_tr = labels[train_nodes]

model = GraphSAGE(d_in=len(train_nodes), d_hid=8, d_out=2, aggr='mean')
# (Real-world inductive 는 일반 X feature, 여기선 단순화)
# ... 학습 코드 생략

# Test: 새 노드 (20~33) 의 embedding 도 forward 가능 (in principle)
# 단 input dim 일치 필요 → real-world 는 graph-agnostic feature 사용
print('Inductive learning: parameter W 가 graph 와 독립')
```

---

## 🔗 실전 활용

### 1. PinSage (Ying 2018)

Pinterest 의 추천 시스템 — 30억 노드급 graph 에서 GraphSAGE 적용. **Random walk-based neighborhood sampling** + **importance pooling**. Industry-grade GNN 의 첫 사례.

### 2. OGB Benchmarks

OGB-Products (2.4M nodes), OGB-Reddit, OGB-Papers100M 에서 GraphSAGE 가 standard baseline. SAGE-mean / pool / LSTM 비교 표준.

### 3. NeighborLoader (PyG)

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],   # layer-wise sampling
    batch_size=128,
    input_nodes=train_mask
)
for batch in loader:
    out = model(batch.x, batch.edge_index)
    # batch.batch_size 의 노드만 prediction (target nodes)
```

### 4. Heterogeneous GraphSAGE

다양한 노드/edge type 처리 (HinSAGE, R-GraphSAGE) — Ch3-05 에서 자세히.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Uniform neighbor sampling | Importance sampling (PinSage) 가 더 효율적 |
| Fixed sample size $S$ | Adaptive sampling (FastGCN, ASGCN) 가능 |
| Mean aggregator | 표현력 < sum (GIN) |
| LSTM aggregator | Permutation invariance 깨짐, slow |
| Train/test feature space 일치 | 새 graph 에서 다른 feature space 시 transfer 어려움 |
| Mini-batch 가정 (full batch도 가능) | Full batch training 시 GCN 과 거의 차이 없음 |
| Independent across layers | Cross-layer dependence (NS bias) 무시 |

---

## 📌 핵심 정리

$$\boxed{h_v^{(l+1)} = \sigma\left( W^{(l)} \cdot [h_v^{(l)} \| \text{Agg}\{h_u^{(l)} : u \in N_s(v)\}] \right)}$$

| 항목 | GraphSAGE |
|------|-----------|
| **Sampling** | Fixed-size $S$, layer-wise |
| **Aggregator** | Mean / Pool / LSTM |
| **Inductive** | ✓ |
| **Mini-batch** | ✓ ($O(B \cdot S^L \cdot d^2)$) |
| **Concat-then-update** | $[h_v \| h_{N(v)}]$ |
| **Self vs neighbor 분리** | ✓ (separate weights) |
| **Variance-based smoothing 완화** | ✓ (sampling) |
| **Real-world scale** | 30B nodes (PinSage) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GraphSAGE-mean (with self in mean, no concat — simplified) 와 GCN 의 차이를 식으로 비교하라.

<details>
<summary>해설</summary>

**Simplified SAGE-mean**: $h_v' = \sigma(W \cdot \frac{1}{|N(v)|+1}(h_v + \sum_{u \in N(v)} h_u))$

**GCN**: $h_v' = \sigma(\sum_{u \in N(v) \cup \{v\}} \frac{1}{\sqrt{\tilde d_v \tilde d_u}} h_u W)$

**차이**:

1. **Normalization**: SAGE-mean 은 $1/(|N(v)|+1)$ — node $v$ 의 degree 만, $u$ 의 degree 무관. GCN 은 $1/\sqrt{\tilde d_v \tilde d_u}$ — 양 노드 degree 둘 다.

2. **Symmetry**: GCN 은 symmetric ($\hat A = \hat A^T$, undirected). SAGE-mean 은 row-stochastic 형태 — directed-friendly.

3. **Self vs neighbor 분리**: 원본 SAGE 는 concat 으로 self/neighbor weight 분리. Simplified mean 은 GCN 과 더 유사.

**결론**: SAGE-mean (simplified) 와 GCN 은 normalization 만 다름. 실제 SAGE 의 우위는 concat-then-linear 형태에서 옴 — 자기 정보와 이웃 정보의 weight 분리 학습.

</details>

**문제 2** (심화): Neighbor sampling 으로 인한 message passing variance 가 mini-batch 학습의 수렴에 어떤 영향을 미치는가? FastGCN (Chen 2018) 의 importance sampling 이 variance 를 어떻게 줄이는가?

<details>
<summary>해설</summary>

**Variance 의 영향**:

Stochastic gradient $\nabla_W \mathcal L$ 의 variance ↑ → 학습 noisy. 단, mini-batch SGD 의 일반 효과 — 적당한 variance 는 implicit regularization (Bayesian SGD 관점).

**문제**: $L$-layer 에서 sampling variance $\propto S^{-L}$ (compound). $L=3, S=5$ 면 매우 noisy.

**FastGCN (Chen 2018)** 의 해결:
- Layer-wise (not node-wise) sampling — 각 layer 에서 fixed number of nodes 만 처리
- **Importance sampling**: degree-proportional sampling. 노드 $u$ 가 sample 될 확률 $q(u) \propto \|h_u\|^2 \cdot \text{deg}(u)$
- Bias correction: $\frac{1}{q(u)} h_u$ 로 weighted sum

**Variance reduction**:

Importance sampling theorem: $q^*(u) \propto |M(u)|$ 가 optimal — variance 최소. FastGCN 은 이 optimal 에 근사.

**결과**: 같은 $S$ 에서 GraphSAGE 보다 작은 variance, 더 안정 학습.

**LADIES (Zou 2019)**: layer-dependent importance, neighborhood sampling 의 variance 더 감소.

</details>

**문제 3** (논문 비평): GraphSAGE 가 inductive 라고 주장하지만, 실전에서 같은 graph 의 unseen nodes (transductive-like with new nodes) 적용이 대부분이다. 진정한 inductive (다른 graph 로 transfer) 가 어려운 이유와 가능한 해결책을 논하라.

<details>
<summary>해설</summary>

**진정한 inductive 의 어려움**:

1. **Feature space mismatch**: 학습 graph $G_1$ 의 feature distribution 이 test graph $G_2$ 와 다름. Cora ↔ Citeseer 도 feature dim/semantic 다름. 일반화 어려움.

2. **Graph statistics 차이**: 
- Density (sparse vs dense)
- Degree distribution (power-law vs uniform)
- Homophily ratio
다른 graph 는 다른 inductive bias 필요.

3. **Position information 부재**: GraphSAGE 는 structural information 만, position-aware 학습 못함. Test graph 의 새 community 가 학습 graph 에 없으면 일반화 X.

4. **Distribution shift**: Edge generating process 가 다름 → learned $W$ 의 자체 mismatch.

**해결책**:

1. **Domain-invariant features**: 텍스트 (BERT embedding), 이미지 (ResNet embedding) 등 graph-agnostic feature 사용. Pre-trained encoder 위 GNN.

2. **Graph contrastive learning** (GraphCL, GraphMAE): self-supervised pre-training on diverse graphs → universal GNN encoder.

3. **Meta-learning** (MAML-GNN): 여러 graph 에서 meta-learn → fast adaptation.

4. **Universal GNN**: 최근 trend — task-agnostic, data-agnostic GNN backbone (GraphGPT, OFA — One For All).

5. **Graph foundation model**: LLM-style pre-training on web-scale graph data (Galileo 2024 등 시도).

**결론**: GraphSAGE 의 "inductive" 는 implementation level (parameter graph-agnostic). 진정한 cross-graph transfer 는 representation learning 의 더 깊은 문제 — 현재 활발 연구. Ch7-04 의 "GNN 의 미래" 에서 더 자세히.

</details>

---

<div align="center">

[◀ 이전](./01-mpnn-framework.md) | [📚 README](../README.md) | [다음 ▶](./03-gat.md)

</div>
