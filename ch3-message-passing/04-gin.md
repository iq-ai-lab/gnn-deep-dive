# 04. Graph Isomorphism Network — GIN (Xu 2019)

## 🎯 핵심 질문

- GIN 의 update rule $h_i^{(l+1)} = \text{MLP}((1 + \epsilon) h_i^{(l)} + \sum_{j \in N(i)} h_j^{(l)})$ 의 각 항이 의미하는 것은?
- 왜 sum + MLP 가 multiset 의 universal injective function 인가?
- GIN 이 "1-WL 등가 표현력" 의 message passing 임이 왜 중요한가?
- Mean / Max aggregator 가 왜 1-WL 보다 weak 한가?
- Graph classification (TU dataset) 에서 GIN 이 GCN/GAT 를 압도하는 이유는?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GCN, GraphSAGE, GAT 등이 등장한 후 자연스러운 질문: **"가장 강력한 message passing GNN 은 무엇인가?"** Xu et al. 2019 (**"How Powerful are Graph Neural Networks?"**) 가 결정적 답:

1. **모든 message passing GNN 의 표현력 ≤ 1-WL** (Weisfeiler-Lehman test, Ch4)
2. **이 상한을 정확히 달성하는 GNN 이 GIN** (Graph Isomorphism Network)
3. **핵심 trick: sum aggregator + MLP** — multiset universal function

GIN 은 **graph classification benchmark (TU dataset)** 에서 SOTA 를 달성하며, message passing 의 이론적 최적임을 실증. 이 문서는 GIN 의 핵심 정리를 직관·정의·증명으로 다룹니다 (정확한 표현력 증명은 Ch4-03 에서).

---

## 📐 수학적 선행 조건

- 이전 문서: [01-mpnn-framework.md](./01-mpnn-framework.md) — MPNN, aggregator
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Universal approximation theorem (UAT)
- 다음 문서 (Ch4-01, Ch4-03): 1-WL test, GIN 의 1-WL 도달 정리

---

## 📖 직관적 이해

### Multiset Function 으로서의 Aggregator

이웃 feature $\{h_j : j \in N(i)\}$ 는 **multiset** (순서 무관, 중복 가능). Aggregator 는 multiset → vector 함수.

**Multiset injectivity**: $f(\{a, b, c\}) \neq f(\{a, b, d\})$ for $c \neq d$. 즉 다른 multiset 은 다른 결과.

- **Sum injective?** Multiset $\{1, 2\}$ vs $\{1.5, 1.5\}$: sum 둘 다 3 — sum 이 직접적으로는 injective X.
- 그러나 **input domain 이 countable (이산)** 이면 sum 이 universal multiset function — 이것이 GIN 의 핵심 (Xu 2019 Lemma 5).

### Mean / Max 의 한계

- **Mean**: $\{1, 2\}$ 와 $\{1.5\}$, 또는 $\{1, 1, 2, 2\}$ 와 $\{1, 2\}$ → 모두 mean 1.5 → 구분 X
- **Max**: $\{1, 2\}$ 와 $\{2\}$ → 모두 max 2 → 구분 X

이 반례들이 mean/max 가 multiset injective 가 아님을 정확히 보임.

### MLP 의 역할

UAT (Cybenko 1989, Hornik 1991): MLP 가 continuous function 의 universal approximator. 따라서 **MLP after sum** 가 multiset → output 의 **임의 injective function** 근사 가능 → **multiset universal** (Xu 2019 Theorem 3).

### $\epsilon$ 의 의미

$h_i^{(l+1)} = \text{MLP}((1 + \epsilon) h_i^{(l)} + \sum_{j \in N(i)} h_j^{(l)})$

$\epsilon$ = **자기 자신과 이웃의 weight ratio** 를 학습 가능하게:
- $\epsilon = 0$: 자기 정보 1 + 이웃 sum
- $\epsilon > 0$: 자기 정보 강조
- $\epsilon < 0$: 이웃 강조 (rare)

GIN-$\epsilon$: $\epsilon$ 학습. GIN-0: $\epsilon = 0$ 고정 (실증적으로 거의 비슷).

이 ratio 가 multiset 의 **distinct** 한 경우 ($h_i$ 가 이웃 multiset 에 포함되는지 여부) 를 구분 가능하게 함.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — GIN Layer

$$
h_i^{(l+1)} = \text{MLP}^{(l)}\left( (1 + \epsilon^{(l)}) h_i^{(l)} + \sum_{j \in N(i)} h_j^{(l)} \right)
$$

- $\epsilon^{(l)}$: learnable scalar (또는 fixed = 0)
- $\text{MLP}^{(l)}$: 2-layer MLP with ReLU between

### 정의 4.2 — GIN-$\epsilon$ vs GIN-0

- **GIN-$\epsilon$**: $\epsilon$ 을 learnable
- **GIN-0**: $\epsilon = 0$ 고정 (단순화, 실증 비슷)

### 정의 4.3 — Graph-level Readout

Graph classification 에서 graph-level representation:
$$
h_G = \text{CONCAT}\left( \text{READOUT}(\{h_v^{(l)} : v \in V\}) : l = 0, 1, \ldots, L \right)
$$

(모든 layer 의 sum-pool 을 concat — Jumping Knowledge-style, Ch5-05)

READOUT: sum (GIN 권장, multiset injective). Mean/max 도 가능하지만 표현력 ↓.

### 정의 4.4 — Multiset Function 과 Injectivity

함수 $f: \mathcal X^* \to \mathbb R^d$ 가 multiset injective:
$$
f(\{\!\{a_1, \ldots, a_n\}\!\}) = f(\{\!\{b_1, \ldots, b_m\}\!\}) \Rightarrow \{\!\{a_i\}\!\} = \{\!\{b_j\}\!\}
$$

(같은 output 이면 같은 multiset)

---

## 🔬 정리와 결과

### 정리 4.1 — Sum 이 Multiset Injective on Countable Domain (Xu 2019 Lemma 5)

**Theorem**: Countable set $\mathcal X$ 에 대해, 함수 $f: \mathcal X \to \mathbb R^N$ ($N$ 충분히 큼) 가 존재하여:
$$
\sum_{x \in S} f(x) \neq \sum_{x \in S'} f(x) \quad \forall S \neq S' \text{ multiset of } \mathcal X
$$

즉 sum aggregator 가 multiset injective (with proper $f$).

**증명 sketch**: $\mathcal X$ countable → bijection $\phi: \mathcal X \to \mathbb N$. $f(x) = (1, \phi(x), \phi(x)^2, \ldots, \phi(x)^N) / N!$ 같은 polynomial moment vector 사용 — $N$ 차원 polynomial moment 가 $N$-element multiset 을 unique 하게 결정 (Vandermonde). 자세한 증명 Xu 2019 Appendix.

**의미**: $f$ 를 학습 가능 MLP 로 두면 sum + MLP = universal multiset injective 함수.

### 정리 4.2 — Mean/Max 가 Sum 보다 Weak

**Counter-example for Mean**:
- $S_1 = \{1, 1\}$, $S_2 = \{1\}$: mean 둘 다 1 — 구분 X
- 단 sum: 2 vs 1 — 구분 O

**Counter-example for Max**:
- $S_1 = \{1, 2\}$, $S_2 = \{2\}$: max 둘 다 2 — 구분 X
- 단 sum: 3 vs 2 — 구분 O

**Theorem**: Mean / Max aggregator 는 multiset injective 가 아님 (countable domain 에서도).

이로부터 mean (GraphSAGE, GCN-like) 와 max (GraphSAGE-pool) 가 sum 보다 strict 표현력 약함.

### 정리 4.3 — GIN 의 1-WL 등가 표현력

(자세한 증명 Ch4-03)

**Theorem (Xu 2019 Theorem 3)**: GIN 은 1-WL 과 등가 표현력 — 두 그래프가 1-WL 로 구분되면 GIN 으로도 구분 가능, 그 역도 성립.

**의미**: GIN 이 message passing GNN 의 **이론적 상한** — 더 강한 GNN 은 더 정밀한 graph isomorphism test 필요 (k-WL, Ch4-04).

### 정리 4.4 — GIN-$\epsilon$ 의 추가 표현력

**Theorem (Xu 2019 informal)**: GIN-$\epsilon$ ($\epsilon$ learnable) 가 GIN-0 보다 strict 더 강함 in some cases. 단 실전 차이는 미미.

**예**: 노드 자체가 이웃 multiset 에 등장하지 않는 형태 — $\epsilon \neq 0$ 으로 self-loop weight 와 분리 가능.

### 정리 4.5 — Graph-level Concat Readout 의 정당화

Layer 마다 다른 hop 정보 보존. Sum-readout 이 multiset injective 이므로 모든 layer concat 이 hierarchical graph representation:
$$
h_G = [\text{sum}(\{h_v^{(0)}\}), \text{sum}(\{h_v^{(1)}\}), \ldots, \text{sum}(\{h_v^{(L)}\})]
$$

이것이 GIN 의 graph classification 우위의 핵심.

---

## 💻 구현

### 실험 1 — GIN Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class GINLayer(nn.Module):
    def __init__(self, d_in, d_out, eps=0.0, train_eps=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(),
            nn.Linear(d_out, d_out))
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer('eps', torch.tensor(eps))
    
    def forward(self, x, edge_index):
        src, dst = edge_index
        # sum aggregation of neighbors
        agg = scatter_add(x[src], dst, dim=0, dim_size=x.size(0))
        # GIN update: MLP((1+ε) x_i + sum_neighbors)
        return self.mlp((1 + self.eps) * x + agg)

class GIN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers=3, train_eps=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(d_in, d_hid, train_eps=train_eps))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(d_hid, d_hid, train_eps=train_eps))
        # Graph-level: sum over all layers + linear
        self.cls = nn.Linear(d_hid * num_layers, d_out)
    
    def forward(self, x, edge_index, batch=None):
        layer_outs = []
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
            layer_outs.append(h)
        # Graph-level readout (sum over each layer)
        if batch is None:
            graph_repr = torch.cat([h.sum(0) for h in layer_outs])
            return self.cls(graph_repr.unsqueeze(0))
        else:
            from torch_scatter import scatter_add
            graph_reprs = [scatter_add(h, batch, dim=0) for h in layer_outs]
            graph_repr = torch.cat(graph_reprs, dim=-1)
            return self.cls(graph_repr)
```

### 실험 2 — Multiset Injectivity 실증

```python
# Sum vs Mean 의 multiset 구분
multisets = [
    [1.0, 1.0],
    [1.0],
    [1.0, 1.0, 2.0, 2.0],
    [1.0, 2.0],
    [1.5, 1.5],
]

for ms in multisets:
    t = torch.tensor(ms)
    print(f'{ms}: sum={t.sum().item():4.1f}, mean={t.mean().item():4.2f}, max={t.max().item():4.1f}')
```

**예상**:
```
[1.0, 1.0]:                sum= 2.0, mean=1.00, max=1.0
[1.0]:                     sum= 1.0, mean=1.00, max=1.0   <- mean/max 같음
[1.0, 1.0, 2.0, 2.0]:      sum= 6.0, mean=1.50, max=2.0
[1.0, 2.0]:                sum= 3.0, mean=1.50, max=2.0   <- mean/max 같음
[1.5, 1.5]:                sum= 3.0, mean=1.50, max=1.5   <- mean 같음
```

### 실험 3 — Karate Club Node Classification

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
n = G.number_of_nodes()
edges = np.array(list(G.edges())).T
ei = torch.tensor(np.concatenate([edges, edges[::-1]], axis=1), dtype=torch.long)

X = torch.eye(n)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True

# GIN node classifier (마지막 layer 만 사용, no graph pool)
class GINNodeClassifier(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(d_in, d_hid))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(d_hid, d_hid))
        self.cls = nn.Linear(d_hid, d_out)
    
    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
        return self.cls(h)

model = GINNodeClassifier(d_in=n, d_hid=8, d_out=2, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
for epoch in range(200):
    model.train(); optimizer.zero_grad()
    out = model(X, ei)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward(); optimizer.step()

acc = (model(X, ei).argmax(-1) == labels).float().mean().item()
print(f'GIN node classification accuracy: {acc:.2%}')
```

### 실험 4 — TU Dataset Graph Classification (간략)

```python
try:
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    
    dataset = TUDataset(root='./data/MUTAG', name='MUTAG')
    train_loader = DataLoader(dataset[:150], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset[150:], batch_size=32)
    
    print(f'MUTAG: {len(dataset)} graphs, '
          f'avg nodes={dataset.data.x.shape[0] // len(dataset):.1f}, '
          f'classes={dataset.num_classes}')
    
    # 학습 코드 (간략)
    model = GIN(d_in=dataset.num_features, d_hid=32, d_out=dataset.num_classes, num_layers=3, train_eps=True)
    # ... train loop
except (ImportError, RuntimeError):
    print('PyG/MUTAG not available')
```

### 실험 5 — WL-fail Graph Pair (CSL 형태)

```python
# Circulant Skip Link (CSL): 1-WL 동등 but non-isomorphic
# 단순화: 두 cycle C_8 with different "skip" → 같은 degree distribution
def build_csl(n, skip):
    """C_n with skip-edge"""
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + skip) % n)
    return G

G1 = build_csl(8, 2)   # skip 2
G2 = build_csl(8, 3)   # skip 3

# 두 그래프 isomorphic? 
print(f'G1, G2 isomorphic: {nx.is_isomorphic(G1, G2)}')

# Degree distribution 동일 확인
print(f'G1 degrees: {sorted([d for _, d in G1.degree()])}')
print(f'G2 degrees: {sorted([d for _, d in G2.degree()])}')

# 1-WL 로 구분 가능?
# (실제로 CSL with skip 2 vs 3 은 1-WL 동등) — Ch4-04 에서 자세히
```

---

## 🔗 실전 활용

### 1. TU Dataset SOTA

GIN 의 표준 결과 (Xu 2019):
- MUTAG: 89.4%
- PROTEINS: 76.2%
- IMDB-B: 75.1%
- COLLAB: 80.2%

GCN/GraphSAGE/GAT 보다 평균 2~5% 높음.

### 2. OGB-molhiv

GIN 이 분자 성질 예측에서 강력. 단, edge feature 가 중요한 경우 GIN-edge (edge feature 통합) 또는 후속 DGN, MPNN-edge 가 우월.

### 3. PyG `GINConv`

```python
from torch_geometric.nn import GINConv

mlp = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_out))
conv = GINConv(mlp, train_eps=True)
out = conv(x, edge_index)
```

### 4. Theoretical Foundation

GIN 이 **1-WL 표현력 상한 도달** 사실은 GNN 이론의 결정적 증명. 후속 연구의 중심:
- k-GNN (Morris 2019): GIN 의 k-WL 일반화
- Provably Powerful Graph Networks (Maron 2019): k-FGNN
- Position-aware GNN: WL 우회 (Ch4-05)

### 5. GIN-edge with Edge Features

원본 GIN 은 edge feature 처리 X. GINEConv (PyG) 가 edge feature 추가:
$$
h_i^{(l+1)} = \text{MLP}((1+\epsilon) h_i + \sum_{j} \text{ReLU}(h_j + e_{ij}))
$$

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Countable feature domain | Continuous feature 시 sum injectivity 정확하지 않음 (approx) |
| Sum aggregator | Sum 의 magnitude scale 이 degree 에 sensitive — degree-aware norm 필요 시 mean 도 |
| 1-WL 상한 | Strongly regular graph (CSL, Paley) 구분 못함 → k-WL, position-aware 필요 |
| MLP universal approx | MLP depth/width 의 실제 표현력은 sample-dependent |
| Self-loop 명시 X | $\epsilon \neq 0$ 으로 self-loop weight 학습 |
| Edge feature 명시 X | GINEConv 등 별도 변종 필요 |

---

## 📌 핵심 정리

$$\boxed{h_i^{(l+1)} = \text{MLP}\left((1 + \epsilon) h_i^{(l)} + \sum_{j \in N(i)} h_j^{(l)}\right)}$$

$$\boxed{\text{sum + MLP} = \text{multiset universal injective function (countable)}}$$

| 항목 | GIN |
|------|-----|
| **Aggregator** | Sum (key) |
| **Update** | MLP |
| **Self vs neighbor** | $(1+\epsilon)$ 로 weight 학습 (or $\epsilon = 0$) |
| **표현력** | 1-WL (이론 상한 달성) |
| **Graph readout** | All-layer sum concat |
| **Mean/Max 와의 차이** | Mean / Max 는 multiset injective X — strict less expressive |
| **TU 벤치마크** | GCN/GAT 대비 +2~5% |
| **이론적 의의** | Message passing GNN 의 최강 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 두 multiset $\{1, 2, 3\}$ 와 $\{6\}$ 를 sum / mean / max 로 비교하라. 어느 aggregator 가 구분 가능한가?

<details>
<summary>해설</summary>

- **Sum**: $1+2+3 = 6$, $6 = 6$ — **같음** (구분 X)
- **Mean**: $6/3 = 2$, $6/1 = 6$ — 다름 (구분 O)
- **Max**: $\max\{1,2,3\} = 3$, $\max\{6\} = 6$ — 다름 (구분 O)

**의미**: Sum 도 multiset 의 size 정보가 있을 때 (sum scale 차이) 도움됨. 이는 sum 이 size 정보를 자동 포함하지 않는 단점.

**GIN 의 보완**: $(1+\epsilon) h_i + \sum_{j} h_j$ 에서 $h_i$ 의 weight 가 size 정보 implicit.

또한 **GIN 의 sum injectivity** 는 학습된 MLP $\phi$ 가 input 을 적절히 transform 후 sum 의 injectivity 를 활용 — raw input 의 sum 이 아닌, $\phi(x_i)$ 의 sum 이 injective 보장.

따라서 raw input 만으로 sum 이 항상 좋은 건 아님 — **MLP transform 후 sum 이 핵심**. $\square$

</details>

**문제 2** (심화): GIN 의 $\epsilon = 0$ 이 어떤 경우에 표현력 손실인지 구체적 반례를 들어라.

<details>
<summary>해설</summary>

**$\epsilon = 0$ vs $\epsilon \neq 0$ 차이**:

$\epsilon = 0$: $\text{MLP}(h_i + \sum_j h_j)$ — $h_i$ 가 이웃 multiset 에 포함되는지 여부 구분 어려움.

**Counter-example**:

Graph 1: 노드 $i$ with $h_i = 1$, neighbors $\{1, 1\}$ ($\sum = 2$, $h_i + \sum = 3$)
Graph 2: 노드 $i'$ with $h_{i'} = 2$, neighbors $\{1\}$ ($\sum = 1$, $h_{i'} + \sum = 3$)

$\epsilon = 0$ 시 MLP input 같음 → 같은 output. **두 다른 graph 구분 X**.

$\epsilon \neq 0$ 시:
- $(1 + \epsilon) \cdot 1 + 2 = 3 + \epsilon$
- $(1 + \epsilon) \cdot 2 + 1 = 3 + 2\epsilon$
- 다름 → 구분 O.

**의미**: $\epsilon$ 이 self / neighbor weight 의 ratio 를 학습 — multiset 에서 self 의 unique 한 contribution 추출.

실전에서 이런 경우는 드물지만 (대부분 multiset 자체가 다름), 이론적으로 GIN-$\epsilon$ 가 GIN-0 보다 strict 강함.

</details>

**문제 3** (논문 비평): GIN 이 "이론상 최강 message passing" 이라는 주장이 graph classification (TU dataset) 에서는 실증되지만, node classification (Cora) 에서는 GAT/GraphSAGE 와 큰 차이 없는 이유를 분석하라.

<details>
<summary>해설</summary>

**Graph classification 에서 GIN 우위의 이유**:

1. **Multiset injectivity**: graph-level representation 이 노드 multiset 의 sum-pool. GIN 의 sum aggregator + sum readout 이 multiset injective — 다른 graph 를 다른 representation 으로.

2. **Hierarchical concat readout**: GIN 의 layer-wise concat readout 이 hierarchical (1-hop, 2-hop, … L-hop) graph statistics 를 보존.

3. **TU dataset 의 특성**: 대부분 분자 (atom-level discrete feature). Sum 이 자연스럽게 valid (countable feature, GIN 의 sum injectivity 가정 만족).

**Node classification 에서 차이가 작은 이유**:

1. **Local information dominant**: Cora 등 citation network 에서 노드 분류는 1~2-hop neighbor 만으로 충분. GCN, GAT 도 충분한 local info.

2. **Continuous feature**: Cora 의 word-bag feature 는 continuous-like — sum injectivity 의 정확한 가정 ↑. 단, 이 경우에도 sum 의 일반적 우위 유지.

3. **Mean's degree-normalization**: Cora 의 hub node (high-degree) 가 mean aggregator 로 정규화 시 더 안정적 학습. GIN 의 sum 은 degree-sensitive — high-degree node 의 magnitude 폭발 위험.

4. **Smoothing 효과**: Node classification 에서 약간의 smoothing (mean) 이 오히려 도움. GIN 의 sharp injectivity 가 noise 에 민감.

5. **Train/Test split 의 transductive 성격**: Cora 의 train/test 가 같은 graph — local pattern 만으로 분류 가능, graph isomorphism-like 표현력 불필요.

**결론**:

- **Graph classification**: GIN 우위 (multiset injectivity 가 graph-level 표현력 결정)
- **Node classification**: 차이 미미 (local pattern 이 dominant, smoothing 이 도움 됨)

이것이 GIN 이 "graph classification 의 표준 baseline" 이지만 node classification 에서는 GCN/GAT 와 함께 다 사용되는 이유.

**Heterophilic node classification** 에서는 GIN 의 sum 이 over-smoothing 회피 측면에서 유리. 따라서 task-dependent.

</details>

---

<div align="center">

[◀ 이전](./03-gat.md) | [📚 README](../README.md) | [다음 ▶](./05-heterogeneous.md)

</div>
