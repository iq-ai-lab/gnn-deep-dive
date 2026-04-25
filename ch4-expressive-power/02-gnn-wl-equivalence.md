# 02. GNN과 1-WL의 동등성 (Xu 2019)

## 🎯 핵심 질문

- "Message Passing GNN 의 표현력 ≤ 1-WL" 이 정확히 무엇을 의미하는가?
- Xu et al. 2019 의 정리는 어떻게 induction 으로 증명되는가?
- 따름정리: GNN 이 어떤 graph 쌍을 구분할 수 있는지 정확히 결정 가능?
- CSL, Paley graph 처럼 1-WL 동등 그래프에서 GNN 도 구분 못함이 실제로 검증되는가?
- Spectral GCN, GAT, GIN, GraphSAGE 모두 1-WL 상한이 적용되는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GNN 이 모든 graph problem 을 풀 수 있을 것 같지만, **이론적으로 명확한 한계** 가 있습니다. Xu et al. (2019) 의 결정적 정리:

**모든 message passing GNN $\phi$ 와 모든 graph $G_1, G_2$ 에 대해:**
$$
G_1 \stackrel{1\text{-WL}}{\equiv} G_2 \Rightarrow \phi(G_1) = \phi(G_2)
$$

즉, **1-WL 이 구분 못하는 graph 는 GNN 도 구분 못함** — 1-WL 이 GNN 표현력의 **upper bound**.

이 정리는 GNN 표현력 이론의 출발점:
1. **Negative result**: GNN 이 못하는 task 의 정확한 경계
2. **Positive guidance**: GIN 이 1-WL 을 정확히 도달하므로 (Ch4-03) 최강 message passing
3. **새 GNN 동기**: k-WL, position-aware 가 1-WL 을 초과하기 위함 (Ch4-04, Ch4-05)

---

## 📐 수학적 선행 조건

- 이전 문서: [01-wl-test.md](./01-wl-test.md) — 1-WL algorithm
- [Ch3-01](../ch3-message-passing/01-mpnn-framework.md) — MPNN framework
- 수학 induction, multiset 함수

---

## 📖 직관적 이해

### Color = Hidden State

1-WL 의 color $c_i^{(l)}$ 와 GNN 의 hidden state $h_i^{(l)}$ 의 평행:

| 1-WL | GNN |
|------|-----|
| $c_i^{(l)}$ | $h_i^{(l)}$ |
| Hash function | Update function $U$ |
| Multiset of neighbor colors | Aggregate of neighbor states |
| Color refinement | Layer-wise propagation |

핵심: **"color" 와 "hidden state" 가 정보적으로 동치** — GNN 이 1-WL 보다 더 정밀한 결과 생성 불가.

### Induction 의 직관

Layer $l$ 에서 만약 $h_i^{(l)} = h_{j}^{(l)}$ ($i \in V_1$, $j \in V_2$) 가 1-WL color 동등에 의해 보장되면 ($c_i^{(l)} = c_j^{(l)}$ in WL), step $l+1$ 도 마찬가지:
- $h_i^{(l+1)} = U(h_i^{(l)}, \bigoplus h_k^{(l)})$, $k \in N(i)$
- 1-WL multiset $\{\!\{c_k^{(l)} : k \in N(i)\}\!\}$ 동등 ⟹ $\{\!\{h_k^{(l)}\}\!\}$ 동등 ⟹ $\bigoplus$ 동등 ⟹ $h_i^{(l+1)} = h_j^{(l+1)}$

이 induction 의 핵심.

### Necessary vs Sufficient

- **1-WL 동등 ⟹ GNN 동등** (정리 2.1): 1-WL 이 못 구분하는 건 GNN 도 못 구분
- 역명제 ⟹ "1-WL 가 다른 결과 시 GNN 이 구분 가능?" — 이는 GIN-같은 maximally expressive GNN 이면 가능 (Ch4-03)
- 일반적 GNN (mean / max aggregator) 에는 strict 부등호 — GCN 이 1-WL 보다도 약함 가능

---

## ✏️ 엄밀한 정의 및 정리

### 정의 2.1 — Message Passing GNN (formal)

$L$-layer GNN $\phi$:
- $h_i^{(0)} = X_i$ (input feature)
- $h_i^{(l+1)} = U_l(h_i^{(l)}, \bigoplus_{j \in N(i)} M_l(h_i^{(l)}, h_j^{(l)}, e_{ij}))$
- Final: $\phi(G) = R(\{h_v^{(L)} : v \in V\})$ (graph-level) or $\phi(G)_i = h_i^{(L)}$ (node-level)

여기서 $M, U, R$ 모두 measurable functions (continuous neural network).

### 정의 2.2 — GNN Equivalence

두 graph $G_1, G_2$ 가 GNN $\phi$ 로 distinguished:
$$
\phi(G_1) \neq \phi(G_2)
$$

GNN $\phi$ 가 graph property $P$ 를 학습 가능: $\phi$ 가 $P$ 가 true / false 인 그래프를 구분.

### 정리 2.3 — 1-WL은 GNN 표현력의 상한 (Xu 2019 Theorem 1)

**Theorem**: Message passing GNN $\phi$ 와 $G_1, G_2$ 에 대해, 만약 $G_1 \stackrel{1\text{-WL}}{\equiv} G_2$ 이면 $\phi(G_1) = \phi(G_2)$.

**증명** (induction on layer $l$):

**Base** ($l = 0$): $h_i^{(0)} = X_i$. 1-WL initial label 도 input feature 또는 default. 따라서 same multiset of initial labels in two graphs ⟹ same multiset of $h^{(0)}$.

**Strong induction**: 모든 $l' \leq l$ 에서 1-WL color partition 이 같으면 multiset of hidden states $\{\!\{h_v^{(l')}\}\!\}$ 가 같다고 가정.

**Step $l \to l+1$**: 1-WL 의 step $l \to l+1$:
$$
c_i^{(l+1)} = \text{hash}(c_i^{(l)}, \{\!\{c_j^{(l)} : j \in N(i)\}\!\})
$$

만약 $G_1, G_2$ 의 1-WL stable color partition 이 같으면, 각 cell $C$ 에 대해 $i \in C \cap V_1$ 와 $j \in C \cap V_2$ 가 같은 신경 표현 가짐 (IH).

GNN step:
$$
h_i^{(l+1)} = U_l\left(h_i^{(l)}, \bigoplus_{k \in N(i)} M_l(h_i^{(l)}, h_k^{(l)})\right)
$$

$i$ 의 1-WL color = $j$ 의 1-WL color ⟹ neighbor multisets equal:
$$
\{\!\{c_k^{(l)} : k \in N_{G_1}(i)\}\!\} = \{\!\{c_{k'}^{(l)} : k' \in N_{G_2}(j)\}\!\}
$$

⟹ $\{\!\{h_k^{(l)} : k \in N_{G_1}(i)\}\!\} = \{\!\{h_{k'}^{(l)} : k' \in N_{G_2}(j)\}\!\}$ (by IH).

$\bigoplus$ 가 multiset function (permutation-invariant) ⟹ same aggregate. 

$U_l$ deterministic + $h_i^{(l)} = h_j^{(l)}$ ⟹ $h_i^{(l+1)} = h_j^{(l+1)}$. $\square$

**Corollary**: $\phi(G_1) = R(\{h_v^{(L)}: v \in V_1\}) = R(\{h_v^{(L)}: v \in V_2\}) = \phi(G_2)$ — 모든 layer 의 multiset 같음 ⟹ readout 같음.

### 정리 2.4 — Strict Inequality Cases

**Theorem (informal)**: Some GNN are strictly weaker than 1-WL:
- Mean aggregator (GCN, GraphSAGE-mean) — multiset injectivity 없음 (Ch3-04)
- Max aggregator (GraphSAGE-pool) — 마찬가지

이들 GNN 은 1-WL 가 구분하는 일부 graph 를 구분 못함 (counter-examples in Ch3-04).

### 정리 2.5 — Sum Aggregator + MLP = 1-WL Tight

GIN 이 1-WL 을 정확히 도달 (Ch4-03 Theorem). 즉:
$$
\text{GIN} \equiv 1\text{-WL} \quad \text{(in distinguishing power)}
$$

따라서 **GIN 이 message passing GNN 의 maximum expressive**.

### 정리 2.6 — Empirical Implication

CSL, Paley graph 등 1-WL 동등 그래프 쌍에서 모든 message passing GNN 이 동일 output → graph classification 시 random guess.

**해결**:
- k-WL (Ch4-04)
- Position-aware GNN (Ch4-05)
- Subgraph GNN (Bevilacqua 2022)

---

## 💻 구현 검증

### 실험 1 — CSL 그래프에서 GIN 이 구분 못함 검증

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import networkx as nx
import numpy as np

def build_csl(n, skip):
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + skip) % n)
    return G

def graph_to_pyg(G):
    edges = np.array(list(G.edges())).T
    edge_index = torch.tensor(np.concatenate([edges, edges[::-1]], axis=1), dtype=torch.long)
    n = G.number_of_nodes()
    x = torch.ones(n, 1)   # 모든 노드 같은 feature (uninformative input)
    return x, edge_index

# GIN 정의 (Ch3-04)
class GINLayer(nn.Module):
    def __init__(self, d_in, d_out, eps=0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out))
        self.eps = eps
    def forward(self, x, edge_index):
        src, dst = edge_index
        agg = scatter_add(x[src], dst, dim=0, dim_size=x.size(0))
        return self.mlp((1 + self.eps) * x + agg)

class GIN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(d_in, d_hid))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(d_hid, d_hid))
        self.cls = nn.Linear(d_hid * num_layers, d_out)
    def forward(self, x, edge_index):
        outs = []
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
            outs.append(h.sum(0))   # graph-level sum readout
        return self.cls(torch.cat(outs))

# CSL(8, 2) vs CSL(8, 3): 1-WL 동등 (모든 노드 같은 색)
G1 = build_csl(8, 2)
G2 = build_csl(8, 3)
x1, ei1 = graph_to_pyg(G1)
x2, ei2 = graph_to_pyg(G2)

# 학습 없이 random init GIN 으로 graph-level repr 비교
torch.manual_seed(0)
model = GIN(d_in=1, d_hid=16, d_out=8, num_layers=3)
model.eval()
with torch.no_grad():
    z1 = model(x1, ei1)
    z2 = model(x2, ei2)
print(f'GIN output diff (CSL 8,2 vs 8,3): {(z1 - z2).norm().item():.6e}')
# Expected: 매우 작음 (numerical noise) — GIN 도 구분 X
```

### 실험 2 — 1-WL 와 GIN 의 동시 검증

```python
from collections import Counter

def wl_step_simple(adj, labels):
    new_labels = []
    for i in range(len(labels)):
        neighbor_labels = sorted(labels[adj[i]].tolist())
        new_labels.append(hash((labels[i].item(), tuple(neighbor_labels))))
    # Canonicalize
    unique = sorted(set(new_labels))
    mapping = {v: i for i, v in enumerate(unique)}
    return torch.tensor([mapping[l] for l in new_labels])

def wl_distinguish(G1, G2, max_iter=10):
    """Returns True if 1-WL distinguishes."""
    A1 = nx.adjacency_matrix(G1).toarray() > 0
    A2 = nx.adjacency_matrix(G2).toarray() > 0
    l1 = torch.zeros(len(A1), dtype=torch.long)
    l2 = torch.zeros(len(A2), dtype=torch.long)
    for _ in range(max_iter):
        l1 = wl_step_simple(A1, l1)
        l2 = wl_step_simple(A2, l2)
    return Counter(l1.tolist()) != Counter(l2.tolist())

print(f'1-WL distinguishes CSL(8,2) and CSL(8,3): {wl_distinguish(G1, G2)}')
# False (1-WL 동등)
```

### 실험 3 — 1-WL 가 구분하는 그래프에서 GIN 도 구분

```python
# C_5 vs P_5 (1-WL 가 구분 — Ch4-01 문제 1)
G_c = nx.cycle_graph(5)
G_p = nx.path_graph(5)

print(f'1-WL distinguishes C_5, P_5: {wl_distinguish(G_c, G_p)}')   # True

# GIN 으로 구분 가능 확인
x_c, ei_c = graph_to_pyg(G_c)
x_p, ei_p = graph_to_pyg(G_p)

torch.manual_seed(0)
model = GIN(d_in=1, d_hid=16, d_out=8, num_layers=3)
model.eval()
with torch.no_grad():
    z_c = model(x_c, ei_c)
    z_p = model(x_p, ei_p)
print(f'GIN diff C_5 vs P_5: {(z_c - z_p).norm().item():.4f}')
# 양수 → 구분 가능
```

### 실험 4 — Mean aggregator 의 strict weakness

```python
# Multiset {1,1} vs {1}: mean 같음, sum 다름
# 노드 의미: A 그래프 노드가 {1,1} 이웃 vs B 그래프 노드가 {1} 이웃

class MeanGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
    def forward(self, x, edge_index):
        from torch_scatter import scatter_mean
        src, dst = edge_index
        agg = scatter_mean(x[src], dst, dim=0, dim_size=x.size(0))
        return self.lin(agg)

# Graph A: 1 node with 2 neighbors of feature 1
# Graph B: 1 node with 1 neighbor of feature 1
# 단순 1-hop GNN 비교

xA = torch.tensor([[0.], [1.], [1.]])   # node 0 의 이웃 1, 2
eiA = torch.tensor([[1, 2, 0, 0], [0, 0, 1, 2]], dtype=torch.long)

xB = torch.tensor([[0.], [1.]])   # node 0 의 이웃 1
eiB = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)

mean_gnn = MeanGNN(1)
zA_mean = mean_gnn(xA, eiA)[0]
zB_mean = mean_gnn(xB, eiB)[0]
print(f'Mean GNN: A={zA_mean.item():.4f}, B={zB_mean.item():.4f} → {"different" if abs(zA_mean - zB_mean) > 1e-6 else "SAME (mean cant distinguish)"}')
# Same (mean of {1,1} = mean of {1})
```

### 실험 5 — GNN 수렴 후 1-WL 가 구분 못하는 그래프

```python
# 학습 후에도 GIN 이 CSL 구분 못함 확인
torch.manual_seed(42)

# Toy classification: G1 = label 0, G2 = label 1
def make_dataset():
    return [
        (G1, 0), (G2, 1)
    ] * 50

dataset = make_dataset()
model = GIN(d_in=1, d_hid=32, d_out=2, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = 0
    for G, y in dataset:
        x, ei = graph_to_pyg(G)
        out = model(x, ei).unsqueeze(0)
        loss = F.cross_entropy(out, torch.tensor([y]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

# 학습 후 두 그래프 representation 비교
model.eval()
with torch.no_grad():
    z1 = model(graph_to_pyg(G1)[0], graph_to_pyg(G1)[1])
    z2 = model(graph_to_pyg(G2)[0], graph_to_pyg(G2)[1])
print(f'After training, GIN diff CSL(8,2) vs CSL(8,3): {(z1 - z2).norm().item():.6e}')
# 여전히 매우 작음 → 학습 안됨
```

---

## 🔗 실전 활용

### 1. New GNN 모델 평가

새 GNN 제안 시 standard checks:
1. **EXP / EXP-classify** dataset (Abboud 2021): WL-fail 그래프 explicit collection
2. **CSL benchmark** (Murphy 2019): 다양한 skip 의 CSL — graph classification
3. **SR25** dataset: 25-node strongly regular graphs

각 dataset 에서 random guess (50%) 가 1-WL 한계.

### 2. Heterophilic Graph

1-WL 가정 ("이웃과 비슷") 이 깨지는 graph (Wikipedia 등) 에서 1-WL 표현력으로도 부족 — 새 inductive bias 필요.

### 3. Theoretical Analysis 의 Foundation

GNN paper 의 표준 contribution:
- "Our model is provably more expressive than 1-WL"
- "Our model achieves $k$-WL expressive power"

이런 statement 를 이해/검증하려면 본 정리 2.3 을 정확히 이해해야 함.

### 4. PyG Benchmarks

PyG, DGL 의 ZINC, OGB-molhiv 등 분자 dataset — chemistry 의 graph classification. Sub-1-WL graph 가 적어 1-WL 한계 marginally 영향.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Permutation invariant aggregator | 비-permutation-invariant (LSTM) 는 다른 위계 |
| Continuous neural function | Discretized hash = 1-WL 정확 |
| Single graph 위 정의 | Multi-graph (subgraph) GNN 은 별도 |
| Static graph | Dynamic, temporal GNN 은 시간축 추가 |
| Node-level / graph-level 둘 다 | Edge prediction 은 별도 분석 필요 |

---

## 📌 핵심 정리

$$\boxed{G_1 \stackrel{1\text{-WL}}{\equiv} G_2 \Rightarrow \phi(G_1) = \phi(G_2) \quad \text{for all message passing GNN } \phi}$$

| 함의 |
|------|
| 1-WL 동등 그래프 (CSL, Paley) → 모든 message passing GNN 같은 output |
| GIN 이 정확히 1-WL 도달 (Ch4-03) — message passing 의 최강 |
| Mean / Max aggregator 는 strict 약함 (GCN, GraphSAGE-pool) |
| 1-WL 우회: k-WL, position-aware, subgraph GNN |
| Real-world graph 의 대부분은 1-WL 로 충분 — 단 highly symmetric 에 한계 |
| 따라서 표현력 vs 일반화 trade-off |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GCN 이 1-WL 보다 strictly 약함을 multiset 반례로 보여라.

<details>
<summary>해설</summary>

**GCN aggregator**: 정규화 weighted mean — $h_i^{(l+1)} = \sigma(\sum_j \frac{1}{\sqrt{d_i d_j}} h_j W)$. 이는 mean 의 변형 (degree-normalized).

**Multiset 반례 ({1,1} vs {1})**:

같은 graph structure (단일 이웃) 가정. 노드 $i$ 가 이웃 multiset:
- Graph A: 두 이웃 (각 feature 1)
- Graph B: 한 이웃 (feature 1)

GCN aggregation:
- A: $\frac{1}{\sqrt{d_i \cdot 1}} (1 + 1) = 2/\sqrt{d_i}$ — 단 $d_i = 2$ 가정 시 $2/\sqrt 2 = \sqrt 2$
- B: $\frac{1}{\sqrt{1 \cdot 1}} \cdot 1 = 1$

다른 결과. **GCN 가 구분**! 단 이건 graph 의 degree 차이로 인한 것 (degree-aware normalization).

**더 정확한 반례**: 같은 degree 같은 graph structure 에서 mean-collapse 하는 multiset.

예: 노드 $i$ 이웃의 multiset $\{a, b\}$ vs $\{c, d\}$ where $a + b = c + d$ but $\{a, b\} \neq \{c, d\}$ as multisets. 

GCN 의 normalized mean: $(a+b)/(2\sqrt{d_i d_j}) = (c+d)/(2\sqrt{d_i d_j})$ → **같음** → GCN 구분 X.

GIN sum: $a + b = c + d$ → 그러면 sum 도 안됨. 단 이는 가정에서 sum = sum 인 경우.

**다른 반례**: Multiset $\{1, 2, 3\}$ vs $\{2, 2, 2\}$ — sum 둘 다 6, mean 둘 다 2. GIN 도 이 두 multiset 의 sum 이 같음 → 단 input 이 통과한 MLP 결과 의 sum 이 같으리란 보장 X. 

핵심: **Sum 자체 같다고 multiset 같지 않음**. Sum + MLP 가 다르게 학습되면 구분 가능.

따라서 GCN 의 mean 은 multiset injectivity 없음 strict 결론, GIN 의 sum + MLP 는 universal multiset injective.

</details>

**문제 2** (심화): 정리 2.3 의 induction 에서 "multiset of $h^{(l)}$ in $G_1$ = multiset in $G_2$" 가 strictly 필요한 가정인가? 더 약한 condition 으로 같은 결과 도출 가능?

<details>
<summary>해설</summary>

**Strict requirement of multiset equality**:

두 그래프 $G_1, G_2$ 의 1-WL color partition 이 같다는 것은 **같은 cell structure** + **same multiset of cell sizes**. 이 multiset equality 는 hash function 의 deterministic 성질에서 옴.

만약 multiset equality 보다 약한 condition (예: equal mean, equal max) 만 보장되면 induction step 깨짐 — 다른 cell 의 hidden state 가 다른 multiset 일 수 있음.

**약한 condition 으로의 확장**:

만약 GNN aggregator 가 strict 더 약하면 (mean, max), induction step 에서 multiset 의 일부만 보존 — strict 약한 GNN 에 대해 strict 약한 동등 condition 이 충분.

예: Mean GNN 은 "multiset 의 mean 이 같음" condition 으로도 같은 output. 따라서:
$$
\{a, b\} \text{ mean} = \{c, d\} \text{ mean} \Rightarrow \text{Mean-GNN}(G_1) = \text{Mean-GNN}(G_2)
$$

이는 1-WL 보다 약한 동등 — Mean GNN 은 1-WL 동등성을 모두 인식하지 못함.

**Generalized statement (informal)**:
$$
\text{Aggregator } \bigoplus \text{ 의 invariance class} \Leftrightarrow \text{GNN equivalence class}
$$

따라서 정리 2.3 은 sum-injective aggregator (GIN) 에 tight, mean / max 에는 strict 약함.

**Modern 일반화 (Geerts 2022)**: GNN 의 표현력을 invariant logic (Counting Logic CL2 등) 으로 정확히 characterize. 1-WL = CL2 — 더 정확한 등가성 정리.

</details>

**문제 3** (논문 비평): 정리 2.3 이 GNN 의 "최악 한계" 를 보여주지만, 실전 GNN 이 거의 모든 task 에서 잘 작동하는 이유는 무엇인가? Real-world graph 의 1-WL 충분성을 분석하라.

<details>
<summary>해설</summary>

**1-WL 가 충분한 이유 (real-world)**:

1. **대부분의 graph 가 1-WL 로 distinguishable**:
- Babai 1980: random graph 에서 1-WL 이 거의 모든 graph 를 canonical form 으로 식별 (확률 1).
- Real-world graph (citation, social) 도 random 에 가까움 — 1-WL 충분.

2. **Node feature 의 정보**:
- 1-WL 가 plain graph 에 적용. Real-world 는 rich node feature (Cora 의 word-bag, 분자의 atom type) 가 있음.
- Initial label 이 다르면 1-WL refinement 도 다른 결과 → 표현력 확장.

3. **Multi-layer 의 implicit composition**:
- 정리 2.3 은 single forward pass. Multi-layer GNN 은 hierarchical representation — 정확히 같은 limit 이지만 implicit 으로 더 풍부.

4. **1-WL 한계가 task 에 적용되지 않을 때**:
- Node classification: local pattern dominant (1-WL 충분).
- Graph classification: global structure 중요 (1-WL 부족할 수 있음 — 그래서 GIN이 우월).
- Link prediction: pair-wise — 1-WL 표현력 직접 영향 X.

**1-WL 한계가 문제되는 task**:

1. **Substructure counting**:
- Triangle, cycle 길이 counting → 1-WL 못함 (Chen 2020).
- 분자에서 functional group counting 등.

2. **Distance-based query**:
- "Node $i$ 에서 $j$ 까지의 shortest path?" — 1-WL 못함.
- Position-aware GNN 필요.

3. **Symmetric graph classification**:
- CSL, regular graph — 1-WL 한계 명확.

**Empirical evidence**:

- TU benchmark: GIN 우위 +2~5%. 작지만 의미 있음.
- OGB-PCBA (분자 활성 예측): GIN > GCN, but Graphormer (Ch7-01) 이 더 강력.
- ZINC: subgraph counting GNN 이 우월 (1-WL 부족).

**결론**: 정리 2.3 은 worst-case 한계. Real-world performance 는 task / dataset 의 1-WL 충분성에 의존. 따라서 GNN 선택은 task-aware:
- Local pattern → GCN/GraphSAGE 충분
- Global / substructure → GIN, Graphormer
- Position-aware → LapPE, Graphormer

이론과 실전의 균형이 GNN 연구의 핵심.

</details>

---

<div align="center">

[◀ 이전](./01-wl-test.md) | [📚 README](../README.md) | [다음 ▶](./03-gin-optimality.md)

</div>
