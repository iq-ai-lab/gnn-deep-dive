# 01. Message Passing Neural Network (Gilmer 2017)

## 🎯 핵심 질문

- Gilmer 2017 의 MPNN framework 가 GCN·GraphSAGE·GAT·GIN 을 어떻게 통일하는가?
- $m_{ij}^{(l)} = M_l(h_i^{(l)}, h_j^{(l)}, e_{ij})$ 와 $h_i^{(l+1)} = U_l(h_i^{(l)}, \bigoplus_{j} m_{ij}^{(l)})$ 의 두 함수가 무엇을 의미하는가?
- Aggregator $\bigoplus \in \{\text{sum}, \text{mean}, \text{max}, \text{attention}\}$ 의 선택이 모델을 결정하는 이유는?
- Belief propagation 과 MPNN 의 관계는?
- QM9 분자 에너지 예측에서 MPNN 이 어떻게 첫 SOTA를 달성했는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GNN 분야는 2014~2017 년까지 여러 모델이 우후죽순 등장했습니다 — Spectral GCN, ChebNet, GCN, GraphSAGE, GG-NN (Gated Graph NN), Interaction Networks (Battaglia 2016) 등. Gilmer et al. (2017) 의 **"Neural Message Passing for Quantum Chemistry"** 가 이들을 **하나의 framework — MPNN — 로 통일**:

1. **모든 spatial GNN 이 message + aggregate + update 의 형태** 임을 보임
2. **Aggregator 선택이 모델의 핵심 차별** — sum, mean, max, attention
3. **분자 chemistry 에서 첫 SOTA** — QM9 데이터셋의 에너지 예측

이 framework 는 PyG 의 `MessagePassing` base class 의 직접적 기반이며, 이후 모든 GNN 연구의 표준 언어. Ch3-02 ~ Ch3-04 의 GraphSAGE / GAT / GIN 모두 MPNN 의 specific instantiation.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch2-04](../ch2-spectral-gcn/04-spectral-vs-spatial.md) — Spatial 관점
- [Graphical Models Deep Dive](https://github.com/iq-ai-lab/graphical-models-deep-dive): Belief propagation, factor graph
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): MLP

---

## 📖 직관적 이해

### MPNN 의 두 핵심 연산

**한 layer 의 노드 update**:

1. **Message 단계**: 각 edge $(i, j)$ 에서 message 계산
   $$m_{ij} = M(h_i, h_j, e_{ij})$$
   ($M$ = MLP, edge feature $e_{ij}$ optional)

2. **Aggregate + Update 단계**: 각 노드가 받은 message 를 모아 자신의 state update
   $$h_i^{\text{new}} = U\left(h_i, \bigoplus_{j \in N(i)} m_{ij}\right)$$

이 두 연산이 모든 spatial GNN 의 공통 구조.

### Belief Propagation 과의 비교

Loopy BP (graphical model 의 inference 알고리즘):
$$
m_{i \to j}(x_j) = \sum_{x_i} \psi(x_i, x_j) \prod_{k \in N(i) \setminus j} m_{k \to i}(x_i)
$$

**유사성**: 노드 $\to$ 노드 message, 모든 이웃의 정보 합산.

**차이**:
- BP: closed-form (sum-product), exact on tree
- MPNN: learned function $M, U$ (MLP), supervised loss

따라서 **MPNN ≈ "학습된 BP"** — 같은 message-passing structure, 다른 update rule.

### Aggregator 선택의 의미

- **Sum**: $h_i + \sum_{j \in N(i)} h_j$ — multiset 정보 보존 (GIN, Ch3-04)
- **Mean**: $\frac{1}{|N(i)|} \sum_{j} h_j$ — degree normalization (GCN, GraphSAGE)
- **Max**: $\max_{j} h_j$ — extreme feature (GraphSAGE pool)
- **Attention**: $\sum_j \alpha_{ij} h_j$ — adaptive (GAT)

각 선택이 다른 표현력 — Ch4 에서 자세한 분석.

### 그림으로 보는 한 step

```
         (이웃 j1)         (이웃 j2)
            \                /
       m_{i,j1}            m_{i,j2}     (message 단계)
              \           /
               ⊕ (sum)                  (aggregate)
                |
              update
              /  \
        h_i (자신) → h_i' (new state)
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — MPNN Layer (Gilmer 2017)

각 layer $l = 0, 1, \ldots, L-1$ 에서:

**Message function** $M_l: \mathbb{R}^{d} \times \mathbb{R}^{d} \times \mathbb{R}^{d_e} \to \mathbb{R}^{d_m}$:
$$
m_{ij}^{(l)} = M_l(h_i^{(l)}, h_j^{(l)}, e_{ij})
$$

**Aggregate function** $\bigoplus$ (permutation-invariant):
$$
m_i^{(l)} = \bigoplus_{j \in N(i)} m_{ij}^{(l)}
$$

**Update function** $U_l: \mathbb{R}^d \times \mathbb{R}^{d_m} \to \mathbb{R}^{d}$:
$$
h_i^{(l+1)} = U_l(h_i^{(l)}, m_i^{(l)})
$$

### 정의 1.2 — Readout Function (Graph-level)

$L$ layer 후 graph-level representation:
$$
y = R(\{h_v^{(L)} : v \in V\})
$$

$R$ = sum, mean, attention pool, set2set 등 (Ch6-02 에서 자세히).

### 정의 1.3 — Standard MPNN Instantiations

| Model | $M$ | $\bigoplus$ | $U$ |
|-------|-----|-------------|-----|
| **GCN** | $\frac{1}{\sqrt{\tilde d_i \tilde d_j}} h_j$ | sum | $\sigma(\cdot W)$ |
| **GraphSAGE** | $h_j$ | mean / max / LSTM | $\sigma(W [h_i \| \text{Agg}])$ |
| **GAT** | $\alpha_{ij} W h_j$ | sum (with attention) | $\sigma$ |
| **GIN** | $h_j$ | sum | $\text{MLP}((1+\epsilon) h_i + \sum)$ |
| **MPNN (orig)** | MLP $(h_i, h_j, e_{ij})$ | sum | GRU |

### 정의 1.4 — Edge Feature 의 포함

분자에서 bond type 등 edge feature $e_{ij}$ 를 message 에 포함:
$$
m_{ij} = \text{MLP}([h_j; e_{ij}])
$$

또는 edge-conditioned weight:
$$
m_{ij} = (e_{ij} \cdot W) h_j
$$

(MPNN-edge: edge feature 로 weight matrix 자체를 modulation)

---

## 🔬 정리와 결과

### 정리 1.1 — Permutation Equivariance of MPNN

**Theorem**: MPNN layer 는 node permutation 에 equivariant. 즉, $\pi$ 가 노드 순서 변경이면:
$$
\text{MPNN}(\pi(G), \pi(X)) = \pi(\text{MPNN}(G, X))
$$

**증명**: 
- $M$ 은 노드별 (local) 함수 — permutation 에 영향 X
- $\bigoplus$ 가 permutation-invariant aggregator (sum/mean/max 등) — set 처리이므로 순서 무관
- $U$ 는 노드별 함수
종합: 전체 layer 가 permutation equivariant. $\square$

이는 GNN 의 **inductive bias 의 핵심** — graph 자체가 isomorphism 에 invariant 이므로 함수도 그래야 함.

### 정리 1.2 — Aggregator의 표현력 위계

**Theorem (informal)**: Multiset 표현력 위계:
$$
\text{sum} > \text{mean}, \text{max} > \text{trivial}
$$

(Ch4-03 에서 정확히 증명. $\{1,1,2,2\}$ vs $\{1,2\}$ 등 반례)

### 정리 1.3 — MPNN 의 K-Hop Receptive Field

**Theorem**: $L$-layer MPNN 의 노드 $i$ 출력은 $i$ 의 $L$-hop neighborhood 에만 의존.

**증명** (induction on layer $l$):
- $l = 0$: $h_i^{(0)} = X_i$ — 0-hop (자기 자신).
- $l \to l+1$: $h_i^{(l+1)}$ 은 $h_i^{(l)}$ 와 $\{h_j^{(l)} : j \in N(i)\}$ 에 의존. 각 $h_j^{(l)}$ 는 $j$ 의 $l$-hop neighborhood — $i$ 기준 $(l+1)$-hop. $\square$

이는 GCN 정리 3.6 (Ch2-03) 의 일반화.

### 정리 1.4 — Computational Cost

**Theorem**: MPNN layer 의 계산 비용:
- Message 단계: $O(m \cdot \text{cost}(M))$ — 모든 edge 처리
- Aggregate: $O(m)$ (scatter)
- Update: $O(n \cdot \text{cost}(U))$

총 $O(m \cdot d^2 + n \cdot d^2)$ for MLP message/update with width $d$.

Sparse graph $m = O(n)$: $O(n d^2)$ — linear in $n$.

### 정리 1.5 — MPNN as Universal Approximator (제한적)

MPNN 의 표현력은 **WL graph isomorphism test** 에 의해 상한 (Xu 2019, Ch4-02). 즉, 1-WL 이 구분 못하는 그래프 쌍은 어떤 MPNN 도 구분 못함.

이것이 MPNN 의 본질적 한계 — k-WL, position-aware GNN, Graphormer 가 이를 우회 (Ch4, Ch7).

---

## 💻 구현

### 실험 1 — MPNN Base Class (PyG-style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max   # PyG 의존성

class MPNNLayer(nn.Module):
    """
    h_i^{l+1} = U(h_i, ⊕_{j ∈ N(i)} M(h_i, h_j, e_{ij}))
    """
    def __init__(self, d, d_e=0, aggr='sum'):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * d + d_e, d), nn.ReLU(), nn.Linear(d, d))
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.aggr = aggr
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        x: [n, d]
        edge_index: [2, m]  (source row 0, target row 1)
        edge_attr: [m, d_e]
        """
        src, dst = edge_index
        m_src, m_dst = x[src], x[dst]
        if edge_attr is not None:
            m = self.message_mlp(torch.cat([m_dst, m_src, edge_attr], dim=-1))
        else:
            m = self.message_mlp(torch.cat([m_dst, m_src], dim=-1))
        
        # Aggregate at destination
        if self.aggr == 'sum':
            agg = scatter_add(m, dst, dim=0, dim_size=x.size(0))
        elif self.aggr == 'mean':
            agg = scatter_mean(m, dst, dim=0, dim_size=x.size(0))
        elif self.aggr == 'max':
            agg, _ = scatter_max(m, dst, dim=0, dim_size=x.size(0))
        
        # Update
        return self.update_mlp(torch.cat([x, agg], dim=-1))
```

### 실험 2 — GCN as MPNN Special Case

```python
def gcn_message(h_j, deg_i, deg_j):
    """GCN-style normalized message"""
    return h_j / torch.sqrt(deg_i * deg_j)

# Equivalent to GCNConv: 위 MPNNLayer 에서
# message: h_j / sqrt(d̃_i d̃_j)
# aggr: sum
# update: σ(W · sum)
```

### 실험 3 — Karate Club Classification (MPNN sum)

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
n = G.number_of_nodes()
edges_arr = np.array(list(G.edges())).T
edge_index = torch.tensor(np.concatenate([edges_arr, edges_arr[::-1]], axis=1), dtype=torch.long)

X = torch.eye(n)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True

class SimpleMPNN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, aggr='sum'):
        super().__init__()
        self.mp1 = MPNNLayer(d_in, aggr=aggr)
        self.mp2 = MPNNLayer(d_hid, aggr=aggr)
        self.cls = nn.Linear(d_hid, d_out)
    
    def forward(self, x, edge_index):
        h = F.relu(self.mp1(x, edge_index))
        h = F.relu(self.mp2(h, edge_index))
        return self.cls(h)

# 학습 코드는 GCN 과 동일
```

### 실험 4 — QM9 분자 데이터 (PyG)

```python
# QM9 의 첫 분자
try:
    from torch_geometric.datasets import QM9
    qm9 = QM9(root='./data/QM9')
    sample = qm9[0]
    print(f'Atoms: {sample.x.shape[0]}, Bonds: {sample.edge_index.shape[1]//2}')
    print(f'Atom features: {sample.x.shape}, Edge features: {sample.edge_attr.shape}')
    print(f'Targets (19개 properties): {sample.y.shape}')
except (ImportError, RuntimeError):
    print('PyG/QM9 not available; skipping')
```

### 실험 5 — Aggregator 비교: WL-fail 예제

```python
# {1, 1, 2, 2} vs {1, 2}: mean 같음, sum 다름
neighbors_A = torch.tensor([1.0, 1.0, 2.0, 2.0]).unsqueeze(-1)
neighbors_B = torch.tensor([1.0, 2.0]).unsqueeze(-1)

agg_sum_A = neighbors_A.sum(0)
agg_sum_B = neighbors_B.sum(0)
agg_mean_A = neighbors_A.mean(0)
agg_mean_B = neighbors_B.mean(0)
agg_max_A = neighbors_A.max(0).values
agg_max_B = neighbors_B.max(0).values

print(f'Sum:   A={agg_sum_A}, B={agg_sum_B}  (다름)')
print(f'Mean:  A={agg_mean_A}, B={agg_mean_B}  (같음)')
print(f'Max:   A={agg_max_A}, B={agg_max_B}  (같음)')
```

**예상 출력**:
```
Sum:   A=tensor([6.]), B=tensor([3.])  (다름)
Mean:  A=tensor([1.5000]), B=tensor([1.5000])  (같음)
Max:   A=tensor([2.]), B=tensor([2.])  (같음)
```

이 차이가 GIN (sum) 의 표현력 우위의 직접 증거 (Ch4-03).

---

## 🔗 실전 활용

### 1. PyG `MessagePassing` Base Class

```python
from torch_geometric.nn import MessagePassing

class MyGNN(MessagePassing):
    def __init__(self, d_in, d_out):
        super().__init__(aggr='sum')   # ⊕ 선택
        self.lin = nn.Linear(d_in, d_out)
    
    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):    # M(h_j)
        return x_j
    
    def update(self, aggr_out):  # U(aggr)
        return aggr_out
```

PyG 가 message/aggregate/update 를 추상화 — 새 GNN 구현이 이 form 만 채우면 됨.

### 2. QM9 Chemistry SOTA 의 의의

Gilmer 2017 의 MPNN 이 13개 property 중 11개에서 화학 정밀도 (chemical accuracy) 달성. 이전의 hand-crafted feature + kernel method 압도. 이는 **GNN 의 chemistry 응용 폭발**의 시작 (DimeNet, SchNet, GemNet 등).

### 3. Recurrent MPNN / GRU Update

원본 Gilmer MPNN 은 update function 을 GRU 로 사용:
$$
h_i^{(l+1)} = \text{GRU}(h_i^{(l)}, m_i^{(l)})
$$

이는 long-range dependency 처리에 유리, 단 메모리 비용 ↑.

### 4. Heterogeneous MPNN

Ch3-05 에서 다룰 R-GCN, HAN: 노드/edge 타입별 다른 $M, U$ 함수.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 1-hop message per layer | Long-range 정보 ↑ → over-smoothing 위험 |
| Permutation-invariant aggregator | Order-aware aggregator (LSTM) 는 학습 불안정 |
| 표현력 ≤ 1-WL | k-WL, position-aware 필요 (Ch4) |
| Static graph | Dynamic graph 는 temporal MPNN |
| Single edge feature | Multi-relational (R-GCN) 별도 처리 |
| Local information only | Global graph property (graph diameter 등) 못 봄 |

---

## 📌 핵심 정리

$$\boxed{m_{ij}^{(l)} = M_l(h_i^{(l)}, h_j^{(l)}, e_{ij}) \quad — \text{message}}$$

$$\boxed{h_i^{(l+1)} = U_l\left(h_i^{(l)}, \bigoplus_{j \in N(i)} m_{ij}^{(l)}\right) \quad — \text{aggregate + update}}$$

| Model | $M$ | $\bigoplus$ | $U$ |
|-------|-----|-------------|-----|
| **GCN** | $h_j / \sqrt{\tilde d_i \tilde d_j}$ | sum | $\sigma(\cdot W)$ |
| **GraphSAGE** | $h_j$ | mean/pool/LSTM | $\sigma(W[h_i \| \text{Agg}])$ |
| **GAT** | $\alpha_{ij} W h_j$ | sum | $\sigma$ |
| **GIN** | $h_j$ | sum | MLP |
| **MPNN-orig** | MLP$(h_i, h_j, e_{ij})$ | sum | GRU |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GCN 의 propagation rule 을 MPNN form ($M, \bigoplus, U$) 으로 명시적으로 작성하라.

<details>
<summary>해설</summary>

GCN: $H^{(l+1)} = \sigma(\hat A H^{(l)} W)$, where $\hat A_{ij} = \tilde A_{ij} / \sqrt{\tilde d_i \tilde d_j}$.

Row $i$ 풀어 쓰기:
$$
h_i^{(l+1)} = \sigma\left( \sum_{j \in N(i) \cup \{i\}} \frac{1}{\sqrt{\tilde d_i \tilde d_j}} h_j^{(l)} W^{(l)} \right)
$$

MPNN form:
- **Message**: $m_{ij} = \frac{1}{\sqrt{\tilde d_i \tilde d_j}} h_j W$ (또는 $W h_j$ — 위치만 차이)
- **Aggregate**: $\bigoplus = $ sum over $N(i) \cup \{i\}$ (self-loop included)
- **Update**: $U(h_i, m_i) = \sigma(m_i)$ — 단순 ReLU (no further mixing of $h_i$ since self-loop 가 message 에 포함)

따라서 GCN 은 MPNN 의 specific instantiation: degree-normalized linear message + sum aggregate + ReLU update. $\square$

</details>

**문제 2** (심화): MPNN layer 의 receptive field 가 정확히 $L$-hop 임을 induction 으로 정확히 증명하라. 단, "exactly $L$-hop" (반례 가능성 포함) 도 분석하라.

<details>
<summary>해설</summary>

**Claim**: $L$-layer MPNN 후 노드 $i$ 의 출력 $h_i^{(L)}$ 는 정확히 $\{X_j : \text{hop}(i, j) \leq L\}$ 에 의존 (선형적으로 $\leq L$, 의미적으로 $= L$ 가능).

**Induction**:

Base ($L=0$): $h_i^{(0)} = X_i$ — 정확히 0-hop ($\{i\}$).

Step ($L \to L+1$): $h_i^{(L+1)} = U(h_i^{(L)}, \bigoplus_{j \in N(i)} M(h_i^{(L)}, h_j^{(L)}))$.

각 $h_j^{(L)}$ ($j \in N(i)$) 는 IH 로 $j$ 의 $L$-hop 에 의존. $j$ 의 $L$-hop $\subseteq$ $i$ 의 $(L+1)$-hop (by triangle inequality on hop distance).

따라서 $h_i^{(L+1)}$ 는 $i$ 의 $(L+1)$-hop 에 의존. $\square$ ($\leq L+1$ direction)

**Exactly $L$ vs $\leq L$**:

- "$\leq L$" 는 보장 — proof 위
- "$= L$": 만약 $j$ 가 $L$-hop ($i$ 에서 정확 $L$-hop neighbor) 이면 일반적으로 $h_i^{(L)}$ 가 $X_j$ 의 비자명한 함수. 단 message function 이 trivial (zero everywhere) 이면 dependency 끊김.

**예외 (degenerate case)**: $M$ 이 모든 message 를 0 으로 보내면 $L$-hop 이라도 영향 X. Generic learnable MPNN 에서는 이런 일 거의 없음.

따라서 **"effectively $L$-hop"** 가 정확한 표현.

</details>

**문제 3** (논문 비평): MPNN 의 "$M, \bigoplus, U$" framework 가 모든 GNN 을 통일하는 것처럼 보이지만, attention-based GNN (GAT, Graphormer) 는 MPNN 의 strict instantiation 으로 보기 어려운 경우가 있다. 그 이유를 분석하고, GNN framework 로서 MPNN 의 한계를 논하라.

<details>
<summary>해설</summary>

**MPNN 으로 attention GNN 표현하기**:

GAT: $\alpha_{ij} = \text{softmax}_j(e_{ij})$ where $e_{ij} = \text{LeakyReLU}(a^T [Wh_i \| Wh_j])$.

Message: $m_{ij} = \alpha_{ij} W h_j$. 

**Subtlety**: $\alpha_{ij}$ 가 softmax 로 정규화 — **이는 단순 edge-wise function 이 아니라 노드 $i$ 의 모든 이웃을 보고 결정** (softmax 는 set 위 연산). 따라서 $M(h_i, h_j, e_{ij})$ form 으로 표현하려면 한 단계 더 trick 필요 (raw $e_{ij}$ 계산 후 별도 softmax aggregation).

**Graphormer (Ch7-01) 의 도전**:
- Fully-connected attention (사실상 $N(i) = V \setminus \{i\}$) — 이는 graph structure 를 message 에 직접 사용하지 않고 bias term 으로 주입
- Dense $O(n^2)$ message — sparse MPNN 의 큰 일탈

**MPNN 한계**:

1. **Set 연산 표현 어려움**: Softmax, top-k, group normalization 등 set-wise 연산이 단순 $\bigoplus$ 로 표현 어려움.
2. **Graph-level information 부족**: MPNN 은 1-hop message 반복이지만, **graph diameter, global pooling 정보** 등이 layer-wise 로 자연스럽게 전파되지 않음 (= Ch4-04 의 k-WL 한계 와 직결).
3. **Message-aggregate-update 의 strict order**: 일부 GNN 은 update 후 message (or simultaneous) — 변형 필요.
4. **Edge update 부재**: 기본 MPNN 은 노드 update 만, edge feature update 는 별도. **GatedGCN** (Bresson 2017) 등이 이를 명시.

**대안 framework**:

- **GDL (Bronstein et al.)**: Group equivariance 기반 더 일반적 framework
- **Message Passing Substrate (Veličković 2022)**: graph rewiring + multi-step + edge update 통합
- **Graph Attention 의 일반화**: Transformer-on-graphs 로 통합 (Ch7)

**결론**: MPNN 은 spatial GNN 의 표준 어휘이지만, 모든 GNN 을 strict 하게 captures 하지는 않음. 현대 GNN (Graphormer, EGNN) 은 MPNN 의 자연스러운 확장으로 봐야 함.

</details>

---

<div align="center">

[◀ 이전](../ch2-spectral-gcn/04-spectral-vs-spatial.md) | [📚 README](../README.md) | [다음 ▶](./02-graphsage.md)

</div>
