# 02. GNN as Attention과 Transformer의 통합

## 🎯 핵심 질문

- GAT 의 attention = sparse Transformer attention 임을 어떻게 formal 하게 보이는가?
- MPNN ⊂ Sparse Graph Transformer (GAT) ⊂ Dense Graph Transformer (Graphormer) 의 위계 증명?
- PNA (Corso 2020) 의 multi-aggregator 의 이론적 정당성?
- GatedGCN 의 edge update + gating 이 어떤 task 에 유리한가?
- GraphGPS (Rampášek 2022) 의 "MPNN + Transformer hybrid" 의 설계 철학?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GNN 과 Transformer 가 원래 다른 분야 (graph vs sequence) 에서 시작했지만, 수학적으로 **같은 framework 의 다른 instantiation**. 이 통합이 modern 연구의 핵심:

1. **GNN = Transformer with graph mask**: 표준 Transformer 에 graph adjacency 를 attention mask 로 사용
2. **Transformer = GNN on complete graph**: 모든 token pair 가 연결된 graph
3. **Graphormer = 사이의 bridge**: structural encoding 으로 graph info 주입

이 관점에서:
- MPNN 의 1-WL 한계 = local attention 의 한계
- Transformer 의 $O(n^2)$ = dense attention 의 cost
- Trade-off 가 task-specific

이 문서는 GNN-Transformer 통합의 formal 분석과 modern 변종 (PNA, GatedGCN, GraphGPS) 을 정리.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-graphormer.md](./01-graphormer.md)
- [Transformer Deep Dive](https://github.com/iq-ai-lab/transformer-deep-dive): Self-attention
- [Ch3-03](../ch3-message-passing/03-gat.md): GAT

---

## 📖 직관적 이해

### GAT as Sparse Transformer

GAT:
$$
\alpha_{ij} = \text{softmax}_{j \in N(i)}(e_{ij}), \quad e_{ij} = a^T [W h_i \| W h_j]
$$

Transformer self-attention:
$$
\alpha_{ij} = \text{softmax}_j(Q_i K_j^T / \sqrt d)
$$

**차이**:
1. GAT: additive scoring $a^T [\cdot; \cdot]$, Transformer: multiplicative $Q K^T$
2. GAT: sparse (only $j \in N(i)$), Transformer: dense (all $j$)

수학적으로 GAT = $\text{Transformer}(h; \text{mask} = A)$ (adjacency as mask).

### 위계

```
MPNN (sum/mean aggregator)
  ⊂ Sparse Graph Transformer (GAT, input-dependent attention)
  ⊂ Dense Graph Transformer (Graphormer, all-pairs)
  ⊂ Transformer on complete graph (no structure)
```

각 level 이 strict 위계 — 앞의 것이 뒤의 특수 case.

### PNA: Multi-aggregator

**Principal Neighborhood Aggregation (Corso 2020)**:
$$
h_i^{(l+1)} = \text{MLP}\left( [\bigoplus_1 \| \bigoplus_2 \| \cdots \| \bigoplus_K] \odot \text{scaler} \right)
$$

여러 aggregator 결합:
- Mean, Max, Min, Std, Sum

각각 multiset 의 다른 aspect 포착. Degree scaler로 magnitude 조정.

### GatedGCN

Edge 도 learnable state:
$$
e_{ij}^{(l+1)} = \text{MLP}([h_i \| h_j \| e_{ij}^{(l)}])
$$
$$
\eta_{ij} = \sigma(e_{ij}^{(l+1)})
$$
$$
h_i^{(l+1)} = h_i^{(l)} + \sum_{j \in N(i)} \eta_{ij} \odot (W h_j)
$$

Edge 가 "gate" 역할 — adaptive edge importance.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Attention Mask

$M \in \{0, 1\}^{n \times n}$ (or $\in \{0, -\infty\}$):
$$
\alpha_{ij} = \text{softmax}_j\left( \frac{Q_i K_j^T}{\sqrt d} + M_{ij} \right)
$$

$M_{ij} = -\infty$ → $\alpha_{ij} = 0$.

### 정의 2.2 — GAT as Masked Transformer

$$
\text{GAT}(X, A) = \text{Transformer}(X; M = \log A)
$$

(where $\log 0 = -\infty$, $\log 1 = 0$)

**Scoring function**: GAT 은 additive, standard Transformer 는 multiplicative — 이 차이는 cosmetic (representation 변형).

### 정의 2.3 — Graphormer as Structure-Biased Transformer

$$
\text{Graphormer}(X, G) = \text{Transformer}(X; \text{bias} = B + C)
$$

($B$ = spatial bias, $C$ = edge bias)

Mask 가 $-\infty$ 가 아닌 **learnable bias**.

### 정의 2.4 — PNA Aggregator Set

$$
\bigoplus_{\text{PNA}}(S) = \left[\sum_S, \text{mean}_S, \max_S, \min_S, \text{std}_S\right] \in \mathbb R^{5d}
$$

Degree scaler $s(d) \in \mathbb R^k$:
$$
s_1(d) = d, \quad s_2(d) = \log(d + 1), \quad s_3(d) = 1/d, \quad \ldots
$$

Final aggregation:
$$
h_i^{(l+1)} = \text{MLP}(s(\text{deg}_i) \odot \bigoplus_{j \in N(i)} \phi(h_j))
$$

### 정의 2.5 — GatedGCN Layer (Bresson 2017)

**Edge update**:
$$
\hat e_{ij}^{(l+1)} = A^{(l)} h_i^{(l)} + B^{(l)} h_j^{(l)} + C^{(l)} e_{ij}^{(l)}
$$

**Edge gate**:
$$
\eta_{ij} = \sigma(\hat e_{ij}^{(l+1)})
$$

**Node update**:
$$
h_i^{(l+1)} = h_i^{(l)} + \text{ReLU}\left( U^{(l)} h_i^{(l)} + \sum_{j \in N(i)} \eta_{ij} \odot (V^{(l)} h_j^{(l)}) \right)
$$

Edge feature 도 matching update — full edge-node joint evolution.

### 정의 2.6 — GraphGPS Layer (Rampášek 2022)

Hybrid:
$$
h_i^{(l+1)} = \text{MPNN}_{\text{local}}(h^{(l)}, \text{edge\_index}) + \text{Transformer}_{\text{global}}(h^{(l)}, \text{PE})
$$

(two parallel branches, output sum or concat)

Local branch: GIN/GatedGCN/GCN 등.
Global branch: Performer, Transformer with positional encoding.

---

## 🔬 정리와 결과

### 정리 2.1 — GAT ⊂ Sparse Transformer

**Theorem**: GAT 은 sparse attention mask ($M_{ij} = -\infty$ if $j \notin N(i)$) 를 가진 Transformer의 special case.

**증명**: GAT attention:
$$
\alpha_{ij} = \begin{cases} \text{softmax}_j(e_{ij}) & j \in N(i) \\ 0 & \text{else} \end{cases}
$$

Transformer with mask:
$$
\alpha_{ij} = \text{softmax}_j(\tilde e_{ij} + M_{ij}), \quad M_{ij} = \begin{cases} 0 & j \in N(i) \\ -\infty & \text{else} \end{cases}
$$

$M = -\infty$ 에서 $\exp = 0$ → softmax 제외. 같은 형태. $\square$

**Scoring function** (additive vs multiplicative) 는 cosmetic — reparameterization 으로 equivalent.

### 정리 2.2 — 위계 Strict

**Theorem**: MPNN ⊊ GAT ⊊ Graphormer (strict inclusion).

**Proof sketch**:

- **MPNN ⊊ GAT**: GAT 의 input-dependent attention $\alpha_{ij}$ vs MPNN 의 fixed weight. GAT 이 additional degree of freedom → strict 더 flexible.
- **GAT ⊊ Graphormer**: GAT 의 sparse (graph edge 만) vs Graphormer 의 dense (all pairs) + structural encoding. Graphormer 가 1-WL 를 넘음 (CSL 구분), GAT 는 1-WL 상한 (multiset injectivity 없으므로 약간 더 약).

### 정리 2.3 — PNA 의 Multi-aggregator 이론

**Theorem (Corso 2020)**: Multiple aggregator + degree scaling 이 single aggregator 보다 더 rich multiset information capture.

**증명 sketch**: $\{$mean, max, min, std, sum$\}$ 결합이 multiset 의 mean, extremes, spread 모두 포착. Single sum 보다 각 aspect 에 더 sensitive.

**표현력**: PNA 가 strict 하게 GIN (sum only) 보다 강한가? 이론상 같음 (1-WL) but empirical 더 좋음.

### 정리 2.4 — Computational Comparison

| 모델 | Cost per layer | Expressive |
|------|----------------|-----------|
| GCN | $O(m d)$ | 1-WL - |
| GAT | $O(m d^2)$ | ≤ 1-WL |
| GIN | $O(m d^2)$ | = 1-WL |
| PNA | $O(m K d^2)$ | ≈ 1-WL |
| GatedGCN | $O(m d^2)$ | 1-WL + edge info |
| Graphormer | $O(n^2 d^2)$ | > 1-WL |
| GraphGPS | $O((m + n^2) d^2)$ | > 1-WL |

### 정리 2.5 — GraphGPS 의 장점

**Empirical (Rampášek 2022)**:
- Local only (GIN): limited 1-WL
- Global only (Transformer + PE): long-range but loses local structure
- Hybrid (GraphGPS): best of both

On OGB and LRGB:
- ZINC: GIN 0.20 MAE, Graphormer 0.12, **GraphGPS 0.09**
- PEPTIDES: GIN 62%, Graphormer 70%, **GraphGPS 72%**

---

## 💻 구현

### 실험 1 — GAT = Sparse Attention 증명

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def sparse_attention(Q, K, V, mask):
    """Masked attention: 0 outside mask."""
    scores = Q @ K.T / Q.size(-1)**0.5
    scores = scores.masked_fill(~mask, float('-inf'))
    alpha = F.softmax(scores, dim=-1)
    return alpha @ V

# GAT-like scoring (additive) vs Transformer (multiplicative)
# Equivalent via reparameterization

n = 10; d = 8
X = torch.randn(n, d)
adj = (torch.rand(n, n) > 0.7).bool()
adj.fill_diagonal_(True)   # self-loop

W_Q = nn.Linear(d, d, bias=False)
W_K = nn.Linear(d, d, bias=False)
W_V = nn.Linear(d, d, bias=False)

Q, K, V = W_Q(X), W_K(X), W_V(X)
out = sparse_attention(Q, K, V, adj)
print(f'Sparse (graph-masked) Transformer output: {out.shape}')
# 이는 formal 하게 GAT-equivalent
```

### 실험 2 — PNA Multi-aggregator

```python
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min, scatter_std

class PNAAggregator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.combine = nn.Linear(5 * d, d)
    
    def forward(self, x, edge_index):
        src, dst = edge_index
        aggs = [
            scatter_add(x[src], dst, dim=0, dim_size=x.size(0)),
            scatter_mean(x[src], dst, dim=0, dim_size=x.size(0)),
            scatter_max(x[src], dst, dim=0, dim_size=x.size(0))[0],
            scatter_min(x[src], dst, dim=0, dim_size=x.size(0))[0],
            scatter_std(x[src], dst, dim=0, dim_size=x.size(0)),
        ]
        combined = torch.cat(aggs, dim=-1)
        return self.combine(combined)

# Degree scaler (PNA standard)
def degree_scaler(degree, mode='identity'):
    if mode == 'identity':
        return degree
    elif mode == 'log':
        return torch.log(degree + 1)
    elif mode == 'inverse':
        return 1.0 / (degree + 1)
```

### 실험 3 — GatedGCN

```python
class GatedGCNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Linear(d, d)
        self.B = nn.Linear(d, d)
        self.C = nn.Linear(d, d)
        self.U = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.bn_h = nn.BatchNorm1d(d)
        self.bn_e = nn.BatchNorm1d(d)
    
    def forward(self, h, e, edge_index):
        """
        h: [n, d] node
        e: [m, d] edge
        edge_index: [2, m]
        """
        src, dst = edge_index
        # Edge update
        e_new = self.A(h[src]) + self.B(h[dst]) + self.C(e)
        # Edge gate
        eta = torch.sigmoid(e_new)
        # Node update
        msg = eta * self.V(h[src])
        agg = scatter_add(msg, dst, dim=0, dim_size=h.size(0))
        h_new = h + F.relu(self.bn_h(self.U(h) + agg))
        e = e + F.relu(self.bn_e(e_new))
        return h_new, e
```

### 실험 4 — GraphGPS-style Hybrid

```python
class GraphGPSLayer(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        # Local: GIN
        self.gin_mlp = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)
        )
        # Global: Transformer
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        # Combine
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    
    def forward(self, h, edge_index):
        # Local GIN branch
        src, dst = edge_index
        local_agg = scatter_add(h[src], dst, dim=0, dim_size=h.size(0))
        local_out = self.gin_mlp(h + local_agg)
        
        # Global Transformer branch (single graph, 단일 sequence)
        h_seq = h.unsqueeze(0)   # [1, n, d]
        global_out, _ = self.attn(h_seq, h_seq, h_seq)
        global_out = global_out.squeeze(0)
        
        # Combine (sum)
        h_combined = h + local_out + global_out
        h_combined = self.ln1(h_combined)
        
        # FFN
        h_final = h_combined + self.ffn(self.ln2(h_combined))
        return h_final
```

### 실험 5 — 모델 비교 on ZINC

```python
# ZINC: 12k 분자, property regression
# (PyG dataset 로드 + 각 모델 학습)

# 기대 결과:
# GIN: MAE ~0.25
# GAT: MAE ~0.22
# PNA: MAE ~0.18
# GatedGCN: MAE ~0.15
# Graphormer: MAE ~0.12
# GraphGPS: MAE ~0.09 (SOTA)

# (상세 학습 코드 생략)
```

---

## 🔗 실전 활용

### 1. GraphGPS 가 현재 SOTA

대부분의 chemistry benchmark (ZINC, PCQM4M, PEPTIDES) 에서 GraphGPS family 가 top:
- GPS (Rampášek 2022)
- GRIT (Ma 2023)
- SGFormer (Wu 2023)

### 2. Long Range Graph Benchmark (LRGB)

Dwivedi 2022 가 5 datasets (PEPTIDES, COCO, PascalVOC etc.) 제안. Long-range dependency 측정.

**LRGB 결과**:
- MPNN alone: weak
- Transformer alone: strong but no local
- Hybrid (GPS): best

### 3. Heterophilic Graph

Wikipedia-like heterophilic graph: Transformer-based (global attention) 이 local MPNN 보다 유리.

### 4. 분자 3D Property

Molecular geometry 가 중요한 task: SE(3)-Transformer, EGNN (Ch7-03) — equivariant attention.

### 5. PyG Integration

```python
from torch_geometric.nn import GPSConv, GINEConv

gps = GPSConv(d, conv=GINEConv(mlp), heads=4)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Dense attention $O(n^2)$ | Large graph 시 linear attention 필요 |
| Structural encoding 효과 | Task 에 따라 diminishing return |
| Hybrid 복잡도 | Tuning 난도 ↑ |
| PNA multi-aggregator 의 redundancy | Head pruning 가능 |
| GatedGCN 의 edge parameter 많음 | Overfitting 위험 |
| Single-graph setting | Batch 의 다양성 부족 시 학습 어려움 |

---

## 📌 핵심 정리

**위계**:
$$
\boxed{\text{MPNN} \subsetneq \text{GAT (sparse)} \subsetneq \text{Graphormer (dense)} \subsetneq \text{Transformer + structural PE}}
$$

| 모델 | Attention | Structure | 표현력 |
|------|-----------|-----------|--------|
| **GCN/GIN** | Fixed weight | Implicit | 1-WL |
| **GAT** | Sparse, input-dep | Graph mask | ≤ 1-WL |
| **GatedGCN** | Edge gate | Edge state | 1-WL + edge |
| **PNA** | Multi-agg | Degree scaler | ≈ 1-WL |
| **Graphormer** | Dense + structural bias | SPD, centrality, edge | > 1-WL |
| **GraphGPS** | Local MPNN + Global attention | PE + SPD | > 1-WL |

핵심 insight: **GNN 과 Transformer 는 같은 framework 의 다른 axis**. 현대 best practice 는 **hybrid** — local (MPNN) + global (attention) + structural (PE).

---

## 🤔 생각해볼 문제

**문제 1** (기초): GAT 의 "additive attention" $e_{ij} = a^T [Wh_i \| Wh_j]$ 가 Transformer 의 "multiplicative" $Q K^T$ 와 동치임을 보여라.

<details>
<summary>해설</summary>

**GAT additive**:
$$
e_{ij} = a^T [Wh_i \| Wh_j] = a_1^T W h_i + a_2^T W h_j
$$

(where $a = [a_1; a_2]$, split)

**Transformer multiplicative**:
$$
s_{ij} = (W_Q h_i)^T (W_K h_j) = h_i^T W_Q^T W_K h_j
$$

**동치성 분석**:

두 form 이 strict 하게 **함수적으로 다름**:
- GAT: Linear in $h_i$ and $h_j$ separately (additive)
- Transformer: Bilinear in $h_i, h_j$ (multiplicative)

하지만 **representation power 는 동치** (at sufficient width):

- Multiplicative form: $h_i^T M h_j$ ($M$ = $W_Q^T W_K$). Rank-$r$ matrix.
- Additive form: $a_1^T W h_i + a_2^T W h_j$. Can express single linear combination per axis.

실전 performance 는 다름:
- Additive: Fewer parameters, but **static ranking** (Ch3-03 문제 2)
- Multiplicative: More flexible, dynamic ranking

**결론**: GAT 의 additive 는 Transformer 의 multiplicative 의 special / restricted form. **GATv2 (Brody 2022)** 는 multiplicative 에 더 가까운 형태로 보강.

그럼에도 둘 다 "attention mechanism on graph" 의 다른 구현 — 개념적 유사성 강조.

</details>

**문제 2** (심화): GraphGPS 의 hybrid (local + global) 가 single branch 보다 우월한 이유를 task 특성 으로 설명하라.

<details>
<summary>해설</summary>

**Local (MPNN) only 의 한계**:
- 1-WL 상한
- Short-range receptive field (O(L) hop)
- Over-smoothing

**Global (Transformer) only 의 한계**:
- Graph structure 무시 (all-pairs 동등 — identity 후 PE 필요)
- Locality bias 없음 (inductive bias 약)
- $O(n^2)$ cost

**Task 별 분석**:

**1. Molecular property prediction (ZINC, QM9)**:

- **Local pattern**: Functional group, ring — 1-2 hop
- **Long-range**: Intramolecular interaction, resonance — multi-hop
- **Hybrid 필요**: Both essential
- **Empirical**: GraphGPS ~ Graphormer + GIN, better than alone

**2. Node classification on citation (Cora)**:

- **Local pattern dominant**: Community structure, 1-2 hop 충분
- **Long-range less critical**: Citation 은 local
- **Hybrid marginal**: GPS ≈ GCN + small Transformer addition
- **실전**: GCN 만으로 OK

**3. Heterophilic (Wikipedia)**:

- **Global attention 필요**: 이웃 heterogeneous, long-range pattern 중요
- **Local 약함**: Standard MPNN fail
- **Hybrid 우월**: Global 이 주, local 은 보조

**4. Large-scale (OGB-Products)**:

- **Sampling 필수**: $n > 10^6$
- **Global attention 비실용**: $n^2$ cost
- **Local dominant + sparse global**: Cluster-GCN + local attention

**5. Long Range Benchmark (LRGB)**:

- **설계 목적 = long-range**: Explicit test
- **Hybrid 우월**: Graphormer / GPS >> MPNN
- **Dataset 설계가 hybrid 지지**

**Conclusion**:

- Chemistry / 분자: Hybrid strict 우월
- Citation: Marginal
- Heterophilic: Hybrid 강력
- Long-range: Hybrid 필수
- Large-scale: Sampling + local 필요

**Modern GNN library**: Hybrid 가 default — GNN 의 future. 단 computational budget 에 따라 local only 또는 global only 선택.

</details>

**문제 3** (논문 비평): PNA 의 multi-aggregator 가 실증적 좋지만 이론상 표현력은 sum 과 같다 (1-WL). 그럼 왜 PNA 가 GIN 을 outperform 하는가?

<details>
<summary>해설</summary>

**PNA 이득의 이론 vs 실증 gap**:

**이론적 한계**: Sum + MLP 가 multiset universal injective (Ch4-03). Multi-aggregator 도 1-WL ceiling.

**실증적 우월**: GIN 75.8 → PNA 79.0 on OGB-molhiv. +3% ROC-AUC.

**이 gap 의 원인**:

1. **Sample efficiency**:
   - GIN sum: Single aggregator — MLP 가 sum 에서 "disentangle" 해야 함.
   - PNA multi-agg: Multiple "views" 의 input. MLP 가 더 easy 하게 학습.
   - 같은 표현력 but easier optimization → faster convergence, better sample efficiency.

2. **Implicit regularization**:
   - Multi-aggregator 이 input redundancy 제공
   - 각 aggregator 가 다른 angle — noise-robust
   - 특히 small dataset / noisy label 에서 도움

3. **Initialization / inductive bias**:
   - Sum: Magnitude 폭발 위험
   - Mean: Size-invariant (중립)
   - Max: Extreme value
   - 각 aggregator 의 bias 가 task 에 적합 → faster alignment with target

4. **Gradient flow**:
   - Single aggregator: Gradient 가 MLP 하나로 집중
   - Multi-aggregator: Gradient 가 여러 path — smoother, easier backprop

5. **Degree scaling 의 효과**:
   - Standard sum: high-degree node 의 magnitude 폭발
   - PNA degree scaler: Adaptive normalization — stable training

**Empirical evidence**:

Corso 2020 ablation:
- Sum only (GIN-like): 75.5
- + Mean: 77.0
- + Max: 77.5
- + Min: 77.8
- + Std: 78.2
- Full PNA (+ scaler): 79.0

각 aggregator 가 +0.5~1% 기여 — no single dominant, combination 의 이득.

**이론적 설명**:

"1-WL ceiling" 은 **asymptotic** limit (infinite data, infinite capacity). 실전 finite data / compute 에서는:
- **Easier optimization** 이 더 큰 model 에서 나은 final performance
- **Inductive bias diversity** 가 generalization ↑

**Lesson**:

이론적 표현력 = ceiling, 실전 performance = ceiling + optimization + inductive bias + regularization.

**Modern 추세**: Diverse aggregator + rich structural encoding (PE) + attention — 모든 axis 의 information 결합이 best practice.

이 원리가 Graphormer (dense attention + PE), GraphGPS (local + global) 의 설계 철학 — 이론적 동치 보다 empirically rich representation.

</details>

---

<div align="center">

[◀ 이전](./01-graphormer.md) | [📚 README](../README.md) | [다음 ▶](./03-equivariant-gnn.md)

</div>
