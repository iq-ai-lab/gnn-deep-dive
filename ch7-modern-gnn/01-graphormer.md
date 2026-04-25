# 01. Graph Transformer — Graphormer (Ying 2021)

## 🎯 핵심 질문

- Graphormer 가 Transformer 를 graph 에 적용하는 3가지 구조적 encoding (centrality, spatial, edge) 은 무엇인가?
- Centrality encoding $h_i^{(0)} += z_{\text{deg}^-(i)} + z_{\text{deg}^+(i)}$ 의 의미와 역할?
- Spatial encoding attention bias $b_{\phi(i, j)}$ ($\phi$ = shortest path distance) 이 어떻게 graph structure 를 주입하는가?
- Edge encoding (SP path 따라 edge feature 평균) 이 어떻게 edge information 을 활용하는가?
- OGB-LSC PCQM4M 에서 Graphormer 가 SOTA 를 달성한 메커니즘?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch3-04, Ch4-03 에서 GIN 이 1-WL 도달 — message passing 의 ceiling. 그러나 1-WL 한계 (CSL 등) 와 local receptive field 문제는 여전. **Graph Transformer** 가 근본적 해결:

1. **Dense attention** (모든 노드 pair): Long-range dependency 자동 capture
2. **Structural encoding**: Graph structure 를 attention 에 주입 — Transformer 의 graph-aware 버전
3. **Permutation invariance**: Self-attention 자연적으로 invariant
4. **SOTA on chemistry**: OGB-LSC PCQM4M 에서 GNN 의 모든 변종 압도

**Graphormer (Ying et al. 2021 — "Do Transformers Really Perform Badly for Graph Representation?")** 가 첫 본격 graph transformer. 후속 GraphGPS, GRIT, SAN 등이 이 framework 확장.

이 문서는 Graphormer 의 3가지 encoding 과 수학적 메커니즘을 정리.

---

## 📐 수학적 선행 조건

- [Transformer Deep Dive](https://github.com/iq-ai-lab/transformer-deep-dive): Self-attention, positional encoding
- [Ch3-03](../ch3-message-passing/03-gat.md): GAT (sparse attention on graph)
- [Ch4-05](../ch4-expressive-power/05-positional-encoding.md): Positional encoding

---

## 📖 직관적 이해

### Transformer 를 Graph 에 왜 적용?

Message passing 의 한계 — local, 1-WL, over-smoothing. Transformer 의 장점:
- **Dense attention**: 모든 노드 쌍 one-shot interaction
- **Positional encoding**: 구조 정보 flexible 주입
- **Multi-head**: 다양한 관계 parallel capture

### 3가지 Encoding

1. **Centrality encoding**: 노드의 degree 에 따른 **intrinsic importance**. Hub node → 높은 centrality → attention 에 더 많은 참여.

2. **Spatial encoding**: 두 노드 사이 **shortest path distance** 를 attention bias 로. 가까운 노드 → 강한 attention.

3. **Edge encoding**: 두 노드 사이 **shortest path 의 edge feature** 평균 → edge type (분자의 bond) 정보 활용.

### Graphormer 의 수식

**Vanilla Transformer**:
$$
\text{Attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt d) V
$$

**Graphormer** (modified):
$$
\alpha_{ij} = \text{softmax}\left(\frac{(h_i W_Q)(h_j W_K)^T}{\sqrt d} + b_{\phi(i,j)} + c_{ij}\right)
$$

- $b_{\phi(i,j)}$: spatial bias (SPD-based)
- $c_{ij}$: edge encoding bias

Centrality: $h_i^{(0)} \leftarrow h_i^{(0)} + z_{\text{deg}(i)}$ (degree embedding at input).

### GAT 와의 대비

GAT: sparse attention (graph edge 만), adaptive weight.
Graphormer: dense attention (모든 pair), structural encoding 으로 graph awareness.

GAT $\subset$ Graphormer in expressive power.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Centrality Encoding

각 노드 $i$ 의 degree-based intrinsic embedding:
$$
h_i^{(0)} \leftarrow h_i^{(0)} + z_{\text{deg}^-(i)} + z_{\text{deg}^+(i)}
$$

- $\text{deg}^-(i), \text{deg}^+(i)$: in/out degree (directed) 또는 same degree (undirected)
- $z_{\cdot}$: learnable embedding of size $d$ for each degree value (0, 1, 2, ..., max_deg)

### 정의 1.2 — Spatial Encoding (Shortest Path Distance Bias)

**$\phi(i, j)$**: shortest path distance in $G$. $\infty$ if disconnected, $0$ if $i = j$.

Attention bias:
$$
b_{\phi(i, j)} = \text{learnable scalar (per SPD value)}
$$

$\phi \in \{0, 1, 2, \ldots, \phi_{\max}, \infty\}$ — finite lookup table.

### 정의 1.3 — Edge Encoding

각 노드 pair $(i, j)$ 의 shortest path 가 edge $e_{k_1}, e_{k_2}, \ldots, e_{k_l}$ 를 지날 때:
$$
c_{ij} = \frac{1}{l} \sum_{k=1}^l x_{e_k}^T w_{k}
$$

- $x_{e_k}$: edge feature
- $w_k$: learnable position weight along path

### 정의 1.4 — Graphormer Attention

$$
\alpha_{ij} = \text{softmax}_j\left( \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d/H}} + b_{\phi(i,j)} + c_{ij} \right)
$$

$H$: number of heads, $\sqrt{d/H}$: Transformer scaling.

각 head 별 별도 $b, c$ parameters.

### 정의 1.5 — Graphormer Layer

Full Transformer encoder layer:
$$
h^{(l)} = h^{(l-1)} + \text{MHA}(h^{(l-1)}) \\
h^{(l)} \leftarrow h^{(l)} + \text{FFN}(h^{(l)})
$$

각 layer 마다 LayerNorm, residual connection (standard Transformer).

### 정의 1.6 — Virtual Node (VN)

Graph-level representation 을 위한 special node:
- $v_{\text{VN}}$ 가 모든 노드와 연결 ($\phi = 1$ to all)
- Final layer $h_{v_{\text{VN}}}^{(L)}$ = graph-level representation

Readout 대체.

---

## 🔬 정리와 결과

### 정리 1.1 — Graphormer 의 표현력

**Theorem (informal, Ying 2021)**: Graphormer 가 1-WL 보다 strict 강함.

**증명 sketch**: Spatial encoding $b_{\phi}$ 가 graph 의 shortest path information 을 attention 에 주입. Highly symmetric graph (CSL) 도 SPD 가 다를 수 있음 → 1-WL 가 못 보는 pattern 구분.

**Empirical**: CSL benchmark 에서 Graphormer 100% accuracy (GIN random guess).

### 정리 1.2 — Computational Cost

**Theorem**: Graphormer layer cost:
- Attention: $O(n^2 d)$ (dense, 모든 pair)
- Position encodings: $O(n^2)$ for SPD lookup
- Total per layer: $O(n^2 d)$

GNN $O(m d)$ 비해 $n^2$ 비용. Large graph ($n > 1000$) 에서 문제:
- Approximation: Performer-like linear attention
- Sampling: subgraph attention

### 정리 1.3 — PCQM4M 성능

**OGB-LSC PCQM4M (HOMO-LUMO gap regression)**:

| Model | Validation MAE |
|-------|----------------|
| GCN | 0.168 |
| GIN | 0.151 |
| GIN + Virtual Node | 0.128 |
| DeeperGCN | 0.120 |
| **Graphormer** | **0.099** |
| Graphormer-L (bigger) | 0.087 |

Graphormer 가 30%+ 개선 — 매우 큼.

### 정리 1.4 — Ablation of 3 Encodings

**Ablation (Ying 2021 Table 5)**:

| Config | MAE |
|--------|-----|
| Vanilla Transformer (no encoding) | 0.178 |
| + Centrality only | 0.164 |
| + Spatial only | 0.115 |
| + Edge only | 0.160 |
| **All three** | **0.099** |

**Spatial encoding 이 가장 큰 효과** — SPD info 가 핵심.

### 정리 1.5 — Connection to MPNN

**Theorem**: Graphormer with $\phi = 1$ neighbor attention 만 keep (sparse attention) = GAT-like.

따라서:
- Graphormer $\supset$ GAT $\supset$ MPNN
- Graphormer 가 가장 expressive

---

## 💻 구현

### 실험 1 — Graphormer Layer (Simplified)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphormerAttention(nn.Module):
    def __init__(self, d, heads, max_deg=100, max_spd=20):
        super().__init__()
        self.heads = heads
        self.d_head = d // heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d)
        # Centrality encoding
        self.deg_emb = nn.Embedding(max_deg + 1, d)
        # Spatial encoding (per head)
        self.spatial_bias = nn.Parameter(torch.randn(heads, max_spd + 1))
    
    def forward(self, x, degree, spd_matrix):
        """
        x: [n, d]
        degree: [n]
        spd_matrix: [n, n]  (clamped to max_spd)
        """
        # Centrality encoding (applied to input x)
        x = x + self.deg_emb(degree.clamp(max=self.deg_emb.num_embeddings-1))
        
        n, d = x.shape
        qkv = self.qkv(x).reshape(n, 3, self.heads, self.d_head).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each [heads, n, d_head]
        
        attn = q @ k.transpose(-2, -1) * self.scale   # [heads, n, n]
        # Add spatial bias
        spd_clamped = spd_matrix.clamp(max=self.spatial_bias.size(1) - 1)
        bias = self.spatial_bias[:, spd_clamped]   # [heads, n, n]
        attn = attn + bias
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(1, 0, 2).reshape(n, d)
        return self.out(out)

class GraphormerLayer(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__()
        self.attn = GraphormerAttention(d, heads)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(),
            nn.Linear(ffn_mult * d, d)
        )
        self.ln2 = nn.LayerNorm(d)
    
    def forward(self, x, degree, spd):
        x = x + self.attn(self.ln1(x), degree, spd)
        x = x + self.ffn(self.ln2(x))
        return x
```

### 실험 2 — SPD Matrix 계산

```python
import networkx as nx
import numpy as np

def compute_spd_matrix(G, max_spd=20):
    """All-pairs shortest path distance matrix."""
    n = G.number_of_nodes()
    spd = np.full((n, n), max_spd)
    for i, paths_from_i in nx.all_pairs_shortest_path_length(G):
        for j, d in paths_from_i.items():
            spd[i, j] = min(d, max_spd)
    return torch.tensor(spd, dtype=torch.long)

G = nx.karate_club_graph()
spd = compute_spd_matrix(G, max_spd=10)
print(f'SPD matrix shape: {spd.shape}')
print(f'SPD[0, 33]: {spd[0, 33].item()}')
print(f'SPD distribution: {np.unique(spd.numpy(), return_counts=True)}')
```

### 실험 3 — Full Graphormer Model

```python
class Graphormer(nn.Module):
    def __init__(self, d_in, d, d_out, num_layers=6, heads=8):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d)
        self.layers = nn.ModuleList([GraphormerLayer(d, heads) for _ in range(num_layers)])
        self.readout = nn.Linear(d, d_out)
    
    def forward(self, x, degree, spd):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, degree, spd)
        # Graph-level: mean pool or first (virtual node)
        h_G = h.mean(0)
        return self.readout(h_G)

# Toy test
n = 20
d = 32
model = Graphormer(d_in=8, d=d, d_out=2, num_layers=3, heads=4)
x = torch.randn(n, 8)
degree = torch.randint(1, 5, (n,))
spd = torch.randint(0, 10, (n, n))
spd = (spd + spd.T) // 2   # symmetric
spd.fill_diagonal_(0)

out = model(x, degree, spd)
print(f'Graphormer output: {out.shape}')
```

### 실험 4 — CSL 에서 Graphormer vs GIN

```python
def csl(n, skip):
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + skip) % n)
    return G

# CSL(8, 2) vs CSL(8, 3): 1-WL 동등, SPD matrix 다름?
G1 = csl(8, 2)
G2 = csl(8, 3)

spd1 = compute_spd_matrix(G1, max_spd=4)
spd2 = compute_spd_matrix(G2, max_spd=4)

print(f'CSL(8,2) SPD distribution: {np.unique(spd1.numpy(), return_counts=True)}')
print(f'CSL(8,3) SPD distribution: {np.unique(spd2.numpy(), return_counts=True)}')
# 같을 수도 다를 수도 — check manually
```

### 실험 5 — Centrality Encoding의 효과

```python
# Vanilla Transformer vs Graphormer
# Karate Club node classification

# Vanilla: no centrality / spatial
class VanillaTransformer(nn.Module):
    def __init__(self, d_in, d, d_out, num_layers=3, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.cls = nn.Linear(d, d_out)
    
    def forward(self, x):
        h = self.input_proj(x).unsqueeze(0)   # [1, n, d]
        for layer in self.layers:
            h = layer(h)
        return self.cls(h.squeeze(0))

# 비교 실험 (상세 학습 코드 생략)
# 예상: Graphormer > Vanilla Transformer (structural encoding 덕에)
```

---

## 🔗 실전 활용

### 1. OGB-LSC PCQM4M-LSC

Graphormer 가 KDD Cup 2021 graph track 에서 1위. Molecular HOMO-LUMO gap prediction.

### 2. Quantum Chemistry

- GEOM-QM9: 분자 conformer prediction
- DrugBank: drug property prediction
- ChEMBL: bioactivity prediction

### 3. Microsoft Graphormer Library

```python
# Graphormer 가 PyG 또는 Microsoft graphormer 로 사용 가능
# https://github.com/microsoft/Graphormer
```

### 4. 후속 Graph Transformer

- **GraphGPS (Rampášek 2022)**: MPNN + Transformer hybrid
- **GRIT (Ma 2023)**: Rethinking Graph Transformer 
- **SAN (Kreuzer 2021)**: Spectral Attention Network
- **EGT (Hussain 2022)**: Edge-augmented Graph Transformer

### 5. Large-scale Adaptations

- **Graphormer-GD**: Global + local attention hybrid
- **Linearized attention**: Performer-style for large graphs

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Dense attention $O(n^2)$ | Large graph ($n > 1000$) 시 memory |
| All-pairs SPD | Pre-computation $O(n^3)$ for Floyd-Warshall |
| Centrality = degree only | 다른 centrality (betweenness, closeness) 도 가능 |
| SPD encoding up to $\phi_{\max}$ | Very long paths 정보 loss |
| Homogeneous edge feature | Multi-relation edge 별도 처리 |
| Static graph | Dynamic graph 는 SPD 재계산 |

---

## 📌 핵심 정리

**Graphormer 의 3가지 structural encoding**:
1. **Centrality**: $h_i^{(0)} += z_{\text{deg}(i)}$
2. **Spatial**: $\alpha_{ij}$ 에 bias $b_{\phi(i,j)}$
3. **Edge**: $\alpha_{ij}$ 에 bias $c_{ij}$ from path edge features

$$\boxed{\alpha_{ij} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + b_{\phi(i,j)} + c_{ij}\right)}$$

| 항목 | Graphormer |
|------|-----------|
| **Attention** | Dense (all-pairs) |
| **Structural encoding** | Centrality, Spatial, Edge |
| **Expressive power** | > 1-WL |
| **Cost** | $O(n^2 d)$ per layer |
| **PCQM4M MAE** | 0.099 (SOTA) |
| **Over-smoothing** | 완화 (attention 의 flexibility) |
| **Permutation invariance** | ✓ (attention 자연 invariant) |

핵심: **Graphormer = Transformer + graph structural encoding = GAT (sparse) 의 dense + structural 확장.**

---

## 🤔 생각해볼 문제

**문제 1** (기초): Graphormer 의 $O(n^2)$ attention cost 가 대규모 graph ($n = 10^6$) 에서 문제. Linear attention (Performer) 로 어떻게 reduce 하는가?

<details>
<summary>해설</summary>

**Linear Attention (Katharopoulos 2020, Choromanski 2021)**:

Softmax attention 의 decomposition:
$$
\alpha_{ij} \propto \exp(Q_i K_j^T) = \phi(Q_i) \phi(K_j)^T  \quad \text{(approx)}
$$

($\phi$ = feature map, e.g., random features)

Then:
$$
\text{Attn} = \phi(Q) (\phi(K)^T V)
$$

Order of operations: $\phi(K)^T V$ 먼저 → $O(n d^2)$, 그 다음 $\phi(Q) \cdot$ → $O(n d^2)$.

**총 $O(n d^2)$ — linear in $n$**.

**Graphormer 에의 적용**:

- Spatial bias $b_{\phi(i,j)}$: $\exp(b_\phi)$ 를 feature map 에 multiplicative 추가. 단, $b_\phi$ 가 $\phi(i, j)$ 에 따라 다름 → pair-specific, linear approximation 어려움.
- 해결: Spatial bias 를 separate low-rank decomposition, 또는 local + global hybrid (GraphGPS).

**Trade-off**:
- Linear attention: speed ↑, accuracy ↓ (approximation quality)
- Dense attention: exact, slow

**Large-scale 현실**:
- $n < 1000$: Dense Graphormer OK
- $n \in [1000, 10000]$: Sparse + global attention hybrid
- $n > 10000$: Sampling (Cluster-GCN on top of Graphormer, or Performer-style linear)

**실전**: Graphormer 는 molecule-level ($n \sim 30-50$) 이 주요 target — dense attention 의 강점 활용. Large graph 는 다른 architecture 필요.

</details>

**문제 2** (심화): Graphormer 의 spatial encoding $b_{\phi(i,j)}$ 가 positional encoding (LapPE, RWPE) 과 어떻게 다른가? 왜 Graphormer 는 LapPE 대신 SPD 를 사용?

<details>
<summary>해설</summary>

**Positional Encoding 들의 비교**:

**LapPE (Ch4-05)**: $u_k(i) \in \mathbb R$ per node — Laplacian eigenvector 좌표.
- 입력 feature 에 concat
- Sign ambiguity 문제
- 모든 eigenvector 사용 → $K$-dim PE

**RWPE**: $P^k_{ii}$ — return probability.
- Node-wise feature
- Sign 문제 없음
- Local structure 강조

**Graphormer 의 SPD**: $\phi(i, j)$ — pair-wise distance.
- **Pair-wise** (not node-wise)
- Attention 의 **bias** — feature 가 아니라 attention weight 수정
- Global structure (long-range SPD 까지)

**왜 SPD?**

1. **Attention 의 자연 bias**: Transformer 의 relative positional encoding (Shaw 2018) 이 같은 원리 — relative position 이 attention 에.

2. **Global reach**: SPD 가 모든 pair 를 직접 표현 — 1-WL 한계 직접 우회.

3. **Interpretable**: SPD = 명확한 graph 정보. LapPE 는 spectral (abstract).

4. **Discrete, lookup**: $\phi \in \{0, 1, ..., \phi_{\max}\}$ — finite lookup table. Efficient.

5. **Sign ambiguity 없음**: Deterministic distance.

**단점**:

- $\phi_{\max}$ 이상 distance 정보 loss
- All-pairs SPD 계산 비용 $O(n^3)$ (Floyd-Warshall)
- Multi-relation 에서 relation-specific SPD 필요

**Hybrid**:

GraphGPS: MPNN (local) + Graphormer (global) + LapPE (spectral) — 모든 positional encoding 결합. Best of both worlds.

**결론**:

- LapPE: Node-wise, spectral information
- RWPE: Node-wise, local random walk
- **SPD (Graphormer)**: Pair-wise, attention bias — graph structure 의 **direct** injection

각 PE 가 다른 axis 의 정보 — 결합이 strongest. 단 single PE 가 필요하다면 Graphormer 의 SPD 가 efficient + interpretable.

</details>

**문제 3** (논문 비평): Graphormer 가 chemistry 에서 SOTA 이지만 node classification (Cora) 에서는 GCN 과 marginal 차이. 이 불균형의 이유는?

<details>
<summary>해설</summary>

**Graphormer 의 chemistry 우위 이유**:

1. **Molecular graph 의 특성**:
   - Small size ($n \sim 20-50$) — dense attention 저비용
   - Rich edge feature (bond type, aromatic)
   - Long-range intramolecular interaction (resonance, steric)
   - Substructure pattern 중요 (functional group)

2. **Graphormer 의 강점 이 정확히 부합**:
   - Edge encoding → bond type
   - Spatial encoding → molecular geometry
   - Dense attention → long-range
   - Global representation via VN → molecular property

**Cora 에서 marginal 차이 이유**:

1. **Citation network 의 특성**:
   - Sparse (avg degree ~4)
   - 1-2 hop neighbor 만 필요 (local pattern dominant)
   - No edge features
   - Homophily — "이웃끼리 비슷" 충분

2. **GCN 이 이미 sufficient**:
   - Local message passing 으로 citation community 충분히 capture
   - Over-smoothing 문제는 small $L$ 로 회피
   - Graphormer 의 추가 capacity 가 overfitting 위험

3. **Training data 작음 (140 labels)**:
   - Graphormer 의 많은 parameter 가 overfitting
   - GCN 의 simpler model 이 better inductive bias

**Empirical pattern**:

| Dataset | GCN | GIN | Graphormer |
|---------|-----|-----|-----------|
| Cora (node) | 81.5% | 80.0% | 82.5% (marginal) |
| PCQM4M (chemistry) | MAE 0.168 | 0.151 | **0.099 (큰 향상)** |
| ZINC (chemistry) | baseline | +5% | **+15%** |

**일반 원리**:

Graphormer 가 우위인 조건:
- Rich structural info (edge types, 3D)
- Long-range dependency 중요
- Small-medium graph (dense attention affordable)
- 충분한 training data

GCN 이 충분한 조건:
- Simple homogeneous graph
- Local pattern sufficient
- Small labeled data
- Large graph (compute constraint)

**Modern consensus**:

- **Chemistry / Molecules**: Graphormer, Specformer, GPS 가 standard
- **Citation / Social (node classification)**: GCN, GAT, APPNP 충분
- **OGB large node benchmark**: 중간 — GNN + light attention hybrid
- **Protein / 3D structure**: SE(3)-equivariant (Ch7-03)

따라서 Graphormer 는 "chemistry 의 ImageNet-level breakthrough" — but node classification 의 ResNet 같은 universal dominance 아님. Task 적합성이 핵심.

</details>

---

<div align="center">

[◀ 이전](../ch6-applications/04-graph-generation.md) | [📚 README](../README.md) | [다음 ▶](./02-gnn-attention-unification.md)

</div>
