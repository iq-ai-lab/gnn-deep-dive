# 05. Edge Features와 Heterogeneous Graphs

## 🎯 핵심 질문

- Edge feature $e_{ij}$ 를 message function 에 어떻게 통합하는가?
- R-GCN (Schlichtkrull 2018) 의 relation-specific weight $W_r$ 가 어떻게 multi-relational graph 를 처리하는가?
- 파라미터 수가 relation 수에 비례하는 문제를 basis decomposition / block diagonal 로 어떻게 해결하는가?
- HAN (Wang 2019) 의 meta-path + hierarchical attention (node-level + semantic-level) 이 어떻게 작동하는가?
- Knowledge graph completion 의 표준 task 와 GNN 의 역할은?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

지금까지 다룬 GCN/GraphSAGE/GAT/GIN 은 **homogeneous graph** 가정 — 노드 1종, edge 1종. 그러나 real-world graph 는 대부분 heterogeneous:

1. **Knowledge graph**: entity (노드 여러 type — 사람, 영화, 도시), relation (edge 여러 type — born_in, acted_in, lives_in)
2. **Citation network**: paper, author, venue (3 node types), authored, cited, published (3+ edge types)
3. **분자**: atom type (H, C, N, O), bond type (single, double, aromatic)
4. **추천 시스템**: user, item, click, purchase 등

**Edge feature** 와 **node/edge type** 을 GNN 에 통합하는 것이 산업적 응용의 핵심. 이 문서는:
1. Edge feature 통합 방법
2. R-GCN: relation-specific GCN
3. HAN: meta-path 기반 hierarchical attention
4. Knowledge graph completion 응용

을 정리합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-mpnn-framework.md](./01-mpnn-framework.md), [03-gat.md](./03-gat.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Block diagonal matrix, low-rank decomposition

---

## 📖 직관적 이해

### Edge Feature 의 통합 방법

세 가지 일반적 방법:

1. **Concat in message**: $m_{ij} = \text{MLP}([h_j; e_{ij}])$
2. **Edge as weight modulation**: $m_{ij} = (W \odot e_{ij}) h_j$ — edge feature 가 weight 자체를 변형
3. **Learned edge type embedding**: discrete edge type → learnable embedding

GIN-edge, NNConv (Gilmer 2017), GatedGCN 등이 이 방법들 사용.

### Multi-Relational Graph 의 도전

Knowledge graph (DBpedia, Wikidata) 는 수백~수천 relation type. 각 relation 마다 다른 GNN 이 필요한데, 단순히 별도 $W_r$ 두면 파라미터 폭발.

**R-GCN** 의 해결: relation-specific weight 를 **basis decomposition** 또는 **block diagonal** 로 공유 — relation 수 ↑ but parameter 효율.

### Meta-path 의 직관

Heterogeneous graph 에서 "user → friend → friend" 같은 multi-hop 경로 = **meta-path**. 같은 meta-path 를 따르는 노드가 의미적으로 연결됨.

**HAN** 의 idea: meta-path 별로 homogeneous subgraph 만들고 각각 GNN 적용 → 다양한 의미 capture.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Heterogeneous Graph

$G = (V, E, \mathcal T_v, \mathcal T_e, \phi_v, \phi_e)$:
- $\mathcal T_v$: node type 집합, $\phi_v: V \to \mathcal T_v$
- $\mathcal T_e$: edge type 집합, $\phi_e: E \to \mathcal T_e$

각 노드 $v$ 의 type $\phi_v(v)$, 각 edge $(u, v)$ 의 type $\phi_e(u, v) = r$.

### 정의 5.2 — Multi-Relational Adjacency

각 relation $r \in \mathcal T_e$ 에 대해 $A_r \in \{0, 1\}^{n \times n}$:
$$
A_r[i, j] = 1 \Leftrightarrow \phi_e(i, j) = r
$$

전체 graph: $A = \sum_r A_r$ (단순 합) — 단 type 정보 손실.

### 정의 5.3 — R-GCN Layer (Schlichtkrull 2018)

$$
h_i^{(l+1)} = \sigma\left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal R} \sum_{j \in N_r(i)} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} \right)
$$

- $N_r(i)$: relation $r$ 로 연결된 이웃
- $c_{i,r} = |N_r(i)|$ (또는 fixed hyperparameter): normalization
- $W_0$: self-loop weight
- $W_r$: relation-specific weight

### 정의 5.4 — Basis Decomposition

R-GCN 의 파라미터 수 ($|\mathcal R| \cdot d^2$) 가 큼. **Basis decomposition**:
$$
W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)} V_b^{(l)}
$$

- $V_b \in \mathbb R^{d \times d}$: basis matrices ($B$ 개, $B \ll |\mathcal R|$)
- $a_{rb} \in \mathbb R$: learned coefficients

파라미터 $O(B d^2 + |\mathcal R| B)$ — relation 수에 linear, $d^2$ 와 분리.

### 정의 5.5 — Block Diagonal Decomposition

$W_r$ 를 block diagonal: $d = \sum_k d_k$, $W_r = \text{diag}(W_r^{(1)}, \ldots, W_r^{(K)})$. 각 block 만 학습 — sparsity-driven 파라미터 절약.

### 정의 5.6 — Meta-path

Heterogeneous graph 에서 type sequence $\mathcal P = (T_1, R_1, T_2, R_2, \ldots, T_l)$. 노드 시퀀스 $v_1, v_2, \ldots, v_l$ 가 meta-path 따른다 ⇔ 각 $v_i$ type $T_i$, edge $(v_i, v_{i+1})$ relation $R_i$.

### 정의 5.7 — HAN Layer (Wang 2019)

**Node-level attention** (within each meta-path):
$$
e_{ij}^{\mathcal P} = a^T \text{LeakyReLU}(W_{\mathcal P} [h_i \| h_j])
$$
$$
\alpha_{ij}^{\mathcal P} = \text{softmax}_j(e_{ij}^{\mathcal P})
$$
$$
z_i^{\mathcal P} = \sigma\left( \sum_{j \in N^{\mathcal P}(i)} \alpha_{ij}^{\mathcal P} W_{\mathcal P} h_j \right)
$$

**Semantic-level attention** (across meta-paths):
$$
\beta_{\mathcal P} = \text{softmax}_{\mathcal P}\left( \frac{1}{|V|} \sum_i q^T \tanh(M z_i^{\mathcal P} + b) \right)
$$
$$
h_i^{(l+1)} = \sum_{\mathcal P} \beta_{\mathcal P} z_i^{\mathcal P}
$$

(meta-path 별 importance 학습)

---

## 🔬 정리와 결과

### 정리 5.1 — R-GCN as MPNN

R-GCN 은 multi-message MPNN form:
- Relation-conditional message: $m_{ij}^r = \frac{1}{c_{i,r}} W_r h_j$
- Aggregation: $\sum_r \sum_{j \in N_r(i)}$ + self-loop $W_0 h_i$
- Update: $\sigma$

**증명**: 정의 직접 확인. $\square$

### 정리 5.2 — Parameter Counting

Single R-GCN layer (without basis):
- $|\mathcal R| \cdot d_{\text{in}} \cdot d_{\text{out}}$ for $W_r$
- $d_{\text{in}} \cdot d_{\text{out}}$ for $W_0$

Total $O(|\mathcal R| d^2)$.

With basis $B$:
- $B \cdot d^2$ for $V_b$
- $|\mathcal R| \cdot B$ for $a_{rb}$

Total $O(B d^2 + |\mathcal R| B) = O(B(d^2 + |\mathcal R|))$ — $|\mathcal R| \gg B$ 시 큰 절감.

### 정리 5.3 — Knowledge Graph Completion

Knowledge graph: triplet $(s, r, t)$ 의 link prediction. R-GCN encoder + decoder:
- Encoder: R-GCN으로 entity embedding
- Decoder: scoring function $f(h_s, r, h_t)$
  - DistMult: $h_s^T \text{diag}(r) h_t$
  - ComplEx: complex-valued
  - RotatE: $h_t = h_s \circ r$ (Hadamard product on complex)

학습: missing triplet 에 대한 negative sampling + cross-entropy.

### 정리 5.4 — Meta-path 의 Composability

여러 meta-path 가 graph 의 다른 의미적 측면 capture:

- $\mathcal P_1$ = "movie → director → movie" (같은 감독의 영화)
- $\mathcal P_2$ = "movie → actor → movie" (같은 배우의 영화)
- $\mathcal P_3$ = "movie → genre → movie" (같은 장르의 영화)

각 meta-path 별 GNN 결과를 **semantic attention** $\beta_{\mathcal P}$ 로 결합 → task-dependent meta-path 중요도 학습.

### 정리 5.5 — HAN 의 표현력

HAN 의 표현력은:
- 단일 meta-path: 해당 homogeneous subgraph 의 GAT 표현력 (1-WL)
- 여러 meta-path 결합: 각 subgraph 의 representation 의 attention-weighted sum — 정확한 1-WL 위계 분석 미해결

실증적으로 R-GCN 보다 우월 (DBLP, ACM, IMDB datasets).

---

## 💻 구현

### 실험 1 — Edge Feature 통합 (간단)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class EdgeMessageLayer(nn.Module):
    """h_i' = σ(W [h_i ; sum_{j} MLP([h_j; e_{ij}])])"""
    def __init__(self, d_node, d_edge, d_out):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_node + d_edge, d_node), nn.ReLU(),
            nn.Linear(d_node, d_node))
        self.update = nn.Linear(2 * d_node, d_out)
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg = self.msg_mlp(torch.cat([x[src], edge_attr], dim=-1))
        agg = scatter_add(msg, dst, dim=0, dim_size=x.size(0))
        return F.relu(self.update(torch.cat([x, agg], dim=-1)))
```

### 실험 2 — R-GCN Layer

```python
class RGCNLayer(nn.Module):
    def __init__(self, d_in, d_out, num_rels, num_bases=None):
        super().__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases or num_rels
        # Self-loop weight
        self.W_0 = nn.Linear(d_in, d_out, bias=False)
        # Basis matrices
        self.V = nn.Parameter(torch.randn(self.num_bases, d_in, d_out) / (d_in ** 0.5))
        # Coefficients (relation × basis)
        if num_bases < num_rels:
            self.a = nn.Parameter(torch.randn(num_rels, num_bases))
        else:
            self.register_buffer('a', torch.eye(num_rels))
    
    def forward(self, x, edge_index, edge_type):
        """
        edge_index: [2, m]
        edge_type: [m] — relation index
        """
        # W_r = sum_b a_{rb} V_b
        W_r = torch.einsum('rb,bij->rij', self.a, self.V)   # [num_rels, d_in, d_out]
        
        src, dst = edge_index
        # Per-edge relation-specific transform
        msgs = torch.einsum('mi,mij->mj', x[src], W_r[edge_type])
        agg = scatter_add(msgs, dst, dim=0, dim_size=x.size(0))
        # Normalize by per-relation degree (간략화: total degree)
        deg = scatter_add(torch.ones_like(dst, dtype=torch.float),
                          dst, dim=0, dim_size=x.size(0))
        agg = agg / (deg.unsqueeze(-1) + 1e-6)
        return F.relu(self.W_0(x) + agg)
```

### 실험 3 — 작은 Knowledge Graph 예제

```python
# Toy KG: 5 entities, 3 relations
# Triplets: (0, 0, 1), (1, 1, 2), (0, 0, 3), (3, 1, 4), (2, 2, 4)
n_entities, n_rels = 5, 3
edge_index = torch.tensor([[0, 1, 0, 3, 2], [1, 2, 3, 4, 4]], dtype=torch.long)
edge_type = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)

x = torch.randn(n_entities, 8)
layer = RGCNLayer(d_in=8, d_out=4, num_rels=n_rels, num_bases=2)
h_new = layer(x, edge_index, edge_type)
print(f'R-GCN output: {h_new.shape}')   # [5, 4]
print(f'Parameters: {sum(p.numel() for p in layer.parameters())}')
```

### 실험 4 — DistMult Decoder

```python
class DistMult(nn.Module):
    def __init__(self, num_rels, d):
        super().__init__()
        self.r_emb = nn.Embedding(num_rels, d)
    
    def forward(self, h_s, r, h_t):
        """h_s, h_t: [batch, d], r: [batch] (rel ids)"""
        return (h_s * self.r_emb(r) * h_t).sum(-1)
    
    def score_all_targets(self, h_s, r, all_h):
        """Score s-r-? for all entities. h_s: [d], r: scalar, all_h: [n, d]"""
        return (h_s * self.r_emb(r) * all_h).sum(-1)

# 사용
decoder = DistMult(n_rels, d=4)
h = layer(x, edge_index, edge_type)   # encoded entity embeddings
# 학습: positive triplet (s, r, t), negative triplet (s, r, t') 
# loss = -log σ(score_pos) - log σ(-score_neg)
```

### 실험 5 — HAN-style Meta-path Subgraph

```python
# 가상의 heterogeneous graph
# 3 types: movie (5), director (2), actor (3)
# Edges: movie-director, movie-actor

# Meta-path 1: movie -> director -> movie (M-D-M)
# Meta-path 2: movie -> actor -> movie (M-A-M)

# 단순화: meta-path 따라 homogeneous subgraph 직접 구성

def meta_path_subgraph_movie_director_movie(movies, directors, m_d_edges, d_m_edges):
    """M-D-M: 같은 director 가 만든 영화 쌍"""
    edges = []
    for d in directors:
        ms = [m for m in movies if (m, d) in m_d_edges]
        for m1 in ms:
            for m2 in ms:
                if m1 != m2:
                    edges.append((m1, m2))
    return edges

# (실제 HAN 은 meta-path subgraph 위에 GAT 적용 후 semantic attention)
```

---

## 🔗 실전 활용

### 1. Knowledge Graph Completion

**Standard datasets**: FB15k-237, WN18RR, OGB-WikiKG2.

**SOTA 비교**:
- TransE (translation): baseline
- DistMult, ComplEx, RotatE: tensor decomposition
- R-GCN, CompGCN (Vashishth 2020): graph-based encoder + tensor decoder

CompGCN 이 R-GCN 의 successor — relation embedding 도 update.

### 2. Recommendation System

User-item bipartite + side info (genre, price, ...). PinSage, LightGCN, NGCF (He 2020) — heterogeneous adaptation.

### 3. Drug-Drug Interaction

Drug type, target type, side-effect type — 분자 + KG 통합. Decagon (Zitnik 2018).

### 4. PyG `HeteroConv`

```python
from torch_geometric.nn import HeteroConv, SAGEConv

conv = HeteroConv({
    ('movie', 'directed_by', 'director'): SAGEConv((-1, -1), 64),
    ('director', 'directed', 'movie'): SAGEConv((-1, -1), 64),
    ('movie', 'starring', 'actor'): SAGEConv((-1, -1), 64),
}, aggr='sum')
out_dict = conv(x_dict, edge_index_dict)
```

### 5. Heterogeneous Graph Transformer (HGT, Hu 2020)

Transformer-based, type-specific attention. 후속 graph transformer 의 heterogeneous 일반화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Discrete relation types | Continuous edge feature 별도 처리 (NNConv, GINE) |
| Fixed type ontology | Schema-free (dynamic relation) 처리 어려움 |
| Basis decomposition 의 rank $B$ | 작은 $B$ 시 표현력 제한, 큰 $B$ 시 overfitting |
| Meta-path 사전 정의 | Meta-path 자동 학습 별도 연구 (MAGNN, GTN) |
| Per-relation degree normalization | Imbalanced relation distribution 시 dominant relation bias |
| Single graph | Multi-graph (Drug A in graph 1, Drug B in graph 2) — graph alignment 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{R-GCN: } h_i^{(l+1)} = \sigma\left(W_0 h_i + \sum_r \sum_{j \in N_r(i)} \frac{1}{c_{i,r}} W_r h_j\right)}$$

$$\boxed{\text{Basis: } W_r = \sum_{b=1}^B a_{rb} V_b \quad (B \ll |\mathcal R|)}$$

$$\boxed{\text{HAN: node-level + semantic-level attention}}$$

| 모델 | 주요 idea |
|------|----------|
| **NNConv (Gilmer)** | Edge feature → weight 자체 generation |
| **R-GCN (Schlichtkrull)** | Relation-specific $W_r$ + basis decomposition |
| **CompGCN (Vashishth)** | Relation embedding 도 update |
| **HAN (Wang)** | Meta-path + 2-level attention |
| **HGT (Hu)** | Type-specific Transformer attention |
| **Knowledge Graph** | Encoder (GNN) + Decoder (DistMult/ComplEx/RotatE) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): R-GCN basis decomposition 으로 $W_r = \sum_{b=1}^B a_{rb} V_b$ 사용 시 파라미터 수를 계산하라. $|\mathcal R| = 100, d = 64, B = 8$ 인 경우.

<details>
<summary>해설</summary>

**Without basis**: $|\mathcal R| \cdot d^2 = 100 \cdot 64^2 = 409,600$

**With basis $B = 8$**:
- $V_b$: $B \cdot d^2 = 8 \cdot 4096 = 32,768$
- $a_{rb}$: $|\mathcal R| \cdot B = 100 \cdot 8 = 800$
- Total: $33,568$

**절감**: $409,600 / 33,568 \approx 12$배.

또한 $W_0$ (self-loop) 추가 $d^2 = 4096$ 양쪽 모두 포함.

**의미**: $|\mathcal R| \gg B$ 일 때 큰 효율 — Wikidata 같은 1000+ relation graph 에서 핵심.

</details>

**문제 2** (심화): Basis decomposition 이 충분히 표현력이 있는지 분석하라. $B$ 가 너무 작으면 어떤 한계가 있고, $B = |\mathcal R|$ 이면 무엇과 동치인가?

<details>
<summary>해설</summary>

**$B = |\mathcal R|$ 시 동치**:
- $a_{rb}$ 가 $|\mathcal R| \times |\mathcal R|$ matrix — full rank 가능
- $V_b$ 가 $|\mathcal R|$ 개 basis
- 학습 후 $W_r = \sum_b a_{rb} V_b$ 가 임의 matrix 표현 가능
- 즉 unconstrained R-GCN 과 동치

단 redundant (over-parameterized): $a$ 의 SVD 로 $V_b$ 와 $W_r$ 사이 trivial transform 가능 → 실질 $B$ 자유도.

**$B$ 가 작을 때 한계**:

$W_r$ 들이 $B$-차원 subspace 내에서 표현 — relation 간 강한 correlation 가정. 만약 두 relation 이 매우 다른 의미 (e.g., "born_in" vs "color_of") 면 같은 basis space 로 잘 표현 못함 → 표현력 손실.

**Empirical**: $B = 10 \sim 30$ 이 보통 좋은 trade-off (Wikidata 의 1000+ relation 대상).

**대안**:
- **Block diagonal**: $W_r = \text{diag}(\ldots)$ — 다른 inductive bias.
- **Mixture of Experts**: 각 relation 이 expert 선택 (sparse activation).
- **Relation-aware attention** (HGT): type-specific attention parameter, basis 보다 풍부.

따라서 basis decomposition 은 단순하지만 R-GCN 의 핵심 trick — relation 수 의 linear scaling 보장.

</details>

**문제 3** (논문 비평): HAN 의 meta-path 가 "사전 정의" 라는 한계가 있다. 자동으로 meta-path 또는 type-aware aggregation 을 학습하는 방법 (GTN, MAGNN, HGT) 을 비교 분석하라.

<details>
<summary>해설</summary>

**HAN 한계**: Meta-path $\mathcal P$ 를 사전 정의 (도메인 지식 필요). 새 task / 새 schema 에서 meta-path 결정이 어려운 문제.

**GTN (Yun 2019) — Graph Transformer Network**:
- 입력: heterogeneous graph $\{A_r\}_{r \in \mathcal R}$
- 학습 가능 weight $\{W_r\}$ 로 weighted sum: $A_{\text{new}} = \sum_r W_r A_r$
- $A_{\text{new}}$ 의 power $A_{\text{new}}^k$ 가 length-$k$ meta-path
- 자동으로 task-relevant meta-path 학습

**MAGNN (Fu 2020)**:
- Meta-path 의 intermediate node feature 도 활용 (HAN 은 시작·끝 노드만)
- Meta-path encoder (RotatE-like)
- 더 풍부한 meta-path representation

**HGT (Hu 2020)**:
- 사전 정의 meta-path 없음
- Type-aware attention: $\text{Attn}(Q_t \cdot K_{t'} \cdot W_{r}) $ — type pair 와 relation 마다 다른 attention parameter
- Layer-wise message 가 자동으로 type-relevant meta-path 학습 (implicit)
- Heterogeneous Graphormer 의 직접 전신

**비교**:

| 모델 | Meta-path 정의 | 표현력 | 효율 |
|------|---------------|--------|------|
| HAN | Manual | Rich (per-path) | $O(P n d^2)$, $P$ = #paths |
| GTN | Auto (linear combination) | Path-as-matmul | $O(K n^2 d)$ — dense matmul |
| MAGNN | Manual | Path-encoder enhanced | similar to HAN |
| HGT | Implicit | Type-aware attention | $O(m d^2)$ — sparse |

**현대 추세**: 
- **HGT-style implicit meta-path** + **graph transformer 통합** (Ch7).
- **Meta-path 자체의 의미** 가 점점 LLM-driven schema understanding 으로 (LLM-augmented heterogeneous graph).

따라서 HAN 의 meta-path 명시성은 interpretability 측면에서 여전히 가치, but expressive power 는 HGT-style 의 implicit 학습이 우월. 후속 Ch7-04 의 LLM 시대 GNN 연계.

</details>

---

<div align="center">

[◀ 이전](./04-gin.md) | [📚 README](../README.md) | [다음 ▶](../ch4-expressive-power/01-wl-test.md)

</div>
