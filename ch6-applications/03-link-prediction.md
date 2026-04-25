# 03. Link Prediction

## 🎯 핵심 질문

- GNN encoder + decoder 구조에서 link prediction 이 어떻게 구성되는가?
- Inner product, bilinear, MLP decoder 의 trade-off 는?
- Negative sampling 의 중요성과 효율적 sampling 전략은?
- Knowledge graph completion 에서 R-GCN, CompGCN, DistMult, ComplEx, RotatE 의 차이?
- Link prediction 의 표준 evaluation metric (AUC, MRR, Hits@K)?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

**Link prediction** 은 missing edge 예측 — 많은 real-world 응용:
1. **추천 시스템**: user-item graph 의 새 preference 예측
2. **Social network**: 친구 추천 (Facebook People You May Know)
3. **Knowledge graph completion**: Wikidata, Freebase 의 missing fact
4. **Drug-drug interaction**: 새 drug pair 의 interaction 예측

Node classification 과 달리 **pair-wise prediction** — 두 노드의 관계. 이는 GNN encoder 로 node embedding 학습 + decoder 로 pair score 계산의 2-stage.

이 문서는 link prediction pipeline, decoder 선택, negative sampling, knowledge graph completion 을 정리.

---

## 📐 수학적 선행 조건

- [Ch3-05](../ch3-message-passing/05-heterogeneous.md): R-GCN, heterogeneous graph
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Bilinear form, tensor factorization

---

## 📖 직관적 이해

### Encoder-Decoder 구조

1. **Encoder**: GNN 으로 node embedding $h_v \in \mathbb R^d$
2. **Decoder**: $(h_u, h_v) \mapsto s_{uv} \in \mathbb R$ — edge score
3. **Loss**: Binary cross-entropy on positive edges + negative samples

$$
\mathcal L = -\log \sigma(s_{uv}^{+}) - \sum_{v_n} \log \sigma(-s_{u v_n}^{-})
$$

### Decoder Families

**Inner product**: $s_{uv} = h_u^T h_v$ — 간단, symmetric.

**Bilinear**: $s_{uv} = h_u^T W h_v$ — learnable interaction.

**MLP**: $s_{uv} = \text{MLP}([h_u; h_v])$ — most flexible.

**Distance-based**: $s_{uv} = -\|h_u - h_v\|^2$ — euclidean similarity.

### Negative Sampling 의 필요성

Link prediction 은 positive edges (존재) + **explicit negative edges** (없는 pair) 필요. 단순:
- **Uniform negative**: 랜덤 pair $(u, v)$ with no edge
- **Hard negative**: 비슷한 nodes 사이의 missing edge (challenging)

Training: $K$ negative per positive — BCE 의 balanced.

### Knowledge Graph 의 Triplet

KG: $(s, r, t)$ — subject, relation, target. Link prediction = triplet completion:
- Given $(s, r, ?)$: 모든 $t$ 에 대해 score 매김, top-k ranking
- Given $(?, r, t)$: similar for subject

표준 decoder (KG-specific):
- **DistMult**: $s = \langle h_s, r, h_t \rangle$ (trilinear)
- **ComplEx**: complex-valued embedding, $s = \text{Re}(\langle h_s, r, \bar h_t \rangle)$
- **RotatE**: $h_t = h_s \circ r$ (Hadamard, rotation in complex plane)

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Link Prediction Problem

Graph $G = (V, E)$, observed edges $E$. Predict probability $p(e) = P[e \in E]$ for $e \notin E$ (미관측).

### 정의 3.2 — Standard Pipeline

```python
# 1. Split: train edges, val edges (positive + negative), test edges
# 2. GNN encoder on train graph → {h_v}
# 3. For each (u, v) edge:
#    s_uv = decoder(h_u, h_v)
# 4. Binary cross-entropy on (positive, negative) pairs
# 5. Evaluate: AUC, Hits@K on held-out edges
```

### 정의 3.3 — Inner Product Decoder (GCN-VAE)

$$
s_{uv} = h_u^T h_v
$$

$P[(u, v) \in E] = \sigma(s_{uv})$.

Kipf-Welling 2016 "Variational Graph Autoencoders".

### 정의 3.4 — DistMult Decoder

Knowledge graph (multi-relation):
$$
s(h_s, r, h_t) = \sum_d h_{s,d} \cdot r_d \cdot h_{t,d}
$$

(elementwise product + sum = trilinear)

단점: symmetric — $s(s, r, t) = s(t, r, s)$. Asymmetric relation 처리 못함.

### 정의 3.5 — ComplEx Decoder

Complex-valued embedding $h_s, h_t, r \in \mathbb C^d$:
$$
s(h_s, r, h_t) = \text{Re}\left( \sum_d h_{s,d} \cdot r_d \cdot \overline{h_{t,d}} \right)
$$

(conjugate on $h_t$ — asymmetric 가능)

### 정의 3.6 — RotatE Decoder

$$
h_t = h_s \circ r, \quad |r_d| = 1
$$

($r$ = rotation in complex plane, $\circ$ = Hadamard product)

$$
s(h_s, r, h_t) = -\|h_s \circ r - h_t\|
$$

(거리 기반, inversion/composition relation 까지 표현 가능)

### 정의 3.7 — Evaluation Metrics

**AUC (Area Under ROC)**:
$$
\text{AUC} = P[s_{\text{pos}} > s_{\text{neg}}]
$$

**Hits@K**: Rank-based. 각 test triplet 에 대해 target 이 top-K 에 있는지.

**MRR (Mean Reciprocal Rank)**:
$$
\text{MRR} = \frac{1}{N} \sum_i \frac{1}{\text{rank}_i}
$$

---

## 🔬 정리와 결과

### 정리 3.1 — Inner Product Decoder 의 한계

**Theorem**: Inner product 는 symmetric — $h_u^T h_v = h_v^T h_u$. 따라서 **directed graph** 또는 **asymmetric relation** 표현 불가.

**해결**: Bilinear $h_u^T W h_v$ with non-symmetric $W$ — asymmetric 가능.

### 정리 3.2 — DistMult 의 Asymmetry 한계

**Theorem**: DistMult 의 elementwise product 가 symmetric. 따라서 (bob, mother, alice) 와 (alice, mother, bob) 를 구분 못함.

**실전**: Wordnet 의 hypernym (is-a) relation 은 asymmetric. DistMult 가 이를 학습 못함.

ComplEx, RotatE 가 해결.

### 정리 3.3 — Negative Sampling 의 영향

**Empirical (Bordes 2013)**: 
- 1 negative per positive: slow convergence
- 10 negatives: OK
- 50+ negatives: diminishing returns

**Hard negative mining**: similar entity 간 negative — 학습 효과 ↑ but bias 위험.

### 정리 3.4 — Knowledge Graph Embedding 성능

**FB15k-237** (표준 KG benchmark):

| Model | MRR | Hits@10 |
|-------|-----|---------|
| TransE | 0.26 | 0.42 |
| DistMult | 0.29 | 0.44 |
| ComplEx | 0.32 | 0.50 |
| RotatE | 0.34 | 0.53 |
| R-GCN + DistMult | 0.25 | 0.42 |
| CompGCN (GNN+ComplEx) | 0.37 | 0.54 |

GNN encoder + tensor decoder 가 tensor-only 보다 우월 (CompGCN).

### 정리 3.5 — Training/Test Split 의 Leakage

**Edge split 이 주의 필요**:
- Test edges 를 training graph 에 포함 X (obvious)
- 그러나 test edges 의 node pair 가 training 에서 connected path (via intermediate nodes) 이면 partial leakage

**Strict split**: Time-based (예: Nov 2019 이전 edge 로 학습, 이후로 test) — real-world 설정.

---

## 💻 구현

### 실험 1 — GCN + Inner Product Link Prediction

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_scatter import scatter_add

class GCNEncoder(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.conv1 = nn.Linear(d_in, d_hid)
        self.conv2 = nn.Linear(d_hid, d_out)
    def forward(self, x, edge_index, A_hat):
        x = F.relu(A_hat @ self.conv1(x))
        return A_hat @ self.conv2(x)

class InnerProductDecoder(nn.Module):
    def forward(self, h_u, h_v):
        return (h_u * h_v).sum(-1)

# Karate Club link prediction
G = nx.karate_club_graph()
n = G.number_of_nodes()
edges = list(G.edges())
np.random.seed(42)
np.random.shuffle(edges)

# Train 80%, test 20%
split = int(0.8 * len(edges))
train_edges = edges[:split]
test_edges = edges[split:]

# Train graph (observed)
G_train = nx.Graph(train_edges)
G_train.add_nodes_from(range(n))
A_train = nx.adjacency_matrix(G_train, nodelist=range(n)).toarray().astype(float)
A_t = A_train + np.eye(n); d_t = A_t.sum(1)
A_hat = torch.tensor(
    np.diag(1/np.sqrt(d_t)) @ A_t @ np.diag(1/np.sqrt(d_t)),
    dtype=torch.float32
)

X = torch.eye(n)
encoder = GCNEncoder(n, 16, 8)
decoder = InnerProductDecoder()

# Negative sampling
def sample_negatives(A, num_neg):
    negs = []
    n = len(A)
    while len(negs) < num_neg:
        u, v = np.random.randint(n, size=2)
        if u != v and A[u, v] == 0:
            negs.append((u, v))
    return negs

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
for epoch in range(100):
    encoder.train()
    h = encoder(X, None, A_hat)
    
    # Positive scores
    pos_u = [e[0] for e in train_edges]; pos_v = [e[1] for e in train_edges]
    pos_scores = decoder(h[pos_u], h[pos_v])
    
    # Negative scores
    neg_edges = sample_negatives(A_train, len(train_edges))
    neg_u = [e[0] for e in neg_edges]; neg_v = [e[1] for e in neg_edges]
    neg_scores = decoder(h[neg_u], h[neg_v])
    
    # BCE loss
    loss = -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Evaluate
encoder.eval()
with torch.no_grad():
    h = encoder(X, None, A_hat)
    test_u = [e[0] for e in test_edges]; test_v = [e[1] for e in test_edges]
    test_pos = decoder(h[test_u], h[test_v]).numpy()
    # Sample negatives for test
    test_neg_edges = sample_negatives(A_train, len(test_edges))
    test_neg_u = [e[0] for e in test_neg_edges]; test_neg_v = [e[1] for e in test_neg_edges]
    test_neg = decoder(h[test_neg_u], h[test_neg_v]).numpy()

# AUC
from sklearn.metrics import roc_auc_score
y_true = [1]*len(test_pos) + [0]*len(test_neg)
y_scores = np.concatenate([test_pos, test_neg])
auc = roc_auc_score(y_true, y_scores)
print(f'Karate Club link prediction AUC: {auc:.3f}')
```

### 실험 2 — Bilinear Decoder

```python
class BilinearDecoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) / d**0.5)
    def forward(self, h_u, h_v):
        return (h_u @ self.W * h_v).sum(-1)
```

### 실험 3 — DistMult for Knowledge Graph

```python
class DistMult(nn.Module):
    def __init__(self, num_rels, d):
        super().__init__()
        self.r_emb = nn.Embedding(num_rels, d)
    def forward(self, h_s, r_ids, h_t):
        r = self.r_emb(r_ids)
        return (h_s * r * h_t).sum(-1)

# 사용: score(s, r, t) = (h_s * r * h_t).sum
# 학습: positive triplet (s, r, t) vs negative (s, r, t_rand)
```

### 실험 4 — RotatE (Complex-valued)

```python
class RotatE(nn.Module):
    def __init__(self, num_rels, d):
        super().__init__()
        self.d = d
        # Relation as rotation: |r_d| = 1
        self.r_theta = nn.Embedding(num_rels, d)
    def forward(self, h_s, r_ids, h_t):
        # h_s, h_t: [batch, 2d]  (split real/imaginary halves)
        h_s_re, h_s_im = h_s[:, :self.d], h_s[:, self.d:]
        h_t_re, h_t_im = h_t[:, :self.d], h_t[:, self.d:]
        r = self.r_theta(r_ids)
        r_re, r_im = torch.cos(r), torch.sin(r)
        # Rotate h_s by r: (h_s * r)
        rot_re = h_s_re * r_re - h_s_im * r_im
        rot_im = h_s_re * r_im + h_s_im * r_re
        # Distance
        return -torch.sqrt((rot_re - h_t_re)**2 + (rot_im - h_t_im)**2 + 1e-8).sum(-1)
```

### 실험 5 — Evaluation (AUC, Hits@K, MRR)

```python
def evaluate_kg_link_prediction(h, decoder, test_triplets, all_entities, num_samples=100):
    """For each test triplet (s, r, t), rank t among all entities."""
    ranks = []
    for (s, r, t) in test_triplets:
        h_s_batch = h[[s] * len(all_entities)]
        r_batch = torch.tensor([r] * len(all_entities))
        h_t_batch = h[all_entities]
        scores = decoder(h_s_batch, r_batch, h_t_batch)
        rank = (scores > scores[all_entities.index(t)]).sum().item() + 1
        ranks.append(rank)
    mrr = np.mean([1/r for r in ranks])
    hits_10 = np.mean([r <= 10 for r in ranks])
    hits_3 = np.mean([r <= 3 for r in ranks])
    return {'MRR': mrr, 'Hits@10': hits_10, 'Hits@3': hits_3}
```

---

## 🔗 실전 활용

### 1. 추천 시스템 (Industry)

- **Netflix**: User-movie bipartite graph + collaborative filtering
- **Amazon**: User-item + co-purchase relation
- **Pinterest PinSage**: 30B node visual search
- **Google Maps**: Location graph link prediction

### 2. Knowledge Graph

- **Freebase, Wikidata**: Missing fact completion
- **Medical KG**: Drug-disease relation
- **Academic KG**: Paper-author-venue triplet

### 3. Social Network

- **LinkedIn People You May Know**: Graph-based friend suggestion
- **Twitter Who to Follow**: Similar user recommendation

### 4. PyG Implementation

```python
from torch_geometric.utils import negative_sampling

# Sample negatives with same #positives
neg_edge_index = negative_sampling(
    edge_index=data.edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=data.edge_index.size(1)
)

# 학습 loop
for epoch in ...:
    h = encoder(x, edge_index)
    pos_scores = decoder(h, data.edge_index)
    neg_scores = decoder(h, neg_edge_index)
    loss = -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()
```

### 5. DGL-KE (KG Embedding Library)

```python
import dgl.nn as dglnn
# ... KG embedding with TransE, ComplEx, RotatE in scalable form
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Pairwise scoring | n-ary relation (multi-way) 처리 어려움 |
| Homogeneous edge | Relation-specific 시 별도 model (CompGCN) |
| Uniform negative sampling | Hard negative mining 중요 but biased |
| Transductive split | Inductive (new nodes) 어려움 |
| Static graph | Temporal link prediction 별도 |
| Top-K ranking metric | Calibration 중요 |

---

## 📌 핵심 정리

| Decoder | Symmetric | Expressiveness | Best for |
|---------|-----------|----------------|----------|
| **Inner Product** | ✓ | Limited | Simple, undirected |
| **Bilinear $h^T W h'$** | ✗ (if non-sym) | Medium | General |
| **MLP** | ✗ | High | Flexible, expensive |
| **DistMult** | ✓ | Limited | Symmetric KG relations |
| **ComplEx** | ✗ | High | Asymmetric relations |
| **RotatE** | ✗ | Composition | Inversion, composition relations |

**Evaluation**: AUC (general), Hits@K / MRR (ranking-aware).

**Negative sampling**: 10-50 per positive, uniform or hard.

**Encoder + Decoder**: GNN (R-GCN, CompGCN) + tensor factorization — strong combination.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Inner product decoder 가 symmetric 이라 directed graph 에서 부적합. 이를 non-symmetric 하게 만드는 2가지 방법은?

<details>
<summary>해설</summary>

**Method 1: Separate Head/Tail Projection**

Encoder 출력 $h_v$ 에 두 가지 projection:
$$
h_v^{\text{head}} = W_h h_v, \quad h_v^{\text{tail}} = W_t h_v
$$

Decoder: $s_{u \to v} = h_u^{\text{head}^T} \cdot h_v^{\text{tail}}$

$W_h \neq W_t$ ⟹ $s_{u \to v} \neq s_{v \to u}$ (asymmetric).

**Method 2: Bilinear with Non-symmetric Matrix**

$$
s_{uv} = h_u^T W h_v
$$

with $W \neq W^T$ (general matrix).

$s_{uv} - s_{vu} = h_u^T (W - W^T) h_v \neq 0$ if $W$ asymmetric.

**Comparison**:

- Method 1: 간단, GNN 의 output 을 두 번 project. 잘 작동.
- Method 2: 더 general, learnable pairwise interaction.

**KG relation 의 경우**:
DistMult 가 Method 3 처럼 elementwise product 사용 but symmetric.
ComplEx 가 conjugate 로 asymmetry 부여 (Method 1 의 복소수 version).

**Empirical**: DistMult (symmetric) < ComplEx (asymmetric) on FB15k-237 — asymmetric encoding 이 실전 중요.

</details>

**문제 2** (심화): Hard negative sampling (비슷한 pair 중 negative) 가 uniform 보다 왜 우위이며, bias 는 무엇인가?

<details>
<summary>해설</summary>

**Hard negative 의 동기**:

Uniform negative: 랜덤 pair — 대부분 trivially distinguishable (e.g., 같은 domain 도 아닌 entity). Gradient small.

Hard negative: similar entity pair (e.g., (cat, dog) 같은 domain) — challenging. Gradient large → faster learning.

**구현**:

1. **Embedding-similarity based**: $h_u$ 와 비슷한 $h_v$ 선택 — top-K nearest neighbor.
2. **Co-occurrence based**: 같은 context (community) 내 entity.
3. **Learned**: GAN-style adversarial negative generation.

**Bias**:

1. **Overfitting to hard examples**: 너무 hard 면 impossible, 학습 못함.

2. **False negative**: Missing edge 를 hard negative 로 sample → 모델이 정답 edge 를 "not edge" 로 학습.
   - 특히 KG 의 incomplete nature 에서 심각.

3. **Sampling bias**:
   - Popular entity (high-degree) 가 over-sample
   - Evaluation 에도 반영 — test metric inflation

**Mitigation**:

1. **Mixture**: 50% uniform + 50% hard
2. **False negative filtering**: Known edges 를 negative pool 에서 제외
3. **Self-adversarial negative sampling** (RotatE 원본): Dynamic reweighting based on current model's prediction.
4. **Subsampling frequency-based**: Popular entity 의 sample 확률 down-weight.

**Empirical**:

- Uniform: baseline
- Hard (top-5 nearest): +3% AUC
- Self-adversarial: +5% AUC, fast convergence

**Modern recommendation**: RotatE 의 self-adversarial 또는 mixed sampling — bias control + fast learning.

</details>

**문제 3** (논문 비평): KG completion 의 score function 들 (TransE, DistMult, ComplEx, RotatE) 이 표현 가능한 relation pattern (symmetric, asymmetric, inversion, composition) 을 비교 분석하라.

<details>
<summary>해설</summary>

**Relation patterns**:
- **Symmetric**: $(s, r, t) \Leftrightarrow (t, r, s)$ (e.g., sibling, married_to)
- **Asymmetric**: $(s, r, t)$ true but $(t, r, s)$ false (e.g., parent_of, author_of)
- **Inversion**: $r_1$ inverse of $r_2$ — $(s, r_1, t) \Leftrightarrow (t, r_2, s)$ (e.g., parent_of vs child_of)
- **Composition**: $r_3$ 이 $r_1 \circ r_2$ — $(s, r_1, x) \wedge (x, r_2, t) \Rightarrow (s, r_3, t)$ (e.g., grandparent = parent of parent)

**TransE** ($t = s + r$, translation):
- Symmetric: ✗ ($r + s = t$ 와 $r + t = s$ 요구 → $r = 0$)
- Asymmetric: ✓
- Inversion: ✓ ($r_1 = -r_2$)
- Composition: ✓ ($r_3 = r_1 + r_2$)

**DistMult** ($s(s, r, t) = \sum h_s \odot r \odot h_t$):
- Symmetric: ✓
- Asymmetric: ✗ (elementwise product commutative)
- Inversion: ✗
- Composition: ✓ (limited)

**ComplEx** (complex-valued):
- Symmetric: ✓
- Asymmetric: ✓ (conjugate 로)
- Inversion: ✓
- Composition: limited

**RotatE** ($h_t = h_s \circ r$, rotation in complex plane):
- Symmetric: ✓ ($r = 1$ 또는 $r = -1$)
- Asymmetric: ✓
- Inversion: ✓ ($r_1 \cdot r_2 = 1$)
- Composition: ✓ ($r_3 = r_1 \cdot r_2$ rotation 합성)

**요약 표**:

| | Sym | Asym | Inv | Comp |
|---|-----|------|-----|------|
| TransE | ✗ | ✓ | ✓ | ✓ |
| DistMult | ✓ | ✗ | ✗ | part |
| ComplEx | ✓ | ✓ | ✓ | part |
| **RotatE** | **✓** | **✓** | **✓** | **✓** |

RotatE 가 4 가지 모두 표현 가능 — empirical best.

**실전 성능 (FB15k-237 MRR)**:
- TransE: 0.29
- DistMult: 0.29
- ComplEx: 0.32
- **RotatE**: 0.34

**Modern beyond**:

- **HAKE** (Zhang 2020): Hierarchical relation → modulus + phase
- **QuatE** (Zhang 2019): Quaternion space
- **BoxE** (Abboud 2020): Geometric box representation

**GNN + KG**:

CompGCN (Vashishth 2020) = R-GCN encoder + ComplEx/RotatE decoder. Strong combination — node representation 이 structural info 반영 → decoder 의 base embedding quality ↑.

따라서 score function selection 이 task-specific (relation type 에 dependent). RotatE 가 범용 baseline, GNN encoder 가 일반적 추가 이득. 

최근 trend: **Pre-trained KG embedding + LLM** — LLM 의 semantic knowledge + KG 의 structural precision.

</details>

---

<div align="center">

[◀ 이전](./02-graph-classification.md) | [📚 README](../README.md) | [다음 ▶](./04-graph-generation.md)

</div>
