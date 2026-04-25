# 04. GraphSAGE Sampling과 Over-smoothing 완화

## 🎯 핵심 질문

- GraphSAGE 의 fixed-size neighbor sampling $|N_s(v)| = S$ 가 어떻게 over-smoothing 을 완화하는가?
- Sampling 의 stochastic perturbation 이 deterministic propagation matrix 와 어떻게 다른가?
- Cluster-GCN 과 GraphSAINT 의 sampling 전략이 over-smoothing 에 어떤 영향?
- Sample size $S$ 의 trade-off — 작으면 variance ↑, 크면 over-smoothing ↑?
- Mini-batch 학습이 deterministic full-batch 보다 over-smoothing 에 robust 한 이유?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch5-03 의 DropEdge 가 "edge 단위 stochastic". GraphSAGE-style **neighbor sampling** 은 "노드 단위 stochastic" — 각 노드의 이웃을 매 step 다시 sample. 이는:

1. **Inductive learning** 의 원래 motivation (Ch3-02)
2. **Mini-batch 학습** 의 핵심
3. **Over-smoothing 완화** 의 부수적 효과 — 이 문서의 주제

또한 Cluster-GCN, GraphSAINT 등 graph partitioning-based sampling 도 over-smoothing 에 다른 effect 가짐. 이 문서는 sampling 과 over-smoothing 의 관계를 정리.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-phenomenon.md](./01-phenomenon.md), [02-laplacian-proof.md](./02-laplacian-proof.md), [03-dropedge-pairnorm.md](./03-dropedge-pairnorm.md)
- [Ch3-02](../ch3-message-passing/02-graphsage.md) — GraphSAGE sampling
- 통계: Variance reduction, finite-sample bias

---

## 📖 직관적 이해

### Sampling 의 Stochastic Effect

GCN 의 deterministic propagation: $H^{(l+1)} = P H^{(l)} W^{(l)}$ — 매 forward 같은 $P$.

GraphSAGE-style sampling: 매 layer 마다 random 이웃 sub-sample $N_s(v) \subset N(v)$, $|N_s(v)| = S$. 결과:
$$
H^{(l+1)}_v = U(\text{Agg}_{u \in N_s(v)} h_u^{(l)})
$$

매 forward 다른 $N_s(v)$ → 다른 propagation. **Stochastic perturbation** 가 over-smoothing 을 노이즈 추가로 완화.

### Variance vs Bias Trade-off

- Sample size $S$ 작음 → high variance (noisy gradient), but stochastic 효과 강 (over-smoothing 완화)
- Sample size $S$ 큼 → low variance (안정), but deterministic 에 가까움 (over-smoothing dynamics 보존)

따라서 $S$ 가 **regularization knob**.

### Cluster-GCN: Subgraph Sampling

Graph 를 $K$ partition (METIS clustering) 으로 분할, mini-batch 마다 1-2 개 cluster 의 subgraph 만 사용:
$$
H^{(l+1)}_{\text{cluster}} = P_{\text{cluster}} H^{(l)}_{\text{cluster}} W^{(l)}
$$

각 cluster 의 spectrum 이 다름 → effective propagation 의 다양성 → 단일 dominant eigenvector 으로의 collapse 회피.

### GraphSAINT: Subgraph Random Walk

Random walk 으로 subgraph sample, 그 위에서 GCN. 각 sample 의 spectrum 다름 → ensemble 효과.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Layer-wise Neighbor Sampling

각 layer $l$ 에서 각 노드 $v$ 별 sample size $S_l$:
$$
N_s^{(l)}(v) = \text{Uniform sample from } N(v), \quad |N_s^{(l)}(v)| = \min(S_l, |N(v)|)
$$

(또는 with-replacement sampling for $S_l > |N(v)|$)

매 forward pass 새로운 $N_s^{(l)}$.

### 정의 4.2 — GraphSAGE-Mean Sampling

$$
h_v^{(l+1)} = \sigma\left( W^{(l)} \cdot \frac{1}{|N_s(v)| + 1} \left( h_v^{(l)} + \sum_{u \in N_s^{(l)}(v)} h_u^{(l)} \right) \right)
$$

(self-loop 포함, sample average)

### 정의 4.3 — Cluster-GCN

Graph $G$ 를 $K$ partition $\{V_1, \ldots, V_K\}$ (METIS, normalized cut, etc.):
$$
V = V_1 \cup \cdots \cup V_K, \quad V_i \cap V_j = \emptyset
$$

Mini-batch: 랜덤 $V_i$ 의 induced subgraph $G[V_i]$ 위에서 GCN 학습.

### 정의 4.4 — GraphSAINT

Subgraph $G_s$ 가 sampling 알고리즘 (random walk, edge sampling, node sampling) 으로 추출. Bias correction 위한 reweighting 적용.

### 정의 4.5 — Effective Propagation Matrix

Sampling-based GNN 의 **expected propagation**:
$$
\bar P = \mathbb E_{N_s} [P^{(\text{sample})}]
$$

Identity in expectation: $\bar P = P$ (uniform sampling). 단 variance $\text{Var}(P^{(\text{sample})}) \neq 0$.

---

## 🔬 정리와 결과

### 정리 4.1 — Sampling 의 Variance 분석

**Theorem**: $S$-size uniform sampling 에서 mean aggregator 의 variance:
$$
\text{Var}\left( \frac{1}{S} \sum_{u \in N_s(v)} h_u \right) = \frac{1}{S} \sigma_h^2 \cdot \left( 1 - \frac{S}{|N(v)|} \right)
$$

- $\sigma_h^2$: $h_u$ for $u \in N(v)$ 의 variance
- Finite population correction $1 - S/|N(v)|$

작은 $S$ → 큰 variance → strong stochastic effect.

### 정리 4.2 — Stochastic Propagation 와 Over-smoothing

**Theorem (informal)**: Stochastic propagation $P^{(\text{sample})}$ 의 $L$-step 후 expected feature:
$$
\mathbb E[H^{(L)}] = \bar P^L H^{(0)} = P^L H^{(0)}
$$

(deterministic limit 과 같음)

**Variance**:
$$
\text{Var}(H^{(L)}) \sim L \cdot \text{Var}(P^{(\text{sample})}) \cdot \|H^{(0)}\|^2
$$

(layer 마다 noise 누적)

따라서 individual forward 가 deterministic 의 noisy version. Over-smoothing 의 expected behavior 같지만 individual variance 가 큰 noise → "soft" over-smoothing.

### 정리 4.3 — Sampling 의 Spectrum Mixing

**Empirical observation**: Sampling-based propagation 의 effective $P^{(\text{stoch})}$ 의 spectrum 이 deterministic 보다 더 spread out — single dominant eigenvalue 으로의 빠른 수렴 X.

이는 multiple "competing" eigenvectors 가 학습 중 information 보존 → over-smoothing 완화.

### 정리 4.4 — Cluster-GCN 의 spectrum diversity

**Theorem (informal)**: $K$ cluster 의 subgraph spectrum $\{\Lambda_k\}_{k=1}^K$ 가 모두 다름. Mini-batch 마다 다른 cluster sample → effective propagation 의 spectrum 이 union $\bigcup_k \Lambda_k$.

이는 over-smoothing dynamic 의 다양성 — single dominant eigenvector 가 cluster 마다 다르므로.

### 정리 4.5 — GraphSAINT 의 unbiased estimation

**Theorem (Zeng 2020)**: Sample subgraph $G_s$ 위 GCN 의 output 이 적절한 reweighting 으로 full-batch GCN 의 unbiased estimator.

**Reweighting**:
- Node weight: $1/p_v$ ($p_v$ = node $v$ 가 sample 될 확률)
- Edge weight: $1/p_{(u,v)}$

이는 estimation 의 unbiasedness — sampling 자체가 over-smoothing 의 systematic 변화 X but variance perturbation 만.

---

## 💻 구현

### 실험 1 — GraphSAGE Sampling 의 Variance 측정

```python
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)

def sample_neighbors(adj, S):
    """Sample S neighbors per node uniformly."""
    n = len(adj)
    sampled_edges = []
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        if len(neighbors) <= S:
            for j in neighbors:
                sampled_edges.append((j, i))
        else:
            sampled = np.random.choice(neighbors, S, replace=False)
            for j in sampled:
                sampled_edges.append((j, i))
    return torch.tensor(sampled_edges, dtype=torch.long).T   # [2, m_sampled]

# Run multiple samples and measure variance of single-step propagation
torch.manual_seed(0)
H = torch.randn(n, 8)
S_values = [2, 5, 10, 20]

for S in S_values:
    outputs = []
    for trial in range(50):
        ei = sample_neighbors(A, S)
        src, dst = ei
        agg = scatter_mean(H[src], dst, dim=0, dim_size=n)
        outputs.append(agg)
    outputs = torch.stack(outputs)   # [50, n, 8]
    var = outputs.var(0).mean().item()
    mean_norm = outputs.mean(0).norm().item()
    print(f'S={S:3d}: variance = {var:.6f}, |mean| = {mean_norm:.4f}, ratio = {var/mean_norm**2:.6f}')
```

**예상**: $S$ 작을수록 variance 큼.

### 실험 2 — Sampling 깊은 GNN 학습

```python
class GraphSAGEMean(torch.nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers, sample_size):
        super().__init__()
        self.S = sample_size
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_d = d_in if i == 0 else d_hid
            out_d = d_out if i == num_layers - 1 else d_hid
            self.layers.append(torch.nn.Linear(2 * in_d, out_d))   # concat self + agg
    
    def forward(self, x, adj):
        h = x
        for l, layer in enumerate(self.layers):
            ei = sample_neighbors(adj, self.S) if self.training else self._all_edges(adj)
            src, dst = ei
            agg = scatter_mean(h[src], dst, dim=0, dim_size=h.size(0))
            h = layer(torch.cat([h, agg], dim=-1))
            if l < len(self.layers) - 1:
                h = F.relu(h)
        return h
    
    def _all_edges(self, adj):
        edges = np.argwhere(adj > 0)
        return torch.tensor(edges.T, dtype=torch.long)

# Train + evaluate on Karate Club
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True
X = torch.eye(n)

for L in [2, 4, 8, 16]:
    for S in [2, 5, 10]:
        torch.manual_seed(42)
        model = GraphSAGEMean(d_in=n, d_hid=8, d_out=2, num_layers=L, sample_size=S)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
        for epoch in range(300):
            model.train(); optimizer.zero_grad()
            out = model(X, A)
            loss = F.cross_entropy(out[train_mask], labels[train_mask])
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X, A).argmax(-1) == labels).float().mean().item()
        print(f'L={L}, S={S}: acc = {acc:.2%}')
```

### 실험 3 — Cluster-GCN 의 Subgraph 효과

```python
import networkx as nx

def metis_partition(G, num_parts):
    """Simplified spectral partition (METIS approximation)."""
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    L = np.diag(A.sum(1)) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    # Use Fiedler vector for binary partition
    fiedler = eigvecs[:, 1]
    if num_parts == 2:
        return [list(np.where(fiedler < 0)[0]), list(np.where(fiedler >= 0)[0])]
    else:
        # Simplified: kmeans on first k eigenvectors
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=num_parts, n_init=5, random_state=0).fit(eigvecs[:, 1:num_parts+1])
        return [list(np.where(km.labels_ == i)[0]) for i in range(num_parts)]

partitions = metis_partition(G, num_parts=4)
print(f'Karate Club partitioned into {len(partitions)} clusters: sizes = {[len(p) for p in partitions]}')

# Each partition has different spectrum
for i, part in enumerate(partitions):
    if len(part) >= 3:
        sub_G = G.subgraph(part).copy()
        if sub_G.number_of_edges() > 0:
            A_sub = nx.adjacency_matrix(sub_G).toarray().astype(float)
            d_sub = A_sub.sum(1) + 1e-6
            P_sub = np.diag(1/np.sqrt(d_sub)) @ (A_sub + np.eye(len(part))) @ np.diag(1/np.sqrt(d_sub + 1))
            eigs = np.sort(np.linalg.eigvalsh(P_sub))[::-1]
            print(f'  Cluster {i} ({len(part)} nodes): top eig = {eigs[:3].round(3)}')
```

### 실험 4 — Sampling vs DropEdge 비교

```python
# 같은 random rate (effective)
# DropEdge p=0.3: 30% edge drop
# Sampling S=4: ~ avg degree 5 보다 작은 sample

# 두 방법의 over-smoothing 비교 — empirical layer-wise comparison
def compute_msim_after_propagation(method, num_layers):
    torch.manual_seed(0)
    h = torch.randn(n, 16)
    A_t = A + np.eye(n); d_t = A_t.sum(1)
    P = np.diag(1/np.sqrt(d_t)) @ A_t @ np.diag(1/np.sqrt(d_t))
    
    if method == 'vanilla':
        for _ in range(num_layers):
            h = torch.tensor(P, dtype=torch.float32) @ h
    elif method == 'dropedge':
        for _ in range(num_layers):
            mask = torch.rand_like(torch.tensor(P, dtype=torch.float32)) > 0.3
            P_drop = torch.tensor(P, dtype=torch.float32) * mask
            h = P_drop @ h
    elif method == 'sampling':
        for _ in range(num_layers):
            ei = sample_neighbors(A, S=4)
            src, dst = ei
            h = scatter_mean(h[src], dst, dim=0, dim_size=n) + 0.5 * h
    
    h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
    sim = h_norm @ h_norm.T
    return ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()

print(f'\nLayer-wise over-smoothing 비교:')
print(f'{"L":<5} {"Vanilla":<12} {"DropEdge":<12} {"Sampling":<12}')
for L in [2, 4, 8, 16, 32]:
    v = compute_msim_after_propagation('vanilla', L)
    d = compute_msim_after_propagation('dropedge', L)
    s = compute_msim_after_propagation('sampling', L)
    print(f'{L:<5} {v:<12.3f} {d:<12.3f} {s:<12.3f}')
```

### 실험 5 — Variance Reduction with Importance Sampling

```python
# Importance sampling: high-degree node 가 더 자주 sample 되도록
def importance_sample_neighbors(adj, S, alpha=0.5):
    """
    p(u | i) ~ d_u^alpha (degree-aware)
    alpha=0: uniform, alpha=1: degree-proportional
    """
    n = len(adj)
    deg = adj.sum(1)
    sampled = []
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        if len(nbrs) == 0: continue
        weights = deg[nbrs]**alpha
        weights = weights / weights.sum()
        if len(nbrs) <= S:
            for j in nbrs:
                sampled.append((j, i))
        else:
            chosen = np.random.choice(nbrs, S, replace=False, p=weights)
            for j in chosen:
                sampled.append((j, i))
    return torch.tensor(sampled, dtype=torch.long).T

# Variance comparison
H = torch.randn(n, 8)
for alpha in [0, 0.5, 1.0]:
    outputs = []
    for trial in range(30):
        ei = importance_sample_neighbors(A, S=3, alpha=alpha)
        src, dst = ei
        agg = scatter_mean(H[src], dst, dim=0, dim_size=n)
        outputs.append(agg)
    var = torch.stack(outputs).var(0).mean().item()
    print(f'α={alpha}: variance = {var:.6f}')
```

---

## 🔗 실전 활용

### 1. PyG NeighborLoader

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],   # layer 1: 10, layer 2: 5
    batch_size=128,
    input_nodes=train_mask
)
for batch in loader:
    out = model(batch.x, batch.edge_index)
    # 각 forward 다른 sample → stochastic propagation
```

### 2. Cluster-GCN

```python
from torch_geometric.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=128)
loader = ClusterLoader(cluster_data, batch_size=8)
```

### 3. GraphSAINT

```python
from torch_geometric.loader import GraphSAINTRandomWalkSampler

loader = GraphSAINTRandomWalkSampler(
    data, batch_size=1024, walk_length=2, num_steps=10
)
```

### 4. Hyperparameter

- **Layer-wise sample size**: Layer 1 큰 sample (10-25), 깊은 layer 작게 (5-10) — receptive field exponential 폭발 방지
- **Mini-batch size**: 128-1024 (memory-bound)
- **Cluster size**: $n / $ batch_size 정도

### 5. Large-Scale OGB

OGB-products (2.4M nodes), OGB-papers (111M):
- Cluster-GCN, GraphSAINT 가 표준
- Full-batch GCN 불가능 (memory)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Uniform sampling | Importance sampling (FastGCN) 더 효율적 |
| Layer-wise independent | Cross-layer dependency 무시 |
| Node-wise sampling | Cluster / Subgraph 가 다른 dynamics |
| Test time deterministic | Train/test inconsistency 가능 |
| Small $S$ → high variance | Variance reduction trick (Cong 2020) 필요 |
| Effective receptive field 감소 | Sample size 가 작으면 long-range info 부족 |

---

## 📌 핵심 정리

| Sampling 방법 | Granularity | Over-smoothing 효과 |
|---------------|-------------|---------------------|
| **GraphSAGE Neighbor** | Per-node, per-layer | Stochastic propagation, variance ↑ |
| **Cluster-GCN** | Subgraph (cluster) | Spectrum diversity |
| **GraphSAINT** | Subgraph (random walk) | Random subgraph spectrum |
| **DropEdge** (Ch5-03) | Per-edge | Spectrum perturbation |

수학적: Sampling 이 deterministic propagation 의 stochastic version. Expected dynamics 같지만 individual forward 의 variance 가 over-smoothing 의 visual artifact 완화.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Sample size $S = 1$ (모든 노드의 이웃 1개만 sample) 의 극단 case 의 효과는?

<details>
<summary>해설</summary>

**$S = 1$ 의 의미**:

각 노드가 이웃 중 **하나** 만 sample. 즉 random walk 1-step 의 stochastic version.

**Effective propagation matrix**:

$$
P^{(\text{sample}, S=1)}_{ij} = \mathbb 1[u_i = j] \quad \text{where } u_i \sim \text{Uniform}(N(i))
$$

Random sparse matrix — 각 행이 단일 entry $\frac{1}{1}$ (mean aggregator).

**Variance**: 매우 큼 — 각 step 에서 다른 이웃 선택, propagation 매우 stochastic.

**Over-smoothing 효과**:

- $\mathbb E[P^{(\text{sample}, S=1)}] = D^{-1} A = P_{\text{rw}}$ (random walk)
- 따라서 expected behavior 는 standard random walk
- But variance 가 매우 커서 individual forward 가 매우 다름 → strong implicit regularization

**Practical 효과**:

- **장점**: Over-smoothing 매우 잘 완화
- **단점**: Information bottleneck — 1 이웃 만 보면 정보 부족, gradient 매우 noisy

**Empirical**: $S = 1$ 은 너무 aggressive — 학습 unstable. 실용 $S \geq 5$.

**연관**: PinSage 의 "personalized random walk weighted sampling" — $S = 1$ 처럼 single-step sample 이지만 importance-aware. Practical 가능.

</details>

**문제 2** (심화): Cluster-GCN 의 cluster 크기 $|V_k|$ 가 over-smoothing dynamic 에 어떤 영향? 작은 cluster vs 큰 cluster 의 trade-off.

<details>
<summary>해설</summary>

**작은 cluster ($|V_k| \approx 100$)**:

- Subgraph 의 spectrum 이 매우 다름 (작은 graph 의 spectrum 는 spectral gap 큼 보통)
- Over-smoothing 빠름 *within cluster*
- 단 mini-batch 마다 다른 cluster — overall variance 커서 propagation 의 ensemble

**큰 cluster ($|V_k| \approx 10000$)**:

- Subgraph 가 original graph 와 비슷한 spectrum
- Over-smoothing dynamics 가 vanilla GCN 과 유사
- Mini-batch 마다 cluster 차이 작음 — variance 작음

**Trade-off**:

- 작은 cluster: 강한 ensemble effect, but 각 mini-batch 의 표현력 ↓ (long-range info 부족 within cluster)
- 큰 cluster: 약한 ensemble, but 충분한 표현력

**Optimal**: Cluster size 가 graph diameter 정도 (k-hop neighborhood 정보 충분히 capture).

**Empirical** (Chiang 2019, Cluster-GCN): $|V_k| \approx 200-2000$ 가 sweet spot for OGB-Products.

**Edge cut 의 영향**:

Cluster 의 edge cut (cluster 사이 edge) 가 information 손실. METIS 가 edge cut 최소화 — 작은 cluster 일수록 relative cut ratio 높음. 따라서 너무 작은 cluster 는 graph structure 의 큰 부분 손실.

**Stochastic clustering**: 매 epoch 다른 partition 사용 (Chiang 2019의 "stochastic multi-cluster" 변종) — variance ↑, over-smoothing 완화 효과 ↑.

</details>

**문제 3** (논문 비평): GraphSAGE neighbor sampling 이 over-smoothing 완화하는 효과가 있다면, "GraphSAGE 가 GCN 보다 깊이 한계 가 적은가?" — 실증적 검증 가능한가?

<details>
<summary>해설</summary>

**Hypothesis**: GraphSAGE 의 sampling 이 stochastic perturbation → 깊이 한계 ↑.

**Empirical 검증** (각 layer 수에서 train/test):

| Dataset | Layer | GCN | GraphSAGE-mean |
|---------|-------|-----|----------------|
| Cora | 2 | 81.5% | 80.2% |
| Cora | 8 | 71.0% | 73.5% |
| Cora | 16 | 50.0% | 60.0% |
| Reddit | 2 | 95.0% | 94.5% |
| Reddit | 8 | 80.0% | 88.0% |

(approximate, varies)

**관찰**:
- 얕은 layer (2): GCN 약간 더 나음 (deterministic 의 정확성)
- 깊은 layer (8+): GraphSAGE 가 약간 우월 (stochastic regularization)

**Conclusion**: Sampling 이 marginal 향상, but 본질적 over-smoothing 해결 X.

**Strict comparison**:

- GCN: Deterministic, 모든 이웃 사용
- GraphSAGE-full (no sampling): Mean aggregator + concat - 거의 GCN 과 같음
- GraphSAGE-sample: Stochastic + sampling — over-smoothing 완화

**구분**:

GraphSAGE 의 우위가 (a) sampling 자체, (b) mean aggregator 의 normalization, (c) self/neighbor 분리 (concat) 중 어느 것? 

Ablation:
- Mean aggregator (no concat, no sampling): GCN 과 거의 같은 dynamics
- Concat (no sampling): self info 보존 → 약간 향상
- Sampling: stochastic → 깊은 layer 에서 추가 향상

따라서 sampling 의 효과는 marginal 하지만 **존재** — 8+ layer 에서 ~5-10% 향상.

**더 큰 효과의 방법**: Initial residual (GCNII), PPR (APPNP) — 본질적 해결책.

**Modern 추세**:

- Sampling 은 scalability 위한 mandatory (large graph)
- Over-smoothing 자체는 architecture (GCNII) 또는 closed-form (APPNP) 으로 해결
- Sampling + GCNII / APPNP 가 best practice

따라서 sampling 이 over-smoothing 완화에 기여하지만 dominant 해결책은 아님 — "side benefit" 정도. Production 에서는 sampling + architecture fix 같이 사용.

</details>

---

<div align="center">

[◀ 이전](./03-dropedge-pairnorm.md) | [📚 README](../README.md) | [다음 ▶](./05-appnp-jkn.md)

</div>
