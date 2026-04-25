# 03. DropEdge, PairNorm, DGN

## 🎯 핵심 질문

- DropEdge (Rong 2020) 가 random edge removal 로 어떻게 spectrum 을 perturb 하여 over-smoothing 을 늦추는가?
- PairNorm (Zhao 2020) 의 "feature distance preservation" 메커니즘은 무엇인가?
- DGN (Differentiable Group Normalization, Chen 2020) 의 group-wise normalization 의 의미?
- 각 방법의 수학적 trade-off 와 실증 비교는?
- 해결책 selection 의 task-dependent guideline?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch5-02 의 over-smoothing 정리를 우회하는 세 가지 일반적 architecture 변형:

1. **DropEdge**: 학습 중 edge sampling — propagation matrix $P$ 자체를 stochastic perturb
2. **PairNorm**: Layer-wise feature normalization — pairwise distance 의 평균 유지
3. **DGN**: Group-based normalization — 다른 community 끼리 다른 normalization

각각 다른 수학적 메커니즘으로 over-smoothing 완화. 실전에서 pure GCN 의 깊이 한계 (2-3 layer) 를 8-16 layer 까지 확장 가능.

이 문서는 세 방법의 이론과 비교를 정리.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-phenomenon.md](./01-phenomenon.md), [02-laplacian-proof.md](./02-laplacian-proof.md)
- 통계: Random sampling, expectation
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Batch normalization

---

## 📖 직관적 이해

### DropEdge 의 직관

매 학습 step 에서 random subset of edges 를 drop. 즉:
$$
P^{(\text{stoch})} = \tilde D^{-1/2} (M \odot \tilde A) \tilde D^{-1/2}, \quad M_{ij} \sim \text{Bernoulli}(1 - p)
$$

(masked $\tilde A$)

**효과**:
1. **Spectrum perturbation**: $P^{(\text{stoch})}$ 의 eigenvalue 가 deterministic $P$ 와 다름 → 다른 over-smoothing dynamics
2. **Stochastic regularization**: 다양한 propagation 학습 — robust feature
3. **$\ker(L)$ 으로의 결정론적 수렴 회피**: 매 step 다른 $\ker$

### PairNorm 의 직관

매 layer 후 feature vector 의 pairwise distance 를 정규화:

**Step 1 (re-center)**: $h_i \leftarrow h_i - \bar h$ (전체 평균 빼기). 모든 노드가 같은 limit으로 가는 것 차단.

**Step 2 (rescale)**: $h_i \leftarrow s \cdot h_i / \sqrt{\frac{1}{n} \sum_j \|h_j\|^2}$. 평균 norm 을 hyperparameter $s$ 로 고정.

**효과**: Feature distance 의 평균이 layer 마다 일정하게 유지 — collapse 방지.

### DGN 의 직관

Graph 의 community 별로 다른 normalization. Karate club: Mr. Hi 그룹 vs Officer 그룹의 평균이 다른 axis. Over-smoothing 시 두 그룹 모두 같은 평균으로 collapse 하지만, group-aware normalization 이 differential 보존.

```
Pre-PairNorm: 모든 노드 평균 0 으로 -- over-smoothing 막음
Pre-DGN: 그룹별 평균 0 으로 -- 그룹 차이 보존
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — DropEdge (Rong 2020)

매 epoch 또는 batch 에서:
$$
\tilde A^{(\text{drop})}_{ij} = \begin{cases} \tilde A_{ij} & \text{with prob } 1 - p \\ 0 & \text{with prob } p \end{cases}
$$

(독립적으로 각 edge sample)

Propagation matrix:
$$
P^{(\text{drop})} = \tilde D^{(\text{drop})}{}^{-1/2} \tilde A^{(\text{drop})} \tilde D^{(\text{drop})}{}^{-1/2}
$$

Edge drop probability $p \in [0, 1]$. 보통 $p = 0.1 \sim 0.3$.

### 정의 3.2 — PairNorm (Zhao 2020)

Layer 의 output $H^{(l+1)}$ 에 PairNorm 적용:

**Centering**:
$$
H^{(l+1)}_c \leftarrow H^{(l+1)} - \frac{1}{n} \mathbb 1 \mathbb 1^T H^{(l+1)}
$$

(노드별 feature 평균 0 으로)

**Rescaling**:
$$
H^{(l+1)}_{\text{norm}} \leftarrow s \cdot \frac{H^{(l+1)}_c}{\sqrt{\frac{1}{n} \sum_i \|h_i^c\|^2}}
$$

(평균 norm $s$ 로 정규화)

Hyperparameter $s$: 보통 1.

### 정의 3.3 — DGN (Differentiable Group Normalization, Chen 2020)

Group $\{G_k\}_{k=1}^K$ — soft assignment $S \in [0, 1]^{n \times K}$ ($S_{ik}$ = node $i$ 의 group $k$ 비율, $\sum_k S_{ik} = 1$).

각 group 별 normalization:
$$
H_k = \text{LayerNorm}(\{h_i : S_{ik} > 0\})
$$

Final: $H_{\text{out}} = \sum_k S_{ik} H_k$.

Group assignment $S$ 도 learnable.

### 정의 3.4 — DropEdge with Spectrum

**Theorem (Rong 2020 informal)**: $P^{(\text{drop})}$ 의 expected behavior:
$$
\mathbb E[P^{(\text{drop})}] = (1 - p) P + p \cdot (\text{lower-rank version})
$$

(정확하지 않지만 직관)

Eigenvalue distribution 이 vanilla $P$ 와 다름 — 더 spread out, 단일 dominant eigenvector 으로의 빠른 collapse 회피.

---

## 🔬 정리와 결과

### 정리 3.1 — DropEdge 의 Variance Reduction

**Theorem (informal)**: DropEdge 가 GNN 의 representation variance 증가시켜 같은 graph 의 다른 forward pass 가 다른 결과 — over-smoothing 의 stochastic perturbation.

$\mathbb E_{P^{(\text{drop})}} [P^{(\text{drop})}^L H] \neq P^L H$ (with $P$ 의 dominant eigenvector projection)

- Variance: $\sim p \cdot$ graph statistics
- Mean: between $P^L H$ 와 zero (drop 다)

### 정리 3.2 — PairNorm Energy Preservation

**Theorem (Zhao 2020)**: PairNorm-augmented GNN 의 Dirichlet energy:
$$
E(H^{(l+1)}_{\text{PairNorm}}) = s^2 \cdot \text{const}
$$

(layer-independent, $s$ 의 함수)

**증명 sketch**: PairNorm 의 rescaling 후 $H$ 의 평균 norm = $s$. Dirichlet energy = $\sum_{ij \in E} \|h_i - h_j\|^2$ — bounded by $4 \cdot s^2 \cdot m$ (per-edge bound). $\square$

이는 over-smoothing 이 energy → 0 인 것에 비해 PairNorm 가 energy 를 lower-bound 시킴.

### 정리 3.3 — Computational Cost

| Method | Forward cost | Backward cost |
|--------|--------------|---------------|
| Vanilla GCN | $O(L m d^2)$ | same |
| DropEdge | Same (drop 후 더 sparse) | same |
| PairNorm | $O(L (m + n) d^2)$ — extra norm | same |
| DGN | $O(L (m + n K) d^2)$ — group norm | extra |

DropEdge 가 가장 cheap, DGN 이 가장 expensive.

### 정리 3.4 — 표현력 trade-off

| Method | 표현력 영향 |
|--------|-------------|
| **DropEdge** | 학습 중 random — 표현력 약간 ↓ but generalization ↑ |
| **PairNorm** | Information-preserving (rescale 만) — 표현력 보존 |
| **DGN** | Group-aware feature — graph community structure 명시적 학습 |

### 정리 3.5 — Empirical Results (Rong, Zhao, Chen 2020)

**Cora layer-wise accuracy**:

| Layer | Vanilla GCN | + DropEdge | + PairNorm | + DGN |
|-------|-------------|------------|------------|-------|
| 2 | 81.5% | 82.8% | 81.5% | 82.0% |
| 4 | 79.0% | 80.5% | 81.7% | 81.5% |
| 8 | 71.0% | 78.0% | 78.2% | 79.0% |
| 16 | 50.0% | 65.0% | 73.5% | 75.0% |
| 32 | 40.0% | 60.0% | 70.0% | 71.0% |

**관찰**:
- Layer 16+ 에서 DropEdge < PairNorm < DGN
- 단 PairNorm 의 hyperparameter (rescale $s$) tuning 중요
- DGN 이 가장 강력 but 가장 복잡

---

## 💻 구현

### 실험 1 — DropEdge 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import networkx as nx
import numpy as np

def drop_edge(edge_index, p=0.2, training=True):
    if not training or p == 0:
        return edge_index
    src, dst = edge_index
    mask = torch.rand(src.size(0)) > p
    return torch.stack([src[mask], dst[mask]], dim=0)

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
    
    def forward(self, x, edge_index):
        from torch_scatter import scatter_add
        # GCN propagation
        n = x.size(0)
        src, dst = edge_index
        # Compute degree from edge_index
        deg = scatter_add(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=n) + 1
        deg_inv_sqrt = 1.0 / torch.sqrt(deg)
        # Self-loop: x_self contribution
        x_lin = self.W(x)
        out = x_lin * (deg_inv_sqrt * deg_inv_sqrt).unsqueeze(-1)   # self-loop
        # Neighbor messages
        msgs = x_lin[src] * (deg_inv_sqrt[src] * deg_inv_sqrt[dst]).unsqueeze(-1)
        out = out + scatter_add(msgs, dst, dim=0, dim_size=n)
        return out

class DropEdgeGCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers, drop_p=0.2):
        super().__init__()
        self.drop_p = drop_p
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(d_in, d_hid))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(d_hid, d_hid))
        self.layers.append(GCNLayer(d_hid, d_out))
    
    def forward(self, x, edge_index):
        h = x
        for l, layer in enumerate(self.layers):
            ei = drop_edge(edge_index, self.drop_p, self.training)
            h = layer(h, ei)
            if l < len(self.layers) - 1:
                h = F.relu(h)
        return h
```

### 실험 2 — PairNorm

```python
class PairNorm(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        # Center
        x = x - x.mean(0, keepdim=True)
        # Rescale
        rownorm_mean = (x.pow(2).sum(-1).mean()).sqrt() + 1e-6
        return self.scale * x / rownorm_mean

class PairNormGCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers.append(GCNLayer(d_in, d_hid))
        self.norms.append(PairNorm())
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(d_hid, d_hid))
            self.norms.append(PairNorm())
        self.layers.append(GCNLayer(d_hid, d_out))
    
    def forward(self, x, edge_index):
        h = x
        for l, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            h = layer(h, edge_index)
            h = norm(h)
            h = F.relu(h)
        return self.layers[-1](h, edge_index)
```

### 실험 3 — Karate Club Layer-wise 비교

```python
G = nx.karate_club_graph()
n = G.number_of_nodes()
edges = np.array(list(G.edges())).T
edge_index = torch.tensor(np.concatenate([edges, edges[::-1]], axis=1), dtype=torch.long)

X = torch.eye(n)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True

def train_eval(model, num_epochs=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    for epoch in range(num_epochs):
        model.train(); optimizer.zero_grad()
        out = model(X, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
    model.eval()
    return (model(X, edge_index).argmax(-1) == labels).float().mean().item()

# Compare for various layers
import matplotlib.pyplot as plt

results = {'GCN': [], 'DropEdge': [], 'PairNorm': []}
layers_list = [2, 4, 8, 16]

for L in layers_list:
    torch.manual_seed(42)
    results['GCN'].append(train_eval(DropEdgeGCN(n, 8, 2, L, drop_p=0.0)))
    torch.manual_seed(42)
    results['DropEdge'].append(train_eval(DropEdgeGCN(n, 8, 2, L, drop_p=0.2)))
    torch.manual_seed(42)
    results['PairNorm'].append(train_eval(PairNormGCN(n, 8, 2, L)))

print(f'Layer-wise accuracy:')
print(f'{"L":<5} {"GCN":<12} {"DropEdge":<12} {"PairNorm":<12}')
for i, L in enumerate(layers_list):
    print(f'{L:<5} {results["GCN"][i]:<12.2%} {results["DropEdge"][i]:<12.2%} {results["PairNorm"][i]:<12.2%}')

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
for method, accs in results.items():
    ax.plot(layers_list, accs, 'o-', label=method)
ax.set_xlabel('Layer count'); ax.set_ylabel('Accuracy')
ax.set_title('Karate Club: over-smoothing 완화 비교'); ax.legend(); ax.grid()
plt.show()
```

### 실험 4 — Cosine Similarity Layer-wise

```python
def compute_msim(h):
    h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
    sim = h_norm @ h_norm.T
    n = h.size(0)
    return ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()

torch.manual_seed(0)
h = torch.randn(n, 16)
A = nx.adjacency_matrix(G).toarray().astype(float)
A_t = A + np.eye(n); d_t = A_t.sum(1)
P = torch.tensor(np.diag(1/np.sqrt(d_t)) @ A_t @ np.diag(1/np.sqrt(d_t)), dtype=torch.float32)

# Vanilla, with DropEdge, with PairNorm
sims_vanilla, sims_dropedge, sims_pairnorm = [], [], []
pn = PairNorm(scale=1.0)

h_v = h.clone(); h_d = h.clone(); h_p = h.clone()
for L in range(20):
    sims_vanilla.append(compute_msim(h_v))
    sims_dropedge.append(compute_msim(h_d))
    sims_pairnorm.append(compute_msim(h_p))
    
    h_v = P @ h_v
    # DropEdge: 각 step random P
    P_drop = P * (torch.rand_like(P) > 0.2).float()
    h_d = P_drop @ h_d
    # PairNorm
    h_p = pn(P @ h_p)

plt.plot(sims_vanilla, 'o-', label='Vanilla')
plt.plot(sims_dropedge, 's-', label='DropEdge')
plt.plot(sims_pairnorm, '^-', label='PairNorm')
plt.xlabel('Layer'); plt.ylabel('Avg Pairwise Cos Sim')
plt.title('Over-smoothing 측정'); plt.legend(); plt.grid()
plt.show()
```

### 실험 5 — DGN-style Group Norm (간략)

```python
class GroupNorm(nn.Module):
    """Simplified DGN: K groups, soft assignment"""
    def __init__(self, d, K=2):
        super().__init__()
        self.assign = nn.Linear(d, K)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(K)])
    
    def forward(self, x):
        # Soft assignment
        S = F.softmax(self.assign(x), dim=-1)   # [n, K]
        # Per-group LayerNorm (simplified)
        normed = torch.stack([norm(x) for norm in self.norms])   # [K, n, d]
        # Weighted sum
        return (S.unsqueeze(-1) * normed.permute(1, 0, 2)).sum(1)

# DGN-augmented GCN 사용 (구현 생략)
```

---

## 🔗 실전 활용

### 1. Hyperparameter Selection

- **DropEdge $p$**: 보통 0.1~0.3. 작은 graph 0.1, large graph 0.3.
- **PairNorm $s$**: 1.0 표준, 작은 dim 0.5, 큰 dim 2.0 가능.
- **DGN $K$**: 데이터 community 수 추정 (Cora ~7, OGB-Arxiv ~40).

### 2. Combined Methods

DropEdge + PairNorm 같이 사용 가능 — 추가 효과 (각 별 dynamics 다름).

### 3. PyG Implementation

```python
from torch_geometric.utils import dropout_edge

# DropEdge during forward
edge_index_drop, _ = dropout_edge(edge_index, p=0.2, training=self.training)
```

PairNorm 은 separate module 로 추가.

### 4. Adversarial Robustness

DropEdge 가 adversarial graph attack (edge perturbation) 에도 robust — random sampling 이 자연스러운 robustness.

### 5. Heterophilic Graph

Heterophilic graph 에서 over-smoothing 더 심함 (이웃 이질성). PairNorm 가 특히 효과 — feature distance 보존이 heterophily 대응.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| **DropEdge**: edge 가 i.i.d. drop | Graph structure 의 important edge 도 random drop |
| **PairNorm $s$ 고정** | Dataset-specific tuning 필요 |
| **DGN**: group structure 가정 | Group 정의가 어려운 graph 가능 |
| **Layer 간 normalization 독립** | Cross-layer dependency 무시 |
| **Training 만 stochastic** | Test time deterministic — bias 가능 |
| **표현력 보존 가정** | 일부 task 에서 약간의 손실 |

---

## 📌 핵심 정리

| 방법 | 메커니즘 | Best for |
|------|---------|----------|
| **DropEdge** | Random edge sampling → spectrum perturbation | Generalization, robustness |
| **PairNorm** | Feature distance 평균 유지 | Deep GNN (16+ layers) |
| **DGN** | Group-wise normalization | Heterophilic, multi-community |

수학적 본질:
- DropEdge: stochastic $P$ → escape from $\ker(L)$ collapse
- PairNorm: bound Dirichlet energy from below
- DGN: group-specific dynamics 보존

---

## 🤔 생각해볼 문제

**문제 1** (기초): DropEdge $p = 0.5$ 가 매우 aggressive. 이것이 GNN 학습에 미칠 영향과 적정 $p$ 의 결정 기준은?

<details>
<summary>해설</summary>

**$p = 0.5$ 의 문제**:

1. **Information loss**: 절반의 edge 가 drop → 그래프 정보의 절반 학습 못 봄
2. **Variance 폭발**: 각 step 의 propagation 이 매우 다름 → 학습 unstable
3. **Effective receptive field 감소**: 같은 layer 수로 더 작은 hop 까지만 도달

**적정 $p$ 의 결정**:

- **그래프 density**: Sparse graph 는 $p$ 작게 (이미 information 부족), dense graph 는 $p$ 크게.
- **Layer 수**: 깊은 GNN 에서 $p$ 크게 (over-smoothing 더 심하므로).
- **Empirical search**: 0.1, 0.2, 0.3 cross-validation.

**Rong 2020 원본 결과**:
- 2-layer GCN: $p = 0.1$ 최적
- 4-layer: $p = 0.2$
- 8-layer: $p = 0.3$
- 16-layer: $p = 0.4$ 가능
- 32+: $p = 0.5$ 도 의미 있음

**일반 가이드**: $p \approx 0.05 \cdot L$ ($L$ = layer 수). Vanilla 가 약하지면 (sparse graph, low-degree), 더 작게.

</details>

**문제 2** (심화): PairNorm 의 rescaling factor $s$ 를 어떻게 선택해야 하며, $s = 0.5$ vs $s = 2.0$ 의 효과는?

<details>
<summary>해설</summary>

**$s$ 의 의미**: Layer 마다 feature norm 의 평균을 $s$ 로 고정.

**$s$ 작은 경우 ($s = 0.5$)**:
- 각 layer 후 norm $0.5$ → small magnitude
- Information bottleneck — 학습 어려움
- 단 over-smoothing 전혀 없음 (norm 작아서 cosine 도 무의미)

**$s$ 큰 경우 ($s = 2.0$)**:
- Norm 이 큼 → expressive power ↑
- 단 numerical instability 가능 (gradient explode)
- Over-smoothing 일부 회피 but normalization 약함

**Sweet spot ($s = 1.0$)**:
- LayerNorm-like, BatchNorm 의 default
- 대부분의 dataset 에서 best

**Tuning guide**:
- 작은 dim ($d \leq 16$): $s$ 작게 (information density 충분)
- 큰 dim ($d \geq 64$): $s$ 크게 (expressive 필요)
- Scale-sensitive task: validation 으로 search

**대안 — Adaptive $s$**: Learnable scalar $s_l$ per layer (Zhao 2020 in supplementary). 약간의 추가 향상.

</details>

**문제 3** (논문 비평): DropEdge, PairNorm, DGN 이 모두 over-smoothing 완화하지만 본질적 해결인가, 아니면 cosmetic mitigation 인가? GCNII / APPNP 와의 비교.

<details>
<summary>해설</summary>

**Cosmetic vs essential**:

**DropEdge / PairNorm 의 한계**:
- Random / normalization 의 surface-level fix
- 원본 GCN 의 dynamics ($\hat A^L \to v_1 v_1^T$) 자체는 변화 X
- "Over-smoothing 의 visual artifact 제거" — 단 underlying dynamics 동일

**증거**:
- Cosine similarity 는 PairNorm 후 보존되지만, **mutual information** $I(X; H^{(L)})$ 는 여전히 감소 (Oono 2020 의 결과 적용 시)
- DropEdge 가 stochastic 이지만 $\mathbb E[H^{(L)}]$ 의 collapse 막지 못함

**Essential 해결책 (GCNII, APPNP)**:
- **GCNII**: Initial residual $\alpha I$ 가 spectral gap 자체 변화 — $\hat \mu_2 = (1-\alpha)\mu_2 + \alpha$ 로 1 에 더 가까움
- **APPNP**: PPR closed-form 이 모든 hop 의 information geometric 보존 — limit 이 collapse 가 아닌 풍부한 distribution

**비교**:

| 방법 | 본질적 vs Cosmetic | 깊이 한계 (실증) |
|------|---------------------|------------------|
| DropEdge | Cosmetic (random regularization) | ~16 layer |
| PairNorm | Cosmetic (normalization) | ~16 layer |
| DGN | Semi-essential (group-aware) | ~32 layer |
| GCNII | Essential (initial residual) | 64+ layer |
| APPNP | Essential (closed-form PPR) | ∞ (closed-form) |

**Empirical layer-wise 비교 (Cora, Chen 2020)**:

- L=2: 모두 ~81%
- L=8: DropEdge 78%, PairNorm 78%, GCNII 84.5%, APPNP 83%
- L=64: DropEdge 60%, PairNorm 73%, **GCNII 85.4%**, APPNP 83% (saturated)

**결론**:

- DropEdge / PairNorm: **단기 fix** — 8-16 layer 까지 개선
- GCNII / APPNP: **본질적 해결** — 깊이 한계 사실상 제거

**현대 추세**:
- DropEdge / PairNorm: standard regularization (orthogonal to depth issue)
- 깊이가 정말 필요한 task: GCNII, APPNP, **Graphormer** (Ch7-01) — 다른 architecture 자체

따라서 over-smoothing 완화는 hierarchy:
1. Surface (DropEdge / PairNorm): 광범위 사용
2. Architecture (GCNII / APPNP): 깊이 효과 명시적
3. Modern (Graphormer): 완전 다른 paradigm

</details>

---

<div align="center">

[◀ 이전](./02-laplacian-proof.md) | [📚 README](../README.md) | [다음 ▶](./04-sampling-mitigation.md)

</div>
