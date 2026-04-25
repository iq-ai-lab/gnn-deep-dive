# 01. Over-smoothing 현상

## 🎯 핵심 질문

- 깊은 GCN 에서 노드 feature 가 collapse 하는 over-smoothing 이 정확히 무엇인가?
- Pairwise cosine similarity 가 1 에 수렴하는 현상을 어떻게 측정하는가?
- 왜 2-3 layer 가 GCN 의 sweet spot 이고 그 너머는 성능 저하되는가?
- Karate Club, Cora, Citeseer 에서 깊이별 정확도가 어떻게 단조 감소하는가?
- Over-smoothing 이 Residual connection 없는 GNN 의 근본 한계라는 의미?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

CNN 에서는 layer 를 깊게 (ResNet-152) 쌓아 성능 향상. GNN 도 같은 직관으로 깊은 layer 시도되지만 **반대 결과**: 2-3 layer 후 성능 단조 감소.

Li et al. 2018 (**"Deeper Insights into Graph Convolutional Networks"**) 가 이 현상을 이론·실증으로 분석:

1. **현상**: Layer 깊어질수록 노드 feature pairwise similarity ↑ → 모든 노드 같아짐 (collapse)
2. **결과**: Node classification 성능 단조 감소
3. **원인**: GCN 의 propagation matrix $\hat A$ 의 $L \to \infty$ 극한 — Ch5-02 에서 수학적 증명

이는 GNN 의 **근본적 한계** — 단순 message passing 만으로 깊이 효과 X. 해결책 (DropEdge, PairNorm, APPNP, JKN) 은 Ch5-03~05.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch2-03](../ch2-spectral-gcn/03-gcn-derivation.md) — GCN propagation
- [Ch1-04](../ch1-graph-laplacian/04-spectral-theory.md) — Spectral theory
- 통계: Cosine similarity, pairwise distance

---

## 📖 직관적 이해

### Over-smoothing 의 본질

GCN propagation: $H^{(l+1)} = \sigma(\hat A H^{(l)} W^{(l)})$. $\hat A$ 가 weighted average — 노드 자신 + 이웃 의 평균.

**한 layer 의 효과**: 노드 feature 가 이웃과 비슷해짐 — 1-step smoothing.

**여러 layer 의 효과**: 점점 더 큰 area 에서의 smoothing → 모든 노드 feature 가 graph 전체의 average 로 수렴.

**결과**: 노드 자체의 구별 정보 손실. 모든 노드가 같은 prediction → random guess.

### Pairwise Similarity 측정

$$
\text{sim}(L) = \frac{1}{\binom{n}{2}} \sum_{i < j} \frac{h_i^{(L)} \cdot h_j^{(L)}}{\|h_i^{(L)}\| \|h_j^{(L)}\|}
$$

Layer $L$ 의 모든 노드 쌍의 cosine similarity 평균. $L \to \infty$ 시 1 에 수렴 (모든 노드 같은 방향).

### Smoothness vs Over-smoothing

- **Smoothness (적당)**: 인접 노드 비슷 — graph 의 local structure 활용. 좋음.
- **Over-smoothing**: 모든 노드 같음 — graph structure 정보 완전 손실. 나쁨.

GCN 에서 이 두 사이의 sweet spot 이 매우 좁음 (보통 2-3 layer).

### CNN 과의 대비

CNN 의 ResNet 이 깊이로 성능 향상. 반면 GCN 은 깊이로 성능 저하.

**차이**:
- CNN: receptive field 확장 = local pattern 의 hierarchical composition
- GCN: receptive field 확장 = 전체 graph 평균 → information collapse

이는 **graph 의 finite-diameter** 가 결정적 — short distance 안에 graph 전체 도달. CNN 의 image 는 large size, hierarchical composition 가능.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Over-smoothing

GNN 의 representation $H^{(L)}$ 이 layer $L$ 증가에 따라 **모든 노드의 feature 가 비슷해지는 현상**:
$$
\lim_{L \to \infty} h_i^{(L)} = \lim_{L \to \infty} h_j^{(L)} \quad \forall i, j \in V
$$

(또는 같은 $\ker$ subspace 로의 projection)

### 정의 1.2 — Pairwise Similarity Metric

여러 측정 metric:

**Mean cosine similarity** (Li 2018):
$$
\text{MSim}(H) = \frac{2}{n(n-1)} \sum_{i<j} \frac{h_i^T h_j}{\|h_i\| \|h_j\|}
$$

**Mean Average Distance (MAD, Chen 2020)**:
$$
\text{MAD}(H) = \frac{2}{n(n-1)} \sum_{i<j} (1 - \cos(h_i, h_j))
$$

(MSim 의 보완 — 작을수록 over-smoothed)

**Dirichlet Energy** (Cai 2020):
$$
E(H) = \text{tr}(H^T L H) = \sum_{(i,j) \in E} \|h_i - h_j\|^2
$$

(graph Laplacian quadratic form)

### 정의 1.3 — Critical Layer

성능이 최대인 layer 수 $L^*$. 보통 $L^* \in \{2, 3\}$ for GCN. $L > L^*$ 부터 단조 감소.

### 정의 1.4 — Information Loss

Information-theoretic: $H^{(L)}$ 의 mutual information with input $X$ 가 layer 증가에 따라 감소:
$$
I(X; H^{(L)}) \to 0 \text{ as } L \to \infty
$$

(data processing inequality, but tight)

---

## 🔬 정리와 결과

### 정리 1.1 — GCN Layer 가 Smoothing Operation

**Theorem (Li 2018, informal)**: GCN propagation $\hat A H$ 가 graph Laplacian smoothing — Dirichlet energy $E(H) = H^T L H$ 가 layer 마다 감소.

**증명 sketch**: $H' = \hat A H$ 시:
$$
E(H') = (H')^T L H' = (\hat A H)^T L (\hat A H) = H^T \hat A^T L \hat A H \leq H^T L H
$$

(spectral 분석: $\hat A$ 가 low-pass filter, $L$ smoothing 이 누적)

### 정리 1.2 — Empirical Layer-wise Performance

**Empirical observation (Li 2018)**:

| Dataset | Layer=2 | Layer=4 | Layer=8 | Layer=16 |
|---------|---------|---------|---------|----------|
| Cora | 81.5% | 79.0% | 71.0% | 50.0% |
| Citeseer | 70.3% | 68.0% | 60.0% | 38.0% |
| Pubmed | 79.0% | 77.0% | 70.0% | 55.0% |

(approximate, varies by random seed)

Layer ≥ 4 부터 빠른 감소. Layer 16 은 거의 random guess (Cora 50%, 7 class — random ~14%).

### 정리 1.3 — Over-smoothing 의 Universality

**Theorem (Li 2018)**: GCN 뿐 아니라 GraphSAGE, GAT 도 깊은 layer 에서 over-smoothing.

(GAT 가 attention 으로 약간 완화 가능 but 근본 해결 X — Ch7-02)

### 정리 1.4 — Spectral Gap 의 영향

**Theorem (informal)**: Over-smoothing 속도 $\propto (1 - \tilde \lambda_2)^L$ where $\tilde\lambda_2$ = augmented Laplacian 의 second smallest eigenvalue.

큰 $\tilde\lambda_2$ (well-connected expander) → 빠른 over-smoothing.
작은 $\tilde\lambda_2$ (path-like, low-conductance) → 느린 over-smoothing.

---

## 💻 NumPy/PyTorch 측정

### 실험 1 — Karate Club Over-smoothing

```python
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)

# GCN propagation matrix
A_tilde = A + np.eye(n)
d_tilde = A_tilde.sum(1)
A_hat = np.diag(1/np.sqrt(d_tilde)) @ A_tilde @ np.diag(1/np.sqrt(d_tilde))
A_hat_t = torch.tensor(A_hat, dtype=torch.float32)

# Random initial features
torch.manual_seed(0)
h = torch.randn(n, 16)

similarities = []
energies = []
n_layers = 30

for l in range(n_layers):
    # Pairwise cosine similarity
    h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
    sim = h_norm @ h_norm.T
    avg_sim = (sim.sum() - sim.diag().sum()) / (n * (n - 1))
    similarities.append(avg_sim.item())
    
    # Dirichlet energy
    L = np.diag(A.sum(1)) - A
    L_t = torch.tensor(L, dtype=torch.float32)
    energy = torch.trace(h.T @ L_t @ h).item()
    energies.append(energy)
    
    # GCN propagation (no learnable W for simplicity)
    h = A_hat_t @ h

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(similarities, 'o-')
axes[0].axhline(1.0, color='r', linestyle='--', label='collapse')
axes[0].set_xlabel('Layer L'); axes[0].set_ylabel('Avg pairwise cos sim')
axes[0].set_title('Over-smoothing on Karate Club')
axes[0].legend(); axes[0].grid()

axes[1].semilogy(energies, 's-', color='orange')
axes[1].set_xlabel('Layer L'); axes[1].set_ylabel('Dirichlet Energy (log)')
axes[1].set_title('Energy decay → ker(L)')
axes[1].grid()
plt.tight_layout(); plt.show()
```

### 실험 2 — Layer 별 GCN 정확도 (Cora-style)

```python
# Karate Club 에서 학습된 GCN 의 layer 별 정확도

class DeepGCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in, d_hid))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(d_hid, d_hid))
        self.cls = nn.Linear(d_hid, d_out)
    def forward(self, x, A_hat):
        h = x
        for layer in self.layers:
            h = F.relu(A_hat @ layer(h))
        return self.cls(h)

X = torch.eye(n)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True

results = {}
for L in [1, 2, 3, 4, 8, 16]:
    torch.manual_seed(42)
    model = DeepGCN(d_in=n, d_hid=8, d_out=2, num_layers=L)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    for epoch in range(300):
        model.train(); optimizer.zero_grad()
        out = model(X, A_hat_t)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
    acc = (model(X, A_hat_t).argmax(-1) == labels).float().mean().item()
    results[L] = acc

print('Layer-wise accuracy:')
for L, acc in results.items():
    print(f'  L={L}: {acc:.2%}')
```

### 실험 3 — MAD Metric Implementation

```python
def MAD(H):
    """Mean Average Distance (smaller = more over-smoothed)."""
    H_norm = H / (H.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = H_norm @ H_norm.T
    cos_dist = 1 - cos_sim
    n = H.size(0)
    # Off-diagonal mean
    return (cos_dist.sum() - cos_dist.diag().sum()) / (n * (n - 1))

# 측정
for L in [1, 2, 4, 8, 16]:
    h = torch.randn(n, 16)
    A_pow_l = torch.linalg.matrix_power(A_hat_t, L)
    h_l = A_pow_l @ h
    print(f'L={L}: MAD = {MAD(h_l).item():.4f}')
```

**예상**: $L$ 증가 → MAD 감소 (모든 노드 비슷)

### 실험 4 — Layer 별 Eigenvector Projection

```python
# H^(L) 가 ker(L_sym^aug) 으로 projection 되는지 확인

eig_aug, U_aug = np.linalg.eigh(np.eye(n) - A_hat)   # augmented L_sym
# Smallest eigenvalue 의 eigenvector (largest A_hat eigenvalue)
v_principal = U_aug[:, 0]   # corresponds to lambda_min = 0

# H^(L) 의 v_principal 방향 projection 비율
torch.manual_seed(0)
H = torch.randn(n, 16)

projections = []
for L in range(20):
    H = A_hat_t @ H
    # Each column 의 v_principal alignment
    H_np = H.numpy()
    proj = np.abs(v_principal @ H_np / (np.linalg.norm(H_np, axis=0) + 1e-8))
    projections.append(proj.mean())

plt.plot(projections, 'o-')
plt.xlabel('Layer L'); plt.ylabel('|projection on dominant eigenvector|')
plt.title('H^(L) 가 dominant eigenvector 로 alignment')
plt.show()
```

### 실험 5 — GCN vs MLP 의 Layer 효과 비교

```python
# MLP 는 layer 증가 시 일반화 ↑ (일반)
# GCN 은 layer 증가 시 성능 ↓ (over-smoothing)

class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in if i==0 else d_hid, d_hid) for i in range(num_layers)])
        self.cls = nn.Linear(d_hid, d_out)
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.cls(x)

mlp_results = {}
for L in [1, 2, 4, 8]:
    torch.manual_seed(42)
    model = MLP(d_in=n, d_hid=8, d_out=2, num_layers=L)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    for epoch in range(300):
        model.train(); optimizer.zero_grad()
        out = model(X)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
    acc = (model(X).argmax(-1) == labels).float().mean().item()
    mlp_results[L] = acc

print('GCN vs MLP layer-wise accuracy:')
print(f'{"L":<5} {"GCN":<10} {"MLP":<10}')
for L in [1, 2, 4, 8]:
    print(f'{L:<5} {results.get(L, 0):<10.2%} {mlp_results.get(L, 0):<10.2%}')
```

---

## 🔗 실전 활용

### 1. Layer 수 선택의 정확한 지침

대부분의 GNN 응용에서 **2-3 layer 가 표준**:
- Cora, Citeseer: 2 layer 최적
- OGB-Arxiv: 3 layer
- Reddit: 2 layer (large graph 의 작은 diameter)

### 2. Over-smoothing 진단 도구

학습 중 monitoring:
1. **MAD (Mean Average Distance)**: 매 epoch 측정, 0 에 가까우면 over-smoothed
2. **Layer-wise representation visualization**: t-SNE 또는 UMAP 으로 시각화
3. **Test accuracy vs Layer count**: 단조 감소 시 over-smoothing 의심

### 3. 해결책 (Ch5-03~05)

- **DropEdge**: 학습 중 random edge removal
- **PairNorm**: layer 마다 feature distance 정규화
- **APPNP**: PageRank-based propagation, $\ker(L)$ collapse 회피
- **Jumping Knowledge**: layer-wise concat 으로 hierarchical info 보존
- **Initial residual** (GCNII, Chen 2020): 매 layer 초기 feature 와 residual

### 4. CNN 과의 다른 inductive bias

GNN: 깊이로 receptive field 빠르게 확장 (graph diameter $O(\log n)$ 도달)
CNN: 깊이로 receptive field 천천히 확장 (image 의 큰 size)

따라서 GNN 깊이의 효과는 CNN 과 다른 mechanism 필요.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Connected graph | Disconnected 시 component-wise over-smoothing |
| Static GCN propagation | Dynamic / attention-based 는 다른 dynamics |
| Cosine similarity metric | 다른 metric 가 다른 답 가능 (MAD, Dirichlet energy) |
| Linear layer (no nonlinearity 효과 무시) | ReLU 등이 약간의 information bottleneck — 작은 영향 |
| Pure GCN propagation | Residual / dense connection 가 over-smoothing 완화 |
| Initialization 의존 | Random init 마다 dynamics 약간 다름 |

---

## 📌 핵심 정리

$$\boxed{\text{Over-smoothing: } \lim_{L \to \infty} h_i^{(L)} = h_j^{(L)} \quad \forall i, j}$$

| 현상 | 측정 |
|------|------|
| **Pairwise similarity** | MSim $\to 1$ |
| **Mean Average Distance** | MAD $\to 0$ |
| **Dirichlet energy** | $E(H) \to 0$ |
| **Test accuracy** | 단조 감소 (after $L^*$) |
| **Optimal layer** | 2-3 (GCN), 3-5 (GAT, GraphSAGE) |
| **수학적 원인** | $\hat A^L \to$ projection on dominant eigenvector (Ch5-02) |
| **해결책** | DropEdge, PairNorm, APPNP, JKN (Ch5-03~05) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 작은 graph (3-node path) 에서 GCN 한 layer 후 cosine similarity 가 어떻게 변하는지 손으로 계산하라.

<details>
<summary>해설</summary>

**3-node path**: $1 - 2 - 3$. 
- $A = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$
- $\tilde A = A + I = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}$
- $\tilde d = (2, 3, 2)$
- $\tilde D^{-1/2} = \text{diag}(1/\sqrt 2, 1/\sqrt 3, 1/\sqrt 2)$

$\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$:

$\hat A_{11} = 1/2$, $\hat A_{12} = 1/(\sqrt 2 \cdot \sqrt 3) = 1/\sqrt 6$, $\hat A_{13} = 0$
$\hat A_{21} = 1/\sqrt 6$, $\hat A_{22} = 1/3$, $\hat A_{23} = 1/\sqrt 6$
$\hat A_{31} = 0$, $\hat A_{32} = 1/\sqrt 6$, $\hat A_{33} = 1/2$

**Initial feature** (random orthogonal): $H^{(0)} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ -1 & 0 \end{pmatrix}$ (예시).

$H^{(1)} = \hat A H^{(0)}$:

Row 1: $(1/2 \cdot 1 + 1/\sqrt 6 \cdot 0, 1/2 \cdot 0 + 1/\sqrt 6 \cdot 1) = (1/2, 1/\sqrt 6)$
Row 2: $(1/\sqrt 6 \cdot 1 + 1/3 \cdot 0 + 1/\sqrt 6 \cdot (-1), 1/\sqrt 6 \cdot 0 + 1/3 \cdot 1 + 1/\sqrt 6 \cdot 0) = (0, 1/3)$
Row 3: $(0 + 1/\sqrt 6 \cdot 0 + 1/2 \cdot (-1), 0 + 1/\sqrt 6 \cdot 1 + 0) = (-1/2, 1/\sqrt 6)$

**Cosine sim 변화**:

Initial: orthogonal pairs → cos = 0.
After 1 layer: pair (1, 3) 의 cos = $(1/2 \cdot (-1/2) + 1/\sqrt 6 \cdot 1/\sqrt 6) / (\|h_1\| \|h_3\|) = (-1/4 + 1/6)/(\ldots) = -1/12 / (\ldots)$

직접 계산하면 (1, 3) 의 cos 가 약간 음수 (구조적 거리 멀어서). 단 (1, 2), (2, 3) 의 cos 는 가까워짐 (smoothing).

**Multi-layer**: 더 많은 layer 후 모든 pair cos → 1 (모두 같은 방향).

이는 Section 정리 1.4 의 $\hat A^L \to$ dominant eigenvector projection.

</details>

**문제 2** (심화): Over-smoothing 의 spectral 분석으로 $\hat A^L H \to c \cdot \sqrt{\tilde d}$ ($\sqrt{\tilde d}$ 은 dominant eigenvector) 임을 증명하라.

<details>
<summary>해설</summary>

$\hat A$ 의 spectral 분해: $\hat A = U \Lambda U^T$ where $\Lambda = \text{diag}(1, \mu_2, \mu_3, \ldots)$ (largest eigenvalue 1).

Dominant eigenvector: $\hat A v_1 = v_1$ where $v_1 = \sqrt{\tilde d} / \|\sqrt{\tilde d}\|$.

(Ch2-03 문제 2 에서 증명: $\hat A \sqrt{\tilde d} = \sqrt{\tilde d}$.)

다른 eigenvalue $|\mu_k| < 1$ for $k \geq 2$.

**$L \to \infty$ 극한**:

$\hat A^L H = U \Lambda^L U^T H = \sum_k \mu_k^L v_k v_k^T H$

각 항: $\mu_1^L = 1$ (보존), $\mu_k^L \to 0$ for $k \geq 2$.

따라서:
$$
\lim_{L \to \infty} \hat A^L H = v_1 v_1^T H = v_1 (v_1^T H)
$$

이는 $H$ 의 각 column 이 $v_1$ 방향으로 projection. 즉:
- 모든 노드 의 feature 가 $v_1 = \sqrt{\tilde d}$ 의 scalar multiple
- $h_i \propto \sqrt{\tilde d_i}$ (degree 의 sqrt)
- 모든 노드가 같은 방향 — $\cos(h_i, h_j) = 1$ → over-smoothing $\square$

**수렴 속도**: $\hat A^L H - v_1 v_1^T H = \sum_{k \geq 2} \mu_k^L v_k v_k^T H$.

$L^2$ norm: $\|\sum_{k \geq 2} \mu_k^L v_k (v_k^T H)\| \leq |\mu_2|^L \cdot \|H_\perp\|$.

$|\mu_2|^L \to 0$ exponentially. **Spectral gap $1 - |\mu_2|$** 가 클수록 빠른 over-smoothing.

</details>

**문제 3** (논문 비평): Over-smoothing 이 GCN 의 본질적 한계인가, 아니면 architecture 설계 (initialization, regularization) 로 해결 가능한 문제인가?

<details>
<summary>해설</summary>

**본질적 한계 측면**:

1. **수학적 inevitability**: $\hat A^L \to $ projection on dominant eigenvector — pure linear propagation 의 limit. 어떤 weight matrix $W^{(l)}$ 도 이를 우회 못함 (linear part 만큼은).

2. **모든 message passing GNN 영향**: GCN, GraphSAGE, GAT 모두 비슷한 dynamics — 모두 mixing matrix 형태.

3. **그래프 의 finite diameter**: Random graph 의 diameter $O(\log n)$ → small $L$ 에서 이미 graph 전체 보임. 깊이의 의미 없음.

**Architecture 로 해결 가능한 측면**:

1. **Skip connection (residual)**: GCNII (Chen 2020) 가 initial residual + identity mapping → 64-layer 가능. 깊이 효과 회복.

2. **PairNorm (Ch5-03)**: Feature distance 정규화 — over-smoothing rate 감소.

3. **DropEdge (Ch5-03)**: Random edge removal → effective propagation matrix 변화 → $\ker$ 으로 단조 수렴 X.

4. **APPNP (Ch5-05)**: PPR propagation 의 closed-form — 모든 hop information 보존 + initial feature retention.

5. **Jumping Knowledge (Ch5-05)**: All-layer concat → hierarchical info 보존.

6. **Edge dropping during training**: Stochastic propagation 가 collapse 방지.

**현대 결론**:

- Pure GCN: 본질적 over-smoothing (수학적)
- Modified GNN (residual, normalization, sampling): **practical 해결 가능**
- 깊이 효과 (50+ layer): GCNII, ResGCN 등 가능 but marginal 향상

**Empirical evidence**:

- GCN 깊이 increase → 성능 ↓
- GCNII 64-layer ≈ 2-layer GCN (slight improvement on some tasks)
- 따라서 over-smoothing 해결 가능하지만 실전 효과 제한적

**왜 ResNet-152 같은 깊이가 GNN 에서 무용한가?**:

- CNN: image size 큼, hierarchical composition 자연스러움
- GNN: graph diameter 작음 ($O(\log n)$), 4-5 layer 면 receptive field = full graph

따라서 깊이 자체가 GNN 의 핵심 challenge 가 아님 — 다른 dimension (표현력 ceiling, sampling efficiency, position-aware) 이 더 중요.

**현대 추세**: GNN 깊이 보다 **width** + **attention** + **PE** 의 향상이 주류 — Graphormer (Ch7-01) 가 이런 trend.

</details>

---

<div align="center">

[◀ 이전](../ch4-expressive-power/05-positional-encoding.md) | [📚 README](../README.md) | [다음 ▶](./02-laplacian-proof.md)

</div>
