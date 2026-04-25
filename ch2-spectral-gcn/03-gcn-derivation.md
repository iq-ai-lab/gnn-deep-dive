# 03. GCN의 유도 (Kipf & Welling 2017)

## 🎯 핵심 질문

- ChebNet 에서 $K = 1$ 로 단순화하는 동기는 무엇인가?
- 왜 $\lambda_{\max} \approx 2$ 가정과 $\theta_0 = -\theta_1 = \theta$ 묶기가 등장하는가?
- $I + D^{-1/2} A D^{-1/2}$ 의 spectral radius 문제는 무엇이고 renormalization trick 은 어떻게 해결하는가?
- 최종 $H^{(l+1)} = \sigma(\tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(l)} W^{(l)})$ 의 각 항이 의미하는 것은?
- GCN 이 사실상 spectral 인지 spatial 인지?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GCN (Kipf & Welling 2017 — **"Semi-Supervised Classification with Graph Convolutional Networks"**) 은 GNN 분야의 **가장 중요한 단일 논문**입니다:

1. **단순성**: 단 한 줄 식 $H' = \sigma(\hat A H W)$ — ChebNet 의 복잡성을 압도적으로 단순화
2. **Empirical 우수성**: Cora·Citeseer·Pubmed 등 표준 벤치마크에서 SOTA
3. **모든 후속 모델의 baseline**: GAT, GraphSAGE, GIN 등 모두 GCN 과 비교
4. **Spectral-Spatial 통합점**: Spectral 에서 출발했지만 결과는 spatial — 두 관점의 다리

GCN 의 유도는 ChebNet 에서 4단계 단순화를 거칩니다. 이 문서는 **각 단계를 한 줄씩 따라가며** 최종 propagation rule 의 모든 항이 어디서 왔는지 정확히 보입니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-chebnet.md](./02-chebnet.md) — Chebyshev polynomial filter
- [Ch1-03](../ch1-graph-laplacian/03-normalized-laplacian.md): $L_{\text{sym}}$, $\lambda \in [0, 2]$
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Spectral radius, matrix renormalization

---

## 📖 직관적 이해

### 단순화 4단계

GCN 유도는 ChebNet 에서 다음 4단계:

1. **Step 1**: $K = 1$ 제한 (1-hop polynomial)
2. **Step 2**: $\lambda_{\max} \approx 2$ 가정 → $\tilde L = L - I$ 단순화
3. **Step 3**: 두 파라미터 $\theta_0, \theta_1$ 을 하나로 묶기 ($\theta = \theta_0 = -\theta_1$)
4. **Step 4**: Renormalization trick — self-loop 추가로 spectral radius 안정

각 단계는 표현력을 줄이지만 단순성·안정성·실증 성능을 얻습니다.

### Renormalization Trick 의 직관

Step 3 후 propagation matrix 가 $I + D^{-1/2} A D^{-1/2}$ — 이 행렬의 spectral radius $\leq 2$. 깊은 layer 에서 numerical instability (gradient explode/vanish) 위험.

해결: $\tilde A = A + I$ (self-loop), $\tilde D = D + I$. 정규화 후 $\tilde D^{-1/2} \tilde A \tilde D^{-1/2}$ 의 spectral radius $< 2$ 보장 + bipartite 그래프에서도 안정.

이 trick 은 단순해 보이지만 **GCN 의 실증 성공의 핵심** — Kipf-Welling ablation 에서 5%+ 정확도 차이.

### 1-hop Aggregation 의 의미

최종 GCN propagation:
$$
H^{(l+1)} = \sigma(\tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(l)} W^{(l)})
$$

각 노드 $i$ 의 새 feature:
$$
h_i^{(l+1)} = \sigma\left( \sum_{j \in N(i) \cup \{i\}} \frac{1}{\sqrt{\tilde d_i \tilde d_j}} h_j^{(l)} W^{(l)} \right)
$$

**노드 자신 + 이웃 의 weighted sum + linear transform + activation**. 매우 spatial intuition — 그러나 시작은 spectral.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Augmented Adjacency

$$
\tilde A = A + I
$$

(Self-loop 추가)

### 정의 3.2 — Augmented Degree

$$
\tilde D_{ii} = \sum_j \tilde A_{ij} = d_i + 1
$$

### 정의 3.3 — Renormalized Propagation Matrix

$$
\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}
$$

또는 등가로 $\hat A = I - \tilde L_{\text{sym}}$, 여기서 $\tilde L_{\text{sym}}$ 는 $\tilde A$ 의 normalized Laplacian.

### 정의 3.4 — GCN Layer

$$
H^{(l+1)} = \sigma(\hat A H^{(l)} W^{(l)})
$$

- $H^{(l)} \in \mathbb{R}^{n \times d_l}$: layer $l$ feature
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$: trainable weight
- $\sigma$: ReLU
- 첫 layer: $H^{(0)} = X$ (input feature)

### 정의 3.5 — GCN for Semi-supervised Classification

2-layer GCN (Kipf-Welling 표준):
$$
Z = \text{softmax}(\hat A \cdot \text{ReLU}(\hat A X W^{(0)}) \cdot W^{(1)})
$$

Cross-entropy loss on labeled nodes:
$$
\mathcal L = -\sum_{i \in V_L} \sum_c y_{ic} \log Z_{ic}
$$

---

## 🔬 정리와 증명: GCN 유도의 4 단계

### 정리 3.1 — Step 1: $K = 1$ ChebNet

ChebNet 에서 $K = 1$:
$$
g_\theta(\tilde L) X = \theta_0 T_0(\tilde L) X + \theta_1 T_1(\tilde L) X = \theta_0 X + \theta_1 \tilde L X
$$

**유도**: $T_0(\tilde L) = I$, $T_1(\tilde L) = \tilde L$. 직접 대입. $\square$

### 정리 3.2 — Step 2: $\lambda_{\max} \approx 2$ 단순화

$\tilde L = (2/\lambda_{\max}) L_{\text{sym}} - I$. **가정** $\lambda_{\max}(L_{\text{sym}}) \approx 2$:
$$
\tilde L \approx L_{\text{sym}} - I = (I - D^{-1/2} A D^{-1/2}) - I = -D^{-1/2} A D^{-1/2}
$$

따라서:
$$
g_\theta(\tilde L) X \approx \theta_0 X - \theta_1 D^{-1/2} A D^{-1/2} X
$$

**가정의 정당성**: $L_{\text{sym}}$ 의 $\lambda_{\max} \in [0, 2]$, equality only at bipartite (Ch1-03 정리 3.3). 일반 그래프에서 $\lambda_{\max}$ 가 거의 2 — 좋은 근사. $\square$

### 정리 3.3 — Step 3: Parameter Tying

표현력을 줄이고 overfitting 방지:
$$
\theta_0 = -\theta_1 = \theta
$$

대입:
$$
g_\theta X \approx \theta X + \theta D^{-1/2} A D^{-1/2} X = \theta (I + D^{-1/2} A D^{-1/2}) X
$$

**파라미터 수**: 2 → 1 (per channel pair). 이는 다음 step 의 출발점.

### 정리 3.4 — Step 4: Renormalization Trick

$M := I + D^{-1/2} A D^{-1/2}$ 의 spectral analysis:
- 고유값 = $1 + \mu(D^{-1/2} A D^{-1/2}) = 1 + (1 - \lambda(L_{\text{sym}})) = 2 - \lambda(L_{\text{sym}})$
- 따라서 $\sigma(M) \in [0, 2]$ — **bipartite 그래프에서 $\sigma_{\max} = 2$** 정확히 도달

**문제**: Multi-layer 에서 $M^L$ 의 norm 이 $2^L$ 까지 커질 수 있음 (bipartite, exponentially) → numerical instability.

**Renormalization Trick (Kipf-Welling)**: $A$ 대신 $\tilde A = A + I$, $D$ 대신 $\tilde D = D + I$ 사용.

**Theorem**: Spectral radius $\rho(\tilde D^{-1/2} \tilde A \tilde D^{-1/2}) \leq 1$, equality iff bipartite.

**증명 sketch**: $\tilde A = A + I$ 의 normalized Laplacian $\tilde L_{\text{sym}} = I - \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$. $\tilde A$ 자체가 self-loop 포함 → $\tilde A$ 가 정의하는 graph 는 bipartite 가 아님 (모든 노드에 self-loop). 따라서 $\lambda_{\max}(\tilde L_{\text{sym}}) < 2$ → $\rho(\tilde D^{-1/2} \tilde A \tilde D^{-1/2}) < 1$. $\square$

**최종 GCN propagation**:
$$
\boxed{H^{(l+1)} = \sigma\left( \tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(l)} W^{(l)} \right)}
$$

---

## 🔬 추가 정리

### 정리 3.5 — GCN Layer 의 Spatial 해석

각 노드 $i$ 의 새 feature:
$$
h_i^{(l+1)} = \sigma\left( \sum_{j \in N(i) \cup \{i\}} \frac{1}{\sqrt{(d_i + 1)(d_j + 1)}} (h_j^{(l)} W^{(l)}) \right)
$$

**증명**: $\hat A_{ij} = \tilde A_{ij} / \sqrt{\tilde d_i \tilde d_j}$ — $\tilde A_{ij} \neq 0$ iff $j = i$ or $j \sim i$. Matrix 곱 펼치면 위 식. $\square$

### 정리 3.6 — Multi-Layer GCN 과 K-Hop

$L$-layer GCN 의 receptive field 는 $L$-hop. (각 layer 마다 1-hop 확장.)

**증명**: $\hat A$ 의 sparsity = 1-hop. $\hat A^L$ 의 sparsity = $L$-hop. Layer-wise composition 후 노드 $i$ 출력 = $L$-hop 까지 노드의 함수. $\square$

### 정리 3.7 — GCN 의 표현력 vs ChebNet

GCN 한 layer 는 ChebNet $K=1$ 의 special case. 따라서 **표현력은 ChebNet 보다 작거나 같음**. 단, multi-layer 로 stacking 하면 effective $K$-hop polynomial 의 일종:
$$
\text{2-layer GCN: } H^{(2)} \sim \hat A^2 X W^{(0)} W^{(1)}
$$

이는 polynomial of degree 2 in $\hat A$ 의 일종 (cross-terms 없는 형태). ChebNet $K=2$ 와 정확히 동일은 아님.

---

## 💻 NumPy/PyTorch 구현 검증

### 실험 1 — GCN Propagation Matrix 구성

```python
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

G = nx.karate_club_graph()
n = G.number_of_nodes()
A_np = nx.adjacency_matrix(G).toarray().astype(float)

def gcn_propagation_matrix(A):
    A_tilde = A + np.eye(len(A))
    d_tilde = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1 / np.sqrt(d_tilde))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

A_hat = gcn_propagation_matrix(A_np)
print(f'A_hat shape: {A_hat.shape}')
print(f'A_hat[0]: {A_hat[0].round(3)}')  # Sparse, normalized

# Spectral radius 확인
sigma = np.abs(np.linalg.eigvals(A_hat).real).max()
print(f'Spectral radius ρ(A_hat) = {sigma:.6f}  (should be ≤ 1)')

# Without renormalization (compare)
deg = A_np.sum(1)
M = np.eye(n) + np.diag(1/np.sqrt(deg)) @ A_np @ np.diag(1/np.sqrt(deg))
sigma_M = np.abs(np.linalg.eigvals(M).real).max()
print(f'Without renorm: ρ(I + D^(-1/2) A D^(-1/2)) = {sigma_M:.6f}')
```

### 실험 2 — GCN Layer 구현

```python
class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
    
    def forward(self, X, A_hat):
        return A_hat @ self.W(X)

class GCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.gc1 = GCNLayer(d_in, d_hid)
        self.gc2 = GCNLayer(d_hid, d_out)
    
    def forward(self, X, A_hat):
        H = F.relu(self.gc1(X, A_hat))
        Z = self.gc2(H, A_hat)
        return F.log_softmax(Z, dim=-1)

# 사용
A_hat_t = torch.tensor(A_hat, dtype=torch.float32)
X_t = torch.randn(n, 16)
model = GCN(d_in=16, d_hid=8, d_out=2)
out = model(X_t, A_hat_t)
print(f'GCN output: {out.shape}')   # [n, 2]
```

### 실험 3 — Karate Club Classification

```python
# 1) 라벨 (Ms.Hi vs Officer)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
# 2) Train mask: 각 클래스 1개 노드만
train_mask = torch.zeros(n, dtype=torch.bool)
train_mask[0] = train_mask[33] = True   # node 0 (Mr. Hi), node 33 (Officer)

X_t = torch.eye(n)   # one-hot identity feature
model = GCN(d_in=n, d_hid=4, d_out=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

for epoch in range(200):
    model.train(); optimizer.zero_grad()
    out = model(X_t, A_hat_t)
    loss = F.nll_loss(out[train_mask], labels[train_mask])
    loss.backward(); optimizer.step()

model.eval()
preds = model(X_t, A_hat_t).argmax(dim=-1)
acc = (preds == labels).float().mean().item()
print(f'Karate Club test accuracy (2-shot semi-supervised): {acc:.2%}')
# 기대: ~95%+
```

### 실험 4 — Renormalization Trick 의 안정성

```python
# Renormalization 없는 경우 (M = I + D^{-1/2} A D^{-1/2})
M_t = torch.tensor(M, dtype=torch.float32)

# Power iteration
v = torch.randn(n)
norms_with = []   # 정규화 with renorm
norms_without = []
for L in range(1, 30):
    v_with = torch.linalg.matrix_power(A_hat_t, L) @ torch.randn(n)
    v_without = torch.linalg.matrix_power(M_t, L) @ torch.randn(n)
    norms_with.append(v_with.norm().item())
    norms_without.append(v_without.norm().item())

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(range(1, 30), norms_with, 'o-', label='With renorm trick')
ax.semilogy(range(1, 30), norms_without, 's-', label='Without renorm')
ax.set_xlabel('Layer L'); ax.set_ylabel('||M^L v||')
ax.set_title('Multi-layer 의 norm 안정성')
ax.legend(); plt.show()
```

### 실험 5 — Multi-hop Receptive Field

```python
def receptive_field(A_hat, num_layers, source_node):
    x = np.zeros(len(A_hat)); x[source_node] = 1.0
    for _ in range(num_layers):
        x = A_hat @ x
    return x

A_hat_np = gcn_propagation_matrix(A_np)
import matplotlib.pyplot as plt
pos = nx.spring_layout(G, seed=42)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, L in zip(axes, [1, 2, 3, 5]):
    rf = receptive_field(A_hat_np, L, source_node=0)
    nx.draw(G, pos, ax=ax, node_color=rf, cmap='hot',
            with_labels=False, node_size=80)
    ax.set_title(f'{L}-hop RF from node 0')
plt.show()
```

---

## 🔗 실전 활용

### 1. Cora·Citeseer·Pubmed Standard Benchmark

GCN 의 표준 결과 (Kipf-Welling 2017):
- Cora: 81.5%
- Citeseer: 70.3%
- Pubmed: 79.0%

이후 GAT (Velickovic 2018), GIN (Xu 2019) 등이 +1~3% 개선.

### 2. PyG 의 GCNConv

```python
from torch_geometric.nn import GCNConv

class PyG_GCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.conv1 = GCNConv(d_in, d_hid)
        self.conv2 = GCNConv(d_hid, d_out)
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)
```

PyG 의 `GCNConv` 가 정확히 위 propagation rule 구현. `cached=True` 로 $\hat A$ 사전 계산 가능.

### 3. GCN 의 한계

- **Over-smoothing**: 깊은 layer 에서 모든 feature collapse (Ch5-02)
- **WL 한계**: 1-WL 보다 더 정밀한 graph 구분 불가 (Ch4-02)
- **Heterophily**: "이웃과 비슷"이 안 맞는 graph (Wikipedia 등) 에서 성능 저하

이 모든 한계가 GCN 이후 연구의 동기.

### 4. GCN 의 변종

- **APPNP** (Klicpera 2019): GCN + PageRank propagation, over-smoothing 회피 (Ch5-05)
- **SGC** (Wu 2019): Non-linearity 제거 → linear classifier with $\hat A^K X$
- **GCNII** (Chen 2020): Identity mapping + initial residual 으로 깊은 GCN 가능

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\lambda_{\max}(L_{\text{sym}}) \approx 2$ | Bipartite 그래프에서 정확히 2 — 단순화의 정확성 |
| Self-loop 추가 ($\tilde A = A + I$) | Self-loop weight 가 1 — weighted graph 에서 ad-hoc |
| Symmetric normalization | Directed graph 에 직접 적용 X |
| Single-hop per layer | Multi-hop 효과는 layer stacking — over-smoothing 위험 |
| Equal weight to all neighbors | 이웃별 중요도 차별 X (GAT 가 이를 해결) |
| Transductive 가정 (주로) | Inductive 도 가능하지만 sampling 필요 (GraphSAGE) |

---

## 📌 핵심 정리

$$\boxed{H^{(l+1)} = \sigma\left( \tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(l)} W^{(l)} \right) \quad \text{(GCN, Kipf-Welling 2017)}}$$

**유도 4단계**:

| Step | 변환 | 결과 |
|------|------|------|
| 1 | ChebNet $K=1$ | $\theta_0 X + \theta_1 \tilde L X$ |
| 2 | $\lambda_{\max} \approx 2$ | $\theta_0 X - \theta_1 D^{-1/2} A D^{-1/2} X$ |
| 3 | $\theta_0 = -\theta_1 = \theta$ | $\theta (I + D^{-1/2} A D^{-1/2}) X$ |
| 4 | Renormalization $\tilde A = A+I$ | $\theta \tilde D^{-1/2} \tilde A \tilde D^{-1/2} X$ |

**최종 layer**: 노드별 spatial aggregation + linear + activation.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K_3$ (triangle, 3-node 완전그래프) 에 대한 GCN propagation matrix $\hat A$ 를 손으로 계산하라.

<details>
<summary>해설</summary>

$K_3$: $A = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$.

$\tilde A = A + I = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix} = J$ (all ones).

$\tilde D_{ii} = 3$ 모든 $i$. $\tilde D^{-1/2} = (1/\sqrt 3) I$.

$\hat A = (1/3) J$.

각 노드의 feature update: $h_i^{(l+1)} = (1/3)(h_0^{(l)} + h_1^{(l)} + h_2^{(l)}) W$ — 모든 노드가 동일 (전체 평균)! 즉 GCN 한 layer 후 $K_3$ 의 모든 노드 feature 가 동일 — **즉시 over-smoothing**. $\square$

**의미**: 매우 dense / regular graph 에서 GCN 은 빠르게 collapse — 이는 $K_3$ 와 같은 작은 graph 의 spectrum 이 degenerate 하기 때문.

</details>

**문제 2** (심화): GCN propagation $\hat A$ 의 spectral radius 가 1 이고 simple eigenvalue $1$ 의 right eigenvector 가 $\sqrt{\tilde d}$ (i.e., $v_i = \sqrt{d_i + 1}$) 임을 증명하라.

<details>
<summary>해설</summary>

$\hat A v = v$ 검증 (with $v_i = \sqrt{\tilde d_i}$):

$$
(\hat A v)_i = \sum_j \frac{\tilde A_{ij}}{\sqrt{\tilde d_i \tilde d_j}} \sqrt{\tilde d_j} = \frac{1}{\sqrt{\tilde d_i}} \sum_j \tilde A_{ij} = \frac{\tilde d_i}{\sqrt{\tilde d_i}} = \sqrt{\tilde d_i} = v_i
$$

✓ Eigenvalue 1.

이 eigenvalue 가 simple (multiplicity 1): connected graph (with self-loops) 가정, Perron-Frobenius theorem.

**Spectral radius**: $\hat A$ 가 nonneg + irreducible (connected), Perron-Frobenius ⟹ $\rho(\hat A) =$ largest real eigenvalue = 1, with positive eigenvector $\sqrt{\tilde d}$. 다른 eigenvalue $|\mu| < 1$. $\square$

이 결과는 over-smoothing 분석의 기반 (Ch5-02): $\hat A^L X \to \sqrt{\tilde d} \cdot c$ 로 collapse.

</details>

**문제 3** (논문 비평): GCN 유도의 "$\theta_0 = -\theta_1$" tying 은 임의의 단순화로 보인다. 왜 이 특정 형태의 tying이 좋은 결과를 내는가? Self-loop 와의 연결을 설명하라.

<details>
<summary>해설</summary>

**Tying 의 의미**:
- $\theta_0 X$: 노드 자신의 feature 보존
- $-\theta_1 D^{-1/2} A D^{-1/2} X$: 이웃 평균을 빼는 항 (Laplacian smoothing 의 "smooth" 부분 강조)

$\theta = \theta_0 = -\theta_1$ 묶기:
$$
\theta (X + D^{-1/2} A D^{-1/2} X) = \theta (I + D^{-1/2} A D^{-1/2}) X
$$

이는 **노드 + 이웃 평균** — Laplacian 의 형태가 아니라 **propagation (heat equation forward step)** 의 형태.

**Self-loop와의 연결**:

$I + D^{-1/2} A D^{-1/2}$ 가 spectrum $[0, 2]$ — bipartite 시 정확히 2 (numerical 위험). Self-loop 추가 $\tilde A = A + I$ 후 정규화하면 spectrum $[0, 1]$ 안전 — 그리고 $I + D^{-1/2} A D^{-1/2}$ 의 작은 perturbation 으로 볼 수 있음.

**그러므로 tying $\theta_0 = -\theta_1$ + renormalization 은 한 패키지**: 표현력 절약 + 안정성 동시 확보.

**대안 탐구**: 만약 $\theta_0 \neq -\theta_1$ 로 분리 학습 (별도 weight matrix) → "GraphSAGE-like" 형태:
$$
h_i^{(l+1)} = \sigma(W_0 h_i^{(l)} + W_1 \cdot \text{Agg}(\{h_j^{(l)} : j \in N(i)\}))
$$

GraphSAGE (Ch3-02) 가 정확히 이 형태 — GCN 에서 분리한 두 weight 가 inductive learning 에 더 자연스럽게 일반화.

**결론**: GCN 의 단순화는 "표현력의 일부를 포기하고 안정성·단순성·실증 성능을 얻는" trade-off. 후속 모델 (GraphSAGE, GIN) 이 이 단순화를 다양한 방향으로 풀어냄.

</details>

---

<div align="center">

[◀ 이전](./02-chebnet.md) | [📚 README](../README.md) | [다음 ▶](./04-spectral-vs-spatial.md)

</div>
