# 05. APPNP와 Jumping Knowledge Network

## 🎯 핵심 질문

- APPNP (Klicpera 2019) 의 closed-form $Z = \alpha (I - (1 - \alpha) \tilde P)^{-1} H^{(0)}$ 가 어떻게 모든 hop 의 information 을 보존하는가?
- $\alpha$ teleport probability 의 의미와 over-smoothing 회피 메커니즘?
- Jumping Knowledge Network (Xu 2018) 의 layer-wise concat / max-pool / LSTM 결합?
- APPNP 의 propagation 이 personalized PageRank 와 정확히 어떻게 일치하는가?
- $L \to \infty$ 에서도 APPNP 가 collapse 안 되는 이유?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch5-03, Ch5-04 의 DropEdge / PairNorm / Sampling 이 cosmetic mitigation. **APPNP** 와 **JKN** 은 본질적 architecture 변경:

1. **APPNP**: Personalized PageRank 의 closed-form 으로 $\ker$ collapse 회피. Layer 무한대도 가능.
2. **JKN**: 모든 layer 의 representation 을 concat — 어느 layer 에서 collapse 되어도 earlier 가 보존.

이들은 GCN 의 single dominant eigenvector projection 자체를 우회 — over-smoothing 의 본질적 해결.

이 문서가 Chapter 5 의 마무리, modern GNN architecture 의 직접적 전신.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-phenomenon.md](./01-phenomenon.md), [02-laplacian-proof.md](./02-laplacian-proof.md)
- [Ch1-06](../ch1-graph-laplacian/06-random-walk-pagerank.md) — PageRank, personalized random walk
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix inverse, geometric series

---

## 📖 직관적 이해

### APPNP 의 핵심 idea

**Propagation 과 prediction 을 분리**:
- 첫째, MLP 로 노드별 prediction $H^{(0)} = \text{MLP}(X)$
- 둘째, PageRank-style propagation $Z = (1 - \alpha)(I - \alpha \tilde P)^{-1} H^{(0)}$

**$\alpha$**: teleport probability — 매 step $\alpha$ 확률로 initial $H^{(0)}$ 으로 reset.

### Geometric Series 형태

$$
Z = (1 - \alpha) \sum_{k=0}^\infty \alpha^k \tilde P^k H^{(0)}
$$

(where $\tilde P = $ GCN propagation matrix)

각 hop $k$ 의 information $\tilde P^k H^{(0)}$ 가 weight $(1 - \alpha) \alpha^k$ 로 합산. **모든 hop 의 weighted average** — single dominant eigenvector 으로 collapse 안 됨.

### $\alpha$ 의 효과

- $\alpha \to 0$: 1-hop propagation 만 dominant — GCN-like
- $\alpha \to 1$: $H^{(0)}$ 만 — MLP-like (graph 무시)
- $\alpha \in (0.1, 0.3)$: sweet spot — graph + initial 의 balance

### JKN 의 idea

GCN 의 모든 layer output $H^{(1)}, H^{(2)}, \ldots, H^{(L)}$ 를 final representation 에 결합:
- **Concat**: $H_{\text{final}} = [H^{(1)} \| H^{(2)} \| \cdots \| H^{(L)}]$
- **Max-pool**: $H_{\text{final}} = \max_l H^{(l)}$
- **LSTM**: $H_{\text{final}} = \text{LSTM}(H^{(1)}, \ldots, H^{(L)})$

각 layer 가 different hop receptive field — 깊은 layer collapse 해도 earlier layers 가 정보 유지.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — APPNP (Klicpera 2019)

**MLP step**: $H^{(0)} = f_\theta(X)$ — input feature 를 MLP 로 transform.

**Power iteration**:
$$
Z^{(0)} = H^{(0)}, \quad Z^{(k+1)} = (1 - \alpha) \tilde P Z^{(k)} + \alpha H^{(0)}
$$

$K$ step 후 $Z^{(K)}$ 가 final representation. $K \to \infty$ 시 closed-form:
$$
Z^* = \alpha (I - (1 - \alpha) \tilde P)^{-1} H^{(0)}
$$

(power iteration 의 fixed point)

**$\tilde P = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$**: GCN propagation matrix.

### 정의 5.2 — Closed-Form vs Iterative

**Closed-form**: $Z^* = \alpha (I - (1 - \alpha) \tilde P)^{-1} H^{(0)}$
- 한 번 계산
- Matrix inverse $O(n^3)$ — sparse 시 conjugate gradient $O(m \log n)$

**Iterative ($K$-step)**: $K$ 번 power iteration
- Each step $O(m d)$
- $K \approx \log_2(\epsilon^{-1}) / (1 - \alpha)$ for $\epsilon$ accuracy

Klicpera 2019: $K = 10$ 정도면 sufficient.

### 정의 5.3 — Jumping Knowledge Network (Xu 2018)

$L$-layer GCN: $H^{(0)}, H^{(1)}, \ldots, H^{(L)}$.

**JK-concat**: $h_v^{\text{JK}} = [h_v^{(1)} \| h_v^{(2)} \| \cdots \| h_v^{(L)}]$
**JK-max**: $h_v^{\text{JK}} = \max_l h_v^{(l)}$ (element-wise)
**JK-LSTM**: $h_v^{\text{JK}} = \text{LSTM}(h_v^{(1)}, \ldots, h_v^{(L)})$ — bidirectional optional

Final classifier: $\text{cls}(h_v^{\text{JK}})$.

### 정의 5.4 — APPNP-LSTM 또는 APPNP-Concat (Hybrid)

APPNP + JKN: 매 power iteration step $Z^{(k)}$ 의 concat 또는 LSTM-aggregate.

---

## 🔬 정리와 결과

### 정리 5.1 — APPNP 의 Geometric Series 형태

**Theorem**: APPNP closed-form 의 Neumann series 전개:
$$
Z^* = \alpha (I - (1-\alpha) \tilde P)^{-1} H^{(0)} = \alpha \sum_{k=0}^\infty (1-\alpha)^k \tilde P^k H^{(0)}
$$

**증명**: $\tilde P$ 의 spectral radius $\leq 1$, $1 - \alpha < 1$ ⟹ $(1 - \alpha) \tilde P$ 의 spectral radius $< 1$ ⟹ Neumann series convergent:
$$
(I - (1-\alpha) \tilde P)^{-1} = \sum_{k=0}^\infty ((1-\alpha) \tilde P)^k = \sum_{k=0}^\infty (1-\alpha)^k \tilde P^k
$$

곱하기 $\alpha$. $\square$

### 정리 5.2 — APPNP 의 Spectral 분석

**Theorem**: $\tilde P = U \Lambda U^T$, $\Lambda = \text{diag}(\mu_1, \ldots, \mu_n)$. APPNP 의 effective spectral filter:
$$
\hat g_{\text{APPNP}}(\mu_k) = \frac{\alpha}{1 - (1-\alpha) \mu_k}
$$

**증명**:
$$
Z^* = \alpha (I - (1-\alpha) U \Lambda U^T)^{-1} H^{(0)} = U \cdot \alpha \cdot \text{diag}\left(\frac{1}{1 - (1-\alpha)\mu_k}\right) \cdot U^T H^{(0)}
$$

$\square$

**관찰**: 모든 spectral component $v_k$ 가 보존 — collapse 없음.

- Dominant ($\mu_1 = 1$): $\hat g = \alpha / (1 - (1-\alpha)) = \alpha / \alpha = 1$ (full weight)
- Other ($\mu_k < 1$): $\hat g < 1$ but **non-zero**

### 정리 5.3 — APPNP 와 PageRank 의 정확한 일치

**Theorem**: APPNP 의 propagation 이 personalized PageRank 의 vector form:
$$
\pi_v = \alpha (I - (1-\alpha) \tilde P)^{-1} e_v
$$

(node $v$ 에서 시작하는 personalized random walk 의 stationary distribution)

APPNP 는 **각 노드에 대한 personalized PageRank 를 모든 feature dimension 에 동시 적용**.

### 정리 5.4 — Over-smoothing 회피 증명

**Theorem**: APPNP 의 dominant eigenvector projection 은 collapse 가 아닌 weighted feature.

**증명**: $L \to \infty$ 시 vanilla GCN: $\tilde P^L \to v_1 v_1^T$ — rank-1.

APPNP: $Z^* = \alpha (I - (1-\alpha) \tilde P)^{-1} H^{(0)}$ — full-rank operator (모든 $(1 - (1-\alpha)\mu_k)^{-1}$ nonzero, including $\mu_k$ for $k \geq 2$).

Unlike $\tilde P^L$, **모든 spectral component 가 살아남음** → no rank collapse → no over-smoothing. $\square$

### 정리 5.5 — JKN 의 Multi-hop Information Preservation

**Theorem (Xu 2018, informal)**: JKN-concat 의 final representation $h_v^{\text{JK}}$ 가 1-hop 부터 $L$-hop 까지 모든 information 보유.

각 layer $l$ 의 receptive field 가 $l$-hop. Final 에서 모두 concat → multi-hop hierarchical info.

**vs vanilla GCN**: Final layer 의 $L$-hop info 만, earlier 손실.

### 정리 5.6 — APPNP vs JKN 비교

| 항목 | APPNP | JKN |
|------|-------|-----|
| **Architecture** | Closed-form / power iteration | Layer-wise concat |
| **Parameter** | $\theta$ for MLP (independent of $L$) | $\{W^{(l)}\}_{l=1}^L$ |
| **Implicit depth** | $\sim \log_{1/(1-\alpha)} (\epsilon^{-1})$ | $L$ |
| **Closed-form**: ✓ | ✗ (depth-parameterized) |
| **Spectral analysis**: 가능 | 어려움 |
| **Empirical performance**: 비슷 | 비슷 (task-dependent) |

---

## 💻 구현

### 실험 1 — APPNP 구현

```python
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

def gcn_propagation_matrix(A):
    n = len(A)
    A_t = A + np.eye(n)
    d_t = A_t.sum(1)
    return np.diag(1/np.sqrt(d_t)) @ A_t @ np.diag(1/np.sqrt(d_t))

class APPNP(nn.Module):
    """Approximate Personalized Propagation of Neural Predictions."""
    def __init__(self, d_in, d_hid, d_out, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.K = K
        self.alpha = alpha
        # MLP for initial prediction
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_out)
        )
    
    def forward(self, x, P):
        H_0 = self.mlp(x)   # [n, d_out]
        # Power iteration
        Z = H_0
        for _ in range(self.K):
            Z = (1 - self.alpha) * P @ Z + self.alpha * H_0
        return Z

# Karate Club 학습
G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)
P = torch.tensor(gcn_propagation_matrix(A), dtype=torch.float32)

X = torch.eye(n)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True

model = APPNP(d_in=n, d_hid=8, d_out=2, K=10, alpha=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

for epoch in range(300):
    model.train(); optimizer.zero_grad()
    out = model(X, P)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward(); optimizer.step()

acc = (model(X, P).argmax(-1) == labels).float().mean().item()
print(f'APPNP accuracy: {acc:.2%}')
```

### 실험 2 — Closed-form vs Iterative APPNP

```python
import time

def appnp_closed_form(P, H_0, alpha):
    n = P.size(0)
    return alpha * torch.linalg.solve(torch.eye(n) - (1 - alpha) * P, H_0)

def appnp_iterative(P, H_0, alpha, K):
    Z = H_0.clone()
    for _ in range(K):
        Z = (1 - alpha) * P @ Z + alpha * H_0
    return Z

H_0 = torch.randn(n, 8)

# Closed-form
t0 = time.time()
Z_closed = appnp_closed_form(P, H_0, alpha=0.1)
t_closed = time.time() - t0

# Iterative K=10
t0 = time.time()
Z_iter_10 = appnp_iterative(P, H_0, alpha=0.1, K=10)
t_iter_10 = time.time() - t0

# Iterative K=100
t0 = time.time()
Z_iter_100 = appnp_iterative(P, H_0, alpha=0.1, K=100)
t_iter_100 = time.time() - t0

print(f'Closed-form vs Iterative K=10:  diff = {(Z_closed - Z_iter_10).norm().item():.4e}')
print(f'Closed-form vs Iterative K=100: diff = {(Z_closed - Z_iter_100).norm().item():.4e}')
print(f'Closed-form time: {t_closed*1000:.2f}ms, K=10: {t_iter_10*1000:.2f}ms, K=100: {t_iter_100*1000:.2f}ms')
```

### 실험 3 — APPNP의 Layer 무한대 (No Over-smoothing)

```python
# K 증가 시에도 representation collapse 안 됨 vs vanilla GCN
def vanilla_gcn_iterations(P, H_0, K, num_layers):
    H = H_0.clone()
    for _ in range(K):
        H = P @ H
    return H

import matplotlib.pyplot as plt

def msim(H):
    h_norm = H / (H.norm(dim=-1, keepdim=True) + 1e-8)
    sim = h_norm @ h_norm.T
    n = H.size(0)
    return ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()

torch.manual_seed(0)
H_init = torch.randn(n, 16)

K_values = [1, 2, 5, 10, 20, 50, 100]
gcn_msims, appnp_msims = [], []

for K in K_values:
    H_gcn = vanilla_gcn_iterations(P, H_init, K, num_layers=K)
    H_appnp = appnp_iterative(P, H_init, alpha=0.1, K=K)
    gcn_msims.append(msim(H_gcn))
    appnp_msims.append(msim(H_appnp))

plt.plot(K_values, gcn_msims, 'o-', label='Vanilla GCN ($\\hat A^K$)')
plt.plot(K_values, appnp_msims, 's-', label='APPNP (α=0.1)')
plt.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='collapse')
plt.xlabel('K'); plt.ylabel('Avg pairwise cos sim')
plt.title('APPNP: No over-smoothing as K → ∞')
plt.legend(); plt.grid()
plt.show()
```

### 실험 4 — JKN 구현

```python
class JKNet(nn.Module):
    """Jumping Knowledge Network with concat / max / lstm."""
    def __init__(self, d_in, d_hid, d_out, num_layers=4, mode='concat'):
        super().__init__()
        self.mode = mode
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in, d_hid))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(d_hid, d_hid))
        
        if mode == 'concat':
            self.cls = nn.Linear(num_layers * d_hid, d_out)
        elif mode == 'max':
            self.cls = nn.Linear(d_hid, d_out)
        elif mode == 'lstm':
            self.lstm = nn.LSTM(d_hid, d_hid, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(2 * d_hid, 1)
            self.cls = nn.Linear(d_hid, d_out)
    
    def forward(self, x, P):
        h = x
        layer_outputs = []
        for layer in self.layers:
            h = F.relu(P @ layer(h))
            layer_outputs.append(h)
        
        # JK aggregation
        if self.mode == 'concat':
            h_jk = torch.cat(layer_outputs, dim=-1)
        elif self.mode == 'max':
            h_jk = torch.stack(layer_outputs).max(0).values
        elif self.mode == 'lstm':
            stacked = torch.stack(layer_outputs, dim=1)   # [n, L, d]
            lstm_out, _ = self.lstm(stacked)   # [n, L, 2d]
            attn_weights = F.softmax(self.attn(lstm_out), dim=1)
            h_jk = (lstm_out[:, :, :h.size(-1)] * attn_weights).sum(1)
        
        return self.cls(h_jk)

# 학습 비교
for mode in ['concat', 'max', 'lstm']:
    torch.manual_seed(42)
    model = JKNet(d_in=n, d_hid=8, d_out=2, num_layers=4, mode=mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    for epoch in range(300):
        model.train(); optimizer.zero_grad()
        loss = F.cross_entropy(model(X, P)[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
    acc = (model(X, P).argmax(-1) == labels).float().mean().item()
    print(f'JKNet-{mode}: acc = {acc:.2%}')
```

### 실험 5 — APPNP $\alpha$ Sensitivity

```python
results = {}
for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
    torch.manual_seed(42)
    model = APPNP(d_in=n, d_hid=8, d_out=2, K=10, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    for epoch in range(300):
        model.train(); optimizer.zero_grad()
        loss = F.cross_entropy(model(X, P)[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
    acc = (model(X, P).argmax(-1) == labels).float().mean().item()
    results[alpha] = acc

print('APPNP α sensitivity:')
for alpha, acc in results.items():
    print(f'  α={alpha}: {acc:.2%}')
```

---

## 🔗 실전 활용

### 1. PyG APPNPConv

```python
from torch_geometric.nn import APPNP as APPNPLayer

class MyAPPNP(torch.nn.Module):
    def __init__(self, d_in, d_hid, d_out, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(d_in, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_out)
        self.prop = APPNPLayer(K=K, alpha=alpha)
    
    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return self.prop(x, edge_index)
```

### 2. JK-Net (PyG)

```python
from torch_geometric.nn.models import JumpingKnowledge

# Layer outputs collect
jk = JumpingKnowledge(mode='concat')   # or 'max', 'lstm'
final = jk(layer_outputs)
```

### 3. Hyperparameter

- **APPNP $\alpha$**: 0.1 (Cora 표준), 0.2 (Citeseer), 0.05 (큰 graph)
- **APPNP $K$**: 10 (충분한 수렴), 20+ (large graph)
- **JKN mode**: concat (단순), LSTM (best but slow), max (작은 dim)

### 4. Combined: GCNII (Chen 2020)

GCNII = APPNP + identity mapping per layer. 이론과 실증 모두 강력. Cora 84.4%, 64-layer 가능.

### 5. Heterophilic Graph

APPNP 가 heterophily 에서 약함 (low-pass 강조). 변종: GPRGNN (Chien 2021) — learnable polynomial coefficient $\theta_k$ 로 high-pass도 가능.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| **APPNP**: low-pass filter $\alpha/(1-(1-\alpha)\mu)$ | Heterophilic graph 약함 |
| Closed-form $O(n^3)$ inverse | Large graph 시 iterative ($K$ steps) 필요 |
| Single $\alpha$ | Adaptive (per-node, per-channel) $\alpha$ 가능 (PPNP, GPR-GNN) |
| **JKN**: linear concat | Layer-wise weight 학습 $\Rightarrow$ overfitting 위험 |
| 모든 layer $H^{(l)}$ 같은 dim | Heterogeneous dim 시 별도 처리 |
| Initial $H^{(0)} = $ MLP(X) | 깊이가 필요한 task 시 limitation |

---

## 📌 핵심 정리

$$\boxed{\text{APPNP: } Z^* = \alpha (I - (1-\alpha) \tilde P)^{-1} H^{(0)}}$$

$$\boxed{\hat g_{\text{APPNP}}(\mu) = \frac{\alpha}{1 - (1-\alpha)\mu} \quad \text{(rational filter, no collapse)}}$$

$$\boxed{\text{JKN-concat: } h_v^{\text{JK}} = [h_v^{(1)} \| \cdots \| h_v^{(L)}]}$$

| 방법 | Over-smoothing 해결 | 깊이 한계 |
|------|---------------------|----------|
| APPNP | 모든 hop weighted geometric series | 무제한 (closed-form) |
| JKN | All-layer concat 으로 multi-hop 보존 | $L$ 까지 |
| GCNII | Initial residual + identity | 64+ layer |
| GPR-GNN | Learnable polynomial coeffs | 무제한 |

핵심: APPNP / JKN 이 **본질적 over-smoothing 해결** — DropEdge / PairNorm 의 cosmetic mitigation 보다 강력.

---

## 🤔 생각해볼 문제

**문제 1** (기초): APPNP $\alpha = 0.5$ 의 spectral filter $\hat g(\mu)$ 를 dominant eigenvalue $\mu_1 = 1$ 과 second $\mu_2 = 0.8$ 에서 계산하라.

<details>
<summary>해설</summary>

$\hat g_{\text{APPNP}}(\mu) = \alpha / (1 - (1-\alpha)\mu)$.

$\alpha = 0.5$, $1 - \alpha = 0.5$.

- $\mu_1 = 1$: $\hat g(1) = 0.5 / (1 - 0.5 \cdot 1) = 0.5 / 0.5 = 1$
- $\mu_2 = 0.8$: $\hat g(0.8) = 0.5 / (1 - 0.5 \cdot 0.8) = 0.5 / 0.6 \approx 0.833$

**비교 with GCN $L = 1$**: $\hat g_{\text{GCN}}(\mu) = \mu$:
- $\mu_1 = 1$: 1
- $\mu_2 = 0.8$: 0.8

**비교 with GCN $L = \infty$**: $\hat g(\mu) = \mu^L$:
- $\mu_1 = 1$: 1
- $\mu_2 = 0.8$: $0.8^L \to 0$

따라서:
- APPNP: 모든 component 가 거의 비슷한 weight (1 vs 0.83) — non-collapse
- GCN $L = \infty$: dominant 만 살아남음 — collapse

이는 APPNP 의 "no over-smoothing" 의 spectral 증명. $\square$

</details>

**문제 2** (심화): $\alpha$ 가 작을수록 graph 정보 ↑, 클수록 initial 정보 ↑. 이 trade-off 의 optimal point 가 dataset 마다 다른 이유?

<details>
<summary>해설</summary>

**Optimal $\alpha$** 의 결정 요소:

1. **Graph 의 information density**:
   - Sparse, weak structural info: 큰 $\alpha$ (initial feature 더 의존)
   - Dense, rich structural: 작은 $\alpha$ (graph 더 활용)

2. **Initial feature $X$ 의 quality**:
   - Strong feature (Cora 의 word-bag): 큰 $\alpha$ — feature 만으로도 분류 가능
   - Weak feature (one-hot only): 작은 $\alpha$ — graph 에 의존

3. **Task 의 long-range dependency**:
   - Local pattern (Cora citation classification): 작은 $\alpha$ ok (1-3 hop info 충분)
   - Long-range (chemistry, social network): 큰 $\alpha$ may help (initial info preservation)

4. **Graph diameter**:
   - Small diameter ($\log n$): 작은 $\alpha$ ok (full graph 빠르게 reach)
   - Large diameter (path-like): 큰 $\alpha$ helpful (avoid over-smoothing)

**Empirical optimal** (Klicpera 2019):
- Cora: $\alpha = 0.1$
- Citeseer: $\alpha = 0.2$
- Pubmed: $\alpha = 0.1$
- MS Academic: $\alpha = 0.05$

**일반화 가이드**:
- $\alpha$ 가 **stationary distribution to initial 의 trade-off knob**
- $\alpha = 0.1$ is the universal safe default
- Cross-validation tuning 으로 $\pm 0.05$ 조정

**Adaptive $\alpha$**: PPNP variant 에서 per-node 또는 per-feature $\alpha_v$ 학습 (Klicpera 2019 supplementary). 약간의 향상.

</details>

**문제 3** (논문 비평): APPNP 가 "본질적 over-smoothing 해결" 이라 주장하지만, low-pass filter 형태로 high-frequency information 을 못 살린다. Heterophilic graph 에서의 한계와 GPR-GNN 의 보강을 분석하라.

<details>
<summary>해설</summary>

**APPNP 의 spectral filter limitation**:

$\hat g_{\text{APPNP}}(\mu) = \alpha / (1 - (1-\alpha)\mu)$.

- $\mu \to 1$: $\hat g \to 1$ (smooth/low-freq emphasized)
- $\mu \to 0$ or negative: $\hat g \to \alpha < 1$ (high-freq dampened)

따라서 **low-pass filter** — graph signal 의 smooth component 강조.

**Heterophilic graph (이웃이 다른 class)** 에서의 문제:

Homophily 가정: "이웃끼리 비슷" — low-pass (smoothing) 가 도움.
Heterophily: "이웃끼리 다름" — high-pass (분리) 가 필요. APPNP 가 low-pass 만 → 정보 손실.

**Empirical evidence**:

- Texas, Wisconsin, Cornell (heterophilic citation): GCN, APPNP < MLP. 그래프 정보 가 오히려 해.
- Squirrel, Chameleon (Wikipedia): 같은 pattern.

**GPR-GNN (Generalized PageRank GNN, Chien 2021)** 의 보강:

APPNP 의 fixed coefficient $(1-\alpha) \alpha^k$ 를 **learnable** $\gamma_k$ 로 replace:
$$
Z = \sum_{k=0}^K \gamma_k \tilde P^k H^{(0)}
$$

(coefficients $\gamma_k$ 자체 학습)

**Learned $\gamma_k$ patterns**:
- Homophilic graph: $\gamma_k \approx (1-\alpha) \alpha^k$ — APPNP-like
- Heterophilic graph: 일부 $\gamma_k < 0$ — high-pass 효과

**다른 보강**:

1. **FAGCN (Bo 2021)**: Signed attention $\alpha_{ij} \in [-1, 1]$ — high-pass possible.
2. **H2GCN (Zhu 2020)**: Higher-order neighbors + ego/neighborhood separation.
3. **JacobiConv (Wang 2022)**: Jacobi polynomial basis for spectral filter.
4. **Specformer (Bo 2023)**: Transformer on eigenvalue tokens.

**Modern conclusion**:

- Homophilic graph: APPNP, GCN 충분
- Heterophilic graph: GPR-GNN, FAGCN, H2GCN 등 specific architecture 필요
- "Universal" GNN: 어려움 — task-specific filter 학습 (GPR-GNN family)

따라서 APPNP 는 graph signal processing literature 의 **low-pass filter 의 closed-form** — 좋은 baseline 이지만 universal solution X. Modern GNN 의 spectral filter 연구는 이 한계 보강이 주제.

</details>

---

<div align="center">

[◀ 이전](./04-sampling-mitigation.md) | [📚 README](../README.md) | [다음 ▶](../ch6-applications/01-node-classification.md)

</div>
