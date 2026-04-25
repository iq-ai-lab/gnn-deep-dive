# 01. Spectral Convolution의 정의 (Bruna 2014)

## 🎯 핵심 질문

- Graph signal 과 filter 의 convolution을 어떻게 정의해야 classical CNN의 convolution 과 일치하는가?
- 왜 Bruna 2014는 "eigenbasis 에서의 element-wise 곱"으로 정의했는가?
- $g *_G x = U(\hat{g}(\Lambda) \odot U^T x)$ 의 학습 파라미터는 무엇이고 표현력은 어떠한가?
- Spectral GCN 의 세 가지 한계 ($O(n^3)$ 분해, non-localized, transferability) 는 무엇인가?
- 왜 ChebNet 으로 대체되었는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

CNN의 핵심은 convolution이지만, "convolution"이 그래프에서 무엇인지는 자명하지 않습니다:

1. **그래프에는 translation 이 없음** — Pixel은 위치 좌표가 있고 평행이동이 자연스럽지만, graph node는 절대 위치가 없음
2. **Local kernel 정의 불명확** — CNN의 $3\times3$ kernel은 spatial neighborhood. 그래프에서는 노드별 degree 가 다름 → fixed kernel size 불가능
3. **이론적 출발점 필요** — GNN 의 모든 후속 연구 (ChebNet, GCN, MPNN) 는 spectral 정의를 출발점으로 함

Bruna et al. 2014 의 **"Spectral Networks and Deep Locally Connected Networks on Graphs"** 가 첫 번째 답을 줍니다: classical convolution theorem 을 이용해 **frequency domain 에서 정의**. 이는 후속 모든 spectral 방법의 기반.

이 문서에서는 spectral convolution 의 정의·표현력·한계를 정리하고, ChebNet (Ch2-02) 으로 가는 동기를 마련합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch1-05](../ch1-graph-laplacian/05-graph-fourier.md) — Graph Fourier Transform $\hat{x} = U^T x$
- [CNN Deep Dive Ch1-05](https://github.com/iq-ai-lab/cnn-deep-dive): Classical convolution theorem $\widehat{f * g} = \hat{f} \cdot \hat{g}$
- Spectral theorem for symmetric matrices

---

## 📖 직관적 이해

### Classical Convolution Theorem 의 복습

연속·이산 convolution 의 유명한 사실:
$$
\widehat{f * g} = \hat{f} \cdot \hat{g}
$$

**Convolution = frequency domain 의 element-wise 곱**. CNN 에서 fast convolution 알고리즘 (FFT-based conv) 의 기반.

### Graph로의 일반화

그래프에서는 spatial convolution 정의가 자명하지 않지만, Fourier transform 은 GFT 로 자연스러움. 따라서 **convolution 을 frequency domain 에서 정의**:

$$
g *_G x := \text{IGFT}(\text{GFT}(g) \odot \text{GFT}(x)) = U(\hat{g} \odot U^T x)
$$

이 정의는 cycle graph $C_n$ 에서 classical circular convolution 과 정확히 일치 (Ch1-05 정리 5.5 의 확장).

### Filter 의 학습 파라미터

Bruna 의 정의: 학습 파라미터 = $\hat{g}(\lambda_k) \in \mathbb{R}$ for each $k = 1, \ldots, n$. 즉 **$n$ 개 자유 파라미터** — 각 frequency 에서의 filter response.

**문제**: $n$ 개 파라미터가 그래프 size에 비례 → overfitting 위험 + scalability 문제 + non-transferable.

### Spatial Locality 와의 단절

CNN의 $3 \times 3$ kernel 은 spatial locality 보장 — 인접 픽셀만 본다. 하지만 spectral filter $\hat{g}(\lambda_k)$ 가 임의 함수이면 spatial locality 가 깨질 수 있음:

- Smooth $\hat{g}$ (작은 $\lambda$ 만 강조) → local
- Oscillatory $\hat{g}$ → highly non-local

이 문제를 해결하기 위해 ChebNet 은 polynomial 제약 (Ch2-02).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Graph Signal

Graph signal $x: V \to \mathbb{R}$, vector form $x \in \mathbb{R}^n$. Multi-channel: $X \in \mathbb{R}^{n \times d}$ (each column = one feature).

### 정의 1.2 — Spectral Convolution (Bruna 2014)

Graph $G$ 와 그 Laplacian $L = U \Lambda U^T$ 가 주어졌을 때, signal $x$ 와 spectral filter $\hat{g}: [0, \lambda_{\max}] \to \mathbb{R}$ (여기서 $\hat{g}(\lambda_k)$ 는 $k$-th frequency 에서의 response) 의 spectral convolution:
$$
g *_G x := U \cdot \text{diag}(\hat{g}(\lambda_1), \ldots, \hat{g}(\lambda_n)) \cdot U^T x = U \hat{g}(\Lambda) U^T x
$$

학습 파라미터: $\theta_k = \hat{g}(\lambda_k)$ for $k = 1, \ldots, n$, 즉 $n$ 차원 vector.

### 정의 1.3 — Spectral GCN Layer

다중 입력·출력 채널 ($d_{\text{in}}, d_{\text{out}}$) 의 spectral conv layer:
$$
H^{(l+1)}_{:, j} = \sigma \left( \sum_{i=1}^{d_{\text{in}}} U \hat{g}_{ij}^{(l)}(\Lambda) U^T H^{(l)}_{:, i} \right) \quad j = 1, \ldots, d_{\text{out}}
$$

각 input·output 채널 쌍마다 별개의 spectral filter $\hat{g}_{ij}^{(l)}$. Total parameters per layer = $d_{\text{in}} \cdot d_{\text{out}} \cdot n$.

### 정의 1.4 — Smoothness Regularization

$\hat{g}_{ij}^{(l)}$ 가 임의 함수이면 overparameterized → Bruna는 smooth parameterization 제안:
$$
\hat{g}(\lambda) = \sum_{p=0}^P \theta_p \cdot \mathcal{B}_p(\lambda)
$$

$\mathcal{B}_p$ = cubic spline basis. 파라미터 수 $P \ll n$. 이는 spatial locality 의 첫 시도.

---

## 🔬 정리와 증명

### 정리 1.1 — Spectral Convolution 의 동치 표현

$$
g *_G x = U \hat{g}(\Lambda) U^T x = \hat{g}(L) x
$$

(단, $\hat{g}(\lambda)$ 가 well-defined function 일 때)

**증명**: Spectral theorem $L = U \Lambda U^T$. Function calculus: $\hat{g}(L) = U \hat{g}(\Lambda) U^T$ (대각화 가능 matrix 의 함수 정의). $\square$

**의미**: Spectral filter 는 $L$ 의 함수 — Laplacian 의 polynomial / power series 등으로 표현 가능 → ChebNet 으로 자연스럽게 일반화 (정리 1.3).

### 정리 1.2 — Cycle Graph 에서 Classical Conv 와 일치

$G = C_n$ 일 때 spectral conv $g *_G x$ 는 classical circular convolution과 정확히 같다.

**증명 sketch**: $C_n$ 의 $A$ (그리고 $L$) 는 circulant. Eigenvector = DFT basis. GFT = DFT. 따라서:
$$
g *_G x = \text{IDFT}(\text{DFT}(g) \cdot \text{DFT}(x))
$$
이는 classical circular convolution 의 정의 그 자체. $\square$

이는 **spectral convolution 이 graph 일반화로서 옳음**을 보여주는 결정적 증거.

### 정리 1.3 — Polynomial Filter는 Localized

**Theorem**: $\hat{g}(\lambda) = \sum_{k=0}^K \theta_k \lambda^k$ ($K$-th polynomial) 이면 spectral conv 는 **$K$-hop localized**:
$$
g *_G x = \sum_{k=0}^K \theta_k L^k x
$$

각 항 $L^k x$ 는 $k$-hop neighbor 의 weighted sum.

**증명**: 정리 1.1 + Polynomial 가 well-defined matrix function:
$$
\hat{g}(L) = \sum_k \theta_k L^k
$$

$L^k$ 의 $(i, j)$ 성분은 $i, j$ 사이 길이 $k$ walk 수의 weighted form (Ch1-01 정리 1.1 의 일반화). 따라서 $i$ 의 새 값은 $\leq k$-hop 노드의 weighted sum. $\square$

이 정리가 **ChebNet 의 직접 동기** — polynomial 로 제약하면 localized + parameter-efficient.

### 정리 1.4 — 학습 파라미터의 한계

Spectral GCN 의 layer-wise parameter 수: $d_{\text{in}} \cdot d_{\text{out}} \cdot n$.

- Cora ($n \approx 3000$): 적당
- Reddit ($n \approx 230000$): $230k \times d^2$ 파라미터 → 비현실적
- OGB-products ($n \approx 2.4M$): 완전 비현실적

또한 $U$ 는 $O(n^3)$ 분해 비용 + $O(n^2)$ 메모리. → **scalability 문제**.

### 정리 1.5 — Non-Transferability

Spectral filter $\hat{g}(\lambda_k)$ 는 graph-specific eigenvalue $\lambda_k$ 에 의존. 새 graph (다른 $\Lambda$) 에서는 학습된 $\hat{g}$ 가 의미 잃음.

**Example**: Cora 에서 학습한 spectral filter 를 Citeseer 에 적용 — 두 graph 의 spectrum 이 다르므로 inductive transfer 불가.

이는 **inductive learning 의 본질적 문제** — ChebNet (smooth function $\hat{g}$) 또는 GCN (graph-agnostic polynomial) 으로 해결.

---

## 💻 NumPy 구현 검증

### 실험 1 — Bruna 식 Spectral Convolution

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(1)
D_inv_sqrt = np.diag(1 / np.sqrt(deg))
L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

eigvals, U = np.linalg.eigh(L_sym)

def spectral_conv(L_eigvals, U, x, g_hat):
    """
    g_hat: function lambda -> response
    """
    g_diag = g_hat(L_eigvals)
    return U @ (g_diag * (U.T @ x))

# 입력 신호: 노드 0에 delta
x = np.zeros(n); x[0] = 1.0

# Filter 1: Low-pass (Heat kernel)
def lowpass(lam, t=2.0):
    return np.exp(-lam * t)

# Filter 2: High-pass
def highpass(lam):
    return lam   # increasing in λ

y_low = spectral_conv(eigvals, U, x, lowpass)
y_high = spectral_conv(eigvals, U, x, highpass)

print(f'Low-pass filtered (top-5 nodes by value): {np.argsort(-y_low)[:5]}')
print(f'High-pass filtered (top-5 nodes by abs value): {np.argsort(-np.abs(y_high))[:5]}')

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pos = nx.spring_layout(G, seed=42)
for ax, vec, title in zip(axes, [x, y_low, y_high],
                           ['Input δ at node 0', 'Low-pass (heat)', 'High-pass (Laplacian)']):
    nx.draw(G, pos, ax=ax, node_color=vec, cmap='RdBu',
            with_labels=False, node_size=80)
    ax.set_title(title)
plt.show()
```

### 실험 2 — Polynomial Filter의 K-hop Locality

```python
# K-th polynomial filter
def poly_filter(L, x, theta):
    """g(L) x = sum_k theta_k L^k x"""
    K = len(theta)
    out = theta[0] * x
    Lk_x = x.copy()
    for k in range(1, K):
        Lk_x = L @ Lk_x
        out = out + theta[k] * Lk_x
    return out

# K=2 polynomial
theta = np.array([1.0, -0.5, 0.1])

# Delta at node 0 → K=2 propagation
x_delta = np.zeros(n); x_delta[0] = 1.0
y_poly = poly_filter(L_sym, x_delta, theta)

# 직접 BFS로 hop 거리 계산
dist = nx.single_source_shortest_path_length(G, 0)
hop = np.array([dist.get(i, np.inf) for i in range(n)])

# K=2 hop 까지만 nonzero 인지 확인
nonzero_mask = np.abs(y_poly) > 1e-10
print(f'Nodes with nonzero output: {np.where(nonzero_mask)[0]}')
print(f'Their hop distances from 0: {hop[nonzero_mask]}')
print(f'Max hop = {hop[nonzero_mask].max()}  (should be ≤ K=2)')
```

### 실험 3 — Bruna Layer의 Parameter Count

```python
def bruna_layer_params(n, d_in, d_out, K_smooth=None):
    if K_smooth is None:
        return d_in * d_out * n   # 자유 spectral
    else:
        return d_in * d_out * K_smooth   # smooth basis

print(f'Karate ({n}=34): Bruna full = {bruna_layer_params(n, 16, 16)}')   # 8704
print(f'Cora-like (n=3000), full = {bruna_layer_params(3000, 16, 16):,}')  # 768k
print(f'Reddit-like (n=230k), full = {bruna_layer_params(230_000, 16, 16):,}')   # 60M
print(f'Smooth basis K=10:  {bruna_layer_params(230_000, 16, 16, K_smooth=10)}')   # 2560
```

### 실험 4 — Eigendecomposition 비용 측정

```python
import time

sizes = [50, 100, 200, 500, 1000]
times = []
for sz in sizes:
    G_rand = nx.erdos_renyi_graph(sz, 0.05, seed=0)
    A_rand = nx.adjacency_matrix(G_rand).toarray().astype(float)
    deg_r = A_rand.sum(1) + 1e-6
    L_r = np.eye(sz) - np.diag(1/np.sqrt(deg_r)) @ A_rand @ np.diag(1/np.sqrt(deg_r))
    t0 = time.time()
    np.linalg.eigh(L_r)
    times.append(time.time() - t0)
    print(f'n={sz}: {times[-1]:.3f}s')

plt.loglog(sizes, times, 'o-')
plt.xlabel('n'); plt.ylabel('time (s)')
plt.title('Eigendecomposition cost ~ O(n^3)')
plt.show()
```

---

## 🔗 실전 활용

### 1. Bruna 의 historical 의의

GNN 의 "0번째" 모델. 이후 모든 spectral GNN 의 출발점. 단, 실전에서는 거의 사용되지 않음 — ChebNet 또는 GCN 사용.

### 2. Scalable Spectral 의 시도

- **Lanczos / Arnoldi**: Top-$k$ eigenvector 만 계산, $O(k m + k^2 n)$. 단 spectral filter 가 truncated representation 에 대해서만 정의.
- **Sparse approximation**: Power iteration + Chebyshev recurrence (ChebNet 의 직접 동기)

### 3. 현대 spectral 방법

- **Specformer** (Bo 2023): Transformer for spectral filtering with eigenvalue tokens
- **JacobiConv** (Wang 2022): Jacobi polynomial 기반 graph filter
- 이들도 모두 Bruna 의 spectral convolution framework 위에서 작동

### 4. Spectral Audio / GSP

Image processing, audio signal processing on graphs (sensor network) 등에서 spectral filter 자체는 중요. GNN에서는 polynomial 근사로 단순화되지만, GSP 분야에서는 raw spectral 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $L$ 의 full eigendecomposition 가능 | $O(n^3)$ — large graph 불가 |
| Filter parameter $n$ 차원 | Overparameterized, scalability X |
| Graph-specific $\hat{g}(\lambda_k)$ | Inductive transfer 불가 |
| Spatial locality 보장 X | Polynomial parametrization 필요 (ChebNet) |
| Connected, undirected 가정 | Disconnected 시 component-wise, directed 시 별도 일반화 |
| Static graph | Dynamic graph 시 매 step decomposition 비용 |

---

## 📌 핵심 정리

$$\boxed{g *_G x = U \hat{g}(\Lambda) U^T x = \hat{g}(L) x \quad \text{(Bruna 2014)}}$$

| 항목 | Bruna Spectral GCN |
|------|--------------------|
| **Filter parameterization** | $\hat{g}(\lambda_k) \in \mathbb{R}$ — full spectrum |
| **Parameter count per layer** | $d_{\text{in}} \cdot d_{\text{out}} \cdot n$ |
| **Locality** | None (자유 함수) |
| **Eigendecomposition cost** | $O(n^3)$ (한 번) |
| **Inference** | $O(n^2)$ (matmul $U \cdot \cdot \cdot U^T$) |
| **Inductive** | ✗ |
| **현대 사용** | 거의 없음 — ChebNet/GCN 으로 대체 |
| **이론적 의의** | Spectral GCN family 의 foundation |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $C_4$ (4-cycle) 의 spectral convolution을 손으로 계산하라. Filter $\hat{g}(\lambda) = \lambda$ (high-pass) 와 input $x = (1, 0, 0, 0)$ (delta at node 0).

<details>
<summary>해설</summary>

$C_4$: $L_{\text{sym}}$ 고유값 $\{0, 1, 1, 2\}$. 고유벡터:
- $u_1 = (1,1,1,1)/2$ ($\lambda=0$)
- $u_2, u_3$: $\lambda=1$ degenerate, 예: $u_2 = (1,0,-1,0)/\sqrt 2$, $u_3 = (0,1,0,-1)/\sqrt 2$
- $u_4 = (1,-1,1,-1)/2$ ($\lambda=2$)

GFT: $\hat{x}_k = u_k^T x$ for $x = (1,0,0,0)$:
- $\hat{x}_1 = 1/2$, $\hat{x}_2 = 1/\sqrt 2$, $\hat{x}_3 = 0$, $\hat{x}_4 = 1/2$

Filter $\hat{g}(\lambda) = \lambda$: $\hat{g}(\Lambda) \hat{x} = (0 \cdot 1/2, 1 \cdot 1/\sqrt 2, 1 \cdot 0, 2 \cdot 1/2) = (0, 1/\sqrt 2, 0, 1)$.

IGFT:
$$
y = U \hat{g}(\Lambda) U^T x = u_2 \cdot 1/\sqrt 2 + u_4 \cdot 1
$$

$$
= (1, 0, -1, 0)/2 + (1, -1, 1, -1)/2 = (1, -1/2, 0, -1/2)
$$

확인: $L_{\text{sym}} x$ 직접 계산 — node 0의 deg 정규화: $(L_{\text{sym}} x)_0 = 1 \cdot 1 = 1$, $(L_{\text{sym}} x)_1 = -1/\sqrt{d_0 d_1} \cdot 1 = -1/2$ (모두 deg=2), $(L_{\text{sym}} x)_3 = -1/2$, $(L_{\text{sym}} x)_2 = 0$ (1-hop 거리 X). → 일치. ✓ $\square$

</details>

**문제 2** (심화): Spectral filter $\hat{g}(\lambda)$ 가 polynomial of order $\leq K$ 인 것과 spatial conv 가 $K$-hop localized 인 것이 동치임을 증명하라. (Hint: $L^k$ 의 sparsity 패턴)

<details>
<summary>해설</summary>

**($\Rightarrow$)** $\hat{g}(\lambda) = \sum_{k=0}^K \theta_k \lambda^k$. 정리 1.3 에서 $g *_G x = \sum_k \theta_k L^k x$.

$L^k$ 의 $(i, j)$ 성분은 $i$ 와 $j$ 사이 path of length $\leq k$ 가 있을 때만 nonzero (정확히는 paths of length $k$, but $L^k = (D-A)^k$ 의 expansion이 lower power $L^l$ for $l \leq k$ 를 포함). 따라서 hop distance $> K$ 인 $(i, j)$ 에서 모든 $L^k_{ij} = 0$ → output 도 $K$-hop 까지만 nonzero.

**($\Leftarrow$)** $g *_G x$ 가 $K$-hop localized라고 가정. 즉 $\hat{g}(L)_{ij} = 0$ for hop$(i, j) > K$. $L^k$ 들이 "$\leq k$-hop" 까지의 정보를 인코딩 — $\{L^0, L^1, \ldots, L^K\}$ 가 $K$-hop locality 를 가진 matrix space 의 basis (확장 보장 by Cayley-Hamilton-like argument). 따라서 $\hat{g}(L) = \sum_{k=0}^K \theta_k L^k$ 형태로 표현 가능 → $\hat{g}$ 가 polynomial of order $\leq K$. $\square$

**수치 보강**: 더 엄밀한 증명은 graph polynomial algebra 와 minimal polynomial 의 degree 분석 (Hammond 2011).

</details>

**문제 3** (논문 비평): Bruna 2014 가 "smooth filter" 로 cubic spline basis 를 제안한 이유와, 이것이 ChebNet (Chebyshev polynomial basis) 으로 대체된 이유를 비교하라.

<details>
<summary>해설</summary>

**Bruna 의 Cubic Spline**:
- 동기: Smooth $\hat{g}(\lambda)$ 가 spatial locality 일부 보장 (smoothness 가 Heisenberg-like 으로 spatial decay 유도)
- 파라미터 수 ↓ ($n \to P \ll n$)
- 한계: 여전히 $U$ 필요 (full eigendecomposition), inductive 불가

**ChebNet 의 Chebyshev Polynomial**:
- Polynomial 로 제약 → $L^k x$ 직접 계산 ($U$ 불필요)
- Chebyshev 의 equioscillation 성질 — uniform approximation 최적
- Recurrence $T_{k+1}(L) = 2L T_k(L) - T_{k-1}(L)$ 로 $O(mK)$ 계산
- $K$-hop locality 자동 보장 (정리 1.3 + 정리 1.5)
- Inductive: 파라미터 $\theta_k$ 가 graph 와 독립

**결정적 차이**: ChebNet 은 $U$ 없이 작동 — graph-agnostic + scalable + inductive. Cubic spline 은 여전히 $U$ 필요한 spectral 방법.

**현대적 관점**: Specformer 등은 다시 spectral domain 으로 회귀하지만, eigenvalue 만 사용 (eigenvector 는 모호) → spectral 의 통찰을 보존하면서 polynomial 의 효율 결합. 따라서 spectral vs polynomial 의 trade-off 는 여전히 활발 — Ch2-04 에서 통합 관점.

</details>

---

<div align="center">

[◀ 이전](../ch1-graph-laplacian/06-random-walk-pagerank.md) | [📚 README](../README.md) | [다음 ▶](./02-chebnet.md)

</div>
