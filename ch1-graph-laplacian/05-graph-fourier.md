# 05. Graph Fourier Transform

## 🎯 핵심 질문

- Graph Fourier Transform (GFT) 의 정의와 classical Fourier transform 과의 유비는 무엇인가?
- 왜 Laplacian의 고유벡터가 그래프에서의 "frequency basis" 역할을 하는가?
- 작은 고유값 = smooth signal, 큰 고유값 = oscillatory signal 인 이유는?
- Parseval-like identity $x^T L x = \sum_k \lambda_k |\hat{x}_k|^2$ 의 의미는?
- GFT가 Spectral GCN (Bruna 2014) 의 출발점이 되는 이유는?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Classical Fourier transform이 신호 처리·CNN의 이론적 기반이듯, Graph Fourier Transform은 GNN의 spectral 분석 전체의 기반:

1. **Spectral Convolution 정의** (Bruna 2014) — graph signal 과 filter 의 convolution을 GFT로 정의 (Ch2-01)
2. **ChebNet 의 polynomial filter** — eigenbasis 에서의 element-wise 곱 = polynomial of $L$ (Ch2-02)
3. **Smoothness 측정** — $\hat{x}_k$ 분포가 high-frequency 성분 인덱스
4. **Graph Signal Processing (GSP)** — 그래프 위 신호의 filtering, denoising, sampling 이론
5. **Positional Encoding** — LapPE (Ch4-05) 가 GFT 좌표를 노드 PE로 사용

이 문서에서는 **GFT의 정의·성질·classical Fourier 와의 유비**를 정리하고, Spectral GCN 의 무대를 마련합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-unnormalized-laplacian.md](./02-unnormalized-laplacian.md), [03-normalized-laplacian.md](./03-normalized-laplacian.md), [04-spectral-theory.md](./04-spectral-theory.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Spectral theorem, orthonormal basis
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): Fourier transform, Plancherel theorem (선택)

---

## 📖 직관적 이해

### Classical Fourier 의 복습

연속 함수 $f: \mathbb{R} \to \mathbb{R}$ 의 Fourier transform:
$$
\hat{f}(\xi) = \int f(t) e^{-2\pi i \xi t} dt
$$

핵심 사실: $e^{2\pi i \xi t}$ 는 **Laplace operator** $-\Delta = -d^2/dt^2$ 의 eigenfunction. 즉:
$$
-\Delta e^{2\pi i \xi t} = (2\pi \xi)^2 e^{2\pi i \xi t}
$$

작은 $\xi$ = smooth (천천히 변화), 큰 $\xi$ = oscillatory (빠른 변화).

### Graph 에서의 자연스러운 일반화

Graph Laplacian $L_{\text{sym}}$ 이 $-\Delta$ 의 이산판. 그 eigenvector $\{u_k\}$ 가 "graph frequency basis":

$$
L_{\text{sym}} u_k = \lambda_k u_k
$$

**Graph Fourier Transform**:
$$
\hat{x}_k = u_k^T x = \langle u_k, x \rangle
$$

작은 $\lambda_k$ → smooth basis vector $u_k$ (인접 노드끼리 비슷한 값).
큰 $\lambda_k$ → oscillatory basis vector (인접 노드끼리 부호 반대).

### 직관: Path Graph $P_n$

$P_n$ 의 Fiedler vector ($\lambda_2$): 한쪽 끝부터 다른 끝까지 부드럽게 변화 (low frequency).
높은 $\lambda$ 의 eigenvector: $(+1, -1, +1, -1, \ldots)$ — 이웃 노드 부호 반대 (high frequency).

이는 Discrete Fourier Transform (DFT) 의 sinusoid 와 정확히 같은 패턴.

### Cycle $C_n$ → DFT

$C_n$ 의 Laplacian eigenvector = $(1, \omega^k, \omega^{2k}, \ldots) / \sqrt n$ ($\omega = e^{2\pi i/n}$) — **DFT basis 그대로**. 즉, GFT 가 cycle graph 위에서 **classical DFT**.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Graph Fourier Transform

Graph Laplacian $L = U \Lambda U^T$, $U = [u_1 | u_2 | \cdots | u_n]$, $u_k$ orthonormal.

**Graph Fourier transform** $\hat{x} \in \mathbb{R}^n$:
$$
\hat{x} = U^T x \quad \Leftrightarrow \quad \hat{x}_k = u_k^T x
$$

**Inverse GFT**:
$$
x = U \hat{x} = \sum_k \hat{x}_k u_k
$$

(Spectral decomposition / synthesis)

### 정의 5.2 — Frequency

$k$-번째 GFT coefficient $\hat{x}_k$ 의 "frequency" = $\lambda_k$ (Laplacian eigenvalue). 작은 $\lambda$ = low freq, 큰 $\lambda$ = high freq.

### 정의 5.3 — Bandwidth and Smoothness

신호 $x$ 가 $\lambda_K$-bandlimited 이려면 $\hat{x}_k = 0$ for all $k > K$. **Smooth signal** = 대부분의 에너지가 작은 $\lambda$ 에 집중된 signal.

### 정의 5.4 — Graph Filtering

Filter $g: \mathbb{R} \to \mathbb{R}$ 가 적용된 신호:
$$
y = U g(\Lambda) U^T x \quad \Leftrightarrow \quad \hat{y}_k = g(\lambda_k) \hat{x}_k
$$

(GFT 후 element-wise 곱, IGFT — classical convolution theorem 유사)

---

## 🔬 정리와 증명

### 정리 5.1 — Parseval-like Identity

$$
\|x\|^2 = \sum_k |\hat{x}_k|^2 = \|\hat{x}\|^2
$$

**증명**: $U$ orthogonal, $\|x\|^2 = x^T x = (U \hat{x})^T (U \hat{x}) = \hat{x}^T U^T U \hat{x} = \hat{x}^T \hat{x} = \|\hat{x}\|^2$. $\square$

### 정리 5.2 — Smoothness 와 Spectrum

$$
x^T L x = \sum_k \lambda_k |\hat{x}_k|^2
$$

**증명**:
$$
x^T L x = x^T U \Lambda U^T x = (U^T x)^T \Lambda (U^T x) = \hat{x}^T \Lambda \hat{x} = \sum_k \lambda_k \hat{x}_k^2
$$
$\square$

**해석**: $x^T L x$ (graph energy) 는 GFT 좌표의 weighted L2 norm. Weight = eigenvalue. 작은 $\lambda$ 에 집중된 신호 = 작은 energy = smooth.

### 정리 5.3 — Filter as Polynomial

Filter $g(\lambda)$ 가 polynomial $g(\lambda) = \sum_{k=0}^K \alpha_k \lambda^k$ 이면:
$$
y = U g(\Lambda) U^T x = g(L) x = \sum_{k=0}^K \alpha_k L^k x
$$

**증명**: $L^k = U \Lambda^k U^T$ ($U$ orthogonal). $\sum \alpha_k L^k = U (\sum \alpha_k \Lambda^k) U^T = U g(\Lambda) U^T$. $\square$

**의미**: Polynomial filter는 $U$ 없이 직접 $L^k x$ 반복 계산 가능 — ChebNet 의 핵심 (Ch2-02). $K$-th polynomial = $K$-hop neighbor 정보.

### 정리 5.4 — Convolution Theorem (Graph 버전)

Graph signal $x$ 와 filter $g$ 의 spectral convolution:
$$
g *_G x := U (g(\Lambda) \odot \hat{x}) = U g(\Lambda) U^T x
$$

(Bruna 2014 의 정의)

이는 classical convolution theorem $\widehat{f * g} = \hat{f} \cdot \hat{g}$ 의 graph 일반화.

### 정리 5.5 — Cycle Graph → DFT 정확 일치

$C_n$ 의 Laplacian eigenvector $u_k$ 가 DFT basis $f_k = (1, \omega^k, \omega^{2k}, \ldots, \omega^{(n-1)k}) / \sqrt n$ 와 일치 (실 형태로).

**증명 sketch**: $C_n$ adjacency $A$ 가 circulant, eigenvector = DFT basis (Ch2-04 에서 자세히). $L = 2I - A$ (regular) → 같은 eigenvector, 고유값만 shift. $\square$

이는 GFT 가 classical DFT 의 **graph 일반화**임을 정확히 보여줌.

### 정리 5.6 — Heat Equation on Graph

Graph 위의 heat diffusion:
$$
\frac{d x(t)}{dt} = -L x(t)
$$

Closed-form: $x(t) = e^{-Lt} x(0) = U e^{-\Lambda t} U^T x(0)$. 각 frequency 성분이 $e^{-\lambda_k t}$ 로 감쇠 — high frequency가 빠르게 사라짐.

이는 GCN propagation 과 매우 유사한 형태로, **그래프 위 diffusion**이 GNN forward의 본질임을 시사.

---

## 💻 NumPy 구현 검증

### 실험 1 — GFT와 IGFT 검증

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

# 임의 signal
x = np.random.randn(n)
x_hat = U.T @ x       # GFT
x_recon = U @ x_hat   # IGFT
print(f'IGFT(GFT(x)) == x? {np.allclose(x_recon, x)}')   # True

# Parseval
print(f'||x||^2  = {(x**2).sum():.4f}')
print(f'||x_hat||^2 = {(x_hat**2).sum():.4f}')   # equal
```

### 실험 2 — Smooth vs Oscillatory Signal의 Spectrum

```python
# Smooth signal: 작은 eigenvector 들의 선형결합
x_smooth = U[:, :3] @ np.array([3.0, 2.0, 1.0])
# Oscillatory signal: 큰 eigenvector
x_osc = U[:, -3:] @ np.array([3.0, 2.0, 1.0])

x_smooth_hat = U.T @ x_smooth
x_osc_hat = U.T @ x_osc

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].stem(eigvals, x_smooth_hat**2, basefmt=' ')
ax[0].set_title('Smooth signal: 에너지가 낮은 λ에 집중')
ax[0].set_xlabel('λ_k'); ax[0].set_ylabel('|x̂_k|²')

ax[1].stem(eigvals, x_osc_hat**2, basefmt=' ')
ax[1].set_title('Oscillatory signal: 에너지가 높은 λ에 집중')
ax[1].set_xlabel('λ_k'); ax[1].set_ylabel('|x̂_k|²')
plt.tight_layout(); plt.show()

# x^T L x = sum λ_k |x̂_k|²
print(f'Smooth:  x^T L x = {x_smooth @ L_sym @ x_smooth:.4f}')
print(f'         Σ λ_k |x̂_k|² = {(eigvals * x_smooth_hat**2).sum():.4f}')
print(f'Osc:     x^T L x = {x_osc @ L_sym @ x_osc:.4f}')
```

### 실험 3 — Low-pass Filter

```python
# Low-pass filter: g(λ) = 1 if λ < 0.5, else 0
def filter_lowpass(eigvals, threshold=0.5):
    return (eigvals < threshold).astype(float)

g_diag = filter_lowpass(eigvals, 0.5)
G_filter = U @ np.diag(g_diag) @ U.T

# 노이즈 신호 + low-pass
x_clean = U[:, :3] @ np.array([2.0, 1.0, 0.5])
x_noisy = x_clean + 0.5 * np.random.randn(n)
x_filtered = G_filter @ x_noisy

print(f'Original error  : {np.linalg.norm(x_noisy - x_clean):.4f}')
print(f'Filtered error  : {np.linalg.norm(x_filtered - x_clean):.4f}')
# Low-pass 후 error 감소 확인
```

### 실험 4 — Heat Diffusion

```python
# Heat equation: x(t) = exp(-L t) x(0)
def heat_kernel(L, t):
    eigvals, U = np.linalg.eigh(L)
    return U @ np.diag(np.exp(-eigvals * t)) @ U.T

# 한 점에서 시작 (delta function)
x0 = np.zeros(n); x0[0] = 1.0

times = [0, 0.5, 2.0, 10.0]
fig, axes = plt.subplots(1, len(times), figsize=(16, 3))
pos = nx.spring_layout(G, seed=42)
for ax, t in zip(axes, times):
    H = heat_kernel(L_sym, t)
    xt = H @ x0
    nx.draw(G, pos, ax=ax, node_color=xt, cmap='hot',
            with_labels=False, node_size=80)
    ax.set_title(f't = {t}')
plt.suptitle('Heat diffusion on graph')
plt.show()
```

### 실험 5 — Cycle Graph → DFT 일치

```python
n_c = 8
A_c = np.zeros((n_c, n_c))
for i in range(n_c):
    A_c[i, (i+1)%n_c] = 1
    A_c[(i+1)%n_c, i] = 1
L_c = np.diag(A_c.sum(1)) - A_c
eig_c, U_c = np.linalg.eigh(L_c)

# DFT basis (real)
F = np.fft.fft(np.eye(n_c)) / np.sqrt(n_c)
print(f'Cycle eigenvalues:    {eig_c.round(4)}')
print(f'2 - 2cos(2πk/n):      {[round(2-2*np.cos(2*np.pi*k/n_c), 4) for k in range(n_c)]}')

# 일부 eigenvector 시각화
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for idx, k in enumerate([0, 1, 3, 7]):
    axes.flat[idx].plot(U_c[:, k], 'o-')
    axes.flat[idx].set_title(f'u_{k}, λ={eig_c[k]:.3f}')
plt.suptitle('Cycle graph eigenvectors = DFT basis (real)')
plt.show()
```

---

## 🔗 실전 활용

### 1. Spectral GCN (Bruna 2014) 의 출발점

Spectral conv = $g *_G x = U(\hat{g} \odot U^T x)$. 학습 파라미터 = $\hat{g}(\Lambda)$ (각 frequency에서의 filter response).

한계 (Ch2-01): $U$ 전체 필요 → $O(n^3)$ 분해, non-localized filter.

### 2. ChebNet (Ch2-02): Polynomial Filter

$\hat{g}(\lambda) = \sum_k \theta_k T_k(\lambda)$ 로 근사 → $g(L) x$ 를 $L$ 의 polynomial로 직접 계산. $K$-hop locality 자동 보장.

### 3. Graph Signal Processing

GFT-domain에서 filtering·denoising·compression. Image graph → image processing 과 동등.

### 4. Heat Kernel as Similarity

$H_t = e^{-Lt}$ 는 노드 간 "diffusion similarity" — node embedding (Heat Kernel SimRank) 또는 PE 로 사용.

### 5. Laplacian Positional Encoding (Ch4-05)

LapPE = $\sqrt{\lambda_k} u_k$ 같은 형태로 GFT 좌표를 PE로 주입. WL 한계 우회 가능.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Connected graph (단일 component) | Disconnected 시 component-wise GFT |
| Symmetric Laplacian | Directed graph는 non-symmetric, 일반 SVD 또는 Hermitian 확장 필요 |
| Distinct eigenvalues 가정 | 동일 eigenvalue 시 eigenvector 자체가 모호 (eigenspace 임의 회전) → BasisNet 필요 |
| $O(n^3)$ 분해 비용 | 대규모 그래프에서는 불가 → polynomial approximation (ChebNet), Lanczos |
| Static graph | Dynamic graph는 시간 의존 GFT 필요 |
| Real signal | Complex signal 도 가능 but rare in GNN |

---

## 📌 핵심 정리

$$\boxed{\hat{x} = U^T x, \quad x = U \hat{x} \quad \text{(GFT / IGFT)}}$$

$$\boxed{x^T L x = \sum_k \lambda_k |\hat{x}_k|^2 \quad \text{(에너지 = 가중 GFT 스펙트럼)}}$$

$$\boxed{g *_G x = U g(\Lambda) U^T x \quad \text{(spectral convolution)}}$$

| 개념 | 정의 | 응용 |
|------|------|------|
| **GFT** | $\hat{x} = U^T x$ | spectral conv |
| **Frequency** | $\lambda_k$ | smoothness 측정 |
| **Smooth signal** | low-$\lambda$ 에너지 집중 | regularization |
| **Polynomial filter** | $g(L) = \sum \alpha_k L^k$ | ChebNet (Ch2-02) |
| **Heat kernel** | $e^{-Lt}$ | similarity, PE |
| **Cycle ↔ DFT** | $C_n$ 의 GFT = DFT | classical 일반화 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K_3$ (triangle) 의 Laplacian $L$ 의 모든 고유벡터를 손으로 계산하고 GFT basis로서의 의미를 해석하라.

<details>
<summary>해설</summary>

$K_3$: $A = J - I$, $D = 2I$, $L = 3I - J$.

$J$ 의 eigenvector: $\mathbb{1}/\sqrt 3$ (eigenvalue 3), 그리고 $\mathbb{1}^\perp$ 의 임의 직교 basis (eigenvalue 0).

$L$ 의 eigenvector:
- $u_1 = \mathbb{1}/\sqrt 3$, $\lambda_1 = 0$
- $u_2, u_3$: $\mathbb{1}^\perp$ basis, $\lambda_2 = \lambda_3 = 3$

예: $u_2 = (1, -1, 0)/\sqrt 2$, $u_3 = (1, 1, -2)/\sqrt 6$

**GFT 의미**: $\lambda_1 = 0$ → constant signal. $\lambda_2 = \lambda_3 = 3$ → 같은 frequency (degenerate). 모든 non-constant signal이 같은 "frequency"로 간주됨 — $K_n$ 의 spectrum 매우 단순.

**의미**: Highly symmetric graph 는 spectrum이 degenerate, eigenvector 자체가 모호. PE로 사용 시 sign/basis ambiguity 문제. $\square$

</details>

**문제 2** (심화): Path graph $P_n$ 의 Laplacian eigenvector가 cosine basis $u_k(i) = \cos((i - 1/2) \pi k / n)$ 와 일치함을 보여라. (Hint: discrete cosine transform)

<details>
<summary>해설</summary>

$P_n$ Laplacian $L$ 의 $i$-th row:
$$
(L)_{i,j} = \begin{cases} 1 & i = j = 1, n \\ 2 & i = j, 1 < i < n \\ -1 & |i - j| = 1 \\ 0 & \text{otherwise} \end{cases}
$$

(끝 노드는 degree 1, 내부는 degree 2)

Eigenvector candidate $u_k(i) = \cos((i - 1/2) \pi k / n)$ for $k = 0, 1, \ldots, n-1$.

검증: $(L u_k)_i = (\text{deg}_i) u_k(i) - u_k(i-1) - u_k(i+1)$ (boundary 처리 필요).

내부 $i$: $2 \cos(\theta_k(i)) - \cos(\theta_k(i-1)) - \cos(\theta_k(i+1))$. Sum-to-product:
$$
= 2 \cos(\theta_k(i)) - 2 \cos(\theta_k(i)) \cos(\pi k / n) = 2 (1 - \cos(\pi k / n)) u_k(i)
$$

따라서 $\lambda_k = 2 - 2 \cos(\pi k / n)$. Boundary 도 (끝 노드 deg=1) 같은 식 만족 (cosine half-shift trick) → 전체에서 eigenvector 임. $\square$

이는 **Discrete Cosine Transform (DCT-II)** 와 정확히 일치 — Path graph 의 GFT = DCT.

</details>

**문제 3** (논문 비평): Bruna 2014 의 Spectral GCN 이 이론적으로는 우아하지만 실전에서는 사용되지 않는다. 그 이유 3가지를 들고, ChebNet (Defferrard 2016) 이 어떻게 이를 해결했는지 설명하라.

<details>
<summary>해설</summary>

**Spectral GCN의 한계**:

1. **계산 비용 $O(n^3)$**: $L$ 의 full eigendecomposition 필요. 큰 그래프 (Cora $n \sim 3000$, OGB $n \sim 10^6$) 에서 불가능.

2. **Non-localized filter**: 학습 파라미터 $\hat{g}(\lambda_k)$ 가 자유로운 함수. Spatial domain 에서는 noisy / global filter 가능. CNN의 small kernel locality 결여.

3. **Graph-specific learning**: $U$ 가 graph 마다 다름 → 학습된 filter $\hat{g}$ 가 새로운 graph 에 transferable 하지 않음 (inductive 불가).

**ChebNet 의 해결**:

1. **Polynomial approximation**: $\hat{g}(\lambda) = \sum_{k=0}^K \theta_k T_k(\lambda)$ → $g(L) x = \sum_k \theta_k T_k(L) x$. $L$ 의 sparsity 활용 → $O(m K)$ 비용.

2. **K-hop locality**: $T_k(L)$ 은 $k$-hop neighbor 까지만 영향 → Spatial locality 자동 보장 (이는 정리 5.3 의 polynomial filter 정리에서 직접 따름).

3. **Transferable parameter**: $\theta_k \in \mathbb{R}^K$ 가 graph 와 독립 → 새 graph 에 적용 가능 (단, $\lambda_{\max}$ scale 정규화 필요).

ChebNet → GCN ($K=1$ 단순화) 의 경로는 Ch2-02, Ch2-03 에서 자세히. 따라서 spectral 의 이론적 우아함을 polynomial 근사로 실용화한 것이 ChebNet/GCN 의 핵심 기여.

</details>

---

<div align="center">

[◀ 이전](./04-spectral-theory.md) | [📚 README](../README.md) | [다음 ▶](./06-random-walk-pagerank.md)

</div>
