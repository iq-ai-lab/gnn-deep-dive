# 02. ChebNet — Chebyshev Polynomial 근사 (Defferrard 2016)

## 🎯 핵심 질문

- Chebyshev polynomial 이란 무엇이며 왜 polynomial approximation 의 "최적" 선택인가?
- $g_\theta(L) = \sum_{k=0}^K \theta_k T_k(\tilde{L})$ 형태로 spectral filter 를 근사하는 이유는?
- $\tilde{L} = 2L/\lambda_{\max} - I$ 의 rescaling은 왜 필요한가?
- ChebNet 이 어떻게 $K$-hop localized 임을 증명할 수 있는가?
- Recurrence relation $T_{k+1}(\tilde L) = 2\tilde L T_k(\tilde L) - T_{k-1}(\tilde L)$ 가 어떻게 $O(mK)$ 효율로 이어지는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Bruna 2014 의 spectral GCN (Ch2-01) 은 세 가지 한계를 가집니다:
1. $O(n^3)$ eigendecomposition 비용
2. Non-localized filter
3. Non-transferable parameters

ChebNet (Defferrard 2016) 은 **Chebyshev polynomial 로 spectral filter 를 parameterize** 함으로써 세 한계를 모두 해결:
- $U$ 없이 작동 → eigendecomposition 불필요
- $K$-hop localized (정리 1.3 by polynomial)
- Polynomial coefficient $\theta_k$ 가 graph-agnostic → inductive

또한 ChebNet 은 GCN (Kipf-Welling 2017) 의 직접적 전신 — GCN 은 ChebNet 의 $K=1$ 단순화 버전. 따라서 ChebNet 이해는 **GCN 유도의 사전조건**.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-spectral-convolution.md](./01-spectral-convolution.md) — Bruna spectral conv
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Polynomial of matrix, recurrence
- 수치해석: Polynomial approximation, Chebyshev nodes

---

## 📖 직관적 이해

### Chebyshev Polynomial 의 직관

Chebyshev polynomial of the first kind $T_k(x)$ on $[-1, 1]$:
$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)
$$

또는 trigonometric form: $T_k(\cos \theta) = \cos(k \theta)$.

**핵심 성질**:
1. **Equioscillation**: $T_k$ 는 $[-1, 1]$ 에서 $k+1$ 개 극점, 모두 절댓값 1
2. **최적 uniform approximation**: 임의 continuous function 을 polynomial 로 근사할 때 max error 최소화
3. **Recurrence**: $T_{k+1} = 2 x T_k - T_{k-1}$ — 2-step 만으로 계산 (Horner-like)

### Spectral Filter 의 Chebyshev 근사

Bruna spectral filter $\hat{g}(\lambda)$ (자유 함수) 를 Chebyshev polynomial sum 으로 근사:
$$
\hat{g}_\theta(\lambda) = \sum_{k=0}^K \theta_k T_k(\tilde\lambda), \quad \tilde\lambda = \frac{2\lambda}{\lambda_{\max}} - 1
$$

(rescaling 으로 $\tilde\lambda \in [-1, 1]$ — Chebyshev 영역)

**파라미터 수**: $K+1$ (작음, 보통 $K = 2 \sim 5$).

### $L$ 영역에서 직접 계산

정리 1.1 (Ch2-01): $\hat{g}(L) = U \hat{g}(\Lambda) U^T$. Polynomial 의 경우:
$$
g_\theta(L) = \sum_k \theta_k T_k(\tilde L), \quad \tilde L = \frac{2L}{\lambda_{\max}} - I
$$

$U$ 없이 $\tilde L^k$ 를 직접 sparse matrix 곱으로 계산 → $O(m K)$ 비용.

### K-Hop Locality

$T_k(\tilde L) x$ 의 sparsity pattern: $\tilde L^k$ 의 nonzero entry = $k$-hop 까지 reachable 노드. 따라서 ChebNet layer 한 번 = $K$-hop neighborhood aggregation.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Chebyshev Polynomial of the First Kind

$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2 x T_{k-1}(x) - T_{k-2}(x) \quad (k \geq 2)
$$

도메인 $[-1, 1]$ 에서 정의 (확장 가능). 다음 trigonometric form 과 동치:
$$
T_k(\cos \theta) = \cos(k \theta)
$$

### 정의 2.2 — Rescaled Laplacian

Spectral GCN 에서 $L$ 또는 $L_{\text{sym}}$ 의 고유값이 $[0, \lambda_{\max}]$ 에 있음 ($\lambda_{\max} \leq 2$). Chebyshev domain $[-1, 1]$ 으로 rescale:
$$
\tilde L = \frac{2 L}{\lambda_{\max}} - I
$$

새로운 고유값 $\tilde\lambda \in [-1, 1]$.

### 정의 2.3 — ChebNet Spectral Filter

$$
\hat{g}_\theta(\tilde \lambda) = \sum_{k=0}^K \theta_k T_k(\tilde \lambda)
$$

학습 파라미터 $\theta = (\theta_0, \theta_1, \ldots, \theta_K) \in \mathbb{R}^{K+1}$.

### 정의 2.4 — ChebNet Layer

Multi-channel input $X \in \mathbb{R}^{n \times d_{\text{in}}}$, output $H \in \mathbb{R}^{n \times d_{\text{out}}}$:
$$
H = \sigma \left( \sum_{k=0}^K T_k(\tilde L) X \Theta_k \right)
$$

각 $\Theta_k \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ — input/output 채널 mixing weight matrix.

전체 파라미터: $(K+1) \cdot d_{\text{in}} \cdot d_{\text{out}}$.

### 정의 2.5 — Recurrence Computation

$T_k(\tilde L) X$ 를 한 번에 계산하지 않고 sequential:
$$
T^{(0)} = X, \quad T^{(1)} = \tilde L X, \quad T^{(k)} = 2 \tilde L T^{(k-1)} - T^{(k-2)}
$$

각 step 은 sparse matrix-vector 곱 → $O(m d)$ per step, total $O(K m d)$.

---

## 🔬 정리와 증명

### 정리 2.1 — Chebyshev Polynomial Recurrence

$$
T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)
$$

**증명** (trigonometric form): $T_k(\cos\theta) = \cos(k\theta)$ 가정. Sum-to-product:
$$
\cos((k+1)\theta) + \cos((k-1)\theta) = 2 \cos\theta \cos(k\theta)
$$
$$
\Rightarrow \cos((k+1)\theta) = 2\cos\theta \cos(k\theta) - \cos((k-1)\theta)
$$
$$
\Rightarrow T_{k+1}(\cos\theta) = 2\cos\theta \cdot T_k(\cos\theta) - T_{k-1}(\cos\theta)
$$
$\square$

### 정리 2.2 — ChebNet Filter 의 K-Hop Locality

**Theorem**: $g_\theta(\tilde L) X$ 의 노드 $i$ 출력값은 $i$ 의 $K$-hop neighborhood 노드의 입력값에만 의존.

**증명**: $T_k(\tilde L)$ 은 $\tilde L$ 의 polynomial of degree $k$. $\tilde L = (2/\lambda_{\max}) L - I = (2/\lambda_{\max})(D - A) - I$ — sparsity pattern 이 $L$ 과 같음 (1-hop neighbor 만 nonzero except diagonal).

따라서 $\tilde L^k$ 의 $(i, j)$ 가 nonzero ⟹ $j$ 가 $i$ 의 $\leq k$-hop neighbor.

$T_k(\tilde L)$ 도 $\tilde L^j$ ($j \leq k$) 의 선형결합 → 같은 sparsity. 따라서 $g_\theta(\tilde L) = \sum_{k=0}^K \theta_k T_k(\tilde L)$ 의 nonzero 영역 = $K$-hop neighborhood. $\square$

### 정리 2.3 — Computational Cost

**Theorem**: ChebNet layer forward pass cost $= O(K m d_{\text{in}} d_{\text{out}})$.

**증명**: Recurrence 로 $T^{(k)} = 2\tilde L T^{(k-1)} - T^{(k-2)}$ 를 $K$ 번 — 각 step sparse matvec $O(m d_{\text{in}})$. Total $K$ matvec = $O(K m d_{\text{in}})$.

각 $T^{(k)} \Theta_k$ 의 dense matmul: $O(n d_{\text{in}} d_{\text{out}})$. 합산 $O(K n d_{\text{in}} d_{\text{out}}) = O(K m d_{\text{in}} d_{\text{out}})$ (sparse graph $m \sim n$). $\square$

비교: Bruna spectral conv 는 $O(n^2 d_{\text{in}} d_{\text{out}})$ + $O(n^3)$ decomposition 일회. ChebNet 은 $K \ll n$ 에서 훨씬 효율.

### 정리 2.4 — Inductive Transferability

ChebNet 의 학습 파라미터 $\theta_k$ 는 graph-agnostic (스칼라 polynomial coefficient).

**Theorem**: $\theta$ 가 graph $G_1$ 에서 학습되었을 때, 다른 graph $G_2$ ($\lambda_{\max}$ 다를 수 있음) 에 적용 가능. 단, $\tilde L_{G_2}$ 로 다시 계산.

(증명: Definition 직접 — 파라미터가 $\Lambda$ 자체에 의존하지 않음. 단, $\lambda_{\max}$ 의 estimation 정확도가 영향)

이는 inductive learning 의 기반 — 새 graph 에 사전훈련된 ChebNet 적용 가능.

### 정리 2.5 — Chebyshev 의 Optimal Approximation

**Theorem (Equioscillation, Chebyshev)**: $f \in C([-1, 1])$ 의 best uniform approximation by polynomial of degree $K$ 는 unique 하고 $f - p_K^*$ 가 $K+2$ 개 alternating extrema 를 가짐.

(이는 numerical analysis 의 표준 결과; Trefethen "Approximation Theory and Approximation Practice" 참조)

**ChebNet에의 의미**: $\hat{g}(\lambda)$ 가 임의 함수일 때, Chebyshev basis 가 uniform approximation 의미에서 **optimal** — 같은 $K$ 에 대해 maximum error 가 최소.

---

## 💻 NumPy 구현 검증

### 실험 1 — Chebyshev Recurrence 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 200)

def chebyshev_recurrence(x, K):
    T = [np.ones_like(x), x.copy()]
    for k in range(2, K + 1):
        T.append(2 * x * T[-1] - T[-2])
    return T

T_list = chebyshev_recurrence(x, 5)
fig, ax = plt.subplots(figsize=(8, 5))
for k, T in enumerate(T_list):
    ax.plot(x, T, label=f'$T_{k}$')
ax.set_title('Chebyshev polynomials of the first kind')
ax.legend(); ax.grid()
plt.show()
```

### 실험 2 — ChebNet Filter Recurrence on Graph

```python
import networkx as nx

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(1)
D_inv_sqrt = np.diag(1/np.sqrt(deg))
L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

# Rescaled Laplacian
lambda_max = np.linalg.eigvalsh(L_sym).max()
print(f'λ_max = {lambda_max:.4f}')
L_tilde = (2/lambda_max) * L_sym - np.eye(n)

# Chebyshev recurrence on graph
def cheb_recurrence(L_tilde, X, K):
    T0 = X.copy()
    out = [T0]
    if K >= 1:
        T1 = L_tilde @ X
        out.append(T1)
    for k in range(2, K + 1):
        T_new = 2 * L_tilde @ out[-1] - out[-2]
        out.append(T_new)
    return out

# Delta input at node 0
x = np.zeros(n); x[0] = 1.0
T_x = cheb_recurrence(L_tilde, x, K=4)

print('K-hop coverage of T_k(L̃) x:')
for k, T_k_x in enumerate(T_x):
    nonzero = np.where(np.abs(T_k_x) > 1e-10)[0]
    max_hop = max(nx.shortest_path_length(G, 0, j) for j in nonzero)
    print(f'  k={k}: {len(nonzero)} nodes nonzero, max_hop={max_hop}')
```

**예상**: $T_k$ 가 정확히 $k$-hop 까지만 영향.

### 실험 3 — ChebNet Layer 정의

```python
import torch
import torch.nn as nn

class ChebConvLayer(nn.Module):
    def __init__(self, d_in, d_out, K):
        super().__init__()
        self.K = K
        self.thetas = nn.ParameterList([
            nn.Parameter(torch.randn(d_in, d_out) / np.sqrt(d_in))
            for _ in range(K + 1)
        ])
    
    def forward(self, X, L_tilde):
        """X: [n, d_in], L_tilde: [n, n]"""
        T_prev = X
        out = T_prev @ self.thetas[0]
        if self.K >= 1:
            T_curr = L_tilde @ X
            out = out + T_curr @ self.thetas[1]
            for k in range(2, self.K + 1):
                T_new = 2 * L_tilde @ T_curr - T_prev
                T_prev, T_curr = T_curr, T_new
                out = out + T_curr @ self.thetas[k]
        return out

# 사용 예
n_, d_in, d_out, K_ = n, 16, 32, 3
X = torch.randn(n_, d_in)
L_tilde_t = torch.tensor(L_tilde, dtype=torch.float32)

layer = ChebConvLayer(d_in, d_out, K=K_)
H = layer(X, L_tilde_t)
print(f'ChebNet layer: input {X.shape} → output {H.shape}')
print(f'Parameters: {sum(p.numel() for p in layer.parameters())}')
print(f'Theory: (K+1)*d_in*d_out = {(K_+1)*d_in*d_out}')
```

### 실험 4 — ChebNet vs Bruna Parameter Count

```python
def cheb_params(K, d_in, d_out):
    return (K + 1) * d_in * d_out

def bruna_params(n, d_in, d_out):
    return n * d_in * d_out

ns = [100, 1000, 10_000, 100_000]
K = 3
d_in = d_out = 32

print(f'{"n":>10} {"Bruna":>15} {"ChebNet":>15} {"ratio":>10}')
for nv in ns:
    b = bruna_params(nv, d_in, d_out)
    c = cheb_params(K, d_in, d_out)
    print(f'{nv:>10} {b:>15,} {c:>15,} {b/c:>10.1f}x')
```

### 실험 5 — Chebyshev Polynomial 의 Function Approximation 정확도

```python
# Approximate target function (Heat kernel-like)
def target(lam, t=2.0):
    return np.exp(-lam * t)

# Chebyshev coefficients via discrete cosine sampling
def chebyshev_coeffs(f, K):
    nodes = np.cos(np.pi * (np.arange(K + 1) + 0.5) / (K + 1))
    f_vals = f(nodes)
    coeffs = np.zeros(K + 1)
    for k in range(K + 1):
        coeffs[k] = (2 / (K + 1)) * (f_vals * np.cos(np.pi * k * (np.arange(K + 1) + 0.5) / (K + 1))).sum()
    coeffs[0] /= 2
    return coeffs

# 주의: 이 target function 은 [0, 2] 영역. Chebyshev 는 [-1, 1] → rescale needed
def heat_chebyshev_approx(lam, K, lambda_max=2.0, t=2.0):
    lam_t = 2 * lam / lambda_max - 1
    f = lambda x: np.exp(-(x + 1) * lambda_max / 2 * t)
    coeffs = chebyshev_coeffs(f, K)
    Ts = chebyshev_recurrence(lam_t, K)
    return sum(c * T for c, T in zip(coeffs, Ts))

lams = np.linspace(0, 2, 100)
target_vals = target(lams, t=2)

for K in [2, 5, 10]:
    approx = heat_chebyshev_approx(lams, K, lambda_max=2.0, t=2.0)
    plt.plot(lams, approx, label=f'K={K}')

plt.plot(lams, target_vals, 'k--', label='target $e^{-2\\lambda}$')
plt.xlabel('λ'); plt.ylabel('filter response')
plt.legend(); plt.title('Chebyshev approximation of heat kernel')
plt.show()
```

---

## 🔗 실전 활용

### 1. ChebNet 의 실전 위치

ChebNet 자체는 Cora·Citeseer 등 작은 그래프에서 GCN 보다 약간 나은 성능을 보이지만, GCN 의 단순성 (K=1) 이 압도적 인기 → ChebNet 은 "이론적 stepping stone" 으로 자리매김.

### 2. MoNet (Monti 2017)

ChebNet 의 generalization. Mesh, point cloud 등 비유클리드 데이터에서 polynomial filter 의 learnable kernel.

### 3. CayleyNet (Levie 2017)

Chebyshev 대신 Cayley polynomial 사용 — 더 narrow band-pass 가능. Rational filter 의 일반화.

### 4. Deep ChebNet 과 Over-smoothing

Layer 를 깊게 쌓아도 $K$-hop locality 가 유지되어, GCN보다 over-smoothing 약간 완화 가능. 단, 기본적으로 over-smoothing 문제는 polynomial 형태 자체에서 옴 (Ch5).

### 5. APPNP 와의 관계

APPNP (Ch5-05) 는 polynomial 대신 **rational filter** $(1 - (1-\alpha)\hat A)^{-1}$ 사용. ChebNet 의 polynomial 와 대비되는 또 하나의 spectral 일반화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\lambda_{\max}$ 추정 가능 | Power iteration 으로 추정, error 시 numerical 문제 |
| Polynomial 표현력 (degree $K$) | 매우 narrow filter 는 high $K$ 필요 → cost 증가 |
| Chebyshev basis 가 모든 함수에 좋음 | 특정 function family 에는 다른 basis 더 효율 (CayleyNet) |
| 모든 layer 에서 fixed $K$ | Adaptive $K$ 는 미해결 |
| Self-loop 미포함 | GCN 의 $\tilde A = A + I$ 와 차이 |

---

## 📌 핵심 정리

$$\boxed{g_\theta(\tilde L) X = \sum_{k=0}^K \theta_k T_k(\tilde L) X \quad — \text{ChebNet filter}}$$

$$\boxed{T_{k+1}(\tilde L) = 2 \tilde L T_k(\tilde L) - T_{k-1}(\tilde L) \quad — \text{recurrence}}$$

| 항목 | ChebNet (Defferrard 2016) |
|------|--------------------------|
| **Filter family** | Chebyshev polynomial of degree $K$ |
| **Parameter count** | $(K+1) \cdot d_{\text{in}} \cdot d_{\text{out}}$ |
| **Locality** | $K$-hop guaranteed |
| **Eigendecomposition** | 불필요 ($\lambda_{\max}$ 만 추정) |
| **Computation** | $O(K m d_{\text{in}} d_{\text{out}})$ |
| **Inductive** | ✓ |
| **GCN과의 관계** | $K=1$ 단순화 → GCN |
| **현대 사용** | 이론·실증 baseline, GCN 으로 대체된 경우 多 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T_2(x) = 2x^2 - 1$, $T_3(x) = 4x^3 - 3x$ 임을 recurrence 로 유도하라.

<details>
<summary>해설</summary>

$T_0 = 1$, $T_1 = x$.
$T_2 = 2 x \cdot T_1 - T_0 = 2x \cdot x - 1 = 2x^2 - 1$ ✓
$T_3 = 2x \cdot T_2 - T_1 = 2x(2x^2 - 1) - x = 4x^3 - 2x - x = 4x^3 - 3x$ ✓ $\square$

</details>

**문제 2** (심화): $K=2$ ChebNet layer 의 explicit 계산식을 $L_{\text{sym}}$ 와 $\lambda_{\max}$ 로 작성하라. (Hint: $\tilde L = (2/\lambda_{\max}) L - I$)

<details>
<summary>해설</summary>

$T_0(\tilde L) = I$, $T_1(\tilde L) = \tilde L = (2/\lambda_{\max}) L - I$, $T_2(\tilde L) = 2 \tilde L^2 - I$.

$$
H = \theta_0 X + \theta_1 \tilde L X + \theta_2 T_2(\tilde L) X
$$

$\tilde L^2 = ((2/\lambda_{\max}) L - I)^2 = (4/\lambda_{\max}^2) L^2 - (4/\lambda_{\max}) L + I$.

$T_2 = 2 \tilde L^2 - I = (8/\lambda_{\max}^2) L^2 - (8/\lambda_{\max}) L + 2 I - I = (8/\lambda_{\max}^2) L^2 - (8/\lambda_{\max}) L + I$.

전체:
$$
H = (\theta_0 - \theta_1 + \theta_2) X + (\frac{2 \theta_1}{\lambda_{\max}} - \frac{8 \theta_2}{\lambda_{\max}}) L X + \frac{8 \theta_2}{\lambda_{\max}^2} L^2 X
$$

이는 $L^0, L^1, L^2$ 의 polynomial 형태로 재배열 가능 → polynomial filter of order 2. $\square$

</details>

**문제 3** (논문 비평): ChebNet 이 Bruna 의 spectral 한계를 모두 극복했음에도, GCN ($K=1$ 단순화) 이 더 인기 있는 이유는 무엇인가? GCN 이 잃은 표현력이 실전에서 문제가 되지 않는 이유를 분석하라.

<details>
<summary>해설</summary>

**GCN 이 더 인기인 이유**:

1. **단순성**: Hyperparameter $K$ 가 1로 고정 → 튜닝 불필요. 또한 1-hop 만 보므로 매 layer 의미가 명확.
2. **Renormalization trick**: $\tilde A = A + I$ + symmetric norm 이 $\lambda_{\max} < 2$ 보장 → numerical stability. ChebNet 은 $\lambda_{\max}$ 추정 필요.
3. **Stack 가능**: $K$-layer GCN ≈ $K$-hop ChebNet (대략, but not exactly equal). 깊이로 receptive field 확장 가능 — 단, over-smoothing.
4. **Empirical 성능**: Cora·Citeseer·Pubmed 에서 GCN 이 ChebNet 과 유사 또는 더 나은 성능. Hyperparameter 적은 GCN 의 robustness 가 우위.

**잃은 표현력이 문제 안 되는 이유**:

1. **Multi-layer = multi-hop**: 1-hop GCN 을 $L$ 번 쌓으면 effective receptive field $L$-hop. ChebNet $K=L$ 1-layer 와 representation power 비슷 (단, 정확히 동일 X).
2. **Real-world graph 의 작은 hop diameter**: 6-degree of separation — most graph 가 3~5 hop 안에 충분한 정보. 큰 $K$ polynomial 불필요.
3. **Non-linearity 의 효과**: Multi-layer GCN 은 layer 사이 ReLU 가 있어 단순 polynomial 보다 표현력 큼 (Cybenko UAT 적용).
4. **Over-smoothing 관점**: 큰 $K$ 단일 layer ChebNet 은 over-smoothing 더 심함 — multi-layer 의 local smoothing 이 더 controllable.

**최근 연구**: ChebNet-style polynomial filter 가 "specific spectrum 학습" 이 필요한 task (heterophilic graph, signal classification) 에서 다시 부각 — JacobiConv (Wang 2022), ChebNetII (He 2022). 따라서 polynomial vs 1-hop 의 trade-off 는 task-dependent — Ch2-04, Ch7 에서 통합 관점.

</details>

---

<div align="center">

[◀ 이전](./01-spectral-convolution.md) | [📚 README](../README.md) | [다음 ▶](./03-gcn-derivation.md)

</div>
