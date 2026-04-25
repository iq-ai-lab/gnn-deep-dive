# 02. Unnormalized Laplacian과 그 성질

## 🎯 핵심 질문

- Graph Laplacian $L = D - A$ 는 어떻게 정의되고, 연속 Laplacian $-\Delta$ 와 무엇이 같은가?
- 왜 $L$ 은 항상 PSD (positive semi-definite) 인가? 이 성질이 GNN에 어떤 의미를 주는가?
- $\dim \ker(L)$ 가 connected component 수와 같다는 사실의 의미는?
- $L = B B^T$ (incidence factorization) 가 가르쳐주는 것은 무엇인가?
- Quadratic form $x^T L x$ 가 왜 "smoothness 측정자"인가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Graph Laplacian은 GNN의 거의 모든 이론적 결과의 출발점입니다:

1. **Spectral GCN의 기반** — Bruna 2014, ChebNet, GCN 모두 $L$ 의 고유분해를 활용
2. **Smoothness regularization** — Manifold learning, label propagation 의 $\frac{1}{2} \sum (f_i - f_j)^2 = f^T L f$ 페널티
3. **Spectral clustering** — $\ker(L)$ 와 Fiedler vector로 community detection (Ch1-04)
4. **Over-smoothing 분석** — 깊은 GCN의 feature collapse를 $\ker(L)$ 로 설명 (Ch5-02)

이 문서에서는 Laplacian의 **정의와 핵심 성질 두 가지** ($L \succeq 0$, $\dim \ker(L) = $ # components) 를 엄밀히 증명합니다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Symmetric matrix, PSD, kernel/null space, rank-nullity
- 이전 문서: [01-graph-basics.md](./01-graph-basics.md) — adjacency, degree, incidence
- 미적분: 연속 Laplacian $\Delta f = \sum_i \partial^2 f / \partial x_i^2$ (선택)

---

## 📖 직관적 이해

### 연속 Laplacian과의 유비

연속 함수 $f: \mathbb{R}^d \to \mathbb{R}$ 의 Laplacian:
$$
\Delta f(x) = \sum_{i=1}^d \frac{\partial^2 f}{\partial x_i^2}
$$

이는 평균-편차 측정: $\Delta f(x) \approx \frac{1}{|N(x)|} \sum_{y \in N(x)} (f(y) - f(x))$ (small neighborhood). 그래프에서 자연스러운 이산판:

$$
(L f)_i = \sum_{j \in N(i)} (f_i - f_j) = d_i f_i - \sum_j A_{ij} f_j = ((D - A) f)_i
$$

**부호 주의**: 그래프 Laplacian $L = D - A$ 는 연속 $-\Delta$ 와 같은 부호 (PSD). 일부 문헌은 $A - D$ 를 쓰기도 함.

### Quadratic form: smoothness 측정

$$
x^T L x = \frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2
$$

이 식은 **각 엣지 양 끝의 값 차이의 제곱 합**. $x$ 가 인접 노드끼리 비슷한 값을 가질수록 (smooth) $x^T L x$ 는 작음. **GCN이 학습 후 노드 feature가 smooth해지는 것이 over-smoothing의 근원**.

### Kernel과 Connected Components

$L \mathbb{1} = (D - A) \mathbb{1}$ 의 $i$ 성분 = $d_i - d_i = 0$, 따라서 $\mathbb{1} \in \ker(L)$. Connected graph에서는 이것이 유일한 (1차원) kernel이지만, $k$ 개 component가 있으면 각 component의 indicator vector도 kernel에 속함 — $\dim \ker(L) = k$.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Unnormalized Graph Laplacian

Undirected graph $G$ 의 adjacency $A$, degree matrix $D$ 에 대해:
$$
L := D - A
$$

Symmetric, real-valued $n \times n$ matrix.

### 정의 2.2 — Quadratic Form

벡터 $x \in \mathbb{R}^n$ 에 대해:
$$
x^T L x = \sum_i d_i x_i^2 - \sum_{i,j} A_{ij} x_i x_j
$$

### 정의 2.3 — Positive Semi-Definite (PSD)

Symmetric matrix $M \in \mathbb{R}^{n \times n}$ 이 PSD ($M \succeq 0$) 이려면:
$$
\forall x \in \mathbb{R}^n: \quad x^T M x \geq 0
$$

동치로: 모든 고유값 $\lambda_i(M) \geq 0$.

---

## 🔬 정리와 증명

### 정리 2.1 — Laplacian Quadratic Form

$$
x^T L x = \frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2
$$

**증명**:

$$
\frac{1}{2} \sum_{i,j} A_{ij} (x_i - x_j)^2 = \frac{1}{2} \sum_{i,j} A_{ij} (x_i^2 - 2 x_i x_j + x_j^2)
$$

$$
= \frac{1}{2} \left( \sum_i x_i^2 \sum_j A_{ij} + \sum_j x_j^2 \sum_i A_{ij} - 2 \sum_{i,j} A_{ij} x_i x_j \right)
$$

Symmetric ($A = A^T$) 이므로 첫 두 항이 같음, $\sum_j A_{ij} = d_i$:
$$
= \sum_i d_i x_i^2 - \sum_{i,j} A_{ij} x_i x_j = x^T D x - x^T A x = x^T L x
$$

(undirected에서 $\sum_{(i,j) \in E}$ 는 ordered pair $\sum_{i,j}$ 의 절반, 따라서 $1/2$). $\square$

### 정리 2.2 — Laplacian PSD

$L \succeq 0$.

**증명**: 정리 2.1에서 $x^T L x = \frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2 \geq 0$ — 제곱의 합이 항상 비음. $\square$

**따름**: 모든 고유값 $\lambda_i(L) \geq 0$. 항상 $\lambda_1 = 0$ ($\mathbb{1}$ 이 eigenvector).

### 정리 2.3 — Kernel과 Connected Components

Undirected graph $G$ 가 $k$ 개의 connected component $C_1, \ldots, C_k$ 로 분해될 때:
$$
\dim \ker(L) = k
$$

**증명** (양방향):

**($\geq k$ 방향)**: 각 component $C_l$ 의 indicator vector $e^{(l)}$ ($e^{(l)}_i = 1$ if $i \in C_l$, else 0) 에 대해:
$$
(L e^{(l)})_i = d_i e^{(l)}_i - \sum_j A_{ij} e^{(l)}_j
$$

- $i \in C_l$: $e^{(l)}_i = 1$, $\sum_j A_{ij} e^{(l)}_j = \sum_{j \in C_l} A_{ij} = d_i$ (이웃 모두 $C_l$ 안에). $\Rightarrow d_i - d_i = 0$.
- $i \notin C_l$: $e^{(l)}_i = 0$, 이웃 $j$ 도 $C_l$ 이 아님 (component 정의), 따라서 $A_{ij} e^{(l)}_j = 0$. $\Rightarrow 0$.

따라서 $L e^{(l)} = 0$, 즉 $e^{(l)} \in \ker(L)$. $\{e^{(1)}, \ldots, e^{(k)}\}$ 는 disjoint support로 linearly independent → $\dim \ker(L) \geq k$.

**($\leq k$ 방향)**: $x \in \ker(L)$ 이라 가정. $x^T L x = 0$, 정리 2.1에서:
$$
\frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2 = 0
$$

각 항이 비음이므로 모든 $(i, j) \in E$ 에 대해 $x_i = x_j$. 따라서 $x$ 는 각 connected component 내에서 상수. Component마다 1개 자유도, 총 $k$ 개. $\dim \ker(L) \leq k$.

종합: $\dim \ker(L) = k$. $\square$

### 정리 2.4 — Incidence Factorization

$$
L = B B^T
$$

(이미 Ch1-01 정리 1.2 에서 증명. 여기서는 **PSD의 또 다른 증명**으로 활용)

**대안 PSD 증명**: $x^T L x = x^T B B^T x = \|B^T x\|^2 \geq 0$. $\square$

또한 $\text{rank}(L) = \text{rank}(B) = n - k$ ($k$ = #components). 이로부터 $\dim \ker(L) = n - \text{rank}(L) = k$ 가 직접 따라옴.

### 정리 2.5 — 고유값 정렬

$L$ 의 고유값을 $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$ 으로 정렬하면:
- $\lambda_1 = 0$ (always, $\mathbb{1}$ eigenvector)
- $\lambda_2 = 0$ iff disconnected
- $\lambda_2 > 0 \Leftrightarrow$ connected, **algebraic connectivity** (Ch1-04 Fiedler value)

### 정리 2.6 — Trace와 엣지 수

$$
\text{tr}(L) = \sum_i d_i = 2m
$$

**증명**: $\text{tr}(L) = \text{tr}(D) - \text{tr}(A) = \sum_i d_i - 0 = 2m$ (handshake lemma). $\square$

**따름**: $\sum_i \lambda_i = 2m$ — 평균 고유값 $= 2m/n = \bar{d}$.

---

## 💻 NumPy 구현 검증

### 실험 1 — Laplacian 구성과 PSD 확인

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 작은 그래프 (Ch1-01 동일)
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
n = 4
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(edges)

A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(axis=1)
D = np.diag(deg)
L = D - A
print('Laplacian L:')
print(L)

# 고유분해
eigvals, eigvecs = np.linalg.eigh(L)
print(f'Eigenvalues: {eigvals.round(4)}')
print(f'All non-negative? {np.all(eigvals >= -1e-10)}')   # PSD
print(f'λ_1 = {eigvals[0]:.6f}  (≈ 0)')
print(f'λ_2 = {eigvals[1]:.4f}  (Fiedler value > 0 if connected)')
```

**출력**:
```
Eigenvalues: [-0.  2.  2.  4.]
All non-negative? True
λ_1 = 0.000000  (≈ 0)
λ_2 = 2.0000  (Fiedler value > 0 if connected)
```

### 실험 2 — Quadratic Form 검증

```python
def quadratic_form(L, x):
    return float(x.T @ L @ x)

def edge_sum_sq(edges, x):
    return 0.5 * sum((x[i] - x[j])**2 for i, j in edges) * 2   # both directions

x = np.array([1.0, 2.0, 3.0, 4.0])
qf = quadratic_form(L, x)
es = sum((x[i] - x[j])**2 for i, j in edges)
print(f'x^T L x = {qf}')
print(f'Σ (x_i - x_j)^2 = {es}')
print(f'Equal? {np.isclose(qf, es)}')   # True (정리 2.1)
```

### 실험 3 — Disconnected Graph의 Kernel

```python
# 두 component: {0,1,2}와 {3,4}
G2 = nx.Graph([(0,1), (1,2), (0,2), (3,4)])
n2 = 5
A2 = nx.adjacency_matrix(G2, nodelist=range(n2)).toarray().astype(float)
L2 = np.diag(A2.sum(1)) - A2

eigvals2, eigvecs2 = np.linalg.eigh(L2)
print(f'Eigenvalues: {eigvals2.round(4)}')
zero_count = np.sum(np.abs(eigvals2) < 1e-8)
print(f'Zero eigenvalues: {zero_count}  (= # components = 2)')

# Indicator vectors가 kernel에 있는지 확인
e1 = np.array([1, 1, 1, 0, 0])
e2 = np.array([0, 0, 0, 1, 1])
print(f'L e1 = {L2 @ e1}  (should be 0)')
print(f'L e2 = {L2 @ e2}  (should be 0)')
```

### 실험 4 — Smoothness와 Laplacian Energy

```python
G3 = nx.path_graph(10)
A3 = nx.adjacency_matrix(G3).toarray().astype(float)
L3 = np.diag(A3.sum(1)) - A3

# Smooth signal (linear)
x_smooth = np.linspace(0, 1, 10)
# Oscillatory signal
x_osc = np.array([(-1)**i for i in range(10)], dtype=float)
# Random
x_rand = np.random.randn(10)

print(f'Smooth   x^T L x = {x_smooth @ L3 @ x_smooth:.4f}')   # 작음
print(f'Osc.     x^T L x = {x_osc @ L3 @ x_osc:.4f}')         # 큼
print(f'Random   x^T L x = {x_rand @ L3 @ x_rand:.4f}')

# 시각화
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(x_smooth, 'o-', label=f'smooth (E={x_smooth@L3@x_smooth:.2f})')
ax.plot(x_osc, 's-', label=f'oscillatory (E={x_osc@L3@x_osc:.2f})')
ax.set_title('Laplacian energy x^T L x measures (non-)smoothness')
ax.legend()
plt.show()
```

### 실험 5 — Trace 검증

```python
print(f'tr(L) = {np.trace(L)}, sum of degrees = {deg.sum()}, 2m = {2 * G.number_of_edges()}')
# 모두 같음 (8)
print(f'Sum of eigenvalues = {eigvals.sum():.4f}')   # = tr(L)
```

---

## 🔗 실전 활용

### 1. Laplacian Regularization

준지도 학습에서 label propagation:
$$
\min_f \frac{1}{2} \sum_{i \in L} (f_i - y_i)^2 + \lambda f^T L f
$$

(라벨된 노드는 정확하게, 라벨 없는 노드는 smooth) — closed-form $f = (\Lambda + \lambda L)^{-1} \Lambda y$.

### 2. Spectral Clustering의 출발점

$\dim \ker(L) = $ # components 이므로, 두 cluster를 분리하려면 $\lambda_2$ 의 eigenvector (Fiedler vector) 사용 — Ch1-04에서 자세히.

### 3. GCN의 forward에서 보이지 않는 $L$

GCN propagation matrix $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} = I - L_{\text{sym}}^{(\text{aug})}$ 형태 — Ch2-03에서 유도. $L$ 의 고유분해가 GCN의 표현력을 결정.

### 4. Mesh / Point Cloud Laplacian

3D 메시에서는 cotangent Laplacian (geometric weight) 가 더 정확. 이는 discrete differential geometry 의 표준.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Symmetric $A$ (undirected) | Directed graph에서 $L$ 은 비대칭 → magnetic Laplacian, Hermitian Laplacian 등 일반화 |
| Real-valued | Complex-valued signal에는 Hermitian 확장 |
| Static graph | Temporal Laplacian, edge time stamp 필요 |
| Unweighted 시 $\lambda_{\max} \leq 2 \Delta$ ($\Delta$ = max degree) | Weighted edge에서 bound 다름 |
| Connected 시 $\mathbb{1}$ 만이 ker | Disconnected 시 component-wise 분석 필요 |
| Self-loop 없는 simple graph | Self-loop 시 $L$ 의 $\ker$ 변화 (GCN의 $\tilde{A}$ 와 직결) |

---

## 📌 핵심 정리

$$\boxed{L = D - A = B B^T \succeq 0}$$

$$\boxed{x^T L x = \frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2 \quad — \text{smoothness 측정}}$$

$$\boxed{\dim \ker(L) = \text{# connected components}}$$

| 성질 | 진술 |
|------|------|
| **Symmetric** | $L = L^T$ |
| **PSD** | $\lambda_i(L) \geq 0$ |
| **$\lambda_1 = 0$** | $\mathbb{1}$ 이 eigenvector |
| **$\dim \ker(L) = k$** | $k$ = #components, indicator vector basis |
| **Trace** | $\text{tr}(L) = 2m$ |
| **Factorization** | $L = B B^T$ (incidence) |
| **Quadratic form** | $x^T L x = \frac{1}{2} \sum_E (x_i - x_j)^2$ |
| **Connectedness** | $\lambda_2 > 0 \Leftrightarrow$ connected |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K_n$ (complete graph, $n$ 개 노드 + 모든 엣지) 의 Laplacian 고유값을 구하라.

<details>
<summary>해설</summary>

$K_n$ 에서 $A = J - I$ ($J$ = all-ones), $D = (n-1) I$. 따라서:
$$
L = (n-1) I - (J - I) = n I - J
$$

$J$ 의 고유값: $\lambda(J) = n$ ($\mathbb{1}$, multiplicity 1) 와 $0$ (multiplicity $n-1$).

$L = nI - J$ 의 고유값: $n - n = 0$ ($\mathbb{1}$) 과 $n - 0 = n$ (multiplicity $n-1$).

따라서 $K_n$ 의 spectrum: $\{0, n, n, \ldots, n\}$. $\square$

**의미**: 모든 nontrivial 고유값이 $n$ — 매우 큰 spectral gap, 빠른 random walk mixing.

</details>

**문제 2** (심화): $G$ 가 $d$-regular 라면 $L = dI - A$ 임을 보이고, $A$ 와 $L$ 의 고유값 관계 $\lambda_i(L) = d - \lambda_{n+1-i}(A)$ 를 증명하라.

<details>
<summary>해설</summary>

$d$-regular: $D = d I$. $L = D - A = dI - A$. ✓

고유값 관계: $A v = \mu v$ 이면 $L v = (dI - A) v = (d - \mu) v$. 따라서:
- $A$ 의 고유값 $\mu_1 \geq \mu_2 \geq \cdots \geq \mu_n$
- $L$ 의 고유값 $\lambda_i = d - \mu_{n+1-i}$ (역순)

특히 $\mu_1 = d$ ($\mathbb{1}$ eigenvector) 이면 $\lambda_1 = 0$. 일반 (non-regular) graph에서는 이 단순 관계가 성립하지 않음. $\square$

</details>

**문제 3** (논문 비평): GCN의 propagation rule이 $L$ 이 아니라 $\tilde{A}$ ($= A + I$) 를 정규화한 것을 사용하는 이유를 추측하라. Self-loop 없는 $L$ 만 사용하면 어떤 문제가 있는가?

<details>
<summary>해설</summary>

**Self-loop의 동기**:

1. **수치 안정성**: Bipartite graph 또는 disconnected graph에서 $L$ 의 spectral radius $\lambda_{\max}(L)$ 는 정확히 2 (normalized) 까지 갈 수 있어 polynomial 근사가 불안정. $\tilde{A} = A + I$ 의 normalized 버전 $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 는 $\lambda_{\max} < 2$ 보장.

2. **Identity preservation**: $L = D - A$ 만 사용하면 노드 자신의 feature가 propagation에 포함 안 됨 (mean aggregator 형태로 보면). Self-loop 추가는 "노드 자신 + 이웃" 로 자연스러움.

3. **GCN propagation matrix의 의미**:
   - $L$ 만: $H' = \sigma((I - L) H W)$ 형태 → $\lambda \in [-1, 1]$, oscillation 위험
   - $\tilde{L}$: $H' = \sigma(\tilde{P} H W)$, $\tilde{P}$ 의 모든 고유값 $\in (-1, 1]$, stable (Ch2-03 renormalization trick)

4. **Empirically**: Kipf-Welling 2017 ablation에서 self-loop 없으면 정확도 5%+ 떨어짐.

따라서 self-loop는 단순한 trick이 아니라 spectral 안정성·정보 보존·실증 성능의 합리화. Ch2-03에서 자세히.

</details>

---

<div align="center">

[◀ 이전](./01-graph-basics.md) | [📚 README](../README.md) | [다음 ▶](./03-normalized-laplacian.md)

</div>
