# 04. Spectral Graph Theory — 고유값과 그래프 성질

## 🎯 핵심 질문

- 왜 $\lambda_1 = 0$ 이 항상 성립하고, $\lambda_2$ (Fiedler value) 가 connectivity의 지표인가?
- Cheeger's inequality $\frac{h^2}{2} \leq \lambda_2 \leq 2h$ 는 무엇을 말하고, 왜 spectral clustering이 작동하는가?
- Fiedler vector로 graph bisection이 가능한 이유와 Karate club 사례에서 실제 community와 일치하는 이유는?
- $\lambda_n$ (max eigenvalue) 는 어떤 그래프 성질을 인코딩하는가?
- Spectral gap $\lambda_2$ vs $\lambda_n - \lambda_2$ 가 random walk mixing time 과 어떻게 연결되는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Graph Laplacian의 고유분해 $L = U \Lambda U^T$ 는 GNN 이론 전체를 떠받칩니다:

1. **Spectral GCN 의 기반** — Bruna 2014 의 spectral conv 가 직접 $U$ 를 사용 (Ch2-01)
2. **Over-smoothing 의 설명** — $P^L x \to \ker(L)$ 의 수렴 속도가 $\lambda_2$ 의 함수 (Ch5-02)
3. **Community detection** — Fiedler vector로 graph bisection
4. **Mixing time** — Random walk 가 stationary에 수렴하는 속도 ∝ $1 / (1 - \lambda_2)$
5. **Positional encoding** — LapPE (Dwivedi 2020) 가 Laplacian eigenvector를 그대로 PE로 사용 (Ch4-05)

이 문서에서는 **Fiedler value의 의미와 Cheeger's inequality** 를 통해 spectral graph theory의 핵심을 정리합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-unnormalized-laplacian.md](./02-unnormalized-laplacian.md), [03-normalized-laplacian.md](./03-normalized-laplacian.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Spectral theorem for symmetric matrices, Rayleigh quotient
- 변분 최적화: min-cut 문제, NP-hardness

---

## 📖 직관적 이해

### Fiedler Value: 연결성의 정량화

$\lambda_1 = 0$ 은 항상 성립 (eigenvector $\mathbb{1}$). **$\lambda_2$ 가 0** 이면 graph가 disconnected (Ch1-02 정리 2.3). $\lambda_2$ 가 작을수록 graph가 "거의 disconnect"에 가까움 — 두 community를 자르는 edge가 적음.

이를 정량화한 것이 **Cheeger's inequality**: $\lambda_2$ 와 graph conductance $h$ 사이의 양방향 부등식. **Spectral clustering 의 이론적 근거**.

### Fiedler Vector와 Graph Bisection

$\lambda_2$ 의 eigenvector $v_2$ (Fiedler vector). 부호로 노드를 두 그룹으로 분리: $V_+ = \{i : v_{2,i} > 0\}$, $V_- = \{i : v_{2,i} < 0\}$. 이 분리가 conductance를 거의 최소화 (Cheeger 보장).

**유명한 예**: Zachary's Karate Club — 실제 분열 직전 두 파벌이 Fiedler vector 부호와 정확히 일치 (Newman 2006).

### Spectral Gap과 Mixing Time

Random walk가 stationary $\pi$ 로 수렴하는 속도:
$$
\| \pi^{(t)} - \pi \|_{\text{TV}} \leq C \cdot |1 - \lambda_2|^t
$$

$\lambda_2$ 가 클수록 (1에 가까울수록) 빠른 mixing. **Expander graph** ($\lambda_2$ 큼) 는 random walk가 빠르게 균등 분포에 도달.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Fiedler Value (Algebraic Connectivity)

$$
\lambda_2 := \text{2nd smallest eigenvalue of } L
$$

(또는 $L_{\text{sym}}$. 같은 connectivity 정보, 다른 scale.)

### 정의 4.2 — Graph Cut

분할 $V = S \cup \bar{S}$ ($S \cap \bar{S} = \emptyset$) 에 대해:
$$
\text{cut}(S, \bar{S}) := \sum_{i \in S, j \in \bar{S}} A_{ij}
$$

(두 집합 사이를 잇는 엣지 수)

### 정의 4.3 — Graph Conductance

$$
h(G) := \min_{S: |S| \leq n/2} \frac{\text{cut}(S, \bar{S})}{\text{vol}(S)}, \quad \text{vol}(S) = \sum_{i \in S} d_i
$$

작은 conductance = 좋은 community 분할.

### 정의 4.4 — Fiedler Vector

$\lambda_2$ 에 대응하는 eigenvector $v_2$ (정규화: $\|v_2\| = 1$, $v_2 \perp \mathbb{1}$).

### 정의 4.5 — Spectral Embedding

$L_{\text{sym}} = U \Lambda U^T$ 에서 처음 $k$ 개 (작은) eigenvector $U_{[k]} \in \mathbb{R}^{n \times k}$. 각 노드 $i$ 의 spectral embedding $= U_{[k]}^T e_i$ — k-means 등 으로 클러스터링.

---

## 🔬 정리와 증명

### 정리 4.1 — Rayleigh Quotient 공식 ($\lambda_2$)

$\mathbb{1}$ 에 직교한 vector 중 quadratic form을 최소화:
$$
\lambda_2 = \min_{\substack{x \perp \mathbb{1} \\ x \neq 0}} \frac{x^T L x}{x^T x}
$$

(Courant-Fischer min-max 의 특수 케이스). 이를 통해 $\lambda_2$ 의 변분 의미가 명확.

### 정리 4.2 — Cheeger's Inequality

$$
\boxed{\frac{h^2(G)}{2} \leq \lambda_2(L_{\text{rw}}) \leq 2 h(G)}
$$

(또는 $L_{\text{sym}}$ 으로 같은 형태)

**해석**:
- 상한 $\lambda_2 \leq 2h$: $\lambda_2$ 는 conductance에 의해 제한됨 (분할 $S$ 에 대응하는 indicator vector를 test vector로 → upper bound).
- 하한 $h^2/2 \leq \lambda_2$: Fiedler vector를 sweep cut으로 변환하면 $\lambda_2$ 가 $h$ 의 quadratic 보장 — **spectral clustering의 approximation guarantee**.

**증명 idea (상한)**:

$S$ 에 대한 indicator vector $\mathbb{1}_S$. $x = \mathbb{1}_S - \frac{|S|}{n} \mathbb{1}$ (centered). $x \perp \mathbb{1}$.

$$
\frac{x^T L x}{x^T x} = \frac{\text{cut}(S, \bar{S})}{\frac{|S| |\bar{S}|}{n}}
$$

이를 $h$ 로 bound (fast computation 후): $\lambda_2 \leq 2h$. (정밀한 증명은 Chung 1997 §1.) $\square$

**증명 idea (하한)**: 더 어려움. Fiedler vector $v_2$ 를 정렬 후 sweep — 각 prefix $S_t = \{i : v_{2,i} \leq t\}$ 의 conductance 분석. Cauchy-Schwarz로 $h(G) \leq \sqrt{2 \lambda_2}$ → $\lambda_2 \geq h^2/2$.

### 정리 4.3 — Disconnected Iff $\lambda_2 = 0$

**Theorem**: Connected graph $G$ 는 $\lambda_2 > 0$ $\Leftrightarrow$ disconnected는 $\lambda_2 = 0$.

**증명**: Ch1-02 정리 2.3: $\dim \ker(L) = $ # components. Connected ⟹ $\dim \ker = 1$ ⟹ $\lambda_2 > 0$. Disconnected ⟹ $\dim \ker \geq 2$ ⟹ $\lambda_2 = 0$. $\square$

### 정리 4.4 — Karate Club Fiedler Bisection

**Empirical Theorem (Newman 2006)**: Zachary's Karate Club ($n = 34$) 의 Fiedler vector 부호 분리는 실제 club 분열 (Mr. Hi vs Officer) 와 1개 노드 (node 8 또는 9) 외에는 정확히 일치.

(이 증명은 실증; 이론적으로는 Cheeger guarantee로 conductance가 거의 최적임이 보장)

### 정리 4.5 — Cycle 의 Spectrum

$n$-cycle $C_n$ 의 $L$ 고유값: $\lambda_k = 2 - 2\cos(2\pi k / n)$ for $k = 0, 1, \ldots, n-1$.

**증명** sketch: $C_n$ 의 $A$ 가 circulant matrix. Circulant의 eigenvector는 DFT basis $v_k = (1, \omega^k, \omega^{2k}, \ldots) / \sqrt n$ ($\omega = e^{2\pi i/n}$). $A$ 의 고유값 $2\cos(2\pi k/n)$. $L = 2I - A$ ($d=2$ regular) → $\lambda(L) = 2 - 2\cos(2\pi k/n)$. $\square$

이는 ChebNet과 GCN 에서 사용되는 polynomial filter의 spectral 분포를 직관적으로 보여줌.

### 정리 4.6 — Random Walk Mixing Time

**Theorem**: Random walk transition $P = D^{-1} A$ 가 stationary $\pi$ 로 수렴하는 속도:
$$
\| \pi^{(t)} - \pi \|_{\text{TV}} \leq \frac{1}{2 \min_i \pi_i^{1/2}} \cdot \mu_*^t
$$

여기서 $\mu_* = \max(|\mu_2|, |\mu_n|)$, $\mu_i$ = $P$ 의 고유값. **Mixing time** $\tau_{\text{mix}} = O(\log(1/\epsilon) / (1 - \mu_*))$.

(증명: spectral decomposition + bounding TV by $L^2$)

---

## 💻 NumPy 구현 검증

### 실험 1 — Karate Club Fiedler Bisection

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

eigvals, eigvecs = np.linalg.eigh(L_sym)
print(f'λ_1 = {eigvals[0]:.6f},  λ_2 = {eigvals[1]:.6f}  (Fiedler value)')

fiedler = eigvecs[:, 1]
labels_pred = (fiedler > 0).astype(int)

# 실제 club 라벨
labels_true = np.array([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=int)

# 일치율 (label flip 가능성 고려)
acc = max(
    (labels_pred == labels_true).mean(),
    (labels_pred != labels_true).mean(),
)
print(f'Fiedler bisection accuracy vs true club: {acc:.2%}')

# 시각화
pos = nx.spring_layout(G, seed=42)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
nx.draw(G, pos, ax=axes[0], node_color=labels_pred, cmap='RdBu', with_labels=True)
axes[0].set_title('Fiedler bisection')
nx.draw(G, pos, ax=axes[1], node_color=labels_true, cmap='RdBu', with_labels=True)
axes[1].set_title('True clubs')
plt.show()
```

**예상**: 정확도 ~97% (1~2개 노드 차이).

### 실험 2 — Cheeger Bound 수치 확인

```python
def conductance(A, S):
    Sb = list(set(range(len(A))) - set(S))
    cut = A[np.ix_(S, Sb)].sum()
    vol_S = A[S].sum()
    vol_Sb = A[Sb].sum()
    return cut / min(vol_S, vol_Sb)

# Fiedler-based sweep cut
order = np.argsort(fiedler)
best_h = float('inf')
for t in range(1, n):
    S = order[:t].tolist()
    h_t = conductance(A, S)
    if h_t < best_h:
        best_h = h_t

print(f'Best sweep conductance h ≈ {best_h:.4f}')
print(f'λ_2 / 2 (Cheeger lower bound on h) = {eigvals[1] / 2:.4f}')   # 안 맞음, h ≥ √(λ_2/2)
print(f'√(λ_2 / 2) (estimated h lower) = {np.sqrt(eigvals[1] / 2):.4f}')
print(f'2 √λ_2 (Cheeger upper) ≥ h ≥ λ_2/2 형태 검증')
```

### 실험 3 — Cycle $C_n$ Spectrum

```python
def cycle_laplacian(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i+1) % n] = 1
        A[(i+1) % n, i] = 1
    D = np.diag(A.sum(1))
    return D - A

n = 8
L_c = cycle_laplacian(n)
eig_c = np.linalg.eigvalsh(L_c)
theory = sorted([2 - 2*np.cos(2*np.pi*k/n) for k in range(n)])
print(f'Numerical: {eig_c.round(4)}')
print(f'Theory   : {np.round(theory, 4)}')
```

### 실험 4 — Random Walk Mixing

```python
P = np.diag(1/deg) @ A
pi = deg / deg.sum()

# 초기 분포: 노드 0
v = np.zeros(n); v[0] = 1.0

tv_distances = []
for t in range(50):
    tv = 0.5 * np.abs(v - pi).sum()
    tv_distances.append(tv)
    v = v @ P

mu = np.sort(np.linalg.eigvals(P).real)[::-1]   # 내림차순
mu_star = max(abs(mu[1]), abs(mu[-1]))
print(f'mu_star = {mu_star:.4f}')

plt.semilogy(tv_distances, 'o-')
plt.semilogy([mu_star**t for t in range(50)], '--', label=f'$\\mu_*^t$ = {mu_star:.3f}^t')
plt.xlabel('step t'); plt.ylabel('TV distance to π')
plt.title('Random walk mixing on Karate Club')
plt.legend(); plt.show()
```

### 실험 5 — Spectral Embedding과 k-means

```python
from sklearn.cluster import KMeans

k = 3
U_k = eigvecs[:, 1:k+1]   # 작은 k개 nontrivial
# Row-normalize (Ng-Jordan-Weiss)
U_k_norm = U_k / np.linalg.norm(U_k, axis=1, keepdims=True)
labels_km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(U_k_norm)
print(f'Spectral clustering labels: {labels_km}')
```

---

## 🔗 실전 활용

### 1. Spectral Clustering의 표준 알고리즘

1. $L_{\text{sym}}$ 계산
2. 작은 $k$ eigenvector $U_k$ 추출
3. 각 노드를 row vector로 보고 ($U_k$ 의 행), row-normalize
4. k-means

이는 Ng-Jordan-Weiss (2001) 의 표준 spectral clustering. Cheeger guarantee로 quasi-optimal cut.

### 2. Image Segmentation as Graph Cut

이미지를 weighted graph (픽셀 노드, 이웃 픽셀 사이 similarity edge) 로 보면 spectral clustering = image segmentation. Shi-Malik (2000) "Normalized Cuts and Image Segmentation" 의 기반.

### 3. Community Detection

소셜 네트워크에서 $\lambda_2, \lambda_3$ 등을 보고 community 수 결정 (eigengap heuristic — eigenvalue gap이 큰 곳에서 cluster 수 결정).

### 4. Spectral Gap → GCN 수렴 분석

Over-smoothing 속도 = $\lambda_2(L_{\text{sym}}^{(\text{aug})})$ 의 함수. Karate club과 같이 spectral gap 작으면 over-smoothing 느림 (good); expander 같으면 빠른 collapse — Ch5-02.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Connected (또는 component-wise) | Disconnected 시 $\lambda_2 = 0$, Fiedler 의미 X |
| Two-cluster 분해 (Fiedler) | 여러 cluster는 multi-eigenvector + k-means 필요 |
| Cheeger inequality는 lower/upper만 | 정확한 conductance 계산은 NP-hard |
| Eigenvector sign 모호 | LapPE 사용 시 sign flip 처리 (Ch4-05) |
| $L_{\text{sym}}$ vs $L_{\text{rw}}$ | 두 변종 존재, 경험적으로 $L_{\text{sym}}$ + row-normalize 가 안정 |
| Connected component별 분석 | Multi-component 시 component identification 선행 |

---

## 📌 핵심 정리

$$\boxed{0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n \leq 2 \quad (\text{normalized})}$$

$$\boxed{\frac{h^2(G)}{2} \leq \lambda_2 \leq 2 h(G) \quad \text{(Cheeger)}}$$

| 양 | 의미 | 응용 |
|----|------|------|
| $\lambda_1 = 0$ | $\mathbb{1}$ eigenvector, always | trivial |
| $\lambda_2$ (Fiedler value) | algebraic connectivity | spectral clustering |
| Fiedler vector $v_2$ | bisection 부호 | community detection |
| Spectral gap | mixing time, smoothness | random walk |
| $\lambda_n = 2$ | bipartite 신호 | Ch1-03 정리 3.3 |
| $h(G)$ | conductance | min-cut approx |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Path graph $P_n$ ($n$ 노드 일렬 연결) 의 첫 4개 Laplacian 고유값을 계산하고 Fiedler vector를 그려라.

<details>
<summary>해설</summary>

$P_n$ 의 unnormalized $L$ 고유값: $\lambda_k = 2 - 2\cos(\pi k / n)$ for $k = 0, 1, \ldots, n-1$.

$n = 10$: $\lambda_0 = 0$, $\lambda_1 \approx 0.0979$, $\lambda_2 \approx 0.382$, $\lambda_3 \approx 0.824$.

Fiedler vector $v_2$: $v_{2,i} \propto \cos(\pi (2i-1) / (2n))$ — sign 부호가 인덱스 절반에서 바뀜.

이는 path를 두 절반으로 분할 (자연스러움). $\square$

</details>

**문제 2** (심화): Cheeger's lower bound $\lambda_2 \geq h^2/2$ 의 등호 달성 조건을 추측하라. (Hint: tight한 그래프 family는?)

<details>
<summary>해설</summary>

**Tight 한 family**: Long path / barbell graph (양 끝에 dense cluster, 중간 긴 path).

- Path $P_n$: $h \approx 1/n$, $\lambda_2 \approx 1/n^2$ (정확히 $h^2/2$ 형태).
- Barbell: 두 dense cluster 가 단일 edge로 연결, conductance 매우 작음, $\lambda_2$ 도 매우 작음.

이런 그래프는 random walk가 cluster 안에서 섞이고 cluster 사이를 건너기 어려움 → 작은 spectral gap. $\square$

**반대**: Expander graph (random regular graph, Ramanujan graph) 는 $\lambda_2 \geq c$ for some constant — large spectral gap.

</details>

**문제 3** (논문 비평): LapPE (Dwivedi 2020) 가 GNN에 Laplacian eigenvector를 직접 PE로 주입한다. 이것이 어떤 한계를 우회하며, sign-flip 모호성 ($v$ 와 $-v$ 둘 다 valid) 은 어떻게 처리하는가?

<details>
<summary>해설</summary>

**WL 한계 우회**: 1-WL은 symmetric graph (CSL, Paley) 를 구분 못함. LapPE는 노드별 unique한 spectral 좌표 제공 → WL이 동등하게 취급하던 노드를 구분.

예: $C_4$ 의 모든 노드는 1-WL에서 동등 (모두 $d=2$, 이웃도 $d=2$). 하지만 $v_2$ 는 부호로 구분.

**Sign flip 모호성 처리 방법**:

1. **Random sign flip during training**: 매 epoch 또는 batch마다 $v_k \leftarrow \pm v_k$ 랜덤. 모델이 sign-invariant 학습.

2. **Sign-equivariant networks**: $\phi(v) = \phi(-v)$ 보장하는 architecture (e.g., absolute value, squared, even-power features).

3. **Sign-Net (Lim 2022)**: SignNet — eigenvector $v$ 와 $-v$ 모두를 network에 입력, output을 결합 ($\phi(v) + \phi(-v)$ 형태).

4. **Eigenvalue로 weighting**: $\sqrt{\lambda_k} v_k$ 사용, scale을 통해 모호성 일부 완화.

**문제점**: Repeated eigenvalue 시 eigenvector 자체가 모호 (eigenspace 내 임의 회전). 이는 더 어려운 문제로, BasisNet (Lim 2022) 등이 다룸.

따라서 LapPE는 강력하지만 sign/basis ambiguity 처리가 핵심 — Ch4-05에서 자세히.

</details>

---

<div align="center">

[◀ 이전](./03-normalized-laplacian.md) | [📚 README](../README.md) | [다음 ▶](./05-graph-fourier.md)

</div>
