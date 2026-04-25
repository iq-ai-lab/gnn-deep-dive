# 06. Random Walk와 PageRank

## 🎯 핵심 질문

- Stochastic matrix $P = D^{-1} A$ 가 random walk transition을 어떻게 정의하는가?
- 왜 stationary distribution $\pi_i = d_i / (2|E|)$ 이고 detailed balance로 어떻게 증명하는가?
- PageRank는 personalized random walk 의 어떤 variant 인가?
- $L_{\text{rw}}$ 의 spectral 성질이 random walk 의 mixing time 을 어떻게 결정하는가?
- APPNP (Klicpera 2019) 의 closed-form $Z = \alpha (I - (1-\alpha) \tilde{P})^{-1} H^{(0)}$ 는 어떻게 PageRank 에서 유도되는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Random walk와 PageRank는 GNN에 두 가지 결정적 영향을 미칩니다:

1. **Message passing 의 확률적 해석** — GraphSAGE 의 mean aggregator = 1-step random walk, GCN 의 propagation = symmetric variant
2. **Over-smoothing 분석** — $P^L$ 의 수렴 속도가 over-smoothing 의 정량화 (Ch5-02)
3. **APPNP 의 동기** — Personalized PageRank propagation 으로 over-smoothing 회피 (Ch5-05)
4. **Node embedding (DeepWalk, node2vec)** — Random walk sampling 으로 word2vec-style embedding
5. **Graph kernel 과 graph similarity** — Random walk graph kernel, return probability 등

이 문서에서는 random walk 의 **수학적 기반과 PageRank 유도** 를 정리하고, 후속 챕터의 응용을 위한 토대를 마련합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [03-normalized-laplacian.md](./03-normalized-laplacian.md) — $L_{\text{rw}}$, $P = D^{-1} A$
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Markov chain, Perron-Frobenius theorem
- 확률론: 조건부 확률, stationary distribution

---

## 📖 직관적 이해

### Random Walk on Graph

각 step에 현재 노드의 이웃 중 균등 랜덤 선택:
$$
\Pr[X_{t+1} = j \mid X_t = i] = \frac{A_{ij}}{d_i} = P_{ij}
$$

$P = D^{-1} A$ 는 **row-stochastic** (각 행 합 = 1, 모두 nonneg). $t$-step 후 분포 = $\pi_0^T P^t$ (initial distribution).

### Stationary Distribution

$\pi$ 가 stationary 이면 $\pi^T P = \pi^T$ — 한 step 지나도 분포 불변. Random walk 가 충분히 오래 진행되면 어떤 초기에서든 stationary 에 수렴 (irreducible + aperiodic 가정).

**놀라운 사실**: Undirected graph 에서 stationary $\pi_i \propto d_i$. **고차수 노드일수록 자주 방문**. 직관: hub 노드는 들어오는 path 가 많음.

### PageRank: Personalized Random Walk

원본 PageRank (Brin-Page 1998):
$$
\pi = \alpha P^T \pi + (1 - \alpha) v
$$

각 step에서 $1 - \alpha$ 확률로 "teleport" (vector $v$ 로 reset), $\alpha$ 확률로 random walk. $v$ 가 uniform 이면 standard PageRank, $v = e_s$ (특정 노드) 이면 **personalized PageRank** (PPR).

**해석**: $\alpha$ 작을수록 starting node $s$ 근처에 집중, $\alpha$ 클수록 long-range diffusion.

### APPNP 의 등장

PPR 의 closed form: $\pi = (1 - \alpha) (I - \alpha P^T)^{-1} v$. 

GNN 관점: 초기 feature $H^{(0)}$ 에 PPR 적용:
$$
Z = (1 - \alpha) (I - \alpha \tilde{P})^{-1} H^{(0)}
$$

(또는 $\alpha$ 와 $1 - \alpha$ 의 정의를 뒤집은 형태). 이는 over-smoothing 회피 — Ch5-05.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Markov Chain on Graph

확률 공간 $(V, P)$ 위의 Markov chain: state $X_t$, transition $\Pr[X_{t+1} = j | X_t = i] = P_{ij}$.

Random walk on graph $G$: $P = D^{-1} A$.

### 정의 6.2 — Stationary Distribution

분포 $\pi: V \to [0, 1]$ ($\sum_i \pi_i = 1$) 이 stationary 이려면:
$$
\pi^T = \pi^T P \quad \Leftrightarrow \quad \pi_j = \sum_i \pi_i P_{ij}
$$

### 정의 6.3 — Detailed Balance (Reversibility)

분포 $\pi$ 가 detailed balance 조건 만족:
$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \forall i, j
$$

(rate from $i$ to $j$ = rate from $j$ to $i$)

### 정의 6.4 — PageRank

Damping factor $\alpha \in (0, 1)$, teleport vector $v$ ($\sum v_i = 1$).
$$
\pi = (1 - \alpha) v + \alpha P^T \pi
$$

(원 PageRank 에서는 column-stochastic $P^T$ 사용. 여기서 row-stochastic $P$ 와 일관 위해 $P^T$ 명시.)

**Personalized PageRank** $\pi_s$: $v = e_s$ (one-hot at source $s$).

### 정의 6.5 — Mixing Time

$\epsilon$-mixing time:
$$
\tau_{\text{mix}}(\epsilon) = \min \{ t : \max_{\pi_0} \| \pi_0^T P^t - \pi^T \|_{\text{TV}} \leq \epsilon \}
$$

---

## 🔬 정리와 증명

### 정리 6.1 — Stationary $\pi_i \propto d_i$

**Theorem**: Connected, undirected graph 에서 random walk $P = D^{-1} A$ 의 stationary distribution:
$$
\pi_i = \frac{d_i}{2 |E|}
$$

**증명** (detailed balance 사용):

확인할 것: $\pi_i P_{ij} = \pi_j P_{ji}$.

$$
\pi_i P_{ij} = \frac{d_i}{2|E|} \cdot \frac{A_{ij}}{d_i} = \frac{A_{ij}}{2|E|}
$$

$$
\pi_j P_{ji} = \frac{d_j}{2|E|} \cdot \frac{A_{ji}}{d_j} = \frac{A_{ji}}{2|E|}
$$

Undirected: $A_{ij} = A_{ji}$. 따라서 등식 성립 ($\pi$ reversible).

Detailed balance ⟹ stationary (직접 확인): $(\pi^T P)_j = \sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j$. ✓

또한 $\sum_i \pi_i = \sum_i d_i / (2|E|) = 2|E| / (2|E|) = 1$. ✓ $\square$

### 정리 6.2 — Convergence to Stationary

**Theorem**: Connected, non-bipartite graph 에서 임의 초기 분포 $\pi_0$:
$$
\lim_{t \to \infty} \pi_0^T P^t = \pi^T
$$

**증명 sketch**: $P$ 는 row-stochastic이고 irreducible (connected) + aperiodic (non-bipartite ⟹ gcd of cycle lengths = 1). Perron-Frobenius theorem으로 simple eigenvalue 1 (with eigenvector $\mathbb{1}$ right, $\pi$ left), 다른 모든 eigenvalue $|\mu| < 1$. $P^t$ 의 spectral 분해에서 $\mu^t \to 0$ for $|\mu| < 1$. $\square$

**Bipartite 의 문제**: $\pi_0^T P^{2t}$ 와 $\pi_0^T P^{2t+1}$ 가 다른 값으로 진동 (period 2). 이를 **lazy random walk** $P_{\text{lazy}} = (I + P)/2$ 로 해결.

### 정리 6.3 — Mixing Time과 Spectral Gap

**Theorem**:
$$
\tau_{\text{mix}}(\epsilon) \leq \frac{1}{1 - \mu_*} \ln \left( \frac{1}{\epsilon \cdot \min_i \pi_i^{1/2}} \right)
$$

여기서 $\mu_* = \max(|\mu_2|, |\mu_n|)$, $\mu_i$ = $P$ 고유값.

(spectral gap $1 - \mu_2$ 가 클수록 빠른 mixing)

### 정리 6.4 — PageRank Closed Form

PageRank equation $\pi = (1 - \alpha) v + \alpha P^T \pi$ 의 해:
$$
\pi = (1 - \alpha) (I - \alpha P^T)^{-1} v
$$

**증명**: Equation 재배열: $(I - \alpha P^T) \pi = (1 - \alpha) v$. $\alpha < 1$, $\|P^T\| \leq 1$ (row-stochastic ⟹ $\|P\|_\infty = 1$, 그 transpose 도 bounded) ⟹ $(I - \alpha P^T)$ invertible. 양변에 inverse 곱. $\square$

**Geometric series 형태**:
$$
\pi = (1 - \alpha) \sum_{k=0}^\infty \alpha^k (P^T)^k v = (1 - \alpha) \sum_k \alpha^k P^k v_{\text{symmetric}}
$$

(undirected) — 각 hop 의 random walk 분포를 $\alpha^k$ 로 가중 평균.

### 정리 6.5 — Symmetric Variant of PageRank

GNN 에서는 종종 symmetric normalization 사용 ($P$ 대신 $\hat{A} = D^{-1/2} A D^{-1/2}$):
$$
Z = (1 - \alpha)(I - \alpha \hat{A})^{-1} H^{(0)}
$$

**해석**: 각 노드의 representation 이 모든 hop 에서의 propagation $\hat{A}^k H^{(0)}$ 의 geometric series. $\alpha$ 가 hop weight를 제어 — $\alpha \to 0$ 이면 1-hop, $\alpha \to 1$ 이면 모든 hop.

이는 APPNP 의 propagation rule (Ch5-05). $H^{(0)}$ 에 MLP 적용 후 PPR propagation → 학습 가능 GNN.

### 정리 6.6 — Random Walk Probabilities and $A$

**Theorem**: $(P^k)_{ij} \cdot d_i = \sum_{\text{walks of length } k \text{ from } i \text{ to } j} \prod_{\text{edges}} \frac{1}{d_{\text{intermediate}}}$ 형태.

(증명은 induction. 직관: 각 step에서 $1/d_{\text{current}}$ 확률로 다음 노드 선택)

특히 $(P^k)_{ii}$ = $i$ 로 돌아오는 random walk 확률 — return probability, 그래프 spectrum 과 직접 연결 (Cayley-Hamilton 변형).

---

## 💻 NumPy 구현 검증

### 실험 1 — Stationary Distribution 검증

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(1)
P = np.diag(1/deg) @ A
m = G.number_of_edges()

pi_theory = deg / (2 * m)

# Power iteration
pi_emp = np.ones(n) / n
for _ in range(2000):
    pi_emp = pi_emp @ P

print(f'||π_theory - π_empirical||_∞ = {np.abs(pi_theory - pi_emp).max():.2e}')
# 매우 작음 ≈ 0
```

### 실험 2 — Detailed Balance 확인

```python
# π_i P_{ij} = π_j P_{ji}
diff = pi_theory[:, None] * P - (pi_theory[:, None] * P).T
print(f'Detailed balance violation: {np.abs(diff).max():.2e}')   # ≈ 0
```

### 실험 3 — Mixing Time 측정

```python
def tv_distance(p, q):
    return 0.5 * np.abs(p - q).sum()

pi0 = np.zeros(n); pi0[0] = 1.0   # delta at node 0
distances = []
pi_cur = pi0.copy()
for t in range(100):
    distances.append(tv_distance(pi_cur, pi_theory))
    pi_cur = pi_cur @ P

# Spectral gap
mu = np.sort(np.abs(np.linalg.eigvals(P).real))[::-1]
mu_star = mu[1]   # 두 번째 큰 절댓값
print(f'mu_star = {mu_star:.4f},  expected mixing rate ~ mu_star^t')

plt.semilogy(distances, 'o-', label='TV distance')
plt.semilogy([mu_star**t for t in range(100)], '--', label=f'$\\mu_*^t$')
plt.xlabel('step t'); plt.ylabel('TV(π_t, π_∞)')
plt.title('Random walk mixing on Karate Club')
plt.legend(); plt.show()
```

### 실험 4 — PageRank 계산

```python
alpha = 0.85   # 표준 damping factor
v = np.ones(n) / n   # uniform teleport

# Closed form
PR_closed = (1 - alpha) * np.linalg.solve(np.eye(n) - alpha * P.T, v)
print(f'PageRank (closed form) sum = {PR_closed.sum():.6f}')   # 1

# Power iteration
PR_iter = v.copy()
for _ in range(500):
    PR_iter = (1 - alpha) * v + alpha * P.T @ PR_iter
print(f'PageRank (iterative) max diff = {np.abs(PR_closed - PR_iter).max():.2e}')

# Personalized PageRank (source = node 0)
v_s = np.zeros(n); v_s[0] = 1.0
PPR = (1 - alpha) * np.linalg.solve(np.eye(n) - alpha * P.T, v_s)
print(f'PPR (top-5 from node 0): {np.argsort(-PPR)[:5]}')
```

### 실험 5 — APPNP-style Propagation

```python
# APPNP: Z = (1-α)(I - α P̂)^{-1} H^{(0)}
# (여기서 P̂ = symmetric normalized adj for undirected GCN-style)
D_inv_sqrt = np.diag(1/np.sqrt(deg))
A_hat_sym = D_inv_sqrt @ A @ D_inv_sqrt   # without self-loop for simplicity

H0 = np.random.randn(n, 8)   # 초기 feature
alphas = [0.1, 0.3, 0.5, 0.9]

fig, axes = plt.subplots(1, len(alphas), figsize=(16, 3))
for ax, a in zip(axes, alphas):
    Z = (1 - a) * np.linalg.solve(np.eye(n) - a * A_hat_sym, H0)
    # 각 노드의 첫 feature 차원을 시각화
    nx.draw(G, nx.spring_layout(G, seed=42), ax=ax,
            node_color=Z[:, 0], cmap='RdBu',
            with_labels=False, node_size=80)
    ax.set_title(f'α = {a}')
plt.suptitle('APPNP propagation: α 클수록 long-range 영향')
plt.show()
```

---

## 🔗 실전 활용

### 1. PageRank → Web Search

원래 Brin-Page 의 PageRank: 웹페이지 = 노드, 링크 = directed edge. PageRank score 가 페이지 importance. Google의 검색 랭킹 기반.

### 2. Personalized PageRank → Recommendation

Source node $s$ 에 가까운 노드들 = 추천 후보. PPR이 빠른 (sparse) approximation 가능: ForwardPush, BackwardPush.

### 3. APPNP → Deep GNN

Klicpera 2019: 초기 MLP feature → PPR propagation → 분류. Over-smoothing 회피, 깊은 layer 효과 (∞-layer GCN의 closed-form). Ch5-05 참조.

### 4. Random Walk-based Embedding

DeepWalk (Perozzi 2014): random walk를 sentence로 보고 word2vec 적용. node2vec (Grover 2016): biased random walk (BFS / DFS 균형). 두 모델 모두 stationary distribution과 깊은 연결.

### 5. SGC (Wu 2019)

Simplified Graph Convolution: $Z = \hat{A}^K X W$ — 단순 power iteration + linear classifier. Stationary 가까이 가는 propagation.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Connected (irreducible) | Disconnected 시 component-wise stationary |
| Non-bipartite (aperiodic) | Bipartite 시 lazy walk $P_\text{lazy} = (I+P)/2$ 사용 |
| Undirected (reversibility) | Directed 는 detailed balance 일반 X, separate stationary |
| Static graph | Time-varying transition은 inhomogeneous Markov chain |
| Perfect random selection | 실제 implementation 은 sampling noise |
| $|V|$ tractable | $|V| \gg 10^6$ 에서는 closed-form $O(n^3)$ 불가 → iterative 또는 sparse |

---

## 📌 핵심 정리

$$\boxed{\pi_i = \frac{d_i}{2|E|} \quad \text{(stationary, undirected)}}$$

$$\boxed{\pi^T P = \pi^T \quad \text{(detailed balance ⟹ stationary)}}$$

$$\boxed{\text{PageRank: } \pi = (1 - \alpha) (I - \alpha P^T)^{-1} v}$$

| 개념 | 정의 | 응용 |
|------|------|------|
| **Transition $P$** | $P = D^{-1} A$ | random walk |
| **Stationary $\pi$** | $\pi \propto d$ | hub identification |
| **Detailed balance** | $\pi_i P_{ij} = \pi_j P_{ji}$ | reversibility |
| **Mixing time** | $\sim 1/(1 - \mu_2)$ | spectral gap 의존 |
| **PageRank** | teleport-augmented walk | importance, recommendation |
| **PPR** | $v = e_s$ | personalized score |
| **APPNP propagation** | $(1-\alpha)(I - \alpha \hat{A})^{-1}$ | 깊은 GNN (Ch5-05) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K_3$ (triangle) 의 random walk transition $P$ 를 손으로 구하고 stationary 가 uniform 임을 확인하라.

<details>
<summary>해설</summary>

$K_3$: 모든 노드 $d = 2$, $A = J - I$:

$$
P = D^{-1} A = \frac{1}{2} \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}
$$

Stationary: $\pi_i = d_i / (2|E|) = 2 / 6 = 1/3$ — uniform.

Verify: $\pi^T P = (1/3, 1/3, 1/3) \cdot P$ 의 $j$ 성분 = $(1/3)(0 + 1/2 + 1/2) = 1/3$ ✓. $\square$

**일반화**: $d$-regular graph 에서 stationary 항상 uniform.

</details>

**문제 2** (심화): Path graph $P_n$ 의 stationary distribution 을 손으로 구하고, mixing time 이 $O(n^2)$ 임을 spectral gap 으로 설명하라.

<details>
<summary>해설</summary>

$P_n$: 끝 노드 $d_1 = d_n = 1$, 내부 $d_i = 2$. $|E| = n - 1$, $2|E| = 2n - 2$.

$$
\pi_1 = \pi_n = \frac{1}{2n - 2}, \quad \pi_i = \frac{2}{2n - 2} = \frac{1}{n - 1} \text{ for } 1 < i < n
$$

(끝 노드는 절반의 stationary 확률, hub-bias 미약)

Mixing time: Path graph $P_n$ Laplacian $L_{\text{rw}}$ 의 spectral gap = 1 - $\mu_2(P)$. $L$ 의 $\lambda_2 = 2 - 2\cos(\pi/n) \approx \pi^2 / n^2$ (small $n$).

$L_{\text{rw}}$ 도 같은 order — $\lambda_2(L_{\text{rw}}) \approx \pi^2 / n^2$ → $\mu_2(P) \approx 1 - \pi^2/n^2$.

Mixing time $\tau \sim 1/(1 - \mu_2) \sim n^2/\pi^2 = O(n^2)$. $\square$

**의미**: 긴 path 에서는 random walk 가 한쪽 끝에서 다른 끝으로 가는 데 quadratic time. 이는 "diffusion is slow" — over-smoothing 도 path-like 구조에서는 천천히 일어남.

</details>

**문제 3** (논문 비평): APPNP (Klicpera 2019) 의 closed-form $Z = \alpha(I - (1-\alpha)\hat{A})^{-1} H^{(0)}$ 가 over-smoothing을 어떻게 회피하는지, 그리고 $\alpha \to 0$ 과 $\alpha \to 1$ 의 극한이 어떤 모델과 동등한지 설명하라.

<details>
<summary>해설</summary>

**Over-smoothing 회피 메커니즘**:

GCN 의 propagation $\hat{A}^L H$ 는 $L \to \infty$ 에서 $\ker(L_{\text{sym}})$ 으로 collapse — 모든 노드 feature 가 같아짐.

APPNP 는 **모든 hop 의 weighted geometric series**:
$$
Z = \alpha \sum_{k=0}^\infty (1-\alpha)^k \hat{A}^k H^{(0)}
$$

$\alpha$ 가 teleport probability — 매 step 에서 $\alpha$ 확률로 $H^{(0)}$ 으로 reset. 따라서:

- 초기 feature $H^{(0)}$ 가 보존됨 (geometric series 의 0-th term $\alpha H^{(0)}$)
- 모든 hop $k \geq 1$ 의 정보 합산되지만 $(1-\alpha)^k$ 로 감쇠 → 멀리 갈수록 영향 감소
- $\ker(L)$ 으로의 단순 수렴 차단 (teleport이 dominant 방향 유지)

**극한**:

- $\alpha \to 0$: $Z \to (1-0)^{\to\infty}$ 항이 dominant... 정확히는 $Z \to \lim_{k \to \infty} \hat{A}^k H^{(0)}$ = $\ker(L_{\text{sym}})$ 으로 projection. **GCN 무한 layer (= over-smoothed)**.

- $\alpha \to 1$: $Z \to \alpha H^{(0)} = H^{(0)}$ (1-step 만, 사실상 propagation 없음). **MLP 만**.

- $\alpha \in (0, 1)$: 두 극한 사이의 **balance** — graph structure (이웃 정보) 와 node-self feature (initial) 의 trade-off.

따라서 $\alpha$ 가 single-knob 으로 over-smoothing 과 under-utilization 사이를 조절 — 보통 $\alpha = 0.1 \sim 0.2$ 가 sweet spot. Ch5-05 에서 자세히.

</details>

---

<div align="center">

[◀ 이전](./05-graph-fourier.md) | [📚 README](../README.md) | [다음 ▶](../ch2-spectral-gcn/01-spectral-convolution.md)

</div>
