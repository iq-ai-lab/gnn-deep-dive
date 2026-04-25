# 01. Weisfeiler-Lehman Graph Isomorphism Test

## 🎯 핵심 질문

- 1-WL (color refinement) algorithm은 무엇이고 어떻게 graph isomorphism 을 (부분적으로) 판별하는가?
- 왜 1-WL 이 graph isomorphism 의 **필요조건은 되지만 충분조건은 아닌가**?
- Strongly regular graph (Paley graph, Circulant Skip Link) 가 왜 1-WL 의 결정적 반례인가?
- 1-WL refinement 의 색깔 분할이 어떻게 GNN 의 표현력 상한이 되는가? (Ch4-02 에서 자세히)
- WL test 의 알고리즘 복잡도와 GNN 학습 복잡도의 비교는?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GNN 의 표현력 한계를 이해하기 위해서는 **graph isomorphism problem (GI)** 을 알아야 합니다:

- **GI**: 두 graph $G_1, G_2$ 가 isomorphic 한지 ($V_1 = \pi(V_2)$ for some permutation $\pi$) 판별
- 결정 복잡도 : NP 에 속하지만 NP-complete 도 P 도 알려지지 않은 매우 특수한 위치 (Babai 2016: quasi-polynomial)
- 실전: heuristic algorithm 으로 대부분의 graph 구분 가능

**Weisfeiler-Lehman test** 는 GI 의 가장 유명한 **다항시간 heuristic**:
- **k-WL** ($k = 1, 2, 3, \ldots$) 의 위계
- **1-WL** = color refinement = **각 노드의 multiset hashing 반복**

이 1-WL 이 message passing GNN 의 표현력 상한 (Ch4-02). 따라서 1-WL 의 메커니즘과 한계를 이해하는 것이 GNN 표현력 이론의 핵심.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch3-04](../ch3-message-passing/04-gin.md) — GIN 의 sum + MLP
- 그래프 이론: graph isomorphism 정의
- 알고리즘: hash function

---

## 📖 직관적 이해

### Color Refinement = 1-WL Algorithm

각 노드에 "color" (label) 을 할당하고, 매 step 에서 자신과 이웃의 color multiset 을 보고 새 color 부여:

**Step 0**: 모든 노드 같은 color (또는 input feature)

**Step $l \to l+1$**: 노드 $i$ 의 새 color
$$
c_i^{(l+1)} = \text{hash}\left(c_i^{(l)}, \{\!\{c_j^{(l)} : j \in N(i)\}\!\}\right)
$$

이 hash 함수가 noise 없이 unique 매핑이라면, **multiset 자체** 가 색깔 결정.

### 종료 조건과 색 분할

각 step 마다 노드들이 색에 의해 분할 (partition). Refinement 가 더 이상 진행 안 되면 (= 어떤 step 에서 partition 이 변하지 않으면) **stable color partition** 도달.

두 그래프의 stable partition 이 같은 multiset of colors 를 가지면 1-WL 동등 (potentially isomorphic). 다르면 non-isomorphic 확실.

### 직관적 예: Cycle vs Path

- $C_4$ (4-cycle): 모든 노드 $d = 2$, 이웃도 모두 $d = 2$. 1-WL stable color = 모든 노드 같은 color.
- $P_4$ (4-path): 끝 2 노드 $d = 1$, 내부 2 노드 $d = 2$. 1-WL 첫 step 에서 두 그룹 분리.
- 따라서 1-WL 이 $C_4$ vs $P_4$ 구분 — 다행히 정확.

### CSL 의 반례

**Circulant Skip Link** $\text{CSL}(n, k)$: $C_n$ + 모든 노드에 distance-$k$ skip edge. $\text{CSL}(8, 2)$ vs $\text{CSL}(8, 3)$: 둘 다 4-regular, 같은 degree distribution — **1-WL 이 구분 불가**.

이런 highly symmetric graph 에서 1-WL 한계 명확.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Graph Isomorphism

두 graph $G_1 = (V_1, E_1)$, $G_2 = (V_2, E_2)$ 가 isomorphic ($G_1 \cong G_2$) 이려면:
$$
\exists \text{ bijection } \pi: V_1 \to V_2 \text{ such that } (u, v) \in E_1 \Leftrightarrow (\pi(u), \pi(v)) \in E_2
$$

### 정의 1.2 — 1-WL (Color Refinement) Algorithm

**Input**: Graph $G$, optional initial labels $\{c_i^{(0)}\}$.

**Iteration**: For $l = 1, 2, \ldots$:
$$
c_i^{(l)} = \text{hash}\left( c_i^{(l-1)}, \{\!\{c_j^{(l-1)} : j \in N(i)\}\!\} \right)
$$

**Termination**: Color partition $\{c_i^{(l)}\} \approx \{c_i^{(l-1)}\}$ (refinement 정지) 시 종료. 보통 $O(n)$ iteration 안에 종료.

**Output**: Stable color partition $\{c_i^*\}$.

### 정의 1.3 — 1-WL Equivalence

두 graph $G_1, G_2$ 가 1-WL 동등:
$$
G_1 \stackrel{1\text{-WL}}{\equiv} G_2 \Leftrightarrow \{\!\{c_i^*(G_1) : i \in V_1\}\!\} = \{\!\{c_j^*(G_2) : j \in V_2\}\!\}
$$

(stable color 의 multiset 일치)

### 정의 1.4 — k-WL (Higher-order)

$k$-tuple of nodes $(v_1, \ldots, v_k) \in V^k$ 를 super-node 로 보고 그 사이 일반화 message passing. (Ch4-04 에서 자세히)

위계: 1-WL ⊊ 2-WL ⊊ 3-WL ⊊ … (strictly increasing power).

### 정의 1.5 — Strongly Regular Graph

$(n, k, \lambda, \mu)$-strongly regular: $n$ 노드, $k$-regular, 인접 두 노드의 공통 이웃 수 = $\lambda$, 비인접 두 노드의 공통 이웃 수 = $\mu$.

이런 graph 들은 매우 symmetric — 1-WL 가 구분 못하는 경우 흔함.

**예**: Paley graph $P(q)$ with $q \equiv 1 \pmod 4$ prime. $P(13)$, $P(17)$ 등이 1-WL 동등하지만 non-isomorphic 가능 (특수 case).

---

## 🔬 정리와 결과

### 정리 1.1 — 1-WL의 Necessary Condition

**Theorem**: $G_1 \cong G_2$ ⟹ $G_1 \stackrel{1\text{-WL}}{\equiv} G_2$.

**증명** (induction on iteration):

Isomorphism $\pi: V_1 \to V_2$ 가정. Initial color: 동일 (둘 다 default 또는 isomorphism-preserving feature).

Step $l \to l+1$: $c_i^{(l)}(G_1) = c_{\pi(i)}^{(l)}(G_2)$ (IH) 가정.

$c_j^{(l)}(G_1) = c_{\pi(j)}^{(l)}(G_2)$ for $j \in N_{G_1}(i)$. $\pi$ isomorphism ⟹ $\pi(j) \in N_{G_2}(\pi(i))$.

따라서:
$$
\{\!\{c_j^{(l)}(G_1) : j \in N_{G_1}(i)\}\!\} = \{\!\{c_{\pi(j)}^{(l)}(G_2) : \pi(j) \in N_{G_2}(\pi(i))\}\!\}
$$

Hash 가 deterministic + multiset 동일 → $c_i^{(l+1)}(G_1) = c_{\pi(i)}^{(l+1)}(G_2)$. 

종합: 모든 step 에서 두 graph 의 color sequence 일치. $\square$

따라서 **1-WL 가 다른 결과 → non-isomorphic 확실**.

### 정리 1.2 — 1-WL의 Insufficient Condition (반례)

**Counter-example**: $\text{CSL}(8, 2)$ 와 $\text{CSL}(8, 3)$.

두 graph 모두:
- 8 노드, 4-regular
- 각 노드 이웃: 2 nearest cycle + 2 skip — 같은 degree pattern

1-WL Step 1: 모든 노드 같은 color (정확히 같은 이웃 multiset of color 0). Refinement 정지 — 두 그래프 1-WL 동등.

하지만 isomorphic? $\text{CSL}(8, 2)$ 는 $C_8 \cup C_8$ (두 4-cycle), $\text{CSL}(8, 3)$ 은 $C_8$ extended — 구조 다름 → non-isomorphic. $\square$

따라서 1-WL 동등이지만 non-isomorphic 인 그래프 쌍 존재. 즉 **1-WL 이 graph isomorphism 의 충분조건 아님**.

### 정리 1.3 — 1-WL의 종료 시간

**Theorem**: 1-WL 은 최대 $n$ iteration 안에 종료.

**증명 sketch**: Color partition 의 refinement 가 매 step 진행하거나 정지. Partition 의 cell 수는 $1 \to n$ 사이, 매 step 에서 strictly 증가하거나 stable. 따라서 최대 $n - 1$ step 후 stable. $\square$

알고리즘 복잡도: $O(n m + n^2)$ — practical 으로 빠름.

### 정리 1.4 — Color Refinement 의 Stable Partition Characterization

**Theorem**: 1-WL stable color partition 은 graph 의 **equitable partition** — 같은 색깔 cell 의 노드들이 다른 cell 에 같은 수의 이웃을 가짐.

**의미**: 두 그래프의 equitable partition 이 같으면 1-WL 동등 — graph 의 "structural symmetry" 가 1-WL 가 capture 할 수 있는 한계.

### 정리 1.5 — 1-WL과 Graph Spectrum 의 관계

**Theorem (informal)**: 1-WL stable partition 이 $L$ 의 spectral 정보 일부 인코딩. 단, spectrum 으로 구분 못하는 cospectral graph (Schwenk 1973) 가 존재 — 1-WL 도 일반적으로 못함.

따라서 spectrum-based GNN (Spectral GCN) + WL 은 모두 비슷한 한계.

### 정리 1.6 — Babai 의 결과

**Babai 2016**: GI 는 quasi-polynomial $\exp(O((\log n)^c))$ 알고리즘 가짐. 단, k-WL 이 충분히 큰 $k$ 에서 GI 해결한다는 가설은 미해결.

---

## 💻 구현

### 실험 1 — 1-WL 직접 구현

```python
import numpy as np
import networkx as nx
from collections import Counter

def wl_step(G, labels):
    """One iteration of 1-WL."""
    new_labels = {}
    for v in G.nodes():
        neighbor_labels = tuple(sorted(labels[u] for u in G.neighbors(v)))
        new_labels[v] = (labels[v], neighbor_labels)
    
    # Re-encode tuples as integers (canonical form)
    label_dict = {}
    next_id = 0
    canonical = {}
    for v, lab in new_labels.items():
        if lab not in label_dict:
            label_dict[lab] = next_id
            next_id += 1
        canonical[v] = label_dict[lab]
    return canonical

def wl_test(G, max_iter=100, init_labels=None):
    """Run 1-WL until stable."""
    if init_labels is None:
        labels = {v: 0 for v in G.nodes()}
    else:
        labels = init_labels.copy()
    
    history = [labels.copy()]
    for _ in range(max_iter):
        new_labels = wl_step(G, labels)
        if Counter(new_labels.values()) == Counter(labels.values()):
            return labels, history
        labels = new_labels
        history.append(labels.copy())
    return labels, history

# Test on simple graph
G = nx.path_graph(4)
labels, hist = wl_test(G)
print(f'P_4 stable colors: {labels}')
print(f'Color multiset: {Counter(labels.values())}')
```

### 실험 2 — Karate Club 의 1-WL Refinement

```python
G = nx.karate_club_graph()
labels, hist = wl_test(G)
print(f'Iterations: {len(hist)}')
print(f'Final # colors: {len(set(labels.values()))}')
print(f'Color multiset: {Counter(labels.values()).most_common(5)}')
```

### 실험 3 — CSL Counter-example

```python
def build_csl(n, skip):
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + skip) % n)
    return G

# CSL(10, 2) vs CSL(10, 3)
G1 = build_csl(10, 2)
G2 = build_csl(10, 3)

print(f'Isomorphic? {nx.is_isomorphic(G1, G2)}')

l1, _ = wl_test(G1)
l2, _ = wl_test(G2)
mc1 = Counter(l1.values())
mc2 = Counter(l2.values())

print(f'CSL(10,2) color multiset: {mc1}')
print(f'CSL(10,3) color multiset: {mc2}')
print(f'Same multiset? {mc1 == mc2}')
```

**예상**: 두 그래프 1-WL 동등 (multiset 같음), but CSL(10,2) 와 CSL(10,3) 구분 가능 (실제로 isomorphic 아님).

### 실험 4 — Strongly Regular Graph

```python
# Paley graph P(13): 13 nodes, q=13 prime, q ≡ 1 mod 4
# 각 노드 i 에 대해 j 가 quadratic residue (i - j) 차이 → edge
def paley_graph(q):
    QR = set((i * i) % q for i in range(1, q))
    G = nx.Graph()
    G.add_nodes_from(range(q))
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in QR or (i - j) % q in QR:
                G.add_edge(i, j)
    return G

P13 = paley_graph(13)
print(f'P(13): {P13.number_of_nodes()} nodes, '
      f'{P13.number_of_edges()} edges, '
      f'all deg = {[P13.degree(i) for i in range(13)]}')

l, _ = wl_test(P13)
print(f'P(13) WL color multiset: {Counter(l.values())}')
# 대칭성으로 인해 모든 노드 같은 color → 한 cell
```

### 실험 5 — WL 가 구분하는 경우

```python
G_path = nx.path_graph(5)
G_cycle = nx.cycle_graph(5)

l_p, _ = wl_test(G_path)
l_c, _ = wl_test(G_cycle)

print(f'Path P_5 colors: {Counter(l_p.values())}')
print(f'Cycle C_5 colors: {Counter(l_c.values())}')
print(f'Different? {Counter(l_p.values()) != Counter(l_c.values())}')
# True — path 의 끝과 내부 노드 색깔 다름, cycle 의 모든 노드 같음
```

---

## 🔗 실전 활용

### 1. WL Graph Kernel

WL test 가 graph kernel (Shervashidze 2011) 의 기반: stable color partition 의 multiset 을 graph fingerprint 로 사용. SVM 등 kernel-based classifier 의 input.

### 2. Graph Isomorphism Software

- **nauty** (McKay 1981): canonical labeling, GI 의 표준 도구. WL 의 변형 사용.
- **bliss** (Junttila 2007): faster nauty.

### 3. GNN 표현력 검증

새 GNN 모델 제안 시 "1-WL 보다 강한가?" "k-WL 도달 가능한가?" 가 표준 질문 — Ch4-02, Ch4-04.

### 4. Graph Database

대규모 graph DB (Neo4j 등) 의 isomorphism query 에 WL-style heuristic 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Hash 가 perfect (no collision) | 실제 hash 충돌 시 false equivalence |
| Static graph | Dynamic graph 의 incremental WL 별도 |
| Undirected | Directed graph 의 WL 일반화 |
| Discrete labels | Continuous feature 시 quantization 필요 |
| 1-WL 충분조건 X | k-WL 이 더 강하지만 컴퓨터 비용 ↑ |
| Symmetric graph 한계 | Position-aware 추가 (Ch4-05) |

---

## 📌 핵심 정리

$$\boxed{c_i^{(l+1)} = \text{hash}(c_i^{(l)}, \{\!\{c_j^{(l)} : j \in N(i)\}\!\})}$$

| 사실 | 진술 |
|------|------|
| **1-WL 종료** | 최대 $n$ iteration |
| **Necessary** | $G_1 \cong G_2 \Rightarrow$ 1-WL 동등 |
| **Not sufficient** | CSL, Paley, etc. 1-WL 동등 but non-iso |
| **Stable partition** | Equitable partition |
| **GNN 상한** | Message passing GNN 표현력 ≤ 1-WL (Ch4-02) |
| **k-WL 위계** | 1-WL ⊊ 2-WL ⊊ … (Ch4-04) |
| **GI 복잡도 (Babai)** | quasi-polynomial |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $C_5$ (5-cycle) 와 $K_3 \cup K_2$ (disconnect 3-clique + edge) 가 1-WL 으로 구분되는지 확인하라.

<details>
<summary>해설</summary>

**$C_5$**: 5 노드 모두 $d = 2$. 이웃 multiset 도 모두 $\{2, 2\}$ → 1-WL 첫 step 후 모든 노드 같은 색. Stable.

Color multiset: $\{\!\{0, 0, 0, 0, 0\}\!\}$ (단순화).

**$K_3 \cup K_2$**: 3 노드 $d = 2$ ($K_3$), 2 노드 $d = 1$ ($K_2$).
- $K_3$ 노드의 이웃: $\{2, 2\}$
- $K_2$ 노드의 이웃: $\{1\}$
- 첫 step 후: $K_3$ 와 $K_2$ 색깔 다름.

Color multiset: $\{\!\{c_1, c_1, c_1, c_2, c_2\}\!\}$ (3 vs 2 분할).

**구분 가능**: 두 multiset 다름 → 1-WL 가 구분. $\square$

**의미**: 단순한 graph 에서 1-WL 의 effectiveness 확인. 복잡한 symmetric graph (CSL) 에서만 1-WL 한계 노출.

</details>

**문제 2** (심화): 두 cospectral graph (같은 Laplacian spectrum 이지만 non-isomorphic) 의 표준 예 (Schwenk 1973) 를 찾고, 이들이 1-WL 동등인지 확인하라.

<details>
<summary>해설</summary>

**Schwenk 1973 의 예**: $K_{1,4}$ (star graph) + extra edge 와 $C_4 + K_1$ — 둘 다 같은 spectrum 가지지만 non-iso.

또는 더 simple: **$K_{3,3} \cup C_2$ 와 $C_6 \cup K_2$** — 6 + 2 = 8 노드, 같은 spectrum.

**1-WL 적용**:
- $K_{3,3}$: 모든 노드 $d = 3$, 이웃 $d = 3$. 1-WL stable: 모든 노드 같은 색 (within $K_{3,3}$). $C_2$ 와 합치면 두 cell.
- $C_6 \cup K_2$: $C_6$ 모든 노드 $d = 2$, $K_2$ 모든 노드 $d = 1$. 두 cell.

두 graph color multiset 다름 ($K_{3,3}$ 의 d=3 vs $C_6$ 의 d=2) → 1-WL 가 구분. ✓

**다른 예**: 더 미묘한 cospectral pair (Cvetkovic 1985) 가 1-WL 동등인 경우 존재. 일반적으로 spectrum vs WL 둘 다 한계 있지만, 정확히 같은 한계는 아님.

**Theorem (Fürer 2010)**: 거의 모든 graph 는 1-WL 로 구분 가능 (random graph). 그러나 worst-case 에는 한계.

</details>

**문제 3** (논문 비평): 1-WL 의 한계를 우회하는 방법으로 (a) k-WL, (b) random ID injection, (c) positional encoding 세 가지가 있다. 각각의 방법이 어떻게 작동하는지 비교하고 trade-off 를 논하라.

<details>
<summary>해설</summary>

**(a) k-WL (Ch4-04)**:
- $k$-tuple 을 super-node 로
- 1-WL ⊊ 2-WL ⊊ 3-WL ⊊ …
- $k = 3$ 부터 strongly regular graph 구분
- 비용: $O(n^k)$ — $k = 3$ 도 $n = 1000$ 에서 $10^9$ 연산
- 표현력: 이론상 무한 ($k \to n$ 시 GI 해결)
- **현실성**: $k \leq 3$ 까지만 실용 (실증적으로 효과 marginal)

**(b) Random ID Injection (Murphy 2019, You 2021)**:
- 각 노드에 랜덤 unique ID 추가
- WL 첫 step 에서 이미 모든 노드 다른 색 → CSL 도 구분
- 단점: random 이라 같은 graph 의 다른 ID 부여마다 다른 결과 → permutation invariance 깨짐
- 해결: Random sampling 평균 (high variance)
- **단점**: training instability, sample inefficiency

**(c) Positional Encoding (Ch4-05)**:
- LapPE (Laplacian eigenvector), Random-walk PE, P-GNN (anchor distances)
- Deterministic structural feature → 1-WL 한계 우회
- LapPE: graph spectrum 자체가 정보 ($U^T x$ 의 unique 성)
- 단점: Sign / basis ambiguity, 동일 eigenvalue 시 모호

**Trade-off**:

| 방법 | 표현력 | 효율 | 안정성 | 일반화 |
|------|--------|------|--------|--------|
| k-WL | 매우 강 | 매우 비쌈 | 결정적 | 강 |
| Random ID | 강 | 빠름 | 불안정 (variance) | 약 |
| Positional Encoding | 강 | 빠름 | 결정적 (sign 문제) | 강 |

**현대 추세**: Positional encoding (특히 LapPE, RWPE) 이 표준. Graphormer (Ch7-01) 의 spatial encoding 도 이 family.

따라서 1-WL 우회는 single answer 없음 — task / graph structure 에 따른 선택.

</details>

---

<div align="center">

[◀ 이전](../ch3-message-passing/05-heterogeneous.md) | [📚 README](../README.md) | [다음 ▶](./02-gnn-wl-equivalence.md)

</div>
