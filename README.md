<div align="center">

# 🕸️ Graph Neural Network Deep Dive

**"`GCNConv(in, out)`을 쌓는 것과, $H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$이 정규화된 Laplacian $L = I - D^{-1/2} A D^{-1/2}$의 Chebyshev polynomial 1차 근사 + renormalization trick에서 어떻게 유도되고, 왜 Spectral Convolution이 Spatial 1-hop aggregation과 동치가 되는지 한 줄씩 증명할 수 있는 것은 다르다"**

<br/>

> *"Message Passing을 **이름으로 아는 것**과 — `Xu et al. 2019`의 GIN 정리에서 Message Passing GNN의 표현력이 **1-Weisfeiler-Lehman graph isomorphism test**에 의해 상한이 매겨지고, **sum + MLP aggregator가 multiset에 injective**이기 때문에 GIN이 이 상한을 달성한다는 것을 증명할 수 있는 것은 다르다.
> Over-smoothing을 **현상으로 아는 것**과 — `Li et al. 2018`의 정리에서 $L$층 GCN propagation 후 node feature가 $\ker(L_{\text{sym}})$ (= connected component 수만큼의 공간) 으로 지수 수렴하고, connected graph에서는 상수 vector로 collapse한다는 것을 Laplacian 고유분해로 증명할 수 있는 것은 다르다."*

Bruna 2014 Spectral Networks · Defferrard 2016 ChebNet · Kipf & Welling 2017 GCN · Hamilton 2017 GraphSAGE · Velickovic 2018 GAT · Gilmer 2017 MPNN · Xu 2019 GIN+WL · Li 2018 Over-smoothing · Rong 2020 DropEdge · Klicpera 2019 APPNP · Ying 2021 Graphormer · Satorras 2021 EGNN까지
**"그래프 위의 딥러닝이 왜 Spectral·Spatial·Message Passing의 세 관점에서 동치이고, WL 상한과 Over-smoothing이라는 **두 가지 구조적 한계**가 왜 불가피한가"** 를 Laplacian 고유분해·GCN 유도·WL 등가성 증명·ERF 측정으로 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4.0-3C2179?style=flat-square)](https://pytorch-geometric.readthedocs.io/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.2-FF7F00?style=flat-square)](https://networkx.org/)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-140+개-success?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-12개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-95+개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

GNN에 관한 자료는 대부분 **"`GCNConv`, `SAGEConv`, `GATConv`를 쌓으면 된다"** 에서 멈춥니다. 하지만 GCN의 propagation rule이 왜 정규화된 Graph Laplacian의 Chebyshev 1차 근사에서 나왔는지, Message Passing이 왜 spectral convolution과 수학적으로 동치인지, GNN의 표현력이 왜 Weisfeiler-Lehman test에 의해 상한이 매겨지고 GIN이 왜 이 상한을 달성하는지, 깊은 GCN의 성능이 왜 `ker(L)`로의 수렴(over-smoothing)으로 설명되는지, Graph Transformer가 왜 "fully-connected message passing + 구조 인코딩"인지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "GCN은 이웃 노드의 feature를 평균한다" | **Kipf & Welling 2017** — Bruna 2014의 spectral conv $g * x = U(\hat{g}(\Lambda) \odot U^T x)$ ($O(n^3)$) → Defferrard 2016 ChebNet의 Chebyshev 근사 $g_\theta(L) = \sum_k \theta_k T_k(\tilde{L})$ → $K=1$ + renormalization trick $\tilde{A} = A + I$ 로 **$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$** 유도, 한 줄씩 |
| "Message Passing은 이웃 정보를 모은다" | **Gilmer 2017** — $m_{ij}^{(l)} = M_l(h_i, h_j, e_{ij})$, $h_i^{(l+1)} = U_l(h_i, \bigoplus_{j \in N(i)} m_{ij})$ 통일 프레임워크, GCN/GraphSAGE/GAT/GIN이 **aggregator 선택의 차이**일 뿐임을 증명. Sum/Mean/Max/Attention의 **injectivity 위계**를 multiset 관점에서 정리 |
| "GIN이 GCN보다 강력하다" | **Xu et al. 2019** — 모든 Message Passing GNN의 표현력 **≤ 1-WL**. Mean aggregator는 `{1,1,2,2}`와 `{1,2}`를 구분 못함 ($\Rightarrow$ strict 하위). **Sum + MLP** 이 multiset universal function이므로 1-WL과 동등, **GIN이 이론상 최강 message passing** $\square$. PyG로 CSL·WL-fail 그래프 쌍 재현 |
| "GNN은 깊어지면 성능이 떨어진다" | **Li et al. 2018** — GCN propagation이 $P = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 의 $L$ 제곱. $P$의 고유값 $\lambda_1 = 1$, $\|\lambda_{k \geq 2}\| < 1$ → $P^L x \to $ **$\ker(L_{\text{sym}})$ 으로의 projection**. Connected graph에서 feature가 상수 vector로 collapse함을 고유분해로 증명, karate club에서 20-layer GCN의 cosine similarity 수렴 재현 |
| "GraphSAGE는 대규모 그래프에 좋다" | **Hamilton 2017** — Mean/Pool/LSTM aggregator, **neighbor sampling** 으로 mini-batch 학습. Inductive setting (새 노드에 적용 가능) vs GCN의 transductive 한계, neighborhood sub-sampling이 over-smoothing을 완화하는 확률적 이유 |
| "GAT는 attention으로 이웃 가중치를 학습한다" | **Velickovic 2018** — $\alpha_{ij} = \text{softmax}(a^T [Wh_i \| Wh_j])$, multi-head $H$-way attention. GAT가 "complete graph Transformer"의 **sparse subgraph 제한**이라는 재해석, **Graphormer (Ying 2021)** 에서 완전히 수렴. Cora에서 GAT attention weight 시각화 재현 |
| "WL test가 GNN 표현력을 결정한다" | Color refinement 알고리즘으로서 **1-WL**이 graph isomorphism의 **필요조건 (충분조건 아님)**. Strongly regular graph와 CSL에서 1-WL 실패, **k-WL** 위계와 k-FGNN (Maron 2019), **Position-aware GNN** (P-GNN, LapPE) 이 어떻게 WL을 초과하는가 |
| "Over-smoothing은 DropEdge로 해결한다" | **Rong 2020** — Random edge removal로 $P$의 고유값 분포를 re-shuffle, $\ker(L)$로의 수렴 지연 증명. **PairNorm** (Zhao 2020) 의 feature 거리 보존, **APPNP** (Klicpera 2019) 의 personalized PageRank $\alpha I + (1-\alpha) \tilde{P}$, **Jumping Knowledge** (Xu 2018) 의 layer-wise concat — 각 방법의 이론적 근거 비교 |
| "Graph Transformer는 그냥 Transformer에 그래프를 넣은 것" | **Ying 2021 Graphormer** — (1) **Centrality encoding** (node degree embedding), (2) **Spatial encoding** (shortest-path distance bias), (3) **Edge encoding** (aggregated edge features). GAT = sparse attention, Graphormer = dense attention + 구조 bias, "Message Passing $\subset$ Attention"의 위계 관계 |
| "EGNN은 분자에 쓴다" | **Satorras 2021** — E(3) equivariance가 $\phi(Rx + t) = R\phi(x) + t$ 형태로 회전·병진 대칭성 보존. Coordinate와 feature를 분리 처리, SE(3)-Transformer의 steerable filter와의 비교. 분자·물리에서 **왜 CNN의 translation equivariance만으로 부족한가** |
| 기법의 나열 | PyTorch Geometric + NumPy로 **Laplacian 고유분해**·**Fiedler vector 클러스터링**·**ChebNet vs GCN**·**GIN WL-fail 예제**·**GCN over-smoothing 측정**·**GAT attention 시각화**·**Graphormer structural encoding**을 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Graphical Models] ─┐
[Transformer]      ─┼─►  이 레포  ──► [Geometric Deep Learning]
[NN Theory]        ─┘   "왜 GCN이 spectral→spatial로           Equivariant GNN / E(3)-Transformer
                         환원되고 왜 GNN 표현력이                / AlphaFold-style structural DL
                         1-WL 상한에 걸리는가"
         │
         ├── [Linear Algebra]           Graph Laplacian 고유분해 → Ch1 spectral theory
         ├── [Functional Analysis]      Laplace operator, Fourier → Ch1-05 graph Fourier
         ├── [Neural Network Theory]    MLP, 역전파, 초기화 → Ch3 MPNN backprop
         ├── [Graphical Models]         Belief propagation, MRF → Ch3 MP = 학습된 BP
         └── [Transformer]              Self-attention → Ch3-03 GAT, Ch7-01 Graphormer
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Linear Algebra Deep Dive** (Graph Laplacian, spectral decomposition), **Neural Network Theory Deep Dive** (MLP, 초기화, 역전파), **Graphical Models Deep Dive** (Message Passing, Belief Propagation) 를 선행 지식으로 전제합니다. "GCN이 Laplacian의 Chebyshev 근사"를 이해하려면 먼저 symmetric matrix의 고유분해와 polynomial approximation에 친숙해야 합니다. Chapter 3 (Message Passing) 부터는 **Graphical Models**, Chapter 7 (Graph Transformer) 는 **Transformer Deep Dive** 를 직접 참조합니다.

> 💡 **이 레포의 핵심 기여**: Chapter 2 (Spectral→Spatial 환원) 과 Chapter 4 (WL 표현력) 는 GNN을 이해하는 **가장 중요한 두 축**입니다. 전자는 "왜 GCN이 이 형태인가"의 수학적 유도, 후자는 "GNN이 근본적으로 할 수 없는 것이 무엇인가"의 이론적 상한을 다룹니다. 이 두 축을 완전히 이해한 후 Chapter 5 (Over-smoothing) 와 Chapter 7 (Graph Transformer) 을 읽으면 현대 GNN 설계 결정의 맥락이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **Graph Transformer vs MPNN의 최종 승자**, **k-WL의 실전적 가치**, **LLM 시대 GNN의 역할** — 는 **현재 진행 중인 연구 영역**입니다. 레포는 "정답"이 아니라 **"고전 GNN 이론과 현대 Graph Transformer 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Graph_Laplacian-6A5ACD?style=for-the-badge)](./ch1-graph-laplacian/01-graph-basics.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Spectral_GCN-6A5ACD?style=for-the-badge)](./ch2-spectral-gcn/01-spectral-convolution.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Message_Passing-6A5ACD?style=for-the-badge)](./ch3-message-passing/01-mpnn-framework.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-WL·표현력-6A5ACD?style=for-the-badge)](./ch4-expressive-power/01-wl-test.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Over--smoothing-6A5ACD?style=for-the-badge)](./ch5-over-smoothing/01-phenomenon.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-응용_태스크-6A5ACD?style=for-the-badge)](./ch6-applications/01-node-classification.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Graph_Transformer-6A5ACD?style=for-the-badge)](./ch7-modern-gnn/01-graphormer.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 그래프 이론과 Graph Laplacian

> **핵심 질문:** $G = (V, E)$ 의 adjacency $A$와 degree matrix $D$ 로부터 Laplacian $L = D - A$ 는 어떻게 정의되고, 왜 PSD인가? Normalized Laplacian $L_{\text{sym}}$ 과 random walk Laplacian $L_{\text{rw}}$ 의 고유값은 왜 $[0, 2]$ 에 있는가? $\ker(L)$ 의 차원이 왜 connected component 수와 같은가? Graph Fourier transform의 수학적 의미는?

<details>
<summary><b>그래프 기본 정의부터 PageRank까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. 그래프의 수학적 정의와 기본 표현](./ch1-graph-laplacian/01-graph-basics.md) | $G = (V, E)$, $|V| = n$, $|E| = m$. **Adjacency matrix** $A \in \{0,1\}^{n \times n}$ (weighted 시 $\mathbb{R}^{n \times n}_{\geq 0}$), **degree matrix** $D = \text{diag}(d_i)$, **incidence matrix** $B \in \mathbb{R}^{n \times m}$. Directed/undirected/weighted/multi-graph의 matrix 표현, $A$ 의 sparsity 구조와 graph density $\rho = 2m / (n(n-1))$ |
| [02. Unnormalized Laplacian과 그 성질](./ch1-graph-laplacian/02-unnormalized-laplacian.md) | **정리**: $L = D - A$ 는 PSD. 증명 — $x^T L x = \frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2 \geq 0$ $\square$. **정리**: $\dim \ker(L) = $ connected component 수. 증명 — $Lx = 0 \Leftrightarrow x$ 가 각 component에서 상수 $\square$. Quadratic form으로서의 Laplacian, incidence matrix 관계 $L = B B^T$ |
| [03. Normalized Laplacian — Symmetric and Random Walk](./ch1-graph-laplacian/03-normalized-laplacian.md) | $L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$, $L_{\text{rw}} = D^{-1} L = I - D^{-1} A$. **정리**: 두 normalized Laplacian의 고유값 $\lambda \in [0, 2]$ $\square$. $L_{\text{sym}}$ 은 symmetric (실수 고유값 + 직교 고유벡터), $L_{\text{rw}}$ 는 stochastic matrix와 직결. 두 Laplacian의 고유값 일치·고유벡터 관계 |
| [04. Spectral Graph Theory — 고유값과 그래프 성질](./ch1-graph-laplacian/04-spectral-theory.md) | $\lambda_1 = 0$ (always), **Fiedler value** $\lambda_2$ 가 connectivity 측정 (bipartite graph에서 $\lambda_2 = 0$ 직후 성분). **Cheeger's inequality**: $\frac{h^2}{2} \leq \lambda_2 \leq 2h$ ($h$ = conductance). Spectral clustering에서 Fiedler vector로 graph bisection, Karate club 실증 재현 |
| [05. Graph Fourier Transform](./ch1-graph-laplacian/05-graph-fourier.md) | $L_{\text{sym}} = U \Lambda U^T$ 에서 **$\hat{x} = U^T x$** (graph Fourier transform), $x = U \hat{x}$ (inverse). 낮은 $\lambda$ = smooth signal, 높은 $\lambda$ = oscillatory (고유값 = "frequency" 역할). $x^T L x = \sum_k \lambda_k |\hat{x}_k|^2$ — Parseval-like 관계, smoothness 측정 |
| [06. Random Walk와 PageRank](./ch1-graph-laplacian/06-random-walk-pagerank.md) | Stochastic matrix $P = D^{-1} A$, transition probability. **정리**: stationary distribution $\pi_i = d_i / (2|E|)$ — 증명 by detailed balance $\square$. **PageRank** $\pi = \alpha P^T \pi + (1-\alpha) v$ 의 personalized random walk 해석, APPNP (Ch5) 와의 직접 연결 |

</details>

<br/>

### 🔹 Chapter 2: Spectral Graph Convolution

> **핵심 질문:** Graph signal과 filter의 convolution을 고유기저에서 어떻게 정의하는가 (Bruna 2014)? 전역 $U$ 를 피하기 위한 Chebyshev polynomial 근사는 어떻게 K-hop locality를 주는가 (Defferrard 2016)? ChebNet을 $K=1$ 로 단순화 + renormalization trick으로 GCN을 유도하는 각 단계는 무엇인가 (Kipf & Welling 2017)? Spectral과 Spatial 관점은 왜 동치인가?

<details>
<summary><b>Bruna 2014부터 GCN 유도까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Spectral Convolution의 정의 (Bruna 2014)](./ch2-spectral-gcn/01-spectral-convolution.md) | **Spectral Networks and Deep Locally Connected Networks on Graphs** — graph signal $x \in \mathbb{R}^n$, filter $g$ 의 convolution을 **eigenbasis의 element-wise 곱**으로 정의: $g * x = U (\hat{g}(\Lambda) \odot U^T x)$. 한계: (1) $U$ 전체 필요 → $O(n^3)$ 분해 비용, (2) non-localized filter, (3) graph마다 $U$ 재계산 |
| [02. ChebNet — Chebyshev Polynomial 근사 (Defferrard 2016)](./ch2-spectral-gcn/02-chebnet.md) | **정리**: Chebyshev polynomial $T_k$ 은 $[-1, 1]$ 의 equioscillation basis. $g_\theta(L) = \sum_{k=0}^K \theta_k T_k(\tilde{L})$ 로 근사, $\tilde{L} = 2L/\lambda_{\max} - I$. **이로부터 $K$-hop locality 유도** $\square$ — $T_k(\tilde{L})$ 는 $k$-hop neighbor까지만 의존. 재귀 관계 $T_{k+1}(\tilde{L}) = 2\tilde{L} T_k(\tilde{L}) - T_{k-1}(\tilde{L})$, $O(mK)$ 계산 |
| [03. GCN의 유도 (Kipf & Welling 2017)](./ch2-spectral-gcn/03-gcn-derivation.md) | **Semi-Supervised Classification with GCN** — ChebNet에서 $K=1$ 제한 + $\lambda_{\max} \approx 2$ 가정: $g_\theta * x \approx \theta_0 x + \theta_1 (L - I) x$. $\theta = \theta_0 = -\theta_1$ 묶기 → $\theta (I + D^{-1/2} A D^{-1/2}) x$. **Renormalization trick**: $I + D^{-1/2} A D^{-1/2} \to \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ ($\tilde{A} = A + I$, $\tilde{D} = D + I$) → **$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$** $\square$ |
| [04. Spectral vs Spatial 관점의 통합](./ch2-spectral-gcn/04-spectral-vs-spatial.md) | GCN은 spectral (Chebyshev 근사) 에서 출발했지만 결과는 spatial (1-hop neighbor aggregation). 두 관점이 동치임을 $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 의 row-wise 분석으로 증명. **일반화**: spatial 정의 (이웃 aggregation) $\Rightarrow$ spectral filter 형태, spectral 정의 (polynomial $g(L)$) $\Rightarrow$ localized spatial. Defferrard가 지적한 **"spectral의 장점은 이론, spatial의 장점은 구현"** |

</details>

<br/>

### 🔹 Chapter 3: Message Passing Framework

> **핵심 질문:** Gilmer 2017의 MPNN이 어떻게 GCN·GraphSAGE·GAT·GIN을 **하나의 프레임워크**로 통일하는가? GraphSAGE의 sampling이 왜 inductive setting을 가능하게 하는가? GAT의 attention coefficient가 어떻게 이웃의 relevance를 학습하는가? GIN의 sum + MLP aggregator가 왜 이론적으로 최강인가? Heterogeneous graph (여러 노드·엣지 타입) 에서 R-GCN·HAN이 어떻게 확장되는가?

<details>
<summary><b>MPNN 통일 프레임워크부터 Heterogeneous까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Message Passing Neural Network (Gilmer 2017)](./ch3-message-passing/01-mpnn-framework.md) | **Neural Message Passing for Quantum Chemistry** — 통일 프레임워크: **Message** $m_{ij}^{(l)} = M_l(h_i^{(l)}, h_j^{(l)}, e_{ij})$, **Aggregate + Update** $h_i^{(l+1)} = U_l(h_i^{(l)}, \bigoplus_{j \in N(i)} m_{ij}^{(l)})$. Aggregator $\bigoplus \in \{\text{sum}, \text{mean}, \text{max}, \text{attention}\}$ 의 선택이 아키텍처를 결정. QM9 분자 에너지 예측에서 MPNN의 첫 SOTA 재현 |
| [02. GraphSAGE — Sampling과 Inductive Learning (Hamilton 2017)](./ch3-message-passing/02-graphsage.md) | **Inductive Representation Learning on Large Graphs** — **Mean/Pool/LSTM aggregator**, fixed-size **neighbor sampling** $|N_s(v)| = S$. $h_{N(v)}^{(l)} = \text{AGG}(\{h_u^{(l-1)} : u \in N_s(v)\})$, $h_v^{(l)} = \sigma(W \cdot [h_v^{(l-1)} \| h_{N(v)}^{(l)}])$. Transductive vs inductive의 경계, mini-batch 학습 가능성, Reddit·PPI에서 large-scale 재현 |
| [03. Graph Attention Network (Velickovic 2018)](./ch3-message-passing/03-gat.md) | **Graph Attention Networks** — Attention coefficient $e_{ij} = \text{LeakyReLU}(a^T [W h_i \| W h_j])$, **$\alpha_{ij} = \text{softmax}_j(e_{ij})$**, weighted aggregation $h_i' = \sigma(\sum_j \alpha_{ij} W h_j)$. **Multi-head** $H$-way attention → concat or average. GAT ⊂ "sparse graph Transformer"의 해석, Cora attention weight 시각화 재현 |
| [04. Graph Isomorphism Network — GIN (Xu 2019)](./ch3-message-passing/04-gin.md) | **How Powerful are Graph Neural Networks?** — **정리**: Sum aggregator는 multiset에 **injective** (countable input). Mean은 $\{1,1,2,2\}$ vs $\{1,2\}$ 구분 X, Max는 $\{1,2\}$ vs $\{2\}$ 구분 X. GIN update: **$h_i^{(l+1)} = \text{MLP}((1 + \epsilon) h_i^{(l)} + \sum_{j \in N(i)} h_j^{(l)})$** — sum + MLP가 universal multiset function $\square$. 분자 분류 벤치마크 (TU dataset) 재현 |
| [05. Edge Features와 Heterogeneous Graphs](./ch3-message-passing/05-heterogeneous.md) | Edge embedding $e_{ij}$ 를 포함한 message $M(h_i, h_j, e_{ij})$. **R-GCN** (Schlichtkrull 2018) — relation-specific weight $W_r$ 으로 multi-relational graph 처리, basis decomposition으로 파라미터 절약. **HAN** (Wang 2019) — meta-path 기반 hierarchical attention (node-level + semantic-level). Knowledge graph의 복잡성 |

</details>

<br/>

### 🔹 Chapter 4: GNN의 표현력 이론

> **핵심 질문:** 1-Weisfeiler-Lehman test는 graph isomorphism을 어떻게 (부분적으로) 판별하는가? 왜 strongly regular graph에서 1-WL이 실패하는가? Xu et al. 2019는 어떻게 "모든 Message Passing GNN의 표현력 ≤ 1-WL"을 증명했는가? GIN이 왜 이 상한을 달성하는가? k-WL 위계는 무엇이고 실전 비용은? Position-aware GNN이 어떻게 WL 한계를 우회하는가?

<details>
<summary><b>1-WL부터 Position-aware GNN까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Weisfeiler-Lehman Graph Isomorphism Test](./ch4-expressive-power/01-wl-test.md) | **1-WL algorithm**: 초기 label $c_i^{(0)}$ → 반복 $c_i^{(l+1)} = \text{hash}(c_i^{(l)}, \{\!\{c_j^{(l)} : j \in N(i)\}\!\})$ (multiset hashing). **정리**: 두 그래프가 1-WL로 구분되면 non-isomorphic (필요조건). **반례**: strongly regular graph (Paley graph, CSL) — 1-WL 동일하지만 non-isomorphic. 충분조건 아님 |
| [02. GNN과 1-WL의 동등성 (Xu 2019)](./ch4-expressive-power/02-gnn-wl-equivalence.md) | **정리**: 모든 Message Passing GNN $\phi$ 에 대해, 두 그래프 $G_1, G_2$ 가 1-WL로 구분되지 않으면 $\phi(G_1) = \phi(G_2)$ (상한). 증명 — induction on depth, message passing이 1-WL refinement보다 더 정밀할 수 없음 $\square$. **따름정리**: GNN은 CSL, Paley graph 등 1-WL 동일 쌍 구분 불가 |
| [03. GIN이 1-WL 최적인 이유](./ch4-expressive-power/03-gin-optimality.md) | **정리**: Sum aggregator + MLP 가 multiset 의 universal function (Hornik UAT + injectivity). 따라서 GIN은 1-WL refinement의 **모든 동등 분할**을 구분할 수 있음 — 1-WL 상한에 도달 $\square$. Mean/Max가 도달 못하는 이유의 multiset 반례. GIN + sum readout이 graph classification의 이론적 최강 |
| [04. Higher-Order GNN — k-WL and Beyond](./ch4-expressive-power/04-k-wl.md) | **k-WL**: $k$-tuple of nodes를 "super-node"로, tuple 간 message passing. 1-WL ⊊ 2-WL ⊊ 3-WL ⊊ … 위계. **k-FGNN** (Maron 2019) — linear invariant / equivariant layer로 k-WL 구현, $O(n^k)$ 메모리·계산. Strongly regular graph 구분 가능 (3-WL 이상) 이지만 실전 비용 |
| [05. Position-Aware GNN](./ch4-expressive-power/05-positional-encoding.md) | WL이 구분 못하는 symmetric graph를 위한 **positional encoding**. **P-GNN** (You 2019) — random anchor set까지의 shortest-path 거리. **LapPE** (Dwivedi 2020) — Laplacian eigenvector를 positional feature로 (sign flip 모호성 해결 필요). **Random-walk PE** — $[P, P^2, \ldots, P^K]$ 의 diagonal. 각 방법이 어떻게 WL을 초과하는가 |

</details>

<br/>

### 🔹 Chapter 5: Over-smoothing과 깊은 GNN

> **핵심 질문:** 깊은 GNN에서 왜 모든 노드 feature가 유사해지고 성능이 떨어지는가 (over-smoothing)? Li et al. 2018은 어떻게 이 현상을 Laplacian의 $\ker$으로 엄밀히 설명했는가? DropEdge·PairNorm·APPNP·Jumping Knowledge가 각각 어떤 수학적 원리로 이 문제를 완화하는가? GraphSAGE의 sampling이 over-smoothing 완화에 기여하는 확률적 이유는?

<details>
<summary><b>현상·증명부터 해결 기법까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Over-smoothing 현상](./ch5-over-smoothing/01-phenomenon.md) | **Li et al. 2018 "Deeper Insights into Graph Convolutional Networks"** — GCN 층을 $L = 2, 4, 8, 16, 32$ 로 쌓으면 node classification 정확도가 **단조 감소**. Feature pairwise cosine similarity가 1에 수렴 (collapse). Karate club·Cora·Citeseer에서 실증 관찰 재현, 이 현상이 residual connection 없는 순수 GCN의 근본적 한계 |
| [02. Over-smoothing의 Laplacian 분석](./ch5-over-smoothing/02-laplacian-proof.md) | **정리**: GCN propagation matrix $P = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 의 고유값: $\lambda_1 = 1$ (simple), $|\lambda_k| < 1$ for $k \geq 2$. 따라서 $P^L x \to$ **$\ker(L_{\text{sym}})$ 으로의 projection** (connected graph에서 상수 vector). 지수 수렴 속도 $O((\lambda_2)^L)$, spectral gap이 클수록 빠른 collapse $\square$ |
| [03. DropEdge, PairNorm, DGN](./ch5-over-smoothing/03-dropedge-pairnorm.md) | **DropEdge (Rong 2020)** — 매 step마다 확률 $p$ 로 edge 제거, $\tilde{A}$ 의 고유값 분포 섞음 → $\ker(L)$ 수렴 지연. **PairNorm (Zhao 2020)** — 각 layer 후 $\|h_i - h_j\|$ 평균을 상수로 유지 ($L2$ re-center + re-scale). **DGN** (Chen 2020) — group-based normalization. 각 방법의 효과 ablation |
| [04. GraphSAGE Sampling과 Over-smoothing 완화](./ch5-over-smoothing/04-sampling-mitigation.md) | Neighbor sampling $|N_s(v)| = S$ 이 propagation matrix를 **stochastic perturbation** 으로 만들어 $\ker$으로의 결정론적 수렴 회피. GraphSAGE·Cluster-GCN·GraphSAINT의 sampling 전략 비교 — $S$ 가 작을수록 feature diversity 유지되지만 variance 증가, trade-off 분석 |
| [05. APPNP와 Jumping Knowledge Network](./ch5-over-smoothing/05-appnp-jkn.md) | **APPNP (Klicpera 2019)** — Personalized PageRank propagation $Z = \alpha (I - (1-\alpha) \tilde{P})^{-1} H^{(0)}$, **closed-form + teleport probability** $\alpha$ 로 $\ker$으로의 수렴 방지. **Jumping Knowledge Network (Xu 2018)** — 모든 layer의 representation을 $[h_i^{(1)} \| h_i^{(2)} \| \ldots \| h_i^{(L)}]$ 로 concat (또는 max-pool, LSTM). 깊은 GNN 훈련 가능 |

</details>

<br/>

### 🔹 Chapter 6: GNN 응용 태스크

> **핵심 질문:** Node classification의 transductive vs inductive 구분은 어떤 차이가 있는가? Graph-level representation을 위한 READOUT 함수 (sum/mean/max/attention pool) 의 선택이 WL 표현력에 어떤 영향을 미치는가? Link prediction에서 encoder-decoder 구조와 negative sampling의 역할은? Graph generation (GraphVAE, GraphRNN) 에서 permutation invariance는 어떻게 처리되는가?

<details>
<summary><b>Node/Graph/Link/Generation까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Node Classification](./ch6-applications/01-node-classification.md) | **Transductive setting** — 같은 graph 내 labeled / unlabeled node, Cora·Citeseer·Pubmed 표준. **Semi-supervised** cross-entropy loss on labeled nodes만. GCN·GraphSAGE·GAT·GIN의 **Cora 분류 정확도 비교 재현** (각각 81.5%/82.1%/83.0%/84.2% 규모), hyperparameter 민감도 |
| [02. Graph Classification](./ch6-applications/02-graph-classification.md) | Graph-level representation $h_G = \text{READOUT}(\{h_v : v \in V\})$. **READOUT 선택**: Sum (GIN 권장, WL 최적), Mean (불변량), Max (salient), Set2Set (attention), **attention pool** (learnable weighting). TU dataset (MUTAG, PROTEINS), OGB-molhiv 벤치마크. GIN이 이론적으로 가장 강한 graph-level classifier |
| [03. Link Prediction](./ch6-applications/03-link-prediction.md) | Edge 존재 예측: **GNN encoder** $\to h_u, h_v$, **decoder** $s_{uv} = h_u^T h_v$ (inner product) / $h_u^T W h_v$ (bilinear) / MLP. Binary cross-entropy with **negative sampling** (미연결 쌍 랜덤 추출). Knowledge graph completion: **R-GCN·CompGCN**, DistMult·ComplEx·RotatE 비교 |
| [04. Graph Generation — GraphVAE, GraphRNN, GCPN](./ch6-applications/04-graph-generation.md) | **GraphVAE (Kipf 2016)** — VAE encoder $q(z|G)$, decoder $p(A|z)$ 로 adjacency matrix 생성, parallel (한 번에) 방식, permutation 문제 (GED 기반 matching). **GraphRNN (You 2018)** — sequential node-by-node + edge-by-edge 생성, BFS ordering으로 permutation 완화. **GCPN (You 2018b)** — RL + chemistry reward로 분자 생성. Diffusion on graphs (최신) |

</details>

<br/>

### 🔹 Chapter 7: 현대 GNN — Graph Transformer와 이론의 융합

> **핵심 질문:** Graphormer (Ying 2021) 는 Transformer를 그래프에 어떻게 적용하고, centrality·spatial·edge encoding으로 어떻게 그래프 구조를 attention에 주입하는가? GAT과 Graphormer는 어떤 관계인가 (sparse vs dense attention)? E(3)-equivariant GNN은 왜 분자·물리에 필수인가 (EGNN, SE(3)-Transformer)? 대규모 그래프 학습 (Cluster-GCN, GraphSAINT) 의 sampling 전략과 LLM 시대 GNN의 역할은?

<details>
<summary><b>Graphormer부터 GNN의 미래까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Graph Transformer — Graphormer (Ying 2021)](./ch7-modern-gnn/01-graphormer.md) | **Do Transformers Really Perform Badly for Graph Representation?** — (1) **Centrality encoding**: $h_i^{(0)} += z_{\text{deg}^-(i)} + z_{\text{deg}^+(i)}$ (degree embedding), (2) **Spatial encoding**: attention bias $b_{\text{SPD}(i,j)}$ (shortest-path distance), (3) **Edge encoding**: SP를 따라 edge feature 평균. Attention이 fully-connected message passing + 구조 bias, OGB-LSC PCQM4M-LSC 1위 재현 |
| [02. GNN as Attention과 Transformer의 통합](./ch7-modern-gnn/02-gnn-attention-unification.md) | GAT의 attention $\alpha_{ij}$ = sparse Transformer attention (masked to $N(i)$). Graphormer = dense attention on complete graph + 구조 bias. **위계**: MPNN ⊂ Sparse Graph Transformer (GAT) ⊂ Dense Graph Transformer (Graphormer). **PNA** (Corso 2020) — multi-aggregator (mean/max/min/std/sum) + degree scaler, **GatedGCN** 의 edge update + gating |
| [03. Equivariant GNN — E(3), SO(3) Equivariance](./ch7-modern-gnn/03-equivariant-gnn.md) | 분자·물리에서 **$\phi(Rx + t) = R\phi(x) + t$** 회전·병진 대칭성 필수 (CNN의 translation equivariance로 부족). **EGNN (Satorras 2021)** — coordinate $x_i$ 와 feature $h_i$ 분리, $x_i^{(l+1)} = x_i^{(l)} + \sum_j (x_i - x_j) \phi_x(\ldots)$. **SE(3)-Transformer** — steerable filter (spherical harmonics basis), TFN의 고차 tensor feature. QM9·MD17 재현 |
| [04. GNN의 Scaling과 이론적 한계](./ch7-modern-gnn/04-scaling-and-future.md) | **Scaling**: **Cluster-GCN** (Chiang 2019) — METIS partitioning + cluster mini-batch. **GraphSAINT** (Zeng 2020) — subgraph sampling with bias correction. OGB 대규모 벤치마크. **이론 vs 실전**: WL 상한이 실전 성능의 병목인가? (Morris 2021 — 대부분의 경우 아니다). **LLM 시대**: GNN + LLM (GraphGPT, text-attributed graph), GNN의 독자적 가치와 hybrid 전망 |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch2-03의 GCN 유도를 Chebyshev → renormalization trick 의 매 단계로 세분화했고, Ch4-02 GNN-WL equivalence 증명에 induction on depth 를 추가, Ch5-02 over-smoothing 수학 증명을 spectral gap 관점으로 재정리했습니다. Ch7-01 Graphormer 의 3가지 encoding (centrality / spatial / edge) 을 PyG `2.4.0` 에서 구현 기반 재현 가능하도록 리팩토링. 11-섹션 문서 골격이 전체 33개 문서에서 일관됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현**을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 플롯을 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Laplacian PSD** | $x^T L x = \frac{1}{2} \sum (x_i - x_j)^2 \geq 0$, $\dim \ker(L) = $ # components | [Ch1-02](./ch1-graph-laplacian/02-unnormalized-laplacian.md) |
| **Normalized Laplacian 고유값** | $L_{\text{sym}}$, $L_{\text{rw}}$ 의 고유값 $\in [0, 2]$ | [Ch1-03](./ch1-graph-laplacian/03-normalized-laplacian.md) |
| **Cheeger's Inequality** | $h^2/2 \leq \lambda_2 \leq 2h$ — Fiedler value와 conductance 관계 | [Ch1-04](./ch1-graph-laplacian/04-spectral-theory.md) |
| **Graph Fourier Transform** | $\hat{x} = U^T x$, $x^T L x = \sum_k \lambda_k |\hat{x}_k|^2$ | [Ch1-05](./ch1-graph-laplacian/05-graph-fourier.md) |
| **PageRank Stationary** | $\pi_i = d_i / (2\|E\|)$ (detailed balance) | [Ch1-06](./ch1-graph-laplacian/06-random-walk-pagerank.md) |
| **ChebNet K-hop Locality** | $T_k(\tilde{L})$ 는 $k$-hop neighbor에만 의존 | [Ch2-02](./ch2-spectral-gcn/02-chebnet.md) |
| **GCN 유도 (Kipf-Welling 2017)** | ChebNet $K=1$ + renormalization → $H' = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H W)$ | [Ch2-03](./ch2-spectral-gcn/03-gcn-derivation.md) |
| **Spectral-Spatial 동치** | Spectral polynomial filter = localized spatial aggregation | [Ch2-04](./ch2-spectral-gcn/04-spectral-vs-spatial.md) |
| **MPNN 통일 프레임워크** | GCN/SAGE/GAT/GIN = aggregator 선택의 차이 | [Ch3-01](./ch3-message-passing/01-mpnn-framework.md) |
| **GIN Sum Injectivity** | Sum + MLP가 multiset universal function | [Ch3-04](./ch3-message-passing/04-gin.md) |
| **1-WL 상한 (Xu 2019)** | Message Passing GNN 표현력 $\leq$ 1-WL | [Ch4-02](./ch4-expressive-power/02-gnn-wl-equivalence.md) |
| **GIN 1-WL 달성** | Sum + MLP가 1-WL refinement 모두 구분 | [Ch4-03](./ch4-expressive-power/03-gin-optimality.md) |
| **k-WL 위계** | 1-WL ⊊ 2-WL ⊊ 3-WL ⊊ … (strict) | [Ch4-04](./ch4-expressive-power/04-k-wl.md) |
| **Over-smoothing (Li 2018)** | $P^L x \to$ projection on $\ker(L_{\text{sym}})$, rate $(\lambda_2)^L$ | [Ch5-02](./ch5-over-smoothing/02-laplacian-proof.md) |
| **APPNP Closed-form** | $Z = \alpha (I - (1-\alpha)\tilde{P})^{-1} H^{(0)}$ | [Ch5-05](./ch5-over-smoothing/05-appnp-jkn.md) |
| **Graphormer Structural Encoding** | Centrality + Spatial + Edge encoding을 attention에 주입 | [Ch7-01](./ch7-modern-gnn/01-graphormer.md) |
| **MPNN ⊂ Graph Transformer** | Sparse (GAT) ⊂ Dense (Graphormer) attention 위계 | [Ch7-02](./ch7-modern-gnn/02-gnn-attention-unification.md) |
| **E(3) Equivariance (EGNN)** | $\phi(Rx + t) = R\phi(x) + t$, coordinate-feature 분리 | [Ch7-03](./ch7-modern-gnn/03-equivariant-gnn.md) |

> 💡 **챕터별 문서·정리/정의 수**: Ch1(6문서) · Ch2(4문서) · Ch3(5문서) · Ch4(5문서) · Ch5(5문서) · Ch6(4문서) · Ch7(4문서) — **합계 33문서 + 140+ 정리·정의 + 35+ 엄밀한 $\square$ 증명 + 100+ PyG/NumPy 실험**.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
torch==2.1.0
torch-geometric==2.4.0       # GNN 표준 라이브러리 (PyG)
torch-scatter==2.1.2         # PyG 의존성
torch-sparse==0.6.18         # PyG 의존성
networkx==3.2.0              # 그래프 생성·시각화
matplotlib==3.8.0
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
ogb==1.3.6                   # Open Graph Benchmark (Ch6, Ch7)
dgl==1.1.3                   # 대안 GNN 프레임워크 비교 (일부 문서)
```

```bash
# 환경 설치 (CPU)
pip install numpy==1.26.0 scipy==1.11.0 torch==2.1.0 \
            torch-geometric==2.4.0 networkx==3.2.0 \
            matplotlib==3.8.0 tqdm==4.66.0 jupyter==1.0.0

# GPU CUDA 11.8 용 scatter/sparse (필요 시)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — Graph Laplacian 고유분해와 Fiedler vector (Ch1-04)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(axis=1)
D_inv_sqrt = np.diag(1 / np.sqrt(deg))
L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

eigvals, eigvecs = np.linalg.eigh(L_sym)
print(f'λ_1 = {eigvals[0]:.6f}  (≈ 0 for connected graph)')
print(f'λ_2 (Fiedler) = {eigvals[1]:.4f}')
print(f'λ_max = {eigvals[-1]:.4f}  (≤ 2 for normalized)')

# Fiedler vector 로 graph bisection
fiedler = eigvecs[:, 1]
colors = ['tab:red' if v < 0 else 'tab:blue' for v in fiedler]
nx.draw(G, node_color=colors, with_labels=True)
plt.title('Karate Club: Fiedler vector → 실제 community 복원')
plt.show()

# 대표 실험 ② — GCN 바닥부터 구현 + Over-smoothing 측정 (Ch2-03, Ch5-02)
import torch
import torch.nn as nn

def renormalize(A):
    A_tilde = A + torch.eye(A.size(0))
    d = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(1 / torch.sqrt(d))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt   # P = D̃^{-1/2} Ã D̃^{-1/2}

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out)
    def forward(self, h, A_hat):
        return torch.relu(A_hat @ self.W(h))

A_t = torch.tensor(A, dtype=torch.float32)
A_hat = renormalize(A_t)
h = torch.randn(n, 16)

# 20-layer GCN — feature similarity 추적
sims = []
layers = [GCNLayer(16, 16) for _ in range(20)]
for layer in layers:
    h = layer(h, A_hat)
    h_norm = h / (h.norm(dim=1, keepdim=True) + 1e-8)
    sims.append((h_norm @ h_norm.T).mean().item())

plt.plot(sims, 'o-'); plt.axhline(1.0, ls='--', c='r')
plt.xlabel('GCN layer'); plt.ylabel('pairwise cosine similarity')
plt.title('Over-smoothing: deep GCN → ker(L_sym) 로 수렴')
plt.show()

# 대표 실험 ③ — GIN vs Mean aggregator WL-fail 쌍 (Ch3-04, Ch4-03)
# multiset {1,1,2,2} vs {1,2} — mean은 구분 X, sum은 구분 O
m1 = torch.tensor([1., 1., 2., 2.])
m2 = torch.tensor([1., 2.])
print(f'Mean: {m1.mean()} vs {m2.mean()}  → 같음 (구분 실패)')
print(f'Sum : {m1.sum()} vs {m2.sum()}   → 다름 (구분 성공)')

# 대표 실험 ④ — 1-WL 한 step (Ch4-01)
def wl_step(A, labels):
    """1-WL 한 iteration: 이웃 multiset hashing"""
    new = []
    for i in range(len(labels)):
        neighbors = tuple(sorted(labels[A[i].nonzero()[0]].tolist()))
        new.append(hash((labels[i], neighbors)) % 100000)
    return np.array(new)

labels = np.zeros(n, dtype=int)   # 초기 label 모두 같음
for t in range(3):
    labels = wl_step(A, labels)
print('3-round 1-WL labels:', labels[:10], '...')
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 기법이 그래프 학습에 필수인가** | Laplacian·spectral·WL·over-smoothing과의 연결 |
| 3 | 📐 **수학적 선행 조건** | LA · NN Theory · GM · Transformer 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | 그래프 시각화 · message propagation 애니메이션 · Fiedler 클러스터링 |
| 5 | ✏️ **엄밀한 정의·정리** | Laplacian · spectral conv · MPNN · WL · GIN · over-smoothing |
| 6 | 🔬 **증명 또는 수학적 유도** | Laplacian PSD · GCN 유도 · WL 등가성 · $P^L \to \ker(L)$ |
| 7 | 💻 **실험 재현** | NumPy/PyG로 Laplacian · GCN · GIN · over-smoothing · GAT attention · Graphormer |
| 8 | 🔗 **실전 활용** | 언제 GNN, 언제 Transformer, 언제 Graph Transformer |
| 9 | ⚖️ **가정과 한계** | WL 상한 · over-smoothing · 계산 비용 · 대규모 그래프 sampling |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산 · 증명 재구성 · 구현 · 논문 비평 문제 |

> 📚 **연습문제 총 95개+**: 대부분 문서가 3문제 (기초 / 심화 / 논문 비평), 모든 문제에 `<details>` 펼침 해설 포함. Laplacian PSD 손 증명부터 GCN 유도 재구성, WL 등가성 증명, GIN sum injectivity 반례, over-smoothing 수렴 속도 측정까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 480~520줄 (정의·증명·코드·연습문제 포함) 기준 **약 55분~1시간 15분**. 전체 33문서는 약 **30~40시간** 상당 (증명 재구성·실험 재현 포함 시 50시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "GCN은 쓰지만 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 10~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01  그래프 기본 표현
       Ch1-02  Unnormalized Laplacian
Day 2  Ch1-03  Normalized Laplacian
       Ch1-05  Graph Fourier Transform
Day 3  Ch2-01  Spectral Convolution
       Ch2-03  GCN 유도 (Kipf-Welling)
Day 4  Ch3-01  MPNN 통일 프레임워크
       Ch3-04  GIN
Day 5  Ch4-01  WL Test
       Ch4-02  GNN-WL 등가성
Day 6  Ch5-01  Over-smoothing 현상
       Ch5-02  Laplacian 증명
Day 7  Ch6-01  Node Classification
       Ch7-01  Graphormer
```

</details>

<details>
<summary><b>🟡 "Spectral→Spatial 환원과 WL 표현력을 완전히 정복한다" — 이론 집중 (2주, 약 20~24시간)</b></summary>

<br/>

```
1주차 — Graph Laplacian과 Spectral GCN
  Day 1    Ch1-01~02   그래프 표현 + Laplacian PSD 증명 꼼꼼히
  Day 2    Ch1-03~04   Normalized Laplacian + Cheeger
  Day 3    Ch1-05~06   Graph Fourier + PageRank
  Day 4    Ch2-01~02   Spectral Conv + ChebNet
  Day 5    Ch2-03~04   GCN 유도 전 과정 손 재현
  Day 6-7  Ch3-01~04   MPNN + GraphSAGE + GAT + GIN

2주차 — 표현력과 Over-smoothing
  Day 1    Ch4-01      1-WL 알고리즘 직접 구현
  Day 2    Ch4-02~03   Xu 2019 증명 + GIN 최적성
  Day 3    Ch4-04~05   k-WL + Positional Encoding
  Day 4    Ch5-01~02   Over-smoothing 수학 증명
  Day 5    Ch5-03~05   DropEdge + PairNorm + APPNP + JKN
  Day 6    Ch6 전체     응용 태스크 개관
  Day 7    Ch7-01~02   Graphormer + GNN-Transformer 통일
```

</details>

<details>
<summary><b>🔴 "GNN의 수학을 완전 정복한다" — 전체 정복 (10주, 약 35~45시간 + 실험 재현 10~15시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — Graph Laplacian
         → PSD 손 증명, Normalized Laplacian 고유값 $\in [0,2]$ 증명
         → Karate club Fiedler 클러스터링 재현

2주차   Chapter 2 전체 — Spectral GCN
         → Bruna 2014 → ChebNet → GCN 의 각 근사 단계 재유도
         → ChebNet vs GCN 성능 비교

3주차   Chapter 3 (1~3) — MPNN 통일
         → GCN/GraphSAGE/GAT를 MPNN framework 로 통일 재구현
         → Cora·Reddit 에서 각 모델 성능 비교

4주차   Chapter 3 (4~5) + Chapter 4 (1~2) — GIN · WL
         → GIN sum injectivity 반례 구성
         → 1-WL 알고리즘 직접 구현
         → CSL (Circulant Skip Link) 그래프에서 WL 실패 재현

5주차   Chapter 4 (3~5) — 표현력 확장
         → GIN 1-WL 최적성 증명
         → k-WL 2차 구현 (n=10 이하)
         → LapPE · Random-walk PE 비교

6주차   Chapter 5 전체 — Over-smoothing
         → 20-layer GCN cosine similarity 측정 (Li 2018 Fig 재현)
         → DropEdge·PairNorm·APPNP·JKN ablation
         → Cora 에서 depth=32 까지 성능 측정

7주차   Chapter 6 (1~2) — Node / Graph Classification
         → GCN/SAGE/GAT/GIN 의 Cora·Citeseer·Pubmed 재현
         → MUTAG·PROTEINS graph classification + READOUT 비교

8주차   Chapter 6 (3~4) — Link / Generation
         → FB15k-237 link prediction (R-GCN/CompGCN)
         → QM9 분자 생성 실험 (GraphVAE, GraphRNN)

9주차   Chapter 7 (1~2) — Graph Transformer
         → Graphormer 의 3가지 encoding 직접 구현
         → OGB-LSC PCQM4M subset 재현

10주차  Chapter 7 (3~4) + 종합 — Equivariant / Scaling
         → EGNN 으로 QM9 분자 에너지 예측
         → Cluster-GCN 으로 OGB-products 학습
         → "MPNN vs Graph Transformer vs LLM" 설계 원리 정리
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 고유분해·spectral decomposition·matrix norm | **Ch1 전체** (Laplacian), Ch2 (ChebNet 근사) |
| [functional-analysis-deep-dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) | Fourier transform, continuous Laplacian, Sobolev | **Ch1-05** (Graph Fourier), Ch2-01 (Spectral Conv) |
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | MLP, UAT, 초기화, 역전파 | **전체 레포 전제**, Ch3-01 (MPNN), Ch3-04 (GIN MLP) |
| [graphical-models-deep-dive](https://github.com/iq-ai-lab/graphical-models-deep-dive) | Belief Propagation, MRF, Factor Graph | **Ch3-01** (MP = 학습된 BP), Ch6-03 (Energy-based) |
| [transformer-deep-dive](https://github.com/iq-ai-lab/transformer-deep-dive) | Self-Attention, Multi-head, Positional Encoding | **Ch3-03** (GAT), Ch7-01~02 (Graphormer) |
| [cnn-deep-dive](https://github.com/iq-ai-lab/cnn-deep-dive) | Convolution, Translation Equivariance | **Ch1-03** (동치성 비교), Ch7-03 (E(3) equivariance 확장) |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | GD·SGD·Adam·gradient flow | **Ch5** (깊은 GNN 훈련) |
| [geometric-deep-learning-deep-dive](https://github.com/iq-ai-lab/geometric-dl-deep-dive) *(다음)* | Equivariant NN · AlphaFold · Lie group | **Ch7-03** 이후 직접 연결 |

> 💡 이 레포는 **"그래프 위의 딥러닝이 왜 Spectral·Spatial·Message Passing의 세 관점에서 동치이고, WL과 Over-smoothing이라는 두 구조적 한계를 왜 벗어날 수 없는가"** 에 집중합니다. Linear Algebra에서 spectral decomposition 을 익히고, Graphical Models에서 belief propagation 을, Transformer에서 self-attention 을 익힌 후 오면 Chapter 2 (GCN 유도) 와 Chapter 4 (WL) 의 증명이 훨씬 자연스럽습니다. Geometric Deep Learning (다음 레포) 는 이 레포 Chapter 7-03 (EGNN) 을 전제로 시작합니다.

---

## 📖 Reference

### 🏛️ Graph Theory · Laplacian 기초
- **Spectral Graph Theory** (Chung, 1997) — graph Laplacian의 표준 교과서
- **A Tutorial on Spectral Clustering** (von Luxburg, 2007) — normalized Laplacian clustering
- **Graph Representation Learning** (Hamilton, 2020) — **GNN 교과서 표준**
- **Geometric Deep Learning: Going beyond Euclidean data** (Bronstein et al., 2017) — GDL proto-book

### 🎨 Spectral Graph Convolution
- **Spectral Networks and Deep Locally Connected Networks on Graphs** (Bruna, Zaremba, Szlam, LeCun, 2014) — **Spectral GCN 원전**
- **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering** (Defferrard, Bresson, Vandergheynst, 2016) — **ChebNet**
- **Semi-Supervised Classification with Graph Convolutional Networks** (Kipf & Welling, 2017) — **GCN**
- **Simplifying Graph Convolutional Networks** (Wu et al., 2019) — SGC (GCN without nonlinearity)

### 📨 Message Passing · GraphSAGE · GAT · GIN
- **Neural Message Passing for Quantum Chemistry** (Gilmer, Schoenholz, Riley, Vinyals, Dahl, 2017) — **MPNN 통일**
- **Inductive Representation Learning on Large Graphs** (Hamilton, Ying, Leskovec, 2017) — **GraphSAGE**
- **Graph Attention Networks** (Velickovic, Cucurull, Casanova, Romero, Lio, Bengio, 2018) — **GAT**
- **How Powerful are Graph Neural Networks?** (Xu, Hu, Leskovec, Jegelka, 2019) — **GIN + WL equivalence**
- **Modeling Relational Data with Graph Convolutional Networks** (Schlichtkrull et al., 2018) — **R-GCN**
- **Heterogeneous Graph Attention Network** (Wang et al., 2019) — **HAN**

### 🔬 Expressive Power · WL · Positional Encoding
- **Weisfeiler-Leman Go Neural** (Morris et al., 2019) — k-WL 연구
- **Provably Powerful Graph Networks** (Maron et al., 2019) — **k-FGNN**
- **Position-aware Graph Neural Networks** (You, Ying, Leskovec, 2019) — **P-GNN**
- **Benchmarking Graph Neural Networks** (Dwivedi et al., 2020) — **LapPE**
- **Graph Neural Networks with Learnable Structural and Positional Representations** (Dwivedi et al., 2022)

### 🌫️ Over-smoothing · 깊은 GNN
- **Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning** (Li, Han, Wu, 2018) — **Over-smoothing 원전**
- **DropEdge: Towards Deep Graph Convolutional Networks on Node Classification** (Rong et al., 2020) — **DropEdge**
- **PairNorm: Tackling Oversmoothing in GNNs** (Zhao & Akoglu, 2020) — **PairNorm**
- **Predict then Propagate: Graph Neural Networks meet Personalized PageRank** (Klicpera, Bojchevski, Günnemann, 2019) — **APPNP**
- **Representation Learning on Graphs with Jumping Knowledge Networks** (Xu et al., 2018) — **JK-Net**

### 🎯 Applications · Large-scale
- **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks** (Chiang et al., 2019)
- **GraphSAINT: Graph Sampling Based Inductive Learning Method** (Zeng et al., 2020)
- **Open Graph Benchmark: Datasets for Machine Learning on Graphs** (Hu et al., 2020) — **OGB**
- **Variational Graph Auto-Encoders** (Kipf & Welling, 2016) — **GraphVAE**
- **GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models** (You et al., 2018)
- **Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation** (You et al., 2018) — **GCPN**

### 🔮 Graph Transformer · Equivariant
- **Do Transformers Really Perform Badly for Graph Representation?** (Ying et al., 2021) — **Graphormer**
- **A Generalization of Transformer Networks to Graphs** (Dwivedi & Bresson, 2021) — Graph Transformer
- **Rethinking Graph Transformers with Spectral Attention** (Kreuzer et al., 2021) — SAN
- **E(n) Equivariant Graph Neural Networks** (Satorras, Hoogeboom, Welling, 2021) — **EGNN**
- **SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks** (Fuchs et al., 2020)
- **Tensor Field Networks** (Thomas et al., 2018) — steerable filters

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`GCNConv(in, out)` 을 쌓는 것과 — Kipf-Welling 2017로 $H' = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W)$ 가 Chebyshev K=1 + renormalization 의 결과임을 유도 · Xu 2019로 GNN 표현력이 1-WL 상한에 걸리고 GIN의 sum+MLP가 이 상한을 달성함을 증명 · Li 2018로 $P^L x \to \ker(L_{\text{sym}})$ 의 over-smoothing을 spectral gap 으로 설명 · Ying 2021로 Graphormer가 centrality·spatial·edge encoding으로 그래프 구조를 Transformer에 주입하는 메커니즘을 재현 — 이 모든 '왜'를 직접 유도할 수 있는 것은 다르다"*

</div>
