# 04. Graph Generation — GraphVAE, GraphRNN, GCPN

## 🎯 핵심 질문

- Graph generation 의 핵심 challenge — **permutation invariance** 를 어떻게 처리하는가?
- GraphVAE (Kipf 2016) 의 parallel generation 과 graph edit distance 매칭의 trade-off?
- GraphRNN (You 2018) 의 sequential (node-by-node) generation 이 어떻게 permutation 문제 완화?
- GCPN (You 2018) 의 RL + chemistry reward 가 분자 생성에 어떤 의미?
- 최신 Diffusion on graph (DiGress, GDSS) 가 기존 방법 을 어떻게 압도하는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

**Graph generation** — 새 graph 를 sampling 으로 만드는 task. Real-world 응용:
1. **Drug discovery**: 특정 성질의 분자 설계
2. **Material science**: 새 결정 구조
3. **Social network synthesis**: 익명화된 synthetic data
4. **Program synthesis**: Code AST 생성

Node/Graph classification 과 달리:
- **Generative** — probability distribution $p(G)$ 학습
- **Variable size** — $|V|$ 가 다름
- **Permutation invariance** — 같은 graph 의 $n!$ 개 representation

이 문서는 GraphVAE, GraphRNN, GCPN, 최신 diffusion 의 메커니즘을 정리.

---

## 📐 수학적 선행 조건

- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): VAE, RNN
- 확률: Density estimation, variational inference
- 조합론: Permutation, graph isomorphism

---

## 📖 직관적 이해

### Permutation 문제의 핵심

Graph $G$ 의 adjacency $A$ 는 node ordering 에 의존 — $n!$ 개 equivalent representation. Generation model 이 "이 중 어떤 것" 을 출력해야?

**두 접근**:

1. **Parallel (GraphVAE)**: $A$ 전체 한 번에 생성 → matching 으로 permutation 해결
2. **Sequential (GraphRNN)**: 노드 하나씩 추가 → specific ordering (BFS, canonical) 선택

### GraphVAE: Parallel + Matching

1. Encoder: $q(z | G)$ — graph → latent vector
2. Decoder: $p(A | z)$ — latent → adjacency prediction
3. Loss: reconstruction + KL
4. **문제**: reconstruction $\|A - \hat A\|$ 가 ordering-dependent
5. **해결**: Graph matching (permutation 탐색) 후 loss 계산 — $O(n!)$ 또는 heuristic

### GraphRNN: Sequential + BFS Ordering

1. Node 하나씩 추가, 각 노드에 대해 edge 결정
2. BFS ordering 으로 "canonical" ordering 선택
3. RNN (LSTM/GRU) 으로 state 관리
4. Graph-level RNN (new node 생성) + Edge-level RNN (edge 결정)

장점: tractable, 표현력 ↑.
단점: BFS 가 unique canonical 아님 (같은 graph 에 여러 BFS ordering).

### GCPN: RL for 분자

Molecular graph generation with chemistry reward:
1. Action: atom/bond 추가 또는 수정
2. State: current partial graph + chemistry features
3. Reward: drug-likeness (QED), synthetic accessibility, target affinity
4. Policy network = GNN → action distribution
5. PPO 또는 Q-learning

### Diffusion on Graphs (최신)

GDSS (Jo 2022), DiGress (Vignac 2022):
1. Forward: clean graph → noisy graph (node + edge noise)
2. Reverse: learned model denoise step-by-step
3. Samples: generate by denoising from pure noise

표현력 강, permutation invariant by design.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Graph Generation Problem

Training dataset $\{G_i\}_{i=1}^N$ — independent sample from $p^*(G)$.

**Goal**: Learn $p_\theta(G) \approx p^*(G)$. Sample $G \sim p_\theta$.

### 정의 4.2 — GraphVAE (Kipf 2016 원본 + Simonovsky 2018 세부)

**Encoder**:
$$
q_\phi(z | G) = \mathcal N(z; \mu(G), \Sigma(G))
$$

$\mu, \Sigma$ = GNN on $G$.

**Decoder**:
$$
p_\theta(A | z) = \text{Bernoulli}(\sigma(\text{MLP}(z)))
$$

Output: node count $n$, node feature $X \in \mathbb R^{n_{\max} \times d_x}$, adjacency $A \in [0, 1]^{n_{\max} \times n_{\max}}$ — fixed-size ($n_{\max}$ = max nodes).

**Loss**:
$$
\mathcal L = \mathbb E_{q_\phi} [\log p_\theta(G | z)] - \text{KL}(q_\phi \| p)
$$

Reconstruction 부분: graph matching 필요 (GED 또는 Hungarian algorithm).

### 정의 4.3 — GraphRNN (You 2018)

**BFS ordering** $\pi$: 시작 노드에서 BFS traversal. 노드 순서 $v_1, v_2, \ldots, v_n$.

**Sequential factorization**:
$$
p(G) = p(\pi(G)) \cdot |\text{Aut}(G)|
$$

(orbit 처리)

각 step $t$:
- **Node-level RNN**: state $h_t$ update
- **Edge-level RNN**: $v_t$ 의 이전 노드들 $\{v_1, \ldots, v_{t-1}\}$ 에 대한 edge 결정

$$
p(v_t \text{에 대한 edge } e_{t,j}) = f(h_t, h_{\text{edge}, j})
$$

### 정의 4.4 — GCPN (Goal-Directed Molecular Generation)

**Markov Decision Process**:
- State $S_t$: current partial molecule
- Action $a_t$: (new atom type, connection location, bond type)
- Transition: deterministic graph update
- Reward $R$: chemistry-aware score (QED, LogP, etc.)

**Policy**: GCN on current molecule → action logits.

**Algorithm**: PPO or REINFORCE.

### 정의 4.5 — DiGress (Discrete Diffusion)

**Forward process**: 각 step $t$, 확률 $\beta_t$ 로 edge/node type randomly flip.

**Reverse process**: 학습된 $p_\theta(G^{t-1} | G^t)$ — denoising.

**Training**: ELBO on forward/reverse pair.

---

## 🔬 정리와 결과

### 정리 4.1 — GraphVAE 의 Matching Complexity

**Theorem**: Exact graph reconstruction loss 는 $O(n!)$ 의 permutation 고려. 실전 heuristic:
- Hungarian algorithm: $O(n^3)$ for node matching
- Graph Edit Distance: $O(n^3)$ approximation

이는 GraphVAE 가 small graph ($n \leq 40$) 에서만 실용적 이유.

### 정리 4.2 — GraphRNN 의 BFS 유용성

**Theorem (You 2018)**: BFS ordering 은 local structure 를 보존하는 canonical. 같은 graph 의 BFS ordering 이 multiple 가능하지만, practical 다양성 줄임.

**장점**: Sequential generation 의 graph-level complexity $O(n^2)$ (max edges) — GraphVAE 의 matching 보다 tractable.

**Empirical**: GraphRNN 이 100+ node graph 도 가능 (GraphVAE 는 30-40).

### 정리 4.3 — GCPN 의 Chemistry Reward

**Drug-likeness score (QED)**:
$$
\text{QED}(m) = \exp\left(-\frac{1}{8} \sum_i d_i\right), \quad d_i = \log\left(\frac{1 + p_i}{1 - p_i}\right)
$$

where $p_i$ = 다양한 physicochemical property (MW, LogP, HBA, HBD 등).

**Reward**: $R = \alpha \cdot \text{QED} + \beta \cdot \text{SA}^{-1}$ etc.

**장점**: Target-directed generation (specific property maximization).

### 정리 4.4 — Diffusion Model 의 우위

**Empirical (DiGress, Vignac 2022)**:

| Method | MUTAG validity | QM9 validity | ZINC FCD |
|--------|----------------|--------------|----------|
| GraphVAE | 0.72 | 0.85 | 25.1 |
| GraphRNN | 0.95 | 0.99 | 8.7 |
| GCPN | 0.95 | 0.99 | 5.2 |
| **DiGress** | **0.99** | **0.99** | **3.1** |

(Fréchet ChemNet Distance — lower better)

Diffusion model 이 SOTA across all chemistry benchmarks.

### 정리 4.5 — Permutation Invariance Comparison

| Method | Perm Invariance | Handling |
|--------|-----------------|---------|
| GraphVAE | Manual (matching) | GED / Hungarian |
| GraphRNN | BFS ordering | Sequential factorization |
| GCPN | Step-wise | RL state |
| DiGress | By design | Discrete diffusion + equivariant |

Diffusion 이 가장 natural permutation 처리.

---

## 💻 구현

### 실험 1 — Simple GraphVAE (단순화)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGraphVAE(nn.Module):
    def __init__(self, n_max, d_z=32):
        super().__init__()
        self.n_max = n_max
        # Encoder (simplified: 단일 MLP on flattened A)
        self.enc_mu = nn.Linear(n_max * n_max, d_z)
        self.enc_logvar = nn.Linear(n_max * n_max, d_z)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_z, 64), nn.ReLU(),
            nn.Linear(64, n_max * n_max)
        )
    
    def encode(self, A):
        A_flat = A.view(A.size(0), -1)
        return self.enc_mu(A_flat), self.enc_logvar(A_flat)
    
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    
    def decode(self, z):
        logits = self.decoder(z)
        return logits.view(-1, self.n_max, self.n_max)
    
    def forward(self, A):
        mu, logvar = self.encode(A)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

### 실험 2 — GraphRNN (간략, BFS)

```python
class GraphRNN(nn.Module):
    def __init__(self, max_nodes=20, d_hidden=64):
        super().__init__()
        self.max_nodes = max_nodes
        # Graph-level RNN (new node generation)
        self.graph_rnn = nn.GRU(max_nodes, d_hidden, batch_first=True)
        # Edge-level RNN (edge decision)
        self.edge_rnn = nn.GRU(1, d_hidden, batch_first=True)
        # Edge prediction head
        self.edge_head = nn.Linear(d_hidden, 1)
    
    def generate(self):
        """Sample new graph (simplified)."""
        edges = torch.zeros(self.max_nodes, self.max_nodes)
        h_g = torch.zeros(1, 1, 64)
        
        for t in range(self.max_nodes):
            # Node-level: given prev edges, predict this node's edge pattern
            input_vec = edges[t-1] if t > 0 else torch.zeros(self.max_nodes)
            _, h_g = self.graph_rnn(input_vec.unsqueeze(0).unsqueeze(0), h_g)
            
            # Edge-level: for each previous node, decide edge
            h_e = torch.zeros(1, 1, 64)
            for j in range(t):
                # Predict edge (t, j)
                _, h_e = self.edge_rnn(torch.zeros(1, 1, 1), h_e)
                logit = self.edge_head(h_e[0])
                edges[t, j] = edges[j, t] = torch.sigmoid(logit).round().item()
        return edges
```

### 실험 3 — 작은 Dataset 에서 GraphVAE Training

```python
# 3-4 node 의 간단한 graph (cycle, path, star) 생성
import numpy as np

def make_simple_graphs(num_samples=1000):
    graphs = []
    for _ in range(num_samples):
        type_choice = np.random.choice(['cycle', 'path', 'star'])
        n = np.random.randint(3, 5)
        if type_choice == 'cycle':
            A = np.zeros((5, 5))  # pad to 5
            for i in range(n):
                A[i, (i+1) % n] = A[(i+1) % n, i] = 1
        elif type_choice == 'path':
            A = np.zeros((5, 5))
            for i in range(n-1):
                A[i, i+1] = A[i+1, i] = 1
        else:  # star
            A = np.zeros((5, 5))
            for i in range(1, n):
                A[0, i] = A[i, 0] = 1
        graphs.append(torch.tensor(A, dtype=torch.float32))
    return torch.stack(graphs)

data = make_simple_graphs(500)
model = SimpleGraphVAE(n_max=5, d_z=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    recon, mu, logvar = model(data)
    # BCE reconstruction (no matching — simplified)
    recon_loss = F.binary_cross_entropy_with_logits(recon, data)
    # KL
    kl = -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))
    loss = recon_loss + 0.01 * kl
    loss.backward(); optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}: loss {loss.item():.4f}')
```

### 실험 4 — Generated Samples Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx

# Sample from prior
z = torch.randn(5, 8)
samples = torch.sigmoid(model.decode(z)).detach().numpy()

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, A_gen in zip(axes, samples):
    A_bin = (A_gen > 0.5).astype(int)
    np.fill_diagonal(A_bin, 0)
    A_bin = np.maximum(A_bin, A_bin.T)   # symmetric
    G = nx.from_numpy_array(A_bin)
    nx.draw(G, ax=ax, node_color='lightblue', with_labels=True, node_size=300)
plt.suptitle('GraphVAE generated samples')
plt.show()
```

### 실험 5 — GraphRNN BFS Ordering

```python
def bfs_order(G, root=0):
    return list(nx.bfs_tree(G, source=root).nodes)

G_test = nx.karate_club_graph()
order = bfs_order(G_test)
print(f'BFS ordering: {order[:10]}...')
# GraphRNN 은 이 ordering 대로 sequential generate
```

---

## 🔗 실전 활용

### 1. Drug Discovery

- **Target-directed**: Specific disease target 에 대한 분자 설계
- **Scaffold hopping**: 기존 drug 의 새 변종
- **Lead optimization**: 기존 candidate 의 성질 개선

### 2. Material Design

- **Crystal structure generation**: Stable crystal 찾기
- **Polymer design**: 특정 mechanical property

### 3. Social Network Anonymization

- **Synthetic data generation**: Privacy-preserving social network 생성

### 4. PyG / RDKit Integration

```python
from rdkit import Chem

# SMILES → graph
mol = Chem.MolFromSmiles('CCO')   # ethanol
# ... convert to PyG Data

# Graph → SMILES (validity check)
mol_gen = graph_to_rdkit(generated_graph)
smiles = Chem.MolToSmiles(mol_gen)
```

### 5. Evaluation Metrics

- **Validity**: Chemically valid fraction
- **Uniqueness**: Unique fraction of valid
- **Novelty**: Not in training set
- **FCD (Fréchet ChemNet Distance)**: Distribution similarity
- **Diversity**: Structural diversity (Tanimoto similarity mean)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| GraphVAE matching | $O(n!)$ 또는 approximation → small graph 만 |
| GraphRNN BFS | Ordering 선택이 distribution bias |
| GCPN chemistry reward | Reward hacking 가능 |
| Diffusion 의 compute | Slow (많은 denoising step) |
| Permutation invariance | 강제 design 필요 |
| Evaluation metric | Validity ≠ usefulness, domain-specific |

---

## 📌 핵심 정리

| Method | Architecture | Permutation | Scale | SOTA |
|--------|-------------|-------------|-------|------|
| **GraphVAE** | VAE encoder-decoder | Matching | n≤40 | Legacy |
| **GraphRNN** | Sequential (RNN) | BFS ordering | n≤100 | Mid |
| **GCPN** | RL on partial graph | Step-wise | n≤50 | Drug-specific |
| **DiGress** | Discrete diffusion | By design | n≤100 | Current |
| **GDSS** | Continuous diffusion (score) | By design | n≤50 | Competitive |

**Modern recommendation**:
- Drug discovery: DiGress + chemistry-specific diffusion (MolDiff)
- General graph: DiGress or GraphGDP
- Small-scale: GraphRNN (tractable)

Permutation 처리가 graph generation 의 핵심 challenge — diffusion 이 가장 natural.

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-node graph 의 가능한 adjacency matrix 경우 수 (permutation 고려)?

<details>
<summary>해설</summary>

**3-node undirected graph**:
- Edge 가능 위치: $\binom{3}{2} = 3$ (pairs (0,1), (0,2), (1,2))
- 각 edge: 있음/없음 → $2^3 = 8$ 개 adjacency matrix

**Isomorphism classes** (permutation 제외):
1. Empty graph (no edges)
2. Single edge (0,1)-connected
3. Path (2 edges, 3-node path)
4. Triangle (3 edges)

총 4 class. 각 class 의 $n!$ adjacency representation:
- Empty: 1
- Single edge: $\binom{3}{2} = 3$ representation
- Path: $3!/2 = 3$ (두 끝 swap = same)
- Triangle: 1 (모든 permutation same)

Total: $1 + 3 + 3 + 1 = 8$ ✓ = $2^3$.

**의미**: Generation model 이 "path" 와 "triangle" 같은 equivalence class 를 학습해야지, specific adjacency matrix 를 외우면 안됨.

**GraphVAE 의 문제**: 출력 $\hat A \in \mathbb R^{3 \times 3}$ 이 $3!$ permutation 중 하나 — training 시 matching 으로 handle. Test 시 generated $\hat A$ 의 class 를 평가 (isomorphism check).

</details>

**문제 2** (심화): GraphRNN 의 BFS ordering 이 unique 하지 않음 (starting node 선택). 이것이 generation distribution 에 어떤 bias 를 야기하는가?

<details>
<summary>해설</summary>

**BFS ordering 의 비유일성**:

같은 graph 에 대해 시작 노드 $n$ 개 → $n$ 개 BFS ordering. Each ordering 이 다른 factorization $\pi(G)$.

**Training 의 impact**:

Training set 의 각 graph 가 $n$-fold duplicate (with different orderings) 로 학습. 

**Bias 1: Root node 의 degree**

High-degree node 를 root 로 선택 시 BFS tree 가 wide 하고 shallow. Low-degree root → deep tree.

Model 이 학습하는 edge 분포가 root choice 에 bias.

**Bias 2: Tree-like vs cyclic**

BFS tree 의 edges 먼저 예측, back-edges (cycle 만드는) 는 나중. 따라서 model 이 cyclic structure 에 대한 representation 이 "BFS-tree + back-edge" 형태로 factored — natural distribution 과 다를 수 있음.

**Bias 3: Symmetric graph**

Highly symmetric graph (cycle, complete) 는 모든 ordering equivalent. 단 asymmetric graph 는 BFS 가 specific structure 강조.

**Handling**:

1. **Random starting node**: Training 시 each graph 를 random root 로 sample — bias 평균화.

2. **Canonical ordering** (nauty): Unique canonical form 사용 — 결정적 but 복잡.

3. **Diffusion**: Permutation invariance by design — 이런 문제 없음.

**Empirical**: Random starting node BFS 가 GraphRNN 의 표준 — 학습된 distribution 이 true graph distribution 에 더 가까움.

**Modern 대안**: DiGress 가 BFS 의존 없이 permutation invariant generation — 이 문제 완전 해결.

</details>

**문제 3** (논문 비평): Diffusion on graphs (DiGress, GDSS) 가 기존 GraphVAE/GraphRNN 을 압도하는 이유와, 여전히 남은 challenge 는?

<details>
<summary>해설</summary>

**Diffusion 의 압도적 장점**:

1. **Permutation invariance by design**: Equivariant network 사용 → 학습·generation 모두 natural handling.

2. **Iterative refinement**: 한 번에 안 맞아도 점차 수정 — large graph 에서 robust.

3. **Expressive**: Score-based learning 이 complex distribution 학습 가능. VAE 의 Gaussian posterior 보다 flexible.

4. **Stable training**: GAN-like instability 없음, log-likelihood based.

5. **Conditional generation**: Classifier-free guidance 로 controllable generation.

**Empirical**:

- Validity: 99%+ (vs GraphVAE 72%)
- Diversity: 높음
- FCD: 낮음 (distribution match good)
- Conditional: Property-specific generation 가능

**Remaining Challenges**:

1. **Scalability**: $n \leq 100$ 노드. 큰 graph (>1000) 에서 memory / compute 폭발.
   - **Solution**: Hierarchical diffusion, patch-based

2. **Compute cost**: 많은 denoising steps (50-1000) — slow inference.
   - **Solution**: DDIM-like acceleration, one-step generation (diffusion distillation)

3. **Discrete vs continuous**: 
   - DiGress: Discrete (atom type, bond type) — chemistry 맞음 but math 복잡
   - GDSS: Continuous (relaxation) — math simple but discrete interpretation 필요

4. **3D integration**: Molecular 3D structure 가 중요 (conformer), 현재 모델은 2D graph 만. 
   - **Solution**: SE(3)-equivariant diffusion (GeoDiff, Xu 2022)

5. **Constrained generation**: Hard constraint (specific scaffold, synthetic accessibility) 어려움.
   - **Solution**: Conditional guidance, rejection sampling

6. **Large-scale KG generation**: Knowledge graph 의 schema-aware generation 미해결.

7. **Interpretability**: Generation process 가 black-box — chemist 가 이해·manipulate 하기 어려움.

**미래 방향**:

- **Foundation model for chemistry** (MolFM, ChemBERTa): Large-scale pre-training + diffusion fine-tune
- **Symbolic + neural hybrid**: SMILES-like sequence 와 graph 의 combination
- **LLM + graph diffusion**: Chemistry knowledge 를 LLM 으로 주입

따라서 diffusion 이 현재 SOTA 이지만 완성된 solution 아님 — 여전히 활발한 연구 영역. Ch7-04 의 "GNN 미래" 에서 더 자세히.

</details>

---

<div align="center">

[◀ 이전](./03-link-prediction.md) | [📚 README](../README.md) | [다음 ▶](../ch7-modern-gnn/01-graphormer.md)

</div>
