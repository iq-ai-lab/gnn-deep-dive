# 04. GNN의 Scaling과 이론적 한계

## 🎯 핵심 질문

- Cluster-GCN, GraphSAINT 가 대규모 graph 학습의 표준이 된 이유와 trade-off?
- OGB-LSC PCQM4M (3.7M 분자), OGB-Papers100M (111M 노드) 의 규모에서 GNN 이 어떻게 작동하는가?
- WL 상한이 실전 성능의 병목인가, 아니면 cosmetic 이론 한계인가? (Morris 2021 의 관점)
- LLM 시대의 GNN 역할 — GraphGPT, text-attributed graph, graph foundation model?
- 미래 GNN 의 방향: hybrid (neuro-symbolic, LLM+GNN), foundation model, ∞-layer?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GNN 이 지난 10년 beautiful 이론 + 실전 성공 이루었지만, **근본적 과제** 들이 여전:

1. **Scaling**: 10M+ 노드 graph 에서 sampling-based 가 필수 but bias
2. **이론 vs 실전의 gap**: WL 상한의 실전 영향?
3. **Foundation model**: NLP / vision 의 pre-trained 대규모 모델 에 해당하는 GNN 은?
4. **LLM 시대**: 분자 를 LLM 이 SMILES 로 처리 가능 — GNN 의 가치?

이 마지막 문서는 GNN 의 scaling, 이론적 한계 의 실전 의미, 그리고 **GNN 의 미래** 를 정리. 전체 학습의 큰 그림 을 제공.

---

## 📐 수학적 선행 조건

- 이전 문서들 전체 (Ch1~Ch7-03)
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Scaling laws

---

## 📖 직관적 이해

### Scaling 의 도전

Graph 의 특성 이 scaling 을 어렵게:
- **Non-i.i.d.**: 노드가 서로 연결 → mini-batch 독립성 없음
- **Variable degree**: Hub node 가 message explosion
- **Memory**: Full graph storage 가 $O(n + m)$ — $n = 100M$ 이면 수백 GB

**해결책 종류**:
1. **Subgraph sampling**: Cluster-GCN, GraphSAINT
2. **Historical embedding**: GNNAutoScale (Fey 2021) — 이전 embedding cache
3. **Linear attention**: Performer-style for Graphormer
4. **Distillation**: Large GNN → small GNN 의 knowledge distillation

### 이론 vs 실전

**WL 상한** (Ch4-02) 의 실전 의미:
- **강한 이론 한계**: CSL, Paley 구분 못함
- **실전 무관**: Real-world graph 의 99%+ 는 1-WL 로 distinguishable (Babai 1980)

**Morris 2021** ("Weisfeiler and Leman go Machine Learning: The Story So Far"):
- 이론적 개선 (k-WL, position-aware) 이 대부분 benchmark 에서 marginal
- Real-world 의 performance gap 은 표현력 이 아닌 **inductive bias + training data + hyperparameter** 에서 옴

### LLM 의 도전

LLM 이 SMILES (분자 의 string representation) 직접 이해 가능:
- ChatGPT-4: "Tell me the property of C1=CC=CC=C1 (benzene)"
- 정확히 answer: aromatic, $\pi$-bond 등

**GNN 의 역할**?

1. **LLM weak at**: Quantitative property prediction, 3D geometry
2. **GNN strong at**: Structural pattern, graph symmetry, numerical precision

**Hybrid**:
- GNN encoder + LLM generator (text explanation)
- LLM for text-attributed graph (TAG) — node에 rich text feature
- Graph RAG (Retrieval-Augmented Generation)

### GNN Foundation Model

NLP: BERT, GPT — massive pretraining.
Vision: CLIP, DINO.
**Graph**: ? 

Candidates:
- **GROVER** (Rong 2020): Molecular pre-training
- **MGSSL** (Rong 2022): Self-supervised
- **OFA (One for All, Liu 2023)**: Multi-task GNN
- **GraphFM** (현재 연구 중)

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Cluster-GCN (Chiang 2019)

Graph $G$ 를 METIS로 $K$ partition:
$$
V = \bigcup_{k=1}^K V_k, \quad V_i \cap V_j = \emptyset
$$

Mini-batch $b$ 에서 몇 개 partition $\{V_{k_1}, V_{k_2}, \ldots, V_{k_B}\}$ 선택, induced subgraph 위에서 학습.

### 정의 4.2 — GraphSAINT (Zeng 2020)

Subgraph sampler 로 $G_s \subset G$ 추출 (random walk, node, edge sampler). Bias correction:
$$
\hat L = \sum_{v \in V_s} \frac{L_v}{p_v}
$$

($p_v$ = node $v$ 가 sample 될 확률, bias correction for unbiased gradient)

### 정의 4.3 — GNNAutoScale (Fey 2021)

Mini-batch 마다 일부 노드 만 실제 계산, 나머지 는 historical embedding cache 사용:
$$
h^{(l+1)}_v = \sigma\left(\text{AGG}\{h^{(l)}_u : u \in N(v) \cap \text{batch}\} \cup \{\tilde h^{(l)}_u : u \in N(v) \setminus \text{batch}\}\right)
$$

($\tilde h$: historical, stored from previous iterations)

### 정의 4.4 — Text-Attributed Graph (TAG)

Node 에 text attribute $t_v$ 가 있는 graph. 
$$
X_v = \text{LM}(t_v) \in \mathbb R^d
$$

($\text{LM}$ = frozen or fine-tuned language model)

GNN 이 graph structure + LM representation 결합.

### 정의 4.5 — Graph RAG

1. Query $q$
2. **Retrieval**: $q$ 에 관련된 subgraph $G_q \subset G$ 선택 (similarity search, community detection)
3. **Augmentation**: $G_q$ 의 정보를 LLM context 에 추가
4. **Generation**: LLM 이 augmented context 로 답변

---

## 🔬 정리와 결과

### 정리 4.1 — Cluster-GCN 의 Memory Complexity

**Theorem**: $K$-partition, batch size $B$ (partition 개수) 사용 시:
$$
\text{Memory} = O\left(\frac{n}{K} \cdot B \cdot d\right)
$$

full-batch $O(n d)$ 대비 $B/K \ll 1$ factor 감소.

**Empirical**: OGB-Products ($n = 2.4M$) 에서 full-batch 불가능 ($\sim 100$ GB), Cluster-GCN 으로 8-16 GB GPU 가능.

### 정리 4.2 — WL Expressivity vs Real-world Performance

**Empirical (multiple surveys, Dwivedi 2022 Benchmark)**:

| Dataset | GIN (1-WL) | Graphormer (> 1-WL) | Gap |
|---------|-----------|---------------------|-----|
| ZINC (chemistry) | 0.20 | 0.09 | **55% better** |
| Cora | 80% | 82.5% | **3% better** |
| PCQM4M | 0.151 | 0.099 | **34% better** |
| OGB-products | 78% | 79% | **1% better** |

**관찰**: Chemistry (3D/complex structure) 에서 WL 상한 초과가 큰 이득. Citation / general 에서는 marginal.

**결론**: 표현력 상한의 실전 영향은 **task-specific** — universal statement X.

### 정리 4.3 — Graph Foundation Model 의 Challenge

**NLP foundation model 의 성공 요인**:
1. Transformer architecture (universal on sequence)
2. Massive pretraining data
3. Tokenization (universal input)
4. Scaling laws

**Graph 의 도전**:
1. **Architecture diversity**: Graph 마다 다른 structure (homogeneous, heterogeneous, spatial, temporal) — universal architecture 어려움
2. **Pretraining data scarcity**: Graph 는 text보다 적음. 수집 / alignment 어려움
3. **Tokenization problem**: Node / edge token 어떻게 정의? Subgraph as token?
4. **Scaling laws unclear**: GNN 이 parameter 증가로 monotonic 향상 X (over-smoothing)

### 정리 4.4 — LLM + GNN Synergy

**LLM strengths** (with text attribute):
- Semantic understanding
- Zero/few-shot learning
- Natural language explanation

**GNN strengths**:
- Structural precision
- Numerical accuracy
- Efficient reasoning on large graph

**Hybrid approach**:
1. **LLM encoder + GNN reasoning**: BERT-encoded text features + GNN propagation
2. **GNN + LLM decoder**: GNN output → natural language
3. **Graph RAG**: LLM + graph-based retrieval

**Empirical**: OGB-TAG (Text-Attributed), LLM-based embedding + GNN 이 baseline GNN 보다 +5~10% improvement.

### 정리 4.5 — Distillation 과 Efficiency

**Knowledge distillation** (Hinton 2015, GNN 변종):
$$
\mathcal L_{\text{KD}} = \mathcal L_{\text{task}} + \lambda \cdot \text{KL}(p_{\text{student}} \| p_{\text{teacher}})
$$

Large teacher (Graphormer, 100M params) → small student (GIN, 1M params).

**장점**:
- Inference speed ↑
- Memory ↓
- Marginal performance loss

**Empirical**: MoleculeKit (Luo 2023) — Graphormer teacher → GIN student, 95% performance, 20x speedup.

---

## 💻 구현

### 실험 1 — Cluster-GCN 구현 (PyG)

```python
try:
    from torch_geometric.loader import ClusterData, ClusterLoader
    from torch_geometric.datasets import Planetoid
    
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    
    # METIS partitioning
    cluster_data = ClusterData(data, num_parts=32, recursive=False)
    loader = ClusterLoader(cluster_data, batch_size=4, shuffle=True)
    
    for batch in loader:
        print(f'Batch: {batch.num_nodes} nodes, {batch.num_edges} edges')
        break
except (ImportError, RuntimeError):
    print('PyG / Cora not available')
```

### 실험 2 — GraphSAINT Sampler

```python
try:
    from torch_geometric.loader import GraphSAINTRandomWalkSampler
    
    sampler = GraphSAINTRandomWalkSampler(
        data,
        batch_size=256,
        walk_length=2,
        num_steps=5,
        sample_coverage=100
    )
    
    for subgraph in sampler:
        print(f'Subgraph: {subgraph.num_nodes} nodes, '
              f'{subgraph.num_edges} edges')
        # 학습 loop 에서 subgraph 사용
        break
except (ImportError, RuntimeError):
    print('PyG not available')
```

### 실험 3 — 간단한 LLM + GNN Hybrid

```python
import torch
import torch.nn as nn

class LLMGNNHybrid(nn.Module):
    def __init__(self, llm_dim=768, gnn_hidden=64, num_classes=10):
        super().__init__()
        # Pre-computed LLM embedding (예: BERT)
        self.input_proj = nn.Linear(llm_dim, gnn_hidden)
        # GNN layers
        self.gnn1 = nn.Linear(gnn_hidden, gnn_hidden)
        self.gnn2 = nn.Linear(gnn_hidden, gnn_hidden)
        # Classifier
        self.cls = nn.Linear(gnn_hidden, num_classes)
    
    def forward(self, llm_embeddings, edge_index):
        """
        llm_embeddings: [n, llm_dim] - node features from LLM
        edge_index: [2, m]
        """
        h = self.input_proj(llm_embeddings)
        # GCN-like propagation (simplified)
        src, dst = edge_index
        from torch_scatter import scatter_mean
        h = torch.relu(self.gnn1(h + scatter_mean(h[src], dst, dim=0, dim_size=h.size(0))))
        h = self.gnn2(h + scatter_mean(h[src], dst, dim=0, dim_size=h.size(0)))
        return self.cls(h)

# Usage (hypothetical):
# 1. LLM encode text attributes: llm_emb = BERT(text_per_node)
# 2. GNN propagate on citation graph
# 3. Classify papers
```

### 실험 4 — Graph RAG Simple Pipeline

```python
def graph_rag(query, graph, llm, k=5):
    """
    Retrieval-Augmented Generation on graph.
    1. Embed query → find similar nodes in graph
    2. Extract k-hop subgraph around similar nodes
    3. Serialize subgraph → LLM context
    4. Generate answer
    """
    query_emb = llm.encode(query)   # hypothetical LLM encode
    # Find top-k most similar nodes
    node_embs = graph.node_features
    similarities = query_emb @ node_embs.T
    top_nodes = similarities.argsort()[-k:]
    
    # k-hop subgraph
    subgraph_nodes = set(top_nodes.tolist())
    for _ in range(2):   # 2-hop
        new_neighbors = set()
        for v in subgraph_nodes:
            new_neighbors.update(graph.neighbors(v))
        subgraph_nodes |= new_neighbors
    
    # Serialize to text
    subgraph_text = "\n".join([f"Node {v}: {graph.text[v]}" for v in subgraph_nodes])
    prompt = f"Context:\n{subgraph_text}\n\nQuery: {query}\nAnswer:"
    
    # LLM generate
    return llm.generate(prompt)
```

### 실험 5 — Knowledge Distillation: Graphormer → GIN

```python
# Teacher: Graphormer (large)
# Student: GIN (small)

# KD loss
def kd_loss(student_logits, teacher_logits, temperature=2.0):
    T = temperature
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * T**2

# Training
# total_loss = task_loss + alpha * kd_loss
# ...
```

---

## 🔗 실전 활용

### 1. Large-Scale Graph Benchmarks

- **OGB-LSC**: 1M+ graphs (PCQM4M-LSC), 100M+ nodes (MAG240M)
- **Open Catalyst**: 150M DFT relaxations
- **AlphaFold DB**: 200M proteins

이 규모에서 sampling-based + equivariant + foundation model 이 표준.

### 2. Graph LLM Applications

- **Drug discovery**: Target 쿼리 → candidate molecule (LLM + molecular GNN)
- **Scientific literature**: Paper citation graph + GPT-4 의 하이브리드
- **Enterprise knowledge graph**: SAP, Salesforce 의 KG + LLM 결합

### 3. Production Systems

- **Google**: Graph Neural Network for YouTube recommendation
- **Facebook (Meta)**: Social graph GNN
- **Alibaba**: Taobao recommendation at trillion-scale
- **Pinterest**: PinSage 30B node

### 4. Open Research Frontiers

- **Universal graph encoder**: Any graph → universal embedding
- **Neural algorithmic reasoning**: Simulate algorithms on graph
- **Continuous-depth GNN**: Neural ODE on graph
- **Lifted / Relational ML**: Graph + logic integration

### 5. Industry Adoption Timeline

- 2017-2019: Academic (Cora, Citeseer)
- 2020-2022: Industry early (Pinterest, Alibaba)
- 2023-2024: Foundation model era begins
- 2025+: Graph foundation model 의 universal adoption 예상

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Sampling 의 unbiasedness | Minibatch dependency, corrected sampling 필요 |
| Efficient Partition (METIS) | Large graph 의 partition 자체 비쌈 |
| LLM embedding 의 quality | Domain-specific 재훈련 필요할 수도 |
| Scaling laws for graphs | GNN 은 monotonic 안 함 (over-smoothing) |
| Foundation model availability | 아직 성숙 X |
| LLM context length | 큰 subgraph 전부 fit 안 됨 |

---

## 📌 핵심 정리

**Scaling 방법**:
- Cluster-GCN: Graph partition
- GraphSAINT: Subgraph sampling + bias correction
- GNNAutoScale: Historical embedding cache

**이론 vs 실전**:
- WL 상한이 chemistry 에서 실전 bottleneck
- Citation / general 에서 marginal
- Task-specific, not universal

**미래 방향**:
1. **Foundation model**: Pretrained graph encoder
2. **LLM + GNN hybrid**: Text-attributed graph, Graph RAG
3. **Equivariant scaling**: SE(3), O(3) for physics/chemistry
4. **Efficient architectures**: Linear attention, distillation

$$\boxed{\text{GNN 의 미래} = \text{Scale} + \text{Multi-modal (text+graph)} + \text{Equivariant} + \text{Foundation Model}}$$

---

## 🤔 생각해볼 문제

**문제 1** (기초): Cluster-GCN 의 inter-cluster edge 손실 문제. 이를 완화하는 방법?

<details>
<summary>해설</summary>

**문제**: Cluster $V_k$ 의 induced subgraph 를 사용 → $V_k$ 와 다른 cluster 사이 edge 가 training 시 고려 안 됨.

**결과**:
- Information loss — cluster boundary 에서 noise
- Over-smoothing 패턴 변화 (cluster 내 만 smoothing)

**완화 방법**:

1. **Stochastic multi-cluster (Chiang 2019)**:
- 매 epoch 다른 partition 사용
- Inter-cluster edge 가 다른 iteration 에서 포함
- 평균적으로 모든 edge 가 학습에 기여

2. **Overlap clusters**:
- Cluster boundary 주변 노드 를 여러 cluster 에 duplicate
- Inter-cluster edge 가 contextually 포함
- Memory overhead + but information loss 완화

3. **Boundary message passing**:
- Cluster 간 경계 edge 를 별도 처리
- Cluster-GCN-L (extension): inter-cluster 의 subset 도 매 step 샘플

4. **Multi-level GNN**:
- First: Cluster-level GNN (coarse)
- Second: Within-cluster GNN (fine)
- Hierarchical: DiffPool-like

5. **Historical embedding (GNNAutoScale)**:
- Inter-cluster 이웃 의 cached embedding 사용
- Exact message passing approximation

**Empirical**: Stochastic multi-cluster 가 단순하면서 효과적 — Cluster-GCN 표준 practice.

**Trade-off**: Memory vs accuracy. Large graph 일수록 sampling approximation 이 이득.

</details>

**문제 2** (심화): LLM 이 SMILES 를 직접 처리 가능하므로 분자 GNN 이 필요 없다는 주장이 있다. 이에 대한 반박?

<details>
<summary>해설</summary>

**LLM only 의 한계**:

1. **Quantitative accuracy**:
   - LLM: "benzene 은 aromatic" (qualitative)
   - GNN: HOMO-LUMO gap = 4.72 eV (quantitative precision)
   - Chemistry benchmark: LLM 의 numerical error >> GNN

2. **Graph structure precision**:
   - Isomorphism: Same structure 의 SMILES variation 처리. LLM 이 SMILES 문자열 의 변형에 민감.
   - GNN 은 isomorphism-invariant by design.

3. **3D geometry**:
   - SMILES 는 2D structure (connectivity)만. 3D conformer 정보 X.
   - GNN (EGNN) 이 3D coordinate 를 equivariant 하게 처리.

4. **Large-scale efficiency**:
   - Drug screening: 10B candidate molecules
   - LLM forward pass: $O(n^2)$ in molecule size + expensive tokenization
   - GNN: $O(m d)$ — 훨씬 빠름

5. **Symbolic precision**:
   - Functional group (OH, NH2, aromatic ring) 인식: LLM 은 context-based (fuzzy), GNN 은 explicit graph match.

6. **Small data regime**:
   - Drug discovery: 특정 target 에 대한 experimental data 는 수백 개
   - GNN 의 inductive bias → better generalization
   - LLM fine-tune 이 작은 데이터에 overfit 위험

**Hybrid 의 강점**:

- LLM: 분자 description 해석, 일반 지식
- GNN: Precision prediction, structural reasoning
- 둘 combined: ChemBERTa + GNN, LLaMol + EGNN

**Examples of hybrid success**:

- **Chemformer** (2022): SMILES-BERT + molecular property prediction
- **MoLFormer** (2022): 1B molecule pretraining + GNN downstream
- **GALACTICA** (2022): LLM for science + molecular graph understanding

**Modern consensus**:

"**LLM for general understanding, GNN for precision**". Hybrid 가 현재 SOTA.

**5년 후 예상**: Graph-native foundation model — LLM 과 GNN 의 완전 통합. 단 GNN 의 core mechanisms (equivariance, multiset aggregation) 은 반드시 포함될 것.

</details>

**문제 3** (논문 비평): "Graph Foundation Model" 의 가능성과 현재 bottleneck 을 NLP 의 BERT/GPT 와 비교 분석하라.

<details>
<summary>해설</summary>

**NLP foundation model 의 성공 요인**:

1. **Universal architecture**: Transformer (unified sequence processing)
2. **Massive pretraining**: Internet-scale text (100B+ tokens)
3. **Unified tokenization**: BPE, SentencePiece
4. **Self-supervised objectives**: MLM (BERT), causal LM (GPT)
5. **Scaling laws**: Parameter / data / compute 의 predictable scaling
6. **Transfer learning**: Single model → many tasks

**Graph 에서의 도전**:

**1. Universal Architecture?**

NLP: Transformer works universally.
Graph: 
- Chemistry: EGNN, Graphormer
- Social: GCN, GAT
- Heterogeneous: R-GCN, HGT
- 각 subdomain 이 다른 architecture optimal

**Challenge**: Single GNN 이 all graph types 를 cover?

**2. Pretraining Data**:

NLP: 100B+ tokens (Common Crawl, Wikipedia)
Graph: 
- Molecular: 1B (ZINC 1.4B molecules, approaching text-scale)
- Social: Billions of nodes but privacy issues
- Scientific KG: Millions of entities

**Gap**: Graph pretraining data 가 훨씬 작음, 또한 diversity 부족.

**3. Tokenization / Representation**:

NLP: Text token universal
Graph: 
- Node as token? (most common)
- Edge as token? (transformer on edges)
- Subgraph as token? (hierarchical)
- Random walk as sequence? (DeepWalk)

**Challenge**: "Universal graph token" 없음.

**4. Self-supervised Objectives**:

NLP MLM: Predict masked word
Graph candidates:
- **Masked node/edge prediction** (GraphMAE, 2022)
- **Contrastive learning** (GraphCL, GCC)
- **Graph completion** (GPT-GNN)
- **Generative** (DiGress pretraining)

**Status**: No single objective 가 NLP MLM 수준으로 universal.

**5. Scaling Laws**:

NLP: Predictable loss vs parameter/data/compute
GNN: 
- Over-smoothing 가 scaling 방해
- Architecture-specific
- **Graph foundation model scaling 연구 미성숙**

**6. Transfer Learning**:

NLP: BERT → QA, classification, generation
Graph: 
- Chemistry GNN → specific property prediction (OK)
- Cross-domain transfer (chemistry → social) 어려움
- **Domain adaptation 필요**

**현재 Graph Foundation Model 후보**:

- **GROVER** (Rong 2020): Molecular, 1B parameters
- **OFA (Liu 2023)**: Multi-task GNN, cross-dataset transfer
- **GraphFM** (연구중): Universal graph encoder
- **LLaMol, MoLFormer** (2023): Chemistry-specific FM

**Bottleneck**:
1. Data scale (text 의 1/1000)
2. Architecture universality
3. Task diversity
4. Compute resources (scientific 연구 limit)

**5-10년 전망**:

- **Chemistry-specific FM**: 2-3년 내 maturation
- **General graph FM**: 5-10년 (NLP 가 2017-2019 GPT 시기, Graph 는 2022-2024 초기 단계)
- **LLM-Graph merged FM**: 가능성 있음 — Graph 의 structure + LLM 의 semantic

따라서 GNN 이 LLM 에 흡수될 risk 보다, **LLM + GNN = multi-modal foundation model** 의 synergy 가 더 가능한 미래.

**마지막 insight**: GNN 연구는 NLP Transformer 의 2017 Pre-BERT 시점과 유사. 우리가 "ChatGPT for graphs" 를 보는 데 5-10년 더. 하지만 path 는 clear — scale + architecture + data + compute.

이 레포의 33개 문서 학습 완료 시 readers 가 이 path 의 forefront 에서 연구·개발 가능. GNN deep dive 의 목적.

</details>

---

<div align="center">

[◀ 이전](./03-equivariant-gnn.md) | [📚 README](../README.md) | [🎉 완주 축하합니다! 🎉]

</div>
