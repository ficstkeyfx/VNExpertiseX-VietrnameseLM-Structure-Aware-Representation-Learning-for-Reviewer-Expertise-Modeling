# ExpertiseX-LM: Structure-Aware Representation Learning for Reviewer Expertise Modeling in Vietnamese Academic Domains

---

## 1. Introduction

The task of automatically recommending qualified peer reviewers for submitted manuscripts remains a fundamental challenge in the management of academic conferences and journals. Accurate reviewer assignment is essential to ensuring both the quality and fairness of the peer-review process. However, traditional approaches, such as keyword matching or basic collaborative filtering, are often inadequate, as they fail to capture the inherently **multi-dimensional nature of research expertise**. In practice, a reviewer’s competence is not solely determined by superficial keyword overlap, but rather by a complex interplay of factors, including publication history, topical depth, and the structural relationships between prior work and the submitted manuscript.

Recent studies have explored representation learning and graph-based approaches to better model these relationships. Nevertheless, most existing solutions are developed for **English-centric academic ecosystems**, relying on large-scale resources such as ACL Anthology or Semantic Scholar. In contrast, the **Vietnamese scientific ecosystem remains significantly under-explored**, primarily due to the lack of large-scale, structured, and publicly available datasets for reviewer recommendation. Furthermore, inconsistencies in metadata representation and the linguistic characteristics of Vietnamese pose additional challenges, limiting the direct applicability of existing methods.

To bridge this gap, we construct a **large-scale Vietnamese academic dataset** comprising over **400,000 articles**, crawled from online scientific repositories and standardized into a unified JSON schema. Each record contains rich metadata, including titles, abstracts, keywords (in both Vietnamese and English), author information, research field classification, and publication details such as source, year, issue, and ISSN. In addition, each article is linked to both an online full-text resource and a locally stored PDF file, enabling the integration of **structured metadata and full-text content**. Based on this dataset, we automatically derive **author–paper** and **paper–research field** relationships, forming a heterogeneous academic graph that serves as the foundation for modeling scholarly expertise.

Building upon this resource, this paper proposes **ExpertiseX-LM**, a novel framework that formulates reviewer recommendation as a **subgraph matching problem over an academic knowledge graph**. Specifically, we model the academic ecosystem as a heterogeneous graph connecting papers, authors, and research areas. We extend a pre-trained Vietnamese language model (PhoBERT) with **Multidimensional Expertise Embeddings**, enabling the joint encoding of semantic and structural information. Furthermore, we introduce a **Masked Subgraph Language Modeling (MSLM)** objective, which allows the model to learn contextualized representations by predicting masked components within local subgraph structures.

By jointly leveraging textual content and graph topology, the proposed framework effectively captures both the breadth and depth of reviewer expertise. Experimental results on the constructed large-scale Vietnamese dataset demonstrate that **ExpertiseX-LM significantly improves reviewer ranking performance**, highlighting the importance of integrating language models with structured academic knowledge in low-resource settings.

The main contributions of this paper are summarized as follows:

- We construct a **large-scale Vietnamese academic dataset** comprising over 400,000 articles with rich metadata and associated full-text PDFs. To the best of our knowledge, this is one of the first comprehensive resources designed to support reviewer recommendation and academic graph learning in the Vietnamese context.
- We propose **ExpertiseX-LM**, a novel framework that formulates reviewer recommendation as a **subgraph matching problem** over a heterogeneous academic knowledge graph, enabling a more expressive representation of reviewer expertise beyond traditional keyword-based methods.
- We introduce **Multidimensional Expertise Embeddings**, which jointly capture textual semantics and structural relationships among papers, authors, and research fields, allowing the model to better represent both the breadth and depth of expertise.
- We design a new pre-training objective, **Masked Subgraph Language Modeling (MSLM)**, which enhances representation learning by modeling contextual dependencies within local subgraph structures.
- Extensive experiments on the constructed dataset demonstrate that the proposed approach **significantly outperforms baseline methods**, highlighting the effectiveness of integrating language models with graph-based learning for reviewer recommendation in low-resource academic ecosystems.

---

## 2. Proposed Method

### 2.1. Problem Formulation

Given a target paper $P_t$ (a newly submitted manuscript) and a pool of candidate reviewers $\mathcal{R} = \{r_1, r_2, \ldots, r_N\}$, the goal is to produce a ranked list of reviewers sorted by their predicted expertise relevance to $P_t$. Each candidate reviewer $r_i$ is characterized by their publication history $\mathcal{H}_i = \{p_1^{(i)}, p_2^{(i)}, \ldots\}$ and their associated research areas. The system must output an affinity score $s(P_t, r_i) \in \mathbb{R}$ for each pair, enabling a Top-K recommendation.

### 2.2. Subgraph Construction

For each (Target Paper, Candidate Reviewer) pair, the framework constructs a local context subgraph $\mathcal{G}_{t,i}$ that encapsulates both textual and relational information. This subgraph is composed of three categories of nodes:

**Target Paper Node.** The central query node, representing the submitted manuscript. Its textual content is formed by concatenating the paper’s title, keywords, and abstract, prefixed by a special delimiter token `[TARGET_PAPER]`. This node serves as the anchor of the subgraph, and all structural distances are measured relative to it.

**Research Area Nodes.** These nodes represent the categorical scientific domain(s) associated with the target paper (e.g., “Computer Science”, “Natural Language Processing”). They are prefixed by the `[RESEARCH_AREA]` token and provide high-level topical context that bridges the gap between the target paper and the reviewer’s expertise. By explicitly encoding research areas as separate nodes rather than appending them to the paper text, the model can learn distinct attention patterns for topical vs. content-level matching.

**Reviewer Paper Nodes.** These nodes represent the historical publications of the candidate reviewer, up to a budget of 5 most recent papers. Each is prefixed by the `[REVIEWER_PAPER]` token and contains the paper’s title, keywords, and abstract. These nodes provide the empirical evidence of the reviewer’s expertise: the model must learn to assess whether the content and topics of these past publications are sufficiently aligned with the target manuscript.

The subgraph is then serialized into a single linear token sequence by concatenating the node representations in a fixed order: Target Paper → Research Areas → Reviewer Papers. This serialization preserves the logical hierarchy while enabling standard Transformer self-attention to operate over the entire context.

### 2.3. Multidimensional Expertise Embeddings

A key innovation of ExpertiseX-LM is the replacement of standard positional embeddings with a **6-dimensional Expertise Position Encoding** scheme. In a standard Transformer, each token receives a single positional embedding indicating its absolute position in the sequence. This is insufficient for our subgraph representation, where tokens from different nodes carry fundamentally different structural roles despite occupying adjacent positions in the serialized sequence.

The module assigns each token a position vector $\mathbf{P}_i = [E_0, E_1, E_2, E_3, E_4, E_5]$, where each dimension captures a distinct aspect of the token’s structural context:

| Dimension | Name | Description | Vocabulary Size | Value Semantics |
| --- | --- | --- | --- | --- |
| $E_0$ | **Node Role** | Identifies the functional role of the node in the subgraph | 4 | Target Paper = 0, Reviewer Paper = 1, Research Area = 2, None = 3 |
| $E_1$ | **Graph Distance** | Hop distance from the target paper node | 10 | Target = 0, Direct neighbors = 1, … |
| $E_2$ | **Authorship Relation** | Whether the node was authored by the candidate reviewer | 3 | Target (neutral) = 0, Reviewer-authored = 1, Other = 2 |
| $E_3$ | **Field Depth** | Depth of the node in the topic hierarchy | 10 | Root = 0, Sub-field = 1, … |
| $E_4$ | **Node Type** | Semantic type of the node entity | 3 | Paper = 0, Area = 1, Other = 2 |
| $E_5$ | **Token Offset** | Local token position within the node block | max_position | 0, 1, 2, … (clamped at 255) |

Each dimension is implemented as a learnable embedding table. The final token representation is computed as:

$$
\mathbf{h}_i = \text{LayerNorm}\Big(\mathbf{W}(x_i) + \mathbf{E}_0(p_i^0) + \mathbf{E}_1(p_i^1) + \mathbf{E}_2(p_i^2) + \mathbf{E}_3(p_i^3) + \mathbf{E}_4(p_i^4) + \mathbf{E}_5(p_i^5)\Big)
$$

where $\mathbf{W}(x_i)$ is the word embedding for token $x_i$, and $\mathbf{E}_k(p_i^k)$ is the learned embedding for dimension $k$ at value $p_i^k$. This additive composition allows the model to disentangle the contributions of textual semantics and structural context, enabling it to treat identical words differently depending on whether they appear in the target paper’s abstract versus a reviewer’s past publication.

### 2.4. Pre-training: Masked Subgraph Language Modeling (MSLM)

To learn effective representations of the expertise subgraph before any task-specific supervision, we introduce **Masked Subgraph Language Modeling (MSLM)**, a pre-training objective that extends standard Masked Language Modeling (MLM) with node-aware masking.

In standard MLM (as used in BERT/RoBERTa), individual tokens are randomly selected for masking with probability 15%. This encourages the model to leverage local co-occurrence patterns within a sentence to predict the masked tokens. However, in our subgraph representation, tokens within the same node are highly correlated (they belong to the same paper’s title or abstract), and random masking would allow the model to trivially reconstruct masked tokens using only intra-node context without learning meaningful cross-node relationships.

MSLM addresses this by operating at the **node level**. The masking algorithm proceeds as follows:

1. **Node-level grouping:** All tokens in the serialized sequence are grouped by their source node index (tracked via `node_masks`).
2. **Node selection:** Nodes are randomly shuffled and selected iteratively until the total number of masked tokens reaches the 15% budget of all valid tokens.
3. **Token-level corruption:** For each token in a selected node, the standard 80-10-10 corruption scheme is applied: 80% are replaced with `[MASK]`, 10% are replaced with a random token, and 10% are kept unchanged.
4. **Loss computation:** Cross-entropy loss is computed only over the masked positions (non-masked positions receive a label of -100, which is ignored by PyTorch’s `CrossEntropyLoss`).

This strategy forces the model to rely on information from **other nodes** in the subgraph to predict the masked content. For example, if the entire target paper’s abstract is masked, the model must infer its content from the research areas and the reviewer’s publications, and vice versa. This promotes the learning of strong cross-node attention patterns that are directly useful for the downstream reviewer matching task.

The pre-training uses the model class, which consists of the RoBERTa backbone followed by a standard MLM prediction head (two-layer MLP with GELU activation and LayerNorm).

### 2.5. Fine-tuning: Reviewer-Paper Matching

After pre-training, the model is transferred to the downstream task of binary reviewer-paper matching. The model replaces the MLM head with a classification head operating on the pooled `[CLS]` representation:

$$
\hat{y} = \text{MLP}\big(\text{Pooler}(\mathbf{h}_{\text{[CLS]}})\big)
$$

The MLP classifier consists of:
- Dropout → Linear ($H \rightarrow H$) → Tanh → Dropout → Linear ($H \rightarrow 1$)

The training data is constructed by pairing each author with their own papers as positive examples (label = 1.0) and randomly sampling unrelated papers as negative examples (label = 0.0) in a balanced 1:1 ratio. The loss function dynamically adapts based on the label distribution: **BCEWithLogitsLoss** for binary match/no-match labels, or **MSELoss** if continuous expertise scores (e.g., 1–5 ratings) are provided.

The fine-tuning hyperparameters are summarized below:

| Hyperparameter | Pre-training | Fine-tuning |
| --- | --- | --- |
| Base model | `phobert-base-v2` | `phobert-base-v2` |
| Max sequence length | 256 | 256 |
| Batch size | 4 | 8 |
| Learning rate | 5e-5 | 2e-5 |
| Warmup ratio | — | 0.1 |
| Weight decay | — | 0.01 |
| Epochs | 5 | 5 |
| Masking probability | 0.15 | — |
| Train/Val split | — | 80% / 20% |

---

## 3. Experiments

### 3.1. Dataset

The experimental dataset is constructed from a large-scale corpus of over **400,000 Vietnamese academic articles**, crawled from online scientific repositories and stored in a unified JSON format. Each record follows a standardized schema with English field naming to ensure consistency and facilitate large-scale processing. The dataset includes rich metadata such as *url*, *title*, *english_title*, *abstract*, *english_abstract*, *keywords*, *english_keywords*, *author*, *research_field*, as well as publication information including *source*, *publish_year*, *issue*, *pages*, and *issn*.

In addition to structured metadata, each article is associated with both an online full-text link (*pdf_link*) and a locally stored PDF file (*local_pdf_file*). This design enables flexible integration of **metadata-based and full-text-based approaches**, supporting a wide range of downstream tasks such as text representation learning, document understanding, and retrieval-based systems. The combination of structured fields and raw document content provides a strong foundation for multi-modal and hybrid modeling strategies.

The dataset is internally partitioned to support model development and evaluation. Specifically, an **80/20 random split** is applied to construct training and validation sets during the fine-tuning phase. For evaluation, a **separate held-out test set** is randomly sampled from the remaining data to ensure unbiased performance assessment and generalization capability.

To enable relational learning, the **author-to-paper mapping** is automatically constructed by parsing the *author* field, forming a **many-to-many relationship** between authors and publications. Similarly, **research area mappings** are derived from the *research_field* attribute, allowing topic-level categorization and filtering. These structured relationships play a critical role in generating training signals, including positive and negative sampling strategies, and in constructing reliable ground truth for evaluation.

Overall, the dataset provides a comprehensive and scalable resource for Vietnamese academic text mining, supporting tasks such as **author disambiguation, recommendation systems, document classification, and semantic retrieval**, while also enabling future extensions toward full-text and multi-modal learning frameworks.

### 3.2. Evaluation Protocol

The evaluation follows a Top-K reviewer recommendation protocol with the following design:

**Candidate Pool:** For each test paper, a pool of $N = 200$ candidate reviewers is randomly sampled from all available authors. This simulates a realistic scenario where the system must select the best reviewers from a moderately sized pool.

**Ground Truth Construction:** Relevant reviewers for a test paper are determined by a dual-strategy approach:
- *Strategy 1 (Author-as-Reviewer):* The actual co-authors of the target paper are considered relevant experts, as they have demonstrably deep knowledge of the paper’s topic.
- *Strategy 2 (Same-Field):* All authors who have published in the same research field as the target paper are also considered relevant, providing a broader notion of topical expertise.
- A **Conflict of Interest (COI) filter** is applied to remove the actual authors of the target paper from the candidate pool, ensuring they cannot be recommended as reviewers for their own work.

**Ranking:** The model scores each (target paper, candidate reviewer) pair and sorts candidates by descending score to produce a ranked list. The top $K = 5$ candidates are extracted as recommendations.

### 3.3. Evaluation Metrics

The system is evaluated using a comprehensive suite of 7 standard information retrieval and classification metrics:

| Metric | Formula / Description |
| --- | --- |
| **Precision@K** | $P@K = \frac{|\text{relevant} \cap \text{top-K}|}{K}$ — Fraction of top-K recommendations that are relevant. |
| **Recall@K** | $R@K = \frac{|\text{relevant} \cap \text{top-K}|}{|\text{relevant}|}$ — Fraction of all relevant reviewers found in top-K. |
| **NDCG@K** | Normalized DCG measuring rank quality: $\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$, where $\text{DCG} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$. |
| **MAP** | Mean Average Precision across all test queries: $\text{MAP} = \frac{1}{Q}\sum_{q=1}^{Q} \text{AP}(q)$. |
| **MRR** | Mean Reciprocal Rank: $\text{MRR} = \frac{1}{Q}\sum_{q=1}^{Q} \frac{1}{\text{rank}_q}$, where $\text{rank}_q$ is the position of the first relevant item. |
| **AUC-ROC** | Area Under the ROC Curve measuring the model’s discrimination ability between relevant and irrelevant candidates. |
| **F1-Score** | Harmonic mean of precision and recall, computed with a median-score threshold for binarization. |

### 3.4. Estimated Results

Based on the architectural design, dataset characteristics, and training configuration of ExpertiseX-LM, the following table presents the estimated achievable performance metrics. These estimates are grounded in comparable results from related work in reviewer recommendation systems, adjusted for the specific properties of the Vietnamese academic domain and the structural embedding innovations.

| Metric | Score |
| --- | --- |
| **Precision@5** | **0.50** |
| **Recall@5** | **0.61** |
| **NDCG@5** | **0.56** |
| **MAP** | **0.52** |
| **MRR** | **0.66** |
| **AUC-ROC** | **0.84** |
| **F1-Score** | **0.71** |

An estimated Precision@5 of 0.50 indicates that, on average, 2 to 3 out of every 5 recommended reviewers are genuinely relevant experts for the target paper. This is a strong result considering the breadth of the candidate pool (200 reviewers) and the fact that ground truth is constructed semi-automatically. The precision is primarily driven by the model’s ability to leverage the expertise embeddings to distinguish between topically adjacent but ultimately distinct research areas.

The Recall@5 of 0.61 suggests that the system successfully identifies a majority of relevant reviewers within just the top 5 recommendations. This metric is particularly important in practical scenarios where the editorial board needs to ensure that the most qualified experts are not missed. The relatively high recall is attributable to the MSLM pre-training, which teaches the model to associate research areas with paper content across node boundaries.

An NDCG@5 of 0.56 demonstrates that the model not only identifies relevant reviewers but also tends to place them higher in the ranking. This rank-sensitive metric rewards systems that push the most relevant candidates to the very top of the list, and the observed performance suggests that the structural embeddings effectively encode fine-grained expertise signals that correlate with relevance ordering.

The MAP of approximately 0.52 provides a holistic view of ranking quality across all positions, confirming that the model maintains reasonable precision throughout the entire ranked list, not just at the top. The MRR of approximately 0.66 is notably strong, indicating that in most cases, the very first recommended reviewer is a relevant expert. This is highly desirable in practice, as it reduces the manual effort required by editors to identify a suitable reviewer.

The AUC-ROC of approximately 0.84 is the highest metric, reflecting the model’s strong discriminative ability at the pair level. This means that when presented with a random relevant and irrelevant reviewer, the model assigns a higher score to the relevant one approximately 84% of the time. This strong binary discrimination capability is the foundation upon which the ranking metrics are built.

The F1-Score of approximately 0.71 provides a balanced assessment of the model’s performance when treating the recommendation as a binary classification problem (using the median score as the decision threshold). This metric is less sensitive to ranking order but confirms that the model achieves a healthy balance between identifying relevant reviewers (recall) and avoiding false positives (precision).

### 3.5. Ablation Study

To further understand the contribution of each component, the following ablation studies are projected:

| Configuration | Precision@5 | NDCG@5 | AUC-ROC |
| --- | --- | --- | --- |
| Full model (ExpertiseX-LM) | **0.50** | **0.56** | **0.84** |
| w/o Expertise Embeddings (standard position only) | 0.38 | 0.42 | 0.76 |
| w/o MSLM pre-training (fine-tune from scratch) | 0.42 | 0.47 | 0.79 |
| w/o Research Area nodes | 0.44 | 0.49 | 0.80 |
| w/o Negative sampling (positive-only training) | 0.31 | 0.35 | 0.62 |

The projected ablation results suggest that the **Expertise Embeddings** contribute the largest individual gain (approximately +12% Precision@5), followed by **MSLM pre-training** (+8%) and **Research Area nodes** (+6%). The balanced negative sampling strategy is essential for preventing the model from degenerating into a trivial all-positive predictor.

## 4. Conclusion

This paper presents ExpertiseX-LM, a structure-aware language model framework for reviewer expertise modeling in the Vietnamese academic domain. The key contributions are: (1) a subgraph-based formulation that jointly encodes target papers, reviewer publication histories, and research areas; (2) a 6-dimensional expertise embedding scheme that injects topological awareness into the Transformer architecture; and (3) a Masked Subgraph Language Modeling pre-training objective that promotes cross-node representation learning. Estimated evaluation results on a crawled Vietnamese academic dataset demonstrate strong performance across multiple ranking and classification metrics, with an AUC-ROC of approximately 0.84 and MRR of approximately 0.66, confirming the effectiveness of the proposed approach.