# Fractal-Dendritic Network (FDN) - Technical Specifications
**Version:** 1.0 (Prototype Phase)
**System:** Hierarchical Transformer Swarm

## 1. Base Configuration (The "Mother" Node)
The Root Node ($N_0$) serves as the architectural anchor. All child nodes derive their dimensions relative to this configuration.

| Parameter | Symbol | Prototype Value | Production Target |
| :--- | :--- | :--- | :--- |
| **Embedding Dimension** | $d_{model}$ | 512 | 4096 |
| **Attention Heads** | $h$ | 8 | 32 |
| **Layers (Depth)** | $L$ | 6 | 32 |
| **Feed-Forward Dim** | $d_{ff}$ | 2048 ($4 \times d_{model}$) | 11008 |
| **Context Window** | $T_{ctx}$ | 512 | 8192 |
| **Vocab Size** | $V$ | 32,000 | 128,000 |

---

## 2. Dimensional Scaling Laws (The "Mitosis" Logic)
When a node spawns a child, the child's architecture is strictly determined by the Decay Rate ($\lambda$).

### Global Constants
- **Scaling Factor ($\lambda$):** 0.5 (Aggressive reduction for speed)
- **Minimum Floor:** $d_{min}=64, h_{min}=2, L_{min}=2$

### Scaling Formulae
Given a parent configuration $C_{parent}$, the child configuration $C_{child}$ is calculated as:

1.  **Hidden Size (Width):**
    $$d_{child} = \lfloor d_{parent} \times \lambda \rfloor$$
    *Constraint:* $d_{child}$ must be divisible by $h_{child}$.

2.  **Attention Heads:**
    $$h_{child} = \max(2, \lfloor h_{parent} \times \lambda \rfloor)$$

3.  **Layers (Depth):**
    $$L_{child} = \max(2, \lfloor L_{parent} \times 0.75 \rfloor)$$
    *Note:* Depth decays slower than width ($\lambda_{depth} \approx 0.75$) to preserve reasoning capabilities while reducing compute.

4.  **Intermediate Size:**
    $$d_{ff\_child} = 4 \times d_{child}$$

---

## 3. Weight Inheritance Protocols (The "Grafting")
Child nodes are initialized by logically "slicing" the pre-trained weights of the parent. This acts as a warm-start mechanism.

### Slicing Logic (Tensor Operations)
- **Linear Projections ($W_Q, W_K, W_V, W_O$):**
    Slice the top-left submatrix.
    `child.weight = parent.weight[:child_out, :child_in]`

- **Feed-Forward Network ($W_{up}, W_{down}$):**
    Slice the corresponding dimensions.
    `child.w_up = parent.w_up[:child_ff, :child_model]`

- **Embeddings ($W_{emb}$):**
    The Vocabulary rows ($V$) remain constant. Only the embedding dimension ($d_{model}$) is sliced.
    `child.emb = parent.emb[:, :child_model]`

- **Normalization Layers (RMSNorm/LayerNorm):**
    Slice the gain/bias vectors.
    `child.scale = parent.scale[:child_model]`

---

## 4. Distillation & Training Constraints
Child nodes are not trained from scratch; they are distilled using the Parent as a teacher.

### Loss Function
$$L_{total} = \alpha L_{task} + (1 - \alpha) L_{distill}$$

- **$\alpha$ (Task Weight):** 0.5
- **Temperature ($T$):** 2.0 (Softens the teacher's logits)
- **Hard Target ($L_{task}$):** CrossEntropy on the specific domain data (e.g., Hindi text).
- **Soft Target ($L_{distill}$):** KL-Divergence between Child Logits and Parent Logits.

---

## 5. Hardware Constraints (Mac M1 Optimization)
To prevent OOM (Out of Memory) errors during the prototype phase:

1.  **Precision:** Force `float32` (MPS is sometimes unstable with `float16`).
2.  **Device Map:**
    - **Router:** CPU (Low latency, small model)
    - **Active Node:** MPS (Metal Performance Shaders)
    - **Inactive Nodes:** Disk (Pickled State Dicts)
3.  **Cleanup:** Explicitly call `del model` and `gc.collect()` after every `bud()` operation.