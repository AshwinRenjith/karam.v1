### 1. `api_contracts.md` (The "Interface" Rules)

```markdown
# Internal API Contracts & Interface Definitions
**Version:** 1.0
**Protocol:** Strict Typing (Python 3.10+)

## 1. The Genesis Node (`FractalTransformer`)
**Class Path:** `src.models.fractal_transformer.FractalTransformer`

### Inputs (`forward`)
- **`tokens`**: `torch.LongTensor`
  - Shape: `(batch_size, seq_len)`
  - Constraints: `seq_len <= config.max_seq_len`
- **`targets`** (Optional): `torch.LongTensor`
  - Shape: `(batch_size, seq_len)`
  - Used for training/distillation only.

### Outputs (`forward`)
- **`logits`**: `torch.FloatTensor`
  - Shape: `(batch_size, seq_len, vocab_size)`
- **`loss`** (Optional): `torch.FloatTensor`
  - Scalar value. Only returned if `targets` is provided.

### Method: `bud(specialized_data_loader)`
**Purpose:** Spawns a smaller child node from the current instance.
- **Input:** `specialized_data_loader` (Iterable of `(tokens, targets)`)
- **Returns:** `FractalTransformer` (New Child Instance)
- **Contract:**
  - The Child MUST be strictly smaller than `self`.
  - The Parent (`self`) weights MUST remain frozen/unchanged.
  - The Child MUST be returned in `train()` mode.

---

## 2. The Semantic Router (`SemanticRouter`)
**Class Path:** `src.router.semantic_router.SemanticRouter`

### Method: `route(text_input)`
- **Input:** `text_input` (str) or `List[str]`
- **Output:** `RouteResult` (NamedTuple)
  - `node_id`: `str` (e.g., "node_physics_v1")
  - `confidence_score`: `float` (0.0 to 1.0)
  - `routing_path`: `List[str]` (e.g., `["root", "science", "physics"]`)

### Routing Logic
- **Threshold:** `0.75` (Default)
- **Fallback:** If `confidence_score < 0.75`, return `node_id="root_node"`.

---

## 3. The Swarm Registry (`NodeRegistry`)
**Class Path:** `src.registry.node_registry.NodeRegistry`

### Storage Format (JSON/Dict)
**Key:** `node_id` (Unique String, e.g., "science_physics_v1")

**Value Schema:**
```json
{
  "parent_id": "science_v1",       // ID of the node it sprouted from
  "depth": 2,                      // Hierarchy level (Root=0)
  "centroid_vector": [0.12, ...],  // 384-dim vector (List[float])
  "file_path": "./weights/science_physics_v1.pt", // Relative path
  "config": {                      // Snapshot of architecture
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 4
  },
  "status": "active"               // Enum: "active", "frozen", "archived"
}

```

### Method: `register_node(node_id, metadata)`

* **Contract:**
* Validates that `parent_id` exists.
* Validates that `file_path` is accessible.
* Auto-saves registry to disk upon success.



```

