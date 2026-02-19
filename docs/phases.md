This is the **Master Blueprint**. We will build this architecture in **5 Phases**, ordered strictly by dependency. You cannot build the roof (Inference) before the foundation (Genesis Node).

Here is the complete engineering breakdown of the **Fractal-Dendritic Network (FDN)**.

---

### **Phase 1: The "Genesis" Node (The Fractal Transformer)**

**Status:** *The Foundation*
**Goal:** Build a single neural network class that can be instantiated at any size (Giant Mother or Tiny Child) and supports "weight inheritance."

**Sub-Components:**

1. **`FractalConfig`:** A configuration object that defines the "DNA" (layers, heads, embedding dimension). It must support a `scaling_factor` to automatically shrink dimensions for child nodes.
2. **`FractalAttention`:** A modified Multi-Head Attention block. Unlike standard attention, it needs a "projection mapping" layer to handle cases where a Child Node (dim=512) tries to read weights from a Parent Node (dim=1024).
3. **`WeightSlicer`:** A utility function that takes a massive weight matrix  and mathematically slices it to initialize .

**The Mathematics:**

* **Standard Attention (The Core Mechanism):**


* **The Scaling Law (The Fractal Math):**
When a child is born, its dimensions scale down based on the depth () of the tree:



*(Where  is your decay rate, e.g., 0.5 for half-size).*

---

### **Phase 2: The "Semantic Router" (The Compass)**

**Status:** *The Navigation System*
**Goal:** A system that directs data to the correct node *before* the model even thinks.

**Sub-Components:**

1. **`VectorEncoder`:** A lightweight model (like `all-MiniLM-L6-v2`) that converts user text into a high-dimensional vector.
2. **`CentroidManager`:** A dynamic list that stores the "center point" of every expert node in vector space.
3. **`RouterGate`:** The logic that calculates distance and selects the winner.

**The Mathematics:**

* **Cosine Similarity (The Routing Metric):**
To find the right expert, we measure the angle between the Query Vector () and the Expert's Centroid Vector ():



*(If Similarity > Threshold, route to that Expert).*

---

### **Phase 3: The "Mitosis" Engine (The Reproductive System)**

**Status:** *The Growth Mechanism*
**Goal:** The script that actually creates a new node. It uses **Knowledge Distillation** to transfer "IQ" from the Parent to the Child while shrinking the brain size.

**Sub-Components:**

1. **`EntropyMonitor`:** A background process that tracks the "Loss" (error rate) of the Mother Node on specific topics. If Loss is consistently high for a specific cluster, it triggers Mitosis.
2. **`DistillationLoop`:** The training loop where the Parent (Teacher) teaches the Child (Student).
3. **`SoftTargetProjector`:** Since the Parent and Child have different vocabulary sizes or dimensions, this aligns their outputs for comparison.

**The Mathematics:**

* **Knowledge Distillation Loss:**
We don't just train on the answer (Hard Target). We train the child to mimic the *probability distribution* of the parent (Soft Target).


* : Standard Cross-Entropy Loss (Learning the data).
* : Kullback-Leibler Divergence (Learning how the Teacher thinks).
* : Temperature (Softens the probability curve).



---

### **Phase 4: The "Swarm" Registry (The Nervous System)**

**Status:** *State Management*
**Goal:** A database that keeps track of the entire family tree.

**Sub-Components:**

1. **`NodeRegistry`:** A JSON/SQL structure tracking the lineage.
* *Example:* `{ "id": "node_a1", "parent": "node_A", "depth": 2, "status": "active" }`


2. **`PathFinder`:** A graph traversal algorithm. When a query comes in, it determines the shortest path from Root  Leaf.
3. **`LifeCycleManager`:** A script that "prunes" (deletes) nodes that haven't been used in a long time to save space.

**The Mathematics:**

* **Tree Traversal Complexity:**
Searching for the right node is efficient because it scales logarithmically, not linearly.



*(Where  is total nodes and  is the branching factor).*

---

### **Phase 5: The "Hyper-Stream" Engine (The Physical Body)**

**Status:** *Runtime Execution*
**Goal:** The user-facing interface. It handles memory swapping (loading/unloading models) in milliseconds.

**Sub-Components:**

1. **`ModelSwapper`:** A low-level memory manager. It keeps the Mother Node in VRAM (GPU memory) permanently but swaps Child Nodes in and out of RAM (CPU memory) on demand.
2. **`StreamAggregator`:** If the architecture allows multiple experts to answer (Ensemble Mode), this averages their answers.
3. **`InferenceAPI`:** The REST API (FastAPI) that the user connects to.

**The Mathematics:**

* **VRAM Estimation (Memory Physics):**
To ensure we don't crash your Mac M1, we calculate the memory footprint () before loading:



*(Where  is the parameter count. Example: A 100M param child node takes ~200MB VRAM).*

---

### **Summary of the Build Order**

1. **Build Phase 1 (Genesis):** Write the `FractalTransformer` code. Verify it runs.
2. **Build Phase 2 (Router):** Write the vector logic. Verify it sorts text correctly.
3. **Build Phase 3 (Mitosis):** Connect 1 & 2. Train a small child from a big parent.
4. **Build Phase 4 (Registry):** Add the database to track multiple children.
5. **Build Phase 5 (Hyper-Stream):** Build the final chat interface.
