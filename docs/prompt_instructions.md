### 2. `prompt_instructions.md` (The Copilot Guide)

```markdown
# Copilot & AI Coding Instructions
**Objective:** Build a robust, memory-efficient Hierarchical Transformer for Mac M1 (MPS).

## 1. Coding Standards (PyTorch)
- **Module Structure:** All neural networks must inherit from `torch.nn.Module`.
- **Type Hinting:** strict adherence to `typing`.
  - *Correct:* `def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:`
  - *Incorrect:* `def forward(self, x):`
- **Tensor Operations:** Use `einops` for complex reshapes (e.g., splitting heads).
  - Example: `rearrange(x, 'b s (h d) -> b h s d', h=n_heads)`

## 2. Architectural Constraints
- **Initialization:**
  - Every `__init__` MUST check: `assert d_model % n_heads == 0, "Dimension mismatch"`
- **Hierarchy:**
  - Child nodes MUST verify `self.config.d_model <= parent.config.d_model`.
  - Implement a `scale_config` helper to strictly enforce integer division constraints.

## 3. Memory & Hardware Safety (Crucial for Mac M1)
- **Device Agnosticism:**
  - Code must check `torch.backends.mps.is_available()` and fallback to `cpu` if needed.
  - Use `device` parameter in all `.to(device)` calls.
- **Garbage Collection:**
  - In `Phase 3` (Mitosis) and `Phase 5` (Inference), explicit cleanup is required:
  ```python
  import gc
  del model
  torch.cuda.empty_cache() if torch.cuda.is_available() else None
  gc.collect() # Crucial for MPS memory release

```

## 4. Testing & Verification

* **Self-Contained Tests:**
* At the bottom of EVERY module file, include an `if __name__ == "__main__":` block.
* This block MUST:
1. Instantiate the class with dummy config.
2. Create a dummy input tensor (e.g., `torch.randint(0, 1000, (2, 10))`).
3. Run a `forward()` pass.
4. Print input shape and output shape.
5. Assert that output shape matches expectations.





```
