# karam.v1

Fractal-Dendritic Network (FDN) prototype with hierarchical transformers, semantic routing, mitosis distillation, and a lightweight inference API.

## Structure
- src/models: FractalTransformer and config
- src/router: semantic routing components
- src/registry: node registry and lineage tracking
- src/runtime: model swapping, stream aggregation, and FastAPI app
- src/utils: device and weight slicing helpers

## Quick start
1. Install dependencies from requirements.txt.
2. Run module self-tests (each module includes a __main__ block).
3. Use create_app in src/runtime/inference_api.py to start the FastAPI server.
