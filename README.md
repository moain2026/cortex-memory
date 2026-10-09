# CORTEX — brain-inspired memory for AI agents (prototype, Track A)
Surprise gate → hippocampal graph (spreading activation + Hebbian + decay) → sleep consolidation.
Design doc: see design.md. Run: `python3 bench/run.py`

## First benchmark (synthetic: 2160 items, 1500 noise, 60 fact updates, 260 queries)
| method | accuracy | accuracy on updated facts | tokens/query | ms/query |
|---|---|---|---|---|
| full context | 0.692 | 0.00 | 15,960 | — |
| plain RAG k=6 | 0.692 | 0.00 | 36 | 0.22 |
| **CORTEX** | **0.885** | **0.70** | 70 | 3.05 |
"Correct" = current value retrieved AND stale value absent. Limits: synthetic data, hashing embeddings, no LLM in the loop yet.
