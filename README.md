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

## v0.2 — honest audit (bench/audit.py, 5 seeds, free text, paraphrased queries)
v0.1 numbers were inflated: facts used a `subject::` marker the code relied on, and queries matched fact wording. v0.2 removes all format tricks.
| method | all facts | updated facts | 2-hop | ms/q |
|---|---|---|---|---|
| RAG | 83.1±0.6 | 0.2±0.5 | 0.0 | 0.6 |
| RAG+recency | 40.4±2.8 | 68.2±6.8 | 0.0 | 0.6 |
| **CORTEX v0.2** | **94.4±1.4** | 66.0±4.8 | 9.3±5.2 | 27.6 |
| − gate | 94.4 | 66.0 | 9.0 | 31.2 |
| − supersede | 83.1 | 0.0 | 5.7 | 27.5 |
| − spread | 93.3 | 61.7 | 0.0 | 23.4 |
Findings: supersession is the component that matters (+11 pts all, +66 updated). Gate adds speed, no accuracy. Spreading activation helps a little; 2-hop is still weak (9%). Still synthetic, hashing embeddings, no LLM in loop.
