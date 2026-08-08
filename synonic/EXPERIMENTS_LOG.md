# Experiments log

Running record of every run's **prompt version + model + metrics**, so a
future prompt/model sensitivity analysis has something to compare against
instead of only the latest `results/*.json`. Append a new entry every time
you run `run_experiment.py` / `run_comparison.py` with a different model or
after editing a prompt in `src/agents.py` / `src/baselines.py` /
`src/supervisor/prompts.py`.

Metric names below use the current convention (`hit@1/3/5`, `ndcg@3/5`,
`exact_any`, `mrr`) — see [Metrics note](#metrics-naming-note) for how older
entries map to it.

## Entries

### 2026-07-18 — full comparison, all 11 methods, `llama3.2:3b`
- Source: `results/comparison_results.csv` (100 rows/method, 1100 rows total)
- Model: `llama3.2:3b` (default in `run_comparison.py`/`run_experiment.py`)
- Prompts: as in `src/agents.py`/`src/baselines.py` at this commit — no prompt
  variants tested yet, only method/architecture varies.
- **Includes now-removed `same_term`** (kept here for the historical record —
  see [ARCHITECTURE.txt](ARCHITECTURE.txt) cleanup note) and the pre-rename
  metric names (`exact_top1`/`recall_at_3`/`recall_at_5`) — this file itself
  was not regenerated, only the code that will produce the *next* one.

| method | n | hit@1 | hit@3 | hit@5 | mrr | avg_time_s |
|---|---|---|---|---|---|---|
| llm_zero_shot | 100 | 0.17 | 0.22 | 0.26 | 0.201 | 0.69 |
| hybrid_fusion | 100 | 0.12 | 0.27 | 0.31 | 0.195 | 3.04 |
| llm_rerank | 100 | 0.13 | 0.24 | 0.26 | 0.182 | 1.01 |
| supervisor | 100 | 0.07 | 0.16 | 0.21 | 0.122 | 7.23 |
| mas_safe_hybrid | 100 | 0.04 | 0.18 | 0.25 | 0.121 | 3.76 |
| mas_llm_ranker | 100 | 0.07 | 0.17 | 0.18 | 0.120 | 3.74 |
| mas_base | 100 | 0.07 | 0.15 | 0.20 | 0.119 | 1.02 |
| llm_expansion | 100 | 0.05 | 0.10 | 0.11 | 0.074 | 0.89 |
| embedding_wordnet | 100 | 0.06 | 0.08 | 0.10 | 0.073 | 0.02 |
| wordnet_direct | 100 | 0.05 | 0.07 | 0.07 | 0.060 | 0.01 |
| ~~same_term~~ | 100 | 0.00 | 0.00 | 0.00 | 0.000 | ~0 |

`same_term` never beat 0 on any metric (it can only match if `en == en_synonym`,
which never happens in this dataset) — this is the evidence behind removing it.

### 2026-07-08 — `llm_rerank`, full dataset, `llama3.2:3b`
- Source: `results/llm_rerank_llama3.2_3b_full_summary.json`
- Model: `llama3.2:3b`. Prompt: `baselines.py::llm_generate_rerank_baseline`
  system prompt ("You are a domain terminology expert...").
- hit@1 0.05, exact_any 0.26, hit@3 0.19, hit@5 0.26, mrr 0.124,
  avg_time_s 1.04 (982 rows — full dataset, not the 100-row sample above,
  numbers aren't directly comparable to the table above).

### 2026-07-16 — `supervisor`, 5-row smoke test, `llama3.2:3b`
- Source: `results/supervisor_llama3.2_3b_limit_5_summary.json`,
  `results/supervisor_llama3.2_3b_limit_5_results.csv`
- Model: `llama3.2:3b`. Prompts: `src/supervisor/prompts.py`.
- rows 5, hit@1 0.0, exact_any 0.2, hit@3 0.2, hit@5 0.2, mrr 0.1,
  avg_time_s 6.17. Too small a sample to draw conclusions — smoke test only.

## Metrics naming note

`exact_top1` → `hit@1`, `recall_at_3`/`top_3_accuracy` → `hit@3`,
`recall_at_5`/`top_5_accuracy` → `hit@5` (same computation, renamed
2026-07-19 to match standard IR naming and `18_7/rsc/evaluation.py`).
`ndcg@3`/`ndcg@5` are new as of the same date. `exact_any`, `precision_at_5`,
`mrr`, `fuzzy_similarity`, `semantic_similarity` are unchanged.

## What to log for a future prompt-sensitivity analysis

For each new run, record: date, method(s), model name, which prompt(s)
changed (paste the new system prompt text if it changed, or note "unchanged"
+ file:line), `--limit` used, and the resulting summary dict. The goal is to
be able to later answer "did changing prompt X or swapping model Y move
hit@1/hit@3/hit@5/ndcg@3/ndcg@5, or was the swing just noise" — which needs
more than one data point per prompt/model combination to say anything.
