# synonic — open synonym discovery (leakage-free rebuild)

This folder answers the advisor's feedback on the prior iteration (`new/src2/`):
the task must be treated as **open synonym discovery**. A method may only
ever see the input term (`en`); it must never see the ground-truth synonym
(`en_synonym`) until the final evaluation step.

## The leak this fixes

In `new/src2/`, several methods built their retrieval/candidate index directly
from `df['en_synonym']`:

- `run_experiments.py` (`embedding` method) — `build_output_vector_index`
  embedded `df['en_synonym']` and searched it.
- `run_llm_expansion_retrieval.py` — same `build_output_vector_index` over
  `en_synonym`, searched with LLM-generated query variants.
- `run_mas_langgraph.py` / `run_mas_v2_langgraph.py` / `run_mas_v3_safe_hybrid.py`
  — the `retrieval_agent` node searched a `candidate_pool` built from
  `df["en_synonym"].dropna()...`.

In every case the system was searching *inside the answer column*, which
inflates every metric far beyond what a realistic deployment (no access to
the answer) could achieve.

**Fix applied here:** every method's candidate vocabulary now comes from
**WordNet** (`src/open_vocab.py`, via `nltk.corpus.wordnet`) — a resource
that is completely independent of this dataset and its `en_synonym` column.
`en_synonym` is read in exactly one place in the whole codebase:
`src/run_experiment.py`, inside the main loop, on the line
`ground_truth = row["en_synonym"]` — which comes strictly *after*
`candidates = get_candidates(method, term, resources)` has already produced
its output. Grep for `en_synonym` under `src/` any time to verify this still
holds; every other hit is a comment/docstring, not a data access.

## Dataset

`jensjorisdecorte/TU-Expert-Collection-Topic-Synonyms` (HuggingFace), same
source as the prior iteration. Only `en` (input term) and `en_synonym`
(ground truth) are used — `nl`/`nl_synonym` are dropped, per the advisor's
instruction, since this project only targets English. `topic` is cached on
disk (`topic_synonyms_clean.csv`) for provenance but dropped by
`load_full_dataset()` before it reaches any method: no method ever took it as
input and no summary broke results down by topic, so it made no difference
either way — removed per advisor feedback after confirming that.

- `data/prepare_dataset.py` → `data/topic_synonyms_clean.csv` (970 rows after
  dropping empty/duplicate entries).
- `data/make_eval_sample.py` → `data/eval_sample.csv`: a **fixed, seeded**
  100-row sample (`random_state=42`) so every method is compared on exactly
  the same rows — re-running the script reproduces the identical sample.

## Project layout

Structured to match the other iterations in this repo (see `second/`):

- `data/` — dataset prep scripts + cached CSVs (`topic_synonyms_clean.csv`, `eval_sample.csv`) + `cache/` (WordNet vocab/embeddings, built once).
- `src/data.py` — loads the full dataset or the fixed eval sample.
- `src/llm.py` — single place that talks to Ollama (`ask_llm`) and parses its output into a candidate list (`parse_candidates`).
- `src/open_vocab.py` — builds/caches the WordNet candidate vocabulary (the leakage fix).
- `src/evaluation.py` — all metrics, documented.
- `src/baselines.py` — the 5 non-agentic methods.
- `src/agents.py` — the MAS agent functions, each standalone and independently callable/testable.
- `src/graph.py` — wires `agents.py` into the 3 MAS variants via LangGraph.
- `src/supervisor/` — single-agent supervisor architecture: an LLM decides which deterministic tool to call next (retrieve/generate/merge/score/finalize/finish) instead of a fixed graph topology. See below.
- `src/run_experiment.py` — CLI for a single method, with resume + incremental save.
- `src/run_comparison.py` — CLI for all methods at once, with resume + incremental save, into one `comparison_results.csv`.
- `ARCHITECTURE.txt` — design rationale for each agent/method and the leakage fix (same style as `second/ARCHITECTURE.txt`).
- `run.sh` — convenience launcher with common commands commented in.
- `requirements.txt` — pinned dependencies (`pip freeze` of `synonic/.venv`), so this folder's environment can be rebuilt from scratch independently of any other folder in this repo.

## Methods (`src/baselines.py`, `src/agents.py` + `src/graph.py`)

All take only `term: str` (plus shared, term-independent resources like the
embedder/LLM/WordNet index) — never the dataframe or `en_synonym`.

| method | description |
|---|---|
| `wordnet_direct` | looks up WordNet synsets containing `term` as a lemma and returns the synset's other lemmas. Pure symbolic baseline, no ML. |
| `embedding_wordnet` | embeds `term` with `all-MiniLM-L6-v2` and retrieves the nearest neighbors from the full WordNet vocabulary (~147k phrases), by cosine similarity. |
| `llm_zero_shot` | prompts an LLM (via Ollama) directly for 5 candidate synonyms of `term`. No retrieval. |
| `llm_expansion` | asks the LLM for 6 paraphrase/expansion queries of `term`, embeds each, and searches the WordNet index with all of them, aggregating by max similarity (+ small cross-variant agreement bonus). |
| `mas_base` (v1) | LangGraph pipeline: `generator(LLM)` ∥ `retrieval(embedding over WordNet)` → `merge` → `verification (0.75·semantic + 0.25·fuzzy)` → `ranking` → top 5. |
| `mas_llm_ranker` (v2) | `retrieval` → `generator` → `merge` (capped at 20) → an LLM re-orders the whole list → top 5. Known negative ablation in the original experiments (kept for comparison, not because it's expected to win). |
| `mas_safe_hybrid` (v3) | same as v2, but a `safe_finalizer` blends `0.9·embedding_sim + 0.1·llm_rank_bonus` — the LLM ranking can only add a bonus, never penalize a candidate it left out, protecting recall from an unreliable LLM ranker. |

All MAS variants are a node-for-node port of `new/src2`'s three graphs — only
`retrieval_agent`'s vocabulary source changed (WordNet instead of
`en_synonym`). See `ARCHITECTURE.txt` for the full rationale behind each
agent and the v1→v2→v3 ablation.

| `supervisor` | a single `supervisor_agent` LLM call decides one action per turn (`retrieve` / `generate` / `merge` / `score` / `finalize` / `finish`) based on the current state, instead of following a fixed pipeline order. Reuses the exact same deterministic tools as the MAS variants above (`retrieval_agent`, `generator_agent`, `merge_agent`, `verification_agent`, `ranking_agent` from `agents.py`) — only the *decision of what to run next* is dynamic. |

### `src/supervisor/` — dynamic single-agent architecture

Where `mas_base`/`mas_llm_ranker`/`mas_safe_hybrid` hard-code the node order
in `graph.py`, `supervisor/` lets an LLM choose the order and parameters at
each step, based on an explicit, inspectable state:

- `state.py` — `SupervisorState` (candidates at each stage, `action_history`, iteration/LLM-call counters) and its constructor.
- `schemas.py` — `SupervisorDecision` (Pydantic): `{action, reason, parameters, confidence}`. The LLM can only ever produce one of these, never arbitrary code.
- `prompts.py` — system prompt (available actions + rules) and the per-turn state-to-text renderer.
- `agent.py` — `supervisor_agent` (one LLM call → validated decision), `fallback_decision` (deterministic retrieve→generate→merge→score→finalize→finish backstop used whenever the LLM output is invalid, repeats an identical call, or tries to `finish` before final candidates exist), `repeated_action`/`can_finish` guardrails.
- `tools.py` — `evaluate_candidate_quality`, the one new deterministic helper (turns scored candidates into a sufficiency/confidence signal); every other tool is reused directly from `agents.py`.
- `executor.py` — the only place a `SupervisorDecision` is turned into a real tool call; clamps every LLM-supplied parameter (e.g. `top_k`, `n`, `cap`) to a safe range.
- `runner.py` — `run_supervisor`, the plain-Python decide→execute→observe loop, capped at `MAX_ITERATIONS = 10` with a deterministic forced-finalize if the cap is hit before finishing.
- `graph.py` — the same functions wired into a LangGraph (`supervisor` ⇄ `execute_tool`, conditional routing to `force_finalize`/`END`), for anyone who wants LangGraph's tracing on top of the identical logic.

Wired into `run_experiment.py`/`run_comparison.py` as the `supervisor` method,
so it can be benchmarked against every other method with the same harness.
Note: this is one autonomous agent choosing among deterministic tools, not a
multi-agent system — see `ARCHITECTURE.txt` for the reasoning on why that's
the right next step before (if ever) splitting into separate cognitive
agents (sense-disambiguation, generation, critique).

## Metrics (`src/evaluation.py`)

Every metric is documented in that file's docstring; summary here per the
advisor's request for explicit definitions:

- **hit@1** (was `exact_top1`): the method's #1 candidate exactly equals the
  ground truth (after lowercasing/whitespace normalization).
- **exact_any**: the ground truth appears *anywhere* in the returned
  candidate list. Every method here returns **5** candidates
  (`num_final_candidates` is recorded per row and averaged as
  `avg_num_candidates_returned` in each summary — always read `exact_any`
  together with that number, since allowing more candidates trivially raises
  it).
- **hit@3 / hit@5** (were `recall_at_3` / `recall_at_5`): ground truth
  appears in the top 3 / top 5. With only one gold synonym per term, hit@K is
  exactly "did the correct answer appear in the top K" — there is no
  multi-item recall to average.
- **ndcg@3 / ndcg@5**: like hit@3/hit@5 but graded by position instead of
  binary — `1/log2(rank+1)` when the hit falls within the first K, else 0.
  Penalizes a correct answer at rank 5 more than one at rank 1, even though
  both count equally for hit@5.
- **precision_at_5**: (# of the top-5 slots that exactly match ground truth)
  / 5. Since there is only one gold synonym, at most 1 of 5 slots can ever be
  correct, so this is mechanically capped at 0.2 and is really just
  `hit@5 / 5`. It does not penalize the other 4 candidates for being
  "wrong" (some may be valid synonyms the dataset simply doesn't list).
  Reported for completeness; prefer `hit@5` / `exact_any` for this
  dataset.
- **mrr** (Mean Reciprocal Rank): `1 / rank` of the correct answer in the
  candidate list (0 if absent), averaged across rows. Rewards ranking the
  correct answer near the top even when it isn't first (rank 1 → 1.0, rank 2
  → 0.5, rank 4 → 0.25, absent → 0).
  **Note:** MRR here is computed the same way whether the correct answer
  wasn't returned or wasn't found in a larger pool — it does not distinguish
  those cases beyond the score itself.
- **fuzzy_similarity**: `rapidfuzz.fuzz.ratio(normalized_ground_truth,
  normalized_candidate) / 100`, maxed over all candidates. `fuzz.ratio` is
  normalized indel similarity: `100 × (1 − edit_distance / (len(a)+len(b)))`
  — a **lexical/spelling** closeness score (e.g. catches "optimise" vs
  "optimize"), not a meaning-based one.
- **semantic_similarity**: cosine similarity between the ground-truth
  embedding and each candidate's embedding, both encoded with
  `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized, so cosine
  similarity = dot product), maxed over candidates. This is a
  **meaning**-based score — can be near 1.0 even for paraphrases sharing no
  words with the ground truth.

See [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md) for a running record of every
run's model/prompt/metrics, kept for a future prompt/model sensitivity
analysis.

## Running it

```bash
source .venv/bin/activate   # or prefix commands with .venv/bin/python3

# one-time: build the fixed eval sample (already committed, but reproducible)
python data/make_eval_sample.py --n 100

# quick smoke test (10 terms, all methods, one combined CSV)
python -m src.run_comparison --limit 10 --no-resume

# run a single method over a few rows
python -m src.run_experiment --method embedding_wordnet --limit 5
python -m src.run_experiment --method llm_zero_shot --model llama3.2:3b --limit 5

# full 100-row eval sample, a single method (resumable - safe to Ctrl-C and rerun)
python -m src.run_experiment --method mas_safe_hybrid --model llama3.2:3b

# full 100-row eval sample, every method, one comparison_results.csv (resumable)
python -m src.run_comparison --model llama3.2:3b
```

Or just run `./run.sh` (edit which line is uncommented for the run you want -
see the comments inside).

`--method` / `--methods` choices: `wordnet_direct`,
`embedding_wordnet`, `llm_zero_shot`, `llm_expansion`, `mas_base`,
`mas_llm_ranker`, `mas_safe_hybrid`.

Both CLIs **resume by default**: if the output CSV already has rows for a
given (term, method), they're skipped, and the file is saved after every
term - so a long MAS run is safe to interrupt and continue. Pass
`--no-resume` to start fresh.

All eight methods have been smoke-tested end-to-end (`--limit 2`–`5`,
`llama3.2:3b`) and run without errors, including all three MAS variants —
those were never actually executed in the prior iteration. **The full-scale
100-row comparison has not been run yet** — that's the next step.

## Environment

`.venv/` is a self-contained virtualenv for this folder only - it does not
share or symlink anything from `new/`, `second/`, or any other folder in
this repo. To rebuild it from scratch elsewhere:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"  # if WordNet corpus isn't already cached
```
