# synonim2 - Academic term synonym finding

Task: given an English academic/domain term (`en`), predict its synonym.
Evaluated against the `TU-Expert-Collection-Topic-Synonyms` dataset by
comparing predictions to the gold `en_synonym` for each row.

**Leakage rule** (clarified by the user 2026-07-19, see project memory
`feedback-leakage-definition`): a method may use a global candidate
vocabulary built from the full `en_synonym` column (that is NOT leakage -
it's a fixed vocabulary, not a lookup of the row's own answer). What's
forbidden is consulting the gold `en_synonym` of the row being evaluated
before producing its prediction. Every method below reads ground truth only
after generating candidates - grep `en_synonym` in `run_eval.py`/`data.py`
to confirm before trusting any future change here.

## Task formulation: which of the two approaches is this?

The advisor's 2026-07-22 email distinguished two possible readings of
"synonym finding":

1. **Open generation**: given `en`, generate/produce a synonym from
   scratch (e.g. free-text LLM output); `en_synonym` is used *only* at
   evaluation time to check the generated text against the gold answer.
2. **Ranking**: given `en`, rank/select the correct synonym from a
   predefined candidate pool built from `en_synonym`; `en_synonym` is the
   search pool the system ranks candidates from.

**This project implements approach 2, for every method, including the
"zero-shot" ones.** Even `llm_zero_shot` and `llm_zero_shot_e5` - which
free-generate a synonym proposal with an LLM that never sees the candidate
pool (`src/agents/zero_shot_agent.py`) - do not return that raw LLM text
as the final prediction. The free-text output is used only as a *retrieval
query* against the `en_synonym` vocabulary (`retrieve_top_k`), and the
nearest pool entry is returned. So no method in `src/methods.py` currently
returns unconstrained open-ended generation as its final answer; all ~40
methods ultimately select/rank from the closed candidate pool. This was a
deliberate, evaluation-driven choice made early on (see `EXPERIMENTS_LOG.md`,
2026-07-19): pure open generation has no well-defined way to score against
one gold string per row without some form of matching/ranking step anyway,
and ranking against the real candidate vocabulary is both a cleaner
evaluation protocol and, empirically, the better-performing design (the
zero-shot methods that skip the LLM-expansion+fusion pipeline underperform
it by double digits in hit@1, logged under "What was tried and discarded").

## Why train/validation/test splits, if nothing is trained from scratch?

No component here is trained with gradient descent - `data/make_splits.py`
doesn't feed a training loop. The splits exist because **method and
hyperparameter selection is itself a form of fitting to data**, even
without training weights:

- Over this project's cycles, ~40 candidate pipelines were compared (5
  embedding models, 4 fusion strategies, fusion-weight values, generator
  temperature/variant-count, 3 reranker models, a confidence-gating
  threshold, pool-width variants...). Picking the best one *by looking at
  its score* is a selection procedure with the same overfitting risk as
  tuning hyperparameters of a trained model: whichever config got luckiest
  on the evaluation set looks best on that same set, even if the win isn't
  real signal.
- **`val`** (194 rows) is where every one of those comparisons was made
  and where every numeric knob (fusion weight, n=3 vs n=9 variants,
  `GATE_THRESHOLD`, which embedder) was chosen. It plays the same role a
  validation set plays for a trained model's hyperparameters - it's
  "used" repeatedly, so its own numbers can end up optimistic.
- **`test`** (194 rows) is touched only *once per already-decided config*,
  purely to report an unbiased number - never to pick a winner between two
  configs. Every "current best" table in this README reports both val (for
  transparency about what val looked like when the decision was made) and
  test (the number that should be trusted/quoted), and they've stayed
  close throughout (e.g. current best: val hit@1 0.851, test hit@1 0.830
  on the fallback-verified re-confirmation run - see "Best configuration
  found" below for why there are two test numbers), which is itself
  evidence the val-based selection wasn't overfit.
- **`train`** (581 rows) is deliberately left untouched by every method
  above - reserved for exactly the kind of thing that *would* need real
  training if pursued later (e.g. calibrating the rerank confidence
  threshold with a fitted model instead of the current offline heuristic,
  or fine-tuning an embedder/cross-encoder) - see "Most promising next
  experiments" below.
- **`full`** (969 rows, train+val+test combined) is used exactly once, at
  the very end, purely for descriptive reporting/error-analysis on the
  already-locked-in best config - never for comparing or selecting between
  methods (see "Full-dataset report" below).

Splits are grouped by unique `en` term (not by row) with a fixed seed, so
the same input term never appears in two different splits - see
`data/make_splits.py` docstring for the degenerate-row exclusion
(`en == en_synonym`) and duplicate-term handling.

## Best configuration found

**`llm_expansion_weighted_t0_rerank_qwen`** (2026-07-20, current best):
`llm_expansion_weighted_t0` (temp=0 query expansion + weighted e5 fusion,
n=9) generates a top-5, then **every** term's top-5 is sent to
`qwen/qwen3-32b` via OpenRouter (paid API) to reorder from most-to-least
likely synonym - the LLM only reorders the 5 candidates it receives, never
invents new ones or sees the gold answer (`src/agents/reranker_agent.py::reordenar_candidatos`).

| split | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s | cost/194 rows |
|---|---|---|---|---|---|---|---|---|
| val (n=194) | 0.851 | 0.912 | 0.923 | 0.882 | 0.888 | 0.892 | 9.08 | ~$0.03 |
| test (n=194, held out, 2026-07-20 original run) | 0.851 | 0.923 | 0.949 | 0.892 | 0.895 | 0.907 | 9.27 | ~$0.03 |
| **test (n=194, held out, 2026-07-22 re-confirmation)** | **0.830** | **0.938** | **0.948** | **0.884** | **0.896** | **0.901** | **13.32** | **~$0.03** |

**Two test rows, on purpose**: the original 2026-07-20 test run predates
the `reranker_fallback` tracking added after the credit-exhaustion
incident (see "Known risks"), so it couldn't be verified after the fact
as fallback-free. It was re-run on 2026-07-22 under the current harness
(`reranker_fallback_count: 0` confirmed - every row has a real reranker
response) as the trustworthy number going forward. The two runs differ by
**-2.1pt hit@1 / +1.5pt hit@3 / -0.1pt hit@5 / -0.8pt MRR** despite being
the *identical* config, prompt, and `temperature=0` - this is exactly the
Qwen-API non-determinism-between-calls risk flagged in "Known risks"
below, now measured directly on the final config rather than inferred
from other methods' repeat runs. hit@5 (0.949→0.948) and MRR barely move,
which is reassuring - the reranker's *set* of retrieved-and-considered
candidates is stable, only the top1-vs-top2/3 ordering call flips on a
meaningful minority of close cases between runs. Latency was also
noticeably slower on the re-run (13.3s vs 9.3s/term) - likely OpenRouter-
side load variance for `qwen3-32b`, not a code change (no `run_eval.py` or
`reranker_agent.py` logic changed between the two runs).

vs. the previous best (`llm_expansion_weighted_n3`, no rerank, test hit@1
0.768, mrr 0.835), using the re-confirmed number: **+6.2pt hit@1, +4.9pt
MRR, +1.5pt hit@5** - still by far the largest single jump of this cycle,
just slightly smaller than the original run suggested. Rank-movement
below (**21 terms moved gold into rank 1, only 5 regressed**) was computed
against the *original* 2026-07-20 test run and has not been recomputed
against the re-confirmation - the direction and rough magnitude of the
finding stands (reranking helps broadly), but the specific example list
may shift slightly if regenerated:
`compare_methods.py --method llm_expansion_weighted_t0_rerank_qwen
--baseline llm_expansion_weighted_n3 --split test` → **21 terms moved gold
into rank 1, only 5 regressed** (net +16/194), and the winning/losing
examples are semantically sensible in both directions (e.g. `auditors`:
"financial auditing" → "financial examiners" ✓; `poverty`: "joblessness" →
"destitution" ✓ - vs. a regression like `terrorism`: "extremism" →
"religious extremism", a defensible near-miss, not nonsense).

**This directly supersedes the earlier "reranking adds zero benefit"
finding** (`llm_expansion_rerank`, `e5_llm_rerank`, both discarded further
down this doc) - that finding was specific to a *weak* local reranker
(`qwen3.5:2b`, 2B params); a genuinely strong reranker (`qwen3-32b`, 32B)
helps broadly, not just on ambiguous cases (see "confidence-gated rerank"
below - gating turned out to *underperform* reranking everything).

**Cost/latency tradeoff**: ~$0.03 per 194-row split (real cost read from
OpenRouter's response `usage.cost` field, not estimated), ~9-13s/term
(observed range across the two test runs, vs 0.7s/term for
`llm_expansion_weighted_n3`, ~13-19x slower) - `qwen3-32b`
runs with implicit "thinking" mode by default (200-800 reasoning tokens
per call observed), which is most of both the latency and the cost. This
is still cheap and fast enough to run at this dataset's scale (~$0.15
total spent across every experiment in this section, val+test+smoke
tests+the discarded gated variant), but is a real, non-free dependency on
a paid external API, unlike every other method in this project.

**Model note**: `qwen/qwen3.5-9b` (the model initially tried, matching
what was already wired into `reranker_agent.py`) was swapped for
`qwen/qwen3-32b` after `qwen3.5-9b` proved unreliable on OpenRouter (~2 of
3 calls returned a transient "Service unavailable" 503, delivered as an
embedded `error` field in an HTTP-200 response rather than an HTTP error,
so the OpenAI client's built-in retry logic didn't catch it - see
`openrouter_client.py::_completar_con_reintentos` for the explicit retry
added, and the "Known risks" section below). `qwen3-32b` was 5/5 reliable
in direct testing, faster, and confirmed on OpenRouter's model catalog as
a currently-available model with multiple providers (better redundancy).

### Full-dataset report (descriptive only, 2026-07-21)

`data/splits/full.csv` (train+val+test combined, 969 rows) - reporting
only, config already locked from val/test before this ran, never used to
pick or tune anything. Every row confirmed to have a real reranker
response (`reranker_fallback_count: 0` in the summary JSON, verified via
`run_eval.py`'s built-in check - see "Known risks" for what this
guarantees):

| method | n | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|---|
| `embedding_only` (original baseline) | 969 | 0.545 | 0.742 | 0.802 | 0.646 | 0.661 | 0.685 | 0.006 |
| `embedding_e5` | 969 | 0.645 | 0.835 | 0.892 | 0.743 | 0.757 | 0.781 | 0.010 |
| `llm_expansion_weighted` (no rerank, old best) | 969 | 0.755 | 0.911 | 0.946 | 0.833 | 0.847 | 0.862 | 0.716 |
| **`llm_expansion_weighted_t0_rerank_qwen` (current best)** | **969** | **0.822** | **0.932** | **0.945** | **0.877** | **0.889** | **0.895** | **11.84** |

797/969 exact top-1, 916/969 (94.5%) within top-5. **+27.8pt hit@1 /
+23.1pt MRR** over the original baseline, **+17.8pt / +13.4pt** over
`embedding_e5`, **+6.7pt hit@1 / +4.4pt MRR / +2.1pt hit@3** over the best
pre-rerank config - with hit@5 essentially flat (0.945 vs 0.946, within
noise), confirming the rerank step mainly fixes *ordering*, not recall.
Cost: ~$0.18 (real usage delta from OpenRouter, `GET /api/v1/credits`
before/after).

**Remaining error cases** (`scripts_error_breakdown.py`-style analysis,
`results/eval/error_analysis_rerank_qwen_full.csv`):
- 797 correct (rank 1) - 82.2%.
- 119 recoverable: gold in top-5 but not rank 1 (87 at rank 2, 19 at rank
  3, 11 at rank 4, 2 at rank 5) - reranker ordering mistakes on genuinely
  close calls, e.g. `investment` → gold "investing", predicted "funding";
  `accountancy` → gold "financial auditing", predicted "financial
  accounting" (a defensible alternative, not nonsense).
- 53 retrieval misses (5.5%): gold never appears in the top-5 at all, so
  no reranker can fix it - mostly idiosyncratic/distant gold synonyms
  (abbreviation expansions like `evs` → "European Values Study",
  etymological phrasings like `gospels` → "good news"). Same pattern seen
  on val; pool-widening was tried against exactly this category and made
  hit@1 worse (see "What was tried and discarded" below) - still an open
  problem. **Abbreviation expansion was also tried** (`llm_expansion_rerank_qwen_abbrev`,
  2026-07-21) - discarded, see below: it had literally zero effect on its
  one testable val case.

**Four attempts to close the retrieval-miss gap, all discarded** (val,
n=194):

| method | hit@1 | hit@3 | hit@5 | mrr | verdict |
|---|---|---|---|---|---|
| pool=5 (base, winner) | 0.851 | 0.912 | 0.923 | 0.882 | |
| widen pool to 10 before rerank | 0.809 | 0.918 | 0.933 | 0.862 | hit@5 +1.0pt but hit@1 -4.1pt - more distractors hurt the top pick |
| widen pool to 15 before rerank | 0.799 | 0.907 | 0.928 | 0.853 | worse than pool=10 - monotonically worse with pool size |
| merge Qwen's own direct-guess candidates | 0.794 | 0.902 | 0.923 | 0.849 | worse on every metric, 0 new recall - LLM's guesses don't find anything embeddings didn't already |
| expand abbreviation-like terms as an extra query | 0.825 | 0.918 | 0.923 | 0.870 | **zero effect on its target (EEG's candidate set was byte-identical with/without the expansion query)** - hit@5 unchanged, the aggregate hit@1/mrr drop is most likely Qwen API run-to-run noise on the 187 untouched rows, not a real effect (see "Known risks") |

Common thread: every attempt to fix the ~5-8% "gold never retrieved"
category either failed to change retrieval at all (abbreviation
expansion) or fixed it at a net cost to hit@1 (pool widening, LLM
guesses) - this category looks structurally resistant to query-side
fixes with the current embedder; a lexical (BM25-style) signal specific
to these terms, or a different embedder entirely, are the more promising
untried directions (see "Most promising next experiments").

### Architecture comparison: fixed-role pipeline vs. dynamic single agent (2026-08-03)

The main pipeline (above) is a fixed-role multi-agent system: generator
agent -> deterministic retrieval/fusion -> reranker agent, always in that
order. This project also has a genuinely dynamic alternative,
`src/agents/tool_agent.py` (`agent_tool_calling_e5[_rerank_qwen]` in
`src/methods.py`): one LLM decides at runtime, per term, how many times to
call a `retrieve_candidates` tool and with what queries before finalizing
- the search strategy isn't hard-coded. Full writeup, methodology, and
interpretation: `METHODOLOGY.md` §6. Headline result (val, n=194, both
using `e5-base-v2` retrieval + `qwen/qwen3-32b` reranking, matched
downstream stage):

| architecture | hit@1 | hit@3 | hit@5 | mrr | avg latency/term |
|---|---|---|---|---|---|
| fixed pipeline (`llm_expansion_weighted_t0_rerank_qwen`) | **0.851** | **0.912** | **0.923** | **0.882** | **9.08s** |
| dynamic agent (`agent_tool_calling_e5_rerank_qwen`) | 0.830 | 0.907 | 0.907 | 0.866 | 18.96s |

The fixed pipeline wins on every metric and is ~2x faster. Unlike the
noise-only deltas found for the three discarded methods just above, this
one is confirmed real: 0/10 inspected regressions share an identical
candidate set with the fixed pipeline's output, meaning the agent's
self-chosen search queries genuinely change what reaches the reranker.
The trade is concrete, not just "agent worse": the dynamic agent recovers
5 gold synonyms the fixed pipeline's retrieval misses entirely (e.g.
`EEG`->"brainwave recording"), but regresses 14 already-correct rows and
worsens 1 more into a miss - net -4/194. A candidate-pool fusion of both
architectures before a single shared reranking pass is flagged as an
untried follow-up (`METHODOLOGY.md` §8).

### Failure-pattern analysis by term type (2026-07-22, advisor-requested)

`scripts_failure_analysis.py` splits every one of the 172/969 full-dataset
non-rank1 cases into **ranking error** (gold in top-5, wrong order - the
reranker saw it and misjudged) vs **retrieval miss** (gold never retrieved
- no reranker can fix it), then breaks both down by term type (same
heuristic categories as `scripts_error_breakdown.py`: abbreviation-like,
compound, multi-word phrase, single word):

| category | n | ranking-error rate | retrieval-miss rate |
|---|---|---|---|
| abbreviation_like | 34 | 5.9% | 8.8% |
| compound | 432 | 11.3% | 3.2% |
| multi_word_phrase | 250 | 12.8% | 4.4% |
| **single_word** | 253 | **14.2%** | **9.9%** |

**Single-word terms are the weakest category on both failure types** -
nearly double the retrieval-miss rate of compounds/phrases, and the
highest ranking-error rate. This matches the pre-rerank finding from
2026-07-20 (single-word hit@1 was the worst category there too) - the
strong reranker narrowed the gap but did not close it. Gold-rank
distribution among ranking errors confirms most are near-misses: 87/119 at
rank 2, 19 at rank 3, 11 at rank 4, 2 at rank 5.

Manual inspection of the 53 retrieval-miss examples (full list in
`results/eval/failure_analysis_full_by_category.csv`) surfaces two
recurring, distinct patterns beyond generic "hard cases":

- **Domain jargon / register mismatch**: gold is a much more general or
  colloquial word than the technical input term, e.g. `philosophy` → gold
  `wisdom`, `gospels` → gold `good news`, `communication` → gold
  `information exchange`. The embedder's neighborhood around the technical
  term simply doesn't reach these registers.
- **Abbreviation direction is bidirectional, not one-way**: the earlier
  abbreviation-expansion attempt (discarded) only handled the case where
  the *input* is the abbreviation (`EEG`, `evs`). The full-dataset data
  shows the **reverse** also happens and is untried: the *gold* is the
  abbreviation while the input is the expanded form - e.g. `identity
  management` → gold `IdM`, `information technology` → gold `IT`. A
  query-side fix aimed only at abbreviation *inputs* structurally cannot
  catch these.
- Ranking-error examples at gold-rank 2 are almost all defensible
  near-synonyms rather than mistakes (`investment`→"investing" ranked
  below "funding"; `mediation`→"conflict resolution" ranked below "dispute
  resolution"; `sustainable tourism`→"eco-friendly tourism" ranked below
  "eco-tourism") - suggesting a meaningful chunk of the remaining 12.3%
  ranking-error rate reflects genuine paraphrase ambiguity in the gold
  labels themselves, not a fixable model deficiency.

Reproduce: `python scripts_failure_analysis.py` (reads the existing
`results/eval/error_analysis_rerank_qwen_full.csv`, no new API calls).

### Failure-pattern analysis v2: gold rank before/after reranking, more dimensions (2026-07-22)

`scripts_failure_analysis_v2.py` extends the analysis above with the gold
rank **before** reranking (`llm_expansion_weighted_t0`'s pre-rerank top-5,
`results/eval/llm_expansion_weighted_t0__full.csv`) matched row-by-row
against the rank **after** (`llm_expansion_weighted_t0_rerank_qwen`), plus
three dimensions beyond term category: lexical (token) overlap between
term and gold, a morphological-stem-sharing proxy, and a genuine
reverse-abbreviation flag (gold is an actual acronym, not just a short
word). Full dataset, n=969:

**Rank movement caused by reranking** (of the 916 rows where gold was
retrieved pre-rerank at all):

| | count | rate |
|---|---|---|
| improved (moved closer to rank 1) | 136 | 14.8% |
| unchanged | 732 | 79.9% |
| **worsened (moved away from rank 1, incl. lost from top-5)** | **48** | **5.2%** |

Reranking is a clear net positive (136 vs 48), but it is not free: **43 of
the 48 regressions are already-correct rank-1 predictions that the
reranker talked itself out of** (e.g. `computer simulation`→"computational
modeling" ranked 1st pre-rerank, demoted to 2nd; `christian
liturgy`→"Christian worship" ranked 1st, demoted to 2nd). This is the
full-dataset confirmation of the same effect seen on val (3/179 rank-1
regressions) and directly explains why "confidence-gated rerank" (below)
- which still reranks *all 5* candidates once triggered - could not avoid
this cost: the reranker mis-ordering an already-correct top-1 doesn't
require an ambiguous fusion margin, it's a reranker judgment error
independent of retrieval confidence.

**Lexical overlap is the strongest single correlate of failure found in
this project**: rows with zero shared tokens between term and gold have
close to double the failure rate of rows with high token overlap
(retrieval-miss rate 8.9% vs 1.7%, ranking-error rate 16.0% vs 6.9%). The
morphological-stem-sharing proxy shows the same direction, more weakly.

**Reverse-abbreviation, corrected**: the naive heuristic (gold is a single
token, uppercase or ≤5 characters) flags 8 full-dataset rows, but manual
check shows only 3 are genuine acronyms - `identity management`→"IdM",
`information technology`→"IT", `International Criminal Court`→"ICCt" (all
3 retrieval misses, gold never in top-5 either before or after
reranking). The other 5 (`anxiety`→"worry", `religion`→"faith",
`gender`→"sex", `motivation`→"drive", `solidarity`→"unity") are just
short ordinary words, not abbreviations - the cheap heuristic
over-triggers and this is disclosed rather than silently miscounted. 3/969
is too small a fraction to move any aggregate metric, but each of the 3
genuine cases is a clean, mechanistically-understood failure (see the
discarded-methods table below - all three targeted fixes tried for these
patterns, including the surgical exact-match follow-up, came back as
reranker noise under per-row candidate-set verification, not a real
effect).

Reproduce: `python scripts_failure_analysis_v2.py --pre results/eval/llm_expansion_weighted_t0__full.csv --post results/eval/llm_expansion_weighted_t0_rerank_qwen__full.csv --label full`

### Confidence-gated rerank (tried, underperformed blanket rerank)

Hypothesis going in: reranking only *ambiguous* terms (small margin
between the fused top-1 and top-2 scores) would avoid disturbing
already-confident correct predictions, unlike the two earlier discarded
"rerank everything with a weak model" attempts. Margin threshold
(`GATE_THRESHOLD = 0.01`, `src/methods.py`) was chosen offline from a
zero-cost margin analysis on `llm_expansion_weighted_t0`'s val predictions
(62/194 terms flagged ambiguous, capturing 28/42 of the wrong rank-1s
while only touching 34/152 of the correct ones).

| method | val hit@1 | val mrr | val avg_time_s | API calls/194 rows |
|---|---|---|---|---|
| `llm_expansion_gated_rerank_qwen` (ambiguous only) | 0.835 | 0.873 | 3.78 | ~62 |
| **`llm_expansion_weighted_t0_rerank_qwen` (everyone, winner)** | **0.851** | **0.882** | **9.08** | **194** |

Reranking *everyone* wins by +1.5pt hit@1 / +0.9pt MRR despite touching
3x more rows - the strong model apparently improves genuinely-confident
cases too (not just fixes ambiguous ones), which the gating hypothesis
didn't anticipate. Kept as a documented negative result, not deleted -
`llm_expansion_gated_rerank_qwen` stays in `ALL_METHODS` since the code and
threshold-selection methodology may be useful if a future, even-stronger
or cheaper model changes this balance.

### Previous best (`llm_expansion_weighted_n3`, no LLM reranking)

Kept as a fast, $0-cost fallback: same query-expansion + weighted-fusion
pipeline as above, but temp=0 generation with only **3 paraphrases**
instead of 9 (`generar_candidatos(term, n=3)`) - fewer, more targeted
variants beat more of them, most likely because each additional generated
variant is another chance for a noisy paraphrase to outrank the true match
in the weighted fusion. Fusion weight unchanged (original term weighted 3x
vs each variant, `LLM_EXPANSION_WEIGHTED_T0_WEIGHT = 3.0`).

| split | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|
| val (n=194) | 0.789 | 0.902 | 0.928 | 0.846 | 0.856 | 0.867 | 0.71 |
| test (n=194, held out) | 0.768 | 0.902 | 0.933 | 0.835 | 0.847 | 0.859 | 0.72 |

vs. the previous best (`llm_expansion_weighted`, temp=0.3, n=9) on the same
held-out test split (hit@1 0.753, mrr 0.823, hit@3 0.881, hit@5 0.938): on
the user's stated priority order (Hit@1 first, then MRR) `n3` wins clearly
(**+1.5pt hit@1, +1.2pt MRR, +2.1pt hit@3**), at the cost of a small,
real hit@5 regression (**-0.5pt**, 0.933 vs 0.938 - a genuine tradeoff, not
noise: confirmed via the isolated `llm_expansion_weighted_t0` (n=9, same
temp=0 generator) test run, hit@5 0.9485, so the hit@5 loss is specifically
an n=3-vs-n=9 effect, not a temperature effect). Cost: **$0** (local Ollama
only). Latency ~0.72s/term, same ballpark as before (fewer variants doesn't
meaningfully reduce latency - the generation call itself dominates, not the
embedding step).

**Caveat on the n-variants finding**: an offline sweep script (reusing one
n=9 generation per term, then truncating to simulate smaller n) suggested a
larger gain (+2.05pt hit@1) than what the *official* eval harness confirmed
when calling `generar_candidatos(term, n=3)` directly (+0.52pt hit@1 on
val, stable across 2 repeat official runs). Both calls are prompt-identical
so should be deterministic and equivalent at `temperature=0`, but weren't -
a reminder that even temp=0 local Ollama generation isn't perfectly
reproducible **between separate process runs** (it was, however, perfectly
reproducible **within** repeated sweep-script runs in the same style). The
promotion decision here rests on the officially-confirmed, twice-repeated,
smaller number (Pareto-dominant over n=9 on every val metric), not the
larger sweep estimate.

### Previous best (`llm_expansion_weighted`, temp=0.3, n=9)

| split | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|
| val (n=194) | 0.773 | 0.907 | 0.938 | 0.845 | 0.856 | 0.868 | 0.73 |
| test (n=194, held out) | 0.753 | 0.881 | 0.938 | 0.823 | 0.828 | 0.851 | 0.74 |

`llama3.2:3b` generates up to 9 paraphrases of the term (always including
the original term itself); each of the ~10 queries is embedded with
`e5-base-v2` and searched against the candidate vocabulary; scores are
fused with a **weighted average that trusts the original term 3x more than
any single generated variant**; top-5 returned. No LLM reranking - two
separate methods that added an LLM rerank pass on top of good retrieval
(one on this method, one on plain `embedding_e5`) were tested and both
added zero benefit (see below), so the pipeline stays generate → retrieve →
fuse, nothing after.

vs. previous-best `embedding_e5` (single-query e5 retrieval, no LLM) on the
same held-out test split: hit@1 0.603, mrr 0.704, hit@5 0.871 - **+15.0pt
hit@1, +11.9pt MRR, +6.7pt hit@5**. vs. the original `all-MiniLM-L6-v2`
baseline: **+21.7pt hit@1** (0.536 → 0.753). Cost: **$0**.

**Caveat**: this config's generator ran at `temperature=0.3`, so its own
numbers have real run-to-run variance (~1-2pt hit@1 across 3 repeat runs,
MRR more stable at ~0.3pt). The categorical choice - weight the original
term's query above generated variants - is robust and reproduces every
time; the exact weight value (3.0 vs 5.0 vs other nearby values) is not
reliably distinguishable from that sampling noise at n=194, so a defensible
round value (3.0) was kept rather than chasing a marginal, noisy "optimum".
See `EXPERIMENTS_LOG.md` (2026-07-19 "noche" entry) for the full stability
analysis, and the 2026-07-20 entry for how `temperature=0` fixed this.

### What was tried and discarded on top of this

| method | hit@1 (val) | mrr (val) | avg_time_s | verdict |
|---|---|---|---|---|
| LLM expansion, `max` fusion | 0.655 | 0.743 | 0.76 | below no-expansion baseline - one noisy variant can outrank the true match |
| LLM expansion, `mean` fusion | 0.753 | 0.822 | 0.74 | good, but weighted still wins |
| LLM expansion, `weighted` fusion, temp=0.3 (old winner) | 0.773 | 0.845 | 0.73 | superseded by temp=0, n=3 (current best, above) |
| LLM expansion, `rrf` fusion | 0.629 | 0.728 | 0.72 | worst - same weakness as BM25/RRF fusion below |
| + LLM rerank on top of `weighted` | 0.773 | 0.842 | 13.1 | net 0 rank-1 change vs. no rerank, 18x slower - discarded |
| plain `embedding_e5` + LLM rerank (no expansion) | 0.691 | 0.773 | 4.4 | **identical predictions to no-rerank, 0/194 changed**, 180x slower - discarded |
| `llm_expansion_weighted`, embedder=`mxbai-embed-large-v1` (temp=0, n=9) | 0.753 | 0.827 | ~0.71 | below e5-base-v2 (0.784/0.842) - discarded |
| `llm_expansion_weighted`, embedder=`gte-base-en-v1.5` | - | - | - | can't load in this environment - custom rotary-embedding remote code crashes (IndexError in position_ids) on both GPU and CPU, unrelated to model quality |
| Weighted (non-RRF) score fusion, BM25 + e5, any weight 0.1-2.0 | 0.43-0.68 | 0.58-0.77 | ~0.03 | every tested weight underperforms plain `embedding_e5` (0.691/0.773) - confirms the earlier RRF finding with a different fusion mechanism, BM25 signal too weak/noisy for this task - discarded |
| `llm_expansion_weighted_t0` + `qwen3.5:2b` local rerank, matched base (2026-07-22) | 0.768 | 0.837 | 13.2 | **worse than no rerank at all** (base: 0.784/0.842) - controlled comparison (same upstream top-5 as the qwen3-32b winner) shows a too-small reranker actively hurts rank-1, not just "no benefit" - see `METHODOLOGY.md` §5 for the full model-size table |
| `llm_expansion_rerank_qwen_initialism` (2026-07-22: acrónimo determinista como query extra en la fusión ponderada, apunta a reverse-abbreviation) | 0.866 | 0.893 | 13.3 | **looks positive but isn't real** - net rank-1 +3, but per-row candidate-set verification shows all 5 status-changed rows had a candidate set byte-identical to baseline; the mechanism never altered the pool for any row whose outcome flipped (it did change 36/194 pools, none of which flipped outcome) - the +3 is 100% Qwen-API non-determinism, not the method. Discarded; see surgical follow-up below |
| `llm_expansion_lexical_gated_rerank_qwen` (2026-07-22: inyecta hasta 2 candidatos BM25 cuando el top-5 tiene solape léxico cero con el término) | 0.830 | 0.872 | 12.5 | trigger condition fires on 50/194 rows but only 2 ever changed final status (1 fixed, 1 regressed - net zero from the mechanism itself); the aggregate -4 net rank-1 is dominated by API noise on untouched rows, not caused by the method - discarded (too-frequent trigger for a null real effect) |
| `llm_expansion_rerank_qwen_initialism_exact` (2026-08-03: exact-string-match acronym injection into the pool, surgical follow-up to the query-extra variant above) | 0.835 | 0.874 | 12.7 | same verdict again - net rank-1 -3 (1 fixed, 4 regressed), all 5 status-changed rows have a candidate set byte-identical to baseline, including the case where the injection had been separately confirmed offline to fire. Discarded - third and final targeted fix for the reverse-abbreviation/lexical-overlap patterns to come back as pure reranker noise |

### Previous best (`embedding_e5`, kept as a fast fallback)

| split | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|
| val (n=194) | 0.691 | 0.840 | 0.907 | 0.773 | 0.779 | 0.806 | 0.024 |
| test (n=194, held out) | 0.603 | 0.789 | 0.871 | 0.704 | 0.711 | 0.746 | 0.025 |

Still worth using when latency/cost must stay near-zero (e.g. no LLM
available) - it beat everything except the LLM-expansion family, at 1/30th
the latency of the new winner.

See "Full-dataset report" above (under "Best configuration found") for the
969-row descriptive numbers on the current best config - this
temp=0.3/n=9, no-rerank config's own full-dataset numbers are in the table
there too, for direct comparison.

## Reproduce

```bash
python data/make_splits.py                                # regenerate splits (seed=42, deterministic)
python run_eval.py --method embedding_only --split test    # original baseline
python run_eval.py --method embedding_e5 --split test      # previous-best, no LLM
python run_eval.py --method llm_expansion_weighted_n3 --split test  # best $0/local-only config
python run_eval.py --method llm_expansion_weighted_t0_rerank_qwen --split test  # current best overall (needs OPENROUTER_API_KEY, ~$0.03/run)
python run_eval.py --method llm_expansion_weighted_t0_rerank_qwen --split full --resume  # full-dataset descriptive report (~$0.18, ~2h; --resume makes it safe to interrupt/retry)
python run_eval.py --method <name> --split val             # any method in src/methods.py::ALL_METHODS
python compare_methods.py --method <name> --split val      # rank-movement vs embedding_e5 baseline
python scripts_weight_sweep_t0.py                           # fusion-weight sweep at temp=0 (documentation)
python scripts_variant_count_sweep.py                       # n-variants sweep at temp=0 (documentation)
python scripts_error_breakdown.py                            # per-category error analysis on val
python scripts_margin_analysis.py                            # margin analysis for the gated-rerank threshold (documentation)
```

Detailed per-row output and JSON summaries land in `results/eval/`; every
run also appends one row to `results/experiment_log.csv` (method, split,
all 6 metrics + absolute hit counts, latency, timestamp - never truncated).

`run_experiment.py` and `run_synonym_finding.py` are the original
entry points (predate this experimentation cycle) and still work - they
run the LLM generator+reranker pipeline directly against `all-MiniLM-L6-v2`,
useful as a standalone demo of that pipeline, but `run_eval.py` is now the
harness for anything comparative (consistent splits, consistent logging).

## Full results and error analysis

See `EXPERIMENTS_LOG.md`:
- 2026-07-19 "tarde" entry: the complete comparison table across 21 methods
  on val + test; why cross-encoder reranking, BM25 fusion, morphological
  query variants, a larger e5 model, and an e5+bge ensemble were all tried
  and discarded; why plain LLM methods (zero-shot, tool-calling, the
  original generate+rerank pipeline) underperform plain `embedding_e5`.
- 2026-07-19 "noche" entry: the 4 fusion strategies tested for LLM query
  expansion (`max`/`mean`/`weighted`/`rrf`), the fusion-weight sweep and
  stability analysis, why LLM reranking on top of good retrieval was tested
  twice more (Methods 2 and 3) and discarded both times with clean
  zero-benefit evidence, concrete before/after examples for every method,
  and the full test-set confirmation.
- 2026-07-20 entry (current winner): the `temperature=0` retune (clean,
  byte-identical weight sweep across 2 repeats), the n-variants sweep
  (n=3 beats n=9) and the sweep-script-vs-official-harness discrepancy it
  surfaced, the alternate-embedder attempts (`mxbai` discarded, `gte`
  couldn't load), the weighted BM25+e5 fusion attempt (discarded), the
  per-category error breakdown, the Qwen/OpenRouter reranker (blanket
  rerank with `qwen3-32b` - the actual current best, +8.3pt hit@1 over
  everything else), why `qwen3.5-9b` was swapped for `qwen3-32b`
  mid-session (reliability), the confidence-gated rerank variant that
  underperformed blanket reranking, and the val+test confirmation with
  rank-movement examples.
- 2026-07-20 (continuación 2) / 2026-07-21 entries: the gold-in-top5-vs-
  outside-top5 error breakdown, three more discarded attempts to close the
  retrieval-miss gap (widen pool to 10/15, merge LLM direct-guess
  candidates - all hurt hit@1 more than they helped hit@5), the OpenRouter
  free-tier credit exhaustion mid-cycle and how it was diagnosed (a run's
  numbers matched the no-rerank baseline exactly), the resulting
  `reranker_fallback` tracking + `--resume`/checkpointing infrastructure
  added to `run_eval.py`, and the confirmed-clean full-dataset run (969/969
  rows, 0 fallbacks, 0 retries needed) after credits were restored.

## Known risks / caveats

- n=194 per split is enough to trust a +8-15pt signal but not to rank
  methods that differ by 2-3pt (several methods cluster closely - within
  plausible noise at this sample size). The current best's test-set gain
  over the previous best is in exactly that marginal range (+1.5pt hit@1)
  - real and reproducible (confirmed via 2 identical repeat runs on val),
  but small.
- Even at `temperature=0`, local Ollama generation is not perfectly
  reproducible **across separate process runs** (though it was perfectly
  reproducible **within** repeated same-session sweep runs) - see the
  n-variants sweep-vs-harness discrepancy above. Any future tuning that
  relies on a single run's numbers should budget for ~0.5-1pt hit@1 slop
  even without `temperature=0.3`'s larger noise.
- The same applies to the Qwen reranker (`temperature=0.0`) across
  *separate* API calls/runs: the abbreviation-expansion experiment
  (2026-07-21) found 8 rows regressed and 3 improved on completely
  untouched, non-abbreviation terms between two full val runs of what
  should have been byte-identical retrieval - i.e. the reranker itself
  gave a different ordering for the same term+candidates input on a
  second run. Budget a few points of hit@1/mrr slop on any single-run
  comparison of two Qwen-reranked methods; a real repeat-run stability
  check (like the one done for the local generator) has not yet been done
  for the reranker specifically.
- Splits are grouped by unique `en` term (not by row) and regenerated
  deterministically from `data/synonyms_clean.csv` - re-run
  `data/make_splits.py` if the source CSV ever changes, don't hand-edit
  `data/splits/*.csv`.
- `gte-base-en-v1.5` cannot be loaded in this environment (custom
  remote-code rotary-embedding implementation crashes with an `IndexError`
  in `position_ids`, both on GPU and CPU) - this is an environment/library
  compatibility issue, not evidence about the model's quality.
- The current best method depends on a paid external API (OpenRouter,
  `qwen/qwen3-32b`) - unlike every other method in this project, it costs
  real money (~$0.03/194-row run) and has external-service reliability
  risk. Two real infra issues were found and fixed along the way:
  1. `openrouter_client.py`'s OpenAI client had no `timeout`/`max_retries`
     configured, so a single stalled request could hang indefinitely
     (confirmed - a smoke-test run hung 27+ minutes with no CPU progress
     before being killed manually); fixed with `timeout=90.0,
     max_retries=3`.
  2. The originally-wired model, `qwen/qwen3.5-9b`, returned a transient
     "Service unavailable" error on ~2 of 3 calls in direct testing - but
     as an embedded `error` field inside an HTTP-200 response, not an HTTP
     error, so the OpenAI client's automatic retry logic never caught it.
     Fixed two ways: added explicit retry logic for this specific pattern
     (`openrouter_client.py::_completar_con_reintentos`, 3 attempts, 2s
     apart), and switched the model to `qwen/qwen3-32b` (5/5 reliable in
     direct testing, faster, multiple providers on OpenRouter). If this
     model is ever swapped again, re-run the direct reliability check
     before trusting a full val/test run's numbers - a flaky model would
     silently under-count how often reranking actually happened (a failed
     call falls back to the unreranked order, not a crash, so a bad model
     wouldn't error out, just quietly underperform).
- `llm_expansion_gated_rerank_qwen` is a genuine negative result (blanket
  reranking wins), not an untested one - see "Confidence-gated rerank"
  above. Kept in `ALL_METHODS` for the code/methodology, not as a
  recommended method.
- **OpenRouter free-tier credits ran out entirely mid-cycle** (2026-07-20,
  a ~$0.19 free allowance exhausted by cumulative experimentation) and
  every reranker call silently fell back to the unreranked order - a run
  that "completed successfully" with metrics identical to no-rerank at
  all. This is now impossible to miss: every `run_eval.py` row records
  `reranker_fallback` (True if that row's LLM call failed and fell back),
  the run summary/log include `reranker_fallback_count`, and `run_eval.py`
  now **refuses to report a result** if any row is still stuck in fallback
  after `--max-fallback-retries` (default 3) automatic retries - it raises
  instead of silently mixing valid and corrupted rows. Long runs also
  checkpoint incrementally (one CSV write per row) and support `--resume`
  to reuse already-confirmed-valid rows and only recompute the rest -
  check current balance any time with:
  `GET https://openrouter.ai/api/v1/credits` (needs `OPENROUTER_API_KEY`
  as a Bearer token).

## Most promising next experiments

**Note (2026-07-22)**: the advisor reviewed these results and recommended
shifting focus away from further Hit@1 optimization for now (results are
already strong) toward (a) failure-pattern analysis by term type - done,
see "Failure-pattern analysis by term type" above, (b) a reranker
model-size/latency/performance comparison - done, see `METHODOLOGY.md`
§5, and (c) drafting the methodology/experimental-setup/baseline sections
of the report - started, see `METHODOLOGY.md`. Items 1, 2, 5, 8, 9, 10
below (further Hit@1-chasing ideas) are deprioritized as a result and kept
only for future reference; items 3, 4, 6, 7 (robustness/analysis, not
new accuracy pushes) remain relevant if picked back up later.

1. Try `llm_expansion_weighted_qwen` (Qwen as the query-*generator*, not
   just the reranker - implemented, model string already updated to
   `qwen3-32b`, but never run) - combining a stronger generator with the
   now-confirmed strong reranker could stack further gains, though it adds
   a second paid-API dependency per term (currently only the rerank step
   costs money).
2. Try a mid-size Qwen model (e.g. `qwen/qwen3-14b` or `qwen/qwen3-8b`) as
   the reranker instead of `qwen3-32b` - the gated-rerank result suggests
   the current model is strong enough that even "obviously correct" cases
   improve; a smaller/cheaper/faster model might capture most of the
   +8.3pt hit@1 gain at a fraction of the ~9s/term latency, worth an A/B
   before assuming 32B is the minimum useful size.
3. Re-run splits with 2-3 different seeds and average, to separate real
   signal from small-n variance before ranking close methods.
4. Try `bge` instead of `e5` as the per-query embedder inside
   `llm_expansion_weighted_n3`/the rerank pipeline (only `mxbai` and `e5`
   have been compared inside the expansion step so far; `gte` is
   untestable here).
5. The n-variants sweep only tried {3,5,7,9} - narrower values (n=1, n=2)
   not yet tested, to see if the trend (fewer variants = better) continues
   or reverses once there's too little query diversity. Worth re-checking
   with the reranker downstream too, since `llm_expansion_weighted_t0`
   (n=9, not n=3) was what got reranked to produce the current best - it's
   not yet confirmed whether reranking n=3's candidates instead would do
   even better or worse.
6. Use the untouched `train` split (581 rows) to calibrate any future
   fusion weights or thresholds instead of hand-picking them against val.
7. Error analysis by term category is now done (see
   `scripts_error_breakdown.py` / `EXPERIMENTS_LOG.md` 2026-07-20 entry) -
   the weakest category is single-word terms (hit@1 0.66 vs 0.82-0.86 for
   compounds/multi-word phrases/abbreviations) in the pre-rerank baseline;
   worth re-running against the reranked results to see if the strong
   reranker closes this category gap too.
8. The 119 full-dataset "wrong rank" cases (gold in top-5, not rank 1) are
   the main remaining lever now that retrieval-miss fixes have failed
   twice - 87/119 are at rank 2 specifically, meaning a small nudge in
   reranker judgment would fix most of them; worth trying a second
   reranker pass or self-consistency (multiple reranker calls + majority
   vote) targeted at just the close top1-vs-top2 calls, val-only first.
9. Abbreviation expansion (tried 2026-07-21, discarded - see "What was
   tried and discarded") added the expansion as a weight=1.0 query
   alongside the other ~9 generated variants, which turned out to have no
   measurable pull on the fusion at all (EEG's retrieved candidates were
   byte-identical with/without it). A follow-up worth trying: give the
   expansion a MUCH higher weight (comparable to or above the original
   term's 3.0, since for a genuine abbreviation the expansion arguably
   carries more signal than the raw acronym does), or route it through a
   separate lexical (BM25) lookup instead of the semantic embedding fusion
   - BM25 was weak in general but might specifically help here, since
   abbreviation expansions vs their gold synonyms can share exact
   vocabulary that embeddings miss.
10. Confirm the Qwen-reranker non-determinism finding (see "Known risks")
    with a proper repeat-run stability check, same approach used earlier
    for the local generator - run the current best config on val 2-3
    times and measure the row-level and aggregate variance, before trusting
    any future single-run comparison between two Qwen-reranked methods.





● Best confirmed method: llm_expansion_weighted_t0_rerank_qwen

  Val: hit@1 0.8505, hit@3 0.912, hit@5 0.923, MRR 0.882 (165/194 exact top-1)
  Test: hit@1 0.8299, hit@3 0.938, hit@5 0.948, MRR 0.884 (161/194 exact top-1, 2026-07-22 fallback-verified re-confirmation)

  Workflow (src/methods.py, dispatch block for this method name):

  1. Query expansion — generar_candidatos(term, agente=AGENTE_GENERADOR_T0): local Ollama (llama3.2:3b, temperature=0) generates up to 9 paraphrases/related terms for the input; the original term is always prepended, giving up to 10 queries total.
  2. Embed — encode_queries(...): all 10 queries embedded with intfloat/e5-base-v2 (with the "query: " prefix e5 was trained with).
  3. Weighted fusion retrieval — combine_scores("weighted", ...): each query is scored (cosine) against the full candidate vocabulary (every en_synonym value in the dataset — a fixed pool, not the row's own answer). Scores are combined as a weighted average, with the original term's query weighted 3x over
  each generated variant (LLM_EXPANSION_WEIGHTED_T0_WEIGHT = 3.0) — variants are useful hints but shouldn't outvote the real input. Top-5 candidates come out of this step.
  4. Rerank — reordenar_candidatos_qwen(term, top5) (src/agents/reranker_agent.py): the term + those exact 5 candidates are sent to qwen/qwen3-32b via OpenRouter, asked to reorder them best-to-worst. It can only reorder — never invents new candidates, never sees the gold answer. If it fails or omits
  something, the untouched candidates are appended back so recall is never lost.
  5. Return the reranked top-5.

