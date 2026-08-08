# Methodology, Experimental Setup, and Baseline Comparison (draft)

Draft sections for the thesis report, started 2026-07-22 per the advisor's
request to begin writing while the pipeline details/metrics are fresh.
Cross-references `README.md` (current results/config) and
`EXPERIMENTS_LOG.md` (full experiment history, including discarded
attempts) rather than duplicating their content - this document is the
narrative write-up, those are the working logs.

## 1. Task formulation

Given an English academic/domain term `en`, the task is to identify its
correct synonym from the `TU-Expert-Collection-Topic-Synonyms` dataset,
evaluated against a gold label `en_synonym` per row.

Two formulations of "synonym finding" are possible: (1) open generation,
where a system produces a synonym from scratch and `en_synonym` is used
only to score the output post-hoc, or (2) ranking, where the system
selects/orders candidates from a predefined pool built from `en_synonym`
and `en_synonym` is both the search space and the evaluation target. This
project implements (2) throughout. Even the methods that free-generate a
proposal with an LLM before any retrieval (`llm_zero_shot`,
`llm_zero_shot_e5`, `src/agents/zero_shot_agent.py`) do not return that
raw text as the final answer - the free-text proposal is used only as an
additional retrieval query against the candidate pool, and the closest
real pool entry is returned. This choice was made because (a) scoring
open-ended generation against one gold string per row has no well-defined
metric without some matching/ranking step regardless, and (b) empirically,
ranking against the true candidate vocabulary outperforms free generation
by a wide margin (see §4). A global candidate vocabulary built from every
`en_synonym` value in the dataset is used throughout; this is not
label leakage, since no method looks up the gold answer of the specific
row being predicted before producing its prediction - only the fixed,
row-independent vocabulary is available at prediction time.

## 2. Dataset

`TU-Expert-Collection-Topic-Synonyms` (HuggingFace:
`jensjorisdecorte/TU-Expert-Collection-Topic-Synonyms`), reduced in this
project to its two usable columns, `en` (input term) and `en_synonym`
(gold synonym) - 969 rows after removing one degenerate row where
`en == en_synonym` (`modernisation`/`modernisation`; keeping it would
guarantee a false negative on every method, since the exclude-self logic
in retrieval removes the correct answer from its own candidate pool by
construction). Ten `en` terms appear twice with two different gold
synonyms; the split procedure below keeps every occurrence of a given
term in the same split so no input term is seen across splits.

## 3. Experimental setup

### 3.1 Data splits

`data/make_splits.py` partitions the 969 usable rows into `train` (581,
60%), `val` (194, 20%), and `test` (194, 20%), grouped by unique `en` term
with a fixed seed (42) for reproducibility. No component in this project
is trained with gradient descent, but the splits are still necessary:
choosing among candidate pipelines and tuning their few numeric knobs
(embedder choice, fusion strategy and weight, generator temperature and
variant count, reranker model, confidence-gating threshold) is itself a
form of fitting to data, with the same overfitting risk as tuning
hyperparameters of a trained model. Concretely:

- **`val`** is where every pipeline comparison and knob choice was made
  across the project's experimentation cycles (~40 candidate methods
  compared; see `EXPERIMENTS_LOG.md`).
- **`test`** is touched exactly once per already-decided configuration, to
  report an unbiased number - never to choose between two configurations.
- **`train`** is currently unused by any method (reserved for a future
  component that would need real fitting, e.g. calibrating the rerank
  confidence threshold or fine-tuning an embedder).
- **`full`** (train+val+test combined, 969 rows) is used exactly once, at
  the end, purely for descriptive reporting/error analysis on the
  already-locked-in best configuration.

### 3.2 Pipeline architecture

The best-performing pipeline (`llm_expansion_weighted_t0_rerank_qwen`,
`src/methods.py`) is a four-stage retrieve-then-rerank system:

1. **Query expansion** - a local LLM (`llama3.2:3b` via Ollama,
   `temperature=0`) generates up to 9 paraphrases/related terms for the
   input; the original term is always included, giving up to 10 search
   queries per row.
2. **Embedding** - every query is encoded with `intfloat/e5-base-v2`
   (using the `"query: "` prefix e5 was trained with).
3. **Weighted fusion retrieval** - each query is scored by cosine
   similarity against the full candidate vocabulary; scores are combined
   as a weighted average with the original term's query weighted 3x over
   each generated variant, producing a top-5 shortlist.
4. **LLM reranking** - the term and its top-5 shortlist are sent to
   `qwen/qwen3-32b` (OpenRouter API), which reorders the 5 candidates
   best-to-worst. The reranker can only reorder the given candidates -
   it never invents new ones and never sees the gold answer; if its
   response omits a candidate, the omitted item is appended back so
   recall from step 3 is never lost.

### 3.3 Evaluation metrics

`src/evaluation.py` reports, per prediction and aggregated per run:
Hit@1/3/5 (binary: is the gold synonym among the top-k predictions),
MRR (mean reciprocal rank of the gold synonym, 0 if absent from the
returned candidates), and NDCG@3/5 (rank-position-weighted gain,
Järvelin & Kekäläinen 2002 - see `REFERENCES.md`). Hit@k and NDCG@k are
reported together deliberately: Hit@k is a coarse binary signal, NDCG@k
additionally captures *how close to rank 1* a correct-but-imperfect
prediction landed, which is exactly the phenomenon this project's later
experiments targeted (moving gold candidates from ranks 2-5 into rank 1).

## 4. Baseline comparison

Progression of the best configuration found at each stage of the project
(test split, n=194, held out and never used for selection):

| method | hit@1 | hit@3 | hit@5 | mrr | avg latency/term |
|---|---|---|---|---|---|
| `embedding_only` (`all-MiniLM-L6-v2`, single query) | 0.536 | - | - | - | ~0.006s |
| `embedding_e5` (`e5-base-v2`, single query, no LLM) | 0.603 | 0.789 | 0.871 | 0.704 | ~0.025s |
| `llm_expansion_weighted_n3` (LLM query expansion + weighted fusion, no rerank) | 0.768 | 0.902 | 0.933 | 0.835 | ~0.72s |
| **`llm_expansion_weighted_t0_rerank_qwen`** (+ Qwen3-32B reranking, current best) | **0.830** | **0.938** | **0.948** | **0.884** | ~13.3s |

Each stage's gain is confirmed on the held-out test split, never chosen
using it (§3.1). The reranking row is the 2026-07-22 fallback-verified
re-confirmation run (`reranker_fallback_count: 0`); an earlier run of the
identical config (2026-07-20, predates fallback tracking) scored hit@1
0.851/mrr 0.892 at 9.3s/term - the two repeat runs of the same
`temperature=0` config differ by ~2pt hit@1 and ~4s/term, which is
real measured evidence of the Qwen-API non-determinism/latency-variance
risk noted in §8, not a config difference. The largest single gain in the
entire project (+6.2pt hit@1, using the re-confirmed number) came from
adding the LLM reranking stage with a sufficiently large model - two
earlier attempts at reranking with a much smaller model (`qwen3.5:2b`, 2B
parameters) found this stage can just as easily *hurt* accuracy (see §5
and `README.md`, "What was tried and discarded"), which is why isolating
the model-size dimension mattered.

## 5. Reranker model-size vs. latency/performance comparison

The advisor specifically asked for this comparison given the size of the
gain attributed to switching reranker models. Three model sizes were
tried as the reranking stage on top of an otherwise-identical retrieval
pipeline:

All three rows below rerank the exact same upstream top-5
(`llm_expansion_weighted_t0`'s output, val hit@1 0.784/mrr 0.842/0.74s
with **no** reranking) - isolating the reranker model's own contribution
from any retrieval-side difference:

| reranker model | params | where run | reliability | latency/term | val hit@1 | val mrr | Δ hit@1 vs. no rerank |
|---|---|---|---|---|---|---|---|
| *(no rerank, baseline)* | - | - | - | 0.74s | 0.784 | 0.842 | - |
| `qwen3.5:2b` | 2B | local (Ollama) | reliable (0 fallbacks), but reorders badly | **13.2s** | 0.768 | 0.837 | **-1.6pt (hurts)** |
| `qwen/qwen3.5-9b` | 9B | OpenRouter API | **unreliable** - ~2/3 calls returned a transient "Service unavailable" embedded in an HTTP-200 response (not a normal HTTP error, so the client's built-in retry didn't catch it) | N/A | N/A | N/A | discarded before producing usable numbers |
| `qwen/qwen3-32b` | 32B | OpenRouter API | 5/5 reliable in direct testing; 0 fallbacks across a 969-row full-dataset run | 9.08s | **0.851** | **0.882** | **+6.7pt (helps)** |

(Controlled 2B run: `llm_expansion_weighted_t0_rerank_local`,
`src/methods.py`, confirmed 2026-07-22, 194/194 rows, 0 fallbacks.)

Two findings stand out. First, **the 2B model actively makes rank-1
worse, not just "no better"** - on this matched base it is a genuine
regression (-1.6pt hit@1), not the "zero effect" the earlier,
mismatched-base comparison suggested; a small reranker doesn't just fail
to add signal, it can actively mis-order confidently-correct predictions.
Second, **the local 2B model is slower than the 32B API model** (13.2s
vs. 9.08s per term) despite being 16x smaller - almost certainly a
CPU-bound-hardware artifact of this environment's local Ollama inference
rather than a property of model size itself, but a reminder that "local
= faster" cannot be assumed without measuring it directly.

**Practical implication**: the reranking stage's benefit is not a generic
property of "adding an LLM reranker" - it required a sufficiently capable
model, and an insufficiently capable one is actively counterproductive.
The 9B tier was abandoned purely on reliability grounds before its
accuracy could even be assessed, which is itself a relevant finding for
any deployment decision: OpenRouter model availability/reliability varies
by model and must be spot-checked before committing an evaluation budget
to it, independent of expected accuracy.

## 6. Multi-agent architecture comparison: fixed-role pipeline vs. dynamic single agent

The pipeline described in §3.2 is a **fixed-role multi-agent system**: a
generator agent (query expansion), a deterministic retrieval/fusion stage,
and a reranker agent, always invoked in that order. This section reports
a controlled comparison against a **dynamic single-agent** alternative
already implemented in this project (`src/agents/tool_agent.py`,
`agent_tool_calling_e5[_rerank_qwen]` in `src/methods.py`): one LLM
(`llama3.2:3b`, native Ollama function-calling, ReAct/Toolformer-style -
see `REFERENCES.md`) decides at runtime, per term, how many times to call
a `retrieve_candidates` tool and with which query paraphrases, before
calling `finalize` with its ranked top-5. Unlike the fixed pipeline, the
number and content of retrieval calls is not hard-coded - the model
controls its own search process. A deterministic safety net (mirrored
from the reranker's own fallback design, §3.2) guarantees the agent never
starts from an empty pool and never invents a candidate that wasn't
actually returned by a real retrieval call.

**Motivation for isolating this comparison.** An earlier same-embedder
comparison (2026-07-19, `agent_tool_calling_e5` hit@1 0.660 vs
`hybrid_pipeline_e5` hit@1 0.613, both without the strong reranker) is not
a clean test of "fixed vs. dynamic architecture" - `hybrid_pipeline_e5`'s
fixed order includes the *local* 2B reranker already shown in §5 to
actively hurt rank-1 accuracy, so that comparison conflates architecture
with reranker quality. Both numbers are in fact below the reranker-free
`embedding_e5` baseline (hit@1 0.691, §4) - at that LLM budget, neither
architecture was adding value. To isolate the variable of interest, a new
method, `agent_tool_calling_e5_rerank_qwen`, was implemented: identical
dynamic agent, but its output top-5 is passed through the same
`qwen/qwen3-32b` reranker used by the current best fixed pipeline -
matching the downstream stage exactly and varying only the retrieval
architecture upstream of it.

**Result (val, n=194, both configurations use `e5-base-v2` retrieval and
`qwen/qwen3-32b` reranking, 2026-08-03):**

| architecture | hit@1 | hit@3 | hit@5 | mrr | avg latency/term |
|---|---|---|---|---|---|
| fixed pipeline (`llm_expansion_weighted_t0_rerank_qwen`) | **0.851** | **0.912** | **0.923** | **0.882** | **9.08s** |
| dynamic single agent (`agent_tool_calling_e5_rerank_qwen`) | 0.830 | 0.907 | 0.907 | 0.866 | 18.96s |

The fixed pipeline wins on every metric (+2.1pt hit@1, +1.6pt hit@5, +1.6pt
MRR) and is roughly **2x faster** - the dynamic agent pays for its own
planning turns (deciding whether/what to search again) in addition to the
same final reranking call both architectures share.

**This delta is real, not reranker noise** - unlike the three discarded
methods in §7.1, which were confirmed to be pure API sampling noise via
per-row candidate-set comparison. The same check applied here
(`scripts_method_comparison.py`, then manual candidate-set inspection on
every regressed row) shows the opposite: **0/10** inspected regressions
have a candidate set identical to the fixed pipeline's - the dynamic
agent's different search queries genuinely change what reaches the
reranker, in both directions. Net effect across all 194 rows: 5 retrieval
misses recovered (terms where the agent's self-chosen paraphrase found the
gold synonym and the fixed pipeline's fusion didn't, e.g. `EEG`→"brainwave
recording", `nonresponse`→"no response"), 7 ranking errors corrected, but
14 regressions and 1 case worsened into a miss - net -4 rank-1 vs. the
fixed pipeline. The regressions are not random noise either: they cluster
on terms with multiple plausible near-synonyms (`computer simulation`→
"computational modeling" demoted behind "simulation";
`democracy`→"popular government" lost entirely, replaced by
"Consensus-based democracy") - the agent's self-directed paraphrases
sometimes drift the search toward a plausible-but-wrong sense of the term
that a fixed, narrower expansion strategy avoids by construction.

**Interpretation.** Giving the LLM control over its own retrieval strategy
does surface genuinely different, occasionally better, candidates - the
50% miss-recovery-vs-not-found trade is a real capability the fixed
pipeline lacks. But net-net, on this task and at this dataset size, the
fixed pipeline's narrower, more consistent search strategy still wins by
a small but real margin, at half the latency. This is consistent with the
cross-cutting pattern observed across every earlier iteration of this
research (see project history in `EXPERIMENTS_LOG.md`): giving an LLM more
autonomy over pipeline control flow has not, in any iteration tried so
far, produced a net accuracy win over a well-tuned fixed pipeline - though
this is the first time in this project's cycle that the *reason* has been
isolated to a real, mechanistically-verified trade-off (better recall on
some rows, worse precision on others) rather than measurement noise. The
5 recovered-miss cases suggest a fusion of both retrieval strategies (agent
top-5 ∪ fixed-pipeline top-5, deduplicated, then reranked) as a promising
follow-up not yet implemented - flagged in §8.

## 7. Failure analysis

Of 969 full-dataset predictions with the current best configuration: 797
(82.2%) exact rank-1 matches, 119 (12.3%) **ranking errors** (gold present
in the top-5 but not ranked first - the reranker saw the right candidate
and misjudged its order), and 53 (5.5%) **retrieval misses** (gold never
appears in the top-5 - unrecoverable by any reranking step).

By term type (`scripts_failure_analysis.py`), single-word terms are the
weakest category on both failure types (14.2% ranking-error rate, 9.9%
retrieval-miss rate, vs. 3.2-4.4% retrieval-miss for compounds/phrases).
Manual inspection of retrieval misses surfaces two recurring patterns:
domain-jargon/register mismatches where the gold is a markedly more
general or colloquial word than the input (`philosophy`→"wisdom",
`gospels`→"good news"), and a previously-unaddressed *reverse*
abbreviation direction, where the gold itself is the abbreviation and the
input is the expanded form (`information technology`→"IT", `identity
management`→"IdM") - the opposite of the abbreviation-expansion attempt
already tried and discarded (which only expanded abbreviation *inputs*).
Ranking-error examples at gold-rank 2 are overwhelmingly defensible
near-synonyms (`investment`→"investing" ranked below "funding";
`mediation`→"conflict resolution" ranked below "dispute resolution"),
suggesting part of the remaining ranking-error rate reflects genuine
paraphrase ambiguity in the gold labels rather than a model deficiency.
Full breakdown: `results/eval/failure_analysis_full_summary.csv`,
per-row detail in `results/eval/failure_analysis_full_by_category.csv`.

### 7.1 Extended analysis: rank before/after reranking, causal attribution (2026-07-22)

`scripts_failure_analysis_v2.py` matches each row's gold rank **before**
reranking (`llm_expansion_weighted_t0`'s output) against its rank
**after** (`llm_expansion_weighted_t0_rerank_qwen`), adding lexical
overlap, a morphological-stem proxy, and a corrected reverse-abbreviation
flag. Full dataset, n=969, 916 rows where gold was retrieved at all
pre-rerank:

| rank movement | n | rate |
|---|---|---|
| improved | 136 | 14.8% |
| unchanged | 732 | 79.9% |
| worsened | 48 | 5.2% |

Reranking is net-positive by a wide margin, but **43 of the 48
regressions are already-correct rank-1 predictions the reranker
demoted** - a real, previously-unquantified cost of blanket reranking,
distinct from "reranking doesn't help ambiguous cases" (confidence-gating
already ruled that out, §5-adjacent finding in `README.md`): a reranker
judgment error can happen on a *confident, correct* prediction just as
easily as on an ambiguous one.

**Lexical (token) overlap between term and gold is the strongest single
correlate of failure found in this project**: retrieval-miss rate 8.9%
at zero overlap vs. 1.7% at high overlap; ranking-error rate 16.0% vs.
6.9%. This motivated two targeted follow-up methods (below).

**Reverse-abbreviation, corrected for heuristic false positives**: 8 rows
flagged by a naive short-token heuristic, but manual verification found
only 3 genuine acronyms (`identity management`→"IdM", `information
technology`→"IT", `International Criminal Court`→"ICCt", all 3 retrieval
misses) - the other 5 (`anxiety`→"worry", `gender`→"sex", etc.) are
ordinary short words the heuristic over-flags. This is disclosed rather
than reported as a larger pattern than it is.

**Two methods implemented and tested against this analysis, both
discarded** (val, n=194; full method/verdict detail in
`EXPERIMENTS_LOG.md`, 2026-07-22 "continuación 3"):

| method | target pattern | val hit@1 | net rank-1 | verdict |
|---|---|---|---|---|
| `llm_expansion_rerank_qwen_initialism` | reverse-abbreviation, via extra fusion query | 0.866 (+1.5pt) | +3 | **looks positive, isn't real** - per-row candidate-set check shows every status-changed row had an identical pool to baseline; the +3 is 100% reranker API noise, not the mechanism (which does change 36/194 pools, but never one that flips an outcome) |
| `llm_expansion_lexical_gated_rerank_qwen` | zero-lexical-overlap rows, via gated BM25 injection | 0.830 (-2.1pt) | -4 | trigger fires on 50/194 rows but only 2 ever change final status (1 fix, 1 regression - net zero); aggregate -4 is dominated by noise on untouched rows |

Both negative results were confirmed the same way: comparing the
candidate **set** (not just the final metric) between baseline and
candidate run for every row whose pass/fail status changed. If the set is
identical, the status change is attributable to `qwen3-32b`
non-determinism between API calls, not the code change - regardless of
which direction the aggregate hit@1 moved. This is now the standard
verification step before crediting any small (<3pt) hit@1 delta to a
method change in this project.

A third, more surgical variant (`llm_expansion_rerank_qwen_initialism_exact`
- exact-string-match injection into the pool instead of diluting the
weighted fusion with an extra query) was implemented in response to the
first method's failure mode. Result on val (completed 2026-08-03): hit@1
0.8351 vs baseline 0.8505 (-1.5pt), net rank-1 -3/194 (1 ranking error
corrected, 4 regressions). The same causal check applied to the two
methods above was applied here too: all 5 status-changed rows have a
candidate set identical to the baseline's, including the one case where
the exact-match injection had been separately confirmed offline to fire.
**Discarded**, same verdict as the other two - the aggregate delta is
reranker sampling noise, not a mechanism effect. All three targeted
attempts at the failure patterns identified in the v2 analysis (reverse-
abbreviation via extra query, reverse-abbreviation via exact injection,
zero-lexical-overlap via gated BM25) are now discarded under the same
verification protocol; the underlying failure patterns remain well
characterized (above) but unresolved by any retrieval-side fix tried so
far - carried into §8 as an open limitation rather than an active line of
work.

## 8. Limitations

**Sample size.** n=194 per split is enough to trust large (+8-15pt)
signals but not to reliably rank methods within 2-3pt of each other -
several methods in §4-5 cluster inside that margin. The two candidate
methods evaluated in §7.1 (`llm_expansion_rerank_qwen_initialism[_exact]`,
`llm_expansion_lexical_gated_rerank_qwen`) were both confirmed, via
per-row candidate-set inspection rather than the aggregate delta alone, to
be within-noise null results despite non-trivial-looking aggregate hit@1
swings (+1.5pt and -2.1pt respectively) - the aggregate number alone would
have been misleading in both directions. Any future reported delta under
~3pt should get the same causal check (candidate set identical
before/after → the metric delta is reranker noise, not the code change)
before being cited as a finding.

**Non-determinism, at two separate points in the pipeline.** The local
generator (`llama3.2:3b`, `temperature=0`) is perfectly reproducible
*within* one process's repeated calls but not perfectly so *across*
separate runs. The Qwen reranker (`qwen/qwen3-32b` via OpenRouter, also
`temperature=0`) is additionally not reproducible between separate API
calls - the abbreviation-expansion experiment (2026-07-21) found 8/194
rows regressed and 3/194 improved between two runs of a config that should
have been byte-identical. Budget a few points of hit@1/MRR slop on any
single-run comparison; see `README.md` "Known risks" for the full
inventory (including a stalled-request timeout bug and a since-fixed
silent-fallback failure mode in the OpenRouter client, both caught before
they could corrupt reported numbers).

**Paid external dependency.** The best-performing configuration depends
on OpenRouter (`qwen/qwen3-32b`) for its reranking stage - unlike every
other component in this project, it costs real money (~$0.03/194-row
run, ~$0.18 for the full 969-row report) and carries external-service
reliability risk. A 9B-parameter alternative was abandoned specifically on
reliability grounds (§5) before its accuracy could even be measured,
independent of the raw hit@1 numbers - a reminder that model *quality* and
model *availability* are separate risks that both need checking before an
API-dependent component is adopted.

**Retrieval misses are the harder-to-close error type.** The 5.5%
retrieval-miss category (gold synonym absent from the candidate pool
handed to the reranker) resisted every query-side fix attempted so far:
pool widening, LLM direct-guess candidates, abbreviation expansion, and
(§7.1) both the acronym-injection and lexical-gated-BM25 follow-ups. By
contrast, the 12.3% ranking-error category responded strongly to a single
change (reranker model size, §5). This asymmetry suggests the two error
types have different root causes - ranking errors are largely a model-
capability problem the reranking stage can fix, while retrieval misses
are closer to a candidate-vocabulary coverage problem no reranker can
recover from by construction.

**No formal prompt-sensitivity analysis.** The changes made so far varied
models, temperature, variant count, and fusion weights, but never
systematically varied the reranker/generator prompt wording itself while
holding everything else fixed - flagged by the advisor early on
(2026-07-19) as a desired analysis, not yet done.

**Environment-specific gaps, disclosed for reproducibility.** `gte-base-
en-v1.5` could not be loaded in this environment (a remote-code rotary-
embedding implementation crashes with an `IndexError`, both on GPU and
CPU) - an environment/library compatibility issue, not evidence about the
model's quality, but it means the embedder comparison in this report is
missing one otherwise-relevant model.

**Unused `train` split.** The 581-row `train` split is reserved but
currently unused by any method (§3.1) - a natural next step, not pursued
here for lack of remaining time before the experimental cutoff, would be
using it to calibrate the fusion weights or the confidence-gating
threshold instead of hand-picking them against `val`.

**Unexplored retrieval fusion between the two architectures.** §6 found
that the dynamic agent recovers 5 gold synonyms the fixed pipeline's
retrieval misses entirely, via self-chosen paraphrases the fixed
expansion strategy never generates - a real, complementary capability,
not noise. Whether pooling both architectures' candidate sets before a
single shared reranking pass would capture that recall gain without the
dynamic agent's regressions or ~2x latency cost is untested - flagged as
the most promising unexplored direction from this cycle, not attempted
here for lack of remaining experimental time.
