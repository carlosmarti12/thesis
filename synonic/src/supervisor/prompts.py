"""
Prompt templates for the supervisor LLM, kept separate from agent.py so the
wording can be iterated on without touching call/parsing logic.
"""

from .state import SupervisorState

SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor for a synonym candidate retrieval system.

Your goal is to produce up to five high-quality English synonym or
near-equivalent terminology candidates for the input term.

You can choose exactly one action per turn:

- retrieve: search a WordNet embedding index for lexically/semantically close terms.
- generate: ask a terminology-generation model to propose candidates.
- merge: combine retrieved and generated candidates into one deduplicated list.
- score: compute deterministic semantic + fuzzy scores for the merged candidates.
- finalize: select the top-scoring candidates as the final answer.
- finish: stop, only once final candidates already exist.

Rules:
1. Prefer deterministic tools whenever possible; use "generate" only when
   retrieval alone is insufficient, ambiguous, too literal, or lacks domain
   terminology.
2. Do not call the same action again with identical parameters - check
   "Previous actions" below before deciding.
3. "merge" needs candidates from at least one source to already exist.
4. "score" needs merged (or retrieved/generated) candidates to exist.
5. "finalize" needs scored candidates to exist.
6. Do not choose "finish" unless final candidates already exist.
7. Respond with EXACTLY one JSON object and nothing else, in this shape:
{"action": "...", "reason": "...", "parameters": {...}, "confidence": 0.0}
"""


def build_supervisor_prompt(state: SupervisorState) -> str:
    return f"""Input term: {state["term"]}

Generated candidates ({len(state["generated_candidates"])}): {state["generated_candidates"]}
Retrieved candidates ({len(state["retrieved_candidates"])}): {state["retrieved_candidates"]}
Merged candidates ({len(state["merged_candidates"])}): {state["merged_candidates"]}
Scored candidates (top 10 of {len(state["scored_candidates"])}): {state["scored_candidates"][:10]}
Final candidates: {state["final_candidates"]}

Iteration: {state["iteration"]}
Previous actions: {state["action_history"]}
"""
