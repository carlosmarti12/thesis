"""
Harness unificado de evaluación - reemplaza la lógica duplicada de
run_experiment.py / run_synonym_finding.py con un único punto de entrada
parametrizado por --method y --split, para que todo experimento futuro sea
directamente comparable (mismos splits, mismas métricas, mismo log).

Uso:
    python run_eval.py --method embedding_only --split val
    python run_eval.py --method hybrid_pipeline --split val --limit 60
    python run_eval.py --method cross_encoder_rerank --split test
    python run_eval.py --method llm_expansion_weighted_t0_rerank_qwen --split full --resume

Cada corrida:
  1. Construye el pool de candidatos (en_synonym del dataset COMPLETO -
     no es leakage, ver memoria del proyecto: el ground truth de la fila
     evaluada nunca se consulta antes de generar sus candidatos).
  2. Corre el método sobre el split pedido (train/val/test, ver
     data/make_splits.py - usar val para tunear, test solo para
     confirmación final).
  3. Guarda el detalle por fila incrementalmente (checkpoint tras cada fila,
     sobrevive una interrupción) y el resumen agregado bajo results/eval/.
  4. Añade una fila a results/experiment_log.csv (log maestro, nunca se
     borra - un experimento por fila).

Cada fila registra `reranker_fallback` (True si el método usa un reranker
LLM vía API y esa llamada falló, cayendo al orden original - ver
src/agents/reranker_agent.py). `--resume` reutiliza solo filas con
reranker_fallback=False de una corrida anterior sobre el mismo archivo de
salida; el resto se recalcula. Al final, si queda algún fallback tras
`--max-fallback-retries` reintentos, la corrida falla con error en vez de
reportar métricas que mezclen filas válidas con corruptas.
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.agents import reranker_agent
from src.evaluation import compute_metrics, summarize_results
from src.methods import ALL_METHODS, NEEDS_LLM, MethodContext, get_candidates

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synonyms_clean.csv"
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "eval"
EXPERIMENT_LOG = ROOT / "results" / "experiment_log.csv"


def _compute_row(method: str, term: str, ground_truth: str, ctx: MethodContext, top_k: int) -> dict:
    """Corre un método sobre una fila y devuelve el dict de resultado,
    incluyendo si el reranker (si el método usa uno) cayó al fallback -
    medido por el DELTA de reranker_agent.fallback_count antes/después de
    esta única llamada (las filas se procesan secuencialmente, nunca en
    paralelo, así que el delta se puede atribuir con seguridad a esta
    fila)."""
    fallback_before = reranker_agent.fallback_count
    start = time.time()
    try:
        candidates = get_candidates(method, term, ctx, top_k=top_k)
    except Exception as e:
        tqdm.write(f"ERROR [{method}] con '{term}': {e}")
        candidates = []
    elapsed = time.time() - start
    used_fallback = reranker_agent.fallback_count > fallback_before

    metrics = compute_metrics(ground_truth, candidates)

    return {
        "term": term,
        "ground_truth": ground_truth,
        "prediction": candidates[0] if candidates else "",
        "candidates": json.dumps(candidates, ensure_ascii=False),
        "num_candidates": len(candidates),
        "time_seconds": elapsed,
        "reranker_fallback": used_fallback,
        **metrics,
    }


def run(
    method: str, split: str, limit: int = 0, top_k: int = 5, config_note: str = "",
    resume: bool = False, max_fallback_retries: int = 3,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(DATA_PATH)
    split_path = SPLITS_DIR / f"{split}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}. Run `python data/make_splits.py` first.")
    df = pd.read_csv(split_path)
    if limit > 0:
        df = df.head(limit)

    # El pool sale siempre del dataset COMPLETO (todas las filas, todos los
    # splits) - es un vocabulario fijo, no la respuesta de ninguna fila en
    # particular. Ver [[feedback-leakage-definition]] en la memoria del
    # proyecto: esto no es leakage.
    pool_terms = full_df["en_synonym"].dropna().astype(str).tolist()
    ctx = MethodContext(pool_terms_raw=pool_terms)

    # Contador de fallos del reranker Qwen para esta corrida (ver
    # src/agents/reranker_agent.py) - un fallo sistemático (p.ej. créditos
    # de OpenRouter agotados) hace que TODAS las llamadas caigan al orden
    # original sin ningún error visible en las métricas - resetear aquí y
    # reportarlo en el summary para que sea imposible pasarlo por alto.
    reranker_agent.reset_fallback_count()

    # El sufijo de --limit evita que una corrida de humo pise el archivo de
    # una corrida completa con el mismo método+split (pasó una vez en esta
    # misma carpeta - ver EXPERIMENTS_LOG.md).
    suffix = f"__limit{limit}" if limit > 0 else ""
    detail_path = RESULTS_DIR / f"{method}__{split}{suffix}.csv"

    # --resume: reutiliza filas de una corrida anterior INTERRUMPIDA sobre
    # el mismo archivo, pero SOLO las que ya tienen una respuesta real del
    # reranker (reranker_fallback == False) - las que fallaron, cayeron a
    # fallback, o vienen de una corrida vieja sin esta columna (formato
    # anterior al 2026-07-20) se recalculan. Clave (term, ground_truth), no
    # solo term - hay ~10 términos duplicados con distinto en_synonym (ver
    # data/make_splits.py), mergear solo por term arriesga cruzar filas.
    reusable: dict[tuple[str, str], dict] = {}
    if resume and detail_path.exists():
        prev_df = pd.read_csv(detail_path)
        if "reranker_fallback" in prev_df.columns:
            valid_prev = prev_df[prev_df["reranker_fallback"] == False]  # noqa: E712
            for _, r in valid_prev.iterrows():
                reusable[(str(r["term"]), str(r["ground_truth"]))] = r.to_dict()
            print(f"--resume: {detail_path.name} exists with {len(prev_df)} rows, "
                  f"{len(reusable)} have a confirmed real reranker response and will be reused; "
                  f"{len(prev_df) - len(reusable)} will be recomputed.")
        else:
            print(f"--resume: {detail_path.name} exists but predates fallback-tracking "
                  f"(no 'reranker_fallback' column) - recomputing everything, not reusing.")

    rows = []
    n_reused = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{method}/{split}"):
        term = str(row["en"])
        # ground_truth se lee AQUI, después de que el método ya produjo sus
        # candidatos - nunca antes, y nunca se pasa a get_candidates.
        ground_truth = str(row["en_synonym"])
        key = (term, ground_truth)

        if key in reusable:
            rows.append(reusable[key])
            n_reused += 1
        else:
            rows.append(_compute_row(method, term, ground_truth, ctx, top_k))

        # Checkpoint incremental - si esta corrida se interrumpe, lo ya
        # computado no se pierde y una corrida posterior con --resume puede
        # retomar desde aquí.
        pd.DataFrame(rows).to_csv(detail_path, index=False)

    if n_reused:
        print(f"--resume: reused {n_reused}/{len(df)} rows, recomputed {len(df) - n_reused}.")

    # Reintentos acotados para cualquier fila que siga en fallback tras el
    # primer paso - el usuario pidió explícitamente que la corrida final NO
    # mezcle filas válidas con fallbacks, así que no se reporta como
    # terminada mientras queden fallbacks recuperables.
    for attempt in range(1, max_fallback_retries + 1):
        failed_idx = [i for i, r in enumerate(rows) if r.get("reranker_fallback")]
        if not failed_idx:
            break
        print(f"Retry {attempt}/{max_fallback_retries}: recomputing {len(failed_idx)} "
              f"row(s) still in fallback...")
        for i in failed_idx:
            term, ground_truth = rows[i]["term"], rows[i]["ground_truth"]
            rows[i] = _compute_row(method, term, ground_truth, ctx, top_k)
        pd.DataFrame(rows).to_csv(detail_path, index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(detail_path, index=False)

    n_fallback_final = int(results_df["reranker_fallback"].sum()) if "reranker_fallback" in results_df else 0
    if n_fallback_final > 0:
        failing_terms = results_df.loc[results_df["reranker_fallback"], "term"].tolist()
        raise RuntimeError(
            f"{n_fallback_final}/{len(results_df)} rows still have reranker_fallback=True "
            f"after {max_fallback_retries} retries - refusing to report a result that mixes "
            f"valid and corrupted rows. Failing terms: {failing_terms[:20]}"
            f"{'...' if len(failing_terms) > 20 else ''}. Fix the underlying issue (e.g. check "
            f"OpenRouter credits with GET /api/v1/credits) and re-run with --resume."
        )

    summary = summarize_results(results_df, method=method)
    summary["split"] = split
    summary["reranker_fallback_count"] = n_fallback_final
    summary_path = RESULTS_DIR / f"{method}__{split}{suffix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _append_to_experiment_log(method, split, summary, config_note)

    print(f"\n{method} / {split} (n={summary['rows']}):")
    for key in ("hit@1", "hit@3", "hit@5", "mrr", "ndcg@3", "ndcg@5", "avg_time_seconds"):
        print(f"  {key}: {summary[key]:.4f}")
    print(f"  reranker_fallback_count: {n_fallback_final} (0 = every row has a confirmed real reranker response)")
    print(f"  n_hit@1={summary['n_hit@1']}  n_hit@3={summary['n_hit@3']}  n_hit@5={summary['n_hit@5']}")
    print(f"Detail:  {detail_path}")
    print(f"Summary: {summary_path}")

    return results_df, summary


def _append_to_experiment_log(method: str, split: str, summary: dict, config_note: str) -> None:
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "split": split,
        "rows": summary["rows"],
        "hit@1": summary["hit@1"],
        "hit@3": summary["hit@3"],
        "hit@5": summary["hit@5"],
        "n_hit@1": summary["n_hit@1"],
        "n_hit@3": summary["n_hit@3"],
        "n_hit@5": summary["n_hit@5"],
        "mrr": summary["mrr"],
        "ndcg@3": summary["ndcg@3"],
        "ndcg@5": summary["ndcg@5"],
        "avg_time_seconds": summary["avg_time_seconds"],
        "reranker_fallback_count": summary.get("reranker_fallback_count", 0),
        "config_note": config_note,
    }
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    if EXPERIMENT_LOG.exists():
        log_df = pd.read_csv(EXPERIMENT_LOG)
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([row])
    log_df.to_csv(EXPERIMENT_LOG, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "full"])
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = full split)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--note", type=str, default="", help="Free-text config note for the experiment log")
    parser.add_argument("--resume", action="store_true",
                         help="Reuse rows from an existing detail CSV that already have a confirmed "
                              "real reranker response (reranker_fallback == False); recompute the rest.")
    parser.add_argument("--max-fallback-retries", type=int, default=3,
                         help="How many extra passes to retry rows still stuck in reranker fallback "
                              "before giving up and raising an error.")
    args = parser.parse_args()

    if args.method in NEEDS_LLM and args.limit == 0:
        print(f"Note: '{args.method}' calls a local LLM once (or more) per term - "
              f"a full split can take a while. Use --limit to iterate faster.")

    run(
        args.method, args.split, limit=args.limit, top_k=args.top_k, config_note=args.note,
        resume=args.resume, max_fallback_retries=args.max_fallback_retries,
    )


if __name__ == "__main__":
    main()
