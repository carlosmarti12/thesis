#!/usr/bin/env bash
set -e
source .venv/bin/activate

# Quick smoke test (10 terms, all methods)
# python -m src.run_comparison --limit 10 --no-resume

# Resume/continue a previous run
# python -m src.run_comparison --limit 10

# Full dataset, all methods (saves progress after every term)
# python -m src.run_comparison --limit 0

# Full dataset, MAS only
# python -m src.run_comparison --limit 0 --methods mas

# Full dataset, baselines only (fast)
# python -m src.run_comparison --limit 0 --methods same llm

# Run experiment (MAS only, separate output)
# python -m src.run_experiment --limit 0

# Default: quick comparison, resume if exists
python -m src.run_comparison --limit 10
