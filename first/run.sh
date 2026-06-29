source .venv/bin/activate

python -m src.run_comparison --limit 10 --methods mas
python -m src.run_comparison --limit 10 --methods same llm
python -m src.run_comparison --limit 10