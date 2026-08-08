#!/usr/bin/env bash
set -e
source .venv/bin/activate

# Quick smoke test (10 terms, all methods)
# python -m src.run_comparison --limit 10 --no-resume

# Resume/continue a previous run
# python -m src.run_comparison --limit 10

# Full dataset (~970 terms), all methods (saves progress after every term)
# python -m src.run_comparison --limit 0

# Full dataset, MAS only
# python -m src.run_comparison --limit 0 --methods mas_base mas_llm_ranker mas_safe_hybrid

# Full dataset, baselines only (fast, no LLM)
# python -m src.run_comparison --limit 0 --methods wordnet_direct embedding_wordnet

# Run a single method (separate output file)
# python -m src.run_experiment --method mas_safe_hybrid --limit 0

# Default: quick comparison, resume if exists
python -m src.run_comparison --limit 10


 python -m src.run_comparison --model llama3.2:3b

   python -m src.run_experiment --method mas_safe_hybrid --model llama3.2:3b --limit 5 --no-resume



  - generator_agent: le pregunta a un LLM "dame 8 sinónimos de este término" y parsea la respuesta como lista. Es puro conocimiento del LLM, no mira ninguna base de datos.
  - retrieval_agent: no usa LLM. Convierte el término en un vector (embedding) y busca sus vecinos más cercanos dentro de un índice precalculado de ~147k palabras de WordNet. Es búsqueda por similitud, no generación.
  - merge_agent: junta las dos listas anteriores, quita duplicados y el propio término si aparece.
  - verification_agent / ranking_agent: le pone una nota a cada candidato (0.75×similitud semántica + 0.25×similitud de escritura) y se queda con los 5 mejores.
  - llm_ranking_agent: le da la lista fusionada a un LLM y le dice "ordénamela de mejor a peor, no inventes nada nuevo".
  - safe_finalizer_agent: recalcula todo con embeddings (0.9×similitud) y solo usa el orden que dio el LLM como un empujoncito extra (0.1×bono de posición).



● En el código hay dos métodos "híbridos" distintos — te explico ambos porque el nombre se usa en los dos:

  1. hybrid_fusion_baseline (src/baselines.py:261) — el baseline "hybrid_fusion"

  Combina generación por LLM + recuperación léxica (WordNet), fusionando ambas señales con Reciprocal Rank Fusion (RRF) ponderado, en vez de confiar solo en una de las dos:

  1. Generación LLM con self-consistency: le pide al LLM n_generate candidatos, repitiendo la llamada n_samples veces con temperature=0.7 (cada llamada en su propio try/except para que un fallo no tire todo el pool).
  2. Lista A: agrupa y deduplica los candidatos de las n_samples llamadas, y los reordena por similitud coseno con el término original (embeddings all-MiniLM-L6-v2).
  3. Fallback: si el LLM falla en las 3 llamadas y el pool queda vacío, cae directamente a embedding_wordnet_baseline en vez de devolver una lista vacía.
  4. Lista B: recupera candidatos del índice WordNet vía embedding_wordnet_baseline (señal de recuperación pura, siempre disponible porque no depende del LLM).
  5. Fusión RRF ponderada: combina A y B con pesos distintos — weight_llm=1.0 vs weight_wordnet=0.5. La fórmula RRF es:


python -m src.run_comparison --model llama3.2:3b --output results/comparison_results_full.csv
