# Results summary (`results/` folder)

Resumen de todos los resultados numéricos guardados en `results/` -
generado leyendo directamente `results/experiment_log.csv` (el log
maestro, una fila por cada corrida de cada método) y los archivos de
`results/eval/`. Este documento no reinterpreta ni redondea decisiones -
para el análisis narrativo (por qué se descartó cada método, qué
significan los hallazgos) ver `EXPERIMENTS_LOG.md` y `METHODOLOGY.md`;
aquí solo están los números, organizados.

## Qué hay en `results/`

- **`results/experiment_log.csv`** - log maestro, 73 corridas registradas
  desde 2026-07-19, 43 métodos distintos, columnas: método, split, filas,
  hit@1/3/5, mrr, ndcg@3/5, latencia media, nota de config,
  `reranker_fallback_count`.
- **`results/eval/*_summary.json`** - métricas agregadas por corrida
  individual (una por método+split).
- **`results/eval/*.csv`** (detail) - predicción fila por fila de cada
  corrida, con el candidato devuelto, el rank del gold, y
  `reranker_fallback` por fila.
- **`results/eval/failure_*`, `error_*`** - desgloses de errores por
  categoría (ver §4).
- **`results/pipeline_results.csv`, `synonym_finding_comparison.csv`** -
  corridas exploratorias muy tempranas (2026-07-19, antes del harness
  unificado `run_eval.py`); supersedidas por `experiment_log.csv`, se
  mantienen solo como rastro histórico.
- Archivos con `__limitN` o "smoke test" en `config_note` -  corridas de
  prueba (5-20 filas) para verificar que un método nuevo no rompe antes de
  lanzar la corrida real de 194 filas. **Excluidos de las tablas de abajo**
  porque su n es demasiado pequeño para ser comparable.

## 1. Resultado final sobre el dataset completo (n=969, `full` split)

Config ya congelada de antes (elegida en `val`, nunca ajustada con estos
datos) - `data/splits/full.csv` = train+val+test combinados. Único
propósito: reporte descriptivo.

| método | n | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | latencia/término |
|---|---|---|---|---|---|---|---|---|
| `embedding_only` (baseline original) | 969 | 0.545 | 0.742 | 0.802 | 0.646 | - | - | 0.006s |
| `embedding_e5` | 969 | 0.645 | 0.835 | 0.892 | 0.743 | - | - | 0.010s |
| `llm_expansion_weighted` (sin rerank) | 969 | 0.755 | 0.911 | 0.946 | 0.833 | - | - | 0.716s |
| `llm_expansion_weighted_t0` | 969 | 0.748 | 0.915 | 0.945 | 0.829 | - | - | 0.744s |
| **`llm_expansion_weighted_t0_rerank_qwen`** (mejor config) | **969** | **0.822** | **0.932** | **0.945** | **0.877** | **0.889** | **0.895** | **11.84s** |

`reranker_fallback_count: 0` en la corrida ganadora - ninguna fila
corrompida por fallo de la API. Coste real medido: ~$0.18 (OpenRouter,
delta de créditos antes/después).

No hay corrida del agente dinámico (`agent_tool_calling_e5_rerank_qwen`)
sobre `full` - esa arquitectura solo se evaluó en `val` (§3).

## 2. Confirmación en `test` (n=194, tocado una única vez por config)

| método | hit@1 | hit@3 | hit@5 | mrr | latencia/término |
|---|---|---|---|---|---|
| **`llm_expansion_weighted_t0_rerank_qwen`** (2026-07-22, con verificación de fallback) | **0.851** | 0.923 | 0.948 | 0.892 | 9.27s |
| `llm_expansion_weighted_t0_rerank_qwen` (2026-07-20, corrida anterior, mismo config) | 0.830 | 0.938 | 0.948 | 0.884 | 13.32s |
| `embedding_e5` (una corrida con n distinto por un bug de muestreo puntual) | 0.800 | 1.000 | 1.000 | 0.900 | 0.71s |
| `llm_expansion_weighted_n3` | 0.768 | 0.902 | 0.933 | 0.835 | 0.72s |
| `llm_expansion_weighted` | 0.753 | 0.881 | 0.938 | 0.823 | 0.74s |
| `llm_expansion_weighted_t0` | 0.747 | 0.907 | 0.948 | 0.827 | 0.72s |
| `embedding_e5` | 0.603 | 0.789 | 0.871 | 0.704 | 0.023s |
| `embedding_only` | 0.536 | 0.722 | 0.773 | 0.629 | 0.022s |

Las dos corridas de `llm_expansion_weighted_t0_rerank_qwen` sobre el mismo
config difieren ~2pt hit@1 - evidencia medida de la no-reproducibilidad
del reranker Qwen entre llamadas de API separadas (ver `METHODOLOGY.md`
§8).

## 3. Leaderboard completo en `val` (n=194, todas las corridas reales)

Split donde se hicieron todas las comparaciones y decisiones de diseño
(42 métodos distintos, 47 corridas con n=194 real). Ordenado por hit@1
descendente:

| método | hit@1 | hit@3 | hit@5 | mrr | latencia/término | nota |
|---|---|---|---|---|---|---|
| `llm_expansion_rerank_qwen_initialism` | 0.866 | 0.923 | 0.923 | 0.893 | 13.30s | +1.5pt vs baseline, luego confirmado como ruido de API (ver §5) |
| `llm_expansion_weighted_t0_rerank_qwen` | 0.851 | 0.912 | 0.923 | 0.882 | 9.08s | **config ganadora, usada en test/full** |
| `llm_expansion_gated_rerank_qwen` | 0.835 | 0.907 | 0.923 | 0.873 | 3.78s | rerank condicionado por confianza, pierde contra rerank en bloque |
| `llm_expansion_rerank_qwen_initialism_exact` | 0.835 | 0.912 | 0.923 | 0.873 | 12.70s | descartado, ruido de API (§5) |
| `agent_tool_calling_e5_rerank_qwen` | 0.830 | 0.907 | 0.907 | 0.866 | 18.96s | agente dinámico, mismo reranker que el pipeline fijo - pierde por ~2pt y 2x más lento, pero recupera 5 retrieval-misses que el fijo no encuentra (ver `METHODOLOGY.md` §6) |
| `llm_expansion_lexical_gated_rerank_qwen` | 0.830 | 0.912 | 0.923 | 0.872 | 12.48s | descartado, ruido de API (§5) |
| `llm_expansion_rerank_qwen_abbrev` | 0.825 | 0.918 | 0.923 | 0.870 | 12.28s | expansión de abreviaturas como query extra - efecto nulo confirmado |
| `llm_expansion_rerank_qwen_pool10` | 0.809 | 0.918 | 0.933 | 0.862 | 5.34s | pool de 10 antes de rerank - más distractores empeora rank-1 |
| `llm_expansion_rerank_qwen_pool15` | 0.799 | 0.907 | 0.928 | 0.853 | 5.54s | pool de 15 - empeora monótonamente con el tamaño |
| `llm_expansion_rerank_qwen_llmguess` | 0.794 | 0.902 | 0.923 | 0.849 | 5.47s | candidatos de conjetura directa del LLM fusionados - sin recall nuevo |
| `llm_expansion_weighted_n3` | 0.789 | 0.902 | 0.928 | 0.846 | 0.71s | n=3 variantes (venció a n=9 en el barrido) |
| `llm_expansion_weighted_t0` | 0.784 | 0.902 | 0.923 | 0.842 | 0.74s | mejor config pre-rerank |
| `llm_expansion_gated_pool_rerank_qwen` | 0.784 | 0.902 | 0.923 | 0.842 | 1.07s | pool ampliado solo si la fusión es ambigua |
| `llm_expansion_rerank` | 0.773 | 0.907 | 0.923 | 0.842 | 13.15s | expansión + fusión + rerank, temp=0.3 |
| `llm_expansion_weighted` | 0.773 | 0.892-0.907 | 0.902-0.938 | 0.829-0.845 | 0.73s | temp=0.3, peso=3 (3 corridas de estabilidad) |
| `llm_expansion_weighted_t0_rerank_local` | 0.768 | 0.907 | 0.923 | 0.837 | 13.21s | reranker local qwen3.5:2b - **empeora** el rank-1 (§5 de `METHODOLOGY.md`) |
| `llm_expansion_mean` | 0.753 | 0.897 | 0.912 | 0.822 | 0.74s | fusión por media en vez de ponderada |
| `llm_expansion_weighted_mxbai` | 0.753 | 0.897 | 0.938 | 0.827 | 0.71s | embedder alternativo mxbai-embed-large-v1 |
| `embedding_e5` | 0.691 | 0.840 | 0.907 | 0.773 | 0.024s | e5 solo, sin LLM - baseline fuerte y barato |
| `e5_llm_rerank` | 0.691 | 0.840 | 0.907 | 0.773 | 4.36s | e5 solo + rerank LLM, sin expansión - rerank no ayuda sin expansión previa |
| `ensemble_rrf_e5_bge` | 0.670 | 0.851 | 0.918 | 0.767 | 0.048s | RRF de dos embedders fuertes |
| `hybrid_pipeline` | 0.670 | 0.861 | 0.912 | 0.766 | 5.31s | pipeline original (MiniLM) |
| `agent_tool_calling_e5` | 0.660 | 0.835 | 0.907 | 0.756 | 1.45s | agente dinámico sin reranker fuerte |
| `llm_zero_shot` | 0.660 | 0.814 | 0.840 | 0.738 | 0.65s | generación abierta anclada a MiniLM |
| `embedding_e5_morph_variants` | 0.655 | 0.840 | 0.897 | 0.748 | 0.024s | |
| `llm_expansion_max` | 0.655 | 0.830 | 0.881 | 0.743 | 0.76s | fusión por máximo |
| `llm_zero_shot_e5` | 0.644 | 0.820 | 0.845 | 0.728 | 0.66s | generación abierta anclada a e5 |
| `llm_expansion_rrf` | 0.629 | 0.809 | 0.887 | 0.728 | 0.72s | fusión RRF en vez de ponderada |
| `embedding_bge` | 0.624 | 0.799 | 0.881 | 0.720 | 0.024s | |
| `hybrid_pipeline_e5` | 0.613 | 0.851 | 0.887 | 0.732 | 5.24s | pipeline original anclado a e5, reranker local débil |
| `hybrid_rrf_bm25_e5_weighted10` | 0.582 | 0.742 | 0.892 | 0.687 | 0.022s | |
| `agent_tool_calling` | 0.572 | 0.706 | 0.804 | 0.656 | 1.37s | agente dinámico, config original (MiniLM) |
| `embedding_only` | 0.546 | 0.706 | 0.804 | 0.642 | 0.019s | baseline original |
| `hybrid_rrf_bm25_e5_weighted` | 0.541 | 0.742 | 0.845 | 0.654 | 0.025s | |
| `embedding_morph_variants` | 0.536 | 0.691 | 0.789 | 0.630 | 0.021s | |
| `embedding_e5_large` | 0.485 | 0.603 | 0.629 | 0.542 | 1.23s | variante más grande de e5, peor - no monótono con tamaño |
| `hybrid_rrf_bm25_e5` | 0.469 | 0.711 | 0.820 | 0.600 | 0.023s | |
| `cross_encoder_rerank` | 0.469 | 0.675 | 0.794 | 0.585 | 0.041s | |
| `cross_encoder_rerank_e5` | 0.469 | 0.665 | 0.778 | 0.586 | 0.044s | |
| `hybrid_rrf_bm25_embedding` | 0.407 | 0.644 | 0.758 | 0.537 | 0.019s | |
| `cross_encoder_quora_rerank_e5` | 0.397 | 0.639 | 0.768 | 0.531 | 0.083s | |
| `bm25_only` | 0.222 | 0.320 | 0.376 | 0.277 | 0.0002s | léxico puro - el más débil con diferencia |

**Advertencia de lectura obligatoria**: cualquier diferencia de hit@1 bajo
~3pt entre dos filas de esta tabla **no es fiable sin más verificación**.
El reranker Qwen (`temperature=0`) no es perfectamente reproducible entre
llamadas de API separadas - varias filas de esta tabla que en su momento
parecieron ganancias reales (`llm_expansion_rerank_qwen_initialism`,
`llm_expansion_rerank_qwen_initialism_exact`,
`llm_expansion_lexical_gated_rerank_qwen`) fueron descartadas tras
verificar que el candidate-set no cambiaba en las filas cuyo resultado
cambió - el delta era ruido de muestreo del LLM, no el mecanismo probado.
Detalle de esa verificación: `EXPERIMENTS_LOG.md`, entradas
2026-07-22 "continuación 3" y 2026-08-03.

## 4. Análisis de errores y fallos (dataset completo, config ganadora)

De 969 predicciones con `llm_expansion_weighted_t0_rerank_qwen`:

| categoría | n | % |
|---|---|---|
| correcto en rank-1 | 797 | 82.2% |
| ranking error (el bueno está en top-5, no en rank-1) | 119 | 12.3% |
| retrieval miss (el bueno nunca aparece en el top-5) | 53 | 5.5% |

Desglose de los 119 ranking errors por posición: 87 en rank-2, 19 en
rank-3, 11 en rank-4, 2 en rank-5.

Archivos fuente: `results/eval/error_analysis_rerank_qwen_full.csv`
(detalle fila por fila), `results/eval/failure_analysis_full_summary.csv`
+ `_by_category.csv` (desglose por tipo de término), y el análisis
extendido con atribución causal (rank antes/después del rerank):
`results/eval/failure_v2_full_by_category.csv` +
`failure_v2_full_detail.csv`.

Rank movement (n=916 filas con gold recuperado antes del rerank, dataset
completo): 136 mejoraron (14.8%), 732 sin cambio (79.9%), 48 empeoraron
(5.2%) - de esas 48, 43 eran predicciones ya correctas en rank-1 que el
reranker degradó, el coste real y antes no cuantificado del reranking en
bloque.

## 5. Métodos descartados con verificación causal de ruido

Tres intentos dirigidos a los patrones de fallo identificados en §4,
todos con delta agregado que parecía positivo o negativo pero resultó ser
ruido de API al inspeccionar el candidate-set fila por fila:

| método | delta hit@1 agregado | delta rank-1 neto (verificado) | veredicto |
|---|---|---|---|
| `llm_expansion_rerank_qwen_initialism` | +1.5pt | +3/194 | ruido - candidate-set idéntico en todas las filas cuyo resultado cambió |
| `llm_expansion_lexical_gated_rerank_qwen` | -2.1pt | -4/194 | ruido - el gating solo afecta 2/194 filas realmente |
| `llm_expansion_rerank_qwen_initialism_exact` | -1.5pt | -3/194 | ruido - mismo patrón |

Comparación arquitectónica que **sí resultó real** (no ruido, verificado
con el mismo protocolo): `agent_tool_calling_e5_rerank_qwen` vs. el
pipeline fijo - 0/10 filas regresadas tenían el mismo candidate-set que
el pipeline fijo. Detalle completo: `METHODOLOGY.md` §6.

## Dónde profundizar

- Narrativa completa de por qué se descartó cada método:
  `EXPERIMENTS_LOG.md` (cronológico) y `README.md` ("What was tried and
  discarded").
- Escritura para la tesis con estos mismos números ya integrados en
  prosa: `METHODOLOGY.md`.
- Reproducir cualquier fila de este documento: `run_eval.py --method
  <nombre> --split <val|test|full>` (ver "Reproduce" en `README.md`).
