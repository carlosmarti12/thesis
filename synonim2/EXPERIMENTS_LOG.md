# Experiments log

Registro de cada corrida con **modelo(s) + prompt(s) + métricas**, para poder
hacer más adelante un análisis de sensibilidad a modelo/prompt ("prompt
sensitivity analysis" / "NLP sensitivity analysis" - feedback del asesor,
2026-07-19). Añadir una entrada nueva cada vez que se corra
`run_experiment.py` / `run_synonym_finding.py` con un modelo distinto, o
tras editar un prompt en `rsc/agents/*.py`.

Nomenclatura de métricas: `hit@1/3/5`, `ndcg@3/5`, `mrr` (`rsc/evaluation.py`)
- reemplazan a `exact_top1`/`top_k_accuracy` de iteraciones anteriores
  (`new/`, `synonic/` antes de su propio rename del mismo día).

## Cleanup pedido por el asesor (2026-07-19) - estado en esta carpeta

| Punto del asesor | Estado en `18_7/` |
|---|---|
| Eliminar `same_term` (siempre da 0) | No existe aquí - nunca se implementó en esta carpeta. Nada que borrar. |
| Evaluar la columna `topic` | `data/prepare_dataset.py` ya solo se queda con `en`/`en_synonym` (línea `df[["en", "en_synonym"]]`) - `topic` nunca llega a ningún método. Nada que probar ni quitar. |
| Registrar prompts/métodos/métricas | Este archivo. Antes de hoy no existía ningún registro - solo el JSON de la última corrida. |
| `exact_top1` -> `hit@1`, + `hit@3`, `hit@5`, `ndcg@3`, `ndcg@5` | Ya implementado desde el inicio de esta carpeta (`rsc/evaluation.py`) - de hecho `synonic/` copió esta convención de aquí el mismo día. |
| Mantener la tarea actual tal cual, para comparar con la nueva | Sin cambios de comportamiento en `run_experiment.py` / `rsc/agents/generator_agent.py` / `rsc/agents/reranker_agent_local.py` - solo se refresca abajo el número con una corrida real (el summary guardado en disco era de una prueba de 5 filas con el reranker de OpenRouter, que ya no es el que usa por defecto `run_experiment.py`).

## Entradas

### 2026-07-19 - tarea actual (open discovery), 100 filas, `llama3.2:3b` + `qwen3.5:2b`
- Fuente: `results/pipeline_results.csv` / `results/pipeline_results_summary.json`
- Generador: `llama3.2:3b` (Ollama local), temperatura 0.3, prompt en
  `rsc/agents/generator_agent.py::AGENTE_GENERADOR`.
- Reranker: `qwen3.5:2b` (Ollama local), temperatura 0.0, prompt en
  `rsc/agents/reranker_agent_local.py::AGENTE_RERANKER_LOCAL`.
- El JSON anterior (`gen_llama3.2_3b__rerank_qwen_qwen3.5-9b`, 5 filas,
  hit@k=1.0 perfecto) usaba el reranker de OpenRouter (`qwen/qwen3.5-9b`) y
  ya no era representativo del pipeline actual - reemplazado por esta corrida.
- **Resultado (n=100, method `gen_llama3.2_3b__rerank_qwen3.5_2b`)**:
  hit@1 0.62, hit@3 0.81, hit@5 0.88, ndcg@3 0.728, ndcg@5 0.757,
  avg_time_seconds 20.76.
- Bug encontrado y corregido durante esta corrida: `ollama_client.py` no
  ponía tope a `num_predict`, así que una generación degenerada ocasional
  podía decodificar miles de tokens en vez de la respuesta corta esperada,
  disparando el tiempo por término de ~5s a >100s (confirmado vía
  `ollama ps` + logs de `ollama serve`: una sola llamada llegó a 2000+
  tokens y seguía subiendo). Añadido `DEFAULT_NUM_PREDICT = 300` (mismo fix
  que ya existía en `new/src2/run_llm_rerank.py`). También se detectó que
  correr esta tarea en paralelo con `run_synonym_finding.py` sobrecargaba
  la GPU de 6GB (RTX 4050) - ambas corridas se lanzaron en serie, no en
  paralelo, tras ese hallazgo.
- **La corrida completa (921 filas) no se ha lanzado todavía** - queda para
  cuando quieras correrla tú (mismo criterio que en `synonic/`).

### 2026-07-19 - tarea nueva (Synonym Finding), 100 términos x 4 métodos
- Fuente: `results/synonym_finding_comparison.csv` / `results/synonym_finding_summary.json`
- 4 métodos, todos sobre el mismo pool de candidatos (`en_synonym` completo,
  921 valores únicos, cacheado en `data/cache/`):
  - `embedding_only` - sin LLM, `sentence-transformers/all-MiniLM-L6-v2`.
  - `llm_zero_shot` - `llama3.2:3b`, temp 0.3, prompt en
    `rsc/agents/zero_shot_agent.py::AGENTE_ZERO_SHOT`.
  - `agent_tool_calling` - `llama3.2:3b`, temp 0.2, tool-calling nativo de
    Ollama, prompt + esquema de tools en `rsc/agents/tool_agent.py`.
  - `hybrid_pipeline` - generador `llama3.2:3b` (temp 0.3) + reranker
    `qwen3.5:2b` (temp 0.0), reutilizando `rsc/agents/generator_agent.py` +
    `rsc/agents/reranker_agent_local.py` tal cual.

| método | hit@1 | hit@3 | hit@5 | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|
| embedding_only | 0.62 | 0.81 | 0.85 | 0.731 | 0.747 | 0.01 |
| llm_zero_shot | 0.65 | 0.81 | 0.84 | 0.748 | 0.761 | 1.02 |
| agent_tool_calling | 0.63 | 0.81 | 0.85 | 0.736 | 0.752 | 2.15 |
| hybrid_pipeline | 0.65 | 0.84 | 0.90 | 0.758 | 0.784 | 16.29 |

- Lectura con n=100 (diferencias de 2-3 puntos entre métodos con LLM están
  dentro del margen de ruido, no tratar como concluyentes): el baseline sin
  LLM (`embedding_only`) ya es muy fuerte; `hybrid_pipeline` gana en casi
  todo pero paga ~1600x más tiempo que el baseline y ~8x más que
  `llm_zero_shot` para una mejora modesta; `agent_tool_calling` (la
  arquitectura nueva pedida) queda entre el baseline y `llm_zero_shot`, no
  los supera todavía en este corte.
- Mismo bug de `num_predict` que arriba, encontrado primero aquí (en una
  llamada de `hybrid_pipeline` vía el reranker local) - ver detalle en la
  entrada de la tarea actual.
- **La corrida completa (921 filas x 4 métodos) no se ha lanzado todavía**
  - queda para cuando quieras lanzarla tú (`python run_synonym_finding.py`
  sin `--limit`).

## Qué registrar en cada entrada futura

Fecha, método(s), modelo(s) exacto(s), qué prompt cambió (pegar el texto
nuevo si cambió, o "sin cambios" + archivo:línea), `--limit` usado, y el
summary resultante. El objetivo es poder responder más adelante "¿cambiar
el prompt X o el modelo Y movió hit@1/hit@3/hit@5/ndcg@3/ndcg@5, o fue solo
ruido?" - lo cual necesita más de un punto de dato por combinación
prompt/modelo para decir algo con confianza.

---

## 2026-07-19 (tarde) - Ciclo de experimentación autónomo: splits, harness unificado, y 21 métodos comparados

Pedido del usuario: mejorar el sistema mediante experimentación iterativa,
manteniendo un ciclo completo (baseline -> hipótesis -> implementar -> medir
-> comparar -> quedarse solo con lo que mejora reproduciblemente). Regla de
leakage aclarada por el usuario ese mismo día: el pool de candidatos puede
ser la lista completa de `en_synonym`, lo único prohibido es consultar el
`en_synonym` de la fila que se está evaluando antes de generar su
predicción - ver `[[feedback-leakage-definition]]` en la memoria del
proyecto. Ningún método de este ciclo consulta el gold de su propia fila.

### 0. Estado previo y lo que cambió en la infraestructura

No existían splits train/val/test - todo se evaluaba sobre el dataset
completo o un `head(n)` arbitrario, lo que hacía imposible tunear nada
(modelo de embeddings, pesos, umbrales) sin arriesgar overfitting al propio
conjunto de medición.

- **`data/make_splits.py`** (nuevo): particiona `data/synonyms_clean.csv`
  por término `en` ÚNICO (no por fila - hay 10 términos `en` con dos
  `en_synonym` distintos, ver docstring) con seed fijo (42), 60/20/20 ->
  `data/splits/{train,val,test}.csv` (581/194/194 filas). Excluye además la
  única fila degenerada `en == en_synonym` ("modernisation"/"modernisation"):
  con `exclude_terms=[term]` en `retrieve_top_k`, esa fila es
  estructuralmente imposible de acertar (la respuesta correcta es el propio
  término, que se excluye a propósito del pool de búsqueda) - no es un bug
  del pipeline, es una fila del dataset donde la "sinonimia" es identidad.
- **`run_eval.py` + `src/methods.py`** (nuevos): harness único que
  reemplaza la lógica duplicada de `run_experiment.py`/`run_synonym_finding.py`
  para que todo experimento sea directamente comparable (mismos splits,
  mismas métricas, mismo log). `results/experiment_log.csv` (nuevo, no se
  borra nunca) acumula una fila por corrida: método, split, las 6 métricas,
  n_hit@1/3/5 (conteos absolutos, no solo proporciones - pedido explícito
  del usuario), latencia media, nota de configuración, timestamp.
- **`src/evaluation.py`**: `summarize_results` no incluía `mrr` en el
  agregado (solo en el detalle por fila) pese a que sí se documentaba como
  métrica del proyecto - corregido, ahora el resumen incluye `mrr` y los
  conteos absolutos `n_hit@1/3/5`.
- Nuevas herramientas en `src/tools/`: `bm25_retrieval.py` (BM25 léxico,
  `rank_bm25`), `fusion.py` (Reciprocal Rank Fusion, Cormack et al. 2009 -
  ya citado en `REFERENCES.md`), `cross_encoder_rerank.py` (reranking
  par-a-par con `sentence-transformers.CrossEncoder`). `embedding_retrieval.py`
  ganó `retrieve_scored` (variante de `retrieve_top_k` que devuelve
  `(candidato, score)` para top-N, no solo el top-k final - necesaria para
  alimentar fusión/reranking) y soporte de prefijos "query:"/"passage:" por
  modelo (`EMBEDDER_PREFIXES`) - necesario para que e5/bge rindan como
  fueron entrenados, no solo intercambiar el nombre del modelo.
- Bug de import corregido el mismo día (`rsc` -> `src`, ver historial de
  memoria del proyecto) - sin este fix ningún script de esta carpeta corría.

### 1. Baseline (config original, `all-MiniLM-L6-v2`, sin cambios de código)

| método | split | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|---|
| `embedding_only` | val | 0.546 | 0.706 | 0.804 | 0.642 | 0.642 | 0.682 | 0.019 |
| `embedding_only` | test | 0.536 | 0.722 | 0.773 | 0.629 | 0.644 | 0.665 | 0.022 |
| `llm_zero_shot` | val | 0.660 | 0.814 | 0.840 | 0.738 | 0.753 | 0.764 | 0.652 |
| `agent_tool_calling` | val | 0.572 | 0.706 | 0.804 | 0.656 | 0.653 | 0.693 | 1.366 |
| `hybrid_pipeline` | val | 0.670 | 0.861 | 0.912 | 0.766 | 0.782 | 0.803 | 5.308 |

(`hybrid_pipeline` corrió esta vez a ~5.3s/término, no los ~16-20s/término
de corridas anteriores del mismo pipeline - la caché de embeddings ya
construida y un `num_predict` correctamente acotado explican la diferencia;
no es una modificación de esta sesión.)

### 2. Hipótesis probadas (21 métodos en total, ver `results/experiment_log.csv`
para la tabla completa con todas las corridas)

**Ganadora - cambiar el embedder de retrieval, nada más:**

| método | split | hit@1 | hit@3 | hit@5 | mrr | avg_time_s |
|---|---|---|---|---|---|---|
| **`embedding_e5`** (intfloat/e5-base-v2) | **val** | **0.691** | **0.840** | **0.907** | **0.773** | **0.024** |
| **`embedding_e5`** (intfloat/e5-base-v2) | **test** | **0.603** | **0.789** | **0.871** | **0.704** | **0.025** |
| `embedding_bge` (BAAI/bge-base-en-v1.5) | val | 0.624 | 0.799 | 0.881 | 0.720 | 0.024 |

Sustituir `all-MiniLM-L6-v2` por `intfloat/e5-base-v2` (con el prefijo
`"query: "`/`"passage: "` correcto - sin él e5 rinde mal, es un requisito de
cómo se entrenó el modelo, no un detalle cosmético) sube hit@1 +14.4pt y MRR
+13.1pt en val, sin cambiar la latencia de forma perceptible (0.019s ->
0.024s). Es la única modificación de todo el ciclo que bate a TODOS los
métodos basados en LLM en hit@1/MRR, incluyendo el pipeline completo de 2
LLMs (`hybrid_pipeline`), a ~220x menos latencia.

Diff de errores MiniLM -> e5 en val (hit@1): **35 términos que MiniLM fallaba
y e5 acierta, 7 en sentido contrario** (neto +28, exactamente los +14.4pt
sobre n=194). Patrón cualitativo: e5 rescata pares con más distancia
superficial/léxica pero equivalencia semántica real (`citizenship`->
`nationality`, `feminism`->`women's liberation`, `brain`->`cerebrum`,
`fiscal policy`->`budgetary policy`), casos donde MiniLM se queda con un
vecino léxicamente más parecido pero semánticamente incorrecto
(`citizenship`->`immigration law`, `brain`->`brain bleed`). Los 7 casos que
e5 rompe tienden a sobregeneralizar hacia un término asociado pero no
sinónimo (`demography`->`birth rate` en vez de `population studies`,
`Electronic Business`->`online business` en vez de `e-commerce`).

**Ideas probadas que NO mejoraron (se descartan, documentadas para no repetirlas):**

1. **BM25 (léxico puro)**: hit@1 0.222 en val - muy por debajo de cualquier
   variante de embeddings. Esperado: sinónimos rara vez comparten tokens.
2. **Fusión RRF BM25 + embeddings**, 4 pesos probados (1:1, 4:1, 10:1 a
   favor de embeddings, sobre MiniLM y sobre e5): **todas las variantes
   quedan por debajo del embedder solo** (mejor caso, e5 10:1: hit@1 0.582
   vs 0.691 de e5 solo). RRF pondera por RANGO, no por magnitud de score -
   incluso con peso bajo, el top-1 léxico de BM25 (a menudo irrelevante)
   compite en rango con el top-1 semántico correcto y lo desplaza. Conclusión:
   la señal léxica de BM25 es demasiado ruidosa para este dataset de
   sinónimos - se descarta esta dirección por completo, no solo el peso 1:1.
3. **Cross-encoder reranking** del top-20 (3 combinaciones: stsb-roberta
   sobre MiniLM, stsb-roberta sobre e5, quora-distilroberta sobre e5):
   **las tres empeoran hit@1/MRR respecto al embedder solo** (peor caso,
   quora sobre e5: hit@1 0.397 vs 0.691). Diff de errores e5 -> cross-encoder
   (stsb): en 48/194 términos donde e5 acertaba en el top-1, el reranking lo
   desplaza a favor de un candidato temáticamente relacionado pero no
   sinónimo (`stock market`(correcto)->`capital markets`,
   `nationality`(correcto)->`civic community`). Interpretación: estos
   cross-encoders están entrenados para similitud de ORACIONES completas
   (STS) o duplicados de preguntas (Quora), no para equivalencia estricta de
   frases nominales cortas de 2-4 palabras - su noción de "similar" es
   demasiado laxa para este dataset y sistemáticamente prefiere relacionado
   sobre sinónimo.
4. **Variantes morfológicas** (singular/plural, guión/espacio, quitar
   paréntesis) como queries extra, sobre MiniLM y sobre e5: neutro a
   ligeramente negativo en ambos casos (e5: hit@1 0.655 vs 0.691 sin
   variantes). El embedder ya es robusto a estas variaciones superficiales;
   las queries extra solo añaden candidatos competidores con score parecido
   que a veces desplazan al correcto vía el bono de multi-hit.
5. **Embedder más grande (`e5-large-v2` vs `e5-base-v2`)**: hit@1 0.485,
   peor que incluso el baseline original de MiniLM (0.546), y ~50x más
   lento (1.23s/término). Más parámetros no implica mejor aquí - no se
   investigó la causa raíz en profundidad (posible diferencia de pooling o
   de cómo `sentence-transformers` maneja este checkpoint concreto), se
   descarta por resultado medido, no por intuición.
6. **Ensemble RRF de dos embedders fuertes (e5 + bge)**: hit@1 0.670, MRR
   0.767 - AMBOS ligeramente por debajo de e5 solo (0.691/0.773), pese a que
   hit@5 sí mejora un poco (0.918 vs 0.907). Por el orden de prioridad
   pedido (hit@1 y MRR antes que hit@5), se descarta: la hipótesis de que
   errores "menos correlacionados" entre dos embedders fuertes ayudarían no
   se sostuvo en este dataset.
7. **Todos los métodos basados en LLM** (`llm_zero_shot`, `agent_tool_calling`,
   `hybrid_pipeline`, cada uno probado también anclado con e5 en vez de
   MiniLM): ninguno bate a `embedding_e5` solo en hit@1/MRR, con
   latencias 25x-220x mayores. El mejor de todos (`hybrid_pipeline`,
   MiniLM) queda 2 puntos por debajo en hit@1 y MRR pagando 220x más
   tiempo. Cuando se ancla con e5 en vez de MiniLM, `hybrid_pipeline_e5`
   empeora respecto a la versión MiniLM (posible ruido de la generación a
   temperatura 0.3 - no investigado más a fondo) - de cualquier forma,
   ninguna variante LLM se acerca a justificar su coste frente al embedder
   solo.

### 3. Confirmación final en test (única vez que se toca el split de test)

| método | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|
| `embedding_only` (baseline original) | 0.536 | 0.722 | 0.773 | 0.629 | 0.644 | 0.665 | 0.022 |
| **`embedding_e5` (config final)** | **0.603** | **0.789** | **0.871** | **0.704** | **0.711** | **0.746** | **0.025** |

Mejora en test: **+6.7pt hit@1, +7.5pt MRR, +9.8pt hit@5** - más modesta que
en val (+14.4pt/+13.1pt/+6.5pt) pero consistente en dirección y clara en
magnitud, sobre un split que el sistema no había visto ni una vez antes de
esta confirmación. La brecha val->test (ambos métodos rinden unos puntos
peor en test) sugiere que el split de test cayó, por azar de la partición
por término con n pequeño (194), ligeramente más difícil en general - no un
síntoma de overfitting a val, porque NINGÚN hiperparámetro se tuneó
específicamente para maximizar val (la decisión "usar e5" se tomó por
categoría de modelo, no se barrieron decenas de embedders buscando el que
mejor le fuera a este split concreto).

### 4. Decisión final

**Config ganadora: `embedding_e5` (`intfloat/e5-base-v2`, prefijo
`"query: "`/`"passage: "`, sin ningún otro cambio) reemplaza a
`all-MiniLM-L6-v2` como embedder de retrieval.** Ninguna otra modificación
probada (BM25, fusión, reranking por cross-encoder, variantes morfológicas,
ensemble, LLMs en cualquier combinación) mejora hit@1 o MRR por encima de
este único cambio, y varias lo empeoran claramente. Se cumple el criterio de
aceptación: mejora reproducible en las métricas prioritarias (hit@1, MRR)
sin empeorar desproporcionadamente el resto, a coste de latencia
despreciable.

**Cómo reproducir:**
```bash
python data/make_splits.py                                    # splits (ya generados, seed=42)
python run_eval.py --method embedding_only --split test        # baseline original
python run_eval.py --method embedding_e5 --split test          # config final
python run_eval.py --method <cualquier método de src.methods.ALL_METHODS> --split val   # cualquier experimento de esta lista
```

**Riesgos identificados:**
- n=194 en val/test es suficiente para ver una señal clara (+14pt no es
  ruido) pero no para diferencias de 2-3 puntos entre métodos cercanos
  (p.ej. `hybrid_pipeline` vs `agent_tool_calling_e5` vs `embedding_e5_morph_variants`,
  todos entre 0.65-0.69 hit@1 en val, están dentro de ruido razonable a este
  tamaño de muestra).
- La brecha val->test para `embedding_e5` (-8.8pt hit@1) es mayor que para
  el baseline (-1pt) - vale la pena repetir la partición con otra seed en
  algún momento para confirmar que no es un artefacto de esta partición
  concreta cayendo con casos más difíciles en test para el método que más
  se beneficia del embedder fuerte.
- No hay contaminación entre splits: partición por término `en` único
  (no por fila), pool de candidatos construido del dataset completo mismo
  antes y después del split (no cambia con --limit ni con el split elegido)
  - confirmado en el diseño de `run_eval.py`, no solo declarado.

### 5. Rendimiento sobre el dataset completo (reporte, no tuning)

Pedido del usuario tras cerrar el ciclo: correr la config ganadora sobre
TODO el dataset (no solo val/test) para tener el número global del sistema.
`data/make_splits.py` ahora genera además `data/splits/full.csv`
(train+val+test concatenados, misma fila degenerada excluida, 969 filas) -
**solo para reportar, no se ha vuelto a tunear nada con esto**, train/val/test
siguen siendo los splits reales para cualquier decisión futura.

| método | n | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|---|
| `embedding_only` (baseline original) | 969 | 0.545 | 0.742 | 0.802 | 0.646 | 0.661 | 0.685 | 0.006 |
| **`embedding_e5` (config final)** | **969** | **0.645** | **0.835** | **0.892** | **0.744** | **0.757** | **0.781** | **0.010** |

+10.0pt hit@1, +9.7pt MRR, +9.0pt hit@5 sobre las 969 filas utilizables del
dataset (625/969 exactas en el top-1, 864/969 dentro del top-5). El número
cae, como se esperaba, entre el de val (0.691) y el de test (0.603) - más
cerca de test, consistente con que train+val+test combinados diluyen el
resultado más alto (val) con el más bajo (test) más los ~581 términos de
train que nunca se habían evaluado hasta ahora individualmente.

**Próximos experimentos más prometedores** (no probados aún, por orden de
prioridad):
1. Repetir la partición train/val/test con 2-3 seeds distintas y promediar,
   para separar señal real de varianza de muestreo pequeña - especialmente
   antes de decidir entre `embedding_e5` y candidatos cercanos como
   `hybrid_pipeline`/`agent_tool_calling_e5`.
2. Probar otros embedders orientados a retrieval no probados aún (p.ej.
   `gte-base-en-v1.5`, `mxbai-embed-large-v1`) - e5-base fue el primero en
   ganar claramente, pero no se barrió exhaustivamente la familia.
3. Investigar por qué `e5-large-v2` rindió peor que `e5-base-v2` en vez de
   solo descartarlo - si es un bug de uso (pooling, truncamiento) en vez de
   una limitación real del modelo, podría haber una ganancia adicional sin
   explotar.
4. Calibrar pesos de fusión/umbrales usando el split de TRAIN (581 filas,
   sin usar todavía en este ciclo) en vez de solo probar valores fijos a
   mano sobre val, si se retoma la dirección de fusión con algún otro
   embedder fuerte (no BM25, ya descartado).
5. Error analysis por categoría de término (abreviaturas, términos
   compuestos, frases largas vs cortas) - no se hizo en este ciclo por
   tiempo, y el propio pedido del usuario lo menciona explícitamente como
   dirección de interés.

---

## 2026-07-19 (noche) - Método 1/2/3 pedidos por el usuario: expansión de queries por LLM + fusión, con y sin reranking LLM

Pedido del usuario: implementar y evaluar 3 métodos nuevos frente al
baseline `embedding_e5`, priorizando Hit@1/MRR, con reporte completo
(hit@1/3/5, mrr, ndcg@3/5, latencia, coste estimado, nº de casos que suben a
rank1, nº de casos que empeoran desde rank1). Desarrollo y selección solo en
train/val, test tocado una única vez al final. Todo corre con Ollama local
(`llama3.2:3b` generación, `qwen3.5:2b` reranking) - **coste estimado: $0,
sin llamadas a OpenRouter/API de pago** (existe `openrouter_client.py` con
una key real en `.env`, pero no se usó nada de pago en este ciclo).

### Infraestructura nueva

- `src/tools/multi_query_fusion.py`: 4 estrategias de combinación
  (`max`/`mean`/`weighted`/`rrf`) sobre la MISMA matriz de scores
  (queries x pool), para que comparar entre ellas no mezcle ninguna otra
  variable. `encode_queries()` (nuevo, en `embedding_retrieval.py`) expone
  el paso de embeber queries por separado para poder reusar la matriz
  completa en vez de recalcularla por estrategia.
- `compare_methods.py` (nuevo, raíz): dado un método y un baseline sobre el
  mismo split, cuenta cuántos términos "suben" a rank1 (antes mal, ahora
  bien) y cuántos "bajan" (antes bien, ahora mal) - exactamente las dos
  métricas de movimiento de rank que pidió el usuario. Bug encontrado y
  corregido en el propio desarrollo: el primer merge usaba solo `term` como
  clave, pero hay 10 términos `en` duplicados con dos `en_synonym`
  distintos (documentado en `make_splits.py`) - si ambas copias caen en el
  mismo split, un merge por `term` solo produce un cruce many-to-many
  espurio (194 filas -> 198 tras el merge, detectado por un chequeo de
  tamaño que el propio script imprime como warning). Corregido: merge por
  `(term, ground_truth)`.
- Bug de harness encontrado y corregido en la sesión anterior (ver arriba)
  reafirmado aquí: los nombres de archivo de detalle no llevan el `--limit`
  en el nombre, así que una corrida de humo puede pisar una corrida
  completa - se sigue usando el sufijo `__limit{N}` añadido entonces.

### Método 1 - expansión de queries por LLM + retrieval e5 + fusión, 4 estrategias

Pipeline: `generar_candidatos(term)` (ya existente, siempre antepone el
término original a las variantes generadas - cumple "el término original
debe estar siempre incluido") -> embeber cada query con `e5-base-v2` ->
combinar contra el pool con una de las 4 estrategias -> top-5.

| fusión | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|
| `max` | 0.655 | 0.830 | 0.881 | 0.743 | 0.756 | 0.778 | 0.758 |
| `mean` | 0.753 | 0.897 | 0.912 | 0.822 | 0.839 | 0.845 | 0.740 |
| **`weighted`** (peso 3x al término original) | **0.773** | **0.907** | **0.938** | **0.845** | **0.856** | **0.868** | **0.726** |
| `rrf` | 0.629 | 0.809 | 0.887 | 0.728 | 0.735 | 0.768 | 0.719 |
| (referencia) `embedding_e5` sin expansión | 0.691 | 0.840 | 0.907 | 0.773 | 0.779 | 0.806 | 0.024 |

**`weighted` gana claramente.** `max` y `rrf` quedan incluso por DEBAJO del
baseline sin expansión - mismo patrón que la fusión con BM25 de la sesión
anterior: dar el mismo peso a una query ruidosa (una variante generada por
el LLM) que a la query fiable (el término original) dejar que una variante
mediocre desplace al candidato correcto. `mean` ya corrige bastante (todas
las queries pesan igual, pero al menos promedian en vez de dejar que una
sola domine), y `weighted` termina de arreglarlo dándole al término
original 3x el peso de cada variante individual.

**Barrido de peso** (`scripts_weight_sweep.py`, nuevo - reusa la MISMA
llamada LLM+embedding por término para las 8 relaciones de peso probadas,
evitando 194 llamadas LLM redundantes por valor de peso): pesos 1.5 a 8.0
sobre val, con las queries YA generadas fijas (mismo muestreo LLM para las
8), dieron hit@1 entre 0.763 (peso 1.5) y 0.789 (peso 5.0), todos los pesos
2-6 muy cercanos entre sí.

**Hallazgo de estabilidad (importante)**: `generar_candidatos` usa
`temperature=0.3` en `AGENTE_GENERADOR` - la generación de variantes NO es
determinista. 3 corridas independientes completas de `llm_expansion_weighted`
con peso=5.0 dieron hit@1 = 0.768 / 0.753 / 0.758 (media 0.759, rango 1.5pt)
y mrr = 0.829 / 0.826 / 0.828 (muy estable, rango 0.3pt) - el ruido de
muestreo del LLM mueve hit@1 más que mrr (mrr promedia sobre toda la
posición del rank, hit@1 es un umbral binario más sensible a casos al
límite). La lectura única de peso=3.0 (hit@1 0.773) queda por ENCIMA de esa
banda de peso=5.0, pero con solo 1-2 lecturas por peso no se puede afirmar
con confianza que 3.0 sea mejor que 5.0 de forma reproducible - la
diferencia entre pesos "cercanos" (3-6) está dentro del ruido de muestreo
del propio LLM. Se mantiene **peso=3.0** (el valor original, ya probado dos
veces con resultados consistentes ~0.77 hit@1) en vez de perseguir una
"optimización" de 1-2pt que no se puede distinguir de ruido - decisión
explícita de no sobreajustar un hiperparámetro a variación de muestreo.

### Método 2 - Método 1 (fusión ganadora) + reranking por LLM

Pipeline: top-5 de `llm_expansion_weighted` -> `reordenar_candidatos`
(agente existente, `qwen3.5:2b` - nunca inventa candidatos, nunca ve el
gold, si omite alguno se reañade al final en orden original).

| método | hit@1 | hit@3 | hit@5 | mrr | avg_time_s |
|---|---|---|---|---|---|
| `llm_expansion_weighted` (Método 1, sin rerank) | 0.773 | 0.907 | 0.938 | 0.845 | 0.726 |
| `llm_expansion_rerank` (Método 2, con rerank) | 0.773 | 0.907 | 0.923 | 0.842 | 13.148 |

Diff Método 2 vs Método 1 (mismo split, mismas queries subyacentes):
**9 términos suben a rank1, 9 bajan - cambio neto exactamente 0**, y hit@5
incluso baja un poco (0.938 -> 0.923). El reranker LLM no aporta nada sobre
una lista ya bien ordenada por fusión ponderada, pagando 18x más latencia.
Descartado.

### Método 3 - retrieval e5 de una sola query (sin expansión) + reranking por LLM

Pipeline: `embedding_e5` (una sola query, el término) -> top-5 ->
`reordenar_candidatos`.

| método | hit@1 | hit@3 | hit@5 | mrr | avg_time_s |
|---|---|---|---|---|---|
| `embedding_e5` (sin rerank) | 0.6907 | 0.8402 | 0.9072 | 0.7729 | 0.024 |
| `e5_llm_rerank` (Método 3, con rerank) | 0.6907 | 0.8402 | 0.9072 | 0.7729 | 4.358 |

**Las métricas son IDÉNTICAS hasta el cuarto decimal.** `compare_methods.py`
confirma: 0 términos suben a rank1, 0 bajan - el reranker LLM reprodujo,
término por término, el mismo orden que ya traía `embedding_e5`, sin
cambiar una sola predicción en las 194 filas de val. A 180x la latencia del
retrieval solo. Descartado con la evidencia más limpia posible de que
reordenar un top-5 ya bueno con este LLM (`qwen3.5:2b`, mismo prompt que
Método 2) no aporta nada en este dataset.

### Confirmación en test (única vez que se toca, config ya fija: `llm_expansion_weighted`, peso=3.0)

| método | split | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|---|
| `embedding_e5` (baseline anterior) | test | 0.603 | 0.789 | 0.871 | 0.704 | 0.711 | 0.746 | 0.025 |
| **`llm_expansion_weighted` (config nueva)** | **test** | **0.753** | **0.881** | **0.938** | **0.823** | **0.828** | **0.851** | **0.745** |

**+15.0pt hit@1, +11.9pt MRR, +6.7pt hit@5** sobre el baseline `embedding_e5`
en el split de test, tocado una única vez con la config ya cerrada desde
val. La mejora en test es incluso MAYOR que en val (+8.3pt hit@1 en val) -
buena señal de que no es un artefacto de ajuste fino a val (no se ajustó
nada de forma agresiva: la única decisión de val fue "usar fusión ponderada
en vez de max/mean/rrf", una elección categórica clara, no un barrido fino
de hiperparámetros sobreajustado).

Rank-movement en val (`compare_methods.py`, `llm_expansion_weighted` vs
`embedding_e5`): **25 términos suben a rank1, 9 bajan, neto +16** (encaja
con la diferencia agregada 0.773-0.691=0.082, 0.082*194≈16). Ejemplos que
suben: `stocks`->`shares` (antes `stock market`), `consciousness`->
`awareness` (antes `unconscious state`), `democracy`->`popular government`
(antes `Consensus-based democracy`) - casos donde la query original sola no
encontraba el sinónimo exacto pero alguna variante generada por el LLM sí
lo acercaba lo suficiente. Ejemplos que bajan: `investment`->`investing`
(la fusión eligió `funding`, una variante razonable pero incorrecta),
`sustainable tourism`->`eco-friendly tourism` (fusión eligió `eco-tourism`,
casi sinónimo pero no el gold exacto) - en general, paráfrasis plausibles
pero no exactas que una variante generada acercó más que el término
original.

### Decisión final de esta sesión

**Nueva config ganadora: `llm_expansion_weighted`** (generar hasta 10
queries con `llama3.2:3b` -temperatura 0.3-, incluyendo siempre el término
original; embeber cada una con `e5-base-v2`; fusionar por score ponderado,
3x de peso al término original sobre cada variante; top-5). Reemplaza a
`embedding_e5` solo como mejor config del proyecto. Ningún reranking por
LLM (ni Método 2 sobre esta fusión, ni Método 3 sobre `embedding_e5` solo)
aporta mejora alguna - ambos descartados con evidencia limpia (cambio neto
0 y predicciones idénticas, respectivamente).

**Coste**: $0 - toda la generación y el reranking corren en Ollama local
(`llama3.2:3b`/`qwen3.5:2b`), sin llamadas a APIs de pago. Latencia:
~0.73s/término (30x el retrieval puro de `embedding_e5`, pero sigue siendo
subsegundo, nada comparable a los 5-13s/término de los pipelines con
reranking LLM).

**Riesgo identificado esta sesión**: `AGENTE_GENERADOR` corre a
`temperature=0.3`, así que los resultados de cualquier método que use
`generar_candidatos` (Método 1/2, `hybrid_pipeline`, `llm_zero_shot`, etc.)
tienen varianza real de corrida a corrida (~1-2pt hit@1 observado en 3
repeticiones) - no son perfectamente reproducibles bit a bit. No afecta a
`embedding_e5`/`bm25_only`/cross-encoders (deterministas). Recomendación
para comparaciones finas futuras: fijar `temperature=0` en el generador, o
repetir cada corrida 3x y reportar media±rango en vez de un único número.

**Cómo reproducir:**
```bash
python run_eval.py --method llm_expansion_weighted --split test   # config final
python run_eval.py --method embedding_e5 --split test              # baseline anterior, para comparar
python compare_methods.py --method llm_expansion_weighted --split val   # rank-movement vs baseline
python scripts_weight_sweep.py                                     # barrido de peso (documentación, no producción)
```

**Próximos experimentos más prometedores** (actualizado):
1. Repetir con `temperature=0` en el generador para eliminar la varianza de
   muestreo y poder comparar pesos de fusión (3 vs 5 vs otros) de forma
   limpia - el barrido actual está confundido por el ruido de muestreo.
2. Probar generar MÁS variantes (actualmente hasta 9) o menos, para ver si
   el número de queries por término tiene un óptimo distinto al peso.
3. Explorar si `mean`/`weighted` con el embedder `bge` (segundo mejor
   embedder individual) en vez de `e5` cambia el ranking de estrategias.
4. Seguir sin explorar: error analysis por categoría de término
   (abreviaturas, términos compuestos) - sigue pendiente de sesiones
   anteriores.

### Confirmación sobre el dataset completo (pedido por el usuario tras cerrar el ciclo)

`llm_expansion_weighted` corrido sobre `data/splits/full.csv` (969 filas,
train+val+test combinados, solo reporte - config ya cerrada desde val antes
de esta corrida):

| método | n | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|---|
| `embedding_only` (baseline original) | 969 | 0.545 | 0.742 | 0.802 | 0.646 | 0.661 | 0.685 | 0.006 |
| `embedding_e5` (config anterior) | 969 | 0.645 | 0.835 | 0.892 | 0.744 | 0.757 | 0.781 | 0.010 |
| **`llm_expansion_weighted` (config actual)** | **969** | **0.755** | **0.911** | **0.946** | **0.833** | **0.847** | **0.862** | **0.716** |

732/969 aciertos exactos en top-1, 917/969 (94.6%) dentro del top-5.
+21.1pt hit@1 / +18.7pt mrr sobre el baseline original, +11.0pt hit@1 /
+8.9pt mrr sobre `embedding_e5` - magnitud consistente con los gaps ya
vistos en val y test, sin sorpresas al escalar a las 969 filas. Coste: $0.

## 2026-07-20 - Retune a temperature=0, backend Qwen (pausado), nuevos métodos

Pedido por el usuario: (1) bajar la temperatura del generador a 0 y
re-tunear el peso de fusión solo con val; (2) probar la API de Qwen 3
(OpenRouter, `qwen/qwen3.5-9b`, ya integrada pero nunca usada en el
harness) como backend; (3) explorar métodos adicionales de retrieval/
ranking/agentes/herramientas; (4) objetivo principal: subir Hit@1 y cerrar
el hueco entre Hit@1 y Hit@3/5. Restricción confirmada por el usuario en
plan mode: la API de pago se usa solo con corridas únicas sobre val (sin
barridos), y el modelo Qwen a usar es el ya presente en el repo
(`qwen/qwen3.5-9b`).

### Fase A - temperature=0 + retune limpio del peso

Hipótesis: la varianza de `temperature=0.3` (~1-2pt hit@1 entre repeticiones,
ver entrada "noche" del 2026-07-19) confundía el barrido de pesos anterior.
Cambio: `AGENTE_GENERADOR_T0 = {**AGENTE_GENERADOR, "temperature": 0.0}` en
`src/agents/generator_agent.py` (el original con 0.3 se deja intacto).

Barrido de pesos (`scripts_weight_sweep_t0.py`, val, n=194) corrido DOS
veces completas para comprobar estabilidad:

| peso | hit@1 (run1) | hit@1 (run2) | mrr (run1) | mrr (run2) |
|---|---|---|---|---|
| 1.5 | 0.7732 | 0.7732 | 0.8288 | 0.8288 |
| 2.0 | 0.7835 | 0.7835 | 0.8386 | 0.8386 |
| 2.5 | 0.7835 | 0.7835 | 0.8424 | 0.8424 |
| **3.0** | **0.7835** | **0.7835** | **0.8423** | **0.8423** |
| 4.0 | 0.7784 | 0.7784 | 0.8415 | 0.8415 |
| 5.0 | 0.7835 | 0.7835 | 0.8436 | 0.8436 |
| 6.0 | 0.7835 | 0.7835 | 0.8451 | 0.8451 |
| 8.0 | 0.7680 | 0.7680 | 0.8333 | 0.8333 |

Las dos corridas son **byte-idénticas** en cada peso - a diferencia de
`temperature=0.3`, `temperature=0.0` en Ollama resultó perfectamente
reproducible dentro de la misma sesión/estilo de script. Hit@1 forma una
meseta plana entre pesos 2.0-6.0 (todos 0.7835); se mantiene
`original_weight = 3.0` (ya usado antes, en el centro de la meseta, buen
ndcg@3/5). Confirmación oficial vía `run_eval.py --method
llm_expansion_weighted_t0 --split val`: hit@1 0.7835, mrr 0.8423 (coincide
exactamente con el barrido) - **+1.0pt hit@1 sobre el config anterior
(temp=0.3, 0.773) pero mrr ligeramente por debajo (0.8423 vs 0.845)**: un
resultado mixto, no una victoria clara por sí solo, pero se adopta como la
nueva base de trabajo para el resto del ciclo por su determinismo (necesario
para que las comparaciones posteriores sean confiables), no por el delta en
sí.

### Fase C.3 - Fusión ponderada (no RRF) BM25+e5 - descartada

Idea nueva no probada antes: solo se había fusionado BM25+e5 por RANGO
(RRF), nunca por SCORE normalizado. `weighted_score_fusion()` nueva en
`src/tools/fusion.py` (min-max normaliza cada fuente, suma ponderada).
Barrido de `bm25_weight` (`scripts_bm25_weight_sweep.py`, val, sin LLM):

| bm25_weight | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|
| 0.10 | 0.680 | 0.845 | 0.897 | 0.766 |
| 0.25 | 0.655 | 0.845 | 0.897 | 0.750 |
| 0.50 | 0.598 | 0.814 | 0.892 | 0.713 |
| 1.00 | 0.495 | 0.768 | 0.871 | 0.639 |
| 2.00 | 0.428 | 0.722 | 0.825 | 0.582 |

Incluso el peso BM25 más bajo probado (0.10) queda por debajo de
`embedding_e5` solo (hit@1 0.691, mrr 0.773), y empeora monótonamente al
subir el peso. Confirma con un mecanismo de fusión distinto lo que RRF ya
había mostrado: la señal BM25 es demasiado débil/ruidosa para este dataset.
**Descartado.**

### Fase C.2 - Embedders alternativos dentro del pipeline de expansión

- `mxbai-embed-large-v1`: carga y corre sin problema. `run_eval.py --method
  llm_expansion_weighted_mxbai --split val` → hit@1 0.7526, mrr 0.8274 -
  por debajo de `e5-base-v2` (0.7835/0.8423). **Descartado.**
- `gte-base-en-v1.5`: **no se pudo probar** - su código remoto
  personalizado (rotary embeddings/unpadding) falla al cargar en este
  entorno, tanto en GPU (`CUDA assertion ind >= 0 && ind < ind_dim_size`)
  como en CPU (`IndexError: index ... out of bounds` en `position_ids`,
  ver traceback completo en el historial de comandos de esta sesión) - es
  una incompatibilidad de librerías (transformers/torch/Python 3.14 con el
  código remoto del modelo), no un resultado sobre la calidad del modelo.
  `EMBEDDER_GTE` documentado como no usable en `src/methods.py`.

`e5-base-v2` se mantiene como el mejor embedder disponible para este
pipeline.

### Fase C.1 - Barrido del número de variantes generadas (n)

Idea nueva: ¿menos variantes generadas por el LLM (actualmente hasta 9)
podrían dar mejor señal que más? `scripts_variant_count_sweep.py` genera
UNA vez con n=9 (temperature=0) y trunca `candidatos[:n+1]` para simular
n menor (mismo patrón de reuso que el barrido de pesos):

| n_variants | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|
| 3 | 0.7938 | 0.9072 | 0.9278 | 0.8500 |
| 5 | 0.7938 | 0.9021 | 0.9227 | 0.8485 |
| 7 | 0.7835 | 0.9021 | 0.9227 | 0.8431 |
| 9 | 0.7835 | 0.9021 | 0.9227 | 0.8423 |

n=3/5 superan a n=9 en el script de barrido (+2.05pt hit@1). **Pero** al
confirmar vía el harness oficial llamando directamente a
`generar_candidatos(term, n=3, agente=AGENTE_GENERADOR_T0)` (en vez de
truncar una generación de n=9), el resultado fue más modesto: hit@1 0.7887
(153/194), mrr 0.8461 - **+0.52pt hit@1 sobre n=9, no +2.05pt**. Repetido
una segunda vez de forma independiente: **idéntico** (153/194, 0.7887,
0.8461) - estable, no es ruido de muestreo, pero el barrido offline había
sobreestimado la ganancia. Lección: aunque `temperature=0` es perfectamente
reproducible DENTRO de una misma sesión de script, llamar a
`generar_candidatos` con un `n` distinto desde cero no es matemáticamente
idéntico a truncar una llamada con n=9, pese a que el prompt/instrucciones
enviados al LLM no dependen de `n` (solo el post-procesado de truncado sí)
- alguna fuente de no-determinismo entre procesos separados de Ollama
afecta el resultado exacto, no solo el generador con temperature>0.

Decisión: aun con el delta oficial más pequeño, n=3 **domina a n=9 en TODAS
las métricas de val** (hit@1 +0.52pt, hit@3 empatado, hit@5 +0.51pt, mrr
+0.38pt, ndcg@3/5 ambos mayores) y es reproducible (idéntico en 2 corridas
oficiales separadas) - un caso Pareto genuino, no ruido. **Se adopta n=3**
como nuevo método `llm_expansion_weighted_n3`, con menos tokens generados
que n=9 (ligera reducción de complejidad, no un aumento).

### Fase C.4 - Desglose de errores por categoría de término

`scripts_error_breakdown.py` sobre `llm_expansion_weighted_t0__val.csv`
(n=9, antes de adoptar n=3 - análisis usado para entender el patrón del
gap, no depende del n exacto):

| categoría | n | hit@1 | hit@3 | hit@5 | gap cases (gold en top3-5, no rank1) |
|---|---|---|---|---|---|
| abbreviation_like | 7 | 0.857 | 0.857 | 0.857 | 0 |
| compound | 74 | 0.838 | 0.959 | 0.973 | 9 |
| multi_word_phrase | 57 | 0.825 | 0.912 | 0.930 | 5 |
| single_word | 56 | 0.661 | 0.821 | 0.857 | 9 |

`single_word` es la categoría más débil con diferencia (hit@1 0.66 vs
0.82-0.86 para el resto) y la que más "gap cases" aporta junto con
compound. De los 23 gap cases totales, la distribución del rank del gold es
17 en rank2 y 6 en rank3 - **nunca** en rank4/5, es decir el margen entre
rank1 y rank2 es la señal relevante a explotar (usado en Fase C.5).

### Fase B / C.5 - Backend Qwen (OpenRouter) - iniciado, bug encontrado y arreglado, luego PAUSADO

Se implementó: `AGENTE_GENERADOR_QWEN` (`qwen/qwen3.5-9b`, temp=0) en
`generator_agent.py`; parámetro `llamar_fn` añadido a
`generar_variantes`/`generar_candidatos` para poder inyectar el backend sin
acoplar `generator_agent.py` a `openrouter_client` (el generador local no
debe depender de tener `OPENROUTER_API_KEY` configurada); métodos nuevos en
`src/methods.py`: `llm_expansion_weighted_qwen` (generador Qwen + fusión
ponderada e5), `llm_expansion_weighted_t0_rerank_qwen` (reintento del
rerank en bloque con el modelo FUERTE, para aislar si el hallazgo anterior
de "cero beneficio" era del modelo débil o del enfoque), y
`llm_expansion_gated_rerank_qwen` (rerank CONDICIONADO por margen - idea
nueva, no repite los intentos anteriores). `combine_scores_scored()` nueva
en `multi_query_fusion.py` para exponer el margen rank1-rank2 (verificada
con datos sintéticos: mismo top-k que `combine_scores` en las 4 estrategias
de fusión).

Análisis de margen (`scripts_margin_analysis.py`, val, sin coste - solo
generación local): margen medio 0.027 para rank1 correctos vs 0.009 para
incorrectos - señal útil pero con solape. Con `GATE_THRESHOLD = 0.01`:
62/194 términos "ambiguos", de los cuales 28/42 (67%) de los rank1
incorrectos caen ahí, tocando solo 34/152 (22%) de los ya correctos -
umbral elegido y fijado en `src/methods.py`, cero llamadas a la API pagada
usadas para elegirlo.

**Bug encontrado**: `openrouter_client.py` (el de `synonim2/`, no el de
`~/thesis/openrouter_client.py`) instanciaba el cliente `OpenAI(...)` sin
`timeout` ni `max_retries` - una corrida de humo (`--limit 5`) se quedó
colgada 27+ minutos sin avance de CPU antes de matarla manualmente. Una
llamada mínima aislada con el cliente sin arreglar también reprodujo el
problema; tras añadir `timeout=90.0, max_retries=3` (mismos valores que el
cliente top-level en `~/thesis/openrouter_client.py`), una llamada mínima
respondió en 7.4s. **Arreglo aplicado y confirmado, pero ningún método
Qwen se llegó a correr hasta el final** - el usuario pidió explícitamente
pausar esta línea de trabajo ("forget about the api method for now") justo
cuando se iba a reintentar la corrida de humo tras el arreglo.

Estado: código listo (métodos, umbral, cliente arreglado), pendiente de
ejecución si se retoma. `llm_expansion_weighted_qwen`,
`llm_expansion_weighted_t0_rerank_qwen`, `llm_expansion_gated_rerank_qwen`
están en `ALL_METHODS`/`NEEDS_LLM` pero **sin ningún resultado medido**.

### Decisión final y confirmación en test

`llm_expansion_weighted_n3` (temp=0, n=3, peso=3.0) confirmado en test
(corrida única, después de fijar el config en val):

| método | split | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|---|
| `llm_expansion_weighted` (temp=0.3/n=9, anterior) | test | 0.753 | 0.881 | 0.938 | 0.823 | 0.828 | 0.851 | 0.74 |
| `llm_expansion_weighted_t0` (temp=0/n=9) | test | 0.7474 | 0.9072 | 0.9485 | 0.8271 | 0.8408 | 0.8577 | 0.72 |
| **`llm_expansion_weighted_n3` (temp=0/n=3, nuevo mejor)** | **test** | **0.7680** | **0.9021** | **0.9330** | **0.8345** | **0.8465** | **0.8594** | **0.72** |

Sobre el orden de prioridad del usuario (Hit@1 primero, luego MRR): `n3`
gana claramente al config anterior (+1.5pt hit@1, +1.2pt mrr, +2.1pt
hit@3), con una pérdida real pero pequeña en hit@5 (-0.5pt, 0.933 vs 0.938)
- confirmado como un efecto específico de n=3 vs n=9 (no de la temperatura)
comparando contra `llm_expansion_weighted_t0`, que con el MISMO generador
temp=0 pero n=9 tiene hit@5 0.9485, el más alto de los tres. **Se promueve
`llm_expansion_weighted_n3` a mejor configuración** (ver README.md).

**Ideas descartadas este ciclo**: fusión ponderada BM25+e5 (cualquier
peso), embedder `mxbai-embed-large-v1` dentro del pipeline de expansión.
**No probado por incompatibilidad de entorno**: embedder `gte-base-en-v1.5`.
**Pausado por el usuario, no descartado**: backend Qwen/OpenRouter (3
métodos implementados, cero corridas completadas).

**Comandos de reproducción**:
```bash
python scripts_weight_sweep_t0.py 1                        # barrido de peso a temp=0 (run 1)
python scripts_weight_sweep_t0.py 2                        # repetición para comprobar estabilidad
python run_eval.py --method llm_expansion_weighted_t0 --split val
python scripts_bm25_weight_sweep.py                        # fusión ponderada BM25+e5 (descartada)
python run_eval.py --method llm_expansion_weighted_mxbai --split val   # descartado
python scripts_variant_count_sweep.py                       # barrido de n variantes
python run_eval.py --method llm_expansion_weighted_n3 --split val
python run_eval.py --method llm_expansion_weighted_n3 --split test     # confirmación final
python scripts_error_breakdown.py                           # desglose por categoría
python scripts_margin_analysis.py                            # análisis de margen (para Fase C.5)
```

## 2026-07-20 (continuación) - Backend Qwen retomado: rerank fuerte confirma ganancia grande

El usuario retomó el trabajo pausado con dos pedidos concretos: (1) usar el
LLM de la API para el reranker, comprobando antes que la API con Qwen
funciona; (2) tras encontrar el modelo inicial poco fiable, seleccionar
otro modelo open-source de la API que dé buenos resultados, investigando
en internet cómo usarlo correctamente (el usuario propuso probar
`qwen/qwen3-32b`, con un snippet del SDK oficial de JS/TS de OpenRouter
como referencia - no aplicable directamente a este proyecto en Python,
pero el modelo sugerido sí se adoptó).

### Verificación de la API y diagnóstico del fallo

Antes de nada, reintenté la llamada mínima que había funcionado
previamente (7.4s) - esta vez el proceso se quedó colgado 27+ minutos sin
avance de CPU otra vez, pese al arreglo de `timeout`/`max_retries` de la
sesión anterior. Investigando con una llamada directa sin el wrapper de
reintentos automáticos del SDK: `qwen/qwen3.5-9b` devolvía
`ChatCompletion(choices=None, ..., error={'message': 'Service
unavailable', 'code': 503})` - un fallo real del proveedor upstream, mal
codificado por OpenRouter como una respuesta HTTP 200 con un campo `error`
embebido (en vez de un status HTTP de error), así que la lógica de
reintentos del cliente `openai` (que solo mira status codes HTTP) nunca lo
reintentaba. Confirmado con 3 llamadas directas seguidas: 2 fallos, 1
éxito - ~33% de fallo transitorio en esta muestra pequeña.

Verifiqué que la clave API y OpenRouter en general funcionan bien
(`openai/gpt-4o-mini` respondió sin problema), aislando el fallo al modelo
concreto `qwen/qwen3.5-9b`.

### Cambio de modelo: qwen3.5-9b → qwen3-32b

Consulté la documentación de OpenRouter (`openrouter.ai/docs/features/provider-routing`,
`openrouter.ai/docs/quickstart`) y las páginas de ambos modelos vía
WebFetch:
- Los fallbacks entre proveedores están activados por defecto
  (`allow_fallbacks: true`) - si un modelo solo tiene un proveedor, no hay
  a qué hacer fallback, lo que explica el 503 sin recuperación automática.
- La página de `qwen/qwen3-32b` menciona explícitamente "Multiple
  providers host this model" (mejor redundancia); la de `qwen3.5-9b` no
  lo menciona igual de claro.
- Existe un SDK dedicado de Python para OpenRouter (`pip install
  openrouter`), preferido por la documentación sobre usar el SDK de
  `openai` apuntando a su `base_url` - no se migró a él en este ciclo
  (cambiar de cliente es un cambio mayor, fuera de alcance de este pedido
  puntual), pero queda anotado como posible mejora futura.
- Precios verificados en las páginas de cada modelo (desactualizaban la
  cifra que tenía hardcodeada de una sesión anterior): `qwen3.5-9b`
  $0.10/$0.15 por millón de tokens (prompt/completion), `qwen3-32b`
  $0.08/$0.28.

Prueba directa de `qwen3-32b`: **5/5 llamadas exitosas**, más rápido
(2-6s vs 7-21s de `qwen3.5-9b` con reintentos). Prueba con el prompt real
del reranker (término + 5 candidatos, formato de `reranker_agent.py`):
respuesta JSON limpia envuelta en \`\`\`json, sin bloques `<think>` que
rompieran `parse_candidates` - confirmado antes de cambiar nada en el
código.

Cambios aplicados:
- `AGENTE_RERANKER["modelo"]` en `src/agents/reranker_agent.py`:
  `qwen/qwen3.5-9b` → `qwen/qwen3-32b`.
- `AGENTE_GENERADOR_QWEN["modelo"]` en `src/agents/generator_agent.py`:
  ídem (por consistencia, aunque no es lo que pidió el usuario esta vez -
  sigue sin usarse en ningún método activo).
- `openrouter_client.py`: nueva `_completar_con_reintentos()` compartida
  por `llamar_modelo`/`llamar_modelo_con_uso` - 3 intentos con 2s de
  espera entre ellos, específicamente para el patrón de error embebido
  (no HTTP) que el cliente `openai` no reintenta solo.
- Constantes de precio corregidas: `QWEN_3_32B_PRICE_PER_MTOK = {"prompt":
  0.08, "completion": 0.28}`, `QWEN_35_9B_PRICE_PER_MTOK` actualizado a
  los valores reales verificados.

### Resultado: rerank en bloque con qwen3-32b - la mayor ganancia del proyecto

`llm_expansion_weighted_t0_rerank_qwen` (rerank de TODAS las filas, no solo
las ambiguas) sobre val completo:

| método | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|
| `llm_expansion_weighted_t0` (sin rerank, base) | 0.7835 | 0.9021 | 0.9227 | 0.8423 | 0.8543 | 0.8627 | 0.72 |
| **`llm_expansion_weighted_t0_rerank_qwen`** | **0.8505** | **0.9124** | **0.9227** | **0.8820** | **0.8882** | **0.8924** | **9.08** |

**+6.7pt hit@1, +4.0pt mrr** sobre la base sin rerank - la ganancia más
grande de cualquier cambio individual en todo el proyecto (más grande que
el salto de `all-MiniLM-L6-v2`→`e5-base-v2`, y más grande que la expansión
de queries por LLM). Movimiento de rangos
(`compare_methods.py --baseline llm_expansion_weighted_t0`): **16 términos
subieron el gold a rank1, solo 3 empeoraron** (neto +13/194). Ejemplos de
mejora: `investment` ("funding"→"investing"), `jurisprudence` ("legal
philosophy"→"legal theory") - correcciones semánticas genuinas, no ruido.

**Esto reescribe la conclusión anterior de "el rerank no aporta nada"**
(Métodos 2/3, `llm_expansion_rerank`/`e5_llm_rerank`, ambos con beneficio
cero): esa conclusión era específica del reranker LOCAL DÉBIL
(`qwen3.5:2b`, 2B parámetros) usado en ambos intentos anteriores. Un
reranker realmente fuerte (`qwen3-32b`, 32B) sí aporta, y mucho.

### Rerank condicionado por confianza (Fase C.5, retomada) - pierde contra el rerank en bloque

Con el reranker fuerte ya disponible, corrí también
`llm_expansion_gated_rerank_qwen` (umbral `GATE_THRESHOLD=0.01`, elegido en
la sesión anterior) sobre val completo:

| método | hit@1 | hit@3 | hit@5 | mrr | avg_time_s | llamadas API/194 |
|---|---|---|---|---|---|---|
| `llm_expansion_gated_rerank_qwen` (solo ambiguos) | 0.8351 | 0.9072 | 0.9227 | 0.8733 | 3.78 | ~62 |
| **`llm_expansion_weighted_t0_rerank_qwen` (todos, ganador)** | **0.8505** | **0.9124** | **0.9227** | **0.8820** | **9.08** | **194** |

Rerankear TODO gana por +1.5pt hit@1 / +0.9pt mrr pese a tocar 3x más
filas - la hipótesis original de la Fase C.5 (rerankear solo casos
ambiguos evita "molestar" a los ya correctos) no se sostiene con un modelo
suficientemente fuerte: también mejora casos donde la retrieval ya era
correcta pero no óptima. Se mantiene `llm_expansion_gated_rerank_qwen` en
el código como resultado negativo documentado, no se recomienda su uso.

### Confirmación en test

`llm_expansion_weighted_t0_rerank_qwen` en test (corrida única):

hit@1=0.8505 (idéntico a val, 165/194), hit@3=0.9227, hit@5=0.9485,
mrr=0.8922, ndcg@3=0.8954, ndcg@5=0.9065, avg_time=9.27s. Movimiento de
rangos vs `llm_expansion_weighted_n3` (el mejor sin rerank): **21 subieron
a rank1, 5 empeoraron** (neto +16/194). **+8.3pt hit@1 / +5.8pt mrr** sobre
`llm_expansion_weighted_n3` en test - confirma la magnitud vista en val,
sin señales de sobreajuste al split de validación.

**Se promueve `llm_expansion_weighted_t0_rerank_qwen` a mejor
configuración del proyecto** (ver README.md). `llm_expansion_weighted_n3`
se mantiene documentado como fallback rápido/gratuito.

### Coste real

El campo `usage.cost` que devuelve OpenRouter directamente en cada
respuesta (más fiable que estimar con las constantes de precio
hardcodeadas) dio un coste medio de ~$0.000155/llamada sobre una muestra
de 5 llamadas con el prompt real del reranker (124-125 tokens de prompt,
204-617 tokens de "thinking"/completion - `qwen3-32b` razona internamente
antes de responder, lo que explica tanto la latencia variable por fila
como la mayor parte del coste). **~$0.03 por corrida completa de 194
filas**; gasto total de esta sesión (val + test + smoke tests + la
variante gated descartada) estimado en ~$0.15.

**Comandos añadidos**:
```bash
python run_eval.py --method llm_expansion_weighted_t0_rerank_qwen --split val
python run_eval.py --method llm_expansion_weighted_t0_rerank_qwen --split test
python run_eval.py --method llm_expansion_gated_rerank_qwen --split val
python compare_methods.py --method llm_expansion_weighted_t0_rerank_qwen --baseline llm_expansion_weighted_t0 --split val
python compare_methods.py --method llm_expansion_weighted_t0_rerank_qwen --baseline llm_expansion_weighted_n3 --split test
```

## 2026-07-20 (continuación 2) - Análisis de errores, nuevos intentos, y créditos de OpenRouter agotados a mitad de ciclo

Pedido del usuario: correr el dataset completo con el mejor config y
reportarlo; analizar por separado los casos "gold fuera del top5" vs
"gold en top5 pero mal rankeado"; seguir iterando de forma autónoma
(implementar, correr, analizar, descartar) para acercar Hit@5 al 100% y
mover más casos de rank2-5 a rank1; coste no es prioridad.

### Desglose de errores en val (`llm_expansion_weighted_t0_rerank_qwen`)

| categoría | n | % |
|---|---|---|
| gold en rank1 (correcto) | 165 | 85.1% |
| gold en rank2-5 (recuperable rerankeando mejor) | 14 | 7.2% |
| gold FUERA del top5 (retrieval miss, el reranker no puede arreglarlo) | 15 | 7.7% |

De los 14 casos "rank2-5": 10 en rank2, 2 en rank3, 1 en rank4, 1 en rank5.
De los 15 casos "fuera de top5", se calculó el rank real de cada uno en un
pool ampliado (top-100, misma fusión ponderada): **10/15 tienen el gold en
rank 6-10** (recuperable ampliando el pool antes de rerankear), 1 en
rank14, 1 en rank20, 1 en rank31, 1 en rank81, 1 no aparece ni en el
top-100 (`gregorian` → `plainchant`, caso genuinamente difícil).

### Intento 1: ampliar el pool antes de rerankear (pool10, pool15) - descartado

Hipótesis: si el gold está en rank 6-10 del pool completo, ampliar de 5 a
10-15 candidatos antes de rerankear debería recuperar esos casos. Sobre
val completo:

| método | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|
| pool=5 (base, ganador) | 0.8505 | 0.9124 | 0.9227 | 0.8820 |
| pool=10 | 0.8093 (-4.1pt) | 0.9175 (+0.5pt) | 0.9330 (+1.0pt) | 0.8616 (-2.0pt) |
| pool=15 | 0.7990 (-5.2pt) | 0.9072 (-0.5pt) | 0.9278 (+0.5pt) | 0.8534 (-2.9pt) |

Hit@5 sí mejora un poco como se esperaba, pero **hit@1 y mrr empeoran
mucho más** - más candidatos = más distractores para la decisión de
rank1, incluso con un reranker fuerte. Empeora monótonamente con el
tamaño del pool. **Descartado** (viola la regla de "solo quedarse con
mejoras medibles").

### Intento 2: añadir sugerencias directas del LLM (conocimiento
paramétrico) al pool - descartado

`generar_respuesta_zero_shot_qwen` (Qwen, sin ver el pool ni el gold)
propone hasta 3 sinónimos directamente; se anclan al vocabulario real vía
embeddings y se añaden al pool de 8 antes de rerankear.

| método | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|
| pool=5 (base) | 0.8505 | 0.9124 | 0.9227 | 0.8820 |
| + sugerencias LLM | 0.7938 (-5.7pt) | 0.9021 (-1.0pt) | 0.9227 (+0.0pt) | 0.8491 (-3.3pt) |

Peor en TODAS las métricas, y ni siquiera mejora hit@5 (0 recall nuevo) -
las sugerencias directas del LLM no encuentran nada que la expansión de
queries por embeddings no encontrara ya, solo añaden ruido. **Descartado.**

### Intento 3: ampliar el pool SOLO en casos ambiguos (gating) - resultado no confirmado (ver aviso de créditos abajo)

Idea: en vez de ampliar el pool para TODAS las filas (que dañó hit@1),
ampliar solo cuando la fusión ya es ambigua (mismo `GATE_THRESHOLD=0.01`
de `llm_expansion_gated_rerank_qwen`) - el análisis de margen mostró que
10/17 "retrieval miss" ya caen en el bucket ambiguo (margen<0.01, 66/194
filas), así que la mayoría de las filas seguras (pool=5, ya ganador) no se
tocarían.

**Este experimento NO se pudo completar de forma fiable**: la corrida
completa dio hit@1=0.7835, hit@3=0.9021, hit@5=0.9227, mrr=0.8423 -
**idéntico, hasta el 4º decimal, a `llm_expansion_weighted_t0` (sin
rerank)**. Investigando la causa: la cuenta de OpenRouter (free tier,
`is_free_tier: true`, tope real de ~$0.19) se había agotado por completo
(`GET /api/v1/credits` → `total_credits: 0`) durante las corridas
paralelas anteriores (pool10/pool15/llmguess + el intento de dataset
completo, todas corriendo a la vez). Cada llamada al reranker fallaba con
`402 - insufficient credits` y caía silenciosamente al orden original (el
`try/except` de `reordenar_candidatos` está diseñado para no perder
recall ante un fallo puntual, pero eso significa que un fallo SISTEMÁTICO
produce una corrida que "funciona" sin ningún error, con métricas
indistinguibles de no rerankear en absoluto).

**Bug de observabilidad arreglado** (no es un bug de lógica, es de
visibilidad): `reordenar_candidatos` ahora imprime un `WARNING` en cada
fallo y lleva un contador global (`reranker_agent.fallback_count`,
reseteado por corrida en `run_eval.py::run()`); el summary JSON y
`experiment_log.csv` ahora incluyen `reranker_fallback_count`, y
`run_eval.py` imprime un aviso explícito si `fallback_count > 0` al
terminar la corrida. Esto habría hecho evidente el problema de créditos
en el momento, sin necesidad de investigar manualmente por qué los
números coincidían sospechosamente con el baseline.

**Validez de los resultados de este ciclo, verificada fila por fila**
(`time_seconds` < 1s ⇒ probable fallback, ya que una llamada real tarda
2-30s+): pool10 3/194 filas sospechosas, pool15 1/194, llmguess 7/194 -
todas dispersas, no concentradas al final, consistentes con fallos
transitorios de contención (4 procesos compitiendo por el mismo saldo
menguante) más que con agotamiento total durante esas corridas. **La
conclusión de descartar pool10/pool15/llmguess se mantiene** (si acaso,
el gap real es aún mayor de lo medido, ya que las pocas filas con
fallback favorecen a estos métodos al comportarse como el baseline
pool=5 en vez de mostrar su verdadero rendimiento degradado). El intento
de dataset completo (969 filas) y el intento de gating de pool sí quedan
totalmente invalidados y deben re-correrse cuando haya crédito.

**Pendiente de créditos** (cuenta en $0, `openrouter.ai/settings/credits`
- acción del usuario, no reintentar automáticamente):
1. Re-correr `llm_expansion_gated_pool_rerank_qwen` en val (idea aún sin
   validar).
2. Correr `llm_expansion_weighted_t0_rerank_qwen` sobre `full.csv` (969
   filas) para el reporte descriptivo pedido por el usuario - estimado
   ~$0.15 y ~2.5h a la tasa observada.
3. Continuar iterando sobre ideas nuevas para cerrar el hueco hit@1↔hit@5
   restante (14 casos rank2-5, 15 casos fuera de top5 - de los cuales solo
   `gregorian`→`plainchant` parece genuinamente difícil de recuperar con
   más candidatos).

## 2026-07-21 - Créditos añadidos: harness resumible + corrida completa del dataset

El usuario añadió crédito a la cuenta de OpenRouter y pidió retomar
exactamente donde se había quedado: inspeccionar lo ya hecho antes de
recorrer nada, reutilizar filas ya válidas, recalcular solo las que
fallaron/cayeron a fallback/quedaron a medias, y no reportar nunca un
resultado que mezcle filas válidas con corruptas.

### Verificación antes de gastar nada

`GET /api/v1/credits` seguía devolviendo `total_credits: 0` la primera vez
que se comprobó tras el aviso del usuario - una llamada de prueba directa
reprodujo el mismo error 402 de antes. Se avisó al usuario en vez de
reintentar a ciegas; tras confirmar que había añadido el crédito de
verdad, `total_credits: 7`, `is_free_tier: false`, y una llamada de
prueba respondió "OK" con éxito - confirmado antes de lanzar nada caro.

### Harness resumible (nuevo en `run_eval.py`)

Se inspeccionó `results/eval/` primero: la corrida de dataset completo
anterior (matada tras descubrir el agotamiento de créditos) **nunca llegó
a escribir ningún archivo** - `run_eval.py::run()` solo escribía el CSV al
final del bucle completo, así que no había nada parcial que reutilizar
para esa corrida específica. En vez de simplemente re-lanzarla desde cero,
se construyó capacidad de resumen genérica y reutilizable:

- Cada fila ahora registra `reranker_fallback` (bool) - calculado como el
  delta de `reranker_agent.fallback_count` antes/después de esa fila
  (las filas se procesan secuencialmente, nunca en paralelo, así que el
  delta se atribuye con seguridad).
- Checkpoint incremental: el CSV de detalle se reescribe tras CADA fila
  (no solo al final) - una interrupción ya no pierde el progreso.
- `--resume`: si el archivo de salida ya existe, se cargan las filas con
  `reranker_fallback == False` (confirmadas válidas) por clave
  `(term, ground_truth)` - no solo `term`, por los ~10 términos duplicados
  con distinto `en_synonym` (ver `data/make_splits.py`) - y se recalculan
  solo las que faltan o fallaron. Un archivo de una corrida vieja sin la
  columna `reranker_fallback` (formato anterior al 2026-07-20) se trata
  como "recalcular todo", nunca se asume válido a ciegas.
- Reintentos acotados (`--max-fallback-retries`, default 3): tras el
  primer paso, cualquier fila que siga en fallback se recalcula hasta
  `max_fallback_retries` veces más.
- **Verificación final dura**: si tras los reintentos queda alguna fila en
  fallback, `run()` lanza `RuntimeError` en vez de reportar métricas -
  nunca más una corrida "exitosa" con datos mezclados válidos/corruptos
  (exactamente lo que pasó sin avisar el 2026-07-20).

### Corrida del dataset completo

`llm_expansion_weighted_t0_rerank_qwen` sobre `full.csv` (969 filas) con
`--resume` (sin archivo previo que reutilizar, pero con toda la
infraestructura de seguridad activa desde el minuto uno):

```
969/969 filas procesadas
0 fallbacks (confirmado en el summary: reranker_fallback_count: 0)
0 reintentos necesarios (ninguna fila falló ni una vez)
```

| método | n | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|---|
| `embedding_only` | 969 | 0.545 | 0.742 | 0.802 | 0.646 | 0.661 | 0.685 | 0.006 |
| `embedding_e5` | 969 | 0.645 | 0.835 | 0.892 | 0.743 | 0.757 | 0.781 | 0.010 |
| `llm_expansion_weighted` (sin rerank) | 969 | 0.755 | 0.911 | 0.946 | 0.833 | 0.847 | 0.862 | 0.716 |
| **`llm_expansion_weighted_t0_rerank_qwen`** | **969** | **0.822** | **0.932** | **0.945** | **0.877** | **0.889** | **0.895** | **11.84** |

797/969 top-1 exacto, 916/969 (94.5%) dentro de top-5. **+27.8pt hit@1 /
+23.1pt mrr** sobre el baseline original, **+17.8pt / +13.4pt** sobre
`embedding_e5`, **+6.7pt hit@1 / +4.4pt mrr / +2.1pt hit@3** sobre el
mejor config sin rerank - hit@5 prácticamente igual (0.945 vs 0.946,
dentro de ruido), consistente con que el rerank corrige orden, no recall.
Coste real: ~$0.18 (delta de `usage` en `/api/v1/credits` antes/después).

### Desglose de errores en el dataset completo

| categoría | n | % |
|---|---|---|
| gold en rank1 (correcto) | 797 | 82.2% |
| gold en rank2-5 (recuperable) | 119 | 12.3% |
| gold fuera del top5 (retrieval miss) | 53 | 5.5% |

Distribución de los 119 casos rank2-5: 87 en rank2, 19 en rank3, 11 en
rank4, 2 en rank5 - la mayoría son casos de un solo escalón, coherente con
el patrón ya visto en val. Los 53 retrieval-miss se concentran en
abreviaturas (`evs`→"European Values Study", `EEG`→"brainwave recording")
y parafraseos etimológicos/no obvios (`gospels`→"good news") - mismo
patrón que val, confirma que ampliar el pool (ya descartado por dañar
hit@1) no era la vía correcta para este subconjunto.

**Comandos**:
```bash
python run_eval.py --method llm_expansion_weighted_t0_rerank_qwen --split full --resume
```

## 2026-07-21 (continuación) - Expansión de abreviaturas para los retrieval-miss - descartado

Pedido del usuario: probar la idea #9 de "próximos experimentos" - los
retrieval-miss del dataset completo se concentran en abreviaturas (`evs`,
`EEG`) y frases etimológicas no obvias (`gospels`→"good news"); expandir
la abreviatura a su forma completa antes de generar las queries de
retrieval podría recuperar algunos de estos casos sin el coste en
distractores que tuvo ensanchar el pool.

### Diseño (deliberadamente distinto de los intentos ya descartados)

A diferencia de pool10/pool15/llmguess (que tocaban TODAS las 194 filas de
val), esta idea se activa SOLO si el término parece una abreviatura
(`text_utils.is_abbreviation_like`: un solo token, todo mayúsculas o <=5
caracteres - mismo criterio que `scripts_error_breakdown.py`). Nuevo
agente `src/agents/abbreviation_agent.py::expandir_abreviatura` (local,
Ollama `llama3.2:3b`, gratis) - si el término no parece abreviatura, el
LLM la instrucción le pide devolver el término sin cambios (verificado:
"power"→"Power", "trust"→"Trust", etc. - se filtran como duplicados y no
añaden ninguna query extra). La expansión se usa como una QUERY MÁS de
retrieval (peso=1.0, igual que las variantes generadas), nunca como
candidato directo - cero invención de respuestas.

Solo 7/194 términos de val son "abbreviation_like" con este criterio (y
solo 6/194 de compute son términos genuinamente abreviados: EEG, brain,
imf, locke, power, trust, care - de los cuales solo "EEG" era un
retrieval-miss real en el análisis de errores de val ya hecho). Muestra
pequeña, pero el riesgo también es mínimo: el resto de val (187 filas) no
se toca en absoluto por este cambio.

### Resultado en val (194 filas, corrida oficial)

| método | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|
| pool=5 sin expansión (base) | 0.851 | 0.912 | 0.923 | 0.882 |
| + expansión de abreviaturas | 0.825 | 0.918 | 0.923 | 0.870 |

hit@1 -2.58pt, mrr -1.24pt, hit@5 SIN CAMBIO (0.923 en ambas). A primera
vista parece un resultado negativo claro, pero investigando fila por fila:

- **Las 7 filas "abbreviation_like" no cambiaron EN ABSOLUTO** - EEG sigue
  prediciendo "brain wave test" (incorrecto) en ambas corridas, y las
  otras 6 (ya correctas) siguen igual de correctas. Comparación directa de
  los candidatos de EEG:
  ```
  baseline: ["brain wave test", "bioelectricity", "neural physiology", "seizure disorder", "electromyogram"]
  abbrev:   ["brain wave test", "bioelectricity", "neural physiology", "seizure disorder", "electromyogram"]
  ```
  **Idénticos** - la query extra ("Electroencephalogram", propuesta por el
  LLM local para "EEG") no cambió NADA en el top-5 recuperado. El peso=1.0
  no fue suficiente para desplazar a ningún candidato dado que ya hay
  ~9-10 queries más compitiendo en la fusión ponderada, o simplemente
  `e5-base-v2` no acerca "Electroencephalogram" a "brainwave recording"
  (el gold) más de lo que ya acercaban las variantes generadas.
- **Las 8 filas que SÍ regresaron y las 3 que mejoraron son TODAS términos
  NO-abreviatura** (`administrative procedural law`, `databases`,
  `jurisprudence`, `health care management`, `information law`, `pastoral
  theology`, `pastoral assistance`, `system dynamics`, entre otros) -
  términos donde el código de este método es un no-op total (mismo
  pipeline exacto que el baseline, ni una query extra). Como el pipeline
  de retrieval es idéntico para estas filas, el único componente que pudo
  cambiar es la respuesta del reranker Qwen entre las dos corridas
  SEPARADAS - es decir, **el reranker de Qwen (`temperature=0.0`) no es
  perfectamente reproducible entre llamadas de API separadas**, igual que
  ya se había encontrado con el generador local de Ollama. Esto significa
  que la caída agregada de hit@1/mrr es, muy probablemente, ruido de la
  API, no un efecto real del cambio de código.

### Conclusión

**Descartado** - no por evidencia de que perjudique (el efecto agregado es
indistinguible de ruido de reranking), sino porque **no demuestra ningún
efecto positivo medible**: el único caso de retrieval-miss abreviado
testeable en val (EEG) no se recuperó en absoluto, con el pool de
candidatos recuperados literalmente sin cambios. No cumple la barra de
"solo quedarse con mejoras medibles" - no hay mejora que medir. Se
mantiene el código (`llm_expansion_rerank_qwen_abbrev` en `ALL_METHODS`)
como resultado negativo documentado, no recomendado.

**Hallazgo metodológico nuevo** (más allá de la idea concreta): el
reranker Qwen a temperature=0 tiene el mismo problema de reproducibilidad
imperfecta ENTRE ejecuciones separadas que ya se había caracterizado para
el generador local de Ollama - pendiente un chequeo de estabilidad
propiamente dicho (repetir la misma configuración 2-3 veces) antes de
confiar en comparaciones de una sola corrida entre métodos con rerank Qwen.

**Ideas de seguimiento no perseguidas este ciclo** (documentadas en
README.md): subir el peso de la query de expansión muy por encima de 1.0
(quizás igual o mayor que el peso 3.0 del término original), o probar la
expansión como lookup léxico (BM25) en vez de query semántica (embeddings)
- BM25 fue débil en general pero podría ayudar específicamente aquí, ya
que una expansión de abreviatura y su gold pueden compartir vocabulario
exacto que los embeddings no captan.

**Comandos**:
```bash
python run_eval.py --method llm_expansion_rerank_qwen_abbrev --split val
```

### Seguimiento: subir el peso de la query de expansión - descartado analíticamente, sin gastar en API

Pedido del usuario tras el resultado anterior: probar a subir el peso de
la query de expansión (en vez del peso=1.0 igual que las demás variantes).
Antes de gastar crédito en una corrida completa, se comprobó
analíticamente (solo embeddings, gratis) si CUALQUIER peso podría
funcionar, calculando el rank del gold bajo cada query por separado:

- **EEG**: bajo la query de expansión sola ("Electroencephalogram"), el
  gold "brainwave recording" rankea 13º - PEOR que bajo el término crudo
  "EEG" solo (rank 5º). Ningún peso sobre la expansión puede ayudar
  porque la expansión en sí es una query PEOR que el término crudo para
  este caso.
- **evs**: el modelo local (`llama3.2:3b`) da una expansión INCORRECTA
  ("Evolutionary" en vez de "European Values Study", un nombre de encuesta
  específico de dominio que el modelo local no conoce). Bajo esa
  expansión errónea, el gold rankea 867º - mucho peor que bajo el término
  crudo solo (73º). Subir el peso de una expansión incorrecta empeoraría
  la fusión, no la mejoraría.

Hallazgo adicional: para EEG, el mejor rank posible de cualquier query
individual es 5º (con el término crudo solo) - pero la fusión completa
(término + 6 variantes generadas) lo empeora a 7º, sugiriendo que el
problema real no es "falta peso en la expansión" sino que las variantes
YA generadas diluyen la señal para términos tipo abreviatura.

**Descartado sin correr nada en val/test** - la evidencia analítica ya es
concluyente en ambos casos disponibles (uno muestra que ni con peso
infinito ayudaría, el otro que activamente perjudicaría). Presentado al
usuario, quien decidió no perseguir esta línea más y pasar a la idea de
los 119 casos "wrong rank" (gold en top5, no en rank1) del dataset
completo en su lugar.

## 2026-07-21 (continuación) - Auto-consistencia del reranker (3 llamadas + Borda count) - INTERRUMPIDO, resultados parciales

Idea implementada para atacar los 119 casos "wrong rank" del dataset
completo (87/119 a un solo escalón, rank2): en vez de una sola llamada al
reranker, `reordenar_candidatos_self_consistency`
(`src/agents/reranker_agent.py`) hace 3 llamadas independientes sobre el
mismo término+top5 y agrega por conteo de Borda (posición 1 = 5pts, ...,
posición 5 = 1pt, suma entre las 3 llamadas). Motivado por el hallazgo del
mismo día de que el reranker Qwen a temperature=0 no es perfectamente
reproducible entre llamadas separadas - la idea es tratar esa variabilidad
como señal de mayoría en vez de puro ruido. Nuevo método
`llm_expansion_rerank_qwen_selfconsistency` en `src/methods.py`. Coste:
3x el del método actual (~$0.09/194 filas en vez de ~$0.03).

**Corrida sobre val (194 filas) interrumpida por el usuario a las 141/194
filas** (~73%, ~47s/fila de media - 3 llamadas con "thinking" activado por
defecto en `qwen3-32b` son notablemente más lentas que 1 sola). Gracias al
harness resumible (`--resume`, ver entrada del 2026-07-21 anterior), las
141 filas ya computadas se guardaron en el checkpoint incremental con
`reranker_fallback_count: 0` confirmado - nada que recalcular al retomar,
solo las 53 filas restantes:
```bash
python run_eval.py --method llm_expansion_rerank_qwen_selfconsistency --split val --resume
```

**Resultados PARCIALES (141/194, NO comparables directamente con un val
completo)**:

| métrica | auto-consistencia (parcial, n=141) | baseline, mismas 141 filas |
|---|---|---|
| hit@1 | 0.8369 (118/141) | 0.8511 (120/141) |
| hit@3 | 0.9291 (131/141) | 0.9220 (130/141) |
| hit@5 | 0.9362 (132/141) | 0.9362 (132/141) - empatado |
| mrr | 0.8812 | 0.8874 |

Movimiento de rango sobre este subconjunto: **1 arreglado, 3
empeorados (neto -2)**. De los 10 casos "wrong rank" del baseline dentro
de este subconjunto (justo lo que este método busca arreglar), **solo 1
de 10 se arregló**.

- Arreglado: `computer science law` - base eligió "legal informatics", sc
  acertó "IT law".
- Empeorados: `investment` (base correcto "investing" → sc "funding" -
  el MISMO par ambiguo que ya se había visto voltear en la dirección
  contraria en una corrida anterior de una sola llamada, evidencia de que
  el voto mayoritario de 3 no converge de forma fiable a la respuesta
  CORRECTA, solo a la que el modelo prefiere en la mayoría de muestreos,
  que no siempre es la correcta); `administrative procedural law`
  (correcto → "regulatory law"); `information law` (correcto → "IT law").
  3 de los 4 casos afectados son de dominio legal, pero con solo 4 puntos
  de datos es una muestra demasiado pequeña para confirmar un patrón real.

**Veredicto preliminar (no final, falta el 27% restante)**: la
auto-consistencia NO muestra el efecto esperado hasta ahora - ligeramente
negativa en neto, y apenas toca la categoría de casos que se buscaba
arreglar. Pendiente terminar las 53 filas restantes antes de decidir
mantener o descartar definitivamente.

### 2026-07-22 - Advisor review: task-formulation/split clarification, failure-pattern analysis, reranker model-size comparison

Asesor revisó los resultados y pidió, en orden: (1) confirmar cuál de dos
formulaciones de la tarea se implementó (generación libre vs ranking desde
un pool cerrado), (2) explicar para qué sirven los splits train/val/test
si ningún componente se entrena con gradiente, (3) desglosar los ~15% de
casos fallidos (5.5% retrieval miss vs 12.3% ranking error) por tipo de
término, (4) una tabla comparando tamaño de modelo del reranker vs
latencia/rendimiento, (5) empezar a redactar metodología/setup
experimental/comparación de baselines. Explícitamente pidió DEJAR de
optimizar Hit@1 por ahora - respuesta y trabajo de este ciclo:

**(1) y (2)**: respondidas en `README.md`, nuevas secciones "Task
formulation: which of the two approaches is this?" y "Why
train/validation/test splits, if nothing is trained from scratch?"
(justo después del párrafo de leakage rule). Confirmado: TODOS los
métodos, incluidos los "zero-shot" que generan libremente con un LLM
(`llm_zero_shot`/`llm_zero_shot_e5`), terminan seleccionando/rankeando
desde el pool cerrado de `en_synonym` - el texto libre del LLM solo se usa
como query adicional de retrieval, nunca como respuesta final. Los splits
existen porque la SELECCIÓN de método/hiperparámetros entre ~40
pipelines candidatos es en sí una forma de ajuste a los datos (mismo
riesgo de sobreajuste que tunear hiperparámetros de un modelo entrenado),
aunque no haya backprop involucrado.

**(3) Failure-pattern analysis** (`scripts_failure_analysis.py`, nuevo):
reutiliza `results/eval/error_analysis_rerank_qwen_full.csv` (969 filas,
ya calculado, cero llamadas nuevas a la API) y separa los 172 casos no-
rank1 en "ranking_error" (gold en top5, mal ordenado) vs "retrieval_miss"
(gold nunca recuperado), cruzado con las 4 categorías heurísticas de
término (abbreviation_like/compound/multi_word_phrase/single_word):

| category | n | ranking_error rate | retrieval_miss rate |
|---|---|---|---|
| abbreviation_like | 34 | 5.9% | 8.8% |
| compound | 432 | 11.3% | 3.2% |
| multi_word_phrase | 250 | 12.8% | 4.4% |
| single_word | 253 | 14.2% | 9.9% |

single_word es la categoría más débil en ambos tipos de fallo (confirma el
hallazgo pre-rerank del 2026-07-20 - el reranker fuerte estrechó pero no
cerró esa brecha). Inspección manual de los 53 retrieval-miss (impresos
por el script + `results/eval/failure_analysis_full_by_category.csv`)
encontró dos patrones nuevos, no documentados antes:
- Jerga de dominio / registro: gold es una palabra mucho más general o
  coloquial que el término técnico de entrada (`philosophy`→"wisdom",
  `gospels`→"good news", `communication`→"information exchange").
- **Abreviatura en dirección inversa**: el intento anterior de expansión
  de abreviaturas (2026-07-21, descartado) solo atacaba el caso "input es
  la abreviatura" (EEG, evs). Los datos del dataset completo muestran el
  caso contrario, sin atacar todavía: el GOLD es la abreviatura y el input
  es la forma expandida (`information technology`→"IT", `identity
  management`→"IdM") - un fix que solo expande abreviaturas de INPUT no
  puede tocar estos casos.
- Casos de ranking-error en gold_rank=2 son casi todos sinónimos
  defendibles (`investment`→"investing" bajo "funding";
  `mediation`→"conflict resolution" bajo "dispute resolution") - sugiere
  que parte del 12.3% de ranking-error refleja ambigüedad genuina de
  paráfrasis en las etiquetas gold, no un déficit del modelo.

**(4) Comparación de tamaño de modelo del reranker**: se añadió un método
nuevo, `llm_expansion_weighted_t0_rerank_local` (`src/methods.py`), que
reutiliza la MISMA base (`llm_expansion_weighted_t0`) que el ganador
actual pero rerankea con `qwen3.5:2b` LOCAL (Ollama, gratis) en vez de
`qwen3-32b` por API - antes solo existía una comparación con el modelo
débil sobre una base DISTINTA (temp=0.3, n=9), que mezclaba el efecto del
tamaño del modelo con el de la base de retrieval. Smoke test (n=10, 0
fallbacks) confirmado; corrida completa en val (n=194) lanzada en segundo
plano - resultados pendientes de completar (~194 x ~14s ≈ 45min), se
añadirán a esta entrada cuando terminen. Comando:
```bash
python run_eval.py --method llm_expansion_weighted_t0_rerank_local --split val --note "controlled model-size comparison: qwen3.5:2b local rerank on same base as qwen3-32b winner"
```
Tabla de 3 modelos (2B local / 9B API - descartado por fiabilidad, nunca
llegó a producir números usables / 32B API - ganador) documentada en
`METHODOLOGY.md` §5.

**(5) Metodología**: nuevo `METHODOLOGY.md` con 7 secciones (formulación
de tarea, dataset, setup experimental incluyendo splits y arquitectura del
pipeline, comparación de baselines, comparación de tamaño de reranker,
failure analysis, limitaciones) - primer borrador para el informe final.

También se corrigieron referencias a rutas obsoletas (`rsc/`) en
`REFERENCES.md` y se corrigió la afirmación de que RRF "no está
implementado aún" (sí lo está, `src/tools/fusion.py::reciprocal_rank_fusion`,
usado por `hybrid_rrf_bm25_e5*`/`ensemble_rrf_e5_bge`/`llm_expansion_rrf`).

**Actualización (2026-07-22, corrida completa terminada)**: la corrida
controlada de `llm_expansion_weighted_t0_rerank_local` en val terminó
(194/194 filas, 0 fallbacks):

| config | hit@1 (val) | mrr (val) | avg_time_s |
|---|---|---|---|
| `llm_expansion_weighted_t0` (base, sin rerank) | 0.7835 | 0.8423 | 0.74 |
| + rerank `qwen3.5:2b` local (2B, misma base) | 0.7680 | 0.8369 | 13.21 |
| + rerank `qwen3-32b` API (32B, misma base, ganador) | 0.8505 | 0.8820 | 9.08 |

Hallazgo más claro que "sin efecto": el reranker de 2B, sobre la MISMA
base que el ganador, **empeora** el hit@1 en -1.6pt respecto a no
rerankear en absoluto (no solo "no ayuda") - y encima es más lento
(13.2s) que el modelo de 32B por API (9.08s), casi seguro por hardware
CPU-bound de Ollama en este entorno, no por el tamaño del modelo en sí.
Tabla completa de 3 tamaños de modelo (2B/9B/32B) documentada en
`METHODOLOGY.md` §5 y `README.md` ("What was tried and discarded").

### 2026-07-22 (continuación) - Test-set re-confirmation of the current best config (fallback-verified)

El asesor pidió correr la "test-set confirmation" del config actual
(`llm_expansion_weighted_t0_rerank_qwen`). La confirmación existente
(2026-07-20) predata el tracking de `reranker_fallback` añadido tras el
incidente de créditos agotados (2026-07-21) - su CSV no tiene esa columna,
así que no se puede verificar retroactivamente que las 194 filas tuvieran
una respuesta real del reranker. Se volvió a correr bajo el harness
actual:
```bash
python run_eval.py --method llm_expansion_weighted_t0_rerank_qwen --split test --note "re-confirmation with reranker_fallback tracking (original 2026-07-20 run predates this safety check)"
```
Créditos de OpenRouter verificados antes de gastar ($7 totales, $0.48
usados, sobra margen). Resultado, 194/194 filas, `reranker_fallback_count:
0` confirmado:

| run | hit@1 | hit@3 | hit@5 | mrr | ndcg@3 | ndcg@5 | avg_time_s |
|---|---|---|---|---|---|---|---|
| 2026-07-20 (original, sin verificar fallback) | 0.8505 | 0.9227 | 0.9485 | 0.8922 | 0.8954 | 0.9065 | 9.27 |
| 2026-07-22 (re-confirmación, fallback_count=0) | 0.8299 | 0.9381 | 0.9485 | 0.8840 | 0.8962 | 0.9006 | 13.32 |

Mismo config exacto, mismo prompt, `temperature=0` en ambos - la
diferencia (-2.1pt hit@1, +1.5pt hit@3, hit@5 idéntico, -0.8pt mrr, ~+4s/
término) es la no-determinismo del API de Qwen entre llamadas separadas ya
documentado como riesgo conocido, ahora medido directamente sobre el
config final en vez de inferido de otros métodos. hit@5 prácticamente
idéntico confirma que el CONJUNTO de candidatos considerado es estable;
solo la decisión de orden top1 vs top2/3 cambia en una minoría de casos
reñidos entre corridas. La latencia más alta (13.3s vs 9.3s) es
variación del lado de OpenRouter, no un cambio de código (ni
`run_eval.py` ni `reranker_agent.py` cambiaron entre las dos corridas).

**Decisión**: el número de 2026-07-22 (fallback-verificado) pasa a ser el
número "de confianza" citado en el informe; el de 2026-07-20 se mantiene
documentado como referencia histórica y como evidencia cuantificada de la
variabilidad del reranker entre corridas. `README.md` ("Best configuration
found") y `METHODOLOGY.md` (§4 comparación de baselines) actualizados con
ambos números y la explicación de por qué difieren.

### 2026-07-22 (continuación 3) - Deep-dive failure analysis + two new methods, both discarded with rigorous noise attribution

Asesor (vía usuario) pidió un análisis de errores más profundo (retrieval
miss vs ranking error, con más dimensiones: abreviatura, abreviatura
inversa, jerga, polisemia, variantes morfológicas, distancia léxica,
longitud) y que se implementaran y probaran métodos dirigidos a los
patrones encontrados, no solo propuestas. Se construyó
`scripts_failure_analysis_v2.py` (antes/después del rerank emparejado por
fila, más dimensiones) - hallazgos completos en `README.md` ("Failure-
pattern analysis v2") y `METHODOLOGY.md` §6. Resumen: el solape léxico
término-gold es el correlato de fallo más fuerte encontrado en todo el
proyecto (miss rate 8.9% con solape cero vs 1.7% con solape alto); el
reranking en bloque tiene un coste real no documentado hasta ahora - 43/916
filas ya correctas en rank1 son degradadas por el reranker (no solo "no
mejora", empeora activamente un top1 ya bueno); y de 8 candidatos "reverse-
abbreviation" solo 3 son abreviaturas genuinas tras verificación manual
(`identity management`->IdM, `information technology`->IT, `International
Criminal Court`->ICCt), los otros 5 son palabras cortas normales que la
heurística confunde - se documenta el error de la heurística en vez de
inflar el hallazgo.

**Método 1 probado: `llm_expansion_rerank_qwen_initialism`** - añade el
acrónimo determinista del término (gratis, sin LLM,
`text_utils.generate_initialism`) como query EXTRA en la fusión ponderada,
apuntando a los 3 casos reverse-abbreviation genuinos. Resultado en val:

| métrica | baseline | + initialism (query extra) |
|---|---|---|
| hit@1 | 0.8505 | 0.8660 (+1.5pt) |
| hit@3 | 0.9124 | 0.9227 (+1.0pt) |
| hit@5 | 0.9227 | 0.9227 (0) |
| mrr | 0.8820 | 0.8926 (+1.0pt) |
| retrieval misses recuperados | - | 0 |
| ranking errors corregidos | - | 4 |
| regresiones | - | 1 |
| net rank-1 | - | +3 |

A primera vista parece una mejora real. **Verificación causal fila por
fila (comparando el SET de candidatos, no solo el hit@1 agregado)
descubrió que es enteramente ruido**: de las 5 filas que cambiaron de
estado (4 arregladas + 1 regresión), las 5 tenían un candidate SET
IDÉNTICO al baseline - es decir, el mecanismo nunca tocó esas filas, el
cambio de resultado es 100% no-determinismo del reranker de Qwen entre
llamadas de API separadas (ver "Known risks"). Por separado, sí se
verificó que la query de acrónimo SÍ cambia el pool de candidatos en 36/194
filas (18.6%) - pero en las 36, el status final (correct/error/miss) NUNCA
cambió: la query de un solo token corto nunca pesa lo suficiente frente a
las otras ~10 queries en la media ponderada para desplazar el top-5, el
mismo problema estructural que ya se había visto con la expansión de
abreviaturas de 2026-07-21 y con la idea (descartada analíticamente sin
gastar API) de subir el peso de la expansión. Verificado directamente
fuera del harness para los 2 casos reverse-abbreviation con datos en val-
equivalentes (`information technology`, `identity management`): el top-5
con y sin la query de acrónimo es BYTE-IDÉNTICO en ambos casos.
**Veredicto: DESCARTADO** - el hit@1 agregado positivo no es atribuible al
mecanismo, es ruido de medición.

**Método 1b (variante quirúrgica): `llm_expansion_rerank_qwen_initialism_exact`**
- en vez de sumar el acrónimo como query dentro de la fusión ponderada
(diluida por las otras ~10 queries), busca una coincidencia EXACTA
(case-insensitive) del acrónimo determinista contra el pool real de
candidatos y, si existe, INYECTA esa entrada directamente en el top-5
antes de rerankear (nunca inventa un candidato, solo promueve uno ya
existente en el pool). Confirmado offline (sin harness) que esto SÍ
recupera `information technology` -> "IT" como candidato (coincidencia
exacta encontrada en el pool), aunque no `identity management` -> "IdM"
(el acrónimo determinista da "IM", no "IdM" - no hay match exacto,
limitación conocida y documentada en el docstring de
`generate_initialism`).

**Resultado final en val (completado 2026-08-03, corrida retomada tras
quedar interrumpida en 143/194 filas)**: hit@1 0.8351 (vs baseline 0.8505,
-1.5pt), mrr 0.8735 (vs 0.8820, -0.9pt), net rank-1 -3/194
(`scripts_method_comparison.py`: 1 ranking error corregido - `ict in
education`, gold rank 2->1 - contra 4 regresiones). Verificación causal
(comparación del SET de candidatos pre-rerank, no solo el hit@1 agregado,
mismo protocolo que los dos métodos anteriores de esta sección): las 5
filas que cambiaron de estado (la única corrección + las 4 regresiones)
tienen SET de candidatos IDÉNTICO al baseline. La inyección exacta de
acrónimo nunca disparó en ninguna de las 5 - ni siquiera en `information
technology`, el único caso donde offline sí se había confirmado que el
mecanismo encuentra un match exacto (el término no cayó en las filas
recomputadas de esta corrida por azar de qué 51 filas quedaban
pendientes, o el match no se mantuvo tras el resto del pipeline; no
verificado más a fondo dado el resultado ya concluyente). **Veredicto:
DESCARTADO**, mismo patrón que los dos métodos anteriores - el -3 neto es
enteramente ruido de no-determinismo del reranker Qwen entre llamadas de
API, no un efecto del mecanismo. Con este tercer resultado, los tres
intentos dirigidos a los patrones de fallo encontrados en el análisis v2
(reverse-abbreviation vía query extra, reverse-abbreviation vía inyección
exacta, solape léxico cero vía BM25 gateado) han sido descartados con la
misma verificación causal - el patrón de fallo está bien caracterizado
(ver "Failure-pattern analysis v2" en README.md) pero ningún mecanismo de
retrieval-side barato lo ha corregido todavía; sigue abierto en
`METHODOLOGY.md` §7 como limitación, no como línea de trabajo activa.

**Método 2 probado: `llm_expansion_lexical_gated_rerank_qwen`** - cuando
el top-5 de embeddings no comparte NINGÚN token con el término (la
condición con peor tasa de acierto medida en el análisis), añade hasta 2
candidatos BM25 (que por construcción sí comparten tokens) antes de
rerankear. Resultado en val:

| métrica | baseline | + BM25 gateado |
|---|---|---|
| hit@1 | 0.8505 | 0.8299 (-2.1pt) |
| hit@3 | 0.9124 | 0.9124 (0) |
| hit@5 | 0.9227 | 0.9227 (0) |
| mrr | 0.8820 | 0.8717 (-1.0pt) |
| retrieval misses recuperados | - | 0 |
| ranking errors corregidos | - | 3 |
| regresiones | - | 7 |
| net rank-1 | - | -4 |

Mismo análisis causal aplicado: la condición de disparo (solape léxico
cero) se cumple en 50/194 filas (25.8%) - pero solo 2 de esas 50
cambiaron de estado final (1 arreglada `ict in education`, 1 regresión
`power in organisations` - el mecanismo en sí es un empate 1-a-1, efecto
neto CERO). Las otras 8 filas que cambiaron de estado (2 arregladas + 6
regresiones) tenían pool IDÉNTICO al baseline en las filas donde SÍ hubo
disparo pero no cambiaron... en realidad las 8 filas restantes que
cambiaron de estado NO fueron de las 50 filas con disparo - es decir, son
enteramente ruido del reranker en filas que este método ni siquiera
tocó. **Veredicto: DESCARTADO** - la condición de disparo es demasiado
frecuente (26% de filas) para el efecto real que produce (neto cero
cuando SÍ actúa), y el -4 agregado es predominantemente ruido, no una
señal real de que el método perjudique.

**Lección metodológica reforzada**: en este proyecto, comparar hit@1
agregado entre dos corridas NO basta para atribuir causalidad a un
cambio de código, porque el reranker de Qwen (temperature=0) no es
reproducible entre llamadas de API separadas y esa variabilidad por sí
sola puede producir swings de ±2-4pt en 194 filas. La verificación
correcta es: ¿el SET de candidatos que le llega al reranker cambió para
las filas cuyo resultado cambió? Si no, el cambio de resultado es ruido,
no efecto del método, sin importar en qué dirección apunte el hit@1
agregado. `scripts_method_comparison.py` (nuevo) automatiza el primer
paso (fixes/regresiones/retrieval-misses-recuperados) pero la
verificación de candidate-set-idéntico se hizo a mano aquí y debería
incorporarse al script en una futura iteración.

**Idea adicional evaluada analíticamente, no implementada (ahorro de presupuesto de API)**:
el hallazgo de que 43/916 regresiones del rerank son top1 YA CORRECTOS que
el reranker degrada (ver "Failure-pattern analysis v2" en README.md)
sugiere una posible mitigación: proteger el rank1 pre-rerank cuando su
margen de fusión (`combine_scores_scored`) es alto, revirtiendo si el
reranker intenta reemplazarlo. Antes de gastar API en implementarlo y
probarlo, se calculó el margen real (gratis, local) de las 43 filas
regresadas: media 0.0154, mediana 0.0142, con 16/43 por DEBAJO del
`GATE_THRESHOLD=0.01` ya usado para "ambiguo". La distribución no está
limpiamente separada de la zona ambigua - un umbral de protección
razonable solo protegería una fracción de las 43 (las de margen más alto),
y el efecto neto es incierto sin conocer también la distribución de margen
de las 682 filas que se quedaron correctamente en rank1 (no calculado, 682
llamadas locales adicionales, ~17min). Dado que el pre-chequeo analítico
no fue claramente positivo (a diferencia de la verificación exact-match
para initialism_exact, que sí confirmó el mecanismo funciona para su caso
objetivo antes de gastar en la API), se documenta como idea NO
implementada esta ronda en vez de gastar presupuesto en una validación de
bajo prior. Candidata para "Most promising next experiments" si se retoma
más adelante con el cálculo de margen completo de las filas ya-correctas.

### 2026-08-03 - Cierre del ciclo de fixes dirigidos + comparación MAS controlada (agente dinámico vs pipeline fijo)

Retomada la sesión desde el punto donde había quedado el 2026-07-22. La
corrida de `llm_expansion_rerank_qwen_initialism_exact` en val, que había
quedado interrumpida en 143/194 filas, se completó con `--resume`
(resultado y veredicto documentados arriba, en la entrada de "continuación
3": descartado, mismo patrón de ruido que los dos métodos anteriores).

Pedido del usuario: dar contenido empírico a la pregunta de si este
proyecto constituye un MAS (multi-agent system) genuino. El código ya
tenía un agente de tool-calling dinámico (`src/agents/tool_agent.py`,
método `agent_tool_calling_e5`) que decide en tiempo de ejecución cuántas
veces llamar a `retrieve_candidates` antes de finalizar, en vez del orden
fijo generador->retrieval->reranker del pipeline principal. Comparación
existente (2026-07-19, embedder e5 en ambos, sin reranker fuerte en
ninguno de los dos): `agent_tool_calling_e5` hit@1 0.660 vs
`hybrid_pipeline_e5` hit@1 0.613 - pero `hybrid_pipeline_e5` usa
internamente el reranker LOCAL débil (`qwen3.5:2b`, ya confirmado en §5 de
`METHODOLOGY.md` que activamente empeora el rank-1), así que esa
comparación confunde "pipeline fijo vs agente dinámico" con "reranker
débil vs sin reranker". Plano `embedding_e5` sin ningún LLM (hit@1 0.691)
supera a ambos - ni el pipeline fijo (con el reranker débil) ni el agente
dinámico (sin reranker) añadían valor neto en ese punto del proyecto.

Para aislar la pregunta correctamente, se implementó un método nuevo,
`agent_tool_calling_e5_rerank_qwen` (`src/methods.py`): mismo agente de
tool-calling dinámico, pero su top-5 pasa por el MISMO reranker fuerte
(`qwen/qwen3-32b`) que usa el pipeline fijo ganador
(`llm_expansion_weighted_t0_rerank_qwen`). Esto permite comparar
"arquitectura de agente único con control de flujo dinámico" vs
"pipeline de roles fijos" bajo la etapa de reranking idéntica, en vez de
bajo presupuestos de LLM distintos.

**Resultado en val (n=194, completado 2026-08-03)**: pipeline fijo hit@1
0.851/mrr 0.882/9.08s vs agente dinámico hit@1 0.830/mrr 0.866/18.96s - el
pipeline fijo gana en las 4 métricas y es ~2x más rápido.
`scripts_method_comparison.py` + inspección manual del SET de candidatos
(mismo protocolo de verificación causal que arriba) confirma que esta vez
la diferencia **SÍ es real, no ruido**: 0/10 filas regresadas inspeccionadas
tienen el mismo set de candidatos que el pipeline fijo - las búsquedas
autoelegidas por el agente genuinamente cambian qué llega al reranker. El
agente recupera 5 retrieval-misses del pipeline fijo (p.ej. `EEG`->"brainwave
recording", `nonresponse`->"no response") vía paráfrasis que la expansión
fija nunca genera, corrige 7 ranking errors, pero regresiona 14 casos ya
correctos y empeora 1 a miss - neto -4/194. Detalle completo, interpretación
y la idea de fusión de candidatos como línea futura no implementada: ver
`METHODOLOGY.md` §6 (sección nueva) y §8 (limitaciones).
