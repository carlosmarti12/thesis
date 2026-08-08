# Referencias académicas

Fuentes que respaldan las decisiones metodológicas de este proyecto -
recopiladas para el informe final (petición del asesor, 2026-07-19: "try to
find some academic sources that support your arguments and implementation
choices"). Cada entrada dice explícitamente a qué decisión de este código
se aplica.

## Métricas de evaluación (`src/evaluation.py`)

**Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of
IR techniques. ACM Transactions on Information Systems (TOIS), 20(4),
422-446.** https://doi.org/10.1145/582415.582418

Define NDCG (Normalized Discounted Cumulative Gain), la métrica estándar de
information retrieval que este proyecto usa como `ndcg@3`/`ndcg@5`. Sustenta
por qué se reportan hit@k (acierto binario) *junto con* ndcg@k (penaliza
gradualmente según la posición del acierto) en vez de solo uno de los dos -
son complementarios, no redundantes, y es la pareja de métricas más citada
en literatura de IR para justificar exactamente esta elección.

## Expansión de queries con LLM (`src/agents/generator_agent.py`, `src/agents/zero_shot_agent.py`)

**Wang, L., Yang, N., & Wei, F. (2023). Query2doc: Query expansion with
large language models. Proceedings of EMNLP 2023, 9414-9423.**
(arXiv:2303.07678)

**Jagerman, R., Zhuang, H., Qin, Z., Wang, X., & Bendersky, M. (2023). Query
expansion by prompting large language models.** (arXiv:2305.03653)

Ambos respaldan el patrón usado en `generator_agent.py` (generar variantes
del término con un LLM antes de recuperar) y en `zero_shot_agent.py`
(generación libre anclada después al pool): la evidencia empírica es que
pedirle a un LLM variantes/paráfrasis de una query mejora el retrieval
posterior frente a usar solo la query original - es el mismo mecanismo,
aplicado aquí a sinónimos de términos académicos en vez de a queries de
buscador.

## Fusión de rankings (`src/tools/fusion.py::reciprocal_rank_fusion`)

**Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank
fusion outperforms Condorcet and individual rank learning methods.
Proceedings of the 32nd International ACM SIGIR Conference, 758-759.**
https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf

Sustenta `reciprocal_rank_fusion` (implementado y usado por
`hybrid_rrf_bm25_e5`, `hybrid_rrf_bm25_e5_weighted`, `ensemble_rrf_e5_bge`
y `llm_expansion_rrf` en `src/methods.py`) - RRF es la técnica estándar, no
ad-hoc, para combinar rankings de sistemas heterogéneos (BM25 léxico +
embeddings semánticos, o múltiples queries generadas por LLM) sin
necesitar recalibrar sus scores a la misma escala. Nota: en este proyecto,
RRF perdió consistentemente frente a la fusión `weighted` (ver README.md,
"What was tried and discarded") - la referencia respalda por qué se probó
como alternativa seria, no que haya ganado.

## Agentes que usan tools / tool-calling (`src/agents/tool_agent.py`)

**Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., & Cao,
Y. (2022). ReAct: Synergizing reasoning and acting in language models.**
https://arxiv.org/abs/2210.03629

**Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M.,
Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language
models can teach themselves to use tools.** https://arxiv.org/abs/2302.04761
(pre-imprenta: arXiv:2302.04761)

Sustentan la arquitectura de `tool_agent.py`: un LLM que decide en tiempo de
ejecución cuándo y cuántas veces invocar una tool externa (aquí,
`retrieve_candidates` contra el pool de embeddings) en vez de seguir una
secuencia fija de pasos - ReAct es la referencia original de intercalar
razonamiento y acción vía tools; Toolformer es la referencia de que un LLM
puede aprender/decidir cuándo una llamada a tool mejora su respuesta frente
a responder directamente. Justifica por qué esta implementación es
cualitativamente distinta del pipeline fijo (`hybrid_pipeline`) y de los
grafos MAS de iteraciones anteriores (`new/src2`, `synonic/src/graph.py`),
que fijan el orden de los pasos en el código en vez de dejar que el modelo
decida.

## Pendiente

Buscar 1-2 fuentes específicas sobre WordNet como recurso léxico (para
comparar con `synonic/src/open_vocab.py` si se cita en el informe) y sobre
la tarea de "synonym extraction"/"synonym mining" en terminología de dominio
académico, que es más específica al problema de este dataset que la
literatura general de IR/agentes de arriba.
