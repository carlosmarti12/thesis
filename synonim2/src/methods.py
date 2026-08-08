"""
Dispatcher central de métodos para run_eval.py. Cada método recibe (term,
ctx) y devuelve una lista de candidatos ordenada (mejor primero), nunca ve
el en_synonym de la fila que está resolviendo - el ground truth solo se lee
en run_eval.py DESPUÉS de llamar a get_candidates.

MethodContext precarga y cachea (por proceso) los recursos caros que cada
método necesita: índices de embeddings (uno por nombre de embedder usado),
índice BM25, y el cross-encoder. Todo se construye de forma perezosa (solo
si el método pedido lo necesita) para no pagar el coste de, p.ej., cargar
bge-base si solo se está evaluando embedding_only.
"""

from dataclasses import dataclass, field

from openrouter_client import llamar_modelo as llamar_modelo_openrouter

from .agents.abbreviation_agent import expandir_abreviatura
from .agents.generator_agent import AGENTE_GENERADOR_QWEN, AGENTE_GENERADOR_T0, generar_candidatos
from .agents.reranker_agent import reordenar_candidatos as reordenar_candidatos_qwen
from .agents.reranker_agent import reordenar_candidatos_self_consistency
from .agents.reranker_agent_local import reordenar_candidatos
from .agents.tool_agent import buscar_sinonimo_con_tools
from .agents.zero_shot_agent import AGENTE_ZERO_SHOT_QWEN, generar_respuesta_zero_shot
from .text_utils import generate_initialism, is_abbreviation_like, morphological_variants, normalize
from .tools.bm25_retrieval import build_bm25_index, retrieve_bm25_scored
from .tools.cross_encoder_rerank import get_cross_encoder, rerank as ce_rerank
from .tools.embedding_retrieval import (
    encode_queries,
    get_embedder,
    get_solution_index,
    retrieve_scored,
    retrieve_top_k,
)
from .tools.fusion import reciprocal_rank_fusion, weighted_score_fusion
from .tools.multi_query_fusion import combine_scores, combine_scores_scored

EMBEDDER_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_BGE = "BAAI/bge-base-en-v1.5"
EMBEDDER_E5 = "intfloat/e5-base-v2"
EMBEDDER_E5_LARGE = "intfloat/e5-large-v2"
EMBEDDER_MXBAI = "mixedbread-ai/mxbai-embed-large-v1"
# EMBEDDER_GTE ("Alibaba-NLP/gte-base-en-v1.5") was tried but its custom
# remote-code rotary-embedding implementation crashes on load in this
# environment (IndexError in position_ids, both on GPU and CPU, unrelated
# to transformers-supported architectures) - not usable here, see
# EXPERIMENTS_LOG.md.

# temp=0 config for the generator (see agents/generator_agent.py) - the
# fusion weight chosen at temp=0 after the clean retune (see
# EXPERIMENTS_LOG.md "temp=0 retune"), used by every llm_expansion_weighted_*
# method below (temp=0 outperforms the old temp=0.3 default and removes most
# LLM-sampling noise from further comparisons).
LLM_EXPANSION_WEIGHTED_T0_WEIGHT = 3.0

# Umbral de margen (score rank1 - score rank2 en la fusión ponderada) para
# el rerank condicionado por confianza (llm_expansion_gated_rerank_qwen) -
# elegido offline sobre val con margin_analysis (cero llamadas a la API de
# pago para fijarlo, ver EXPERIMENTS_LOG.md): a este umbral, 62/194 términos
# de val se consideran "ambiguos" (28 de los 42 rank1 incorrectos caen aquí,
# vs solo 34 de los 152 correctos - captura la mayoría de los errores
# tocando una minoría de los aciertos).
GATE_THRESHOLD = 0.01

CROSS_ENCODER_STSB = "cross-encoder/stsb-roberta-base"
CROSS_ENCODER_QUORA = "cross-encoder/quora-distilroberta-base"

ALL_METHODS = [
    "embedding_only",
    "embedding_bge",
    "embedding_e5",
    "embedding_e5_large",
    "embedding_morph_variants",
    "embedding_e5_morph_variants",
    "bm25_only",
    "hybrid_rrf_bm25_embedding",
    "hybrid_rrf_bm25_e5",
    "hybrid_rrf_bm25_e5_weighted",
    "hybrid_rrf_bm25_e5_weighted10",
    "ensemble_rrf_e5_bge",
    "cross_encoder_rerank",
    "cross_encoder_rerank_e5",
    "cross_encoder_quora_rerank_e5",
    "llm_zero_shot",
    "agent_tool_calling",
    "hybrid_pipeline",
    "llm_zero_shot_e5",
    "agent_tool_calling_e5",
    "agent_tool_calling_e5_rerank_qwen",
    "hybrid_pipeline_e5",
    # Método 1 (pedido por el usuario): expansión de queries por LLM +
    # retrieval e5 para cada variante, 4 estrategias de combinación.
    "llm_expansion_max",
    "llm_expansion_mean",
    "llm_expansion_weighted",
    "llm_expansion_rrf",
    # Método 2: la mejor estrategia de Método 1 (elegida en val) + reranking
    # por LLM (reordenar_candidatos - nunca inventa, nunca ve el gold).
    "llm_expansion_rerank",
    # Método 3: retrieval e5 de una sola query (el término) + reranking LLM.
    "e5_llm_rerank",
    # Ciclo 2026-07-20: generador a temperature=0 (retune limpio del peso,
    # ver EXPERIMENTS_LOG.md) + variantes con otros embedders dentro del
    # mismo pipeline de expansión.
    "llm_expansion_weighted_t0",
    "llm_expansion_weighted_mxbai",
    "llm_expansion_weighted_n3",
    # Backend Qwen (OpenRouter, de pago) en vez de Ollama local.
    "llm_expansion_weighted_qwen",
    "llm_expansion_weighted_t0_rerank_qwen",
    "llm_expansion_gated_rerank_qwen",
    # Ciclo 2026-07-20 (continuación) - error analysis en val mostró que
    # 15/194 casos fallan porque el gold NUNCA aparece en el top-5 (el
    # reranker no puede arreglar lo que no ve) - de esos, 10/15 tienen el
    # gold en rank 6-10 del pool completo. Estos métodos amplían el pool
    # enviado al reranker (10/15 candidatos en vez de 5) para darle más
    # margen de recall antes de recortar a top_k=5.
    "llm_expansion_rerank_qwen_pool10",
    "llm_expansion_rerank_qwen_pool15",
    # Añade una(s) sugerencia(s) directa(s) del LLM (conocimiento
    # paramétrico, sin ver el pool ni el gold) al conjunto de candidatos
    # antes de rerankear - apunta a los mismos casos de "retrieval miss"
    # pero por una vía distinta (memoria del LLM en vez de vecinos más
    # cercanos por embedding).
    "llm_expansion_rerank_qwen_llmguess",
    # pool10/pool15/llmguess (arriba) resultaron peor en hit@1/mrr en val
    # (más candidatos = más distractores para la decisión de rank1) pese a
    # mejorar ligeramente hit@5 - este método aplica el ensanchamiento SOLO
    # a los casos ambiguos (mismo GATE_THRESHOLD que
    # llm_expansion_gated_rerank_qwen), dejando el pool=5 ya ganador intacto
    # para los casos donde la fusión ya está segura.
    "llm_expansion_gated_pool_rerank_qwen",
    # 2026-07-21: los retrieval-miss del dataset completo se concentran en
    # abreviaturas (EEG, evs...) - este método expande el término a su
    # forma completa (local, gratis) SOLO si parece una abreviatura
    # (text_utils.is_abbreviation_like) y la añade como query EXTRA de
    # retrieval (no como candidato) - el resto de términos no se toca.
    "llm_expansion_rerank_qwen_abbrev",
    # 2026-07-21 (continuación): los 119 casos "wrong rank" (gold en top5,
    # no en rank1, 87/119 a solo un escalón de distancia en rank2) son el
    # hueco más grande que queda. Auto-consistencia: 3 llamadas al reranker
    # sobre el mismo top-5, agregadas por Borda count - explota el hallazgo
    # de que el reranker de Qwen no es perfectamente determinista entre
    # llamadas (ver EXPERIMENTS_LOG.md) como señal de mayoría, no como ruido
    # a ignorar.
    "llm_expansion_rerank_qwen_selfconsistency",
    # 2026-07-22: comparación controlada tamaño-de-modelo pedida por el
    # asesor - MISMA base (llm_expansion_weighted_t0) que el ganador actual,
    # solo cambia el reranker: qwen3.5:2b local en vez de qwen3-32b vía API.
    # Antes solo existía una comparación con reranker débil sobre una base
    # DISTINTA (llm_expansion_weighted, temp=0.3, n=9) - esto aisla el efecto
    # del tamaño del modelo del efecto de la base de retrieval.
    "llm_expansion_weighted_t0_rerank_local",
    # 2026-07-22 (failure-pattern analysis v2): dos métodos nuevos, cada
    # uno justificado por un patrón medido en el análisis extendido
    # (scripts_failure_analysis_v2.py), no repeticiones de intentos ya
    # descartados.
    #
    # (a) reverse-abbreviation: el gold es la abreviatura y el input es la
    # forma expandida (`information technology` -> gold `IT`) - el intento
    # anterior de expansión de abreviaturas (2026-07-21) solo atacaba la
    # dirección contraria. Añade el acrónimo DETERMINISTA del término
    # (sin LLM, gratis) como query extra de retrieval solo si el término
    # tiene 2+ palabras de contenido.
    "llm_expansion_rerank_qwen_initialism",
    # (a2) versión quirúrgica del anterior - inyección exacta en vez de
    # query extra en la fusión ponderada (ver dispatch block para el
    # porqué del cambio de enfoque).
    "llm_expansion_rerank_qwen_initialism_exact",
    # (b) lexical-gated BM25: el análisis mostró que las filas donde el
    # top-5 de embeddings NO comparte ningún token léxico con el término
    # tienen retrieval-miss rate 12.9% y ranking-error rate 9.7%, contra
    # 0%/4.3% cuando SÍ hay solape léxico alto - la señal léxica es el
    # predictor más fuerte encontrado. A diferencia del hybrid BM25+e5
    # ponderado ya descartado (aplicado a TODAS las filas, perdió contra
    # embedding_e5 solo), esto solo mezcla candidatos BM25 cuando el pool
    # actual de embeddings tiene solape léxico CERO con el término -
    # ámbito mucho más estrecho, dirigido específicamente al patrón
    # medido.
    "llm_expansion_lexical_gated_rerank_qwen",
]

# Métodos que necesitan un LLM local vía Ollama - usados por run_eval.py para
# decidir si hay que avisar de coste/latencia antes de correr sobre un split grande.
NEEDS_LLM = {
    "llm_zero_shot", "agent_tool_calling", "hybrid_pipeline",
    "llm_zero_shot_e5", "agent_tool_calling_e5", "agent_tool_calling_e5_rerank_qwen", "hybrid_pipeline_e5",
    "llm_expansion_max", "llm_expansion_mean", "llm_expansion_weighted", "llm_expansion_rrf",
    "llm_expansion_rerank", "e5_llm_rerank",
    "llm_expansion_weighted_t0", "llm_expansion_weighted_mxbai", "llm_expansion_weighted_n3",
    "llm_expansion_weighted_qwen", "llm_expansion_weighted_t0_rerank_qwen",
    "llm_expansion_gated_rerank_qwen",
    "llm_expansion_rerank_qwen_pool10", "llm_expansion_rerank_qwen_pool15",
    "llm_expansion_rerank_qwen_llmguess", "llm_expansion_gated_pool_rerank_qwen",
    "llm_expansion_rerank_qwen_abbrev", "llm_expansion_rerank_qwen_selfconsistency",
    "llm_expansion_weighted_t0_rerank_local",
    "llm_expansion_rerank_qwen_initialism", "llm_expansion_rerank_qwen_initialism_exact",
    "llm_expansion_lexical_gated_rerank_qwen",
}

# Estrategia de fusión usada por "llm_expansion_rerank" (Método 2) - se fija
# aquí tras elegirla en el split de VALIDACIÓN (ver EXPERIMENTS_LOG.md), no
# se ha vuelto a tocar test para decidirla. "weighted" ganó claramente a
# max/mean/rrf en val (hit@1 0.773 vs 0.655/0.753/0.629).
LLM_EXPANSION_RERANK_FUSION = "weighted"


@dataclass
class MethodContext:
    pool_terms_raw: list[str]  # en_synonym del dataset COMPLETO, sin dedup

    _embedding_indices: dict = field(default_factory=dict)  # embedder_name -> (terms, embeddings)
    _bm25_index: tuple | None = field(default=None)
    _cross_encoders: dict = field(default_factory=dict)  # model_name -> CrossEncoder

    def embedding_index(self, embedder_name: str = EMBEDDER_MINILM):
        if embedder_name not in self._embedding_indices:
            embedder = get_embedder(embedder_name)
            terms, embeddings = get_solution_index(
                self.pool_terms_raw, embedder=embedder, embedder_name=embedder_name,
                cache_name=f"solution_pool_{embedder_name.split('/')[-1]}",
            )
            self._embedding_indices[embedder_name] = (terms, embeddings, embedder)
        return self._embedding_indices[embedder_name]

    def bm25_index(self):
        if self._bm25_index is None:
            self._bm25_index = build_bm25_index(self.pool_terms_raw)
        return self._bm25_index

    def cross_encoder(self, model_name: str = CROSS_ENCODER_STSB):
        if model_name not in self._cross_encoders:
            self._cross_encoders[model_name] = get_cross_encoder(model_name)
        return self._cross_encoders[model_name]


def get_candidates(method: str, term: str, ctx: MethodContext, top_k: int = 5) -> list[str]:
    if method == "embedding_only":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MINILM)
        return retrieve_top_k(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_MINILM,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "embedding_bge":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_BGE)
        return retrieve_top_k(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_BGE,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "embedding_e5":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        return retrieve_top_k(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "embedding_e5_large":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5_LARGE)
        return retrieve_top_k(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5_LARGE,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "embedding_morph_variants":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MINILM)
        queries = [term] + morphological_variants(term)
        return retrieve_top_k(
            queries, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_MINILM,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "embedding_e5_morph_variants":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = [term] + morphological_variants(term)
        return retrieve_top_k(
            queries, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "bm25_only":
        terms, bm25 = ctx.bm25_index()
        scored = retrieve_bm25_scored(term, terms, bm25, top_n=top_k, exclude_terms=[term])
        return [c for c, _ in scored]

    if method == "hybrid_rrf_bm25_embedding":
        emb_terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MINILM)
        emb_scored = retrieve_scored(
            [term], emb_terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_MINILM,
            top_n=20, exclude_terms=[term],
        )
        bm25_terms, bm25 = ctx.bm25_index()
        bm25_scored = retrieve_bm25_scored(term, bm25_terms, bm25, top_n=20, exclude_terms=[term])

        emb_ranked = [c for c, _ in emb_scored]
        bm25_ranked = [c for c, _ in bm25_scored]
        return reciprocal_rank_fusion([emb_ranked, bm25_ranked], weights=[1.0, 1.0], top_k=top_k)

    if method == "hybrid_rrf_bm25_e5":
        emb_terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        emb_scored = retrieve_scored(
            [term], emb_terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_n=20, exclude_terms=[term],
        )
        bm25_terms, bm25 = ctx.bm25_index()
        bm25_scored = retrieve_bm25_scored(term, bm25_terms, bm25, top_n=20, exclude_terms=[term])

        emb_ranked = [c for c, _ in emb_scored]
        bm25_ranked = [c for c, _ in bm25_scored]
        return reciprocal_rank_fusion([emb_ranked, bm25_ranked], weights=[1.0, 1.0], top_k=top_k)

    if method == "hybrid_rrf_bm25_e5_weighted":
        # Peso de embeddings >> BM25: BM25 solo desempata/rescata coincidencias
        # léxicas exactas, no debería poder desplazar al top del embedder fuerte.
        # Peso elegido en el set de validación (ver EXPERIMENTS_LOG.md).
        emb_terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        emb_scored = retrieve_scored(
            [term], emb_terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_n=20, exclude_terms=[term],
        )
        bm25_terms, bm25 = ctx.bm25_index()
        bm25_scored = retrieve_bm25_scored(term, bm25_terms, bm25, top_n=20, exclude_terms=[term])

        emb_ranked = [c for c, _ in emb_scored]
        bm25_ranked = [c for c, _ in bm25_scored]
        return reciprocal_rank_fusion([emb_ranked, bm25_ranked], weights=[4.0, 1.0], top_k=top_k)

    if method == "hybrid_rrf_bm25_e5_weighted10":
        emb_terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        emb_scored = retrieve_scored(
            [term], emb_terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_n=20, exclude_terms=[term],
        )
        bm25_terms, bm25 = ctx.bm25_index()
        bm25_scored = retrieve_bm25_scored(term, bm25_terms, bm25, top_n=20, exclude_terms=[term])

        emb_ranked = [c for c, _ in emb_scored]
        bm25_ranked = [c for c, _ in bm25_scored]
        return reciprocal_rank_fusion([emb_ranked, bm25_ranked], weights=[10.0, 1.0], top_k=top_k)

    if method == "ensemble_rrf_e5_bge":
        # A diferencia de la fusión con BM25 (que consistentemente empeoró
        # hit@1/MRR - señal léxica demasiado débil), e5 y bge son dos
        # embedders FUERTES e independientes (arquitecturas/datos de
        # entrenamiento distintos) - la hipótesis es que sus errores estén
        # menos correlacionados entre sí que con una señal léxica, y que
        # fusionarlos pueda rescatar casos donde uno acierta y el otro no.
        e5_terms, e5_embeddings, e5_embedder = ctx.embedding_index(EMBEDDER_E5)
        e5_scored = retrieve_scored(
            [term], e5_terms, e5_embeddings, embedder=e5_embedder, embedder_name=EMBEDDER_E5,
            top_n=20, exclude_terms=[term],
        )
        bge_terms, bge_embeddings, bge_embedder = ctx.embedding_index(EMBEDDER_BGE)
        bge_scored = retrieve_scored(
            [term], bge_terms, bge_embeddings, embedder=bge_embedder, embedder_name=EMBEDDER_BGE,
            top_n=20, exclude_terms=[term],
        )
        e5_ranked = [c for c, _ in e5_scored]
        bge_ranked = [c for c, _ in bge_scored]
        return reciprocal_rank_fusion([e5_ranked, bge_ranked], weights=[1.0, 1.0], top_k=top_k)

    if method == "cross_encoder_rerank":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MINILM)
        scored = retrieve_scored(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_MINILM,
            top_n=20, exclude_terms=[term],
        )
        candidates = [c for c, _ in scored]
        return ce_rerank(term, candidates, cross_encoder=ctx.cross_encoder(CROSS_ENCODER_STSB), top_k=top_k)

    if method == "cross_encoder_rerank_e5":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        scored = retrieve_scored(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_n=20, exclude_terms=[term],
        )
        candidates = [c for c, _ in scored]
        return ce_rerank(term, candidates, cross_encoder=ctx.cross_encoder(CROSS_ENCODER_STSB), top_k=top_k)

    if method == "cross_encoder_quora_rerank_e5":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        scored = retrieve_scored(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_n=20, exclude_terms=[term],
        )
        candidates = [c for c, _ in scored]
        return ce_rerank(term, candidates, cross_encoder=ctx.cross_encoder(CROSS_ENCODER_QUORA), top_k=top_k)

    if method == "llm_zero_shot":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MINILM)
        guesses = generar_respuesta_zero_shot(term)
        queries = guesses if guesses else [term]
        return retrieve_top_k(
            queries, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_MINILM,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "agent_tool_calling":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MINILM)
        return buscar_sinonimo_con_tools(
            term, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_MINILM, top_k_final=top_k,
        )

    if method == "hybrid_pipeline":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MINILM)
        queries = generar_candidatos(term)
        top_n = retrieve_top_k(
            queries, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_MINILM,
            top_k=top_k, exclude_terms=[term],
        )
        return reordenar_candidatos(term, top_n)

    if method == "llm_zero_shot_e5":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        guesses = generar_respuesta_zero_shot(term)
        queries = guesses if guesses else [term]
        return retrieve_top_k(
            queries, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_k=top_k, exclude_terms=[term],
        )

    if method == "agent_tool_calling_e5":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        return buscar_sinonimo_con_tools(
            term, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5, top_k_final=top_k,
        )

    if method == "agent_tool_calling_e5_rerank_qwen":
        # Comparación MAS controlada: mismo agente dinámico de tool-calling
        # que "agent_tool_calling_e5" (el LLM decide cuántas veces llamar a
        # retrieve_candidates), pero su top-5 pasa por el MISMO reranker
        # fuerte (qwen3-32b) que usa el pipeline fijo en
        # "llm_expansion_weighted_t0_rerank_qwen". Aísla la pregunta "¿la
        # arquitectura de agente único con control de flujo dinámico llega
        # al nivel del pipeline de roles fijos si se le da la misma etapa
        # final de reranking?" del efecto ya conocido del reranker en sí.
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        top_n = buscar_sinonimo_con_tools(
            term, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5, top_k_final=top_k,
        )
        return reordenar_candidatos_qwen(term, top_n)

    if method in ("llm_expansion_max", "llm_expansion_mean", "llm_expansion_weighted", "llm_expansion_rrf"):
        fusion_method = method.removeprefix("llm_expansion_")
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        # generar_candidatos SIEMPRE antepone el término original a las
        # variantes generadas por el LLM (ver agents/generator_agent.py) -
        # así se cumple "el término original siempre debe estar incluido".
        queries = generar_candidatos(term)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        return combine_scores(
            fusion_method, query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k,
        )

    if method == "llm_expansion_rerank":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        top5 = combine_scores(
            LLM_EXPANSION_RERANK_FUSION, query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k,
        )
        return reordenar_candidatos(term, top5)

    if method == "e5_llm_rerank":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        top5 = retrieve_top_k(
            [term], terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_k=top_k, exclude_terms=[term],
        )
        return reordenar_candidatos(term, top5)

    if method == "llm_expansion_weighted_t0":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        return combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )

    if method == "llm_expansion_weighted_n3":
        # n=3 variantes generadas (en vez de las 9 por defecto) rindió mejor
        # en val en un barrido a temperature=0 (hit@1 +2.05pt, mrr +0.8pt
        # sobre n=9 - ver EXPERIMENTS_LOG.md) - menos variantes generadas
        # significa menos oportunidad de que una parafraseo ruidosa del LLM
        # desplace al candidato correcto.
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, n=3, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        return combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )

    if method == "llm_expansion_weighted_qwen":
        # Igual que llm_expansion_weighted_t0 pero generando con Qwen
        # (qwen/qwen3.5-9b vía OpenRouter, de pago) en vez del llama3.2:3b
        # local, para ver si un generador más grande produce mejores
        # variantes de búsqueda. Peso de fusión NO se re-barre con la API de
        # pago - se reutiliza el elegido en Fase A (ver EXPERIMENTS_LOG.md).
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_QWEN, llamar_fn=llamar_modelo_openrouter)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        return combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )

    if method == "llm_expansion_weighted_t0_rerank_qwen":
        # Reintento del rerank "en bloque" (Métodos 2/3, ambos con beneficio
        # cero) pero con el modelo FUERTE (qwen3.5-9b vía API) en vez del
        # reranker local débil (qwen3.5:2b) usado antes - para aislar si el
        # hallazgo anterior era del enfoque (rerankear todo) o del modelo.
        top5 = get_candidates("llm_expansion_weighted_t0", term, ctx, top_k=top_k)
        return reordenar_candidatos_qwen(term, top5)

    if method == "llm_expansion_weighted_t0_rerank_local":
        # Comparación controlada de tamaño de modelo (pedida por el asesor,
        # 2026-07-22): misma base exacta que el ganador
        # (llm_expansion_weighted_t0_rerank_qwen), pero rerankeando con
        # qwen3.5:2b LOCAL (Ollama, gratis) en vez de qwen3-32b vía API. La
        # comparación anterior con este modelo débil ("llm_expansion_rerank",
        # discardada) usaba una base distinta (temp=0.3, n=9) - aquí se
        # aisla el efecto del TAMAÑO del reranker manteniendo todo lo demás
        # fijo.
        top5 = get_candidates("llm_expansion_weighted_t0", term, ctx, top_k=top_k)
        return reordenar_candidatos(term, top5)

    if method == "llm_expansion_rerank_qwen_initialism":
        # Reverse-abbreviation fix (2026-07-22, ver ALL_METHODS arriba para
        # la justificación completa): añade el acrónimo determinista del
        # término (si tiene 2+ palabras) como query EXTRA de retrieval,
        # igual de gratis y sin LLM que morphological_variants - no
        # reemplaza ninguna query existente, solo suma una más antes de la
        # fusión ponderada.
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        initialism = generate_initialism(term)
        if initialism:
            existing_norms = {normalize(q) for q in queries}
            if normalize(initialism) not in existing_norms:
                queries.append(initialism)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        top5 = combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        reranked = reordenar_candidatos_qwen(term, top5)
        return reranked[:top_k]

    if method == "llm_expansion_rerank_qwen_initialism_exact":
        # Variante quirúrgica de llm_expansion_rerank_qwen_initialism
        # (2026-07-22): la versión anterior sumaba el acrónimo como query
        # MÁS dentro de la fusión ponderada - verificado directamente
        # (fuera del harness de evaluación) que esto NUNCA cambia el top-5
        # para los casos objetivo (`information technology`, `identity
        # management`): con ~10 queries en la media ponderada, una query
        # de una sola palabra corta nunca pesa lo suficiente para ganar,
        # el mismo problema estructural que ya se había visto al intentar
        # subir el peso de la expansión en general. Esta variante evita la
        # fusión por completo: si el acrónimo determinista coincide EXACTO
        # (case-insensitive) con una entrada real del pool de candidatos,
        # esa entrada se inyecta directamente en el top-5 antes de
        # rerankear (nunca inventa un candidato nuevo, solo promueve uno
        # que ya existe en el pool). Disparo muy poco frecuente y de
        # riesgo casi nulo (requiere una coincidencia EXACTA de string, no
        # una heurística difusa).
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        top5 = get_candidates("llm_expansion_weighted_t0", term, ctx, top_k=top_k)
        initialism = generate_initialism(term)
        if initialism:
            pool_by_norm = {normalize(t): t for t in terms}
            exact_match = pool_by_norm.get(normalize(initialism))
            if exact_match and exact_match not in top5:
                top5 = top5 + [exact_match]
        reranked = reordenar_candidatos_qwen(term, top5)
        return reranked[:top_k]

    if method == "llm_expansion_lexical_gated_rerank_qwen":
        # Lexical-gated BM25 rescue (2026-07-22, ver ALL_METHODS arriba
        # para la justificación completa): si el top-5 de embeddings no
        # comparte NINGÚN token con el término (solape léxico cero - la
        # condición con peor tasa de acierto medida en el análisis), se
        # añaden hasta 2 candidatos BM25 (que por construcción SÍ comparten
        # tokens, `retrieve_bm25_scored` descarta score<=0) al pool antes
        # de rerankear. El resto de filas (solape >0) no se tocan en
        # absoluto - a diferencia de pool10/pool15 (ensanchaban el pool de
        # embeddings en TODAS las filas y perdieron).
        top5 = get_candidates("llm_expansion_weighted_t0", term, ctx, top_k=top_k)
        term_tokens = set(normalize(term).split())
        top5_tokens = set()
        for c in top5:
            top5_tokens |= set(normalize(c).split())
        pool = list(top5)
        if term_tokens and not (term_tokens & top5_tokens):
            bm25_terms, bm25 = ctx.bm25_index()
            bm25_hits = retrieve_bm25_scored(term, bm25_terms, bm25, top_n=2, exclude_terms=[term])
            existing_norms = {normalize(c) for c in pool}
            for cand, _score in bm25_hits:
                if normalize(cand) not in existing_norms:
                    pool.append(cand)
                    existing_norms.add(normalize(cand))
        reranked = reordenar_candidatos_qwen(term, pool)
        return reranked[:top_k]

    if method == "llm_expansion_gated_rerank_qwen":
        # Rerank CONDICIONADO por confianza (Fase C.5, idea nueva - no repite
        # los intentos anteriores de rerank en bloque): solo se invoca al
        # reranker fuerte (Qwen vía API) cuando el margen entre el rank1 y
        # el rank2 de la fusión ponderada es pequeño (decisión ambigua).
        # Umbral elegido offline sobre el análisis de márgenes de
        # llm_expansion_weighted_t0 en val (ver EXPERIMENTS_LOG.md) - cero
        # llamadas a la API para fijarlo, solo para los casos ambiguos.
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        top5, margin = combine_scores_scored(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        if margin < GATE_THRESHOLD:
            return reordenar_candidatos_qwen(term, top5)
        return top5

    if method in ("llm_expansion_rerank_qwen_pool10", "llm_expansion_rerank_qwen_pool15"):
        # Igual que llm_expansion_weighted_t0_rerank_qwen pero enviando un
        # pool MÁS ANCHO (10 o 15 candidatos en vez de 5) al reranker antes
        # de recortar a top_k=5 - el error analysis en val mostró que 10/15
        # casos "retrieval miss" (gold fuera del top-5) tienen el gold en
        # rank 6-10 del pool completo, así que el reranker nunca llega a
        # verlos con el pool estrecho.
        pool_size = 10 if method == "llm_expansion_rerank_qwen_pool10" else 15
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        pool = combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=pool_size, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        reranked = reordenar_candidatos_qwen(term, pool)
        return reranked[:top_k]

    if method == "llm_expansion_rerank_qwen_llmguess":
        # Añade hasta 3 candidatos anclados a partir de propuestas directas
        # de Qwen (conocimiento paramétrico, SIN ver el pool ni el gold -
        # generar_respuesta_zero_shot_qwen solo recibe el término) al pool
        # de 8 candidatos de la fusión ponderada, antes de rerankear. Las
        # propuestas del LLM se anclan al vocabulario real vía embeddings
        # (retrieve_top_k) - nunca se inyecta texto libre del LLM como
        # candidato final, solo se usa para encontrar el término REAL más
        # cercano en el pool (mismo principio que llm_zero_shot).
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        pool = combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=8, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )

        guesses = generar_respuesta_zero_shot(term, agente=AGENTE_ZERO_SHOT_QWEN, llamar_fn=llamar_modelo_openrouter)
        if guesses:
            guess_matches = retrieve_top_k(
                guesses, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
                top_k=3, exclude_terms=[term],
            )
            for g in guess_matches:
                if g not in pool:
                    pool.append(g)

        reranked = reordenar_candidatos_qwen(term, pool)
        return reranked[:top_k]

    if method == "llm_expansion_gated_pool_rerank_qwen":
        # Ensancha el pool a 10 SOLO cuando la fusión ya es ambigua
        # (mismo GATE_THRESHOLD que llm_expansion_gated_rerank_qwen) - para
        # el resto (mayoría, casos donde pool=5 ya es el ganador probado)
        # se mantiene el pool=5 sin cambios, evitando el daño a hit@1/mrr
        # que mostró ensanchar el pool para TODAS las filas.
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        _, margin = combine_scores_scored(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=5, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        pool_size = 10 if margin < GATE_THRESHOLD else 5
        pool = combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=pool_size, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        reranked = reordenar_candidatos_qwen(term, pool)
        return reranked[:top_k]

    if method == "llm_expansion_rerank_qwen_abbrev":
        # Igual que llm_expansion_weighted_t0_rerank_qwen, pero si el
        # término PARECE una abreviatura (is_abbreviation_like) se le pide
        # a un LLM local (gratis) su forma expandida y se añade como una
        # query MÁS de retrieval (no como candidato) - el resto de
        # términos sigue exactamente el mismo camino que el método
        # ganador, cero riesgo de tocar los casos que ya funcionan.
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        if is_abbreviation_like(term):
            expansion = expandir_abreviatura(term)
            if expansion:
                existing_norms = {normalize(q) for q in queries}
                if normalize(expansion) not in existing_norms:
                    queries.append(expansion)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        top5 = combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        reranked = reordenar_candidatos_qwen(term, top5)
        return reranked[:top_k]

    if method == "llm_expansion_rerank_qwen_selfconsistency":
        # Igual que llm_expansion_weighted_t0_rerank_qwen (pool=5, el
        # ganador), pero el rerank final usa auto-consistencia (3 llamadas
        # + Borda count) en vez de una sola llamada - apunta a los 119
        # casos "wrong rank" del dataset completo, la mayoría a un solo
        # escalón (rank2) de ser correctos.
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
        top5 = combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        reranked = reordenar_candidatos_self_consistency(term, top5, n_calls=3)
        return reranked[:top_k]

    if method == "llm_expansion_weighted_mxbai":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_MXBAI)
        queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
        query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_MXBAI)
        return combine_scores(
            "weighted", query_embeddings, terms, embeddings,
            exclude_terms=[term], top_k=top_k, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )

    if method == "hybrid_pipeline_e5":
        terms, embeddings, embedder = ctx.embedding_index(EMBEDDER_E5)
        queries = generar_candidatos(term)
        top_n = retrieve_top_k(
            queries, terms, embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
            top_k=top_k, exclude_terms=[term],
        )
        return reordenar_candidatos(term, top_n)

    raise ValueError(f"Unknown method: {method}")
