"""
RAG Verification paradigm (Methodology III-A/C): decomposes the output into
atomic claims, embeds them, and checks each against a FAISS-indexed knowledge
corpus. Time complexity: embedding pass O(C) for C atomic claims plus
nearest-neighbor search O(log V); space complexity dominated by the
vector-store, O(V * d_embed).
"""
from __future__ import annotations

import re
import time
from typing import List, Sequence

import numpy as np

from detectors.base import DetectionResult, HallucinationDetector
from engine.memory_profiler import vector_store_memory_mb


def split_into_claims(text: str) -> List[str]:
    """Sentence-level atomic-claim proxy. Swap for an LLM-based claim
    decomposer (as in FActScore [3]) for higher-fidelity atomic claims."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class RAGVerificationDetector(HallucinationDetector):
    name = "rag_verification"

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        corpus: Sequence[str] = (),
        similarity_threshold: float = 0.75,
        dtype_bytes: int = 4,
    ):
        from sentence_transformers import SentenceTransformer
        import faiss

        self._faiss = faiss
        self.embedder = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        self.dtype_bytes = dtype_bytes
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        self.corpus: List[str] = list(corpus)
        self.index = self._faiss.IndexFlatIP(self.embedding_dim)
        if self.corpus:
            corpus_emb = self._normalize(self.embedder.encode(self.corpus, convert_to_numpy=True))
            self.index.add(corpus_emb)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype("float32")

    def add_to_corpus(self, documents: Sequence[str]) -> None:
        self.corpus.extend(documents)
        emb = self._normalize(self.embedder.encode(list(documents), convert_to_numpy=True))
        self.index.add(emb)

    def detect(self, prompt: str, generated_text: str, **kwargs) -> DetectionResult:
        claims = split_into_claims(generated_text)
        if not claims:
            return DetectionResult(False, 0.0, {"n_claims": 0})

        start = time.perf_counter()
        claim_emb = self._normalize(self.embedder.encode(claims, convert_to_numpy=True))
        embedding_latency_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        if self.index.ntotal:
            scores, _ = self.index.search(claim_emb, 1)
            supported_flags = [float(s[0]) >= self.similarity_threshold for s in scores]
        else:
            supported_flags = [False] * len(claims)
        retrieval_latency_ms = (time.perf_counter() - start) * 1000

        n_unsupported = sum(1 for flag in supported_flags if not flag)
        hallucination_ratio = n_unsupported / len(claims)
        is_hallucination = hallucination_ratio > 0.5

        vstore_mb = vector_store_memory_mb(self.index.ntotal, self.embedding_dim, self.dtype_bytes)
        total_latency_s = (embedding_latency_ms + retrieval_latency_ms) / 1000
        # Claims/sec: a throughput proxy comparable across paradigms since RAG
        # has no token-generation step to measure tokens/sec against.
        throughput_cps = len(claims) / total_latency_s if total_latency_s > 0 else 0.0

        return DetectionResult(
            is_hallucination=is_hallucination,
            score=hallucination_ratio,
            metrics=dict(
                n_claims=len(claims),
                n_unsupported=n_unsupported,
                embedding_latency_ms=embedding_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                vector_store_mb=vstore_mb,
                index_size=self.index.ntotal,
                throughput_tps=throughput_cps,
            ),
        )
