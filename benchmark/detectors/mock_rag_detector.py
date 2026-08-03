"""
Dependency-free RAG-verification stand-in for `--mock` runs: replaces the
BGE embedding model + FAISS ANN search with a pure-Python bag-of-words
Jaccard retriever, so the pipeline can be sanity-checked without downloading
any models. Use `RAGVerificationDetector` for methodology-faithful results.
"""
from __future__ import annotations

import time
from typing import Sequence

from detectors.base import DetectionResult, HallucinationDetector
from detectors.rag_verification_detector import split_into_claims
from engine.memory_profiler import vector_store_memory_mb


class MockRAGVerificationDetector(HallucinationDetector):
    name = "rag_verification"

    def __init__(
        self,
        corpus: Sequence[str] = (),
        similarity_threshold: float = 0.3,
        embedding_dim: int = 1024,  # nominal dim matching BGE-large-en-v1.5, for realistic memory-footprint estimates
        dtype_bytes: int = 4,
    ):
        self.corpus = list(corpus)
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.dtype_bytes = dtype_bytes

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        a_tokens, b_tokens = set(a.lower().split()), set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    def detect(self, prompt: str, generated_text: str, **kwargs) -> DetectionResult:
        claims = split_into_claims(generated_text)
        if not claims:
            return DetectionResult(False, 0.0, {"n_claims": 0})

        start = time.perf_counter()
        supported_flags = []
        for claim in claims:
            best = max((self._jaccard(claim, doc) for doc in self.corpus), default=0.0)
            supported_flags.append(best >= self.similarity_threshold)
        retrieval_latency_ms = (time.perf_counter() - start) * 1000

        n_unsupported = sum(1 for f in supported_flags if not f)
        hallucination_ratio = n_unsupported / len(claims)
        vstore_mb = vector_store_memory_mb(len(self.corpus), self.embedding_dim, self.dtype_bytes)
        total_latency_s = retrieval_latency_ms / 1000
        throughput_cps = len(claims) / total_latency_s if total_latency_s > 0 else 0.0

        return DetectionResult(
            is_hallucination=hallucination_ratio > 0.5,
            score=hallucination_ratio,
            metrics=dict(
                n_claims=len(claims),
                n_unsupported=n_unsupported,
                embedding_latency_ms=0.0,
                retrieval_latency_ms=retrieval_latency_ms,
                vector_store_mb=vstore_mb,
                index_size=len(self.corpus),
                throughput_tps=throughput_cps,
            ),
        )
