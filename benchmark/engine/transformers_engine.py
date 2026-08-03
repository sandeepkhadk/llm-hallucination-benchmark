"""
HuggingFace `transformers`-based inference backend. Slower than vLLM and
without PagedAttention KV-cache introspection, but works on any machine
(CPU or GPU) without vLLM's Linux/CUDA requirement. Use `--backend vllm` for
methodology-faithful measurements.
"""
from __future__ import annotations

import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.inference_engine import GenerationResult, InferenceEngine


class TransformersEngine(InferenceEngine):
    def __init__(self, model_name: str, device: str = None, dtype=torch.float16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 256, n: int = 1):
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            per_prompt = []
            for _ in range(n):
                start = time.perf_counter()
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                elapsed_s = time.perf_counter() - start

                n_prompt_tokens = inputs["input_ids"].shape[-1]
                n_out_tokens = output.shape[-1] - n_prompt_tokens
                text = self.tokenizer.decode(output[0][n_prompt_tokens:], skip_special_tokens=True)

                # Approximate TTFT as latency / output tokens (no native
                # streaming timestamp in the offline `generate` call).
                ttft_ms = (elapsed_s * 1000) / max(n_out_tokens, 1)
                throughput = n_out_tokens / elapsed_s if elapsed_s > 0 else 0.0

                per_prompt.append(
                    GenerationResult(
                        prompt=prompt,
                        text=text,
                        ttft_ms=ttft_ms,
                        latency_ms=elapsed_s * 1000,
                        throughput_tps=throughput,
                        prompt_tokens=n_prompt_tokens,
                        output_tokens=n_out_tokens,
                    )
                )
            results.append(per_prompt)
        return results
