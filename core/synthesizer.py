import os
import re
import time
import asyncio as _asyncio
from typing import Callable, Optional
from core import SubQuery, RetrievedChunk, SynthesisResult

_SYNTHESIS_TIMEOUT_S  = float(os.environ.get("SYNTHESIS_TIMEOUT_S",  "45"))
_SYNTHESIS_MAX_RETRIES = int(os.environ.get("SYNTHESIS_MAX_RETRIES", "1"))

# Concurrency limit for LLM calls (e.g. contradiction checks)
rate_limiter = _asyncio.Semaphore(2)

class AnswerSynthesizer:
    
    INSUFFICIENT_EVIDENCE_SENTINEL = "INSUFFICIENT_EVIDENCE"
    
    SYSTEM_PROMPT = """You answer using only the supplied retrieved evidence. Do not use outside knowledge. Write only the final answer for the user. Do not describe these instructions or the reasoning process. Every factual claim must have an inline citation in the format [doc_id §section]. If the evidence is insufficient, return exactly INSUFFICIENT_EVIDENCE followed by one short clarifying question.

Constraints have already been applied by the retrieval controller. Every chunk provided in CONTEXT has already passed the active constraints. Do not independently reject, re-evaluate, or debate whether a provided chunk satisfies a constraint. Treat provided chunks as eligible evidence and answer from them. Only return INSUFFICIENT_EVIDENCE when the context contains no usable evidence for the question."""

    def __init__(self):
        self._model = os.environ.get('LLM_MODEL', 'gemini/gemini-2.5-flash')
        self._api_key = (
            os.environ.get('GEMINI_API_KEY') or
            os.environ.get('OPENAI_API_KEY') or
            os.environ.get('GROQ_API_KEY')
        )
        self._api_base = None
        if self._model.startswith('nvidia_nim/'):
            self._api_key = os.environ.get('NVIDIA_NIM_API_KEY')
            self._api_base = os.environ.get('NVIDIA_NIM_BASE_URL')
            
        # Support local Ollama explicit base mapping
        if self._model.startswith('ollama/'):
            self._api_base = os.environ.get('OLLAMA_API_BASE', 'http://localhost:11434')
            
        import litellm
        litellm.drop_params = True

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks):
            year_line = f'Year: {chunk.metadata["year"]}\n' if 'year' in chunk.metadata else ''
            domain_line = f'Domain: {chunk.metadata["domain"]}\n' if 'domain' in chunk.metadata else ''
            blocks.append(
                f'=== CONTEXT CHUNK {i+1} ===\n'
                f'Source: [{chunk.doc_id} §{chunk.section}]\n'
                f'{year_line}{domain_line}'
                f'Text: {chunk.text}\n'
                f'===========================\n\n'
            )
        return ''.join(blocks)
    def _build_messages(self, query: str, chunks: list[RetrievedChunk], refinement_suffix: str = '') -> list[dict]:
        context_block = self._build_context_block(chunks)
        return [
            {'role': 'system', 'content': self.SYSTEM_PROMPT},
            {'role': 'user', 'content': f'CONTEXT:\n{context_block}\nQUESTION: {query}{refinement_suffix}'}
        ]

    async def stream_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        on_token: Callable,
        on_citation: Callable,
        on_done: Callable,
        refinement_suffix: str = ''
    ) -> None:
        messages = self._build_messages(query, chunks, refinement_suffix)
        start_time = time.time()
        
        import litellm
        
        full_text = ''
        is_insufficient = False
        clarifying_question = None
        
        try:
            last_exc = None
            for attempt in range(_SYNTHESIS_MAX_RETRIES + 1):
                try:
                    await rate_limiter.acquire()
                    response = await _asyncio.wait_for(
                        litellm.acompletion(
                            model=self._model,
                            messages=messages,
                            stream=True,
                            stream_options={"include_usage": True},
                            api_key=self._api_key, api_base=self._api_base,
                            temperature=0.2,
                            max_tokens=1024
                        ),
                        timeout=_SYNTHESIS_TIMEOUT_S,
                    )
                    last_exc = None
                    rate_limiter.release()
                    break  # success
                except (_asyncio.TimeoutError, Exception) as exc:
                    last_exc = exc
                    rate_limiter.release()
                    if attempt < _SYNTHESIS_MAX_RETRIES:
                        delay = 0.0 if attempt == 0 else 1.5 ** attempt
                        await _asyncio.sleep(delay)
            
            if last_exc is not None:
                err_msg = f"\n[Synthesis unavailable: {type(last_exc).__name__}: {str(last_exc)}. Please try again.]"
                await on_token(err_msg)
                await on_done(
                    SynthesisResult(
                        answer=err_msg, citations=[],
                        is_insufficient=True,
                        latency_ms=(time.time() - start_time) * 1000,
                    ),
                    0.0,
                )
                return
            
            ttft_ms = 0.0
            first_token_time = 0.0
            usage = None

            # --- STREAMING PHASE with remaining-budget timeout --------------------
            # The wait_for above covers obtaining the streaming response *object*.
            # Once we have the object, each __anext__ can block indefinitely if
            # Ollama's llama-server crashes or hangs mid-stream.
            # We enforce whatever budget is left from _SYNTHESIS_TIMEOUT_S.
            remaining_s = max(1.0, _SYNTHESIS_TIMEOUT_S - (time.time() - start_time))
            full_text_ref: list[str] = []

            async def _drain_stream():
                nonlocal ttft_ms, first_token_time, usage
                async for chunk in response:
                    if getattr(chunk, 'usage', None):
                        usage = chunk.usage
                    delta = chunk.choices[0].delta if chunk.choices else None
                    content = getattr(delta, 'content', None) if delta else None
                    if content:
                        if first_token_time == 0.0:
                            first_token_time = time.time()
                            ttft_ms = (first_token_time - start_time) * 1000
                        full_text_ref.append(content)
                        await on_token(content)

            try:
                await _asyncio.wait_for(_drain_stream(), timeout=remaining_s)
            except _asyncio.TimeoutError:
                elapsed_ms = (time.time() - start_time) * 1000
                err_msg = (
                    f"\n[Synthesis timed out after {elapsed_ms / 1000:.0f}s "
                    f"waiting for the local LLM. Please try again.]"
                )
                await on_token(err_msg)
                await on_done(
                    SynthesisResult(
                        answer=err_msg, citations=[],
                        is_insufficient=True,
                        latency_ms=elapsed_ms,
                        ttft_ms=ttft_ms,
                    ),
                    0.0,
                )
                return

            full_text = ''.join(full_text_ref)
            latency_ms = (time.time() - start_time) * 1000
            
            cost_usd = 0.0
            if usage:
                try:
                    cost_usd = litellm.completion_cost(completion_response={"model": self._model, "usage": dict(usage)})
                except Exception:
                    pass
            
            citations = re.findall(r'\[([\w_]+) §([^\]]+)\]', full_text)
            seen_citations = set()
            formatted_citations = []
            
            for doc_id, section in citations:
                k = (doc_id, section)
                if k not in seen_citations:
                    seen_citations.add(k)
                    await on_citation(doc_id, section)
                    formatted_citations.append(f"[{doc_id} §{section}]")
            
            if self.INSUFFICIENT_EVIDENCE_SENTINEL in full_text:
                lines = [line.strip() for line in full_text.splitlines() if line.strip()]
                for i, line in enumerate(lines):
                    if line == self.INSUFFICIENT_EVIDENCE_SENTINEL or line.startswith(self.INSUFFICIENT_EVIDENCE_SENTINEL):
                        is_insufficient = True
                        if i + 1 < len(lines):
                            clarifying_question = lines[i+1]
                        break
                        
            result = SynthesisResult(
                answer=full_text,
                citations=formatted_citations,
                is_insufficient=is_insufficient,
                clarifying_question=clarifying_question,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms
            )
            await on_done(result, cost_usd)
            
        except Exception as e:
            await on_token(f'\n[Error: {str(e)}]')
            await on_done(SynthesisResult(answer=''.join(full_text_ref) if 'full_text_ref' in dir() else full_text, citations=[], is_insufficient=True, latency_ms=(time.time() - start_time) * 1000), 0.0)

    async def synthesize_merged(
        self,
        sub_queries: list[SubQuery],
        chunks_by_query: dict[str, list[RetrievedChunk]],
        on_token: Callable,
        on_citation: Callable,
        on_done: Callable,
        refinement_suffix: str = ''
    ) -> None:
        supported = [sq for sq in sub_queries if chunks_by_query.get(sq.id)]
        unsupported = [sq for sq in sub_queries if not chunks_by_query.get(sq.id)]

        if not supported:
            # None have evidence
            merged_query = ' | '.join(sq.text for sq in sub_queries)
            await self.stream_answer(merged_query, [], on_token, on_citation, on_done, refinement_suffix)
            return

        if not unsupported:
            # All have evidence
            merged_query = ' | '.join(sq.text for sq in sub_queries)
            merged_chunks = [c for sq in sub_queries for c in chunks_by_query.get(sq.id, [])]
            
            unique_chunks = {}
            for chunk in merged_chunks:
                key = (chunk.doc_id, chunk.section)
                if key not in unique_chunks or chunk.score > unique_chunks[key].score:
                    unique_chunks[key] = chunk
                    
            dedup_chunks = list(unique_chunks.values())
            dedup_chunks.sort(key=lambda x: x.score, reverse=True)
            
            rerank_top_k = int(os.environ.get("RERANK_TOP_K", 5))
            top_chunks = dedup_chunks[:rerank_top_k]
            
            for i, chunk in enumerate(top_chunks):
                chunk.rank = i + 1
                
            await self.stream_answer(merged_query, top_chunks, on_token, on_citation, on_done, refinement_suffix)
            return

        # PARTIAL EVIDENCE
        merged_query = ' | '.join(sq.text for sq in supported)
        merged_chunks = [c for sq in supported for c in chunks_by_query.get(sq.id, [])]
        
        unique_chunks = {}
        for chunk in merged_chunks:
            key = (chunk.doc_id, chunk.section)
            if key not in unique_chunks or chunk.score > unique_chunks[key].score:
                unique_chunks[key] = chunk
                
        dedup_chunks = list(unique_chunks.values())
        dedup_chunks.sort(key=lambda x: x.score, reverse=True)
        
        rerank_top_k = int(os.environ.get("RERANK_TOP_K", 5))
        top_chunks = dedup_chunks[:rerank_top_k]
        
        for i, chunk in enumerate(top_chunks):
            chunk.rank = i + 1
            
        unsupported_texts = [sq.text for sq in unsupported]
        if len(unsupported_texts) == 1:
            missing_statement = f"\n\nNote: The supplied corpus does not contain sufficient evidence to answer: \"{unsupported_texts[0]}\""
        else:
            joined = '", "'.join(unsupported_texts)
            missing_statement = f"\n\nNote: The supplied corpus does not contain sufficient evidence to answer: \"{joined}\""

        async def intercepted_on_done(result: SynthesisResult, cost_usd: float = 0.0):
            await on_token(missing_statement)
            result.answer += missing_statement
            await on_done(result, cost_usd)

        await self.stream_answer(merged_query, top_chunks, on_token, on_citation, intercepted_on_done, refinement_suffix)

    async def _check_contradictions(
        self,
        chunks_by_query: dict[str, list[RetrievedChunk]],
        sub_queries: list[SubQuery],
    ) -> list[tuple[RetrievedChunk, RetrievedChunk]]:
        if len(sub_queries) < 2:
            return []

        tops: list[RetrievedChunk] = []
        for sq in sub_queries:
            sq_chunks = chunks_by_query.get(sq.id, [])
            if sq_chunks:
                tops.append(sq_chunks[0])

        if len(tops) < 2:
            return []

        contradicting: list[tuple] = []
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                a, b = tops[i], tops[j]
                if a.doc_id == b.doc_id:
                    continue
                prompt = (
                    "Do these two passages contradict each other on any specific fact?\n"
                    "Passage A ({src_a}): {text_a}\n"
                    "Passage B ({src_b}): {text_b}\n"
                    "Reply ONLY '1' (contradict) or '0' (consistent)."
                ).format(
                    src_a=a.doc_id, text_a=a.text[:350],
                    src_b=b.doc_id, text_b=b.text[:350],
                )
                try:
                    import litellm
                    await rate_limiter.acquire()
                    resp = await litellm.acompletion(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, api_key=self._api_key, api_base=self._api_base,
                    )
                    if "1" in resp.choices[0].message.content.strip():
                        contradicting.append((a, b))
                except Exception:
                    pass
                finally:
                    rate_limiter.release()
        return contradicting
