import os
import json
import litellm
from core.rate_limiter import rate_limiter
from dotenv import load_dotenv
from core import SubQuery

load_dotenv()
litellm.drop_params = True

import asyncio as _asyncio

_DECOMP_TIMEOUT_S   = float(os.environ.get("DECOMP_TIMEOUT_S",   "6"))
_DECOMP_MAX_RETRIES = int(os.environ.get("DECOMP_MAX_RETRIES",   "1"))

class IntentDecomposer:
    """Extracts 1-N atomic sub-queries from a compound utterance."""
    
    SYSTEM_PROMPT: str = '''
You are a query decomposition specialist. Your task is to analyze a user utterance and break it into atomic, self-contained sub-queries suitable for vector search retrieval.

Rules:
- Each sub-query must be fully self-contained (no pronouns referring to other sub-queries)
- Use intent_type: 'factual' | 'comparative' | 'procedural' | 'clarification'
- Set is_dependent=true only if a sub-query logically requires the answer to a prior one
- If the utterance contains multiple distinct questions or intents, return multiple separate sub-query objects in the array, one for each distinct intent. Do not combine distinct questions into one sub-query.
- If the utterance is already a single atomic question, return exactly one sub-query
- Do NOT answer the questions — only decompose them
- Return ONLY valid JSON with this exact structure: {"sub_queries": [{"id": "sq_0", "text": "...", "intent_type": "factual", "is_dependent": false}]}
'''

    def __init__(self):
        self._model = os.environ.get("LLM_MODEL_LITE", "gemini/gemini-2.5-flash-lite")
        self._api_key = (
            os.environ.get('GEMINI_API_KEY') or
            os.environ.get('OPENAI_API_KEY') or
            os.environ.get('GROQ_API_KEY')
        )
        self._api_base = None
        if self._model.startswith('nvidia_nim/'):
            self._api_key = os.environ.get('NVIDIA_NIM_API_KEY')
            self._api_base = os.environ.get('NVIDIA_NIM_BASE_URL')



    def deterministic_split(self, query: str) -> list[SubQuery]:
        """Deterministically splits obvious compound queries like '... and ...'."""
        import re
        query_clean = query.strip()
        
        # We only want to split obvious compound queries.
        # Match common compound phrases.
        compound_pattern = re.compile(r'(?i),?\s+and\s+(what|why|how|where|when|who|explain|also)\b')
        match = compound_pattern.search(query_clean)
        
        if not match:
            return [SubQuery(id='sq_0', text=query_clean, intent_type='factual', is_dependent=False)]
            
        # Split at the matched conjunction
        split_idx = match.start()
        p1 = query_clean[:split_idx].strip()
        # Keep the matched interrogative word but remove 'and'
        p2 = query_clean[match.end(1) - len(match.group(1)):].strip()
        
        if len(p1.split()) < 3 or len(p2.split()) < 2:
            return [SubQuery(id='sq_0', text=query_clean, intent_type='factual', is_dependent=False)]
            
        # Explicit exception for pros/cons, advantages/disadvantages, causes/effects
        lower_q = query_clean.lower()
        if "pros and cons" in lower_q or "advantages and disadvantages" in lower_q or "causes and effects" in lower_q or "difference between" in lower_q:
            return [SubQuery(id='sq_0', text=query_clean, intent_type='factual', is_dependent=False)]
            
        # Also check if both sides have question content
        q_words = {"what", "when", "where", "who", "how", "why", "which", "is", "are", "can", "could", "do", "does", "explain", "compare", "tell", "describe"}
        p1_words = set(re.findall(r'\b\w+\b', p1.lower()))
        p2_words = set(re.findall(r'\b\w+\b', p2.lower()))
        
        if not (p1_words.intersection(q_words) and p2_words.intersection(q_words)):
            return [SubQuery(id='sq_0', text=query_clean, intent_type='factual', is_dependent=False)]
            
        if query_clean.endswith('?') and not p1.endswith('?'):
            p1 += '?'
        if query_clean.endswith('?') and not p2.endswith('?'):
            p2 += '?'
            
        p1 = p1[0].upper() + p1[1:] if p1 else p1
        p2 = p2[0].upper() + p2[1:] if p2 else p2
        
        return [
            SubQuery(id='sq_0', text=p1, intent_type='factual', is_dependent=False),
            SubQuery(id='sq_1', text=p2, intent_type='factual', is_dependent=False)
        ]

    async def decompose(self, query: str) -> list[SubQuery]:
        from telemetry.logger import logger
        # Robust single-intent check
        import re
        norm_query = query.lower().strip()
        
        # Look for compound phrase indicators
        # If it doesn't have "and", "also", "plus", it's single intent.
        compound_pattern = re.compile(r'\b(?:and|also|plus|moreover|furthermore|along with)\b')
        
        if not compound_pattern.search(norm_query):
            logger.stats['decomposer_skipped'] += 1
            return [SubQuery(id='sq_0', text=query, intent_type='factual', is_dependent=False)]
            
        logger.stats['decomposer_llm_call'] += 1
        try:
            last_exc = None
            response = None

            for attempt in range(_DECOMP_MAX_RETRIES + 1):
                try:
                    await rate_limiter.acquire()
                    response = await _asyncio.wait_for(
                        litellm.acompletion(
                            model=self._model,
                            api_key=self._api_key, api_base=self._api_base,
                            messages=[
                                {"role": "system", "content": self.SYSTEM_PROMPT},
                                {"role": "user", "content": f'Decompose this utterance into sub-queries:\n\n"{query}"'}
                            ],
                            temperature=0.1
                        ),
                        timeout=_DECOMP_TIMEOUT_S,
                    )
                    last_exc = None
                    break
                except (_asyncio.TimeoutError, Exception) as exc:
                    last_exc = exc
                    if attempt < _DECOMP_MAX_RETRIES:
                        delay = 0.0 if attempt == 0 else 1.5 ** attempt
                        await _asyncio.sleep(delay)

            if last_exc is not None or response is None:
                import logging
                logging.warning(f"Decomposer LLM call failed after retries: {type(last_exc).__name__}: {last_exc}. Triggering deterministic fallback.")
                fallback_sqs = self.deterministic_split(query)
                if len(fallback_sqs) > 1:
                    logging.info("Decomposer: Applied deterministic split during API failure.")
                    return fallback_sqs
                return [SubQuery(id="sq_0", text=query, intent_type="factual", is_dependent=False)]
            
            content = response.choices[0].message.content
            
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Find the first { or [ to handle markdown wrapping for both objects and arrays
                start = -1
                for i, c in enumerate(content):
                    if c in "{[":
                        start = i
                        break
                
                # Find the last } or ]
                end = -1
                for i in range(len(content)-1, -1, -1):
                    if content[i] in "}]":
                        end = i
                        break
                
                if start != -1 and end != -1 and start < end:
                    content = content[start:end+1]
                    data = json.loads(content)
                else:
                    raise ValueError("No JSON found")
                    
            sub_queries = []
            
            # Handle both normal {"sub_queries": [...]} and raw array [...] formats safely
            raw_list = data if isinstance(data, list) else data.get("sub_queries", [])
            
            for sq in raw_list:
                sub_queries.append(SubQuery(
                    id=sq.get("id", "sq_X"),
                    text=sq.get("text", ""),
                    intent_type=sq.get("intent_type", "factual"),
                    is_dependent=sq.get("is_dependent", False)
                ))
            if not sub_queries:
                raise ValueError("Empty list")
                
            if len(sub_queries) == 1:
                sq_text = sub_queries[0].text.strip()
                import string
                def normalize(s):
                    return s.lower().translate(str.maketrans('', '', string.punctuation)).replace(" ", "")
                if normalize(sq_text) == normalize(query):
                    fallback_sqs = self.deterministic_split(query)
                    if len(fallback_sqs) > 1:
                        import logging
                        logging.info("Decomposer: LLM parroted compound query. Applied deterministic split.")
                        return fallback_sqs

            return sub_queries
        except Exception as e:
            import logging
            truncated_content = content[:200] + "..." if 'content' in locals() and isinstance(content, str) else "N/A"
            logging.warning(f"Decomposition fallback triggered. Exception: {type(e).__name__}: {e}. Raw response: {truncated_content}")
            fallback_sqs = self.deterministic_split(query)
            if len(fallback_sqs) > 1:
                logging.info("Decomposer: Applied deterministic split during exception fallback.")
                return fallback_sqs
            return [SubQuery(id='sq_0', text=query, intent_type='factual', is_dependent=False)]
