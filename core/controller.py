import os
import json
from dotenv import load_dotenv
import litellm
from core.rate_limiter import rate_limiter
from core import SessionState, StabilityDecision

load_dotenv()

class IntentController:
    """Analyzes incoming transcript tokens and decides the retrieval action."""
    
    MIN_CONTENT_TOKENS = int(os.environ.get("MIN_CONTENT_TOKENS", 5))
    STABILITY_LLM_THRESHOLD = float(os.environ.get("STABILITY_LLM_THRESHOLD", 0.6))
    
    FILLER_WORDS = {'um', 'uh', 'like', 'so', 'well', 'you', 'know', 'hmm', 'err', 'ah'}
    DANGLING_CONJUNCTIONS = {'and', 'but', 'or', 'because', 'however', 'also', 'then', 'that'}
    INCOMPLETE_ENDINGS = {
        'the', 'a', 'an', 'this', 'these', 'those',
        'about', 'is', 'are', 'was', 'were', 'does', 'do', 'has', 'have', 'had',
        'of', 'for', 'to', 'in', 'on', 'at', 'with', 'by', 'from', 'as', 'into', 'like'
    }
    NAVIGATIONAL_PHRASES = [
        'next slide', 'go back', 'previous slide', 'repeat that', 'show again',
        'move on', 'skip this', 'go to', 'open the', 'close the', 'scroll',
        'zoom in', 'zoom out', 'go forward', 'last slide', 'first slide'
    ]
    QUESTION_MARKERS = {
        'what', 'who', 'where', 'when', 'why', 'how', 'which', 'is', 'are', 
        'does', 'do', 'can', 'could', 'would', 'explain', 'describe', 'compare', 
        'list', 'tell'
    }
    
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
        
    async def assess(self, transcript: str, session_state: SessionState) -> StabilityDecision:
        norm_transcript = transcript.strip().lower()
        
        import re
        for phrase in self.NAVIGATIONAL_PHRASES:
            if re.search(r'\b' + re.escape(phrase) + r'\b', norm_transcript):
                return StabilityDecision(action='SUPPRESS', confidence=0.95, reasoning='navigational command detected')
                
        tokens = [t for t in transcript.split() if t]
        if not tokens:
            return StabilityDecision(action='WAIT', confidence=0.9, reasoning='empty transcript')
            
        content_tokens = [t for t in tokens if t.lower() not in self.FILLER_WORDS]
        
        last_word = "".join(c for c in tokens[-1].lower() if c.isalnum())
        
        if tokens[-1].lower() in self.FILLER_WORDS:
            return StabilityDecision(action='WAIT', confidence=0.85, reasoning='trailing filler word')
            
        if last_word in self.DANGLING_CONJUNCTIONS:
            return StabilityDecision(action='WAIT', confidence=0.85, reasoning='trailing conjunction')
            
        if last_word in self.INCOMPLETE_ENDINGS:
            return StabilityDecision(action='WAIT', confidence=0.85, reasoning='trailing incomplete word')
            
        if transcript.endswith(',') or transcript.endswith('...'):
            return StabilityDecision(action='WAIT', confidence=0.8, reasoning='incomplete phrase')
            
        has_question_marker = any(t.lower() in self.QUESTION_MARKERS for t in content_tokens)
        is_concise_question = has_question_marker and len(content_tokens) >= 2
        
        if len(content_tokens) < self.MIN_CONTENT_TOKENS and not is_concise_question:
            return StabilityDecision(action='WAIT', confidence=0.9, reasoning='too few content tokens')
            
        confidence = 0.85 if has_question_marker else 0.55
        action = 'RETRIEVE'
        
        if confidence < self.STABILITY_LLM_THRESHOLD:
            return await self._llm_stability_check(transcript)
            
        return StabilityDecision(action=action, confidence=confidence, reasoning='rule-based assessment passed')
        
    async def _llm_stability_check(self, transcript: str) -> StabilityDecision:
        try:
            await rate_limiter.acquire()
            response = await litellm.acompletion(
                model=self._model,
                api_key=self._api_key, api_base=self._api_base,
                messages=[
                    {"role": "system", "content": 'You classify speech fragments. Respond with only JSON: {"action": "WAIT|RETRIEVE|SUPPRESS", "confidence": 0.0-1.0, "reasoning": "brief reason"}'},
                    {"role": "user", "content": f'Classify this speech fragment: "{transcript}"'}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content
            if "{" in content and "}" in content:
                content = content[content.find("{"):content.rfind("}")+1]
            data = json.loads(content)
            return StabilityDecision(
                action=data.get('action', 'WAIT'),
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', 'parsed from llm')
            )
        except Exception as e:
            return StabilityDecision(action='WAIT', confidence=0.5, reasoning='llm check failed')
