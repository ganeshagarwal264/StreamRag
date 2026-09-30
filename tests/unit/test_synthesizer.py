import pytest
import asyncio
from unittest.mock import patch, MagicMock
from core.synthesizer import AnswerSynthesizer
from core import RetrievedChunk, SynthesisResult

class AsyncIteratorMock:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

class MockChunk:
    def __init__(self, content, reasoning=None):
        self.choices = [MagicMock()]
        self.choices[0].delta = MagicMock()
        self.choices[0].delta.content = content
        self.choices[0].delta.reasoning_content = reasoning
        self.usage = None

def mock_acompletion(chunks):
    async def _mock(*args, **kwargs):
        return AsyncIteratorMock([MockChunk(c) for c in chunks])
    return _mock

def mock_acompletion_with_reasoning(content_chunks, reasoning_chunks):
    async def _mock(*args, **kwargs):
        items = [MockChunk(None, r) for r in reasoning_chunks] + [MockChunk(c, None) for c in content_chunks]
        return AsyncIteratorMock(items)
    return _mock


@pytest.mark.asyncio
async def test_normal_single_intent_synthesis():
    synth = AnswerSynthesizer()
    chunks = [RetrievedChunk("doc1", "sec1", "Text", 1.0, 1)]
    
    tokens = []
    citations = []
    done_result = None
    
    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): citations.append((d, s))
    async def on_done(res, cost): nonlocal done_result; done_result = res
    
    with patch("litellm.acompletion", new=mock_acompletion(["Hello ", "World [doc1 \u00a7sec1]"])):
        await synth.stream_answer("query", chunks, on_token, on_citation, on_done)
        
    assert "".join(tokens) == "Hello World [doc1 \u00a7sec1]"
    assert citations == [("doc1", "sec1")]
    assert not done_result.is_insufficient
    assert not done_result.clarifying_question

@pytest.mark.asyncio
async def test_no_reasoning_content_leakage():
    synth = AnswerSynthesizer()
    chunks = [RetrievedChunk("doc1", "sec1", "Text", 1.0, 1)]
    
    tokens = []
    citations = []
    done_result = None
    
    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): citations.append((d, s))
    async def on_done(res, cost): nonlocal done_result; done_result = res
    
    with patch("litellm.acompletion", new=mock_acompletion_with_reasoning(["Final Answer"], ["Thinking...", "Wait!"])):
        await synth.stream_answer("query", chunks, on_token, on_citation, on_done)
        
    assert "".join(tokens) == "Final Answer"
    assert "Thinking..." not in "".join(tokens)

@pytest.mark.asyncio
async def test_duplicate_citation_deduplication():
    synth = AnswerSynthesizer()
    chunks = [RetrievedChunk("doc1", "sec1", "Text", 1.0, 1)]
    
    tokens = []
    citations = []
    done_result = None
    
    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): citations.append((d, s))
    async def on_done(res, cost): nonlocal done_result; done_result = res
    
    with patch("litellm.acompletion", new=mock_acompletion(["Part1 [doc1 \u00a7sec1] ", "Part2 [doc1 \u00a7sec1]"])):
        await synth.stream_answer("query", chunks, on_token, on_citation, on_done)
        
    # We should only get ONE citation call
    assert citations == [("doc1", "sec1")]
    assert len(done_result.citations) == 1

@pytest.mark.asyncio
async def test_correct_insufficient_evidence():
    synth = AnswerSynthesizer()
    chunks = []
    
    tokens = []
    citations = []
    done_result = None
    
    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): citations.append((d, s))
    async def on_done(res, cost): nonlocal done_result; done_result = res
    
    with patch("litellm.acompletion", new=mock_acompletion(["INSUFFICIENT_EVIDENCE\n", "What did you mean?"])):
        await synth.stream_answer("query", chunks, on_token, on_citation, on_done)
        
    assert done_result.is_insufficient
    assert done_result.clarifying_question == "What did you mean?"

@pytest.mark.asyncio
async def test_valid_answer_evidence_word():
    synth = AnswerSynthesizer()
    chunks = []
    
    tokens = []
    citations = []
    done_result = None
    
    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): citations.append((d, s))
    async def on_done(res, cost): nonlocal done_result; done_result = res
    
    with patch("litellm.acompletion", new=mock_acompletion(["The evidence shows that...", " constraint is met."])):
        await synth.stream_answer("query", chunks, on_token, on_citation, on_done)
        
    assert not done_result.is_insufficient

from core import SubQuery

@pytest.mark.asyncio
async def test_multi_intent_partial_evidence():
    synth = AnswerSynthesizer()
    
    sq0 = SubQuery(id="sq_0", text="What is RAG?", intent_type="factual")
    sq1 = SubQuery(id="sq_1", text="What is Climate Change?", intent_type="factual")

from core import SubQuery

@pytest.mark.asyncio
async def test_multi_intent_partial_evidence():
    synth = AnswerSynthesizer()
    
    sq0 = SubQuery(id="sq_0", text="What is RAG?", intent_type="factual")
    sq1 = SubQuery(id="sq_1", text="What is Climate Change?", intent_type="factual")
    
    chunks_by_query = {
        "sq_0": [RetrievedChunk("ai_003", "RAG Overview", "Text", 1.0, 1)],
        "sq_1": []
    }
    
    tokens = []
    citations = []
    done_result = None
    
    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): citations.append((d, s))
    async def on_done(res, cost): nonlocal done_result; done_result = res
    
    with patch("litellm.acompletion", new=mock_acompletion(["RAG is cool. [ai_003 §RAG Overview]"])) as mock_llm:
        await synth.synthesize_merged([sq0, sq1], chunks_by_query, on_token, on_citation, on_done)
        
    full_output = "".join(tokens)
    
    assert not done_result.is_insufficient
    assert citations == [("ai_003", "RAG Overview")]
    assert "RAG is cool. [ai_003 §RAG Overview]" in done_result.answer
    assert "Note: The supplied corpus does not contain sufficient evidence to answer: \"What is Climate Change?\"" in done_result.answer
    assert full_output.endswith("Note: The supplied corpus does not contain sufficient evidence to answer: \"What is Climate Change?\"")


@pytest.mark.asyncio
async def test_multi_intent_no_evidence():
    synth = AnswerSynthesizer()
    
    sq0 = SubQuery(id="sq_0", text="Q1", intent_type="factual")
    sq1 = SubQuery(id="sq_1", text="Q2", intent_type="factual")
    
    chunks_by_query = { "sq_0": [], "sq_1": [] }
    
    done_result = None
    async def on_done(res, cost): nonlocal done_result; done_result = res
    async def on_token(t): pass
    async def on_citation(d, s): pass
    
    with patch("litellm.acompletion", new=mock_acompletion(["INSUFFICIENT_EVIDENCE\n", "Can you clarify?"])):
        await synth.synthesize_merged([sq0, sq1], chunks_by_query, on_token, on_citation, on_done)
        
    assert done_result.is_insufficient

@pytest.mark.asyncio
async def test_multi_intent_all_evidence():
    synth = AnswerSynthesizer()
    
    sq0 = SubQuery(id="sq_0", text="Q1", intent_type="factual")
    sq1 = SubQuery(id="sq_1", text="Q2", intent_type="factual")
    
    chunks_by_query = {
        "sq_0": [RetrievedChunk("doc1", "sec1", "Text", 1.0, 1)],
        "sq_1": [RetrievedChunk("doc2", "sec2", "Text", 1.0, 1)]
    }
    
    done_result = None
    async def on_done(res, cost): nonlocal done_result; done_result = res
    async def on_token(t): pass
    async def on_citation(d, s): pass
    
    with patch("litellm.acompletion", new=mock_acompletion(["Both answered."])):
        await synth.synthesize_merged([sq0, sq1], chunks_by_query, on_token, on_citation, on_done)
        
    assert not done_result.is_insufficient
    assert "Both answered." in done_result.answer
    assert "Note: The supplied corpus" not in done_result.answer
