from core.state import augment_query_with_constraints
from core.synthesizer import AnswerSynthesizer
from core import RetrievedChunk

def test_query_augmentation():
    original = "What is retrieval augmented generation? Only information after 2020."
    constraint = {"year_min": 2020}
    
    result = augment_query_with_constraints(original, constraint)
    # The new logic should return exactly the original since "2020" is in it
    # We strip trailing spaces so we'll compare stripped
    assert result.strip() == original.strip()
    
    # Check another
    original_2 = "Tell me about RAG."
    res2 = augment_query_with_constraints(original_2, constraint)
    assert res2 == "Tell me about RAG. after 2020"

def test_context_block_metadata():
    synth = AnswerSynthesizer()
    chunk = RetrievedChunk(doc_id="ai_003", section="RAG Overview", text="Retrieval-Augmented Generation...", score=1.0, rank=1, metadata={"year": 2020, "domain": "AI"})
    
    block = synth._build_context_block([chunk])
    assert "Year: 2020" in block
    assert "Domain: AI" in block
    assert "Source: [ai_003 §RAG Overview]" in block
    assert "Text: Retrieval-Augmented Generation" in block
