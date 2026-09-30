import pytest, json, re
from pathlib import Path

def test_corpus_domains():
    p = Path('data/corpus.json')
    if not p.exists(): pytest.skip('Run generate_mock_data.py first')
    with open(p) as f: corpus = json.load(f)
    domains = {d['metadata']['domain'] for d in corpus}
    assert {'AI', 'Climate', 'Space', 'Medicine'}.issubset(domains)
    assert len(corpus) >= 40

def test_transcript_scenarios():
    p = Path('data/transcripts.json')
    if not p.exists(): pytest.skip('Run generate_mock_data.py first')
    with open(p) as f: scenarios = json.load(f)
    actions = {s['expected_action'] for s in scenarios}
    assert 'RETRIEVE' in actions
    assert 'WAIT' in actions
    assert 'SUPPRESS' in actions
    assert len(scenarios) >= 6

def test_citation_regex():
    pattern = r'\[([\w_]+) §([^\]]+)\]'
    texts = [
        ('Transformers [ai_001 §Introduction].', 1),
        ('Two citations [cl_003 §Findings] and [cl_001 §Overview].', 2),
        ('No citation here.', 0),
    ]
    for text, expected in texts:
        assert len(re.findall(pattern, text)) == expected


def test_session_ephemeral():
    from core.state import StateManager
    sm = StateManager()
    sm.get_or_create('eph_test')
    sm.append_token('eph_test', 'data')
    sm.destroy('eph_test')
    assert 'eph_test' not in sm._sessions
    fresh = sm.get_or_create('eph_test')
    assert fresh.tokens == []

def test_chunk_citation_key_format():
    from core import RetrievedChunk
    chunk = RetrievedChunk(doc_id='sp_005', section='Exoplanets', text='', score=0.9, rank=1)
    assert chunk.citation_key == '[sp_005 §Exoplanets]'
