from core.state import apply_constraints, StateManager
from core import RetrievedChunk
import pytest

def test_apply_constraints_inclusive():
    c2019 = RetrievedChunk(doc_id="1", section="s", text="t", score=1, rank=1, metadata={"year": 2019})
    c2020 = RetrievedChunk(doc_id="2", section="s", text="t", score=1, rank=2, metadata={"year": 2020})
    c2021 = RetrievedChunk(doc_id="3", section="s", text="t", score=1, rank=3, metadata={"year": 2021})
    
    constraints = {"year_min": 2020}
    
    filtered = apply_constraints([c2019, c2020, c2021], constraints)
    ids = [c.doc_id for c in filtered]
    
    assert "2" in ids  # 2020 kept
    assert "3" in ids  # 2021 kept
    assert "1" not in ids # 2019 removed

@pytest.mark.asyncio
async def test_refinement_suffix_wording():
    sm = StateManager()
    sm.get_or_create("ss1")
    
    # year_min
    delta_min = await sm.detect_late_constraints("ss1", ["after", "2020"])
    assert "from 2020 onward" in delta_min.refinement_prompt_suffix
    assert "after 2020" not in delta_min.refinement_prompt_suffix
    
    # year_max
    sm.get_or_create("ss2")
    delta_max = await sm.detect_late_constraints("ss2", ["before", "2020"])
    assert "up to 2020" in delta_max.refinement_prompt_suffix
    
    # year_exact
    sm.get_or_create("ss3")
    delta_exact = await sm.detect_late_constraints("ss3", ["in", "2020"])
    assert "from 2020" in delta_exact.refinement_prompt_suffix
    assert "onward" not in delta_exact.refinement_prompt_suffix
    
    # range
    sm.get_or_create("ss4")
    delta_range = await sm.detect_late_constraints("ss4", ["after", "2020", "before", "2023"])
    assert "from 2020 through 2023" in delta_range.refinement_prompt_suffix
