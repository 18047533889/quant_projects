import random
import types
import pytest
from factor_engine.planner.read_wave_planner import ReadWavePlanner
from factor_engine.planner.projected_column_footprint import ProjectedColumnFootprint


def legacy_best(self, candidates, current_cols, current_reqs):
    best_idx=0; best_score=float('-inf'); cm=self.cost_model
    for idx,req in enumerate(candidates):
        union_mem=self._wave_memory_bytes(current_cols|req.columns,current_reqs+[req])
        current_mem=self._wave_memory_bytes(current_cols,current_reqs) if current_reqs else 0
        incremental=max(0,union_mem-current_mem)
        cost=incremental+self._memory_rent(incremental)+cm.scheduling_delay_bytes
        benefit=self._pack_benefit(len(req.columns&current_cols),req) if current_cols else self._pack_benefit(0,req)+len(req.columns)*cm.scan_per_column_bytes
        score=benefit/max(cost,cm.epsilon)
        if score>best_score: best_idx=idx; best_score=score
    return best_idx


def populate(planner, seed):
    rng=random.Random(seed)
    for i in range(24):
        columns=rng.sample(['a','b','c','d','e'],rng.randrange(1,5))
        footprints={c:ProjectedColumnFootprint(c,decoded_bytes=rng.randrange(10,500),null_bitmap_bytes=5)
                    for c in columns if rng.random()<.7}
        planner.register_scan_task(str(i),dataset='d',source_scope='s',snapshot_id='v',time_range=None,
            columns=columns,column_footprints=footprints,axis_bytes=rng.randrange(100))


@pytest.mark.parametrize('seed',range(8))
def test_marginal_selection_and_budgeted_plan_match_legacy(seed):
    options=dict(wave_memory_budget=1800,rows_estimate=12,axis_bytes=20,metadata_bytes=30,
                 downstream_live_reserve=40,output_reserve=50)
    fast=ReadWavePlanner(**options); slow=ReadWavePlanner(**options)
    populate(fast,seed); populate(slow,seed)
    # Walk every greedy position, including conflicting first-wins footprints,
    # absent footprints, axis maxima, shared columns and ties.
    remaining=list(fast._requests); current=[]; cols=set()
    while remaining:
        chosen=fast._best_marginal(remaining,cols,current)
        assert chosen==legacy_best(fast,remaining,cols,current)
        req=remaining.pop(chosen); current.append(req); cols|=req.columns
    slow._best_marginal=types.MethodType(legacy_best,slow)
    assert fast.plan()==slow.plan()


def test_current_footprints_scanned_once_per_selection(monkeypatch):
    planner=ReadWavePlanner(rows_estimate=12); populate(planner,0)
    original=planner._union_footprints; calls=[]
    def counted(reqs): calls.append(len(reqs)); return original(reqs)
    monkeypatch.setattr(planner,'_union_footprints',counted)
    planner._best_marginal(planner._requests[8:],{'a','b'},planner._requests[:8])
    assert calls==[8]
