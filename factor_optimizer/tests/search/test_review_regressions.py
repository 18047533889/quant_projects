from factor_optimizer.search.pareto import ParetoArchive, ParetoFrontier, ParetoPoint


def test_domination_query_uses_current_objectives_not_trial_id():
    frontier = ParetoFrontier()
    frontier.add(ParetoPoint('leader', (2.0, 2.0)))
    assert frontier.is_dominated(ParetoPoint('retry', (1.0, 1.0)))
    assert not frontier.is_dominated(ParetoPoint('retry', (3.0, 3.0)))


def test_snapshot_does_not_share_mutable_points_or_metadata():
    archive = ParetoArchive(['score', 'robustness'])
    point = ParetoPoint('trial', (1.0, 2.0), {'evidence': {'ref': 'original'}})
    archive.add_point(point)
    archive.snapshot(10)
    point.objectives = (9.0, 9.0)
    point.metadata['evidence']['ref'] = 'changed'
    archived = archive.get_frontier(10).get_point('trial')
    assert archived.objectives == (1.0, 2.0)
    assert archived.metadata['evidence']['ref'] == 'original'
