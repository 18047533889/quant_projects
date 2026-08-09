def test_gate_probe():
    from planner.logical_plan import PlanNode
    from runtime.engine import FactorExecutionScope, assert_execution_scope_contract, _scope_from_factor
    from runtime.production_policy import ProductionPolicyViolation
    rank_plan = PlanNode(op='rank', inputs=(PlanNode(op='column', attrs={'name':'close'}),))
    try:
        assert_execution_scope_contract(FactorExecutionScope(universe_id='CSI300', market='A'), rank_plan, factor_name='x')
        assert False, 'should raise'
    except ProductionPolicyViolation:
        pass
    assert_execution_scope_contract(FactorExecutionScope(universe_id='ALL'), rank_plan, factor_name='x')
    assert True
