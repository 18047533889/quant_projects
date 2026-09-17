"""Legacy panel keywords must use the same contract as positional inputs."""
import pytest
from factor_engine.api.dsl_parser import DSLParser
from factor_engine.ir.analyzer import Analyzer

@pytest.mark.parametrize("op", [
    "ts_markov_persistence", "ts_markov_state_entropy",
    "ts_markov_transition_surprisal", "ts_active_information_storage",
    "ts_kramers_moyal_local_stability", "ts_km_equilibrium_distance",
])
def test_legacy_panel_keyword_matches_positional(op):
    parser = DSLParser(surface="compat_research")
    positional = Analyzer().lower(parser.parse(f"{op}(ret, window=120)"))
    keyword = Analyzer().lower(parser.parse(f"{op}(x=ret, window=120)"))
    assert positional == keyword

def test_panel_value_in_scalar_keyword_still_rejected():
    parser = DSLParser(surface="compat_research")
    with pytest.raises((NotImplementedError, TypeError, ValueError)):
        Analyzer().lower(parser.parse("ts_markov_persistence(x=ret, window=close)"))

def test_duplicate_panel_keyword_still_rejected():
    parser = DSLParser(surface="compat_research")
    with pytest.raises((NotImplementedError, TypeError, ValueError)):
        Analyzer().lower(parser.parse("ts_markov_persistence(ret, x=ret, window=120)"))
