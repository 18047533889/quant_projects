import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore
from factor_optimizer.search.supervised_parameter import FrozenSupervisedParameter


def state():
    return FrozenSupervisedParameter("1", "U_SHAPE_REPAIR", "center", 0.5,
        (0.4, 0.5), "train", "evidence", "objective")


@pytest.mark.parametrize("invalid", [1, True])
@pytest.mark.parametrize("operation", ["freeze", "read_campaign", "read_parent"])
def test_supervised_store_rejects_numeric_aliases(tmp_path, invalid, operation):
    path = tmp_path / "frozen.sqlite3"
    store = SQLiteCampaignStore(path)
    frozen = state()
    store.freeze_supervised_parameter("1", frozen)
    with pytest.raises((ValueError, TypeError)):
        if operation == "freeze":
            store.freeze_supervised_parameter(invalid, frozen)
        else:
            store.supervised_parameter(
                invalid if operation == "read_campaign" else "1",
                invalid if operation == "read_parent" else "1", "U_SHAPE_REPAIR")
    assert SQLiteCampaignStore(path).supervised_parameter("1", "1", "U_SHAPE_REPAIR") == frozen


def test_valid_frozen_parameter_replay_and_isolation(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "valid.sqlite3")
    frozen = state()
    store.freeze_supervised_parameter("1", frozen)
    store.freeze_supervised_parameter("1", frozen)
    assert store.supervised_parameter("1", "1", "U_SHAPE_REPAIR") == frozen
    assert store.supervised_parameter("2", "1", "U_SHAPE_REPAIR") is None
    assert store.supervised_parameter("1", "2", "U_SHAPE_REPAIR") is None
    assert store.supervised_parameter("1", "1", "TAIL_HINGE") is None
