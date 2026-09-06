from importlib import resources

import pytest

from factor_engine.runtime.config import load_profile


@pytest.mark.parametrize("name", ["dev", "prod", "staging"])
def test_builtin_profile_is_a_package_resource(name):
    assert resources.files("factor_engine.runtime").joinpath("profiles", name + ".yaml").is_file()
    payload = load_profile(name)
    assert isinstance(payload, dict) and payload
    if name != "dev":
        assert payload["run"]["market"] == "us"
        assert payload["run"]["calendar"] == "NYSE"
        assert payload["pit"]["enforce"] is True


@pytest.mark.parametrize("name", ["../prod", "/tmp/prod", "a/b", "..", "prod.yaml"])
def test_profile_name_cannot_escape_package(name):
    with pytest.raises(ValueError, match="profile"):
        load_profile(name)
