"""Checkpoint serde allowlist (LangGraph 1.2 deprecation).

A type missing from ``allowed_msgpack_modules`` does not raise — it round-trips
back as a plain dict. So the drift guard matters more than the happy path: it
fails the moment a model is added to app.state and not registered.
"""

import inspect

from pydantic import BaseModel

from app import state as state_module
from app.graph import CHECKPOINT_TYPES, _SERDE


def _roundtrip(serde, obj):
    return serde.loads_typed(serde.dumps_typed(obj))


def test_every_app_state_model_is_registered():
    """Drift guard: a new model in app.state must join CHECKPOINT_TYPES."""
    declared = {
        cls
        for _, cls in inspect.getmembers(state_module, inspect.isclass)
        if issubclass(cls, BaseModel) and cls is not BaseModel
    }
    assert declared, "no models found — did app.state move?"
    assert declared == set(CHECKPOINT_TYPES), (
        "app.state models not registered for checkpoint deserialization: "
        f"{sorted(c.__name__ for c in declared - set(CHECKPOINT_TYPES))}"
    )


def test_registered_types_survive_a_checkpoint_roundtrip():
    for cls in CHECKPOINT_TYPES:
        # model_construct: some models have required fields, and this test is
        # about the serde allowlist, not about validation.
        out = _roundtrip(_SERDE, cls.model_construct())
        assert isinstance(out, cls), f"{cls.__name__} degraded to {type(out).__name__}"


def test_unregistered_type_degrades_to_a_dict():
    """Paired opposite: proves the allowlist is doing the discriminating, and
    documents the failure mode the drift guard above exists to prevent."""

    class Unregistered(BaseModel):
        value: str = "x"

    out = _roundtrip(_SERDE, Unregistered())
    assert not isinstance(out, Unregistered)
    assert out == {"value": "x"}
