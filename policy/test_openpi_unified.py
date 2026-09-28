import pytest

from policy.openpi import Policy
from policy.openpi import deploy_policy


class _Backend:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def needs_observation(self, task):
        self.calls.append(("needs_observation", task))
        return True

    def eval(self, task, observation):
        self.calls.append(("eval", task, observation))

    def close(self):
        self.calls.append(("close",))


@pytest.mark.parametrize("client_type", ["single", "streaming"])
def test_unified_policy_delegates_to_selected_backend(monkeypatch, client_type):
    backend = _Backend()
    monkeypatch.setattr(
        deploy_policy,
        "build_backend",
        lambda selected_type, args: backend,
    )
    task = object()
    policy = Policy({"client_type": client_type})

    policy.reset()
    assert policy.needs_observation(task)
    policy.eval(task, "observation")
    policy.close()

    assert [call[0] for call in backend.calls] == [
        "reset",
        "needs_observation",
        "eval",
        "close",
    ]


def test_unified_policy_rejects_unknown_client_type(monkeypatch):
    from policy.openpi import clients

    with pytest.raises(ValueError, match="client_type"):
        clients.build_backend("unknown", {})


@pytest.mark.parametrize(
    ("client_type", "module_name"),
    [("single", "single"), ("streaming", "streaming")],
)
def test_backend_factory_loads_selected_client(monkeypatch, client_type, module_name):
    from importlib import import_module

    from policy.openpi import clients

    module = import_module(f"policy.openpi.clients.{module_name}")
    sentinel = object()
    monkeypatch.setattr(module, "build", lambda args: sentinel)

    assert clients.build_backend(client_type, {}) is sentinel
