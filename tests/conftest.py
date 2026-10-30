import pytest


@pytest.fixture(autouse=True)
def _no_actrouter_env(monkeypatch):
    """Keep tests off the hosted ActRouter even when the developer's shell
    has an ActRouter key or URL set."""
    for name in ("ACTROUTER_API_KEY", "ACTROUTER_URL", "ACTROUTER_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
