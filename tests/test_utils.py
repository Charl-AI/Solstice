import equinox as eqx
import pytest
import solstice


def test_replace():
    class _DummyCounter(eqx.Module):
        """Dummy counter for testing solstice.replace."""

        count: int

        def __init__(self, initial_count: int) -> None:
            self.count = initial_count

        def increment(self):
            return solstice.replace(self, count=self.count + 1)

    c1 = _DummyCounter(0)
    assert c1.count == 0
    c2 = c1.increment()
    assert c2.count == 1
    c3 = solstice.replace(c1, count=5)
    assert c3.count == 5
    with pytest.raises(AttributeError):
        _ = solstice.replace(c1, z=6)
