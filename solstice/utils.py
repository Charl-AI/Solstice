"""Miscellaneous utilities for Solstice."""

from typing import Any, TypeVar
import equinox as eqx

Module = TypeVar("Module", bound=eqx.Module)


def replace(obj: Module, **changes: Any) -> Module:
    """Make out-of-place changes to a Module, returning a new module with changes
    applied. Just a wrapper around `equinox.tree_at`.

    !!! example
        You can use this in the same way as `dataclasses.replace`, but it only works
        with `eqx.Module`s. The advantage is that it can be used when custom `__init__`
        constructors are defined.
        For more info, see https://github.com/patrick-kidger/equinox/issues/120.

        ```python

        import equinox as eqx
        import solstice

        class Counter(eqx.Module):
            x: int

            def __init__(self, z: int):
                # 'smart' constructor inits x by calculating from z
                self.x = 2 * z

            def increment(self):
                return solstice.replace(self, x=self.x+1)

        C1 = Counter(z=0)
        assert C1.x == 0
        C2 = C1.increment()
        assert C2.x == 1
        ```

    Args:
        obj (Module): Module to make changes to (subclass of `eqx.Module`).
        **changes (Any): Keyword arguments to replace in the module.

    Returns:
        Module: New instance of `obj` with the changes applied.
    """

    keys, vals = zip(*changes.items())
    return eqx.tree_at(lambda c: [getattr(c, key) for key in keys], obj, vals)  # type: ignore


class EarlyStoppingException(Exception):
    """A callback can raise this exception `on_epoch_end` to break the training loop
    early. Useful if you want to write a custom alternative to `EarlyStoppingCallback`."""

    pass
