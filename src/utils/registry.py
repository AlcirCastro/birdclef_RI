"""Generic name → object registry. The spine of the framework's modularity.

Every replaceable component (preprocessor, segmenter, ranker, …) registers
itself here under a string key. The pipeline builder reads `config.<stage>.type`
and calls `REGISTRY.get(key)` to instantiate — no `if/elif` switches anywhere.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._entries:
                raise KeyError(f"{self.name}: duplicate key {key!r}")
            self._entries[key] = obj
            return obj
        return decorator

    def get(self, key: str) -> Any:
        if key not in self._entries:
            raise KeyError(
                f"{self.name}: unknown key {key!r}. Known: {sorted(self._entries)}"
            )
        return self._entries[key]

    def keys(self) -> Iterable[str]:
        return self._entries.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, keys={sorted(self._entries)})"
