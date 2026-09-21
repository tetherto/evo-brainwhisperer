from typing import Type

_REGISTRY: dict[str, Type] = {}

# Using a decoratory factory allows to pass a name (`srt`)
# to the `register_dataset` decorator


def register_dataset(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(name: str) -> Type:
    if name not in _REGISTRY:
        raise KeyError(
            f"Dataset {name!r} is not registered. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
