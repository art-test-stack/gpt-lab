"""Dataloaders with optional native acceleration."""

__all__ = ["DistDataLoader", "build_dataloader", "DataLoaderState"]


def __getattr__(name):
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
