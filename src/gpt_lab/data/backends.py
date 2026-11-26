"""Dataloader modes and optional native dispatch shared by the framework and CLI."""

from importlib import import_module


DATALOADER_BACKENDS = ("auto", "python", "rust-packing", "rust")
PACKING_STRATEGIES = ("rolling-16", "block-best-fit")


def resolve_packing_backend(dataloader_backend, packing_backend, packing_strategy, split):
    if dataloader_backend not in DATALOADER_BACKENDS:
        raise ValueError(f"Unknown dataloader backend {dataloader_backend!r}")
    if packing_backend not in ("python", "rust"):
        raise ValueError(f"Unknown packing backend {packing_backend!r}")
    if dataloader_backend == "rust-packing":
        packing_backend = "rust"
    elif dataloader_backend == "rust" and packing_backend != "python":
        raise ValueError("The Rust dataloader includes its own packing backend")
    if packing_strategy not in PACKING_STRATEGIES:
        raise ValueError(f"Unknown packing strategy {packing_strategy!r}")
    if packing_strategy == "block-best-fit" and (
        dataloader_backend not in ("auto", "rust") or packing_backend != "python" or split != "train"
    ):
        raise ValueError("block-best-fit requires the end-to-end Rust training loader")
    return packing_backend


def native_extension(*, required=True):
    try:
        return import_module("gpt_lab_rust_packer")
    except ImportError as exc:
        if not required and isinstance(exc, ModuleNotFoundError) and exc.name == "gpt_lab_rust_packer":
            return None
        raise ImportError(
            "Rust dataloaders require the native extension. Run `uv sync --locked --extra rust-dataloader` "
            "with your existing device extra (--extra cpu or --extra gpu)."
        ) from exc
