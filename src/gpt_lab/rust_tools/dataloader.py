"""Bind native packing to the framework's CPU batch buffer and checkpoints."""

from importlib import import_module


DATALOADER_BACKENDS = ("auto", "python", "rust-packing", "rust")
PACKING_STRATEGIES = ("rolling-16", "block-best-fit")
_EXTENSION = "gpt_lab_rust_packer"


def _extension(*, required=True):
    try:
        return import_module(_EXTENSION)
    except ImportError as exc:
        if not required and isinstance(exc, ModuleNotFoundError) and exc.name == _EXTENSION:
            return None
        raise ImportError(
            "Rust dataloaders require the native extension. Run `uv sync --locked --extra rust-dataloader` "
            "with your existing device extra (--extra cpu or --extra gpu)."
        ) from exc


def native_batch(loader, dataset, backend, packing_backend):
    """Return a native batch filler, or None to use the Python packer."""
    if backend not in DATALOADER_BACKENDS:
        raise ValueError(f"Unknown dataloader backend {backend!r}")
    if packing_backend not in ("python", "rust"):
        raise ValueError(f"Unknown packing backend {packing_backend!r}")
    if backend == "rust" and packing_backend == "rust":
        raise ValueError("The Rust dataloader includes its own packing backend")
    if backend == "rust-packing" or packing_backend == "rust":
        if loader.block_best_fit:
            raise ValueError("block-best-fit requires the Python or end-to-end Rust training loader")
        return _packing_batch(loader, dataset)
    if backend == "python":
        return None

    from gpt_lab.data.loader import ShardedDataset
    from gpt_lab.tokenizer import Tokenizer

    if not isinstance(dataset, ShardedDataset) or type(dataset.tokenizer) is not Tokenizer:
        if backend == "auto":
            return None
        raise TypeError("The Rust dataloader requires cached text and the repository's tiktoken Tokenizer")
    if backend == "auto" and type(dataset) is not ShardedDataset:
        return None
    sm = dataset.sm
    if sm.base_url:
        missing = next((path for path in sm.shard_paths if not path.is_file()), None)
        if missing is not None:
            if backend == "auto":
                return None
            raise FileNotFoundError(f"The Rust dataloader requires a cached shard at {missing}")
    extension = _extension(required=backend == "rust")
    if extension is None:
        return None

    from gpt_lab.utils.schemas import DataLoaderState

    encoding = dataset.tokenizer.main
    resume = None
    if (state := dataset.start_state) is not None:
        resume = (
            (state.shard_idx, state.global_shard_idx, state.row_group_idx, state.offset_in_row_group, state.epoch),
            state.lookahead_refs, state.pending_document_ref, state.document_token_offset,
        )
    native = extension.TextLoader(
        list(zip(sm.shard_idx, sm.shard_paths, strict=True)), sm.column_name,
        dataset.split == "train", sm.ddp_rank, sm.world_size, (loader.B, loader.T + 1),
        dataset.tokenizer_threads, dataset.tokenizer_batch_size or 64,
        (encoding._mergeable_ranks, encoding._pat_str, encoding._special_tokens, loader.bos_token_id),
        resume, block_exact_pair=loader.block_best_fit,
    )
    output = loader.cpu.numpy()

    def fill(loader):
        (shard, global_shard, group, row, epoch), refs, pending, offset = native.fill(output)
        cursor = DataLoaderState(
            shard_idx=shard, global_shard_idx=global_shard, row_group_idx=group,
            offset_in_row_group=row, epoch=epoch,
        )
        return loader._checkpoint(cursor, tuple(refs), pending, offset)

    return fill


def _packing_batch(loader, dataset):
    import numpy as np
    import torch

    def contiguous(document):
        if isinstance(document, torch.Tensor):
            document = document.detach().to(device="cpu", dtype=torch.long).numpy()
        return np.ascontiguousarray(document, dtype=np.int64)

    pack = _extension().pack
    loader._restore_python(dataset, prepare=contiguous)
    output = loader.cpu.numpy()

    def fill(loader):
        try:
            loader.document, loader.document_ref, loader.offset = pack(
                output, loader.lookahead, lambda: loader._fill_lookahead(prepare=contiguous),
                loader.document, loader.document_ref,
                loader.offset, loader.bos_token_id,
            )
        except StopIteration:
            loader.document = loader.document_ref = None
            loader.offset = 0
            raise
        return loader._python_checkpoint()

    return fill
