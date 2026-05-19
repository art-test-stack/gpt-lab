from __future__ import annotations
from pathlib import Path
import random, pickle
from gpt_lab.utils.default import RANDOM_SEED, DATA_DIR
from gpt_lab.data.normalizers import clean_codeparrot_example
from gpt_lab.utils.logging import log0
from typing import Union, Dict, Callable, Optional, Iterable, Tuple

# TODO: consider using compression.ztsd when python.version >= 3.14 (pi)

from tqdm import tqdm
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

import logging

logger = logging.getLogger(__name__)

_fineweb_2_names_raw = ["rus_Cyrl", "cmn_Hani", "deu_Latn", "jpn_Jpan", "spa_Latn", "fra_Latn", "ita_Latn", "por_Latn", "pol_Latn", "nld_Latn", "ind_Latn", "vie_Latn", "fas_Arab", "arb_Arab", "tur_Latn", "tha_Thai", "ukr_Cyrl", "ell_Grek", "kor_Hang", "ces_Latn", "swe_Latn", "hun_Latn", "ron_Latn", "nob_Latn", "dan_Latn", "fin_Latn", "bul_Cyrl", "hin_Deva", "ben_Beng", "slk_Latn", "slk_Latn", "lit_Latn", "bos_Latn", "slv_Latn", "ekk_Latn", "cat_Latn", "tam_Taml", "hrv_Latn", "lvs_Latn", "zsm_Latn", "azj_Latn", "srp_Cyrl", "kat_Geor", "npi_Deva", "mar_Deva", "nno_Latn"]
_fineweb_2_names = []
alph = []
for lang in _fineweb_2_names_raw:
    lang_code, script = lang.split("_")
    if script not in alph:
        alph.append(script)
        _fineweb_2_names.append(lang)

def apply_temperature_sampling(
    sources,
    alpha: float = 0.5,
    min_weight: float = 0.0,
):
    """
    Temperature sampling over dataset weights.

    p_i ∝ w_i^alpha

    alpha < 1:
        flattens distribution
        boosts smaller datasets

    alpha = 1:
        original distribution

    alpha = 0:
        uniform sampling
    """
    raw = [max(src.get("weight", 1.0), min_weight) for src in sources]
    scaled = [w ** alpha for w in raw]
    total = sum(scaled)

    for src, w in zip(sources, scaled):
        src["weight"] = w / total

    return sources

def safe_byte_truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    # walk backward until valid UTF-8
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""

def _normalize_sources_weights(sources, weights_sum):
    """Attach a stable `name` key to each source dict. Call once before any loop."""
    for src in sources:
        src["weight"] = src.get("weight", 1.0) / weights_sum
    return sources

def load_datasets(
         # { "path": str, "name": str (optional), "weight": float (optional), "hook": Callable (optional) } 
        sources: Iterable[Dict[str, Union[str, float, Callable]]],
        data_dir: Union[str,Path] = DATA_DIR,
        split: str = "train",
        streaming: bool = True,
        shuffle: bool = True,
        random_seed: int = 42,
        loader_kwargs: dict = None
    ) -> Dict[str, Iterable]:
    ds = dict()
    loader_kwargs = loader_kwargs or {}
    for src in sources:
        path, name = src["path"], src.get("name", None)
        ds_name = path if name is None else f"{path}:{name}"
        ds_split = src.get("split", split)
        ds_hook = src.get("hook", lambda x: x)
        _ds = ds_hook(
            load_dataset(
                path, 
                name=name,
                split=ds_split,
                streaming=streaming, 
                cache_dir=data_dir, 
                **loader_kwargs
            )
        )
        if "filter_fn" in src:
            _ds = _ds.filter(src["filter_fn"])
        if shuffle and streaming:
            _ds = _ds.shuffle(seed=random_seed)
        ds[ds_name] = _ds
    return ds

def display_stat_by_source(stat_by_source: Dict[str, Dict[str, int]]):
    from rich.console import Console
    from rich.markdown import Markdown
    print("Stats by source:")
    md = "| Source | Chars | Docs | Percentage chars (%) | Percentage bytes (%) | Size (MB) |\n|-|-|-|-|-|-|\n"
    total_chars = sum(stat["chars"] for stat in stat_by_source.values())
    total_bytes = sum(stat["bytes"] for stat in stat_by_source.values())
    sum_per_char = 0
    sum_per_byte = 0
    for src, stat in stat_by_source.items():
        chars = stat["chars"]
        docs = stat["docs"]
        bytes_ = stat["bytes"]
        per_char = chars / total_chars * 100
        per_byte = bytes_ / total_bytes * 100
        sum_per_char += per_char
        sum_per_byte += per_byte
        md += f"| {src} | {chars:,} | {docs:,} | {per_char:.2f} | {per_byte:.2f} | {bytes_ / 1e6:.2f} |\n"
    md += f"| Total | {total_chars:,} | {sum(stat['docs'] for stat in stat_by_source.values()):,} | {sum_per_char:.2f} | {sum_per_byte:.2f} | {total_bytes / 1e6:.2f} |\n"
    Console().print(Markdown(md))

class TokenizerCorpus:
    def __init__(
            self, 
            total_bytes: int,
            total_docs: int,
            corpus_dir: Union[str, Path],
            random_seed: int = RANDOM_SEED,
            total_chars: Optional[int] = None, 
            sources: Optional[dict] = None,
            compressed: bool = False,
            stat_by_source: Optional[Dict[str, Dict[str, int]]] = None,
        ):
        corpus_dir = TokenizerCorpus.init_corpusdir(corpus_dir)
        self.corpus_dir = corpus_dir
        self.random_seed = random_seed
        self.total_bytes = total_bytes
        self.total_chars = total_chars
        self.total_docs = total_docs
        if sources is not None:
            for source in sources:
                if "filter_fn" in source:
                    # filter functions are not serializable; 
                    # we can re-apply them when loading if needed, 
                    # but for now we just drop them from the metadata
                    source["filter_fn"] = None 
        self.sources = sources 
        self.compressed = compressed
        self.stat_by_source = stat_by_source
    
    @property
    def meta_path(self):
        return TokenizerCorpus._make_meta_path(self.corpus_dir)
    
    def save(self):
        with open(self.meta_path, "wb") as f:
            pickle.dump(self, f)

    def iterator(self, max_bytes: Optional[int] = None) -> Iterable[str]:
        byte_count = 0
        for shard in self.shard_paths():
            with self.open_text_file(shard) as f:
                for line in f:
                    yield line.strip()
                    byte_count += len(line)
                    if max_bytes and byte_count >= max_bytes:
                        return

    @staticmethod
    def _make_meta_path(corpus_dir: Path):
        return (corpus_dir / "meta").with_suffix(".pkl")
    
    @classmethod
    def from_path(cls, path: Union[Path, str]):
        if not isinstance(path, Path):
            path = Path(path)
        path = cls._make_meta_path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def write_from_sources(
            cls,
            corpus_dir: Union[str, Path],
            sources: Optional[dict] = None, # dict ds_name: weight,
            bytes_per_doc: int = 10_000,
            max_bytes: int = 1_000_000_000,
            random_seed: int = RANDOM_SEED,
            split: str = "train",
            temperature_alpha: Optional[float] = None,
            compressed: bool = False, # False: .txt, True: sharded .txt.zst (optimized memory for large corpora)
            shard_size_bytes: Optional[int] = None, # only relevant if compressed=True; if None, defaults to max_bytes (i.e. single shard)
        ):
        if shard_size_bytes is None:
            shard_size_bytes = max_bytes // 10 if compressed else max_bytes # heuristic for shard size; adjust as needed
        # TODO: fix zstd comp
        if compressed:
            log0("Warning: compressed corpus writing is not yet implemented; writing uncompressed .txt files instead", logger=logger)
        compressed = False
        
        corpus_dir = TokenizerCorpus.init_corpusdir(corpus_dir)
        bytes_count, char_count, doc_count, stat_by_source = write_corpus_sample(
            sources=sources,
            bytes_per_doc=bytes_per_doc,
            max_bytes=max_bytes,
            corpus_dir=corpus_dir,
            random_seed=random_seed,
            temperature_alpha=temperature_alpha,
            split=split,
            shard_size_bytes=shard_size_bytes,
            compressed=compressed,
        )
        display_stat_by_source(stat_by_source)
        meta = cls(
            corpus_dir=corpus_dir,
            total_bytes=bytes_count,
            total_chars=char_count,
            total_docs=doc_count,
            compressed=compressed,
            sources=sources,
            stat_by_source=stat_by_source,
        )
        meta.save()
        return meta
    
    def show_stats(self):
        print(f"Corpus directory: {self.corpus_dir}")
        print(f"Total bytes: {self.total_bytes:,}")
        print(f"Total chars: {self.total_chars:,}")
        print(f"Total docs: {self.total_docs:,}")
        if self.stat_by_source:
            display_stat_by_source(self.stat_by_source)
        
    @classmethod
    def from_sources(
            cls,
            corpus_dir: Union[str, Path],
            sources: Optional[dict] = None, # dict ds_name: weight,
            bytes_per_doc: int = 10_000,
            max_bytes: int = 1_000_000_000,
            random_seed: int = RANDOM_SEED,
            split: str = "train",
            compressed: bool = False,
            shard_size_bytes: Optional[int] = None,
            loader_fn: Optional[Callable] = None, # if provided, should be function that takes dataset config and returns iterator of text samples; overrides default loading from datasets library
        ) -> TokenizerCorpus:
        meta = None
        if loader_fn is not None:
            class CustomLoaderCorpus(TokenizerCorpus):
                # overwrite iterator to use custom loader
                def iterator(self, max_bytes: Optional[int] = None) -> Iterable[str]:
                    return loader_fn()
            meta = CustomLoaderCorpus(corpus_dir=corpus_dir, total_bytes=-1, total_docs=-1)
        else:
            try:
                meta = cls.from_path(corpus_dir)
            except (FileNotFoundError, pickle.UnpicklingError, EOFError):
                meta = cls.write_from_sources(
                    corpus_dir=corpus_dir,
                    sources=sources,
                    bytes_per_doc=bytes_per_doc,
                    max_bytes=max_bytes,
                    compressed=compressed,
                    split=split,
                    random_seed=random_seed,
                    shard_size_bytes=shard_size_bytes,
                )
        assert meta is not None, "Failed to create or load corpus metadata"
        return meta
    
    def open_text_file(self, path: Path):
        if path.suffix == ".zst":
            raise NotImplementedError
            # f = open(path, "rb")
            # dctx = zstd.ZstdDecompressor()
            # stream = dctx.stream_reader(f)
            # return stream
        return open(path, "r", encoding="utf-8", errors="ignore")
    
    def shard_paths(self):
        suffix = ".txt.zst" if self.compressed else ".txt"
        return sorted(self.corpus_dir.glob(f"*{suffix}"))

    @staticmethod
    def init_corpusdir(corpus_dir: Union[str, Path]):
        if isinstance(corpus_dir, str):
            corpus_dir = Path(corpus_dir)
        if not corpus_dir.exists():
            corpus_dir.mkdir(parents=True, exist_ok=True)
        return corpus_dir

def weighted_sample_generator(sources, iters, prng, batch_size=10_000):
    names = [src["name"] for src in sources]
    weights = [src["weight"] for src in sources]
    
    while True:
        batch = prng.choices(names, weights=weights, k=batch_size)
        for name in batch:
            yield next(iters[name]), name

def write_corpus_sample(
        sources = None, # dict ds_name: weight
        bytes_per_doc: int = 10_000,
        max_bytes: int = 1_000_000_000,
        shard_size_bytes: int = 1_000_000_000,
        per_dataset_normalizer: Optional[Callable] = None,
        corpus_dir: Path = DATA_DIR / "tokenizer_corpus",
        temperature_alpha: Optional[float] = None,
        split: str = "train",
        show_progress: bool = True,
        random_seed: int = RANDOM_SEED,
        compressed: bool = False,
        streaming: bool = True,
        shuffle: bool = True,
    ):
    if sources is None:
        sources = [
            # base corpus for tokenizer training; mostly web text with some code and math
            { "path": "HuggingFaceFW/fineweb-edu", "weight": 0.30 },
            { "path": "HuggingFaceTB/finemath", "weight": 0.15, "name": "finemath-4plus" },
            { "path": "codeparrot/codeparrot-clean", "weight": 0.15 },
        ]
        multilingual_weight = 0.4
        per_lang = multilingual_weight / len(_fineweb_2_names)
        for name in _fineweb_2_names:
            sources.append({
                "path": "HuggingFaceFW/fineweb-2",
                "weight": per_lang,
                "name": name
            })
        # ronantakizawa/github-top-code with file_language="Python"
    ds = load_datasets(sources, split=split, random_seed=random_seed, streaming=streaming, shuffle=shuffle)
    sources = [ 
        { 
            **src, 
            "weight": src.get("weight", 1.0), 
            "name": src["path"] + (
                f":{src.get('name', None)}" if src.get("name") else ""
            ) 
        }
        for src in sources 
    ] # ensure all sources have weight key
    if temperature_alpha is not None:
        sources = apply_temperature_sampling(sources, alpha=temperature_alpha)
    weights_sum = sum(src["weight"] for src in sources)
    sources = _normalize_sources_weights(sources, weights_sum) 

    def _make_source_weights(sources):
        return { src["name"]: src["weight"] for src in sources }
    
    source_weights = _make_source_weights(sources)

    if max_bytes == -1:
        max_bytes = sum(len(text.encode("utf-8")) for subset in ds.values() for text in subset["text"])
        print(f"Calculated max_bytes from datasets: {max_bytes}")
    
    total_chars = 0
    total_bytes = 0
    total_docs = 0
    shard_index = 0
    shard_bytes = 0
    stat_by_source = { src["name"]: {"chars": 0, "docs": 0, "bytes": 0} for src in sources }
    
    def cycling_iterator(dataset):
        while True:
            yield from dataset

    def open_new_shard(idx):
        suffix = ".txt.zst" if compressed else ".txt"
        shard_path = (corpus_dir / f"shard_{idx:05d}").with_suffix(suffix)
        f = open(shard_path, "w", encoding="utf-8", errors="ignore")
        return f
    
    r = random.Random(random_seed)
    iters = { name: cycling_iterator(subset) for name, subset in ds.items() }
    writer = open_new_shard(shard_index)

    sampler = weighted_sample_generator(sources, iters, r)

    with tqdm(total=max_bytes, disable=not show_progress) as pbar:
        while total_bytes < max_bytes:
            sample, src_name = next(sampler)
            text = sample.get("text") or sample.get("content") or ""
            if not text.strip():
                continue
            if src_name == "codeparrot/codeparrot-clean":
                text = clean_codeparrot_example(text)

            text = safe_byte_truncate(text, bytes_per_doc) # arbitrary truncation
            

            if per_dataset_normalizer:
                text = per_dataset_normalizer(text, dataset_name=src_name) # should be function(text, dataset_name) -> text
            
            if not text.strip():
                continue

            encoded = text.encode("utf-8")
            
            total_chars += len(text)
            writer.write(text + "\n")

            total_bytes += len(encoded)
            shard_bytes += len(encoded)
            total_docs += 1
            stat_by_source[src_name]["chars"] += len(text)
            stat_by_source[src_name]["docs"] += 1
            stat_by_source[src_name]["bytes"] += len(encoded)
            pbar.update(len(encoded))

            if shard_bytes >= shard_size_bytes:
                writer.close()
                shard_index += 1
                shard_bytes = 0
                writer = open_new_shard(shard_index)
    writer.close()
    return total_bytes, total_chars, total_docs, stat_by_source
