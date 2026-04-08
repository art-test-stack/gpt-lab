from pathlib import Path
import random, pickle
from gpt_lib.utils.default import RANDOM_SEED, CACHE_DIR, DATA_DIR
from gpt_lib.data.loader import load_datasets
from gpt_lib.data.normalizers import clean_codeparrot_example
from typing import Union, Dict, Callable, Optional, Iterable, Tuple

# TODO: consider using compression.ztsd when python.version >= 3.14 (pi)
import zstd

from tqdm import tqdm

_fineweb_2_names_raw = ["rus_Cyrl", "cmn_Hani", "deu_Latn", "jpn_Jpan", "spa_Latn", "fra_Latn", "ita_Latn", "por_Latn", "pol_Latn", "nld_Latn", "ind_Latn", "vie_Latn", "fas_Arab", "arb_Arab", "tur_Latn", "tha_Thai", "ukr_Cyrl", "ell_Grek", "kor_Hang", "ces_Latn", "swe_Latn", "hun_Latn", "ron_Latn", "nob_Latn", "dan_Latn", "fin_Latn", "bul_Cyrl", "hin_Deva", "ben_Beng", "slk_Latn", "slk_Latn", "lit_Latn", "bos_Latn", "slv_Latn", "ekk_Latn", "cat_Latn", "tam_Taml", "hrv_Latn", "lvs_Latn", "zsm_Latn", "azj_Latn", "srp_Cyrl", "kat_Geor", "npi_Deva", "mar_Deva", "nno_Latn"]
_fineweb_2_names = []
alph = []
for lang in _fineweb_2_names_raw:
    lang_code, script = lang.split("_")
    if script not in alph:
        alph.append(script)
        _fineweb_2_names.append(lang)

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
            total_chars: int, 
            total_docs: int,
            corpus_dir: Union[str, Path],
            random_seed: int = RANDOM_SEED,
            sources: Optional[dict] = None,
            compressed: bool = False,
            stat_by_source: Optional[Dict[str, Dict[str, int]]] = None,
        ):
        corpus_dir = TokenizerCorpus.init_corpusdir(corpus_dir)
        self.corpus_dir = corpus_dir
        self.random_seed = random_seed
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

    def iterator(self, max_chars: Optional[int] = None) -> Iterable[str]:
        char_count = 0
        for shard in self.shard_paths():
            with self.open_text_file(shard) as f:
                for line in f:
                    yield line.strip()
                    char_count += len(line)
                    if max_chars and char_count >= max_chars:
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
            chars_per_doc: int = 10_000,
            max_chars: int = 1_000_000_000,
            random_seed: int = RANDOM_SEED,
            split: str = "train",
            compressed: bool = False, # False: .txt, True: sharded .txt.zst (optimized memory for large corpora)
            shard_size_chars: Optional[int] = None, # only relevant if compressed=True; if None, defaults to max_chars (i.e. single shard)
        ):
        if shard_size_chars is None:
            shard_size_chars = max_chars // 10 if compressed else max_chars # heuristic for shard size; adjust as needed
        # TODO: fix zstd comp
        compressed = False
        corpus_dir = TokenizerCorpus.init_corpusdir(corpus_dir)
        char_count, doc_count, stat_by_source = write_corpus_sample(
            sources=sources,
            chars_per_doc=chars_per_doc,
            max_chars=max_chars,
            corpus_dir=corpus_dir,
            random_seed=random_seed,
            split=split,
            shard_size_chars=shard_size_chars,
            compressed=compressed,
        )
        display_stat_by_source(stat_by_source)
        meta = cls(
            corpus_dir=corpus_dir,
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
        print(f"Total chars: {self.total_chars:,}")
        print(f"Total docs: {self.total_docs:,}")
        if self.stat_by_source:
            display_stat_by_source(self.stat_by_source)
        
    @classmethod
    def from_sources(
            cls,
            corpus_dir: Union[str, Path],
            sources: Optional[dict] = None, # dict ds_name: weight,
            chars_per_doc: int = 10_000,
            max_chars: int = 1_000_000_000,
            random_seed: int = RANDOM_SEED,
            split: str = "train",
            compressed: bool = False,
            shard_size_chars: Optional[int] = None,
            loader_fn: Optional[Callable] = None, # if provided, should be function that takes dataset config and returns iterator of text samples; overrides default loading from datasets library
        ):
        meta = None
        if loader_fn is not None:
            class CustomLoaderCorpus(TokenizerCorpus):
                # overwrite iterator to use custom loader
                def iterator(self, max_chars: Optional[int] = None) -> Iterable[str]:
                    return loader_fn()
            meta = CustomLoaderCorpus(corpus_dir=corpus_dir, total_chars=-1, total_docs=-1)
        else:
            try:
                meta = cls.from_path(corpus_dir)
            except:
                meta = cls.write_from_sources(
                    corpus_dir=corpus_dir,
                    sources=sources,
                    chars_per_doc=chars_per_doc,
                    max_chars=max_chars,
                    compressed=compressed,
                    split=split,
                    random_seed=random_seed,
                    shard_size_chars=shard_size_chars,
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
        if not corpus_dir.suffix == None:
            corpus_dir = corpus_dir.with_suffix("")
        # if not corpus_dir.suffix == ".txt":
        #     corpus_dir = corpus_dir.with_suffix(".txt")
        # if compressed:
        #     corpus_dir = corpus_dir.with_suffix(".txt.zst")
        
        if not corpus_dir.exists():
            corpus_dir.mkdir(parents=True, exist_ok=True)
        return corpus_dir
    

def write_corpus_sample(
        sources = None, # dict ds_name: weight
        chars_per_doc: int = 10_000,
        max_chars: int = 1_000_000_000,
        shard_size_chars: int = 1_000_000_000,
        per_dataset_normalizer: Optional[Callable] = None,
        corpus_dir: Path = DATA_DIR / "tokenizer_corpus",
        split: str = "train",
        show_progress: bool = True,
        random_seed: int = RANDOM_SEED,
        compressed: bool = False,
        streaming: bool = True,
        shuffle: bool = True,
    ):

    if not sources:
        sources = [
            # base corpus for tokenizer training; mostly web text with some code and math
            { "path": "HuggingFaceFW/fineweb-edu", "weight": 0.45 },
            { "path": "HuggingFaceTB/finemath", "weight": 0.2, "name": "finemath-4plus" },
            { "path": "codeparrot/codeparrot-clean", "weight": 0.2 },
        ]
        for name in _fineweb_2_names:
            sources.append({ "path": "HuggingFaceFW/fineweb-2", "weight": 0.15 / len(_fineweb_2_names), "name": name })
        # ronantakizawa/github-top-code with file_language="Python"
    ds = load_datasets(sources, split=split, random_seed=random_seed, streaming=streaming, shuffle=shuffle)
    sources = [ { **src, "weight": src.get("weight", 1.0), "name": src["path"] + (f":{src.get('name', None)}" if src.get("name") else "") } for src in sources ] # ensure all sources have weight key
    len_ds = sum(1 for _ in ds.values())
    len_src = sum(1 for _ in sources)
    weights_sum = sum(src.get("weight", 1.0) for src in sources)
    if not len_src == len_ds:
        for src in sources:
            src["weight"] = src.get("weight", 1.0) / weights_sum
    print("Len sources vs loaded datasets:", len_src, len_ds)
    print("Sum of dataset weights after adjustment:", sum(src.get("weight", 1.0) for src in sources))
    print(f"Dataset weights scaled by factor of 1 / {weights_sum:.2f} to match number of loaded datasets.")
    r = random.Random(random_seed)
    if max_chars == -1:
        max_chars = sum(len(text) for subset in ds.values() for text in subset["text"])
        print(f"Calculated max_chars from datasets: {max_chars}")
    
    total_chars = 0
    total_docs = 0
    shard_index = 0
    shard_chars = 0
    stat_by_source = { src["name"]: {"chars": 0, "docs": 0, "bytes": 0} for src in sources }

    def open_new_shard(idx):
        suffix = ".txt.zst" if compressed else ".txt"
        shard_path = (corpus_dir / f"shard_{idx:05d}").with_suffix(suffix)
        # shard_path.mkdir(parents=True, exist_ok=True)
        f = open(shard_path, "w", encoding="utf-8", errors="ignore")
        return f
        # cctx = zstd.ZstdCompressor(level=3)
        # return cctx.stream_writer(f)
    
    iters = { name: iter(subset) for name, subset in ds.items() }
    writer = open_new_shard(shard_index)

    with tqdm(total=max_chars, disable=not show_progress) as pbar:
        while total_chars < max_chars:
            p = r.random()
            try:
                for src in sources:
                    weight = src.get("weight", 1.0)
                    if p < weight:
                        sample = next(iters[src["name"]])
                        break
                    else:
                        p -= weight
            except StopIteration:
                break
            text = sample.get("text") or sample.get("content") or ""
            if not text.strip():
                continue
            total_docs += 1
            if src.get("name") == "codeparrot/codeparrot-clean":
                text = clean_codeparrot_example(text)

            text = text[:chars_per_doc] # arbitrary truncation
            if per_dataset_normalizer:
                text = per_dataset_normalizer(text, dataset_name=src.get("name", src["name"])) # should be function(text, dataset_name) -> text
            
            if not text.strip():
                continue
                
            encoded = text.encode("utf-8")

            writer.write(encoded.decode("utf-8"))

            total_chars += len(text)
            shard_chars += len(text)
            total_docs += 1
            stat_by_source[src["name"]]["chars"] += len(text)
            stat_by_source[src["name"]]["docs"] += 1
            stat_by_source[src["name"]]["bytes"] += len(encoded)
            pbar.update(len(encoded))

            if shard_chars >= shard_size_chars:
                writer.close()
                shard_index += 1
                shard_chars = 0
                writer = open_new_shard(shard_index)
    writer.close()
    return total_chars, total_docs, stat_by_source
