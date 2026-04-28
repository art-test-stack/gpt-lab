from importlib import import_module
import gpt_lab.tokenizer.pretokenizer as sp
import rbpe

def _token_generator_from_lines(corpus_iter_fn, corpus_path, config):
    pretok = sp.SimplePreTokenizer(config)
    for line in corpus_iter_fn(corpus_path):
        toks = pretok._split(line, drop_special_tokens=True)
        # emit token strings where each byte-level symbol is represented as its corresponding character
        # If your byte-level mapping maps bytes->private-range chars, ensure pretok returns that.
        for t in toks:
            # Yield as Python str (Rust will encode to bytes)
            yield t

def bpe(corpus_iter_fn, corpus_path, config):
    """
    Compatibility wrapper with your Python bpe signature.
    Streams pretokenized tokens to the Rust trainer.
    """
    token_iter = _token_generator_from_lines(corpus_iter_fn, corpus_path, config)
    # call native trainer
    merges, vocab = rbpe.train_from_token_iterator(
        token_iter,
        vocab_size=config.vocab_size,
        merges_per_pass=config.merges_per_pass,
        special_tokens=list(config.special_tokens.list()) if hasattr(config, "special_tokens") else None
    )
    return merges, vocab
