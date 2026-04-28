from collections import Counter
from gpt_lab.tokenizer.pretokenizer import SimplePreTokenizer

from typing import Dict, Tuple, Union, Generator
from pathlib import Path
from tqdm import tqdm

import heapq
from collections import defaultdict, Counter


def pretokenize_stream(corpus_iter: Generator[str, None, None], pretok) -> Generator[list[bytes], None, None]:
    for line in corpus_iter:
        tokens = pretok._split(line, drop_special_tokens=True)
        for t in tokens:
            yield t.encode("utf-8")

def stream_corpus(corpus_iter: Generator[str, None, None], pretok, merges) -> Generator[bytes, None, None]:
    for line in corpus_iter:
        tokens = pretok._split(line, drop_special_tokens=True)
        for t in tokens:
            for b in t.encode("utf-8"):
                yield bytes([b])

def stream_corpus_tokens(corpus_iter: Generator[str, None, None], pretok, merges) -> Generator[list[bytes], None, None]:
    for line in corpus_iter:
        tokens = pretok._split(line, drop_special_tokens=True)
        for t in tokens:
            token_bytes = [bytes([b]) for b in t.encode("utf-8")]
            # Apply merges
            i = 0
            L = len(token_bytes)
            merged_token_bytes = []
            while i < L:
                j = i + 1
                while j <= L:
                    candidate = b"".join(token_bytes[i:j])
                    if candidate in merges and j - i > 1:
                        j += 1
                    else:
                        break
                merged_token_bytes.append(b"".join(token_bytes[i:j-1]))
                i = j - 1
            yield merged_token_bytes

def bpe(
        corpus_iter_fn,      
        corpus_path: Union[str, Path],
        config
    ) -> Tuple[Dict[bytes, bytes], Dict[bytes, int]]:
    """
    Byte-Level BPE training implementation. Not fast.
    """

    pretknzr = SimplePreTokenizer(config)

    vocab = Counter()
    pair_counts = Counter()
    merges: Dict[bytes, int] = {}

    print("First streaming pass (initial vocab + pair counts)")

    for token_bytes in tqdm(stream_corpus(corpus_iter_fn(corpus_path), pretknzr, merges), desc="Initial pass"):
        vocab.update(token_bytes)
        pair_counts.update(zip(token_bytes, token_bytes[1:]))

    assert len(vocab) > 10, "Vocabulary is empty after initial pass."

    initial_vocab_size = len(vocab)
    nb_merges = config.vocab_size - initial_vocab_size

    print(f"Initial vocab symbols: {initial_vocab_size}")
    print(f"Will perform ~{nb_merges} merges")

    for merge_rank in tqdm(range(nb_merges), desc="BPE Merges"):
        if not pair_counts:
            break

        (A, B), _ = pair_counts.most_common(1)[0]
        merged_symbol = A + B
        merges[merged_symbol] = merge_rank  
        vocab[merged_symbol] = 1
        new_pair_counts = Counter()

        for token_bytes in stream_corpus(corpus_iter_fn(corpus_path), pretknzr, merges):
            A_len = len(A)
            B_len = len(B)

            syms = []
            i = 0
            L = len(token_bytes)

            while i < L:
                if i + A_len + B_len <= L and token_bytes[i:i+A_len] == A and token_bytes[i+A_len:i+A_len+B_len] == B:
                    syms.append(merged_symbol)
                    i += A_len + B_len
                else:
                    syms.append(bytes(token_bytes[i:i+1]))
                    i += 1
            new_pair_counts.update(zip(syms, syms[1:]))

        pair_counts = new_pair_counts

    vocab_list = sorted(vocab.keys()) 
    vocab_dict = {sym: i for i, sym in enumerate(vocab_list)}

    offset = len(vocab_dict)
    for i, special in enumerate(config.special_tokens.list()):
        vocab_dict[special.encode("utf-8")] = offset + i

    print(f"Final vocab size: {len(vocab_dict)}")
    return merges, vocab_dict


class Node:
    __slots__ = ("sym", "prev", "next", "alive")

    def __init__(self, sym):
        self.sym = sym
        self.prev = None
        self.next = None
        self.alive = True

def make_word_nodes(token_bytes):
    """Convert a list of bytes to a linked list of Node objects."""
    prev = None
    nodes = []
    for b in token_bytes:
        n = Node(bytes([b]))
        n.prev = prev
        if prev is not None:
            prev.next = n
        prev = n
        nodes.append(n)
    return nodes[0] if nodes else None

def iter_word_nodes(corpus_iter_fn, corpus_path, pretknzr):
    """Generator: yields HEAD node for each pretokenized token."""
    for text in corpus_iter_fn(corpus_path):
        for tok in pretknzr._split(text, drop_special_tokens=True):
            bs = tok.encode("utf-8")
            head = make_word_nodes(bs)
            if head:
                yield head

def collect_pairs(head):
    """Return list of (node, node.next) pairs for a given linked list head."""
    pairs = []
    cur = head
    while cur and cur.next:
        if cur.alive and cur.next.alive:
            pairs.append(cur)
        cur = cur.next
    return pairs

def bpe_fast(corpus_iter_fn, corpus_path, config):

    vocab_size = config.vocab_size
    assert vocab_size > 256, "Vocab size must be greater than 256 for byte-level BPE."
    
    vocab = {bytes([i]) for i in range(256)}

    all_words = []
    pair_freq = Counter()
    pair_locs = defaultdict(list)

    pretknzr = SimplePreTokenizer(config)
    for head in tqdm(iter_word_nodes(corpus_iter_fn, corpus_path, pretknzr), desc="Building initial pair stats", total=config.max_chars // 10_000):
        all_words.append(head)
        for node in collect_pairs(head):
            pair = (node.sym, node.next.sym)
            pair_freq[pair] += 1
            pair_locs[pair].append(node)

    heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)

    merges = {}
    rank = 0

    target_merges = vocab_size - len(vocab)

    with tqdm(total=target_merges, desc="BPE Merges") as pbar:
        while len(vocab) < vocab_size and heap:
            neg_freq, pair = heapq.heappop(heap)
            freq = -neg_freq

            if pair_freq.get(pair, 0) != freq or freq == 0:
                continue

            A, B = pair
            merged = A + B
            merges[pair] = rank
            rank += 1
            vocab.add(merged)

            nodes_to_update = pair_locs[pair]
            pair_locs[pair] = []

            new_pair_freq = Counter()
            new_pair_locs = defaultdict(list)

            for node in nodes_to_update:
                if not node.alive or not node.next or not node.next.alive:
                    continue
                if node.sym != A or node.next.sym != B:
                    continue

                nxt = node.next
                node.sym = merged
                nxt.alive = False

                node.next = nxt.next
                if nxt.next:
                    nxt.next.prev = node

                if node.prev and node.prev.alive:
                    p = (node.prev.sym, node.sym)
                    new_pair_freq[p] += 1
                    new_pair_locs[p].append(node.prev)

                if node.next and node.next.alive:
                    p = (node.sym, node.next.sym)
                    new_pair_freq[p] += 1
                    new_pair_locs[p].append(node)

            pair_freq[pair] = 0

            for p, f in new_pair_freq.items():
                pair_freq[p] += f
                pair_locs[p].extend(new_pair_locs[p])
                heapq.heappush(heap, (-pair_freq[p], p))
            pbar.update(1)

    vocab_list = [bytes([i]) for i in range(256)]

    for (A, B), _ in sorted(merges.items(), key=lambda x: x[1]):
        vocab_list.append(A + B)

    vocab_dict = {sym: i for i, sym in enumerate(vocab_list)}

    return merges, vocab_dict
