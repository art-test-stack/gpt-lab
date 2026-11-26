import torch

from gpt_lab.model.gpt import DenseTransformer
from gpt_lab.utils.schemas import TransformerConfig


class ByteTokenizer:
    """Deterministic UTF-8 tokenizer with a separate BOS for the small model."""

    def get_bos_token_id(self):
        return 256

    def __call__(self, texts, *, prepend):
        return [[prepend, *text.encode("utf-8")] for text in texts]


def small_model(device, *, head_dim=16):
    torch.manual_seed(29)
    model = DenseTransformer(TransformerConfig(
        vocab_size=257, max_context=2048, n_layers=2, n_heads=2, d_head=head_dim,
        d_model=2 * head_dim, d_ffn=4 * head_dim, window_pattern="SL", window_size=4, attn_impl="sdpa",
    ))
    model.init_weights()
    # Training initialization zeros residual projections. Exercise both layers.
    with torch.no_grad():
        model.lm_head.weight.normal_(std=0.08)
        for block in model.blocks:
            block.attention.o_proj.weight.normal_(std=0.08)
            block.ffn.w_2.weight.normal_(std=0.05)
    return model.to(device).eval()


def freeze_checkpoint(path, device):
    if not path.exists():
        model = small_model(device)
        torch.save({
            "config": model.config.model_dump(),
            "weights": {key: value.cpu() for key, value in model.state_dict().items()},
        }, path)


def load_frozen_checkpoint(path, device):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    config = TransformerConfig.model_validate(saved["config"])
    if config.max_context != 2048 or config.attn_impl != "sdpa":
        raise ValueError("Expected a 2,048-token SDPA checkpoint")
    model = DenseTransformer(config).to(device)
    model.load_state_dict(saved["weights"])
    return model.requires_grad_(False).eval()


def wide_fixture(pad_id):
    def question(prompt_length, answer_lengths):
        prefix = [pad_id, *[97 + pos % 19 for pos in range(prompt_length - 1)]]
        rows = [prefix + [65 + i] * length for i, length in enumerate(answer_lengths)]
        return rows, [prompt_length] * len(rows), [len(row) for row in rows]

    return question(1920, [128, 64, 32, 16]), question(13, [2, 7, 3])
