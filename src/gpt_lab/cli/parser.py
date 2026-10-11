"""Root registration is independent of training stages and workflow imports."""

from .commands import cache, chat, data, tokenizer, training
from .options import ArgumentParser


COMMANDS = (tokenizer, training, data, cache, chat)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="gpt-lab",
        description="Train and use gpt-lab models and tokenizers.",
        epilog="Training can be expensive. Use COMMAND --help for requirements and defaults.",
    )
    subparsers = parser.add_subparsers(required=True, title="commands")
    for command in COMMANDS:
        command.register(subparsers)
    return parser
