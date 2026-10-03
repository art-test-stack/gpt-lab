"""Compatibility wrapper for gpt-lab tokenizer train."""

from collections.abc import Sequence
import sys

from gpt_lab.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    return cli_main(['tokenizer', 'train'] + list(sys.argv[1:] if argv is None else argv))


if __name__ == '__main__':
    raise SystemExit(main())
