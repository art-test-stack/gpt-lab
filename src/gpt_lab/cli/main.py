"""The gpt-lab CLI. Importing this package performs no application setup."""

from collections.abc import Sequence
from importlib import import_module


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit arguments (or sys.argv) and run exactly one workflow."""
    from .parser import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    module_name, handler_name = args._handler.split(":")
    handler = getattr(import_module(module_name), handler_name)
    try:
        return handler(args) or 0
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
