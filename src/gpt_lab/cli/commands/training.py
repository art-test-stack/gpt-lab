"""Register automatic base training and checkpoint resume."""

from. import base


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "train", help="Train models (expensive; supports distributed CUDA training).",
        description="Train from scratch by default, or resume a saved checkpoint.",
        default_subcommand="auto",
    )
    base.register(parser)
