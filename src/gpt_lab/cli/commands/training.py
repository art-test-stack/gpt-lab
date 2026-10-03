"""Register training stages here; the root parser does not know their names."""

from. import base


TRAINING_STAGES = (base,)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "train", help="Train models (expensive; supports distributed CUDA training).",
        description="Model training stages. Distributed launches use torchrun -m gpt_lab.cli train...",
    )
    stages = parser.add_subparsers(required=True, title="training stages")
    for stage in TRAINING_STAGES:
        stage.register(stages)
