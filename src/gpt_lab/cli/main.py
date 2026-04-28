from gpt_lab.utils.common import get_banner
from argparse import ArgumentParser

if __name__ == "__main__":
    get_banner(to_print=True)

    parser = ArgumentParser(description="Main entry point for GPT training, evaluation, monitoring and inference.")

    subparsers = parser.add_subparsers(dest="sub_command", help="Sub-commands for training, evaluation, monitoring and inference.")
    
    # TODO: think about the CLI

    # read, write env vars
    # trigger training, evaluation, monitoring, inference
    # view logs, metrics, results
    # launch interactive sessions (e.g. CLI or gradio interface)
    # manage models, datasets, configs
    # etc.


