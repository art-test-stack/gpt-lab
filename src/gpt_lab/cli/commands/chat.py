"""Chat commands do not import Torch or UI dependencies during parsing."""

from argparse import Namespace

from gpt_lab.cli.options import cache_dir, checkpoint_step, positive_int


def register(subparsers) -> None:
    parser = subparsers.add_parser('chat', help='Console completion and experimental Gradio app.',
                                  description='Use a local checkpoint in the console, or launch the experimental Gradio interface.')
    children = parser.add_subparsers(required=True, title='interfaces')
    console = children.add_parser('console', help='Interactive greedy completion from a local checkpoint.',
                                  description='Load a local base checkpoint and generate greedy completions. This is a base-model prompt loop, without an instruction chat template. Enter /exit or EOF to quit.')
    console.add_argument('--model-name', default='ic1', help='Saved model name.')
    console.add_argument('--run-name', default=None, help='Run name, latest, best, or -N; omitted selects the latest run.')
    console.add_argument('--model-dir', default=str(cache_dir() / 'models'), help='Model cache directory.')
    console.add_argument('--checkpoint-step', type=checkpoint_step, default=None, help='Step, latest, best, or -N; omitted selects latest.')
    console.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto', help='Inference device.')
    console.add_argument('--max-tokens', type=positive_int, default=64, help='Maximum new tokens per prompt.')
    console.set_defaults(_handler='gpt_lab.cli.commands.chat:run_console')
    app = children.add_parser('app', help='Launch the experimental Gradio web UI.',
                              description='Launch the existing experimental web UI. Requires Gradio, OpenAI and dotenv dependencies. The benchmark tab queries Hugging Face; remote chat uses OPENAI_BASE_URL and HF_TOKEN. Local web completion remains unsupported.')
    app.set_defaults(_handler='gpt_lab.cli.commands.chat:run_app')


def run_console(args: Namespace) -> None:
    from gpt_lab.workflows.chat import console
    console(args)


def run_app(args: Namespace) -> None:
    from gpt_lab.workflows.chat import app
    app(args)
