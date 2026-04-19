import sys, os, load_dotenv
load_dotenv.load_dotenv()
try:
    import gpt_lab
except ImportError as e:
    print("Import Error:", e)
    if os.environ["DEVELOPMENT"] == "1":
        print("Development environment detected. Attempting to adjust sys.path.")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.append(parent_dir)
        print("Updated System Path:", sys.path)
        import gpt_lab
    else:
        raise e

from gpt_lab.interface.chat import chatapp_interface
from gpt_lab.interface.benchmark import benchmark_interface
from gpt_lab.utils.common import get_banner
import gradio as gr
from pydantic import BaseModel


class ChatSettings(BaseModel):
    temperature: float = 1.0
    max_tokens: int = 64
    model_name: str | None = None

class ModelSettings(BaseModel):
    nb_parameters_min: int = "1B"
    nb_parameters_max: int = "175B"


with gr.Blocks(title="GPT-lib") as app:
    with gr.Tab("Chat"):
        chatapp_interface()

    with gr.Tab("Benchmark"):
        benchmark_interface()

    with gr.Tab("Training"):
        gr.Markdown("# Training Interface 🏋️‍♂️")


if __name__ == "__main__":
    print("Launching GPT-lib Interface...")
    print(get_banner())
    app.launch()