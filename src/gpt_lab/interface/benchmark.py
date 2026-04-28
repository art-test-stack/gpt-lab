import gradio as gr
from datasets import load_dataset
import transformers
from pydantic import BaseModel
from huggingface_hub import HfApi


class ResultEntry(BaseModel):
    models: list[str]
    results: dict[str, float]

    def get_results(self):
        assert len(self.results) == len(self.models)
        return self.results
    
    def add_result(self, model: str, result: float):
        self.results[model] = result

# TODO: HAVE TO BE CLEANED UP LATER (l18-27)
hgf_api = HfApi()
models = hgf_api.list_models(    
    filter="Text Generation",
    limit=100
)

models = list(models)
models = [models[i].id for i in range(len(models))]

def benchmark_interface():
    gr.Markdown("# Benchmarking Interface 🏆")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Settings")
            available_models = gr.Dropdown(
                models, 
                label="Available Models", 
                info="Select one or more models to benchmark", 
                multiselect=True,
                allow_custom_value=True
            )
            available_datasets = gr.Dropdown(
                ["SQuAD", "GLUE", "SuperGLUE", "WMT", "Custom"], 
                label="Available Datasets",
                info="Select the dataset to use for benchmarking",
                multiselect=True,
            )
            # benchmark_model_name = gr.Textbox(label="Model Name", value="gpt-4")
            benchmark_temperature = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Temperature")
            benchmark_max_tokens = gr.Number(label="Max Tokens", value=64)

        with gr.Column(scale=3):
            gr.Markdown("## Results")
            benchmark_results = gr.Textbox(label="Results", lines=20)
            results = {}
            def get_dataset(name):
                splits = ["test", "validation", "train"]
                for split in splits:
                    try:
                        ds = load_dataset(name, streaming=True, split=split)
                        return ds
                    except Exception as e:
                        print(f"Error loading dataset '{split}' split {name}: {e}")
                return None
            
            def run_benchmark(models, datasets, temperature, max_tokens):
                for dataset in datasets:
                    results[dataset] = ResultEntry(
                        models=models,
                        results={}
                    )
                    for model in models:
                        model = transformers.pipeline(
                            "text-generation",
                            model=model,
                            temperature=temperature,
                            max_new_tokens=max_tokens
                        )
                        prediction = model.predict(dataset[0])

                        results[dataset].add_result(model, prediction == dataset)
                    for model in models:
                        model = transformers.pipeline(
                            "text-generation",
                            model=model,
                            temperature=temperature,
                            max_new_tokens=max_tokens
                        )
                        prediction = model.predict(dataset[0])

                        results[dataset].add_result(model, prediction == dataset)

            gr.Button("Run Benchmark").click(
                run_benchmark,
                [available_models, available_datasets, benchmark_temperature, benchmark_max_tokens],
                benchmark_results
            )
