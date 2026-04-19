import gradio as gr

from gpt_lab.interface.config import ConfigChat
from gpt_lab.interface.completion import ModelCompletion
import time, os

# TODO: HAVE TO BE CLEANED UP LATER (l7-18)
remote_models_base = [
    "moonshotai/Kimi-K2-Thinking",
    "meta-llama/Llama-3.3-70B-Instruct",
    "MiniMaxAI/MiniMax-M2:novita",
]
local_models_base = [
    "local-model",
]
def choose_model_base(use_local):
    return local_models_base if use_local else remote_models_base
reasoning_choices = ["minimal", "low", "medium", "high"]

def chatapp_settings():
    gr.Markdown("# Settings")
    local_model = gr.Checkbox(label="Use Local Models", value=False, interactive=True)
    initial_choices = choose_model_base(False)
    models = gr.State(initial_choices)

    model_name = gr.Dropdown(
        choices=initial_choices, 
        label="Model Name", 
        value=initial_choices[0] if initial_choices else None,
        allow_custom_value=True,
        interactive=True
    )

    def _on_local_model_change(use_local):
        choices = choose_model_base(use_local)
        val = choices[0] if choices else None
        return choices, gr.Dropdown(choices=choices, value=val, allow_custom_value=not use_local)

    local_model.change(
        fn=_on_local_model_change,
        inputs=[local_model],
        outputs=[models, model_name],
    )
    temperature = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Temperature", interactive=True)
    max_tokens = gr.Slider(100, 10000, value=500, step=10, label="Max Tokens", interactive=True)
    seed = gr.Slider(0, 1000, value=42, step=1, label="Random Seed", interactive=True)
    # time_out = gr.Slider(0.1, 1000, value=100, step=10, label="Timeout", interactive=True)

    reasoning_effort = gr.Dropdown(
        reasoning_choices, 
        label="Reasoning Effort", 
        value=reasoning_choices[-1], 
        interactive=True
    )
    remote_completion = gr.Checkbox(label="Use Remote Completion", value=False)
    stream = gr.Checkbox(label="Stream Responses", value=True)
    model_config = gr.State(
        ConfigChat(
            model_name=model_name.value,
            temperature=temperature.value,
            max_tokens=max_tokens.value,
            remote_completion=remote_completion.value,
            seed=int(seed.value),
            stream=stream.value,
            reasoning_effort=reasoning_effort.value,
    ))

    def _update_config(
            model_name_value, temperature_value, max_tokens_value,
            remote_completion_value, seed_value, stream_value,
            reasoning_value
        ):
        try:
            seed_int = int(seed_value)
        except Exception:
            gr.Warning("Seed value must be an integer. Using default seed 42.")
            seed_int = 42

        new_cfg = ConfigChat(
            model_name=model_name_value,
            temperature=temperature_value,
            max_tokens=max_tokens_value,
            remote_completion=remote_completion_value,
            seed=seed_int,
            stream=stream_value,
            reasoning_effort=reasoning_value,
        )
        print("Updated model configuration:", new_cfg)
        return new_cfg

    save_button = gr.Button("Save Settings")

    save_button.click(
        fn=_update_config,
        inputs=[model_name, temperature, max_tokens, remote_completion, seed, stream, reasoning_effort],
        outputs=[model_config],
    )
    return model_config

def chatapp_box(model_config: ConfigChat):
    gr.Markdown("# Chat Interface 💬")
    
    print("Model configuration for chat app:", model_config)

    def update_context(context, history, cfg: gr.State):
        model_name = cfg.model_name
        context_based = f"You are a helpful assistant called {model_name.split('/')[-1]}, made by {model_name.split('/')[0]}."

        if context:
            context = context_based + " Use the following context to answer the user's questions.\n\nContext:\n" + context
        else:
            context = context_based
        updated_context = {"role": "system", "content": context}
        has_system = any(h.get("role") == "system" for h in history)
        history = list(map(lambda h: updated_context if h.get("role") == "system" else h, history))
        history  = history if has_system else [updated_context] + history
        print("Updated context in history:", history)
        return history
    
    context = gr.Textbox(
        placeholder="Enter the context here...", 
        label="Context", 
        lines=2,
        submit_btn=True
    )
    history = gr.State([])
    context.submit(update_context, inputs=[context, history, model_config], outputs=history)

    chatbot = gr.Chatbot(
        # history,
        type="messages", 
        # height=300, 
        show_copy_button=True,
    )
    msg_interactive = gr.State(True)
    default_message = "What is the capital of France?" if os.environ.get("DEVELOPMENT") == "1" else ""
    msg = gr.Textbox(
        default_message,
        placeholder="Enter your message here...", 
        label="Your Message", 
        lines=2,
        show_label=True,
        show_copy_button=True,
        submit_btn=True,
        interactive=msg_interactive,
    )
    show_reasoning = gr.Checkbox(label="Show Reasoning", value=False)


    def submit_user_message(user_message, chat_history, cfg: ConfigChat, show_reasoning=False):
        # Append user message to UI history
        start_time = time.time()

        # Build model messages: system + user only
        messages_for_model = [
            msg for msg in chat_history
        ]

        chat_history = chat_history + [{"role": "user", "content": user_message}]

        msg_itc = False
        yield "", msg_itc, chat_history, chat_history

        reasoning_response = {
            "role": "assistant", 
            "content": "",
            "metadata": {"title": "_Thinking_ step-by-step", "id": 0, "status": "pending"}
        }
        response = {
            "role": "assistant", 
            "content": "",
            "metadata": {"status": "pending"}
        }
        if show_reasoning:
            chat_history.append(reasoning_response)
        
        chat_history.append(response)

        yield "", msg_itc, chat_history, chat_history
        
        model = ModelCompletion(cfg)

        for chunk in model.predict(user_message, messages_for_model):
            if chunk.get("finish_reason"):
                if not reasoning_response["metadata"]["status"] == "done":
                    reasoning_response["metadata"]["status"] = "done"
                    reasoning_response["metadata"]["duration"] = time.time() - start_time
                response["content"] = chunk["content"]

            elif show_reasoning and chunk.get("reasoning"):
                reasoning_response["content"] = chunk["reasoning"]
            
            yield "", msg_itc, chat_history, chat_history

        response["metadata"]["status"] = "done"
        if not show_reasoning:
            response["metadata"]["duration"] = time.time() - start_time

        print("Chat history after completion:", chat_history)
        msg_itc = True
        yield "", msg_itc, chat_history, chat_history

    
    msg.submit(
        submit_user_message,
        inputs=[msg, history, model_config, show_reasoning],
        outputs=[msg, msg_interactive, history, chatbot]
    )
    clear = gr.Button("New conversation")
    clear.click(
        lambda _: ["", [], [], "", True],
        [],
        [context, history, chatbot,  msg, msg_interactive],
    )
    return chatbot

def chatapp_interface():
    with gr.Row():
        with gr.Column(scale=1, render=True):
            config = chatapp_settings()

        with gr.Column(scale=3, variant="panel"):
            chatbot = chatapp_box(config)