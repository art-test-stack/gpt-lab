"""Inference and UI setup are deferred until a chat command is executed."""


def console(args):
    import torch
    from gpt_lab.model.checkpoint import build_model
    from gpt_lab.model.wrapper import Engine
    from gpt_lab.utils.distributed import get_dist_info, get_device_type
    from gpt_lab.utils.schemas import GenerationConfig

    device = get_device_type() if args.device == 'auto' else args.device
    model, tokenizer, _, _ = build_model(
        model_name=args.model_name, run_name=args.run_name, model_cachedir=args.model_dir,
        step=args.checkpoint_step if args.checkpoint_step is not None else 'latest',
        phase='eval', dist_info=get_dist_info(device_type=device),
    )
    engine = Engine(model=model, tokenizer=tokenizer)
    print('Local greedy completion. Enter /exit or press Ctrl-D to quit.')
    while True:
        try:
            prompt = input('You: ')
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.strip() == '/exit':
            return
        if not prompt.strip():
            continue
        tokens = tokenizer.encode(prompt, prepend_bos=True)
        available = model.config.max_context - len(tokens)
        if available <= 0:
            print('Prompt exceeds the model context; enter a shorter prompt.')
            continue
        with torch.inference_mode():
            output = engine.generate(
                torch.tensor([tokens], dtype=torch.long, device=engine.device),
                generation_config=GenerationConfig(max_length=min(args.max_tokens, available)),
            )
        print('Model: ' + tokenizer.decode(output[0]))


def app(args):
    # The old script imported `load_dotenv` as a module. python-dotenv exposes
    # this function from `dotenv` instead.
    from dotenv import load_dotenv
    load_dotenv()
    import gradio as gr
    from gpt_lab.interface.chat import chatapp_interface
    from gpt_lab.interface.benchmark import benchmark_interface
    from gpt_lab.utils.common import get_banner

    with gr.Blocks(title='GPT-lib') as application:
        with gr.Tab('Chat'):
            chatapp_interface()
        with gr.Tab('Benchmark'):
            benchmark_interface()
        with gr.Tab('Training'):
            gr.Markdown('# Training Interface 🏋️‍♂️')
    print('Launching GPT-lib Interface...')
    get_banner(to_print=True)
    application.launch()
