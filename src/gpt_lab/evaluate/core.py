"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794
Adapted from https://github.com/karpathy/nanochat/blob/master/nanochat/core_eval.py
Copyright (c) 2025 Andrej Karpathy
Licensed under the MIT License; see LICENSES/nanochat-MIT.txt.
Modified for GPT-Lab by Arthur Testard.

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""

import os, csv, time, json, yaml, shutil, random, zipfile, tempfile

from jinja2 import Template
import torch
import torch.distributed as dist

from gpt_lab.utils.default import DATA_DIR
from gpt_lab.utils.common import print0, download_file_with_lock

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_rows(model, input_ids, **model_kwargs):
    """Return full logits and autoregressive token losses for CORE rows."""
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    return model(input_ids, target_ids, reduction='none', **model_kwargs)


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    outputs = forward_rows(model, input_ids)
    losses = outputs.loss.view_as(input_ids)

    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.logits.argmax(dim=-1)
    return losses, predictions


def prepare_example(idx, tokenizer, data, task_meta, max_tokens=None):
    """Render, tokenize and crop each row, returning its scored token span."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    if max_tokens is not None:
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    return tokens, start_idxs, end_idxs


def choice_nll(losses, start_idxs, end_idxs):
    return [losses[i, si-1:ei-1].mean().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]


def first_minimum(mean_losses):
    return mean_losses.index(min(mean_losses))


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta, *, score_multiple_choice=None):
    """Evaluate a single example, return True if correct, False otherwise."""
    tokens, start_idxs, end_idxs = prepare_example(
        idx, tokenizer, data, task_meta, getattr(model, 'max_seq_len', None),
    )
    task_type = task_meta['task_type']
    item = data[idx]
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    if task_type == 'multiple_choice' and score_multiple_choice is not None:
        mean_losses = score_multiple_choice(model, tokens, start_idxs, end_idxs, pad_token_id, device)
        return first_minimum(mean_losses) == item['gold']

    # Stack up all the sequences into a batch
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    losses, predictions = forward_model(model, input_ids)

    # See if the losses/predictions come out correctly
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        mean_losses = choice_nll(losses, start_idxs, end_idxs)
        pred_idx = first_minimum(mean_losses)
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def rank_example_indices(num_examples):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return range(rank, num_examples, world_size)


def evaluate_task(model, tokenizer, data, device, task_meta, *, score_multiple_choice=None):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in rank_example_indices(len(data)):
        is_correct = evaluate_example(
            idx, model, tokenizer, data, device, task_meta,
            score_multiple_choice=score_multiple_choice,
        )
        correct[idx] = float(is_correct)
    # sync results across all the processes if running distributed
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = correct.mean().item()
    return mean_correct

# Adapted from https://github.com/karpathy/nanochat/blob/master/scripts/base_eval.py
COLS = {
    "id": 4,
    "task": 32,
    "type": 17,
    "fewshot": 9,
    "acc": 10,
    "cacc": 17,
    "time": 10,
}

CORE_HEADERS = (
    f"{'Id':^{COLS['id']}} | "
    f"{'Task name':^{COLS['task']}} | "
    f"{'Task type':^{COLS['type']}} | "
    f"{'Fewshot':^{COLS['fewshot']}} | "
    f"{'Accuracy':^{COLS['acc']}} | "
    f"{'Centered accuracy':^{COLS['cacc']}} | "
    f"{'Time (s)':^{COLS['time']}}"
)

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = DATA_DIR / "core"
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def load_core_bundle():
    base_dir = DATA_DIR / "core"
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    return data_base_path, tasks, random_baselines


def load_core_task(data_base_path, task, max_per_task):
    task_meta = {
        'task_type': task['icl_task_type'],
        'dataset_uri': task['dataset_uri'],
        'num_fewshot': task['num_fewshot'][0],
        'continuation_delimiter': task.get('continuation_delimiter', ' ')
    }
    data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    # Shuffle for consistent subsampling when using max_per_task
    random.Random(1337).shuffle(data)
    if max_per_task > 0:
        data = data[:max_per_task]
    return data, task_meta


def evaluate_core(model, tokenizer, device, max_per_task=-1, *, score_multiple_choice=None):
    """Evaluate a base model on CORE, with an optional multiple-choice scorer."""
    data_base_path, tasks, random_baselines = load_core_bundle()
    results = {}
    centered_results = {}
    print0(f"Evaluating CORE benchmark with {len(tasks)} tasks...")
    print0(CORE_HEADERS)
    print0("-" * len(CORE_HEADERS))
    for i, task in enumerate(tasks):
        start_time = time.time()
        label = task['label']
        data, task_meta = load_core_task(data_base_path, task, max_per_task)
        print0(
            f"{i:^{COLS['id']}} | "
            f"{label:^{COLS['task']}} | "
            f"{task_meta['task_type']:^{COLS['type']}} | "
            f"{(str(task_meta['num_fewshot']) + '-shot'):^{COLS['fewshot']}} |",
            end=""
        )

        accuracy = evaluate_task(
            model, tokenizer, data, device, task_meta, score_multiple_choice=score_multiple_choice,
        )
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(
            f"{accuracy:^{COLS['acc']}.6f} | "
            f"{centered_result:^{COLS['cacc']}.6f} | "
            f"{elapsed:^{COLS['time']}.2f}"
        )
        results[label] = dict(
            accuracy=accuracy, 
            num_examples=len(data), 
            random_baseline=random_baseline,
            centered_result=centered_result,
            elapsed_time=elapsed,
        )
        if os.getenv("DEVELOPMENT", "0") == "1":
            break

    accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
    time_elapsed = sum(r['elapsed_time'] for r in results.values())
    log_dict = {}
    log_dict["core/accuracy"] = accuracy
    log_dict["core/core"] = sum(centered_results.values()) / len(centered_results)
    log_dict["core/max_per_task"] = max_per_task
    log_dict["all_core_results"] = results
    log_dict["core/step_time_ms"] = time_elapsed * 1000
    return log_dict
