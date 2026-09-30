# Third-party notices

GPT-Lab's original code is distributed under the MIT License in `LICENSE`.
The components listed below include or are derived from third-party work and
remain subject to their original license terms.

## nanochat

- Project: [karpathy/nanochat](https://github.com/karpathy/nanochat)
- License: MIT
- Copyright: Copyright (c) 2025 Andrej Karpathy
- License copy: [LICENSES/nanochat-MIT.txt](LICENSES/nanochat-MIT.txt)
- Affected material: `src/gpt_lab/evaluate/core.py`, portions of the training,
  model, configuration, checkpoint, data, and utility code that identify
  nanochat as their source, and the `nanochat-tasks` Git submodule.

## modded-nanogpt

- Project: [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- License: MIT
- Copyright: Copyright (c) 2024 Keller Jordan
- License copy: [LICENSES/modded-nanogpt-MIT.txt](LICENSES/modded-nanogpt-MIT.txt)
- Affected material: `src/gpt_lab/optim/kernels/muon.py`.

## plainLM

- Project: [Niccolo-Ajroldi/plainLM](https://github.com/Niccolo-Ajroldi/plainLM)
- License: MIT
- Copyright: Copyright (c) 2024 Niccolò Ajroldi
- License copy: [LICENSES/plainLM-MIT.txt](LICENSES/plainLM-MIT.txt)
- Affected material: portions of `scripts/train_base.py` identified there as
  adapted from plainLM.

## pytorch-optimizer

- Project: [jettify/pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)
- License: Apache License 2.0
- License copy: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)
- Affected material:
  - `src/gpt_lab/optim/kernels/shampoo.py`
  - `src/gpt_lab/optim/kernels/adahessian.py`

These files carry notices that they were modified for GPT-Lab.

## Hugging Face Transformers and Nanotron

- Projects:
  - [huggingface/transformers](https://github.com/huggingface/transformers)
  - [huggingface/nanotron](https://github.com/huggingface/nanotron)
- License: Apache License 2.0
- Transformers copyright: Copyright 2018- The Hugging Face team
- License copy: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)
- Affected material: portions of `scripts/train_base.py` identified there as
  adapted from these projects.

This notice is informational and does not replace or modify the referenced
license terms.
