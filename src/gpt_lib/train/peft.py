import torch
from torch import nn

class LoRA(nn.Module):
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.config = config

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if "weight" in name and param.ndim == 2:
                param.requires_grad = True
            self.lora_A.append(nn.Parameter(torch.randn((self.config.lora_rank, param.size(1))) * 0.01))
            self.lora_B.append(nn.Parameter(torch.randn((param.size(0), self.config.lora_rank)) * 0.01))
            
        self.lora_A = nn.Parameter(torch.randn((self.config.lora_rank, self.model.config.hidden_size)) * 0.01)
        self.lora_B = nn.Parameter(torch.randn((self.model.config.d_model, self.config.lora_rank)) * 0.01)

        model.register_forward_hook(self.forward)

    def forward(self, input_ids, *args, **kwargs):
        # Compute the LoRA update and add it to the original weights
        for (name, param), A, B in zip(self.model.named_parameters(), self.lora_A, self.lora_B):
            if param.requires_grad:
                lora_update = A @ B
                param = param + lora_update

        return self.model(input_ids, *args, **kwargs)


class PEFT:
    def __init__(self, model, config):
        self.model = model
        self.config = config


    @classmethod
    def from_config(cls, model, config):
        if config.peft_method == "lora":
            return LoRA(model=model, config=config)
        # elif config.peft_method == "prefix_tuning":
        #     from gpt_lib.train.peft_prefix import PrefixTuningPEFT
        #     return PrefixTuningPEFT(model=model, tokenizer=tokenizer, config=config)
        # elif config.peft_method == "adapter":
        #     from gpt_lib.train.peft_adapter import AdapterPEFT
        #     return AdapterPEFT(model=model, tokenizer=tokenizer, config=config)
        else:
            raise ValueError(f"Unsupported PEFT method: {config.peft_method}")