import torch
from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
    get_peft_model
)


def post_processing(model):
    # Post-processing on the model
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    # class CastOutputToFloat(nn.Sequential):
    #     def forward(self, x): return super().forward(x).to(torch.float32)

    # model.lm_head = CastOutputToFloat(model.lm_head)
    # print(f"model : {model}")

    return model


def lora(model):
    # Apply LoRA
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    return model