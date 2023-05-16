from peft import LoraConfig, get_peft_model


def lora(model):
    """
    Apply LoRA
    PeftModel을 load하고 peft의 get_peft_model 함수를 사용하여 낮은 순위 어탭터를 사용하도록 지정.
    """

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model
