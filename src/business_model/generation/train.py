import hydra
import torch
import torch.nn as nn
from transformers import (
    PreTrainedTokenizerFast,
    GPTNeoXForCausalLM
)
from custom import (
    post_processing,
    lora
)


device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.MODEL.name)
    print(f"tokenizer : {tokenizer}")
    
    # model
    model = GPTNeoXForCausalLM.from_pretrained(
        cfg.MODEL.name,
        load_in_8bit=True,
        device_map='auto'
    )

    # lora
    model = post_processing(model)
    model_lora = lora(model)
    print(f"model_lora : {model_lora}")


if __name__ == "__main__":
    main()