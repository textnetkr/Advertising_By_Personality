import hydra
import wandb
import torch
import transformers
from transformers import (
    PreTrainedTokenizerFast,
    GPTNeoXForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from custom import post_processing, lora
from dataloader import load


device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.MODEL.name)

    # model
    model = GPTNeoXForCausalLM.from_pretrained(
        cfg.MODEL.name, load_in_8bit=True, device_map="auto"
    )

    # transfer lora model - get_peft_model
    model = post_processing(model)
    model_lora = lora(model)
    print(f"model_lora : {model_lora}")

    # dataloder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # wandb
    wandb.init(
        project=cfg.ETC.project,
        entity=cfg.ETC.entity,
        name=cfg.ETC.name,
    )

    # trainer
    args = TrainingArguments(
        **cfg.TRAININGARGS,
    )

    trainer = Trainer(
        model=model_lora,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()

    trainer.save_model(cfg.PATH.save_dir)


if __name__ == "__main__":
    main()
