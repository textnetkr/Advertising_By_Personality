from tqdm import tqdm
import hydra
import torch
from transformers import (
    PreTrainedTokenizerFast,
    GPTNeoXForCausalLM,
    get_linear_schedule_with_warmup,
)
from custom_lora import lora
from dataloader import load, custom_collator
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.MODEL.name)

    # model
    model = GPTNeoXForCausalLM.from_pretrained(cfg.MODEL.name)
    # transfer lora model - get_peft_model
    model = lora(model)

    # dataloder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    train_dataloader = custom_collator(train_dataset, cfg.ARGS.batch_size)
    eval_dataloader = custom_collator(eval_dataset, cfg.ARGS.batch_size)

    # wandb
    wandb.init(
        project=cfg.WANDB.project,
        entity=cfg.WANDB.entity,
        name=cfg.WANDB.name,
    )

    # train
    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ARGS.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * cfg.ARGS.num_epochs),
    )

    # training and evaluation
    print(f"epoch : {cfg.ARGS.num_epochs}")
    model = model.to(device)
    for epoch in range(cfg.ARGS.num_epochs):
        # train
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # evaluate
        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss
            eval_preds.extend(
                tokenizer.batch_decode(
                    torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                    skip_special_tokens=True,
                )
            )

        eval_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_loss)
        train_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_loss)
        print(f"{epoch=}: {train_ppl=} {train_loss=} {eval_ppl=} {eval_loss=}")
        wandb.log({"train/loss": train_loss}, step=epoch)
        wandb.log({"eval/loss": eval_loss}, step=epoch)

    # save
    model.save_pretrained(cfg.PATH.peft_model)


if __name__ == "__main__":
    main()
