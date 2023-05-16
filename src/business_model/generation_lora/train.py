from tqdm import tqdm
import hydra
import torch
from transformers import (
    PreTrainedTokenizerFast,
    GPTNeoXForCausalLM,
    get_linear_schedule_with_warmup,
)
from custom_lora import lora
from dataloader import load, custom_default_collator

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.MODEL.name)

    # model
    model = GPTNeoXForCausalLM.from_pretrained(
        cfg.MODEL.name,
        load_in_8bit=True,  # bitsandbytes 8비트 양자화 모델로 load
        device_map="auto",  # 각 하위 모듈이 이동해야 하는 위치를 지정하는 맵. 사용 가능한 모든 GPU에서 모델을 고르게 분할.
    )
    # transfer lora model - get_peft_model
    model = lora(model)

    # dataloder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    train_dataloader = custom_default_collator(train_dataset, cfg.ARGS.batch_size)
    eval_dataloader = custom_default_collator(eval_dataset, cfg.ARGS.batch_size)

    # train
    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ARGS.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * cfg.ARGS.num_epochs),
    )

    # training and evaluation
    model = model.to(device)

    for epoch in range(cfg.ARGS.num_epochs):
        # train
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            print(f"loss : {loss}")
            total_loss += loss.float()
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
            print(f"loss : {loss}")
            eval_loss += loss.float()
            eval_preds.extend(
                tokenizer.batch_decode(
                    torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                    skip_special_tokens=True,
                )
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
        )
    # print accuracy
    correct = 0
    total = 0
    for pred, true in zip(eval_preds, eval_dataset["text_label"]):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total * 100
    print(f"{accuracy=} % on the evaluation dataset")
    print(f"{eval_preds[:10]=}")
    print(f"{eval_dataset['text_label'][:10]=}")

    # save
    model.save_pretrained(cfg.PATH.peft_model)


if __name__ == "__main__":
    main()
