import re
import time
import hydra
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from pshmodule.processing import processing as p
from pshmodule.utils import filemanager as fm


device = torch.device("cuda")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # data load
    df = fm.load(cfg.PATH.test)
    print(df.head())

    # tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(cfg.PATH.save_dir)

    # model
    model = T5ForConditionalGeneration.from_pretrained(cfg.PATH.save_dir)
    model.eval().to(device)

    inputs = [
        f"Text Generation: <type>{i[1]['type']}<classified>{i[1]['classified']}<advertiser>{i[1]['advertiser']}<product>{i[1]['product']}<product_detail>{i[1]['product_detail']}<purpose>{i[1]['purpose']}<benefit>{i[1]['benefit']}<period>{i[1]['period']}<target>{i[1]['target']}<season>{i[1]['season']}<weather>{i[1]['weather']}<anniv>{i[1]['anniv']}<selling_point>{i[1]['selling_point']}"
        for i in df.iterrows()
    ]

    encoded = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    sample = {k: v.to(device) for k, v in encoded.items()}

    # repetition_penalty: 1.1
    with torch.no_grad():
        outputs = model.generate(
            **sample,
            temperature=cfg.PREDICT.temperature,
            max_length=512,
            do_sample=cfg.PREDICT.do_sample,
            penalty_alpha=cfg.PREDICT.penalty_alpha,
            top_k=cfg.PREDICT.top_k,
            repetition_penalty=cfg.PREDICT.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    # result
    r_list1 = [
        tokenizer.decode(output, skip_special_tokens=True).strip().replace("\\", "\\\\")
        for output in outputs
    ]
    df["repetition_penalty_1.1"] = r_list1

    # max_new_tokens 50
    with torch.no_grad():
        outputs = model.generate(
            **sample,
            temperature=cfg.PREDICT.temperature,
            do_sample=cfg.PREDICT.do_sample,
            penalty_alpha=cfg.PREDICT.penalty_alpha,
            top_k=cfg.PREDICT.top_k,
            repetition_penalty=cfg.PREDICT.repetition_penalty,
            max_new_tokens=cfg.PREDICT.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    # result
    r_list2 = [
        tokenizer.decode(output, skip_special_tokens=True).strip().replace("\\", "\\\\")
        for output in outputs
    ]
    df["max_new_tokens_50"] = r_list2

    # max_length 100
    with torch.no_grad():
        outputs = model.generate(
            **sample,
            temperature=cfg.PREDICT.temperature,
            max_length=cfg.PREDICT.max_length,
            do_sample=cfg.PREDICT.do_sample,
            penalty_alpha=cfg.PREDICT.penalty_alpha,
            top_k=cfg.PREDICT.top_k,
            repetition_penalty=cfg.PREDICT.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    # result
    r_list3 = [
        tokenizer.decode(output, skip_special_tokens=True).strip().replace("\\", "\\\\")
        for output in outputs
    ]
    df["max_length_100"] = r_list3

    # save
    fm.save(cfg.PATH.result_save + ".xlsx", df)


if __name__ == "__main__":
    main()
