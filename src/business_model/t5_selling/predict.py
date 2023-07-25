import re
import time
import hydra
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from pshmodule.processing import processing as p
from pshmodule.utils import filemanager as fm


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # data load
    df = fm.load(cfg.PATH.predict)
    print(df.head())

    # tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(cfg.PATH.save_dir)

    # model
    model = T5ForConditionalGeneration.from_pretrained(cfg.PATH.save_dir)
    model.eval().to(device)

    inputs = [
        f"Text Generation: {i[1]['title']} {i[1]['content']}{tokenizer.eos_token}"
        for i in df.iterrows()
    ]

    encoded = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    sample = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **sample,
            temperature=0.7,
            do_sample=True,
            max_length=512,
            penalty_alpha=0.6,
            top_k=5,
            eos_token_id=tokenizer.eos_token_id,
        )

    # result
    r_list = [
        tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs
    ]
    df["t5_v1.3"] = r_list

    # save
    df.rename(
        columns={"no": "번호", "title": "INPUT_제목", "content": "INPUT_본문"},
        inplace=True,
    )
    print(df.head())
    fm.save(f"{cfg.PATH.temp_save}_1.xlsx", df)


if __name__ == "__main__":
    main()
