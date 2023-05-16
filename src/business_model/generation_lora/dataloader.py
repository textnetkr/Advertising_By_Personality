from os.path import abspath, splitext
from typing import Optional
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset, logging

logging.set_verbosity(logging.ERROR)


def load(
    tokenizer,
    seq_len,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    def _tokenize_function(e):
        result = tokenizer(
            [
                f"""Below is an instruction that describes a task, paired with an input that provides further context.\n
                아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n
                Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n
                ### Instruction(명령어):\n다음 정보를 활용해서 광고 문구를 생성해줘.\n\n
                ### Input(입력):\n마케팅 주체: {t1}, 마케팅 대상: {t2}, 혜택 조건: {t3}, 할인 수치: {t4}, 프로모션 품목: {t5}, 이벤트 기간: {t6}, 시즌 정보: {t7}\n\n
                ### Response(응답):\n{t8}
                """
                for t1, t2, t3, t4, t5, t6, t7, t8 in zip(
                    e["marketing_entity"],
                    e["marketing_target"],
                    e["benefit_conditions"],
                    e["discount_figure"],
                    e["promotional_items"],
                    e["event_period"],
                    e["season_information"],
                    e["label"],
                )
            ],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        del result["token_type_ids"]
        return result

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""),
        data_files=datafiles,
        split=train_test_split,
    )

    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    data = data.map(
        _tokenize_function,
        batched=True,
        num_proc=worker,
        remove_columns=data["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    return data["train"], (data["test"] if is_eval else None)


# Write preprocessor code to run in batches.
def custom_default_collator(data, batch_size):
    dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        pin_memory=True,
    )
    return dataloader
