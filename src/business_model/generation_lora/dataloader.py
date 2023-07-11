from os.path import abspath, splitext
from typing import Optional
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset, logging

logging.set_verbosity(logging.ERROR)


def load(
    tokenizer,
    seq_len: int,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    shuffle_seed: Optional[int] = None,
):
    """
    Method to make data into hugging face dataset.

    Args:
        tokenizer (_type_): _description_
        seq_len (int): Sequence Length for Tokenization.
        train_data_path (str): train jsonl data file path.
        eval_data_path (Optional[str], optional): eval jsonl data file path. Defaults to None.
        train_test_split (Optional[float], optional): The ratio between train data and test data. Defaults to None.
        worker (int, optional): Multi parallel processing. Defaults to 1.
        shuffle_seed (Optional[int], optional): shuffle ratio. Defaults to None.
    """

    def _tokenize_function(e):
        result = tokenizer(
            [
                f"""Below is an instruction that describes a task, paired with an input that provides further context.\n
                아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n
                Write a response that appropriately completes the request.\n
                요청을 적절히 완료하는 응답을 작성하세요.\n\n
                ### Instruction(명령어):\n너는 판매 촉진을 위한 마케팅 문구를 만드는 카피라이터야.\n
                    마케팅 주체, 타겟, 혜택 지급 조건, 혜택, 할인 수치, 프로모션 품목, 프로모션 장소, 이벤트 기간, 요일 정보, 시즌 정보, 소구점으로 광고 문구를 생성할거야.\n\n
                ### Input(입력):\nNT 성향 문구 : {t1}, NF 성향 문구: {t2}, 마케팅 주체: {t3}, 타겟: {t4}, 혜택 지급 조건: {t5}, 혜택: {t6}, 할인 수치: {t7}, 프로모션 품목: {t8}, 프로모션 장소: {t9}, 이벤트 기간: {t10}, 요일 정보: {t11}, 시즌 정보: {t12}, 소구점: {t13}\n\n
                ### Response(응답):\n{t14}
                """
                for t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 in zip(
                    e["NT"],
                    e["NF"],
                    e["marketing_entity"],
                    e["marketing_target"],
                    e["benefit_conditions"],
                    e["benefits"],
                    e["discount_figure"],
                    e["promotional_items"],
                    e["promotional_place"],
                    e["event_period"],
                    e["dow_information"],
                    e["season_information"],
                    e["solicitation_point"],
                    e["label"],
                )
            ],
            max_length=seq_len + 1,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        del result["token_type_ids"]

        return {
            "input_ids": result.input_ids[:-1],
            "attention_mask": result.attention_mask[:-1],
            "labels": result.input_ids[1:],
        }

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
def custom_collator(data, batch_size):
    dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        pin_memory=True,
    )
    return dataloader
