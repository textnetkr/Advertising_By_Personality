from os.path import abspath, splitext
from typing import Optional

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
        result = dict()
        input = dict()
        label = dict()

        input = tokenizer(
            [
                f"Text Generation: <type>{t1}<classified>{t2}<advertiser>{t3}<product>{t4}<product_detail>{t5}<purpose>{t6}<benefit>{t7}<period>{t8}<target>{t9}<season>{t10}<weather>{t11}<anniv>{t12}<selling_point>{t13}"
                for t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13 in zip(
                    e["type"],
                    e["classified"],
                    e["advertiser"],
                    e["product"],
                    e["product_detail"],
                    e["purpose"],
                    e["benefit"],
                    e["period"],
                    e["target"],
                    e["season"],
                    e["weather"],
                    e["anniv"],
                    e["selling_point"],
                )
            ],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # keys = [
        #     "type",
        #     "classified",
        #     "advertiser",
        #     "product",
        #     "product_detail",
        #     "purpose",
        #     "benefit",
        #     "period",
        #     "target",
        #     "season",
        #     "weather",
        #     "anniv",
        #     "selling_point",
        # ]
        # input = tokenizer(
        #     [
        #         "Text Generation: <type>{}<classified>{}<advertiser>{}<product>{}<product_detail>{}<purpose>{}<benefit>{}<period>{}<target>{}<season>{}<weather>{}<anniv>{}<selling_point>{}".format(
        #             *args
        #         )
        #         for args in zip(*[e[k] for k in keys])
        #     ],
        #     max_length=seq_len,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="np",
        # )

        label = tokenizer(
            [t + tokenizer.eos_token for t in e["label"]],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        result["input_ids"] = input["input_ids"]
        result["labels"] = label["input_ids"]

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
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    return data["train"], (data["test"] if is_eval else None)


# Write preprocessor code to run in batches.
def default_collator(data):
    return data
