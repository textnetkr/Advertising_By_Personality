import hydra
import json
import time
import math
import mbti_prompt as mp
import pandas as pd
import openai
from pshmodule.utils import filemanager as fm


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # try:
    # Data Load
    data = []
    with open(cfg.PATH.for_selling, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    df = pd.DataFrame(data)

    # OpenAI Api Key
    openai.api_key = cfg.OPENAI.OPENAI_API_KEY
    model = "gpt-4"

    # $$7.64 - $6.61
    def generate_response(messages: list) -> str:
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return response["choices"][0]["message"]["content"]

    # GPT Extraction
    predict = []
    start = 3500
    df_start = df.iloc[start:3510]

    s_time = time.time()
    print(df_start.head())
    print(f"{start}번째 행부터 시작!")

    for k, i in enumerate(df_start.iterrows()):
        messages = [
            {
                "role": "system",
                "content": mp.mbti["SELLING_POINT"],
            },
            {
                "role": "user",
                "content": f"""{i[1]['label']}\n위 문장에 적합한 소구점을 찾고, 찾아낸 소구점을 반영해서 매력적인 광고 문구로 \n을 기준으로 만들어줘.\n
                            소구점 : \n
                            소구점이 반영된 광고 문구 : """,
            },
        ]

        result = generate_response(messages)
        print(f"{start + k}번째 행")
        print(f"원문 : {i[1]['label']}")
        print(f"답변 : {result}")
        print("-" * 100)
        predict.append(result)

    math.factorial(100000)
    e_time = time.time()
    print(f"{e_time - s_time:.5f} sec")

    # temp save
    df["predict"] = predict
    fm.save(cfg.PATH.temp_save, df)

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     print(f"{start + k}번째 행에서 오류!")
    # finally:
    #     # Processing
    #     df_temp = df.iloc[start : start + len(predict)].copy()
    #     df_temp["mbti"] = predict
    #     df_temp["mbti"] = df_temp.mbti.str.replace("NT 성향 문구 : ", "")
    #     df_temp["mbti"] = df_temp.mbti.str.replace("NT 성향 문구: ", "")
    #     df_temp["mbti"] = df_temp.mbti.str.replace("NF 성향 문구 : ", "")
    #     df_temp["mbti"] = df_temp.mbti.str.replace("NF 성향 문구: ", "")
    #     # df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("\n\n", "\n")
    #     # df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("\n", ",\n")
    #     # df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace(",,", ",")
    #     # df_split_mbti = df_temp.mbti_temp.str.split(",\n", expand=True)
    #     # df_split_mbti_true = df_split_mbti[df_split_mbti[0].str.contains(",\n")]
    #     # df_split_mbti_true = df_split_mbti_true[0].str.split(",\n", expand=True)
    #     # df_split_mbti_false = df_split_mbti[~df_split_mbti[0].str.contains(",\n")]
    #     # df_split_mbti_temp = pd.concat([df_split_mbti_false, df_split_mbti_true])
    #     # df_split_mbti_temp.rename(
    #     #     columns={
    #     #         0: "NT",
    #     #         1: "NF",
    #     #     },
    #     #     inplace=True,
    #     # )
    #     # df_result = pd.concat([df_split_mbti_temp, df_temp], axis=1)
    #     df_result = df_temp.copy()
    #     df_result.sort_index(ascending=True, inplace=True)
    #     df_result.fillna("없음", inplace=True)

    #     # save
    #     temp_dict = [
    #         {
    #             "mbti": row["mbti"].strip(),
    #             "marketing_entity": row["marketing_entity"].strip(),
    #             "marketing_target": row["marketing_target"].strip(),
    #             "benefit_conditions": row["benefit_conditions"].strip(),
    #             "benefits": row["benefits"].strip(),
    #             "discount_figure": row["discount_figure"].strip(),
    #             "promotional_items": row["promotional_items"].strip(),
    #             "promotional_place": row["promotional_place"].strip(),
    #             "event_period": row["event_period"].strip(),
    #             "dow_information": row["dow_information"].strip(),
    #             "season_information": row["season_information"].strip(),
    #             "solicitation_point": row["solicitation_point"].strip(),
    #             "type": row["type"],
    #             "label": row["label"].strip(),
    #         }
    #         for _, row in df_result.iterrows()
    #     ]
    #     with open(eval(f"cfg.PATH.ext_gpt_mbti"), "a", encoding="utf-8") as f:
    #         for line in temp_dict:
    #             json_record = json.dumps(line, ensure_ascii=False)
    #             f.write(json_record + "\n")
    #     print("Save Done!")


if __name__ == "__main__":
    main()
