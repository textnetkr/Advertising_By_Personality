import re
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
    try:
        # Data Load
        data = []
        with open(cfg.PATH.for_selling_ntnf, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip("\n|\r")))
        df = pd.DataFrame(data)

        df_temp = df.iloc[[956, 282, 2002, 2003, 2004], :].copy()
        print(df_temp.head())

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY
        # model = "gpt-3.5-turbo"
        model = "gpt-4"

        def generate_response(messages: list) -> str:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response["choices"][0]["message"]["content"]

        # GPT Extraction
        # start = 0
        # df_start = df.iloc[start:5]

        s_time = time.time()
        # print(df_start.head())
        # print(f"{start}번째 행부터 시작!")

        selling = []
        selling_sent = []
        market_info = []
        for k, i in enumerate(df_temp.iterrows()):
            # selling_point
            messages = [
                {
                    "role": "system",
                    "content": mp.mbti["SELLING_NTNF"],
                },
                {
                    "role": "user",
                    "content": f"""{i[1]['label']}\n위 문장에 적합한 소구점을 만들어줘.\n
                                소구점 : """,
                },
            ]

            sell_result = generate_response(messages)
            # print(f"{start + k}번째 행")
            print(f"{k}번째 행")
            print(f"원문 : {i[1]['label']}")
            print(f"답변 : {sell_result}")
            print("-" * 100)
            selling.append(sell_result)

            # selling_sent
            messages = [
                {
                    "role": "system",
                    "content": mp.mbti["SELLING_SENT_NTNF"],
                },
                {
                    "role": "user",
                    # "content": f"""{i[1]['label']}\n위 문장에 해당하는 마케팅 대상 : {i[1]['marketing_entity']}, 타겟 : {i[1]['marketing_target']}, 혜택 지급 조건 : {i[1]['benefit_conditions']}, 혜택 : {i[1]['benefits']}, 할인 수치 : {i[1]['discount_figure']}, 프로모션 품목 : {i[1]['promotional_items']}, 프로모션 장소 : {i[1]['promotional_place']}, 이벤트 기간 : {i[1]['event_period']}, 요일 정보 : {i[1]['dow_information']}, 시즌 정보 : {i[1]['season_information']}, 소구점 : {sell_result}을 반영하여 매력있는 광고 문구를 만들어줘.\n
                    # 소구점이 반영된 광고 문구 : """,
                    # "content": f"""{i[1]['label']}\n위 문장에 해당하는 요일 정보 : {i[1]['dow_information']}, 시즌 정보 : {i[1]['season_information']}, 소구점 : {sell_result}을 반영하여 매력있는 광고 문구를 만들어줘.\n
                    # 소구점이 반영된 광고 문구 : """,
                    "content": f"""{i[1]['label']}\n위 문장에 해당하는 소구점 : {sell_result}을 반영하여 매력있는 광고 문구를 만들어줘.\n
                                소구점이 반영된 광고 문구 : """,
                },
            ]

            sell_sent = generate_response(messages)
            # print(f"{start + k}번째 행")
            print(f"원문 : {i[1]['label']}")
            print(f"답변 : {sell_sent}")
            print("-" * 100)
            selling_sent.append(sell_sent)

            # marketing_info
            market_info.append(
                f"마케팅 대상 : {i[1]['marketing_entity']}, 타겟 : {i[1]['marketing_target']}, 혜택 지급 조건 : {i[1]['benefit_conditions']}, 혜택 : {i[1]['benefits']}, 할인 수치 : {i[1]['discount_figure']}, 프로모션 품목 : {i[1]['promotional_items']}, 프로모션 장소 : {i[1]['promotional_place']}, 이벤트 기간 : {i[1]['event_period']}, 요일 정보 : {i[1]['dow_information']}, 시즌 정보 : {i[1]['season_information']}"
            )

        math.factorial(100000)
        e_time = time.time()
        print(f"{e_time - s_time:.5f} sec")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # temp save
        # df_temp = df.iloc[start : start + len(selling)].copy()
        df_temp["selling"] = selling
        df_temp["sentence"] = selling_sent
        df_temp["market_info"] = market_info
        df_temp = df_temp[["label", "market_info", "selling", "sentence"]]
        df_temp.rename(
            {
                "label": "원문",
                "market_info": "마케팅 정보",
                "selling": "소구점",
                "sentence": "소구점이 반영된 광고 문구",
            },
            inplace=True,
        )
        fm.save(cfg.PATH.temp_save, df_temp)
        print("Save Done!")


if __name__ == "__main__":
    main()
