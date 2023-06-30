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
    start = 0
    sell = []
    sell_title = []
    sell_content = []
    market_info = []

    try:
        # Data Load
        data = []
        with open(cfg.PATH.for_selling_ntnf, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip("\n|\r")))
        df = pd.DataFrame(data)
        df_ref = fm.load(cfg.PATH.EXT_GPT_SELL)

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY
        model = "gpt-3.5-turbo"

        def generate_response(messages: list) -> str:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                # temperature=temperature,
            )
            return response["choices"][0]["message"]["content"]

        # GPT Extraction
        start = df_ref.shape[0]
        df_start = df.iloc[start:].copy()
        s_time = time.time()
        print(f"start: {start}")
        print(df_start.head())
        print(f"{start}번째 행부터 시작!")

        for k, i in enumerate(df_start.iterrows()):
            # selling_point
            temp = i[1]["label"].split("\\\\")
            title = temp[0]
            # content = temp[1]
            content = temp[1].replace("\\", " ")
            messages = [
                {
                    "role": "system",
                    "content": mp.mbti["SELLING_NTNF"],
                },
                {
                    "role": "user",
                    "content": f"""'''제목 : {title}, 본문 : {content}''' 오직 소구점 찾는 방법에 있는 내용만 활용해서 삼중 따옴표 안의 제목과 본문 내용에 적합한 소구점을 찾아줘.
                                소구점 : """,
                },
            ]

            result1 = generate_response(messages)
            print(f"{start + k}번째 행")
            print(f"제목 : {title}, 본문 : {content}")
            print(f"답변 : {result1}")
            print("\n")
            # sell.append(result1.replace("소구점 : ", ""))

            # selling_sent
            messages = [
                {
                    "role": "system",
                    "content": mp.mbti["SELLING_SENT_NTNF"],
                },
                {
                    "role": "user",
                    # "content": f"""'''제목 : {title}, 본문 : {content}''' 삼중 따옴표 안의 마케팅 문구를 마케팅 주체 : '{i[1]['marketing_entity']}', 타겟 : '{i[1]['marketing_target']}', 혜택 지급 조건 : '{i[1]['benefit_conditions']}', 혜택 : '{i[1]['benefits']}', 할인 수치 : '{i[1]['discount_figure']}', 프로모션 품목 : '{i[1]['promotional_items']}', 프로모션 장소 : '{i[1]['promotional_place']}', 이벤트 기간 : '{i[1]['event_period']}', 요일 정보 : '{i[1]['dow_information']}', 시즌 정보 : '{i[1]['season_information']}', 소구점 : '{result1}'을 반영하여 매력있는 광고 제목과 광고 본문으로 변경해줘.
                    # 결과는 다음 두 개의 키로 JSON 형식으로만 제공해줘: title, content""",
                    # "content": f"""'''제목 : {title}, 본문 : {content}''' 삼중 따옴표 안의 제목과 본문에 요일 정보 : {i[1]['dow_information']}, 시즌 정보 : {i[1]['season_information']}, 소구점 : {result1}을 반영하여 매력있는 광고 제목과 광고 본문을 만들어줘.
                    # 결과는 다음 두 개의 키로 JSON 형식으로만 제공해줘: title, content""",
                    # "content": f"""'''제목 : {title}, 본문 : {content}''' 삼중 따옴표 안의 제목과 본문에 소구점 : {result1}을 반영하여 매력있는 광고 제목과 광고 본문을 만들어줘.
                    # 결과는 다음 두 개의 키로 JSON 형식으로만 제공해줘: title, content""",
                    "content": f"""'''제목 : {title}, 본문 : {content}''' 삼중 따옴표 안의 제목과 본문에 마케팅 주체 : '{i[1]['marketing_entity']}', 혜택 : '{i[1]['benefits']}', 소구점 : '{result1}'을 반영하여 매력있는 광고 제목과 광고 본문을 만들어줘.
                    결과는 다음 두 개의 키로 JSON 형식으로만 제공해줘: title, content""",
                },
            ]

            result2 = generate_response(messages)
            print(f"{start + k}번째 행")
            print(f"제목 : {title}, 본문 : {content}")
            print(f"답변 : {result2}")

            result_json = json.loads(result2)
            # sell_title.append(result_json["title"])
            # sell_content.append(result_json["content"])

            # marketing_info
            # market_info.append(
            #     f"마케팅 주체 : {i[1]['marketing_entity']}, 혜택 : {i[1]['benefits']}"
            # )

            # 1 row save
            temp_dict = [
                {
                    "marketing_entity": i[1]["marketing_entity"],
                    "marketing_target": i[1]["marketing_target"],
                    "benefit_conditions": i[1]["benefit_conditions"],
                    "benefits": i[1]["benefits"],
                    "discount_figure": i[1]["discount_figure"],
                    "promotional_items": i[1]["promotional_items"],
                    "promotional_place": i[1]["promotional_place"],
                    "event_period": i[1]["event_period"],
                    "dow_information": i[1]["dow_information"],
                    "season_information": i[1]["season_information"],
                    "type": i[1]["type"],
                    "label": i[1]["label"].strip(),
                    "use_market": f"마케팅 주체 : {i[1]['marketing_entity']}, 혜택 : {i[1]['benefits']}",
                    "sell_point": result1.replace("소구점 : ", ""),
                    "sell_title": result_json["title"],
                    "sell_content": result_json["content"],
                }
            ]
            with open(cfg.PATH.EXT_GPT_SELL, "a", encoding="utf-8") as f:
                for line in temp_dict:
                    json_record = json.dumps(line, ensure_ascii=False)
                    f.write(json_record + "\n")
            print(f"{start + k}. Save Done!")
            print("-" * 100)

        math.factorial(100000)
        e_time = time.time()
        print(f"{e_time - s_time:.5f} sec")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{start + k}번째 행에서 오류")

    # finally:
    #     # temp save
    #     df_temp = df.iloc[start : start + len(sell_content)].copy()
    #     df_temp["use_market"] = market_info
    #     df_temp["sell_point"] = sell
    #     df_temp["sell_title"] = sell_title
    #     df_temp["sell_content"] = sell_content

    #     # save
    #     temp_dict = [
    #         {
    #             "marketing_entity": row["marketing_entity"],
    #             "marketing_target": row["marketing_target"],
    #             "benefit_conditions": row["benefit_conditions"],
    #             "benefits": row["benefits"],
    #             "discount_figure": row["discount_figure"],
    #             "promotional_items": row["promotional_items"],
    #             "promotional_place": row["promotional_place"],
    #             "event_period": row["event_period"],
    #             "dow_information": row["dow_information"],
    #             "season_information": row["season_information"],
    #             "type": row["type"],
    #             "label": row["label"].strip(),
    #             "use_market": row["use_market"],
    #             "sell_point": row["sell_point"],
    #             "sell_title": row["sell_title"],
    #             "sell_content": row["sell_content"],
    #         }
    #         for _, row in df_temp.iterrows()
    #     ]
    #     with open(cfg.PATH.EXT_GPT_SELL, "a", encoding="utf-8") as f:
    #         for line in temp_dict:
    #             json_record = json.dumps(line, ensure_ascii=False)
    #             f.write(json_record + "\n")

    #     print("Save Done!")

    # Excel Save
    # df_temp = df_temp[
    #     ["label", "use_market", "sell_point", "sell_title", "sell_content"]
    # ]
    # df_temp.rename(
    #     {
    #         "label": "원문",
    #         "market_info": "마케팅 정보",
    #         "selling": "소구점",
    #         "sentence": "소구점이 반영된 광고 문구",
    #     },
    #     inplace=True,
    # )
    # fm.save(cfg.PATH.temp_save, df_temp)


if __name__ == "__main__":
    main()
