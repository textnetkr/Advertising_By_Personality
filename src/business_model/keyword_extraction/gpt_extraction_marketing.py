import hydra
import swifter
import json
from tqdm import tqdm
from pshmodule.utils import filemanager as fm
import mbti_prompt as mp
import pandas as pd
import openai


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    try:
        # Data Load
        df = fm.load(cfg.PATH.for_gpt)
        print(df.head())

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY
        model = "gpt-3.5-turbo"

        def generate_response(messages: list) -> str:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response["choices"][0]["message"]["content"]

        # GPT Extraction
        predict = []
        start = 40
        df_start = df.iloc[start:].copy()
        print(f"{start}번째 행부터 시작!")
        data = [[i[1]["type"], i[1]["input"]] for i in df_start.iterrows()]
        for k, i in tqdm(enumerate(data)):
            messages = [
                {
                    "role": "system",
                    "content": mp.mbti["marketing"],
                },
                {
                    "role": "user",
                    "content": f"""'{i[1]}' 문장에서 아래 정보를 다음 키와 함께 JSON 형식으로 만들어줘: 마케팅 주체-marketing_entity, 타겟-marketing_target, 혜택 지급 조건-benefit_conditions, 혜택-benefits, 할인 수치-discount_figure, 프로모션 품목-promotional_items, 프로모션 장소-promotional_place, 이벤트 기간-event_period, 요일 정보-dow_information, 시즌 정보-season_information""",
                    # : marketing_entity, marketing_target, benefit_conditions, benefits, discount_figure, promotional_items, promotional_place, event_period, dow_information, season_information: marketing_entity, marketing_target, benefit_conditions, benefits, discount_figure, promotional_items, promotional_place, event_period, dow_information, season_information
                },
            ]
            result = generate_response(messages)
            print(f"{start + k}번째 행")
            print(f"원문 : {i[1]}")
            print(f"답변 : {result}")
            print("-" * 100)
            res_dict = json.loads(result)
            res_dict = {i[0]: i[1] for i in res_dict.items()}
            predict.append(res_dict)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{start + k}번째 행에서 오류!")

    finally:
        # Processing
        df_temp = df_start.iloc[: len(predict)].copy()
        df_pred = pd.DataFrame(predict)
        print(f"df_temp : {df_temp.shape}")
        print(f"df_pred : {df_pred.shape}")
        df_result = pd.concat([df_temp, df_pred], axis=1)
        df_result.fillna("없음", inplace=True)
        df_result.replace("", "없음", inplace=True)
        print(df_result.head())

        # save
        temp_dict = [
            {
                "marketing_entity": row["marketing_entity"],
                "marketing_target": row["marketing_target"],
                "benefit_conditions": row["benefit_conditions"],
                "benefits": row["benefits"],
                "discount_figure": row["discount_figure"],
                "promotional_items": row["promotional_items"],
                "promotional_place": row["promotional_place"],
                "event_period": row["event_period"],
                "dow_information": row["dow_information"],
                "season_information": row["season_information"],
                "type": row["type"],
                "label": row["input"],
            }
            for _, row in df_result.iterrows()
        ]
        with open(eval(f"cfg.PATH.EXT_GPT_MARKET"), "a", encoding="utf-8") as f:
            for line in temp_dict:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")


if __name__ == "__main__":
    main()
