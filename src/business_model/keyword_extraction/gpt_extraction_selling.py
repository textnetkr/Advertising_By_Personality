import hydra
import time
import math
import json
import mbti_prompt as mp
import openai
from pshmodule.utils import filemanager as fm


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    start = 0
    l_benefit = []
    l_period = []
    l_season = []
    l_weather = []
    l_anniv = []
    type = "NT"

    try:
        # Data Load
        df = fm.load(cfg.PATH.test)
        df = df[df.type == type]
        df_ref = fm.load(f"{cfg.PATH.EXT_GPT_SELL}_{type}.json")

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY

        def generate_response(messages: list, model: str) -> str:
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
        print("start")

        print(f"start: {start}")
        print(df_start.head())
        print(f"총 건수 : {len(df)}")
        print(f"대상 건수 : {len(df_start)}")
        print(f"{start}번째 행부터 시작!")

        for k, i in enumerate(df_start.iterrows()):
            # selling_point
            messages = [
                {
                    "role": "system",
                    "content": "너는 광고문구에서 정확한 마케팅정보를 찾아내는 카피라이터야.",
                },
                {
                    "role": "user",
                    "content": f"""{mp.mbti["SELLING"]}\n<원문>\n{i[1]['title']} {i[1]['content']}""",
                },
            ]
            # result
            result1 = generate_response(messages, cfg.MODEL.model1)
            benefit = (
                result1.split("혜택기간: ")[0]
                .replace("<마케팅정보>\n혜택/방법: ", "")
                .replace("\n", "")
            )
            period = result1.split("혜택기간: ")[1].split("시즌: ")[0].replace("\n", "")
            season = (
                result1.split("혜택기간: ")[1]
                .split("시즌: ")[1]
                .split("날씨: ")[0]
                .replace("\n", "")
            )
            weather = (
                result1.split("혜택기간: ")[1]
                .split("시즌: ")[1]
                .split("날씨: ")[1]
                .split("기념: ")[0]
                .replace("\n", "")
            )
            anniv = (
                result1.split("혜택기간: ")[1]
                .split("시즌: ")[1]
                .split("날씨: ")[1]
                .split("기념: ")[1]
                .replace("\n", "")
            )
            l_benefit.append(benefit)
            l_period.append(period)
            l_season.append(season)
            l_weather.append(weather)
            l_anniv.append(anniv)

            print(f"{start + k}번째 행")
            print(f"원문 : {i[1]['title']} {i[1]['content']}")
            print(f"답변 : {result1}")

            print("-" * 100)

            # 1 row save
            temp_dict = [
                {
                    "no": i[1]["no"],
                    "origin_no": i[1]["origin_no"],
                    "type": i[1]["type"],
                    "title": i[1]["title"],
                    "content": i[1]["content"],
                    "benefit": benefit,
                    "period": period,
                    "season": season,
                    "weather": weather,
                    "anniv": anniv,
                }
            ]
            with open(
                f"{cfg.PATH.EXT_GPT_SELL}_{type}.json", "a", encoding="utf-8"
            ) as f:
                for line in temp_dict:
                    json_record = json.dumps(line, ensure_ascii=False)
                    f.write(json_record + "\n")
            print(f"{start + k}. Save Done!")
            print("-" * 100)

        math.factorial(100000)
        e_time = time.time()
        print(f"최종 걸린 시간 : {e_time - s_time:.5f} sec")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{start + k}번째 행에서 오류")

    finally:
        print("end")
        # # temp save
        # df_temp = df.iloc[start : start + len(l_benefit)].copy()
        # print(df_temp.head())

        # df_temp["benefit"] = l_benefit
        # df_temp["period"] = l_period
        # df_temp["season"] = l_season
        # df_temp["weather"] = l_weather
        # df_temp["anniv"] = l_anniv

        # # Excel Save
        # fm.save(cfg.PATH.temp_save + "_1.xlsx", df_temp)


if __name__ == "__main__":
    main()
