import hydra
import swifter
import json
from tqdm import tqdm
from pshmodule.utils import filemanager as fm
import pandas as pd
import openai


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    try:
        # Data Load
        df = fm.load(cfg.PATH.for_gpt2)

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY
        model = "gpt-3.5-turbo"

        def generate_response(messages: list) -> str:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response["choices"][0]["message"]["content"]

        # GPT Extraction
        predict = []
        # type = ["NT", "NF"]
        type = ["NT"]
        for t in type:
            df_temp = df[df.type == t]
            df_temp.reset_index(inplace=True, drop=True)
            start = 199
            df_temp = df_temp.iloc[start:]
            print(f"{start}번째 행부터 시작!")
            data = [[i[1]["type"], i[1]["input"]] for i in df_temp.iterrows()]
            for k, i in tqdm(enumerate(data)):
                messages = [
                    {
                        "role": "system",
                        "content": """너는 판매 촉진을 위한 마케팅 문구를 만드는 카피라이터야.\n
                                    아래 예시처럼 ''안에 있는 원문에서 마케팅 주체, 타겟, 혜택 지급 조건, 혜택, 할인수치, 프로모션품목, 프로모션장소, 이벤트기간, 요일정보, 시즌정보, 소구점을 추출할거야\n
                                    '이번 봄 나들이는 빕스로❓\\\\제철 과일 딸기 가득 [딸기홀릭] 시즌♥ 시크릿박스 쿠폰 쓰고 딸기 디저트 저.렴.하.게 즐기자'
                                    마케팅 주체 : 빕스,\n타겟 : 없음,\n혜택 지급 조건 : 없음,\n혜택 : 시크릿박스 쿠폰,\n할인 수치 : 없음,\n프로모션 품목 : [딸기 홀릭] 시즌,\n프로모션 장소 : 없음,\n이벤트 기간 : 없음,\n요일 정보 : 없음,\n시즌 정보 : 봄,\n소구점 : 봄 나들이, 제철 과일\n
                                    '갈 땐 가더라도...(진지)\\\\[뮤지엄오브컬러展] 3O% 할인 쿠폰💌 쓰는 것 정돈 괜찮잖아? 은하수를 닮은 다채로운 색감의 향연✨ 거 쿠폰 쓰기 딱 좋은 전시네💕'
                                    마케팅 주체 : 없음,\n타겟 : 없음,\n혜택 지급 조건 : 할인 쿠폰 쓰기,\n혜택 : 30% 할인 쿠폰,\n할인 수치 : 0.3,\n프로모션 품목 : 뮤지엄오브컬러展,\n프로모션 장소 : 없음,\n이벤트 기간 : 없음,\n요일 정보 : 없음,\n시즌 정보 : 없음,\n소구점 : 없음\n
                                    '한겨울엔 따뜻한 박물관으로 GO!\\\\고객님께만 드리는 단독 할인! 국립중앙박물관 호랑이 전시 1+1! 마감 D-1✔️'
                                    마케팅 주체 : 국립중앙박물관,\n타겟 : 고객님,\n혜택 지급 조건 : 없음,\n혜택 : 단독 할인, 1+1,\n할인 수치 : 1+1,\n프로모션 품목 : 호랑이 전시,\n프로모션 장소 : 없음,\n이벤트 기간 : D-1,\n요일 정보 : 없음,\n시즌 정보 : 겨울맞이,\n소구점 : 한겨울엔 따뜻한 박물관으로\n
                                    '옛날옛날에~호랑이가 살았어요🦁\\\\아직도 안 가본 사람?(어흥)😨 국립중앙박물관에서 호랑이를 만나보세요⭐ 지금 1+1 할인중!
                                    마케팅 주체 : 국립중앙박물관,\n타겟 : 없음,\n혜택 지급 조건 : 없음,\n혜택 : 1+1 할인,\n할인 수치 : 1+1,\n프로모션 품목 : 없음,\n프로모션 장소 : 없음,\n이벤트 기간 : 없음,\n요일 정보 : 없음,\n시즌 정보 : 없음,\n소구점 : 없음\n
                                    '짝짝짝! 친구 맺으면 1,000P~\\\\정답만 맞히면 포인트 주는 이벤트! 퀴즈 맞히고 1,000P 바로 받아가기'
                                    마케팅 주체 : 없음,\n타겟 : 없음,\n혜택 지급 조건 : 친구맺기, 퀴즈 맞히기,\n혜택 : 1000P,\n할인 수치 : 없음,\n프로모션 품목 : 없음,\n프로모션 장소 : 없음,\n이벤트 기간 : 없음,\n요일 정보 : 없음,\n시즌 정보 : 없음,\n소구점 : 없음\n
                                    '🐯: 어흥!(전시 할인 중!)\\\\⌛내/일/까/지/만 [국립중앙박물관 호랑이展] 1+1 할인⚡\\할인도 2배 호랑이 기운도 2배로 받을 기회!✌️ 절대 놓치지 마세흥!'
                                    마케팅 주체 : 국립중앙박물관,\n타겟 : 없음,\n혜택 지급 조건 : 없음,\n혜택 : 1+1 할인,\n할인 수치 : 1+1,\n프로모션 품목 : 호랑이展,\n프로모션 장소 : 없음,\n이벤트 기간 : 내일까지,\n요일 정보 : 없음,\n시즌 정보 : 없음,\n소구점 : 없음\n
                                    '투썸 빙수 먹으면 아메리카노가 공짜!\\\\올 여름 더위 물리치는 투썸 빙수가 왔다!\\빙수 포함 5개 득템프 찍으면 아메리카노가 O원!'
                                    마케팅 주체 : 투썸,\n타겟 : 없음,\n혜택 지급 조건 : 빙수 포함 5개 득템프 찍기,\n혜택 : 1+1 할인,\n할인 수치 : 없음,\n프로모션 품목 : 빙수,\n프로모션 장소 : 없음,\n이벤트 기간 : 없음,\n요일 정보 : 없음,\n시즌 정보 : 여름맞이,\n소구점 : 더위 물리치는 투썸 빙수\n
                                    '힙스터들의 성지, 성수동 갈 사람?\\\\🔥화제의 팝업 전시, 뮤지엄오브컬러가 고객님을 기다려욧!\\힙쟁이라면 안 간 사람 없다던데👁️...단독 3O% 할인 중'
                                    마케팅 주체 : 없음,\n타겟 : 고객님,\n혜택 지급 조건 : 없음,\n혜택 : 30% 할인,\n할인 수치 : 0.3,\n프로모션 품목 : 뮤지엄오브컬러,\n프로모션 장소 : 성수동,\n이벤트 기간 : 없음,\n요일 정보 : 없음,\n시즌 정보 : 없음,\n소구점 : 없음
                                """,
                    },
                    {
                        "role": "user",
                        "content": f"""{i[1]}\n
                                                        위 문장에서 아래 정보를 ',\n'를 기준으로 뽑아줘.
                                                        마케팅 주체 : ,\n
                                                        타겟 : ,\n
                                                        혜택 지급 조건 : ,\n
                                                        혜택 : ,\n
                                                        할인 수치 : ,\n
                                                        프로모션 품목 : ,\n
                                                        프로모션 장소 : ,\n
                                                        이벤트 기간 : ,\n
                                                        요일 정보 : ,\n
                                                        시즌 정보 : ,\n
                                                        소구점 :
                                        """,
                    },
                ]
                result = generate_response(messages)
                print(f"{start + k}번째 행")
                print(f"원문 : {i[1]}")
                print(f"답변 : {result}")
                print("-" * 100)
                predict.append(result)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{start + k}번째 행에서 오류!")

    finally:
        # Processing
        df_temp = df_temp.iloc[:k]
        df_temp["marketing"] = predict
        df_temp["marketing_temp"] = df_temp.marketing.swifter.apply(
            lambda x: x.replace("마케팅 주체 : ", "")
            .replace("타겟 : ", "")
            .replace("혜택 지급 조건 : ", "")
            .replace("혜택 : ", "")
            .replace("할인 수치 : ", "")
            .replace("프로모션 품목 : ", "")
            .replace("프로모션 장소 : ", "")
            .replace("이벤트 기간 : ", "")
            .replace("요일 정보 : ", "")
            .replace("시즌 정보 : ", "")
            .replace("소구점 : ", "")
        )
        df_temp["marketing_temp"] = df_temp.marketing_temp.swifter.apply(
            lambda x: x.replace("마케팅 주체: ", "")
            .replace("타겟: ", "")
            .replace("혜택 지급 조건: ", "")
            .replace("혜택: ", "")
            .replace("할인 수치: ", "")
            .replace("프로모션 품목: ", "")
            .replace("프로모션 장소: ", "")
            .replace("이벤트 기간: ", "")
            .replace("요일 정보: ", "")
            .replace("시즌 정보: ", "")
            .replace("소구점: ", "")
        )
        df_temp["marketing_temp"] = df_temp.marketing_temp.str.replace("\n\n", "\n")
        df_temp["marketing_temp"] = df_temp.marketing_temp.str.replace("\n", ",\n")
        df_temp["marketing_temp"] = df_temp.marketing_temp.str.replace(",,", ",")
        df_split_marketing = df_temp.marketing_temp.str.split(",\n", expand=True)
        df_split_marketing_true = df_split_marketing[
            df_split_marketing[0].str.contains(",\n")
        ]
        df_split_marketing_true = df_split_marketing_true[0].str.split(
            ",\n", expand=True
        )
        df_split_marketing_false = df_split_marketing[
            ~df_split_marketing[0].str.contains(",\n")
        ]
        df_split_marketing_temp = pd.concat(
            [df_split_marketing_false, df_split_marketing_true]
        )
        # 마케팅 주체: marketing_entity, 타겟: marketing_target, 혜택 지급 조건: benefit_conditions, 혜택: benefits, 할인 수치: discount_figure, 프로모션 품목: promotional_items, 프로모션 장소: promotional_place, 이벤트 기간: event_period, 요일 정보: dow_information, 시즌 정보: season_information, 소구점: solicitation_point
        df_split_marketing_temp.rename(
            columns={
                0: "marketing_entity",
                1: "marketing_target",
                2: "benefit_conditions",
                3: "benefits",
                4: "discount_figure",
                5: "promotional_items",
                6: "promotional_place",
                7: "event_period",
                8: "dow_information",
                9: "season_information",
                10: "solicitation_point",
            },
            inplace=True,
        )
        df_temp.rename(columns={"input": "label"}, inplace=True)
        df_temp = df_temp[["type", "label"]]
        df_result = pd.concat([df_split_marketing_temp, df_temp], axis=1)
        df_result.sort_index(ascending=True, inplace=True)
        df_result.fillna("없음", inplace=True)

        # save
        temp_dict = [
            {
                "marketing_entity": row["marketing_entity"].strip(),
                "marketing_target": row["marketing_target"].strip(),
                "benefit_conditions": row["benefit_conditions"].strip(),
                "benefits": row["benefits"].strip(),
                "discount_figure": row["discount_figure"].strip(),
                "promotional_items": row["promotional_items"].strip(),
                "promotional_place": row["promotional_place"].strip(),
                "event_period": row["event_period"].strip(),
                "dow_information": row["dow_information"].strip(),
                "season_information": row["season_information"].strip(),
                "solicitation_point": row["solicitation_point"].strip(),
                "type": row["type"],
                "label": row["label"].strip(),
            }
            for _, row in df_result.iterrows()
        ]
        with open(eval(f"cfg.PATH.EXT_GPT_{t}2"), "a", encoding="utf-8") as f:
            for line in temp_dict:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")


if __name__ == "__main__":
    main()
