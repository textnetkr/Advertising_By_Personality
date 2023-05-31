import hydra
import json
from tqdm import tqdm
from pshmodule.utils import filemanager as fm
import pandas as pd
import openai


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    try:
        # Data Load
        data = []
        with open(cfg.PATH.train_data, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.rstrip('\n|\r')))
        df = pd.DataFrame(data)

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY
        model = "gpt-3.5-turbo"

        def generate_response(messages: list) -> str:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response["choices"][0]["message"]["content"]

        # GPT Extraction
        predict = []
        start = 240
        df_start = df.iloc[start:]
        print(df_start.head())
        print(f"{start}번째 행부터 시작!")
        for k, i in enumerate(df_start.iterrows()):
            messages = [
                    {
                        "role": "system",
                        "content": """너는 판매 촉진을 위한 마케팅 문구를 만드는 카피라이터야.\n
                                        아래 예시처럼 ''안에 있는 원문에서 MBTI 성향 중 NT, NF의 유형으로 각각에 해당하는 마케팅 문구를 추출할거야\n
                                        '이번 봄 나들이는 빕스로❓\\\\제철 과일 딸기 가득 [딸기홀릭] 시즌♥ 시크릿박스 쿠폰 쓰고 딸기 디저트 저.렴.하.게 즐기자'
                                        NT 성향 문구 : 없음, NF 성향 문구 : 없음\n
                                        '갈 땐 가더라도...(진지)\\\\[뮤지엄오브컬러展] 3O% 할인 쿠폰💌 쓰는 것 정돈 괜찮잖아? 은하수를 닮은 다채로운 색감의 향연✨ 거 쿠폰 쓰기 딱 좋은 전시네💕
                                        NT 성향 문구 : 갈 땐 가더라도...(진지), [뮤지엄오브컬러展] 3O% 할인 쿠폰💌 쓰는 것 정돈 괜찮잖아?, 은하수를 닮은 다채로운 색감의 향연✨, 거 쿠폰 쓰기 딱 좋은 전시네💕, NF 성향 문구 : 없음\n
                                        '한겨울엔 따뜻한 박물관으로 GO!\\\\고객님께만 드리는 단독 할인! 국립중앙박물관 호랑이 전시 1+1! 마감 D-1✔️
                                        NT 성향 문구 : 없음, NF 성향 문구 : 없음\n
                                        '옛날옛날에~호랑이가 살았어요🦁\\\\아직도 안 가본 사람?(어흥)😨 국립중앙박물관에서 호랑이를 만나보세요⭐ 지금 1+1 할인중!
                                        NT 성향 문구 : 없음, NF 성향 문구 : 옛날옛날에~호랑이가 살았어요🦁, 아직도 안 가본 사람?(어흥)😨\n
                                        '짝짝짝! 친구 맺으면 1,000P~\\\\정답만 맞히면 포인트 주는 이벤트! 퀴즈 맞히고 1,000P 바로 받아가기'
                                        NT 성향 문구 : 없음, NF 성향 문구 : 없음\n
                                        '🐯: 어흥!(전시 할인 중!)\\\\⌛내/일/까/지/만 [국립중앙박물관 호랑이展] 1+1 할인⚡\\할인도 2배 호랑이 기운도 2배로 받을 기회!✌️ 절대 놓치지 마세흥!
                                        NT 성향 문구 : 🐯: 어흥!(전시 할인 중!), ⌛내/일/까/지/만, NF 성향 문구 : 없음\n
                                        '투썸 빙수 먹으면 아메리카노가 공짜!\\\\올 여름 더위 물리치는 투썸 빙수가 왔다!\\빙수 포함 5개 득템프 찍으면 아메리카노가 O원!
                                        NT 성향 문구 : 없음, NF 성향 문구 : 없음\n
                                        '힙스터들의 성지, 성수동 갈 사람?\\\\🔥화제의 팝업 전시, 뮤지엄오브컬러가 고객님을 기다려욧!\\힙쟁이라면 안 간 사람 없다던데👁️...단독 3O% 할인 중
                                        NT 성향 문구 : 없음, NF 성향 문구 : 힙스터들의 성지, 성수동 갈 사람?, 🔥화제의 팝업 전시, 뮤지엄오브컬러가 고객님을 기다려욧!, 힙쟁이라면 안 간 사람 없다던데👁️\n
                                        NT, NF의 특징을 알려줄게\n
                                        NT 유형 : 비유적, 추상적 수사 방식 사용, 첫 줄에 혜택 정보(마감 정보, 할인 수치 등) 강조, 노래 가사, 영화 대사, 광고 문구, 밈, 이모지 활용 (예 : 외식하기 딱 좋은 날이네, 숨 참고 할인 다이브 등), 제품의 효과, 효능, 전문성, 신뢰성 강조
                                        NF 유형 : 비유적, 추상적 수사 방식 사용, 노래 가사, 영화 대사, 광고 문구, 밈, 이모지 활용, 고객을 위해주는, 특별하게 생각하는 듯한 메시지
                                """,
                    },
                    {
                        "role": "user",
                        "content": f"""{i[1]['label']}\n
                                        위 문장에서 아래 정보를 ',\n'를 기준으로 뽑아줘.
                                        NT 성향 문구 : ,\n
                                        NF 성향 문구 : 
                                        """,
                    },
            ]
            result = generate_response(messages)
            print(f"{start + k}번째 행")
            print(f"원문 : {i[1]['label']}")
            print(f"답변 : {result}")
            print("-" * 100)
            predict.append(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{start + k}번째 행에서 오류!")
    finally:
        # Processing
        df_temp = df.iloc[start:start + k].copy()
        df_temp["mbti_temp"] = predict
        df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("NT 성향 문구 : ", "")
        df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("NT 성향 문구: ", "")
        df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("NF 성향 문구 : ", "")
        df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("NF 성향 문구: ", "")
        df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("\n\n", "\n")
        df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace("\n", ",\n")
        df_temp["mbti_temp"] = df_temp.mbti_temp.str.replace(",,", ",")
        df_split_mbti = df_temp.mbti_temp.str.split(",\n", expand=True)
        df_split_mbti_true = df_split_mbti[
            df_split_mbti[0].str.contains(",\n")
        ]
        df_split_mbti_true = df_split_mbti_true[0].str.split(
            ",\n", expand=True
        )
        df_split_mbti_false = df_split_mbti[
            ~df_split_mbti[0].str.contains(",\n")
        ]
        df_split_mbti_temp = pd.concat(
            [df_split_mbti_false, df_split_mbti_true]
        )        
        df_split_mbti_temp.rename(
            columns={
                0: "NT",
                1: "NF",
            },
            inplace=True,
        )
        df_result = pd.concat([df_split_mbti_temp, df_temp], axis=1)
        df_result.sort_index(ascending=True, inplace=True)
        df_result.fillna("없음", inplace=True)

        # save
        temp_dict = [
            {
                "NT": row["NT"].strip(),
                "NF": row["NF"].strip(),
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
        with open(eval(f"cfg.PATH.final_traiin_data"), "a", encoding="utf-8") as f:
            for line in temp_dict:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")
        print("Save Done!")

if __name__ == "__main__":
    main()
