import re
import hydra
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from pshmodule.processing import processing as p


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(cfg.PATH.save_dir)

    # model
    model = T5ForConditionalGeneration.from_pretrained(cfg.PATH.save_dir)
    model.to(device)

    # convert_to_other_unicode
    nlp = p.Nlp()

    texts = [
        [
            "KT",
            "통신사 결합을 원하는 고객",
            "kt 인터넷, 휴대폰 요금제 가입",
            "금액 할인",
            "최대 20%",
            "결합 할인",
            "전국 KT 매장, 마이케이티 앱",
            "없음",
            "없음",
            "없음",
            "할인율이나 할인 금액 등 고객이 얻을 수 있는 혜택 강조하기, 고객이 특별한 존재로서 특별대우를 받는 것처럼 느끼게 하기",
        ],
        [
            "KT",
            "통신사 결합을 원하는 고객",
            "kt 인터넷, 휴대폰 요금제 가입",
            "금액 할인",
            "최대 20%",
            "결합 할인",
            "전국 KT 매장, 마이케이티 앱",
            "없음",
            "없음",
            "없음",
            "할인율이나 할인 금액, 마감 임박 등 실질적인 혜택 정보는 첫 줄에 언급하기, 고객이 가까운 미래에 얻을 수 있는 보상을 암시적이고 상징적인 표현으로 나타내기",
        ],
        [
            "GS25",
            "도시락을 구매한 고객",
            "도시락 구입 후 QR Code 인증",
            "금액 할인",
            "500원",
            "코카콜라 320ml",
            "전국 GS25",
            "5.11~5.19",
            "없음",
            "여름 한정",
            "혜택 마감 임박 강조, 상품 구매 시 인기 상품 할인 쿠폰 제공.",
        ],
        [
            "AIA 암보험 전문 상담가",
            "암보험에 관심 있는 사람",
            "AIA 암보험 상담 신청",
            "1만 원 증정",
            "없음",
            "AIA 암보험",
            "없음",
            "없음",
            "없음",
            "없음",
            "혜택 마감 임박, 한정수량, 한정판, 선착순 등의 희소성 강조하기",
        ],
        [
            "신세계면세점",
            "여행객",
            "$10 이상 구매",
            "5천원 선불카드 캐시백",
            "없음",
            "없음",
            "없음",
            "없음",
            "매주 주말",
            "여행 시즌",
            "시즌정보나 일상정보를 활용해 예상되는 고객의 니즈 강조하기",
        ],
    ]

    for i in texts:
        text = f"<market_entity>{i[0]}<market_target>{i[1]}<benefit_cond>{i[2]}<benefits>{i[3]}<dis_figure>{i[4]}<prom_items>{i[5]}<prom_place>{i[6]}<event_period>{i[7]}<dow_info>{i[8]}<season_info>{i[9]}<sell_point>{i[10]}"

        encoded = tokenizer(text)
        sample = {k: torch.tensor([v]).to(device) for k, v in encoded.items()}

        with torch.no_grad():
            pred = model.generate(
                **sample,
                temperature=0.7,
                do_sample=True,
                top_k=20,
                penalty_alpha=0.6,
                max_length=512,
                eos_token_id=tokenizer.eos_token_id,
            )
        # 이모지로 역치환 - Predict 후에 적용해야 함
        result = nlp.convert_emojis_in_text(
            nlp.convert_to_python_unicode(
                tokenizer.decode(pred[0], skip_special_tokens=True)
            )
        )
        print(f"마케팅 주체 : {i[0]}")
        print(f"타겟 : {i[1]}")
        print(f"혜택 지급 조건 : {i[2]}")
        print(f"혜택 : {i[3]}")
        print(f"할인 수치 : {i[4]}")
        print(f"프로모션 품목 : {i[5]}")
        print(f"프로모션 장소 : {i[6]}")
        print(f"이벤트 기간 : {i[7]}")
        print(f"요일 정보 : {i[8]}")
        print(f"시즌 정보 : {i[9]}")
        print(f"소구점 : {i[10]}")
        result = result.replace("\\\\", "\n\n").replace("\\", "\n")
        result = result.encode("utf-8", "ignore").decode("utf-8")
        print(f"모델 답변 : {result}")
        print("-" * 100)


if __name__ == "__main__":
    main()
