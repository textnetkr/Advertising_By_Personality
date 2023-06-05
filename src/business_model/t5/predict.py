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
            "도시락을 구매한 고객✔️500원 할인!",
            "없음",
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
            "코카콜라 할인",
        ],
        [
            "없음",
            "KT 결합 할인!\\u2763️, 고객을 위해주는, 특별하게 생각하는 KT",
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
            "요금제 결합 할인",
        ],
        [
            "없음",
            "없음",
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
            "없음",
        ],
        [
            "없음",
            "님 알잘딱깔센 뜻 아세요? 알잘딱깔센 크리스마스 카드 쓰러가기!",
            "없음",
            "지인",
            "크리스마스 카드 보내기",
            "최대 1,000P",
            "없음",
            "없음",
            "없음",
            "없음",
            "없음",
            "크리스마스",
            "없음",
        ],
        [
            "아.묻.따 혜택 받아가세요!, 신세계면세점 5천원 선불카드 캐시백",
            "없음",
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
            "신세계면세점",
        ],
    ]

    for i in texts:
        text = f"<NT>{[i[0]]}<NF>{i[1]}<market_entity>{i[2]}<market_target>{i[3]}<benefit_cond>{i[4]}<benefits>{i[5]}<dis_figure>{i[6]}<prom_items>{i[7]}<prom_place>{i[8]}<event_period>{i[9]}<dow_info>{i[10]}<season_info>{i[11]}<soli_point>{i[12]}"

        encoded = tokenizer(text)
        sample = {k: torch.tensor([v]).to(device) for k, v in encoded.items()}

        with torch.no_grad():
            pred = model.generate(
                **sample,
                penalty_alpha=0.6,
                top_k=5,
                max_length=512,
                eos_token_id=tokenizer.eos_token_id,
            )
        # 이모지로 역치환 - Predict 후에 적용해야 함
        result = nlp.convert_emojis_in_text(
            nlp.convert_to_python_unicode(
                tokenizer.decode(pred[0], skip_special_tokens=True)
            )
        )
        nt = nlp.convert_emojis_in_text(nlp.convert_to_python_unicode(i[0]))
        nf = nlp.convert_emojis_in_text(nlp.convert_to_python_unicode(i[1]))
        print(f"NT 성향 문구 : {nt}")
        print(f"NF 성향 문구 : {nf}")
        print(f"마케팅 주체 : {i[2]}")
        print(f"마케팅 대상 : {i[3]}")
        print(f"혜택 지급 조건 : {i[4]}")
        print(f"혜택 : {i[5]}")
        print(f"할인 수치 : {i[6]}")
        print(f"프로모션 품목 : {i[7]}")
        print(f"프로모션 장소 : {i[8]}")
        print(f"이벤트 기간 : {i[9]}")
        print(f"요일 정보 : {i[10]}")
        print(f"시즌 정보 : {i[11]}")
        print(f"소구점 : {i[12]}")
        result = result.replace("\\\\", "\n\n").replace("\\", "\n")
        print(f"모델 답변 : {result}")
        print("-" * 100)


if __name__ == "__main__":
    main()
