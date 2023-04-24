import re
import hydra
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from pshmodule.processing import processing as p


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.PATH.tokenizer)

    # model
    model = BartForConditionalGeneration.from_pretrained(cfg.PATH.save_dir)
    model.to(device)

    # convert_to_other_unicode
    nlp = p.Nlp()

    texts = [
        ["", "NT", "SF", "당신은 덧셈 천재입니까?\\\\덧셈 게임 참여 시 💰31만 포인트가 [TAG1]님 주머니로 쏘-옥"],
        ["가을맞이", "ST", "SF", "10월이니까 1OOO포인트♬\\\\브랜드별 할인 받고 1천P 적립까지 더!\\알뜰하게 가을 옷 준비하려면 터치▶"],
        ["신학기", "NF", "NT", "우리 아이 성적도 자신감도 홈런⚾\\\\그 유명한 초등학습 1위 [홈런]\\새학기를 맞아 무료 체험을 제공해요❣️ 역사 연대표와 포인트도 드리니까 부담없이 신청하세요😍"],
        ["어버이날", "NT", "ST", "이번 어버이날 선물은 너로 정했다🎁\\\\지금 부모님 선물 구매하시면 2,OOOP를 사은품으로 드립니다💰 선물 사고 선물 받고🎵 꿩 먹고 알 먹고💗"],
        ["", "ST", "NF", "7일 뒤면 사라지는 쿠폰...❗\\\\7/일/뒤 혜택 종료!! 지금 앱에 접속해서 미사용 쿠폰 체크✔️체크✔️"],
    ]

    for i in texts:
        temp = str(i[1]).replace('\\\\', '\n\n').replace('\\', '\n')
        print(f"바꿀 문구 : {temp}, 시즌 정보 : {i[0]}")
        print(i[3])

        text = f"<season>{i[0]}<ctrl1>{i[1]}<ctrl2>{i[2]}{tokenizer.sep_token}{nlp.convert_to_other_unicode(i[3])}"

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
        print(f"바뀐 문구 : {i[2]}")
        # 이모지로 역치환 - Predict 후에 적용해야 함
        result = nlp.convert_emojis_in_text(nlp.convert_to_python_unicode(tokenizer.decode(pred[0], skip_special_tokens=True)))
        print(result.replace('\\\\', '\n\n').replace('\\', '\n'))
        print("-" * 100)


if __name__ == "__main__":
    main()
