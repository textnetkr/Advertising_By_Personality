import re
import hydra
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.PATH.tokenizer)

    # model
    model = BartForConditionalGeneration.from_pretrained(cfg.PATH.save_dir)
    model.to(device)



    # 이모지를 Java, JavaScript, JSON 유니코드 코드로 변경
    def convert_other(emoji, target_format="java"):
        if target_format not in ["java", "javascript", "json"]:
            raise ValueError("Invalid target format. It should be 'java', 'javascript', or 'json'.")

        def to_surrogate_pair(code_point):
            high = (code_point - 0x10000) // 0x400 + 0xD800
            low = (code_point - 0x10000) % 0x400 + 0xDC00
            if target_format == "json":
                return f"\\\\u{high:04x}\\\\u{low:04x}"
            else:
                return f"\\u{high:04x}\\u{low:04x}"

        pattern = re.compile(r"[\U00010000-\U0010FFFF]")
        return pattern.sub(lambda match: to_surrogate_pair(ord(match.group(0))), emoji)

    # Java, JavaScript, JSON 유니코드 이모지 코드를 찾는 정규식
    def convert_unicode(emoji_code):    
        pattern = re.compile(r"(?:\\u[0-9a-fA-F]{4}){2}")
        
        def surrogate_pair(match):
            code_points = [int(x, 16) for x in match.group(0).split('\\u')[1:]]
            high, low = code_points
            code_point = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
            return f"\\U{code_point:08x}"
        
        return pattern.sub(surrogate_pair, emoji_code)

    texts = [
        ["NT", "SF", "당신은 덧셈 천재입니까?\n덧셈 게임 참여 시 💰31만 포인트가 [TAG1]님 주머니로 쏘-옥"],
        [
            "원문",
            "NT",
            "[CJONE VIP] 계절밥상 1만원/5천원 할인쿠폰\n마지막 2일! [계절밥상 디너/주말 1만원, 런치 5천원 할인 쿠폰] 9월30일까지 꼭 사용하세요!",
        ],
        [
            "NF",
            "SF",
            "뭐든 몰래 하는 게 더 짜릿한 법⚡\n매니아 플러스에게만 몰래 드리는 선물🎁 [VIPS] 추가 15% 할인 쿠폰으로 알찬 연말 보내자⭐",
        ],
        ["SF", "원문", "집콕러버 집순이들 여기 손!\nBTV로 요즘 핫한 콘텐츠를 끊김없이 즐긴다! 지금 가입 시 6O만원 혜택 증정!"],
        ["ST", "NF", "7일 뒤면 사라지는 쿠폰...❗\n7/일/뒤 혜택 종료!! 지금 앱에 접속해서 미사용 쿠폰 체크✔️체크✔️"],
    ]

    for i in texts:
        print(f"바꿀 문구 : {i[0]}")
        print(i[2])

        text = f"<ctrl1>{i[0]}<ctrl2>{i[1]}{tokenizer.sep_token}{convert_other(i[2])}"

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
        print(f"바뀐 문구 : {i[1]}")
        # 이모지로 역치환 - Predict 후에 적용해야 함
        print(convert_unicode(tokenizer.decode(pred[0], skip_special_tokens=True)))
        print("-" * 100)


if __name__ == "__main__":
    main()
