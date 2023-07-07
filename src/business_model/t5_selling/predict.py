import re
import hydra
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from pshmodule.processing import processing as p
from pshmodule.utils import filemanager as fm


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # data load
    df = fm.load(cfg.PATH.predict)
    print(df.head())

    # tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(cfg.PATH.save_dir)

    # model
    model = T5ForConditionalGeneration.from_pretrained(cfg.PATH.save_dir)
    model.to(device)

    # convert_to_other_unicode
    nlp = p.Nlp()

    # texts = [
    #     [
    #         "🐰💕춘식이와 함께! 동반귀여움 춘식이 이모티콘 소장하세요\\\\춘식이가 프렌즈와 함께 등장했어요! 지금바로 소장하면 두 배로 귀여운 '춘식이 이모티콘'을 만나보세요. 고객님을 위한 애정과 감사로 가득한 소중한 선물입니다. 🎁💖",
    #     ],
    #     [
    #         "🎮 퇴근 시즌 이벤트! 아이템3종과 이모티콘 무료 증정 중! (2천명 한정)\\\\\퇴근하고 싶을 때, 몰래 하는 게임이 최고예요! 지금 들어오면 아이템3종과 이모티콘을 무료로 증정해 드려요. 희소한 혜택이니 놓치지 마세요! 한정 수량이니 서둘러 참여해주세요. 🎁🕹️🎁",
    #     ],
    #     [
    #         "🌞💃 핫썸머샵 원피스 위크! 최대 50% 할인 중!\\\\월요특가, 무료교환, 3종 쿠폰팩 증정! 다양한 헤택이 시원하게 쏟아지는 썸머 쇼핑 축제에서 원피스 쇼핑을 즐겨보세요. 인기와 품질을 반영한 최대 50% 세일로 마감 임박했습니다. 놓치지 마세요! 💃🛍️🎁",
    #     ],
    #     [
    #         "☕️🎉 오후 3시 쿠폰 폭탄! 아메리카노 공짜로 먹을 수 있는 기회!\\\\오늘 같이 더운 날엔 아메리카노 필수니까~ 간편하고 편리한 방법으로 공짜로 아메리카노를 즐길 수 있는 기회가 오후 3시에 찾아옵니다! 지금바로 카카오 플친하러 고고해서 놓치지 마세요! ☕️🎁😊",
    #     ],
    # ]

    # for i in texts:
    r_list = []
    for i in df.iterrows():
        if {i[1]["content"]} == "":
            text = f"<input>{i[1]['title']}"
        else:
            text = f"<input>{i[1]['title']}\\\\{i[1]['content']}"
        encoded = tokenizer(nlp.convert_to_other_unicode(text))
        sample = {k: torch.tensor([v]).to(device) for k, v in encoded.items()}

        with torch.no_grad():
            pred = model.generate(
                **sample,
                temperature=0.7,
                do_sample=True,
                top_k=10,
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
        print(f"{i[0]}번째 행")
        print(f"원문 : {text}")
        # print(f"gpt 광고 문구 : {i[1]['sell_sent']}")
        result = result.replace("\\\\", "\n\n").replace("\\", "\n")
        result = result.encode("utf-8", "ignore").decode("utf-8")
        print(f"모델 답변 : {result}")
        print("-" * 100)
        r_list.append(result)

    # save
    df["t5"] = r_list
    df.rename(
        columns={"no": "번호", "title": "INPUT_제목", "content": "INPUT_본문"},
        inplace=True,
    )
    fm.save(cfg.PATH.temp_save, df)


if __name__ == "__main__":
    main()
