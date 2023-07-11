import hydra
import torch
from transformers import PreTrainedTokenizerFast, GPTNeoXForCausalLM
from peft import PeftModel, PeftConfig


device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.MODEL.name)

    # model
    config = PeftConfig.from_pretrained(cfg.PATH.peft_model)
    model = GPTNeoXForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, cfg.PATH.peft_model)

    # data instrunction form
    text = f"""Below is an instruction that describes a task, paired with an input that provides further context.\n
                아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n
                Write a response that appropriately completes the request.\n
                요청을 적절히 완료하는 응답을 작성하세요.\n\n
                ### Instruction(명령어):\n너는 판매 촉진을 위한 마케팅 문구를 만드는 카피라이터야.\n
                    마케팅 주체, 타겟, 혜택 지급 조건, 혜택, 프로모션 품목, 프로모션 장소, 이벤트 기간, 시즌 정보으로 광고 문구를 생성할거야.\n\n
                ### Input(입력):\n마케팅 주체: CJ ONE, 타겟: 도시락을 구매한 고객, 혜택 지급 조건: 도시락 구입 후 QR Code 인증, 혜택: 500원 할인, 프로모션 품목: 코카콜라 320ml, 이벤트 기간: 5.11~5.19, 시즌 정보: 여름 한정\n\n
                ### Response(응답):\n
                """

    encoded = tokenizer(text)
    sample = {k: torch.tensor([v]).to(device) for k, v in encoded.items()}

    # predict
    model.eval().to(device)
    with torch.no_grad():
        pred = model.generate(
            input_ids=sample["input_ids"],
            penalty_alpha=0.8,
            top_k=5,
            max_length=512,
        )
        print(pred)
        print(tokenizer.decode(pred[0], skip_special_tokens=True))
        # print(
        #     tokenizer.batch_decode(
        #         pred.detach().cpu().numpy(),
        #         skip_special_tokens=True,
        #     )
        # )


if __name__ == "__main__":
    main()
