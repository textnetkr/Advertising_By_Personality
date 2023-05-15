import hydra
import torch
from transformers import (
    PreTrainedTokenizerFast,
    GPTNeoXForCausalLM
)
from peft import PeftModel, PeftConfig
from pshmodule.processing import processing as p

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # peft
    config = PeftConfig.from_pretrained(cfg.PATH.peft_model)

    # tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.base_model_name_or_path)

    # model
    model = GPTNeoXForCausalLM.from_pretrained(
        config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(model, cfg.PATH.peft_model)
    model.to(device)

    print(f"tokenizer : {tokenizer}")
    print(f"model : {model}")

    # # data instrunction form
    # text = f"""Below is an instruction that describes a task, paired with an input that provides further context.\n
    #             아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n
    #             Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n
    #             ### Instruction(명령어):\n다음 정보를 활용해서 광고 문구를 생성해줘.\n\n
    #             ### Input(입력):\n마케팅 주체: CJ ONE, 마케팅 대상: 도시락을 구매한 고객, 혜택 조건: 도시락 구입 후 QR Code 인증, 할인 수치: 없음, 프로모션 품목: 코카콜라 320ml, 이벤트 기간: 5.11~5.19, 시즌 정보: 여름 한정\n\n
    #             ### Response(응답): 
    #         """

    # encoded = tokenizer(text)

    # sample = {k: torch.tensor([v]).to(device) for k, v in encoded.items()}
    # del sample['token_type_ids']
    # print(sample)

    # with torch.no_grad():
    #     pred = model.generate(
    #         **sample,
    #         penalty_alpha=0.6,
    #         top_k=5,
    #         max_length=512,
    #         eos_token_id=tokenizer.eos_token_id,
    #     )

    # print(tokenizer.decode(pred[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()