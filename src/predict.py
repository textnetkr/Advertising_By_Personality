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

    # texts = [
    #     ["", "NT", "SF", "당신은 덧셈 천재입니까?\\\\덧셈 게임 참여 시 💰31만 포인트가 [TAG1]님 주머니로 쏘-옥"],
    #     ["가을맞이", "ST", "SF", "10월이니까 1OOO포인트♬\\\\브랜드별 할인 받고 1천P 적립까지 더!\\알뜰하게 가을 옷 준비하려면 터치▶"],
    #     ["신학기", "NF", "NT", "우리 아이 성적도 자신감도 홈런⚾\\\\그 유명한 초등학습 1위 [홈런]\\새학기를 맞아 무료 체험을 제공해요❣️ 역사 연대표와 포인트도 드리니까 부담없이 신청하세요😍"],
    #     ["어버이날", "NT", "ST", "이번 어버이날 선물은 너로 정했다🎁\\\\지금 부모님 선물 구매하시면 2,OOOP를 사은품으로 드립니다💰 선물 사고 선물 받고🎵 꿩 먹고 알 먹고💗"],
    #     ["", "ST", "NF", "7일 뒤면 사라지는 쿠폰...❗\\\\7/일/뒤 혜택 종료!! 지금 앱에 접속해서 미사용 쿠폰 체크✔️체크✔️"],
    # ]
    # 정성 평가 10개
    texts = [
        ['', '원문', 'NT', '화제의 갤럭시 노트10\\ud83d\\udcf1\\댓글만 써도 100% 당첨+온라인체험존 ▶', '핸드폰 댓글로 장만했어요😎\\갤럭시 노트10 댓.글.만.달.고 가져가자🎁\\참여만 하면 무조건 1OOP 지급💰 온라인 체험존에서 성능도 확인할 수 있어요▶'],
        ['', 'ST', 'NF', '굶기만 하는 다이어트는 이제 그만❌\\건강하고 맛있는 다이어트 간식 GET하고 3,OOOP 받아가기✔️', '프로 다이어터들 다 모이세요🏃\\이 기회 혼자만 놓치면 땅을 치고 후회할걸요?!😭 요즘 핫한 다이어트 간식 구경하고 가세요! 구입 시 3천P 적립해 드려요!'],
        ['', '원문', 'NF', '사용하신 포인트 돌려드립니다.\\오늘 마감! 페이백 신청하러 가기▶', '왜 내 포인트는 마르지 않는 고양😻\\두근두근❤️ 페이백 신청했더니 사용한 포인트가 작구 되살아나!\\나처럼 페이백 신청할 사람🙋 여기 모/여/라!(🚨오늘까지만)'],
        ['', 'NF', '원문', '아니 진짜 이래도 되는거야?!🔥\\모르고 있었던 분들은 인생의 절반 손해본 겁니다😎 매일 최저가! [CJmall 어메이징 세일위크 2탄]에서 득템각⭐', 'CJmall 어메이징 세일위크 2탄 오픈!\\매일 최저가 도전! 해피한 특가! 지금 바로 득템하세요'],
        ['', 'ST', 'NF', '꽝 없는 당첨 기회 도착★\\주사위 던지고 발뮤다 공기청정기 득템♬\\지금 바로 행운 잡으러 컴컴~!', '행운의 주사위 굴릴 찬스🎲\\고객님께만 드릴거라눙💘 발뮤다 공기청정기 행운이 뙇- 나올지도 몰라요! 참! 포인트는 덤이랍니다~💕'],
        ['새해', 'ST', 'NT', '고객님의 신년 운세 도착!\\CJmall앱에서 공짜로 운세 확인✔️\\행운 선물도 응모하고 새해 福 받으세요♥', '내년 나의 운세는?🔮\\CJmall에서 무/료/로 새해 타로도 보고 푸짐한 경품도 받아가세요🎁 3분만 투자하고 앱에서 응모하기👉'],
        ['겨울맞이', '원문', 'SF', '아직 구매하시지 않으셨나요?\\방문만 해도 기프티콘 100% 증정', '올겨울 SNS대란템 고객님만 없네?!\\날 추울 땐, 침대 위에서 랜선 쇼핑하고 마음까지 따땃하게! 지금 방문만 해도 기프티콘 무조건 지급!'],
        ['설연휴', 'ST', 'SF', '설날은 [더플레이스 쿠폰]♬\\다운받은 쿠폰 확인하고 더플레이스 가자!', '설연휴엔 더플레이스와 함께~\\모던한 분위기의 가족 모임 장소로 PICK❗\\요즘 설날엔 마르게리타 피자 먹는다~ 쿠폰 잊지 말고 꼭 사용하기▶'],
        ['설연휴', '원문', 'NT', 'CJmall 설맞이 혜택받쥐\\2020년 무료 신년운세 + 식품 10% 적립', '복채는 필요하지 않아요🙅‍♀️\\[CJmall]에서 설을 맞아 무.료.로 신년 운세도 봐주고🔮 식품 1O% 적립까지 해주니까요💰 운세 보고 이 참에 설날 장도 보면 딱이겠어요🛒'],
        ['크리스마스', '원문', 'NF', '\\ud83c\\udf84크리스마스 선물 쏩니다!\\ud83c\\udf84\\2021년 운세만 보면 경품 100% 당첨!\\ud83c\\udf85\\BESPOKE 큐브 냉장고도 받아가세요!', '내년에는 남친 생기게 해주세요🙏\\크리스마스 선물로 신년운세에서 알려준다는데?💡\\운세만 봤는데 경품 1OO% 당첨! 경품에 큐브 냉장고가 있다는..😨']
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
