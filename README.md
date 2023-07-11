# 🤖 Advertising by Personality
국내 유통대기업과의 협업으로 설계된 성격유형별 문체 특성 기반 맞춤형 광고 메시지를 활용하여<br>
마케팅 관련 키워드를 입력하여 광고 문구를 생성하는 비지니스 모델 연구<br><br>

# 👉🏻 model
- 기존 광고 메시지를 다른 성격유형 문체로 변경하는 맞춤형 광고 메시지 생성 모델 구현 - koBART<br>
- NT 성향 문구, NF 성향 문구, 마케팅 대상, 소구점 등 총 13개 마케팅 정보를 입력 받아 광고 메시지 생성하는 모델 - koT5<br>
&nbsp;&nbsp;&nbsp;PEFT LoRA 시도 - KoAlpaca polyglot-5.8b<br><br>

# 👉🏻 tree
```bash
.
├── README.md
├── requirements.txt
├── src
│   ├── bart
│   │   ├── dataloader.py
│   │   ├── predict.py
│   │   ├── run.ipynb
│   │   └── train.py
│   ├── business_model
│   │   ├── generation_lora
│   │   │   ├── custom_lora.py
│   │   │   ├── dataloader.py
│   │   │   ├── predict.py
│   │   │   ├── run.ipynb
│   │   │   └── train.py
│   │   ├── keyword_extraction
│   │   │   ├── after_treatment.ipynb
│   │   │   ├── for_extraction.ipynb
│   │   │   ├── gpt_extraction_marketing.py
│   │   │   ├── gpt_extraction_mbti.py
│   │   │   └── keybert.ipynb
│   │   └── t5
│   │       ├── dataloader.py
│   │       ├── predict.py
│   │       ├── run.ipynb
│   │       └── train.py
│   ├── preprocessing
│   │   ├── for_train.ipynb
│   └── └── predict_check.ipynb
└── tree.txt
```