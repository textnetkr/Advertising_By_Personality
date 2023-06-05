# 🤖 Advertising by Personality
국내 유통대기업과의 협업으로 설계된 성격유형별 문체 특성 기반 맞춤형 광고 메시지를 활용하여<br>
마케팅 관련 키워드를 입력하여 광고 문구를 생성하는 비지니스 모델 연구<br><br>

# 👉🏻 model
- 기존 광고 메시지를 다른 성격유형 문체로 변경하는 맞춤형 광고 메시지 생성 모델 구현 - koBART<br>
- NT 성향 문구, NF 성향 문구, 마케팅 대상, 소구점 등 총 13개 마케팅 정보를 입력 받아 광고 메시지 생성하는 모델 - koT5<br>
&nbsp;&nbsp;&nbsp;KoAlpaca-Polyglot 5.8b 모델로 PEFT LoRA <br><br>

# 👉🏻 tree
.<br>
├── README.md<br>
├── requirements.txt<br>
├── src<br>
│   └── bart<br>
│       ├── dataloader.py<br>
│       ├── predict.py<br>
│       ├── run.ipynb<br>
│       └── train.py<br>
│   └── business_model<br>
│       └── generation_lora<br>
│           ├── custom_lora.py<br>
│           ├── dataloader.py<br>
│           ├── predict.py<br>
│           ├── run.ipynb<br>
│           └── train.py<br>
│       └── keyword_extraction<br>
│           ├── for_extraction.ipynb<br>
│           ├── gpt_extraction_mbti.py<br>
│           ├── after_treatment.ipynb<br>
│           ├── gpt_extraction_marketing.py<br>
│           └── keybert.ipynb<br>
│       └── t5<br>
│           ├── dataloader.py<br>
│           ├── predict.py<br>
│           ├── run.ipynb<br>
│           └── train.py<br>
│   └── preprocessing<br>
│       ├── for_train.ipynb<br>
└──     └── predict_check.ipynb<br>