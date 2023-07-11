# 🤖 Advertising by Personality
CJ와의 협업으로 설계된 성격유형별 문체 특성 기반 맞춤형 광고 메시지를 활용하여<br>
마케팅 관련 키워드를 입력하여 광고 문구를 생성하는 비지니스 모델 연구<br><br>

# 👉🏻 model
소구점이 반영된 광고 문구 생성 프로젝트
일반 문장이 들어왔을 때 소구점과 MBTI 성향에 맞는 성격의 광고 문구로 변경하는 모델을 구현.
 1. ChatGPT 프롬프트 튜닝으로 소구점을 추출하고 소구점이 반영된 광고 문구를 생성하여 T5를 이용하여 문체 변환하는 모델 구현.

 2. Koalpaca-Polyglot-5.8b를 활용하여 마케팅 정보를 input으로 Instruct 구조로 광고 문구를 생성하도록 PEFT LoRA finetuning 학습.     src/business_model/generation_lora
 3. CJ 성격유형별 문체 특성 기반 맞춤형 광고메시지 자동생성 복원 모델
 src/bart

# 👉🏻 tree
```bash
.
├── .gitignore
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
│   │   │   ├── gpt_extraction_selling.py
│   │   │   └── gpt_extraction_selling_stsf.py
│   │   └── t5_selling
│   │       ├── dataloader.py
│   │       ├── for_train.ipynb
│   │       ├── predict.py
│   │       ├── run.ipynb
└── └─      └── train.py
```