# Financial RAG Evaluation PoC

금융 고객지원 RAG를 대상으로 12개 AI 안전성·신뢰성 가이드라인의 평가 가능성을 검토한 PoC입니다. 동일한 질문과 가상 정책 문서를 사용하여 DeepEval 및 LangWatch 평가를 실행합니다.

## 구성

- 대상 답변 모델: `gpt-4.1-mini` (환경변수로 변경 가능)
- LLM Judge: `gpt-5.6-sol` (환경변수로 변경 가능)
- RAG: 정책 문서 임베딩 검색 후 상위 문서를 답변 모델에 전달
- DeepEval: 코드·pytest 기반 12개 가이드라인 평가
- LangWatch: RAGAS 및 LLM-as-a-Judge 기반 12개 가이드라인 평가

정책 문서와 QA셋은 실제 고객정보나 실제 운영 정책이 아닌 PoC용 가상 데이터입니다.

## 디렉터리

```text
rag/
  app.py                         # 검색, 답변 생성, DeepEval Trace 기록
  build_index.py                 # 정책 문서 임베딩 인덱스 생성
  knowledge_base/                # 가상 승인 정책 문서
evals/
  test_guideline_12_deepeval.py  # DeepEval 12개 가이드라인 평가
  run_langwatch_rag.py           # LangWatch 12개 가이드라인 평가
  fixtures/                      # 승인 QA셋 및 LangWatch Dataset
```

## 환경 설정

Python 3.11 이상 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.local
```

`.env.local`에 최소 `OPENAI_API_KEY`를 설정합니다. LangWatch 평가도 실행하려면 `LANGWATCH_API_KEY`와 `LANGWATCH_ENDPOINT_URL`도 설정해야 합니다.

## 실행

정책 문서가 변경되었거나 처음 실행하는 경우 검색 인덱스를 생성합니다.

```bash
python -m rag.build_index
```

DeepEval 12개 가이드라인 평가를 실행합니다.

```bash
deepeval test run evals/test_guideline_12_deepeval.py
```

LangWatch 서버와 프로젝트 API Key를 준비한 후 LangWatch 평가를 실행합니다.

```bash
python evals/run_langwatch_rag.py
```

## 가이드라인별 DeepEval Metric

| 가이드라인 | 검증 목적 | 적용 Metric |
| --- | --- | --- |
| 2.1 | 고객 질문 관련성·업무 기준 | Answer Relevancy, GEval |
| 2.2 | 승인 환불 조건 검색 회수 | Contextual Recall |
| 2.3 | 답변 근거 충실도·환각 | Faithfulness, Hallucination |
| 3.2 | 문서 밖 금융정보 생성 방지 | 문서 기반 GEval |
| 3.3 | 검색 문서 관련성 | Contextual Relevancy |
| 4.1·4.2 | 차별 방지 | Bias, GEval |
| 5.2 | 제한 사유·다음 행동 설명 | GEval |
| 7.1 | 비밀번호·OTP 보호 | GEval, Toxicity |
| 7.2 | 프롬프트 인젝션·정책변조 방어 | GEval, Faithfulness, Hallucination |
| 7.3 | 답변-근거 충실도 모니터링 | Faithfulness |
| 9.2 | 외부/검색 문서 기반 응답 검증 | Contextual Precision, Hallucination |

## 유의사항

- `.env.local`에는 API Key가 포함되므로 Git에 올리지 않습니다.
- `rag/data/index.json`은 생성 파일이므로 제외하며, `python -m rag.build_index`로 재생성합니다.
- LLM Judge 기반 평가는 의미·정책 적합성 평가에 유용하지만 결과 변동 가능성이 있으므로, 민감정보 형식·금칙어·필수 문구에는 Rule 기반 검증을 병행해야 합니다.
