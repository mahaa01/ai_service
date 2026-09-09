"""OpenAI-compatible API를 이용한 최소 RAG 앱."""

import json
import math
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "data" / "index.json"
load_dotenv(BASE_DIR.parent / ".env.local")

from deepeval.tracing import observe, update_current_span, update_current_trace


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm) if left_norm and right_norm else 0.0


@observe(type="retriever", embedder=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
def retrieve(question: str, top_k: int = 2) -> list[dict]:
    if not INDEX_PATH.exists():
        raise RuntimeError("RAG index가 없습니다. `python -m rag.build_index`를 먼저 실행하세요.")
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    query_embedding = client.embeddings.create(
        model=index["embedding_model"], input=question
    ).data[0].embedding
    ranked = sorted(
        index["documents"],
        key=lambda document: cosine_similarity(query_embedding, document["embedding"]),
        reverse=True,
    )
    retrieved_documents = ranked[:top_k]
    contexts = [document["text"] for document in retrieved_documents]
    update_current_span(input=question, retrieval_context=contexts)
    update_current_trace(retrieval_context=contexts)
    return retrieved_documents


@observe(type="llm", model=os.getenv("RAG_MODEL", "gpt-4.1-mini"))
def generate_answer(question: str, sources: str) -> str:
    model = os.getenv("RAG_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 금융 고객지원 AI입니다. 제공된 승인 문서에만 근거해 답변하세요. "
                    "문서에 없는 조건, 수치, 절차는 만들지 말고 확인할 수 없다고 답하세요. "
                    "사용한 문서 ID나 내부 문서 메타데이터는 답변 본문에 표시하지 마세요.\n\n"
                    f"[승인 문서]\n{sources}"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content or ""
    update_current_span(input=question, output=answer)
    return answer


@observe(name="Example financial RAG")
def answer_question(question: str) -> dict:
    retrieved_documents = retrieve(question)
    sources = "\n\n".join(
        f"[{document['source_document_id']}]\n{document['text']}"
        for document in retrieved_documents
    )
    answer = generate_answer(question, sources)
    source_metadata = [
        {
            "document_id": document["source_document_id"],
            "version": document["source_version"],
            "effective_date": document["effective_date"],
        }
        for document in retrieved_documents
    ]
    update_current_trace(
        input=question,
        output=answer,
        metadata={"retrieved_sources": source_metadata},
    )
    return {
        "answer": answer,
        "retrieval_context": [document["text"] for document in retrieved_documents],
        "sources": source_metadata,
    }
