"""knowledge_base 문서를 임베딩해 로컬 JSON index를 만든다."""

import json
import os
import re
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
INDEX_PATH = BASE_DIR / "data" / "index.json"
load_dotenv(BASE_DIR.parent / ".env.local")


def parse_document(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    document_id = re.search(r"문서 ID:\s*(.+)", text)
    version = re.search(r"버전:\s*(.+)", text)
    effective_date = re.search(r"시행일:\s*(.+)", text)
    return {
        "source_document_id": document_id.group(1).strip() if document_id else path.stem,
        "source_version": version.group(1).strip() if version else "unknown",
        "effective_date": effective_date.group(1).strip() if effective_date else "unknown",
        "text": text,
    }


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY를 .env.local에 설정하세요.")

    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    documents = [parse_document(path) for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))]
    if not documents:
        raise RuntimeError("knowledge_base에 Markdown 문서가 없습니다.")

    response = OpenAI(api_key=api_key).embeddings.create(
        model=embedding_model,
        input=[document["text"] for document in documents],
    )
    for document, item in zip(documents, response.data):
        document["embedding"] = item.embedding

    INDEX_PATH.parent.mkdir(exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(
            {"embedding_model": embedding_model, "documents": documents},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Indexed {len(documents)} documents: {INDEX_PATH}")


if __name__ == "__main__":
    main()
