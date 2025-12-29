'''
Chunking + Chroma + 임베딩 + RAG (+ documents index)
'''
import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI

def normalize_text(text: str) -> str:
    # 과한 정규화는 금물: 줄바꿈을 살리되, 공백 폭주만 정리
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 2200, overlap: int = 200) -> List[str]:
    """
    MVP용 간단 청킹:
    - 문단 기준으로 합치되 max_chars 넘으면 자름
    - overlap으로 문맥 손실 완화
    """
    text = normalize_text(text)
    if not text:
        return []
    
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            flush()
            if len(p) <= max_chars:
                buf = p
            else:
                # 문단이 너무 길면 강제 슬라이스
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    chunks.append(p[start:end].strip())
                    start = max(0, end - overlap)

    flush()

    # overlap 처리(문단 단위 overlpa은 어려워서, 청크 경계에 일부 복사)
    if overlap > 0 and len(chunks) >= 2:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i-1][-overlap:]
            overlapped.append((prev_tail + "\n" + chunks[i]).strip())
        chunks = overlapped

    return chunks

class RAGStore:
    """
    Collections:
    - lecture_docs: 청크(검색 대상)
    - lecture_doc_index: 문서 단위 메타(목록 조회용)
    """

    def __init__(self, persist_dir: str = "storage/chroma"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text_embedding-3-small")

        self.chat_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        os.makedirs(persist_dir, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # chunks (검색 대상)
        self.col = self.chroma.get_or_create_collection(name="lecture_docs")
        # docs index (목록)
        self.doc_index = self.chroma.get_or_create_collection(name="lecture_doc_index")


# Embedding
def embed(self, texts: List[str]) -> List[List[float]]:
    resp = self.client.embeddings.create(model=self.embed_model, input=texts)
    return [d.embedding for d in resp.data]

# Add documents
def add_document(
    self,
    raw_text: str,
    metadata: Dict[str, Any],
    course: Optional[str],
    professor: Optional[str],
    uploader_id: Optional[str],
) -> Dict[str, Any]:
    raw_text = normalize_text(raw_text)
    if not raw_text:
        return {"ok": False, "reason": "no_text_extracted"}
    
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(raw_text)

    if not chunks:
        return {"ok": False, "reason": "no_chunks"}
    
    created_at = datetime.utcnow().isoformat()

    filename = metadata.get("filename", "") or metadata.get("name", "") or ""
    source_type = metadata.get("type", "") or ""
    ocr_used = metadata.get("ocr", "")

    # 1) 문서 인덱스 저장 (목록 조회용)
    doc_meta = {
        "doc_id": doc_id,
        "course": course or "",
        "professor": professor or "",
        "uploader_id": uploader_id or "",
        "filename": filename,
        "source_type": source_type,
        "ocr": ocr_used,
        "chunks": len(chunks),
        "created_at": created_at,
    }
    # doc_index 컬렉션은 documents 내용이 없어도 되지만, Chroma는 documents가 필요할 수 있어 빈 문자열로 
    self.doc_index.add(
        ids=[doc_id],
        documents=[""],
        metadatas=[doc_meta],
    )

    # 2) 청크 저장 (검색 대상)

    base_meta = {
        "doc_id": doc_id,
        "course": course or "",
        "professor": professor or "",
        "uploader_id": uploader_id or "",
        "filename": filename,
        "source_type": source_type,
        "ocr": ocr_used,
        "created_at": created_at,
    }

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    embs = self.embed(chunks)

    metas: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        m = dict(base_meta)
        m["chunk_index"] = i

        # 출처 표시용 힌트: [PDF p.X], [PPTX slide X] 같은 머리표가 있으면 저장
        first_line = ch.splitlines()[0].strip() if ch.splitlines() else ""
        m["where_hint"] = first_line[:80]

        metas.append(m)

    self.col.add(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)

    return {"ok": True, "doc_id": doc_id, "chunks": len(chunks), "created_at": created_at}

# List documents
def list_documents(
    self,
    course: Optional[str] = None,
    professor: Optional[str] = None,
    uploader_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if course:
        where["course"] = course
    if professor:
        where["professor"] = professor
    if uploader_id:
        where["uploader_id"] = uploader_id

    res = self.doc_index.get(
        where=where if where else None,
        limit=limit,
        include=["metadatas"],
    )

    items: List[Dict[str, Any]] = []
    ids = res.get("ids", []) or []
    metas = res.get("metadatas", []) or []

    for doc_id, meta in zip(ids, metas):
        items.append(
            {
                "doc_id": doc_id,
                "filename": meta.get("filename", ""),
                "source_type": meta.get("source_type", ""),
                "course": meta.get("course", ""),
                "professor": meta.get("professor", ""),
                "uploader_id": meta.get("uploader_id", ""),
                "chunks": meta.get("chunks", 0),
                "created_at": meta.get("created_at", ""),
                "ocr": meta.get("ocr", ""),
            }
        )

        # created_at 기준 내림차순 정렬
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {"ok": True, "items": items}
    
# Search
def search(
    self, 
    query: str, 
    k: int = 5, 
    course: str | None = None, 
    professor: str | None = None
) -> List[Dict[str, Any]]:
    where: Dict[str, Any] = {}
    if course:
        where["course"] = course
    if professor:
        where["professor"] = professor

    q_emb = self.embed([query])[0]
    res = self.col.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"]
    )

    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "meta": meta, "distance": float(dist) if dist is not None else None})
    return hits

# Answer (with sources)
def answer(
    self,
    system_prompt: str,
    user_question: str,
    course: Optional[str],
    professor: Optional[str],
    k: int = 5,
) -> Dict[str, Any]:
    hits = self.search(user_question, k=k, course=course, professor=professor)

    # sources: 프론트 "근거 목록"
    sources: List[Dict[str, Any]] = []

    context_blocks: List[str] = []
    for idx, h in enumerate(hits, start=1):
        m = h["meta"]
        cite_line = f"[{idx}] {m.get('filename','')} | {m.get('where_hint','')}"
        context_blocks.append(f"---\nSOURCE {idx}\n{cite_line}\n\n{h['text']}\n")

        # distance -> score (보기용 변환: 1/(1+dist))
        dist = h.get("distance")
        score = (1.0 / (1.0 + dist)) if isinstance(dist, (int, float)) else 0.0

        sources.append(
            {
                "rank": idx,
                "doc_id": m.get("doc_id", ""),
                "chunk_id": f"{m.get('doc_id', '')}_{m.get('chunk_index','')}",
                "filename": m.get("filename", ""),
                "where_hint": m.get("where_hint", ""),
                "chunk_index": m.get("chunk_index", None),
                "course": m.get("course", ""),
                "professor": m.get("professor", ""),
                "score": round(float(score), 4),
            }
        )

    context = "\n".join(context_blocks).strip()

    # 챗봇 정체성 강화
    final_system = system_prompt.strip() + "\n\n" + (
        "너는 업로드된 공식자료/학생자료를 근거로 답한다.\n"
        "- 근거가 부족하면 추측하지 말고, 필요한 자료/질문을 요청한다.\n"
        "- 먼저 전체 구조(요약/학습 전략)를 제시한 뒤 세부 설명한다.\n"
        "- 답변 마지막에 사용한 SOURCE 번호를 인용 형태로 붙인다. 예: (SOURCE 1, SOURCE 2)\n"
    )

    messages =[
        {"role": "system", "content": final_system},
        {"role": "system", "content": f"[CONTEXT]\n{context}" if context else "[CONTEXT]\n(검색된 자료 없음)"},
        {"role": "user", "content": user_question},
    ]

    resp = self.client.chat.completions.create(
        model=self.chat_model,
        messages=messages,
        temperature=0.2,
    )
    text = resp.choices[0].message.content

    return {
        "ok": True,
        "answer": text, 
        "sources": sources,
        "hits": len(hits)
    }