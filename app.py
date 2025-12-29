'''
FastAPI: /upload + /chat
'''
from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from extractors import extract_text
from rag import RAGStore
from typing import Optional

SYSTEM_PROMPT = """
너는 'AI 디지털 트윈 교수' 기반 학습 지원 챗봇이다.

목표:
- 학생이 업로드한 강의자료/공지/캡처(공식 또는 개인자료)와 학생 체감 정보를 바탕으로 
'이 교수의 이 과목을 듣는 학생에게 가장 현실적인 학습 조언'을 제공한다.

행동 원칙:
1) 한국어로 답한다.
2) 먼저 전체 구조(요약/학습 전략)를 제시한 뒤 세부 설명한다.
3) 자료 근거가 부족하면 추측하지 말고 필요한 자료/질문을 요청한다.
4) 단순 개념 설명이 아니라 시험/과제 대비 관점으로 설명한다.
""".strip()

app = FastAPI(title="KW AI Digital Twin Professor - MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # MVP: 개발 편의. 배포 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RAGStore(persist_dir="storage/chroma")
os.makedirs("storage/uploads", exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    course: str = Form(""),
    professor: str = Form(""),
    uploader_id: str = Form(""),
):
    data = await file.read()
    text, meta = extract_text(file.filename, data)

    if meta.get("type") == "unsupported":
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 파일 형식입니다. MVP에서는 PDF/PPTX/TXT/PNG(JPG)만 지원합니다. HWP는 PDF로 변환 후 업로드해주세요."
        )
    
    if meta.get("type") == "image" and not text:
        # OCR 실패 시 안내
        return {
            "ok": False,
            "reason": "ocr_failed_or_unavailable",
            "meta": meta,
            "tip": "캡처 글자가 크게 보이게 다시 업로드하거나, 텍스트로 복사/붙여넣기 해주세요. (서버에 tesseract 설치가 필요할 수 있어요.)"
        }
    
    res = store.add_document(
        raw_text=text,
        metadata=meta,
        course=course,
        professor=professor,
        uploader_id=uploader_id,
    )
    return {"meta": meta, "result": res}

@app.post("/chat")
async def chat(
    question: str = Form(...),
    course: str = Form(""),
    professor: str = Form(""),
):
    out = store.answer(
        system_prompt=SYSTEM_PROMPT,
        user_question=question,
        course=course or None,
        professor=professor or None,
        k=5,
    )
    store.log_chat(
        question=question,
        answer=out["answer"],
        course=course,
        professor=professor,
    )
    return out

@app.post("/chat/stream")
async def chat_stream(
    question: str = Form(""),
    course: str = Form(""),
    professor: str = Form(""),
):
    return StreamingResponse(
        store.stream_answer(),
        media_type="text/event-stream"
    )

@app.get("/documents")
def documents(
    course: Optional[str] = None,
    professor: Optional[str] = None,
    uploader_id: Optional[str] = None,
    limit: int = 50,
):
    return store.list_documents(
        course=course,
        professor=professor,
        uploader_id=uploader_id,
        limit=limit,
    )