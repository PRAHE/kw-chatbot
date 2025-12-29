# AI 디지털 트윈 교수 챗봇 웹사이트

이 프로젝트는 대학 강의에서 교수의 설명 방식, 강조 포인트, 시험 유형과 학생 후기 데이터를 결합하여 개인화된 학습 조언을 제공하는 **AI 디지털 트윈 교수** 챗봇을 위한 기본 웹 애플리케이션입니다. 백엔드는 FastAPI와 구글의 Generative AI(Gemini) API를 사용하여 스트리밍 방식으로 응답을 생성하고, 프론트엔드는 간단한 HTML/JavaScript로 작성된 채팅 UI를 제공합니다.

> **주의**: 실제로 이 코드를 배포하려면 Google Cloud 콘솔에서 Gemini API를 활성화하고 API 키를 발급받아야 합니다. 키는 서버 사이드 환경변수로 관리해야 하며, 클라이언트 코드에 노출해서는 안 됩니다.

## 기능

- `FastAPI`로 작성된 백엔드 서버
- `/chat` 경로에서 SSE(Server‑Sent Events)로 AI 응답을 스트리밍
- Google Generative AI(PaLM/Gemini) API 사용 예제 포함
- 간단한 채팅 UI: 사용자 입력 후 메시지를 실시간으로 렌더링
- CORS 설정, 오류 처리 및 환경변수 로딩 지원

## 설치 및 실행

1. 저장소를 클론한 후 프로젝트 디렉터리로 이동합니다.

```bash
git clone <repository-url>
cd chatbot_site
```

2. Python 가상환경을 생성하고 필요한 패키지를 설치합니다. 가상환경은 프로젝트마다 라이브러리 버전을 격리하는 데 권장되며 필수는 아니지만 재현성과 충돌 방지를 위해 사용하는 것이 좋습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. 프로젝트 루트에 `.env` 파일을 생성하고 `GEMINI_API_KEY`를 설정합니다. `GEMINI_API_KEY`는 Google Cloud 콘솔에서 발급받은 키입니다. 다음과 같이 입력합니다.

```ini
GEMINI_API_KEY="당신의-제미나이-API-키"
```

4. 개발 서버를 실행합니다.

```bash
uvicorn app:app --reload --port 8000
```

5. 브라우저에서 `http://localhost:8000` 로 접속하면 챗봇 인터페이스가 나타납니다. 질문을 입력하면 AI의 스트리밍 응답을 실시간으로 확인할 수 있습니다.

## 폴더 구조

```
chatbot_site/
├── app.py              # FastAPI 백엔드 애플리케이션
├── requirements.txt     # Python 의존성 목록
├── .env.example         # 환경변수 설정 예시
├── README.md            # 프로젝트 설명
├── templates/
│   └── index.html       # 프론트엔드 HTML 템플릿
└── static/
    └── style.css        # 간단한 스타일시트
```

## 개선 아이디어

이 프로젝트는 기본적인 구조를 제시합니다. 실제 서비스로 확장하려면 다음과 같은 기능을 추가할 수 있습니다.

- **강의자료 분석**: PDF, PPT, KLAS 공지 등 교수의 공식 자료를 파싱하여 요약/키워드 추출
- **학생 후기 DB화**: 에타 등의 비공식 후기 데이터를 수집·정제하여 과목 난이도, 과제량, 시험 유형 등의 메타데이터 구축
- **개인화 전략 추천**: 사용자의 질문 내용과 수집된 데이터베이스를 기반으로 학습 전략, 예상문제 생성, 중요 개념 설명 제공
- **대화 히스토리 유지**: 세션 기반으로 대화 맥락을 저장하여 보다 일관성 있는 답변 생성
- **사용자 인증 및 권한 관리**: 특정 과목 정보에 대한 접근을 제한하거나 사용자 맞춤형 기록 저장

## 라이선스

이 프로젝트는 교육용 예제 코드이며 자유롭게 수정 및 배포할 수 있습니다. Google Generative AI API 사용 시 약관을 반드시 준수하십시오.# kw-chatbot
