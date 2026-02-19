# Spine 2D Landmark Webapp

X-ray 이미지에서 **102개 척추 랜드마크**를 자동 검출하고 **Cobb angle / Kyphosis / Lordosis**를 측정하는 웹앱.

Phase 2 학습 모델(`best.pth`)만으로 독립 동작하며, Cloud Run(CPU-only, stateless) 배포를 지원합니다.

## 기능

- 드래그앤드롭 X-ray 업로드
- T1~L5 (17개 척추 × 6개 랜드마크 = 102점) 자동 검출
- **Cobb angle**: 전체 척추 중 최대 측만 각도
- **Kyphosis**: T1 상판 ↔ T12 하판 (흉추 후만)
- **Lordosis**: L1 상판 ↔ L5 하판 (요추 전만)
- **Segment angles**: 인접 척추 간 각도 (16개)
- 랜드마크 + endplate + 각도 오버레이 이미지

## 프로젝트 구조

```
spine_2d_landmark_webapp/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 서버
│   ├── model.py             # SpineLandmarkNet 모델
│   ├── inference.py          # 추론 파이프라인
│   ├── angles.py             # 각도 계산
│   ├── visualization.py      # 결과 이미지 생성
│   └── config.py             # 상수 정의
├── static/
│   ├── index.html            # 업로드 UI
│   ├── style.css
│   └── app.js
└── models/
    └── .gitkeep              # best.pth 위치
```

## 사전 준비

### 1. 모델 가중치 복사

Phase 2 학습 완료 후:

```bash
cp 3d-spine/phase2_landmark/models/best.pth spine_2d_landmark_webapp/models/best.pth
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

> PyTorch CPU 버전만 필요하다면:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

## 로컬 실행

```bash
uvicorn app.main:app --port 8080
```

브라우저에서 `http://localhost:8080` 접속 → X-ray 이미지 업로드.

## API

### `GET /health`

상태 확인 (Cloud Run probe).

```json
{"status": "ok", "model_loaded": true}
```

### `POST /analyze`

X-ray 이미지를 업로드하면 랜드마크 + 각도 + 어노테이션 이미지를 반환합니다.

**Request**: `multipart/form-data`, 필드명 `file`

```bash
curl -X POST http://localhost:8080/analyze \
  -F "file=@xray.png"
```

**Response**:

```json
{
  "landmarks": [
    {
      "vertebra": "T1",
      "landmarks": {
        "superior_anterior": {"x": 245.3, "y": 52.1},
        "superior_posterior": {"x": 268.7, "y": 53.4},
        "inferior_anterior": {"x": 244.1, "y": 78.9},
        "inferior_posterior": {"x": 267.5, "y": 80.2},
        "pedicle_left": {"x": 238.0, "y": 65.5},
        "pedicle_right": {"x": 275.2, "y": 66.8}
      }
    }
  ],
  "angles": {
    "cobb": {"cobb_angle": 23.5, "upper_vertebra": "T6", "lower_vertebra": "L1"},
    "kyphosis": {"kyphosis_angle": 35.2, "from": "T1_superior", "to": "T12_inferior"},
    "lordosis": {"lordosis_angle": 42.1, "from": "L1_superior", "to": "L5_inferior"},
    "segments": [{"segment": "T1-T2", "angle": 3.2}]
  },
  "annotated_image": "<base64 PNG>",
  "image_size": {"width": 512, "height": 512}
}
```

## Docker 빌드

```bash
docker build -t spine-webapp .
docker run -p 8080:8080 spine-webapp
```

## Cloud Run 배포

```bash
gcloud run deploy spine-landmark \
    --source . \
    --region asia-northeast3 \
    --memory 2Gi --cpu 2 \
    --timeout 300 \
    --min-instances 0 --max-instances 5 \
    --port 8080 \
    --allow-unauthenticated \
    --startup-cpu-boost
```

## 기술 스택

| 구성 | 기술 |
|---|---|
| Backend | FastAPI + Uvicorn |
| Model | SpineLandmarkNet (HRNet-w48 backbone + Heatmap Head) |
| Inference | PyTorch CPU, Soft-Argmax |
| Frontend | Vanilla JS (외부 라이브러리 없음) |
| Container | python:3.11-slim, PyTorch CPU-only |
| Deploy | Google Cloud Run |
