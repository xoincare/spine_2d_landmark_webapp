"""FastAPI 서버: 업로드 → 분석 → 결과"""

import io
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .inference import load_model, preprocess_image, predict, rescale_keypoints
from .angles import compute_all_angles
from .visualization import draw_landmarks, image_to_base64
from .config import VERTEBRA_NAMES, LANDMARK_NAMES, N_LANDMARKS_PER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Spine 2D Landmark Analyzer")

# 모델 전역 변수
_model = None
MODEL_PATH = Path(__file__).parent.parent / "models" / "best.pth"


@app.on_event("startup")
async def startup():
    global _model
    if MODEL_PATH.exists():
        _model = load_model(str(MODEL_PATH), device="cpu")
        logger.info("Model ready")
    else:
        logger.warning(f"Model not found: {MODEL_PATH}. /analyze will return 503.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Place best.pth in models/")

    # 이미지 읽기
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    try:
        tensor, original_size = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # 추론
    keypoints_512 = predict(_model, tensor)
    keypoints_orig = rescale_keypoints(keypoints_512, original_size)

    # 각도 계산
    angles = compute_all_angles(keypoints_512)

    # 시각화
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    annotated = draw_landmarks(original_image, keypoints_orig, angles)
    annotated_b64 = image_to_base64(annotated)

    # 랜드마크를 구조화된 형태로 변환
    landmarks_structured = []
    for v_idx, v_name in enumerate(VERTEBRA_NAMES):
        points = {}
        for lm_idx, lm_name in enumerate(LANDMARK_NAMES):
            idx = v_idx * N_LANDMARKS_PER + lm_idx
            points[lm_name] = {
                "x": round(float(keypoints_orig[idx, 0]), 1),
                "y": round(float(keypoints_orig[idx, 1]), 1),
            }
        landmarks_structured.append({
            "vertebra": v_name,
            "landmarks": points,
        })

    return JSONResponse({
        "landmarks": landmarks_structured,
        "angles": angles,
        "annotated_image": annotated_b64,
        "image_size": {"width": original_size[0], "height": original_size[1]},
    })


# 정적 파일 서빙 (프론트엔드)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
