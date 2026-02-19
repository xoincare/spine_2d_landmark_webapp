"""추론 파이프라인: 모델 로드, 이미지 전처리, 예측"""

import io
import logging
import os

# Windows에서 conda MKL과 PyTorch OpenMP 런타임 충돌 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402  # must import before numpy (Windows MKL DLL conflict)
import numpy as np
from PIL import Image

from .config import IMAGE_SIZE, N_LANDMARKS
from .model import SpineLandmarkNet

logger = logging.getLogger(__name__)


def load_model(path: str, device: str = "cpu") -> SpineLandmarkNet:
    """checkpoint에서 모델 로드"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    n_landmarks = config.get("n_landmarks", N_LANDMARKS)
    backbone = config.get("backbone", "hrnet_w48")

    model = SpineLandmarkNet(
        n_landmarks=n_landmarks,
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {path} (backbone={backbone}, landmarks={n_landmarks})")
    return model


def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    이미지 바이트 → 모델 입력 텐서

    Returns:
        tensor: (1, 1, 512, 512) 정규화된 텐서
        original_size: (width, height) 원본 크기
    """
    img = Image.open(io.BytesIO(image_bytes))
    original_size = img.size  # (width, height)

    # grayscale 변환
    img = img.convert("L")

    # 512x512 리사이즈
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    # numpy → tensor, [0, 1] 정규화
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return tensor, original_size


@torch.no_grad()
def predict(model: SpineLandmarkNet, tensor: torch.Tensor) -> np.ndarray:
    """
    모델 추론 → keypoints (N_landmarks, 2) in 512x512 공간
    """
    output = model(tensor)
    keypoints = output["keypoints"][0].cpu().numpy()  # (N_landmarks, 2)
    return keypoints


def rescale_keypoints(
    keypoints: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    """
    512x512 공간의 키포인트를 원본 이미지 좌표로 변환

    Args:
        keypoints: (N, 2) in 512x512 space, [x, y]
        original_size: (width, height)
    """
    scale_x = original_size[0] / IMAGE_SIZE
    scale_y = original_size[1] / IMAGE_SIZE
    scaled = keypoints.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled
