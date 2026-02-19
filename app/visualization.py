"""결과 이미지 생성: PIL로 랜드마크 + 각도 오버레이"""

import io
import base64

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    VERTEBRA_NAMES, LANDMARK_NAMES, N_LANDMARKS_PER,
    THORACIC_RANGE, LUMBAR_RANGE,
    SUP_ANT, SUP_POST, INF_ANT, INF_POST,
)

# 색상 팔레트
COLOR_THORACIC = (0, 180, 255)    # 파란색 (흉추)
COLOR_LUMBAR = (255, 140, 0)      # 주황색 (요추)
COLOR_ENDPLATE = (255, 255, 0)    # 노란색 (endplate 라인)
COLOR_COBB = (255, 50, 50)        # 빨간색 (Cobb angle)
COLOR_LABEL = (255, 255, 255)     # 흰색 (텍스트)
POINT_RADIUS = 3


def _get_color(vertebra_idx: int) -> tuple:
    if vertebra_idx in THORACIC_RANGE:
        return COLOR_THORACIC
    return COLOR_LUMBAR


def draw_landmarks(
    image: Image.Image,
    keypoints: np.ndarray,
    angles: dict,
) -> Image.Image:
    """
    원본 이미지에 랜드마크, endplate, 각도 오버레이

    Args:
        image: PIL Image (원본 크기)
        keypoints: (102, 2) 원본 좌표 [x, y]
        angles: compute_all_angles() 결과
    """
    # RGB 변환
    if image.mode != "RGB":
        image = image.convert("RGB")

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_large = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_large = font

    # 척추별 랜드마크 그리기
    for v_idx, v_name in enumerate(VERTEBRA_NAMES):
        color = _get_color(v_idx)
        base = v_idx * N_LANDMARKS_PER

        # 6개 랜드마크 점
        for lm_idx in range(N_LANDMARKS_PER):
            x, y = keypoints[base + lm_idx]
            draw.ellipse(
                [x - POINT_RADIUS, y - POINT_RADIUS,
                 x + POINT_RADIUS, y + POINT_RADIUS],
                fill=color,
            )

        # Superior endplate 라인 (anterior → posterior)
        sa = keypoints[base + SUP_ANT]
        sp = keypoints[base + SUP_POST]
        draw.line([tuple(sa), tuple(sp)], fill=COLOR_ENDPLATE, width=1)

        # Inferior endplate 라인
        ia = keypoints[base + INF_ANT]
        ip = keypoints[base + INF_POST]
        draw.line([tuple(ia), tuple(ip)], fill=COLOR_ENDPLATE, width=1)

        # 척추 이름 라벨 (pedicle 중심 근처)
        pl = keypoints[base + 4]  # pedicle_left
        pr = keypoints[base + 5]  # pedicle_right
        label_x = max(pl[0], pr[0]) + 8
        label_y = (pl[1] + pr[1]) / 2 - 6
        draw.text((label_x, label_y), v_name, fill=color, font=font)

    # Cobb angle 표시
    cobb = angles.get("cobb", {})
    if cobb.get("cobb_angle", 0) > 0:
        upper = cobb["upper_vertebra"]
        lower = cobb["lower_vertebra"]
        angle_val = cobb["cobb_angle"]

        # 이미지 우상단에 텍스트
        text = f"Cobb: {angle_val}\u00b0 ({upper}-{lower})"
        draw.text((10, 10), text, fill=COLOR_COBB, font=font_large)

    # Kyphosis / Lordosis 표시
    kyph = angles.get("kyphosis", {})
    lord = angles.get("lordosis", {})
    y_offset = 35
    if kyph.get("kyphosis_angle") is not None:
        draw.text((10, y_offset), f"Kyphosis: {kyph['kyphosis_angle']}\u00b0",
                  fill=COLOR_THORACIC, font=font_large)
        y_offset += 25
    if lord.get("lordosis_angle") is not None:
        draw.text((10, y_offset), f"Lordosis: {lord['lordosis_angle']}\u00b0",
                  fill=COLOR_LUMBAR, font=font_large)

    return overlay


def image_to_base64(image: Image.Image) -> str:
    """PIL Image → base64 PNG string"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
