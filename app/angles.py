"""Cobb angle, Kyphosis, Lordosis 계산"""

import math

import numpy as np

from .config import (
    VERTEBRA_NAMES, N_LANDMARKS_PER, N_VERTEBRAE,
    SUP_ANT, SUP_POST, INF_ANT, INF_POST,
)


def _endplate_angle(p_ant: np.ndarray, p_post: np.ndarray) -> float:
    """
    두 점(anterior, posterior)으로 정의되는 endplate 기울기 (degrees)

    수평선 대비 각도 반환 (반시계 양수)
    """
    dx = p_post[0] - p_ant[0]
    dy = p_post[1] - p_ant[1]
    return math.degrees(math.atan2(dy, dx))


def _angle_between_endplates(angle1: float, angle2: float) -> float:
    """두 endplate 간 각도 (절대값)"""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def _get_landmark(keypoints: np.ndarray, vertebra_idx: int, offset: int) -> np.ndarray:
    """특정 척추의 특정 랜드마크 좌표 반환"""
    idx = vertebra_idx * N_LANDMARKS_PER + offset
    return keypoints[idx]


def compute_cobb_angle(keypoints: np.ndarray) -> dict:
    """
    Cobb angle: 모든 상판/하판 조합 중 최대 각도

    상위 척추의 superior endplate vs 하위 척추의 inferior endplate
    """
    max_angle = 0.0
    upper_vert = ""
    lower_vert = ""

    for i in range(N_VERTEBRAE):
        sup_ant = _get_landmark(keypoints, i, SUP_ANT)
        sup_post = _get_landmark(keypoints, i, SUP_POST)
        angle_sup = _endplate_angle(sup_ant, sup_post)

        for j in range(i + 1, N_VERTEBRAE):
            inf_ant = _get_landmark(keypoints, j, INF_ANT)
            inf_post = _get_landmark(keypoints, j, INF_POST)
            angle_inf = _endplate_angle(inf_ant, inf_post)

            cobb = _angle_between_endplates(angle_sup, angle_inf)
            if cobb > max_angle:
                max_angle = cobb
                upper_vert = VERTEBRA_NAMES[i]
                lower_vert = VERTEBRA_NAMES[j]

    return {
        "cobb_angle": round(max_angle, 1),
        "upper_vertebra": upper_vert,
        "lower_vertebra": lower_vert,
    }


def compute_kyphosis(keypoints: np.ndarray) -> dict:
    """Kyphosis: T1 superior ↔ T12 inferior endplate 각도"""
    t1_idx = 0   # T1
    t12_idx = 11  # T12

    t1_sup_ant = _get_landmark(keypoints, t1_idx, SUP_ANT)
    t1_sup_post = _get_landmark(keypoints, t1_idx, SUP_POST)
    angle_t1 = _endplate_angle(t1_sup_ant, t1_sup_post)

    t12_inf_ant = _get_landmark(keypoints, t12_idx, INF_ANT)
    t12_inf_post = _get_landmark(keypoints, t12_idx, INF_POST)
    angle_t12 = _endplate_angle(t12_inf_ant, t12_inf_post)

    return {
        "kyphosis_angle": round(_angle_between_endplates(angle_t1, angle_t12), 1),
        "from": "T1_superior",
        "to": "T12_inferior",
    }


def compute_lordosis(keypoints: np.ndarray) -> dict:
    """Lordosis: L1 superior ↔ L5 inferior endplate 각도"""
    l1_idx = 12   # L1
    l5_idx = 16   # L5

    l1_sup_ant = _get_landmark(keypoints, l1_idx, SUP_ANT)
    l1_sup_post = _get_landmark(keypoints, l1_idx, SUP_POST)
    angle_l1 = _endplate_angle(l1_sup_ant, l1_sup_post)

    l5_inf_ant = _get_landmark(keypoints, l5_idx, INF_ANT)
    l5_inf_post = _get_landmark(keypoints, l5_idx, INF_POST)
    angle_l5 = _endplate_angle(l5_inf_ant, l5_inf_post)

    return {
        "lordosis_angle": round(_angle_between_endplates(angle_l1, angle_l5), 1),
        "from": "L1_superior",
        "to": "L5_inferior",
    }


def compute_segment_angles(keypoints: np.ndarray) -> list[dict]:
    """인접 척추 간 각도 (16개 세그먼트)"""
    segments = []
    for i in range(N_VERTEBRAE - 1):
        inf_ant = _get_landmark(keypoints, i, INF_ANT)
        inf_post = _get_landmark(keypoints, i, INF_POST)
        angle_inf = _endplate_angle(inf_ant, inf_post)

        sup_ant = _get_landmark(keypoints, i + 1, SUP_ANT)
        sup_post = _get_landmark(keypoints, i + 1, SUP_POST)
        angle_sup = _endplate_angle(sup_ant, sup_post)

        segments.append({
            "segment": f"{VERTEBRA_NAMES[i]}-{VERTEBRA_NAMES[i+1]}",
            "angle": round(_angle_between_endplates(angle_inf, angle_sup), 1),
        })
    return segments


def compute_all_angles(keypoints: np.ndarray) -> dict:
    """모든 각도 계산 통합"""
    return {
        "cobb": compute_cobb_angle(keypoints),
        "kyphosis": compute_kyphosis(keypoints),
        "lordosis": compute_lordosis(keypoints),
        "segments": compute_segment_angles(keypoints),
    }
