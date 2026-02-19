"""상수 정의: 척추 이름, 랜드마크 인덱스 매핑"""

# 17개 척추 (T1~L5)
VERTEBRA_NAMES = [
    "T1", "T2", "T3", "T4", "T5", "T6",
    "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
]

# 각 척추당 6개 랜드마크
LANDMARK_NAMES = [
    "superior_anterior",
    "superior_posterior",
    "inferior_anterior",
    "inferior_posterior",
    "pedicle_left",
    "pedicle_right",
]

N_VERTEBRAE = len(VERTEBRA_NAMES)       # 17
N_LANDMARKS_PER = len(LANDMARK_NAMES)   # 6
N_LANDMARKS = N_VERTEBRAE * N_LANDMARKS_PER  # 102

IMAGE_SIZE = 512


def vertebra_landmark_indices(vertebra_idx: int) -> slice:
    """척추 인덱스 → 랜드마크 슬라이스 (i*6 : i*6+6)"""
    start = vertebra_idx * N_LANDMARKS_PER
    return slice(start, start + N_LANDMARKS_PER)


def landmark_name(vertebra_idx: int, landmark_idx: int) -> str:
    """전체 랜드마크 이름: e.g. 'T1_superior_anterior'"""
    return f"{VERTEBRA_NAMES[vertebra_idx]}_{LANDMARK_NAMES[landmark_idx]}"


# 흉추/요추 인덱스 범위
THORACIC_RANGE = range(0, 12)   # T1-T12
LUMBAR_RANGE = range(12, 17)    # L1-L5

# 각도 계산용 랜드마크 오프셋 (각 척추 내)
SUP_ANT = 0   # superior_anterior
SUP_POST = 1  # superior_posterior
INF_ANT = 2   # inferior_anterior
INF_POST = 3  # inferior_posterior
