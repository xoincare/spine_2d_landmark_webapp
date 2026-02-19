"""
SpineLandmarkNet 모델 정의 (phase2_landmark/train.py에서 복사)

추론 전용: pretrained=False (checkpoint에서 로드)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapHead(nn.Module):
    """히트맵 출력 헤드: 각 랜드마크 위치를 가우시안 히트맵으로 예측"""

    def __init__(self, in_channels: int, n_landmarks: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, n_landmarks, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SimpleCNNBackbone(nn.Module):
    """timm 없을 때 사용하는 간단한 CNN 백본"""

    def __init__(self, in_channels: int = 1, out_channels: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 64, stride=2),
            self._block(64, 128, stride=2),
            self._block(128, 256, stride=2),
            self._block(256, out_channels, stride=1),
        )

    def _block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


class SpineLandmarkNet(nn.Module):
    """
    척추 랜드마크 탐지 네트워크

    백본 + 히트맵 헤드 구조
    입력: X-ray 이미지 (1, H, W)
    출력: 히트맵 (N_landmarks, H/4, W/4)
    """

    def __init__(
        self,
        n_landmarks: int = 102,
        backbone: str = "hrnet_w48",
        pretrained: bool = False,
    ):
        super().__init__()
        self.n_landmarks = n_landmarks

        if backbone == "simple":
            self.backbone = SimpleCNNBackbone(in_channels=1, out_channels=256)
            backbone_channels = 256
        else:
            try:
                import timm
                self.backbone = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    features_only=True,
                    in_chans=1,
                    out_indices=(4,),
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, 256, 256)
                    feat = self.backbone(dummy)
                    backbone_channels = feat[-1].shape[1]
            except ImportError:
                self.backbone = SimpleCNNBackbone(in_channels=1, out_channels=256)
                backbone_channels = 256

        self.heatmap_head = HeatmapHead(backbone_channels, n_landmarks)

    def forward(self, x: torch.Tensor) -> dict:
        if isinstance(self.backbone, nn.Module) and hasattr(self.backbone, 'forward_features'):
            features = self.backbone(x)
            if isinstance(features, list):
                features = features[-1]
        else:
            features = self.backbone(x)

        heatmaps = self.heatmap_head(features)
        keypoints = self._soft_argmax(heatmaps, x.shape[-2:])

        return {"heatmaps": heatmaps, "keypoints": keypoints}

    def _soft_argmax(self, heatmaps: torch.Tensor, original_size: tuple) -> torch.Tensor:
        B, N, H, W = heatmaps.shape

        heatmaps_flat = heatmaps.reshape(B, N, -1)
        probs = F.softmax(heatmaps_flat, dim=-1)
        probs = probs.reshape(B, N, H, W)

        device = heatmaps.device
        y_grid = torch.arange(H, device=device, dtype=torch.float32)
        x_grid = torch.arange(W, device=device, dtype=torch.float32)

        x_coords = (probs.sum(dim=2) * x_grid).sum(dim=-1)
        y_coords = (probs.sum(dim=3) * y_grid).sum(dim=-1)

        x_coords = x_coords * (original_size[1] / W)
        y_coords = y_coords * (original_size[0] / H)

        keypoints = torch.stack([x_coords, y_coords], dim=-1)
        return keypoints
