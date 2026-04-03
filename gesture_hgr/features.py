from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import cv2
import numpy as np

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

HAND_BONES = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
ANGLE_TRIPLETS = [
    (WRIST, THUMB_CMC, THUMB_MCP),
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP, INDEX_PIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP, RING_PIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP, PINKY_PIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]

HAND_LANDMARKER_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
)
DEFAULT_HAND_LANDMARKER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / 'data' / 'mediapipe_models' / 'hand_landmarker.task'
)


@dataclass
class ExtractedFeatures:
    landmarks: np.ndarray  # (63,)
    geom: np.ndarray       # (G,)
    detected: bool
    handedness: str
    score: float


def _ensure_rgb(image: Any) -> np.ndarray:
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))
    else:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected an RGB/BGR image with 3 channels.')
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba)) + 1e-6
    nbc = float(np.linalg.norm(bc)) + 1e-6
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.arccos(cosv))


def canonicalize_landmarks(landmarks: np.ndarray, handedness: str) -> Tuple[np.ndarray, float]:
    """Palm-centric 3D canonical frame.

    1. Mirror left hands into the right-hand coordinate convention.
    2. Translate wrist to the origin.
    3. Build a local palm basis from index-pinky axis and wrist-middle axis.
    4. Normalize by palm scale.
    """
    coords = landmarks.astype(np.float32).copy()
    if handedness.lower().startswith('left'):
        coords[:, 0] = 1.0 - coords[:, 0]

    coords = coords - coords[WRIST]

    x_axis = coords[INDEX_MCP] - coords[PINKY_MCP]
    x_axis = _safe_unit(x_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    y_seed = coords[MIDDLE_MCP]
    y_seed = y_seed - float(np.dot(y_seed, x_axis)) * x_axis
    y_axis = _safe_unit(y_seed)
    if float(np.linalg.norm(y_axis)) < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    z_axis = _safe_unit(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = _safe_unit(np.cross(z_axis, x_axis))

    basis = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    canonical = coords @ basis

    palm_width = float(np.linalg.norm(canonical[INDEX_MCP] - canonical[PINKY_MCP]))
    palm_height = float(np.linalg.norm(canonical[MIDDLE_MCP] - canonical[WRIST]))
    palm_scale = max((palm_width + palm_height) * 0.5, 1e-6)
    canonical = canonical / palm_scale
    return canonical.astype(np.float32), float(palm_scale)


def geometric_features(canonical: np.ndarray, handedness: str, score: float) -> np.ndarray:
    features: List[float] = []

    # Joint articulation angles (15).
    for a, b, c in ANGLE_TRIPLETS:
        features.append(_angle(canonical[a], canonical[b], canonical[c]))

    # Bone lengths (20).
    for a, b in HAND_BONES:
        features.append(float(np.linalg.norm(canonical[b] - canonical[a])))

    # Fingertip pairwise distances (10).
    tips = canonical[FINGERTIPS]
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            features.append(float(np.linalg.norm(tips[i] - tips[j])))

    # Fingertip-to-wrist distances (5).
    for idx in FINGERTIPS:
        features.append(float(np.linalg.norm(canonical[idx] - canonical[WRIST])))

    # Finger extension (tip to MCP distances, 5).
    for tip_idx, mcp_idx in zip(FINGERTIPS, MCPS):
        features.append(float(np.linalg.norm(canonical[tip_idx] - canonical[mcp_idx])))

    # Finger spread angles relative to the wrist (10).
    wrist = canonical[WRIST]
    rays = canonical[FINGERTIPS] - wrist
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            vi = rays[i]
            vj = rays[j]
            ni = float(np.linalg.norm(vi)) + 1e-6
            nj = float(np.linalg.norm(vj)) + 1e-6
            cosv = float(np.dot(vi, vj) / (ni * nj))
            cosv = float(np.clip(cosv, -1.0, 1.0))
            features.append(float(np.arccos(cosv)))

    # Palm frame descriptors (9).
    palm_vec1 = canonical[INDEX_MCP] - canonical[WRIST]
    palm_vec2 = canonical[PINKY_MCP] - canonical[WRIST]
    palm_normal = np.cross(palm_vec1, palm_vec2)
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
    features.extend(palm_vec1.tolist())
    features.extend(palm_vec2.tolist())
    features.extend(palm_normal.tolist())

    # Scalar palm descriptors + reliability (5).
    palm_width = float(np.linalg.norm(canonical[INDEX_MCP] - canonical[PINKY_MCP]))
    palm_height = float(np.linalg.norm(canonical[MIDDLE_MCP] - canonical[WRIST]))
    features.append(palm_width)
    features.append(palm_height)
    features.append(palm_width / max(palm_height, 1e-6))
    handedness_value = -1.0 if handedness.lower().startswith('left') else 1.0
    features.append(handedness_value)
    features.append(float(score))

    return np.asarray(features, dtype=np.float32)


def zero_feature_tensors() -> Tuple[np.ndarray, np.ndarray]:
    landmarks = np.zeros(63, dtype=np.float32)
    geom = np.zeros(15 + 20 + 10 + 5 + 5 + 10 + 9 + 5, dtype=np.float32)
    return landmarks, geom


def feature_dims() -> Dict[str, int]:
    landmarks, geom = zero_feature_tensors()
    return {'landmarks_dim': int(landmarks.shape[0]), 'geom_dim': int(geom.shape[0])}


def _resolve_task_delegate(mp: Any):
    delegate_name = os.environ.get('MEDIAPIPE_TASK_DELEGATE', 'CPU').strip().upper()
    if delegate_name not in {'CPU', 'GPU'}:
        raise ValueError(f'Unsupported MEDIAPIPE_TASK_DELEGATE: {delegate_name}')
    return getattr(mp.tasks.BaseOptions.Delegate, delegate_name)


class MediaPipeFeatureExtractor:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._backend = 'unknown'
        self._video_timestamp_ms = 0
        self._static_image_mode = bool(static_image_mode)
        self._mp = mp

        if hasattr(mp, 'solutions'):
            self._backend = 'solutions'
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        if hasattr(mp, 'tasks') and hasattr(mp.tasks, 'vision'):
            self._backend = 'tasks'
            model_path = self._ensure_hand_landmarker_model()
            running_mode = mp.tasks.vision.RunningMode.IMAGE if static_image_mode else mp.tasks.vision.RunningMode.VIDEO
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=_resolve_task_delegate(mp),
                ),
                running_mode=running_mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
            return

        raise RuntimeError('Unsupported MediaPipe package: neither solutions nor tasks vision API is available.')

    def _ensure_hand_landmarker_model(self) -> Path:
        model_override = Path(
            os.environ.get('MEDIAPIPE_HAND_LANDMARKER_MODEL', str(DEFAULT_HAND_LANDMARKER_MODEL_PATH))
        ).expanduser().resolve()
        if model_override.exists():
            return model_override

        model_override.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = model_override.with_suffix(model_override.suffix + '.tmp')
        with urllib.request.urlopen(HAND_LANDMARKER_MODEL_URL, timeout=120) as response:
            with tmp_path.open('wb') as f:
                shutil.copyfileobj(response, f)
        tmp_path.replace(model_override)
        return model_override

    def close(self) -> None:
        if hasattr(self.hands, 'close'):
            self.hands.close()

    def __enter__(self) -> 'MediaPipeFeatureExtractor':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _best_hand(self, results: Any) -> Optional[Tuple[np.ndarray, str, float]]:
        if self._backend == 'tasks':
            if not getattr(results, 'hand_landmarks', None):
                return None

            best_idx = 0
            best_score = -1.0
            best_label = 'Right'
            for i, handedness_list in enumerate(results.handedness):
                classification = handedness_list[0]
                score = float(classification.score or 0.0)
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_label = classification.category_name or classification.display_name or 'Right'

            coords = []
            for lm in results.hand_landmarks[best_idx]:
                coords.append([lm.x, lm.y, lm.z])
            return np.asarray(coords, dtype=np.float32), best_label, float(best_score)

        if not getattr(results, 'multi_hand_landmarks', None):
            return None

        best_idx = 0
        best_score = -1.0
        best_label = 'Right'
        for i, handedness in enumerate(results.multi_handedness):
            classification = handedness.classification[0]
            score = float(classification.score)
            if score > best_score:
                best_score = score
                best_idx = i
                best_label = classification.label

        landmarks = results.multi_hand_landmarks[best_idx]
        coords = []
        for lm in landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        return np.asarray(coords, dtype=np.float32), best_label, float(best_score)

    def extract(self, image: Any, assume_bgr: bool = False) -> ExtractedFeatures:
        rgb = _ensure_rgb(image)
        if assume_bgr:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if self._backend == 'tasks':
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            if self._static_image_mode:
                results = self.hands.detect(mp_image)
            else:
                self._video_timestamp_ms += 33
                results = self.hands.detect_for_video(mp_image, self._video_timestamp_ms)
        else:
            results = self.hands.process(rgb)
        best = self._best_hand(results)
        if best is None:
            landmarks_zero, geom_zero = zero_feature_tensors()
            return ExtractedFeatures(
                landmarks=landmarks_zero,
                geom=geom_zero,
                detected=False,
                handedness='Unknown',
                score=0.0,
            )

        coords, handedness, score = best
        canonical, _ = canonicalize_landmarks(coords, handedness)
        geom = geometric_features(canonical, handedness, score)
        return ExtractedFeatures(
            landmarks=canonical.reshape(-1).astype(np.float32),
            geom=geom.astype(np.float32),
            detected=True,
            handedness=handedness,
            score=score,
        )
