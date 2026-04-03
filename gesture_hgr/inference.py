from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from .config import load_json
from .utils import softmax_np


class TorchPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = 'cpu') -> None:
        raise RuntimeError('This release only includes the ONNX demo path. Checkpoint inference is not packaged.')

    def predict_proba(self, landmarks: np.ndarray, geom: np.ndarray) -> np.ndarray:
        raise RuntimeError('This release only includes the ONNX demo path.')


class ONNXPredictor:
    def __init__(self, onnx_path: str | Path, meta_path: str | Path, providers: Optional[List[str]] = None) -> None:
        self.meta = load_json(meta_path)
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

    def predict_proba(self, landmarks: np.ndarray, geom: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            None,
            {
                'landmarks': landmarks.astype(np.float32)[None, :],
                'geom': geom.astype(np.float32)[None, :],
            },
        )
        logits = outputs[0]
        return softmax_np(logits, axis=1)[0]


class TemporalGestureFilter:
    """Low-cost deployment-time stabilizer for webcam demos.

    It combines EMA probability smoothing with a small state machine so predictions do not
    flicker when MediaPipe landmarks jitter for a few frames.
    """

    def __init__(
        self,
        num_classes: int,
        no_gesture_idx: int,
        threshold: float = 0.6,
        ema_alpha: float = 0.7,
        stable_frames: int = 3,
        history_size: int = 12,
    ) -> None:
        self.num_classes = int(num_classes)
        self.no_gesture_idx = int(no_gesture_idx)
        self.threshold = float(threshold)
        self.ema_alpha = float(ema_alpha)
        self.stable_frames = int(stable_frames)
        self.history: Deque[np.ndarray] = deque(maxlen=max(1, int(history_size)))
        self.ema: Optional[np.ndarray] = None
        self.current_label = self.no_gesture_idx
        self.current_conf = 1.0
        self.candidate_label = self.no_gesture_idx
        self.candidate_count = 0

    def reset(self) -> Tuple[int, float, np.ndarray]:
        self.history.clear()
        self.ema = None
        self.current_label = self.no_gesture_idx
        self.current_conf = 1.0
        self.candidate_label = self.no_gesture_idx
        self.candidate_count = 0
        probs = np.zeros(self.num_classes, dtype=np.float32)
        probs[self.no_gesture_idx] = 1.0
        return self.current_label, self.current_conf, probs

    def update(self, probs: Optional[np.ndarray]) -> Tuple[int, float, np.ndarray]:
        if probs is None:
            return self.reset()

        probs = probs.astype(np.float32)
        self.history.append(probs)
        hist_avg = np.mean(np.stack(self.history, axis=0), axis=0)
        if self.ema is None:
            self.ema = hist_avg
        else:
            self.ema = self.ema_alpha * self.ema + (1.0 - self.ema_alpha) * hist_avg

        smooth = self.ema / np.clip(self.ema.sum(), 1e-8, None)
        pred_idx = int(np.argmax(smooth))
        conf = float(smooth[pred_idx])
        if conf < self.threshold:
            pred_idx = self.no_gesture_idx

        if pred_idx == self.current_label:
            self.current_conf = conf
            self.candidate_label = pred_idx
            self.candidate_count = 0
            return self.current_label, self.current_conf, smooth

        if pred_idx == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = pred_idx
            self.candidate_count = 1

        if self.candidate_count >= self.stable_frames:
            self.current_label = pred_idx
            self.current_conf = conf
            self.candidate_count = 0

        return self.current_label, self.current_conf, smooth
