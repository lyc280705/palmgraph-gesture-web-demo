from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory
import numpy as np

from .config import load_json, save_json
from .features import MediaPipeFeatureExtractor
from .inference import ONNXPredictor, TemporalGestureFilter, TorchPredictor
from .utils import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
WEB_ASSET_DIR = ROOT / 'assets' / 'web_demo'

GESTURE_LABELS_ZH = {
    'call': '呼叫',
    'dislike': '反对',
    'fist': '握拳',
    'like': '点赞',
    'no_gesture': '无手势',
    'ok': '确认',
    'palm': '手掌',
    'peace': '剪刀手',
}

ACTION_DEFINITIONS: List[Dict[str, Any]] = [
    {
        'id': 'noop',
        'title': '无绑定',
        'description': '不执行任何系统动作。',
        'fields': [],
    },
    {
        'id': 'notify',
        'title': '桌面通知',
        'description': '在当前系统上弹出一条桌面通知或提示。',
        'fields': [
            {'name': 'title', 'label': '通知标题', 'placeholder': 'PalmGraph-MoE'},
            {'name': 'message', 'label': '通知内容', 'placeholder': '已识别到手势'},
        ],
    },
    {
        'id': 'toggle_control_with_notice',
        'title': '通知并切换控制',
        'description': '弹出桌面通知，同时切换手势控制的启用/关闭状态。',
        'fields': [
            {'name': 'title', 'label': '通知标题', 'placeholder': 'PalmGraph-MoE'},
            {'name': 'enabled_message', 'label': '启动通知', 'placeholder': '手势控制已启动'},
            {'name': 'disabled_message', 'label': '关闭通知', 'placeholder': '手势控制已关闭'},
            {'name': 'lockout_seconds', 'label': '锁定秒数', 'placeholder': '2.8'},
        ],
    },
    {
        'id': 'volume_up',
        'title': '系统音量增加',
        'description': '使用当前系统可用的媒体键增加音量。',
        'fields': [],
    },
    {
        'id': 'volume_down',
        'title': '系统音量降低',
        'description': '使用当前系统可用的媒体键降低音量。',
        'fields': [],
    },
    {
        'id': 'mute_toggle',
        'title': '切换静音',
        'description': '使用当前系统可用的媒体键切换静音。',
        'fields': [],
    },
    {
        'id': 'play_pause',
        'title': '播放暂停',
        'description': '使用当前系统可用的媒体键触发播放/暂停。',
        'fields': [],
    },
    {
        'id': 'media_play',
        'title': '播放',
        'description': '使用当前系统可用的媒体键触发播放。',
        'fields': [],
    },
    {
        'id': 'media_pause',
        'title': '暂停',
        'description': '使用当前系统可用的媒体键触发暂停。',
        'fields': [],
    },
    {
        'id': 'next_item',
        'title': '下一项',
        'description': '使用当前系统可用的媒体键发送下一项。',
        'fields': [],
    },
    {
        'id': 'previous_item',
        'title': '上一项',
        'description': '使用当前系统可用的媒体键发送上一项。',
        'fields': [],
    },
    {
        'id': 'fullscreen_toggle',
        'title': '切换全屏',
        'description': '发送当前系统常见的全屏快捷键。',
        'fields': [],
    },
    {
        'id': 'stop_key',
        'title': '停止或返回',
        'description': '发送 Escape 以停止、返回或退出全屏。',
        'fields': [],
    },
    {
        'id': 'screenshot',
        'title': '系统截图',
        'description': '把当前屏幕保存到图片目录。',
        'fields': [
            {'name': 'directory', 'label': '保存目录', 'placeholder': '~/Pictures/PalmGraphDemo'},
        ],
    },
    {
        'id': 'hotkey',
        'title': '自定义快捷键',
        'description': '向前台应用发送自定义按键或组合键。',
        'fields': [
            {'name': 'key', 'label': '按键', 'placeholder': 'space / right / a / 1'},
            {'name': 'modifiers', 'label': '修饰键', 'placeholder': 'command,control'},
            {'name': 'repeat', 'label': '重复次数', 'placeholder': '1'},
        ],
    },
    {
        'id': 'shell',
        'title': '自定义命令',
        'description': '运行一条本机 shell 命令。',
        'fields': [
            {'name': 'command', 'label': '命令', 'placeholder': '例如：xdg-open .'},
        ],
    },
    {
        'id': 'applescript',
        'title': '自定义 AppleScript',
        'description': '仅 macOS：运行一段 AppleScript。',
        'fields': [
            {
                'name': 'script',
                'label': '脚本内容',
                'placeholder': 'display notification "PalmGraph-MoE" with title "手势触发"',
                'multiline': True,
            },
        ],
    },
]

ACTION_DEFINITION_MAP = {item['id']: item for item in ACTION_DEFINITIONS}

SPECIAL_KEY_CODES = {
    'space': 49,
    'left': 123,
    'right': 124,
    'down': 125,
    'up': 126,
    'escape': 53,
    'return': 36,
    'enter': 76,
    'tab': 48,
    'delete': 51,
}

MODIFIER_TOKENS = {
    'command': 'command down',
    'cmd': 'command down',
    'control': 'control down',
    'ctrl': 'control down',
    'option': 'option down',
    'alt': 'option down',
    'shift': 'shift down',
}

WINDOWS_MODIFIER_KEYS = {
    'command': 0x5B,
    'cmd': 0x5B,
    'win': 0x5B,
    'windows': 0x5B,
    'control': 0x11,
    'ctrl': 0x11,
    'option': 0x12,
    'alt': 0x12,
    'shift': 0x10,
}

WINDOWS_SPECIAL_KEYS = {
    'space': 0x20,
    'left': 0x25,
    'up': 0x26,
    'right': 0x27,
    'down': 0x28,
    'escape': 0x1B,
    'return': 0x0D,
    'enter': 0x0D,
    'tab': 0x09,
    'delete': 0x2E,
    'backspace': 0x08,
    'f11': 0x7A,
}

LINUX_MODIFIER_TOKENS = {
    'command': 'super',
    'cmd': 'super',
    'win': 'super',
    'windows': 'super',
    'control': 'ctrl',
    'ctrl': 'ctrl',
    'option': 'alt',
    'alt': 'alt',
    'shift': 'shift',
}

LINUX_SPECIAL_KEYS = {
    'space': 'space',
    'left': 'Left',
    'right': 'Right',
    'down': 'Down',
    'up': 'Up',
    'escape': 'Escape',
    'return': 'Return',
    'enter': 'Return',
    'tab': 'Tab',
    'delete': 'Delete',
    'backspace': 'BackSpace',
    'f11': 'F11',
}

MEDIA_KEY_TYPES = {
    'volume_up': 0,
    'volume_down': 1,
    'mute_toggle': 7,
    'play_pause': 16,
    'next_track': 17,
    'previous_track': 18,
}

# macOS CGEvent modifier flag masks
_CGEVENT_FLAG_CMD = 1 << 20
_CGEVENT_FLAG_CTRL = 1 << 18
LINUX_XDOTOOL_ERROR = 'Hotkey and media-key actions require xdotool on Linux. Install it with your package manager (e.g. `sudo apt install xdotool`, `sudo dnf install xdotool`, or `sudo pacman -S xdotool`).'
LINUX_NOTIFY_ERROR = 'Desktop notifications require notify-send or zenity on Linux. Install libnotify-bin or zenity with your package manager (e.g. `sudo apt install libnotify-bin`, `sudo dnf install libnotify`, or `sudo pacman -S libnotify`).'
LINUX_SCREENSHOT_ERROR = 'Screenshot actions require gnome-screenshot, scrot, import (ImageMagick), or grim on Linux. Install one with your package manager (e.g. `sudo apt install gnome-screenshot`, `sudo dnf install gnome-screenshot`, or `sudo pacman -S gnome-screenshot`).'


def translate_gesture(label: str) -> str:
    return GESTURE_LABELS_ZH.get(label, label)


def action_title(action_id: str) -> str:
    return ACTION_DEFINITION_MAP.get(action_id, {}).get('title', action_id)


def model_option_title(model_path: Path) -> str:
    lower_name = model_path.name.lower()
    if 'int8' in lower_name:
        return 'INT8 ONNX'
    if lower_name.endswith('.onnx'):
        return 'FP32 ONNX'
    return model_path.name


def discover_model_options(
    checkpoint_path: Optional[Path],
    onnx_path: Optional[Path],
    meta_path: Optional[Path],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if checkpoint_path is not None:
        return [
            {
                'id': 'checkpoint',
                'title': 'PyTorch Checkpoint',
                'kind': 'checkpoint',
                'model_path': str(checkpoint_path),
                'meta_path': None,
                'file_name': checkpoint_path.name,
                'size_bytes': int(checkpoint_path.stat().st_size) if checkpoint_path.exists() else 0,
            }
        ]

    if onnx_path is None or meta_path is None:
        return []

    seen_paths: set[str] = set()
    options: List[Dict[str, Any]] = []

    def add_option(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if str(resolved) in seen_paths or not resolved.exists() or resolved.suffix.lower() != '.onnx':
            return
        seen_paths.add(str(resolved))
        options.append(
            {
                'id': resolved.name,
                'title': model_option_title(resolved),
                'kind': 'onnx',
                'model_path': str(resolved),
                'meta_path': str(meta_path),
                'file_name': resolved.name,
                'size_bytes': int(resolved.stat().st_size),
            }
        )

    add_option(onnx_path.with_name('model.int8.onnx'))
    add_option(onnx_path)
    add_option(onnx_path.with_name('model.onnx'))

    quantized_name = str(meta.get('quantized_model', '') or '').strip()
    if quantized_name:
        add_option(meta_path.with_name(quantized_name))

    return options


def preferred_startup_onnx_path(onnx_path: Optional[Path], meta_path: Optional[Path]) -> Optional[Path]:
    if onnx_path is None or meta_path is None:
        return onnx_path

    quantized_candidate = meta_path.with_name('model.int8.onnx')
    if quantized_candidate.exists():
        return quantized_candidate.resolve()
    return onnx_path


def _default_binding_for_command(command_name: str, gesture: str) -> Dict[str, Any]:
    aliases: Dict[str, Dict[str, Any]] = {
        'idle': {'enabled': False, 'action_id': 'noop', 'params': {}},
        'volume_up': {'enabled': True, 'action_id': 'volume_up', 'params': {}},
        'volume_down': {'enabled': True, 'action_id': 'volume_down', 'params': {}},
        'mute_toggle': {'enabled': True, 'action_id': 'mute_toggle', 'params': {}},
        'play_pause': {'enabled': True, 'action_id': 'play_pause', 'params': {}},
        'fullscreen_toggle': {'enabled': True, 'action_id': 'fullscreen_toggle', 'params': {}},
        'next_video': {'enabled': True, 'action_id': 'next_item', 'params': {}},
        'next_scene': {'enabled': True, 'action_id': 'next_item', 'params': {}},
        'previous_video': {'enabled': True, 'action_id': 'previous_item', 'params': {}},
        'stop': {'enabled': True, 'action_id': 'stop_key', 'params': {}},
        'confirm': {'enabled': True, 'action_id': 'notify', 'params': {'title': 'PalmGraph-MoE', 'message': '确认手势已触发'}},
        'wake_video_control': {'enabled': True, 'action_id': 'notify', 'params': {'title': 'PalmGraph-MoE', 'message': '视频控制模式已唤醒'}},
        'wake_assistant': {'enabled': True, 'action_id': 'notify', 'params': {'title': 'PalmGraph-MoE', 'message': '助手唤醒手势已触发'}},
        'turn_on': {'enabled': True, 'action_id': 'notify', 'params': {'title': 'PalmGraph-MoE', 'message': '已识别开启动作，请按需绑定真实控制'}},
        'turn_off': {'enabled': True, 'action_id': 'notify', 'params': {'title': 'PalmGraph-MoE', 'message': '已识别关闭动作，请按需绑定真实控制'}},
        'toggle_mode': {'enabled': True, 'action_id': 'notify', 'params': {'title': 'PalmGraph-MoE', 'message': '模式切换手势已触发'}},
    }
    if command_name in aliases:
        return aliases[command_name]
    return {
        'enabled': gesture != 'no_gesture',
        'action_id': 'notify',
        'params': {
            'title': 'PalmGraph-MoE',
            'message': f'已识别命令: {command_name}',
        },
    }


@dataclass
class ActionEvent:
    time_text: str
    gesture: str
    gesture_zh: str
    action_id: str
    action_title: str
    success: bool
    detail: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': self.time_text,
            'gesture': self.gesture,
            'gesture_zh': self.gesture_zh,
            'action_id': self.action_id,
            'action_title': self.action_title,
            'success': self.success,
            'detail': self.detail,
            'source': self.source,
        }


class GestureActionController:
    def __init__(
        self,
        config_path: Path,
        class_names: List[str],
        command_map: Dict[str, str],
        no_gesture_label: str,
    ) -> None:
        self.config_path = config_path
        self.class_names = list(class_names)
        self.command_map = dict(command_map)
        self.no_gesture_label = no_gesture_label
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='gesture-action')
        self.recent_events: Deque[ActionEvent] = deque(maxlen=18)
        self.last_event: Optional[ActionEvent] = None
        self.last_trigger_label = no_gesture_label
        self.last_trigger_ts = 0.0
        self.pending_trigger_label = no_gesture_label
        self.pending_trigger_ts = 0.0
        self.config = self._load_or_create_config()
        atexit.register(self.close)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _load_or_create_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            data = load_json(self.config_path)
        else:
            data = {
                'control_enabled': True,
                'cooldown_seconds': 1.2,
                'bindings': {},
            }

        bindings = dict(data.get('bindings', {}))
        merged_bindings: Dict[str, Dict[str, Any]] = {}
        for gesture in self.class_names:
            if gesture in bindings:
                merged_bindings[gesture] = self._sanitize_binding(bindings[gesture], gesture)
                continue

            command_name = self.command_map.get(gesture, 'idle' if gesture == self.no_gesture_label else gesture)
            merged_bindings[gesture] = self._sanitize_binding(_default_binding_for_command(command_name, gesture), gesture)

        data['bindings'] = merged_bindings
        data['control_enabled'] = bool(data.get('control_enabled', True))
        data['cooldown_seconds'] = float(data.get('cooldown_seconds', 1.2))
        data['confidence_threshold'] = float(data.get('confidence_threshold', 0.7))
        data['trigger_delay_seconds'] = float(data.get('trigger_delay_seconds', 0.2))
        save_json(data, self.config_path)
        return data

    def _sanitize_binding(self, binding: Dict[str, Any], gesture: str) -> Dict[str, Any]:
        action_id = str(binding.get('action_id', 'noop')).strip() or 'noop'
        if action_id not in ACTION_DEFINITION_MAP:
            action_id = 'noop'

        params = binding.get('params', {})
        if not isinstance(params, dict):
            params = {}

        clean_params: Dict[str, str] = {}
        for key, value in params.items():
            clean_params[str(key)] = str(value)

        enabled_default = gesture != self.no_gesture_label and action_id != 'noop'
        return {
            'enabled': bool(binding.get('enabled', enabled_default)),
            'action_id': action_id,
            'params': clean_params,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'control_enabled': bool(self.config.get('control_enabled', True)),
                'cooldown_seconds': float(self.config.get('cooldown_seconds', 1.2)),
                'confidence_threshold': float(self.config.get('confidence_threshold', 0.7)),
                'trigger_delay_seconds': float(self.config.get('trigger_delay_seconds', 0.2)),
                'bindings': json.loads(json.dumps(self.config.get('bindings', {}), ensure_ascii=False)),
            }

    def update_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any]
        with self.lock:
            self.config['control_enabled'] = bool(payload.get('control_enabled', self.config.get('control_enabled', True)))
            cooldown = float(payload.get('cooldown_seconds', self.config.get('cooldown_seconds', 1.2)))
            self.config['cooldown_seconds'] = max(0.2, min(10.0, cooldown))
            conf_thresh = float(payload.get('confidence_threshold', self.config.get('confidence_threshold', 0.7)))
            self.config['confidence_threshold'] = max(0.0, min(1.0, conf_thresh))
            trigger_delay = float(payload.get('trigger_delay_seconds', self.config.get('trigger_delay_seconds', 0.2)))
            self.config['trigger_delay_seconds'] = max(0.0, min(2.0, trigger_delay))

            bindings_payload = payload.get('bindings', {})
            if isinstance(bindings_payload, dict):
                merged = dict(self.config.get('bindings', {}))
                for gesture in self.class_names:
                    if gesture in bindings_payload:
                        merged[gesture] = self._sanitize_binding(bindings_payload[gesture], gesture)
                self.config['bindings'] = merged

            save_json(self.config, self.config_path)
            snapshot = {
                'control_enabled': bool(self.config.get('control_enabled', True)),
                'cooldown_seconds': float(self.config.get('cooldown_seconds', 1.2)),
                'confidence_threshold': float(self.config.get('confidence_threshold', 0.7)),
                'trigger_delay_seconds': float(self.config.get('trigger_delay_seconds', 0.2)),
                'bindings': json.loads(json.dumps(self.config.get('bindings', {}), ensure_ascii=False)),
            }
        return snapshot

    def describe_binding(self, gesture: str) -> Dict[str, Any]:
        with self.lock:
            binding = dict(self.config.get('bindings', {}).get(gesture, {'enabled': False, 'action_id': 'noop', 'params': {}}))
        return {
            'enabled': bool(binding.get('enabled', False)),
            'action_id': str(binding.get('action_id', 'noop')),
            'action_title': action_title(str(binding.get('action_id', 'noop'))),
            'params': dict(binding.get('params', {})),
        }

    def recent_events_snapshot(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [item.to_dict() for item in reversed(self.recent_events)]

    def last_event_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return None if self.last_event is None else self.last_event.to_dict()

    def update_idle_state(self, gesture: str) -> None:
        if gesture == self.no_gesture_label:
            with self.lock:
                self.last_trigger_label = self.no_gesture_label
                self.pending_trigger_label = self.no_gesture_label
                self.pending_trigger_ts = 0.0

    def maybe_queue_action(self, gesture: str, confidence: float) -> Optional[Dict[str, Any]]:
        with self.lock:
            if gesture == self.no_gesture_label:
                self.last_trigger_label = self.no_gesture_label
                self.pending_trigger_label = self.no_gesture_label
                self.pending_trigger_ts = 0.0
                return None

            binding = dict(self.config.get('bindings', {}).get(gesture, {'enabled': False, 'action_id': 'noop', 'params': {}}))
            action_id = str(binding.get('action_id', 'noop'))
            is_toggle = action_id == 'toggle_control_with_notice'

            if not is_toggle and not bool(self.config.get('control_enabled', True)):
                self.pending_trigger_label = self.no_gesture_label
                self.pending_trigger_ts = 0.0
                return None

            conf_threshold = float(self.config.get('confidence_threshold', 0.7))
            if confidence < conf_threshold:
                self.pending_trigger_label = self.no_gesture_label
                self.pending_trigger_ts = 0.0
                return None

            now = time.monotonic()
            trigger_delay_seconds = float(self.config.get('trigger_delay_seconds', 0.2))

            if gesture == self.last_trigger_label:
                self.pending_trigger_label = gesture
                return None

            if gesture != self.pending_trigger_label:
                self.pending_trigger_label = gesture
                self.pending_trigger_ts = now
                return None

            if now - self.pending_trigger_ts < trigger_delay_seconds:
                return None

            if is_toggle:
                lockout = float(binding.get('params', {}).get('lockout_seconds', '2.8') or '2.8')
                effective_cooldown = max(lockout, float(self.config.get('cooldown_seconds', 1.2)))
            else:
                effective_cooldown = float(self.config.get('cooldown_seconds', 1.2))

            if now - self.last_trigger_ts < effective_cooldown:
                return None

            self.last_trigger_label = gesture
            self.last_trigger_ts = now

        if not binding.get('enabled', False) or binding.get('action_id', 'noop') == 'noop':
            return {
                'gesture': gesture,
                'gesture_zh': translate_gesture(gesture),
                'action_id': str(binding.get('action_id', 'noop')),
                'action_title': action_title(str(binding.get('action_id', 'noop'))),
                'accepted': False,
                'confidence': float(confidence),
            }

        action_id = str(binding.get('action_id', 'noop'))
        params = dict(binding.get('params', {}))
        self.executor.submit(self._execute_and_record, gesture, action_id, params, 'auto')
        return {
            'gesture': gesture,
            'gesture_zh': translate_gesture(gesture),
            'action_id': action_id,
            'action_title': action_title(action_id),
            'accepted': True,
            'confidence': float(confidence),
        }

    def execute_binding_for_gesture(self, gesture: str) -> Dict[str, Any]:
        with self.lock:
            if gesture not in self.class_names:
                raise KeyError(f'Unknown gesture: {gesture}')
            binding = dict(self.config.get('bindings', {}).get(gesture, {'enabled': False, 'action_id': 'noop', 'params': {}}))

        action_id = str(binding.get('action_id', 'noop'))
        params = dict(binding.get('params', {}))
        event = self._execute_action(gesture, action_id, params, 'manual')
        self._record_event(event)
        return event.to_dict()

    def execute_preview_binding(self, gesture: str, binding: Dict[str, Any]) -> Dict[str, Any]:
        if gesture not in self.class_names:
            raise KeyError(f'Unknown gesture: {gesture}')
        clean_binding = self._sanitize_binding(binding, gesture)
        action_id = str(clean_binding.get('action_id', 'noop'))
        params = dict(clean_binding.get('params', {}))
        event = self._execute_action(gesture, action_id, params, 'manual')
        self._record_event(event)
        return event.to_dict()

    def _record_event(self, event: ActionEvent) -> None:
        with self.lock:
            self.last_event = event
            self.recent_events.append(event)

    def _execute_and_record(self, gesture: str, action_id: str, params: Dict[str, Any], source: str) -> None:
        event = self._execute_action(gesture, action_id, params, source)
        self._record_event(event)

    def _execute_action(self, gesture: str, action_id: str, params: Dict[str, Any], source: str) -> ActionEvent:
        time_text = datetime.now().strftime('%H:%M:%S')
        title = action_title(action_id)
        try:
            if action_id == 'toggle_control_with_notice':
                success, detail = self._toggle_control(params)
            else:
                success, detail = execute_system_action(action_id, params)
        except Exception as exc:
            success = False
            detail = str(exc)
        return ActionEvent(
            time_text=time_text,
            gesture=gesture,
            gesture_zh=translate_gesture(gesture),
            action_id=action_id,
            action_title=title,
            success=success,
            detail=detail,
            source=source,
        )

    def _toggle_control(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        with self.lock:
            current = bool(self.config.get('control_enabled', True))
            self.config['control_enabled'] = not current
            save_json(self.config, self.config_path)
            new_state = not current
        msg_key = 'enabled_message' if new_state else 'disabled_message'
        message = str(params.get(msg_key, '控制状态已切换'))
        ntitle = str(params.get('title', 'PalmGraph-MoE'))
        execute_system_action('notify', {'title': ntitle, 'message': message})
        label = '启动' if new_state else '关闭'
        return True, f'控制已{label}'


def _run_subprocess(command: List[str], timeout: float = 8.0) -> Tuple[bool, str]:
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    detail = (proc.stdout or proc.stderr or '').strip()
    if proc.returncode == 0:
        return True, detail or 'OK'
    return False, detail or f'Exit code {proc.returncode}'


def _run_applescript_lines(lines: List[str], timeout: float = 8.0) -> Tuple[bool, str]:
    command = ['osascript']
    for line in lines:
        command.extend(['-e', line])
    return _run_subprocess(command, timeout=timeout)


def _platform_name() -> str:
    if sys.platform == 'darwin':
        return 'macos'
    if sys.platform == 'win32':
        return 'windows'
    if sys.platform == 'linux':
        return 'linux'
    return 'unsupported'


def _parse_modifiers(raw: str, mapping: Dict[str, Any]) -> List[Any]:
    tokens = [item.strip().lower() for item in str(raw or '').split(',') if item.strip()]
    resolved: List[Any] = []
    for token in tokens:
        if token in mapping:
            resolved.append(mapping[token])
    return resolved


def _hotkey_lines(key: str, modifiers: str = '', repeat: int = 1) -> List[str]:
    key = str(key or '').strip().lower()
    if not key:
        raise ValueError('Missing hotkey key.')
    repeat = max(1, min(8, int(repeat)))
    modifiers_list = _parse_modifiers(modifiers, MODIFIER_TOKENS)
    using = f' using {{{", ".join(modifiers_list)}}}' if modifiers_list else ''
    if key in SPECIAL_KEY_CODES:
        press_line = f'key code {SPECIAL_KEY_CODES[key]}{using}'
    elif len(key) == 1:
        press_line = f'keystroke "{key}"{using}'
    else:
        raise ValueError(f'Unsupported hotkey key: {key}')

    lines = ['tell application "System Events"']
    for _ in range(repeat):
        lines.append(press_line)
    lines.append('end tell')
    return lines


def _post_media_key(action_id: str) -> Tuple[bool, str]:
    key_type = MEDIA_KEY_TYPES.get(action_id)
    if key_type is None:
        return False, f'Unsupported media key action: {action_id}'

    try:
        import AppKit
        import Quartz
    except ImportError:
        return False, 'Missing PyObjC media-key bridge. Install pyobjc-core, pyobjc-framework-Cocoa and pyobjc-framework-Quartz.'

    event_type = getattr(AppKit, 'NSEventTypeSystemDefined', getattr(AppKit, 'NSSystemDefined', 14))
    event_tap = getattr(Quartz, 'kCGSessionEventTap', 1)
    event_factory = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_

    for flags, state in ((0xA00, 0xA), (0xB00, 0xB)):
        event = event_factory(
            event_type,
            (0, 0),
            flags,
            0,
            0,
            0,
            8,
            (key_type << 16) | (state << 8),
            -1,
        )
        Quartz.CGEventPost(event_tap, event.CGEvent())
        time.sleep(0.01)

    return True, f'media_key:{action_id}'


def _post_keyboard_event(keycode: int, flags: int = 0) -> Tuple[bool, str]:
    try:
        import Quartz
    except ImportError:
        return False, 'Missing PyObjC Quartz framework.'
    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    tap = getattr(Quartz, 'kCGSessionEventTap', 1)
    for pressed in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(source, keycode, pressed)
        if flags:
            Quartz.CGEventSetFlags(event, flags)
        Quartz.CGEventPost(tap, event)
        time.sleep(0.01)
    return True, f'keyboard:{keycode}'


def _windows_key_code(key: str) -> Optional[int]:
    normalized = str(key or '').strip().lower()
    if not normalized:
        return None
    if normalized in WINDOWS_SPECIAL_KEYS:
        return WINDOWS_SPECIAL_KEYS[normalized]
    if len(normalized) == 1 and normalized.isascii() and normalized.isalnum():
        return ord(normalized.upper())
    return None


def _linux_key_name(key: str) -> Optional[str]:
    normalized = str(key or '').strip()
    lowered = normalized.lower()
    if lowered in LINUX_SPECIAL_KEYS:
        return LINUX_SPECIAL_KEYS[lowered]
    if len(normalized) == 1 and normalized.isascii() and normalized.isprintable():
        return normalized.lower()
    return None


def _press_windows_vk(vk_code: int, modifiers: Optional[List[int]] = None, repeat: int = 1) -> Tuple[bool, str]:
    try:
        import ctypes
    except ImportError:
        return False, 'ctypes is unavailable on this Python runtime.'

    windll = getattr(ctypes, 'windll', None)
    if windll is None:
        return False, 'Windows keyboard API is unavailable (not a Windows system or ctypes limitation).'

    keybd_event = windll.user32.keybd_event
    keyup_flag = 0x0002
    modifiers = modifiers or []
    repeat = max(1, min(8, int(repeat)))

    for modifier in modifiers:
        keybd_event(modifier, 0, 0, 0)
    for _ in range(repeat):
        keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.01)
        keybd_event(vk_code, 0, keyup_flag, 0)
        time.sleep(0.01)
    for modifier in reversed(modifiers):
        keybd_event(modifier, 0, keyup_flag, 0)
    return True, f'vk:{vk_code}'


def _send_hotkey(key: str, modifiers: str = '', repeat: int = 1) -> Tuple[bool, str]:
    platform_name = _platform_name()
    repeat = max(1, min(8, int(repeat)))
    if platform_name == 'macos':
        return _run_applescript_lines(_hotkey_lines(key, modifiers=modifiers, repeat=repeat))

    if platform_name == 'windows':
        vk_code = _windows_key_code(key)
        if vk_code is None:
            return False, f'Unsupported hotkey key: {key}'
        modifier_codes = _parse_modifiers(modifiers, WINDOWS_MODIFIER_KEYS)
        return _press_windows_vk(vk_code, modifier_codes, repeat=repeat)

    if platform_name == 'linux':
        command = shutil.which('xdotool')
        if command is None:
            return False, LINUX_XDOTOOL_ERROR
        key_name = _linux_key_name(key)
        if key_name is None:
            return False, f'Unsupported hotkey key: {key}'
        modifier_tokens = _parse_modifiers(modifiers, LINUX_MODIFIER_TOKENS)
        combo = '+'.join([*modifier_tokens, key_name]) if modifier_tokens else key_name
        for _ in range(repeat):
            success, detail = _run_subprocess([command, 'key', '--clearmodifiers', combo])
            if not success:
                return False, detail
        return True, combo

    return False, f'Unsupported platform: {platform_name}'


def _run_windows_message_box(title: str, message: str) -> Tuple[bool, str]:
    try:
        import ctypes
    except ImportError:
        return False, 'ctypes is unavailable on this Python runtime.'

    windll = getattr(ctypes, 'windll', None)
    if windll is None:
        return False, 'Windows notification API is unavailable (not a Windows system or ctypes limitation).'
    windll.user32.MessageBoxW(None, message, title, 0)
    return True, title


def _send_notification(title: str, message: str) -> Tuple[bool, str]:
    platform_name = _platform_name()
    if platform_name == 'macos':
        title = title.replace('"', "'")
        message = message.replace('"', "'")
        return _run_applescript_lines([f'display notification "{message}" with title "{title}"'])

    if platform_name == 'windows':
        return _run_windows_message_box(title, message)

    if platform_name == 'linux':
        notify_send = shutil.which('notify-send')
        if notify_send is not None:
            return _run_subprocess([notify_send, title, message])
        zenity = shutil.which('zenity')
        if zenity is not None:
            return _run_subprocess([zenity, '--info', f'--title={title}', f'--text={message}'])
        return False, LINUX_NOTIFY_ERROR

    return False, f'Unsupported platform: {platform_name}'


def _post_media_key_cross_platform(action_id: str) -> Tuple[bool, str]:
    platform_name = _platform_name()
    if platform_name == 'macos':
        return _post_media_key(action_id)

    if platform_name == 'windows':
        media_vks = {
            'volume_up': 0xAF,
            'volume_down': 0xAE,
            'mute_toggle': 0xAD,
            'play_pause': 0xB3,
            'next_track': 0xB0,
            'previous_track': 0xB1,
        }
        vk_code = media_vks.get(action_id)
        if vk_code is None:
            return False, f'Unsupported media key action: {action_id}'
        return _press_windows_vk(vk_code)

    if platform_name == 'linux':
        command = shutil.which('xdotool')
        if command is None:
            return False, LINUX_XDOTOOL_ERROR
        media_keys = {
            'volume_up': 'XF86AudioRaiseVolume',
            'volume_down': 'XF86AudioLowerVolume',
            'mute_toggle': 'XF86AudioMute',
            'play_pause': 'XF86AudioPlay',
            'next_track': 'XF86AudioNext',
            'previous_track': 'XF86AudioPrev',
        }
        key_name = media_keys.get(action_id)
        if key_name is None:
            return False, f'Unsupported media key action: {action_id}'
        return _run_subprocess([command, 'key', '--clearmodifiers', key_name])

    return False, f'Unsupported platform: {platform_name}'


def _take_screenshot(directory: str) -> Tuple[bool, str]:
    base_dir = Path(str(directory or '~/Pictures/PalmGraphDemo')).expanduser()
    ensure_dir(base_dir)
    out_path = base_dir / f'gesture-shot-{datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
    platform_name = _platform_name()

    if platform_name == 'macos':
        success, detail = _run_subprocess(['screencapture', '-x', str(out_path)])
        return (True, str(out_path)) if success else (False, detail)

    if platform_name == 'windows':
        script = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            'Add-Type -AssemblyName System.Drawing; '
            '$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen; '
            '$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; '
            '$graphics = [System.Drawing.Graphics]::FromImage($bitmap); '
            '$graphics.CopyFromScreen($bounds.Left, $bounds.Top, 0, 0, $bitmap.Size); '
            '$bitmap.Save($args[0], [System.Drawing.Imaging.ImageFormat]::Png); '
            '$graphics.Dispose(); '
            '$bitmap.Dispose()'
        )
        powershell = shutil.which('powershell') or shutil.which('pwsh')
        if powershell is None:
            return False, 'PowerShell could not be found in PATH. Ensure powershell.exe or pwsh.exe is accessible on Windows.'
        success, detail = _run_subprocess([powershell, '-NoProfile', '-Command', script, str(out_path)], timeout=15.0)
        return (True, str(out_path)) if success else (False, detail)

    if platform_name == 'linux':
        screenshot_commands = [
            ('gnome-screenshot', ['-f', str(out_path)]),
            ('scrot', [str(out_path)]),
            ('import', ['-window', 'root', str(out_path)]),
            ('grim', [str(out_path)]),
        ]
        last_detail = ''
        for name, args in screenshot_commands:
            command = shutil.which(name)
            if command is None:
                continue
            success, detail = _run_subprocess([command, *args], timeout=15.0)
            if success:
                return True, str(out_path)
            last_detail = detail
        if last_detail:
            return False, last_detail
        return False, LINUX_SCREENSHOT_ERROR

    return False, f'Unsupported platform: {platform_name}'


def execute_system_action(action_id: str, params: Dict[str, Any]) -> Tuple[bool, str]:
    if action_id == 'noop':
        return True, 'Preview only'

    if action_id == 'notify':
        title = str(params.get('title', 'PalmGraph-MoE'))
        message = str(params.get('message', '已识别到手势'))
        return _send_notification(title, message)

    if action_id == 'volume_up':
        return _post_media_key_cross_platform('volume_up')

    if action_id == 'volume_down':
        return _post_media_key_cross_platform('volume_down')

    if action_id == 'mute_toggle':
        return _post_media_key_cross_platform('mute_toggle')

    if action_id == 'play_pause':
        return _post_media_key_cross_platform('play_pause')

    if action_id == 'media_play':
        return _post_media_key_cross_platform('play_pause')

    if action_id == 'media_pause':
        return _post_media_key_cross_platform('play_pause')

    if action_id == 'next_item':
        return _post_media_key_cross_platform('next_track')

    if action_id == 'previous_item':
        return _post_media_key_cross_platform('previous_track')

    if action_id == 'fullscreen_toggle':
        if _platform_name() == 'macos':
            return _post_keyboard_event(3, _CGEVENT_FLAG_CMD | _CGEVENT_FLAG_CTRL)
        return _send_hotkey('f11')

    if action_id == 'stop_key':
        if _platform_name() == 'macos':
            return _post_keyboard_event(53)
        return _send_hotkey('escape')

    if action_id == 'screenshot':
        return _take_screenshot(str(params.get('directory', '~/Pictures/PalmGraphDemo')))

    if action_id == 'hotkey':
        key = str(params.get('key', '')).strip()
        modifiers = str(params.get('modifiers', ''))
        repeat = params.get('repeat', '1')
        return _send_hotkey(key, modifiers=modifiers, repeat=int(repeat or 1))

    if action_id == 'shell':
        command = str(params.get('command', '')).strip()
        if not command:
            return False, 'Missing shell command.'
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=12)
        detail = (proc.stdout or proc.stderr or '').strip()
        if proc.returncode == 0:
            return True, detail or command
        return False, detail or f'Exit code {proc.returncode}'

    if action_id == 'applescript':
        if _platform_name() != 'macos':
            return False, 'AppleScript is only available on macOS.'
        script = str(params.get('script', '')).strip()
        if not script:
            return False, 'Missing AppleScript.'
        lines = [line for line in script.splitlines() if line.strip()]
        if not lines:
            return False, 'Missing AppleScript.'
        return _run_applescript_lines(lines)

    return False, f'Unsupported action: {action_id}'


def load_predictor(checkpoint: Optional[str], onnx_path: Optional[str], meta_path: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
    if checkpoint:
        predictor = TorchPredictor(checkpoint, device='cpu')
        meta = predictor.meta['data_meta'] | {
            'rejection_threshold': predictor.meta.get('rejection_threshold', 0.5),
            'param_count': predictor.meta.get('param_count', 0),
            'model_name': predictor.meta.get('model_config', {}).get('model_name', 'torch_checkpoint'),
        }
        return predictor, meta

    if onnx_path and meta_path:
        predictor = ONNXPredictor(onnx_path, meta_path)
        return predictor, predictor.meta

    raise ValueError('Provide either checkpoint or both onnx and meta paths.')


class GestureWebDemoService:
    def __init__(
        self,
        checkpoint: Optional[str],
        onnx_path: Optional[str],
        meta_path: Optional[str],
        config_path: Path,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        threshold: Optional[float] = None,
        ema_alpha: float = 0.70,
        stable_frames: int = 3,
        history_size: int = 12,
        mirror: bool = True,
    ) -> None:
        self.checkpoint_path = None if checkpoint is None else Path(checkpoint).expanduser().resolve()
        requested_onnx_path = None if onnx_path is None else Path(onnx_path).expanduser().resolve()
        self.meta_path = None if meta_path is None else Path(meta_path).expanduser().resolve()
        self.onnx_path = preferred_startup_onnx_path(requested_onnx_path, self.meta_path)
        self.model_lock = threading.Lock()
        self.predictor, self.meta = load_predictor(
            None if self.checkpoint_path is None else str(self.checkpoint_path),
            None if self.onnx_path is None else str(self.onnx_path),
            None if self.meta_path is None else str(self.meta_path),
        )
        self.model_options = discover_model_options(self.checkpoint_path, self.onnx_path, self.meta_path, self.meta)
        self.active_model_id = 'checkpoint' if self.checkpoint_path is not None else (self.onnx_path.name if self.onnx_path is not None else 'default')
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.mirror = bool(mirror)
        self.class_names = list(self.meta['class_names'])
        label_to_id = dict(self.meta.get('label_to_id', {}))
        self.no_gesture_label = 'no_gesture' if 'no_gesture' in self.class_names else self.class_names[0]
        self.no_gesture_idx = int(label_to_id.get(self.no_gesture_label, 0))
        self.threshold = float(threshold if threshold is not None else self.meta.get('rejection_threshold', 0.5))
        self.stabilizer = TemporalGestureFilter(
            num_classes=len(self.class_names),
            no_gesture_idx=self.no_gesture_idx,
            threshold=self.threshold,
            ema_alpha=ema_alpha,
            stable_frames=stable_frames,
            history_size=history_size,
        )
        self.action_controller = GestureActionController(
            config_path=config_path,
            class_names=self.class_names,
            command_map=dict(self.meta.get('command_map', {})),
            no_gesture_label=self.no_gesture_label,
        )
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.frame_lock = threading.Condition()
        self.frame_bytes: Optional[bytes] = None
        self.frame_seq = 0
        self.state_lock = threading.Lock()
        self.state: Dict[str, Any] = {
            'running': False,
            'camera_ready': False,
            'last_error': None,
            'gesture': self.no_gesture_label,
            'gesture_zh': translate_gesture(self.no_gesture_label),
            'confidence': 1.0,
            'raw_gesture': self.no_gesture_label,
            'raw_gesture_zh': translate_gesture(self.no_gesture_label),
            'raw_confidence': 1.0,
            'detected': False,
            'fps': 0.0,
            'model_name': self.meta.get('model_name', 'palmgraph_moe'),
            'model_display_name': self._current_model_display_name(),
            'param_count': int(self.meta.get('param_count', 0)),
            'binding': self.action_controller.describe_binding(self.no_gesture_label),
            'pending_action': None,
        }

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_loop, name='gesture-web-demo', daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self.action_controller.close()

    def _update_state(self, **kwargs: Any) -> None:
        with self.state_lock:
            self.state.update(kwargs)

    def _find_model_option_locked(self, model_id: str) -> Optional[Dict[str, Any]]:
        for item in self.model_options:
            if str(item.get('id')) == model_id:
                return item
        return None

    def _current_model_title_locked(self) -> str:
        option = self._find_model_option_locked(self.active_model_id)
        if option is None:
            return ''
        return str(option.get('title', ''))

    def _current_model_display_name(self) -> str:
        with self.model_lock:
            base_name = str(self.meta.get('model_name', 'unknown'))
            active_title = self._current_model_title_locked()
        return base_name if not active_title else f'{base_name} / {active_title}'

    def models_snapshot(self) -> Dict[str, Any]:
        with self.model_lock:
            options = [
                {
                    'id': str(item.get('id', '')),
                    'title': str(item.get('title', '')),
                    'kind': str(item.get('kind', '')),
                    'file_name': str(item.get('file_name', '')),
                    'size_bytes': int(item.get('size_bytes', 0)),
                    'active': str(item.get('id', '')) == self.active_model_id,
                }
                for item in self.model_options
            ]
            active_title = self._current_model_title_locked()
            model_display_name = str(self.meta.get('model_name', 'unknown'))
            if active_title:
                model_display_name = f'{model_display_name} / {active_title}'
            return {
                'active_model_id': self.active_model_id,
                'active_model_title': active_title,
                'model_display_name': model_display_name,
                'model_options': options,
            }

    def state_snapshot(self) -> Dict[str, Any]:
        with self.state_lock:
            data = json.loads(json.dumps(self.state, ensure_ascii=False))
        config_snapshot = self.action_controller.snapshot()
        data.update(self.models_snapshot())
        data['control_enabled'] = config_snapshot['control_enabled']
        data['cooldown_seconds'] = config_snapshot['cooldown_seconds']
        data['confidence_threshold'] = config_snapshot['confidence_threshold']
        data['trigger_delay_seconds'] = config_snapshot['trigger_delay_seconds']
        data['last_action'] = self.action_controller.last_event_snapshot()
        data['recent_actions'] = self.action_controller.recent_events_snapshot()
        data['class_names'] = self.class_names
        data['gesture_labels'] = {name: translate_gesture(name) for name in self.class_names}
        return data

    def config_snapshot(self) -> Dict[str, Any]:
        snapshot = self.action_controller.snapshot()
        snapshot.update(self.models_snapshot())
        snapshot['gesture_labels'] = {name: translate_gesture(name) for name in self.class_names}
        snapshot['model_command_map'] = dict(self.meta.get('command_map', {}))
        return snapshot

    def update_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.action_controller.update_from_payload(payload)
        return self.config_snapshot()

    def test_binding(self, gesture: str) -> Dict[str, Any]:
        return self.action_controller.execute_binding_for_gesture(gesture)

    def test_preview_binding(self, gesture: str, binding: Dict[str, Any]) -> Dict[str, Any]:
        return self.action_controller.execute_preview_binding(gesture, binding)

    def switch_model(self, model_id: str) -> Dict[str, Any]:
        with self.model_lock:
            option = self._find_model_option_locked(model_id)
            current_id = self.active_model_id

        if option is None:
            raise KeyError(f'Unknown model: {model_id}')
        if model_id == current_id:
            return self.state_snapshot()
        if str(option.get('kind')) != 'onnx':
            return self.state_snapshot()

        new_model_path = str(option['model_path'])
        new_meta_path = str(option['meta_path'])
        predictor, meta = load_predictor(None, new_model_path, new_meta_path)

        if list(meta.get('class_names', [])) != self.class_names:
            raise ValueError('Selected model is incompatible with current class names.')
        if dict(meta.get('label_to_id', {})) != dict(self.meta.get('label_to_id', {})):
            raise ValueError('Selected model is incompatible with current label mapping.')
        if dict(meta.get('command_map', {})) != dict(self.meta.get('command_map', {})):
            raise ValueError('Selected model is incompatible with current command mapping.')

        with self.model_lock:
            self.predictor = predictor
            self.meta = meta
            self.active_model_id = model_id
            self.stabilizer.reset()

        self.action_controller.update_idle_state(self.no_gesture_label)
        self._update_state(
            last_error=None,
            gesture=self.no_gesture_label,
            gesture_zh=translate_gesture(self.no_gesture_label),
            confidence=1.0,
            raw_gesture=self.no_gesture_label,
            raw_gesture_zh=translate_gesture(self.no_gesture_label),
            raw_confidence=1.0,
            detected=False,
            model_name=meta.get('model_name', 'palmgraph_moe'),
            model_display_name=self._current_model_display_name(),
            param_count=int(meta.get('param_count', 0)),
            binding=self.action_controller.describe_binding(self.no_gesture_label),
            pending_action=None,
        )
        return self.state_snapshot()

    def _encode_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with self.frame_lock:
            self.frame_bytes = encoded.tobytes()
            self.frame_seq += 1
            self.frame_lock.notify_all()

    def stream_frames(self):
        last_seq = -1
        while not self.stop_event.is_set():
            with self.frame_lock:
                if self.frame_seq == last_seq:
                    self.frame_lock.wait(timeout=1.0)
                frame = self.frame_bytes
                seq = self.frame_seq
            if frame is None or seq == last_seq:
                continue
            last_seq = seq
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def _draw_overlay(
        self,
        frame: np.ndarray,
        pred_label: str,
        confidence: float,
        raw_label: str,
        raw_confidence: float,
        fps: float,
    ) -> np.ndarray:
        return frame

    def _run_loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            self._update_state(running=False, camera_ready=False, last_error='Failed to open webcam.')
            return

        self._update_state(running=True, camera_ready=True, last_error=None)
        frame_count = 0
        started = time.perf_counter()

        try:
            with MediaPipeFeatureExtractor(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as extractor:
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self._update_state(last_error='Failed to read frame from webcam.')
                        time.sleep(0.05)
                        continue

                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    feat = extractor.extract(frame, assume_bgr=True)
                    if feat.detected:
                        with self.model_lock:
                            raw_probs = self.predictor.predict_proba(feat.landmarks, feat.geom)
                            pred_idx, confidence, _ = self.stabilizer.update(raw_probs)
                        raw_idx = int(np.argmax(raw_probs))
                        raw_label = self.class_names[raw_idx]
                        raw_confidence = float(raw_probs[raw_idx])
                    else:
                        with self.model_lock:
                            pred_idx, confidence, _ = self.stabilizer.update(None)
                        raw_label = self.no_gesture_label
                        raw_confidence = 1.0

                    pred_label = self.class_names[pred_idx]
                    pending_action = self.action_controller.maybe_queue_action(pred_label, confidence)
                    self.action_controller.update_idle_state(pred_label)

                    frame_count += 1
                    fps = frame_count / max(time.perf_counter() - started, 1e-6)
                    frame = self._draw_overlay(frame, pred_label, confidence, raw_label, raw_confidence, fps)
                    self._encode_frame(frame)
                    self._update_state(
                        running=True,
                        camera_ready=True,
                        last_error=None,
                        gesture=pred_label,
                        gesture_zh=translate_gesture(pred_label),
                        confidence=float(confidence),
                        raw_gesture=raw_label,
                        raw_gesture_zh=translate_gesture(raw_label),
                        raw_confidence=float(raw_confidence),
                        detected=bool(feat.detected),
                        fps=float(fps),
                        binding=self.action_controller.describe_binding(pred_label),
                        pending_action=pending_action,
                    )
        except Exception as exc:
            self._update_state(last_error=str(exc), running=False, camera_ready=False)
        finally:
            cap.release()
            self._update_state(running=False, camera_ready=False)


def create_web_app(service: GestureWebDemoService) -> Flask:
    app = Flask(__name__, static_folder=str(WEB_ASSET_DIR), static_url_path='/assets')

    @app.get('/')
    def index() -> Response:
        return send_from_directory(WEB_ASSET_DIR, 'index.html')

    @app.get('/styles.css')
    def styles() -> Response:
        return send_from_directory(WEB_ASSET_DIR, 'styles.css')

    @app.get('/app.js')
    def app_js() -> Response:
        return send_from_directory(WEB_ASSET_DIR, 'app.js')

    @app.get('/stream.mjpg')
    def stream() -> Response:
        return Response(service.stream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.get('/api/state')
    def api_state() -> Response:
        return jsonify(service.state_snapshot())

    @app.get('/api/config')
    def api_config() -> Response:
        return jsonify(service.config_snapshot())

    @app.get('/api/actions')
    def api_actions() -> Response:
        return jsonify({'actions': ACTION_DEFINITIONS, 'gesture_labels': {name: translate_gesture(name) for name in service.class_names}})

    @app.get('/api/models')
    def api_models() -> Response:
        return jsonify(service.models_snapshot())

    @app.post('/api/config')
    def api_save_config() -> Response:
        payload = request.get_json(silent=True) or {}
        return jsonify(service.update_config(payload))

    @app.post('/api/test-binding')
    def api_test_binding() -> Response:
        payload = request.get_json(silent=True) or {}
        gesture = str(payload.get('gesture', '')).strip()
        if not gesture:
            return jsonify({'ok': False, 'error': 'Missing gesture.'}), 400
        try:
            binding = payload.get('binding')
            if isinstance(binding, dict):
                event = service.test_preview_binding(gesture, binding)
            else:
                event = service.test_binding(gesture)
        except KeyError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 404
        return jsonify({'ok': True, 'event': event})

    @app.post('/api/model')
    def api_switch_model() -> Response:
        payload = request.get_json(silent=True) or {}
        model_id = str(payload.get('model_id', '')).strip()
        if not model_id:
            return jsonify({'ok': False, 'error': 'Missing model_id.'}), 400
        try:
            return jsonify(service.switch_model(model_id))
        except KeyError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400

    return app
