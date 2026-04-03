#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_hgr.utils import configure_torch_threads
from gesture_hgr.web_demo import GestureWebDemoService, create_web_app


DEFAULT_ONNX = ROOT / 'models' / 'model.int8.onnx'
if not DEFAULT_ONNX.exists():
    DEFAULT_ONNX = ROOT / 'models' / 'model.onnx'
DEFAULT_META = ROOT / 'models' / 'model_meta.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PalmGraph-MoE ONNX web demo.')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--onnx', type=str, default=str(DEFAULT_ONNX) if DEFAULT_ONNX.exists() else None)
    parser.add_argument('--meta', type=str, default=str(DEFAULT_META) if DEFAULT_META.exists() else None)
    parser.add_argument('--config', type=str, default=None, help='Path to saved web demo bindings JSON.')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--ema-alpha', type=float, default=0.70)
    parser.add_argument('--stable-frames', type=int, default=3)
    parser.add_argument('--history-size', type=int, default=12)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--mirror', dest='mirror', action='store_true')
    parser.add_argument('--no-mirror', dest='mirror', action='store_false')
    parser.add_argument('--open-browser', dest='open_browser', action='store_true')
    parser.add_argument('--no-browser', dest='open_browser', action='store_false')
    parser.set_defaults(mirror=True, open_browser=True)
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    if args.meta:
        return Path(args.meta).expanduser().resolve().with_name('web_demo_bindings.json')
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve().with_name('web_demo_bindings.json')
    return (ROOT / 'models' / 'web_demo_bindings.json').resolve()


def maybe_open_browser(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        webbrowser.open(url)

    thread = threading.Thread(target=_open, name='open-browser', daemon=True)
    thread.start()


def main() -> None:
    args = parse_args()
    configure_torch_threads(args.torch_threads)

    if not args.checkpoint and not (args.onnx and args.meta):
        raise ValueError('Provide either --checkpoint or both --onnx and --meta.')

    service = GestureWebDemoService(
        checkpoint=args.checkpoint,
        onnx_path=args.onnx,
        meta_path=args.meta,
        config_path=resolve_config_path(args),
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        stable_frames=args.stable_frames,
        history_size=args.history_size,
        mirror=args.mirror,
    )
    service.start()
    app = create_web_app(service)
    url = f'http://{args.host}:{args.port}'
    print(f'PalmGraph-MoE 中文网页 demo 已启动: {url}')
    if args.open_browser:
        maybe_open_browser(url)

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        service.stop()


if __name__ == '__main__':
    main()