"""
api/[...path].py
-----------------
Vercel Python 서버리스 함수 (catch-all). /api/* 요청 전체를 이 파일 하나가
처리한다. mri_infer_core.py를 server.py와 그대로 공유해서 쓴다.

체크포인트는 저장소에 없으므로(GitHub 100MB 제한) CHECKPOINT_URL 환경변수
(Hugging Face Hub 등 외부 파일 URL)에서 받아와 /tmp에 캐싱한다
(mri_infer_core.py의 _resolve_ckpt_path 참고). Vercel 프로젝트 환경변수에
CHECKPOINT_URL을 설정해야 모델이 로드된다.

이 함수는 vercel.json에서 VERCEL_SUPPORT_LARGE_FUNCTIONS=1로 5GB 번들
한도를 켜야 torch 설치본이 들어간다(기본 500MB 한도로는 부족).
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mri_infer_core as core  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors()
        self.end_headers()

    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, status: int, data: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_cors()
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        sub = _strip_api(parsed.path)
        if sub == "infer-t1":
            try:
                length    = int(self.headers.get("Content-Length", 0))
                png_bytes = self.rfile.read(length)
                result    = core.infer_from_bytes(png_bytes)
                self._send_png(200, result)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        sub    = _strip_api(parsed.path)

        if sub.startswith("mri/"):
            file_path = core.find_mri_file(sub[len("mri/"):])
            if file_path is not None:
                self._send_png(200, file_path.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
            return

        if sub == "mri-slices":
            self._send_json(200, {
                "count": len(core.MRI_PATHS),
                "files": [p.name for p in core.MRI_PATHS],
            })
            return

        if sub == "infer-t1":
            try:
                qs  = parse_qs(parsed.query)
                idx = int(qs.get("idx", ["0"])[0])
                idx = max(0, min(idx, len(core.MRI_PATHS) - 1))
                png = core.infer_slice(idx)
                self._send_png(200, png)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        if sub == "status":
            core.ensure_model_load_started()
            # 서버리스 환경에선 요청 사이에 백그라운드 스레드가 이어서 돈다는 보장이
            # 없어서, 이 요청 안에서 다운로드+로딩이 끝날 때까지(최대 maxDuration 여유
            # 안에서) 기다린 뒤 상태를 보고한다.
            core.wait_for_model_ready(timeout=260)
            self._send_json(200, core.model_status())
            return

        self.send_response(404)
        self.end_headers()


def _strip_api(path: str) -> str:
    """'/api/status' → 'status', '/api/mri/foo.png' → 'mri/foo.png'"""
    p = path.lstrip("/")
    if p.startswith("api/"):
        p = p[len("api/"):]
    return p
