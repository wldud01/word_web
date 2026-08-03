"""
api/[...path].py
-----------------
Vercel Python 서버리스 함수. /api/* 요청 전체를 이 파일 하나가 처리한다
(catch-all 동적 라우트). vercel.json의 rewrite로 /mri/:filename도
/api/mri/:filename 으로 들어와 여기서 함께 처리된다.

주의 (실험적 배포): 이 함수는 ~200MB PyTorch 체크포인트 + torch(CPU)
추론을 서버리스 함수 안에서 돌리는 구조라, Vercel의 함수 크기/실행시간
제한에 걸려 배포 자체가 실패하거나 추론이 타임아웃될 가능성이 크다.
또한 체크포인트 파일은 GitHub 100MB 제한 때문에 저장소에 커밋하지
않았으므로(.gitignore), 배포된 함수에서는 "체크포인트 없음" 상태로
동작한다 — 로컬 server.py에서만 실제 추론이 된다.
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

        # mri/<filename>  ( /mri/:filename → rewrite → /api/mri/:filename )
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
