"""
mri-server/server.py
--------------------
MRI T1→T2 추론 독립 서버 (로컬 상시 구동용).

폴더 구조:
    mri-server/
    ├── server.py           ← 이 파일 (로컬 개발/구동용, 포트 8765)
    ├── mri_infer_core.py   ← server.py / api/[...path].py 공용 추론 코어
    ├── api/[...path].py    ← Vercel 서버리스 배포용 (동일 로직)
    ├── mri_rf/             ← 모델 코드 + checkpoint.88.pt (placeholder)
    └── patient_mri/        ← 환자별 슬라이스 PNG (지금은 p1의 T2 슬라이스를
                               T1 원본 자리표시자로 사용 중)

실행:
    cd mri-server
    python3 server.py      # venv 활성화 후
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import mri_infer_core as core

BASE       = Path(__file__).parent
PUBLIC_DIR = BASE / "public"
PORT       = 8765

if core.MRI_PATHS:
    print(f"[INFO] MRI 슬라이스: {len(core.MRI_PATHS)}장")
else:
    print(f"[WARN] MRI 폴더 없음: {core.MRI_DIR}")

core.ensure_model_load_started()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
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
        if parsed.path == "/api/infer-t1":
            try:
                length    = int(self.headers.get("Content-Length", 0))
                png_bytes = self.rfile.read(length)
                print(f"[PREDICT POST] {length//1024}KB 추론 중...", flush=True)
                result = core.infer_from_bytes(png_bytes)
                print(f"[PREDICT POST] 완료 ({len(result)//1024}KB)")
                self._send_png(200, result)
            except Exception as e:
                print(f"[PREDICT POST ERROR] {e}")
                self._send_json(500, {"error": str(e)})
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        # /mri/<filename>.png  (서브폴더 포함 검색)
        if path.startswith("/mri/"):
            file_path = core.find_mri_file(path[5:])
            if file_path is not None:
                self._send_png(200, file_path.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
            return

        # /api/mri-slices
        if path == "/api/mri-slices":
            self._send_json(200, {
                "count": len(core.MRI_PATHS),
                "files": [p.name for p in core.MRI_PATHS],
            })
            return

        # /api/infer-t1?idx=N
        if path == "/api/infer-t1":
            try:
                qs  = parse_qs(parsed.query)
                idx = int(qs.get("idx", ["0"])[0])
                idx = max(0, min(idx, len(core.MRI_PATHS) - 1))
                print(f"[PREDICT] 슬라이스 #{idx+1}/{len(core.MRI_PATHS)} 추론 중...", flush=True)
                png = core.infer_slice(idx)
                print(f"[PREDICT] #{idx+1} 완료 ({len(png)//1024}KB)")
                self._send_png(200, png)
            except Exception as e:
                print(f"[PREDICT ERROR] {e}")
                self._send_json(500, {"error": str(e)})
            return

        # /api/status
        if path == "/api/status":
            self._send_json(200, core.model_status())
            return

        # 정적 파일 (public/)
        if path == "/":
            path = "/index.html"
        file_path = PUBLIC_DIR / path.lstrip("/")
        if file_path.is_file():
            mime, _ = mimetypes.guess_type(str(file_path))
            mime = mime or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"\n{'='*50}")
    print(f"  MRI T1→T2 서버 (placeholder checkpoint)")
    print(f"  http://localhost:{PORT}")
    print(f"  슬라이스: {len(core.MRI_PATHS)}장  |  모델: {core.CKPT_PATH.name}")
    print(f"{'='*50}\n")

    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")
