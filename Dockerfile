# Hugging Face Spaces (Docker SDK)용 이미지.
# 프론트엔드(viewer-src)는 안 담고, 추론 API 서버(server.py)만 돌린다.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-local.txt .
RUN pip install --no-cache-dir -r requirements-local.txt

COPY mri_infer_core.py server.py ./
COPY mri_rf ./mri_rf
COPY patient_mri ./patient_mri

ENV HOST=0.0.0.0
ENV PORT=7860
ENV MRI_SERVER_NO_BROWSER=1

EXPOSE 7860

CMD ["python", "server.py"]
