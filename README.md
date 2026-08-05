# MRI T1→T2 추론 서버

MRI T1 슬라이스를 T2로 변환하는 독립 실행 서버입니다. 원본(T1)과 예측
결과(T2)를 나란히 비교하는 뷰어를 제공합니다.

> **현재 상태 (placeholder):** `patient_mri/p1`에는 아직 T2 슬라이스만
> 있어서 T1 원본 자리표시자로 쓰고 있고, `checkpoint.88.pt`도 원래
> T2→T1용으로 학습된 가중치입니다. 실제 T1 원본 데이터와 T1→T2
> 체크포인트가 준비되면 교체하세요.

## 폴더 구조

```
mri-server/
├── server.py           # Python 추론 서버 — 로컬 상시 구동용 (포트 8765)
├── mri_infer_core.py   # server.py / api 서버리스 함수 공용 추론 코어
├── api/[...path].py    # Vercel 서버리스 함수 (프론트+백엔드 한 프로젝트로 배포)
├── vercel.json         # Vercel 빌드/함수 설정
├── requirements.txt    # Python 패키지 목록 (pillow, numpy, torch만 — 최소 구성)
├── Dockerfile           # (선택) Docker로 별도 호스팅하고 싶을 때용
├── viewer-src/         # 프론트엔드 소스 (React + Vite)
├── public/             # 빌드된 프론트엔드
├── mri_rf/             # 모델 코드 + 체크포인트 (checkpoint.88.pt, git 미포함)
└── patient_mri/        # 환자별 슬라이스 PNG (p1)
```

## 아키텍처: Vercel 하나로 프론트+백엔드

`GeneratorSPADE`(단일 forward pass, RectifiedFlow 미사용)로 바꾸면서
실제 필요한 패키지가 `pillow`/`numpy`/`torch` 세 개뿐으로 줄었고, Vercel의
["Large Functions"](https://vercel.com/docs/functions/limitations#large-functions-beta)
(서버리스 함수 크기 한도를 500MB→5GB로 올리는 옵트인 기능)를 켜면 torch
설치본도 그 안에 들어갑니다. 그래서 지금은 `api/[...path].py` 하나로
프론트엔드와 추론 백엔드를 **같은 Vercel 프로젝트**에서 서빙합니다.

체크포인트(379MB)는 GitHub 100MB 제한 때문에 이 저장소엔 못 올리므로,
**Hugging Face Hub(Model 저장소, 파일 저장 전용— Space/컴퓨트 아님)** 에
올려두고 `CHECKPOINT_URL` 환경변수로 그 다운로드 링크를 알려주면,
서버리스 함수가 콜드 스타트 시 `/tmp`로 받아서 캐싱합니다
(`mri_infer_core.py`의 `_resolve_ckpt_path`).

### Vercel 프로젝트에 설정할 환경변수

- `VERCEL_SUPPORT_LARGE_FUNCTIONS` = `1` — 5GB 함수 크기 한도 활성화 (필수)
- `CHECKPOINT_URL` = Hugging Face Hub의 체크포인트 직접 다운로드 URL
  (예: `https://huggingface.co/<user>/<repo>/resolve/main/checkpoint.88.pt`)

로컬 `server.py`는 `mri_rf/checkpoint.88.pt`가 이미 디스크에 있으면 그걸
그대로 쓰고, `CHECKPOINT_URL`은 신경 쓸 필요 없습니다.

> `pyproject.toml`(Poetry)은 제거했습니다 — Vercel의 Python 빌더가
> `pyproject.toml`을 발견하면 `[project]` 테이블이 있는 PEP 621 형식으로
> 간주하고 `uv lock`을 시도하는데, Poetry 형식과 충돌해 빌드가 실패했습니다.

## 사전 조건

- Python 3.9 ~ 3.12
- Node.js 18 이상 (프론트 빌드 시 필요 / 최초 1회)

---

## 최초 설치 (처음 한 번만)

### 1. Node.js 18 설치 (없는 경우)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18
```

### 2. 프론트엔드 빌드

```bash
cd viewer-src
npm install
npm run build
cd ..
```

빌드 결과가 자동으로 `public/` 폴더에 저장됩니다.

### 3. Python 환경 설정 및 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 실행

```bash
source .venv/bin/activate   # 가상환경 활성화 (이미 되어 있으면 생략)
python3 server.py
```

브라우저가 자동으로 열립니다. 수동으로 접속하려면:
**http://localhost:8765**

---

## 사용 방법

1. 접속하면 `patient_mri/p1` 슬라이스가 자동으로 로드되고, 왼쪽에 **T1 원본**이
   나타납니다.
2. 슬라이스 탐색: `←` `→` 키 또는 마우스 휠 (두 패널이 함께 이동)
3. 헤더의 **T2 예측 실행** 버튼 클릭 → 서버에서 T1→T2 추론 시작
   - 헤더에 진행률 표시 (`T2 예측 중... N/전체`)
   - 현재 슬라이스가 추론 중이면 오른쪽 패널에 스피너 오버레이 표시
4. 추론이 끝난 슬라이스부터 오른쪽 **T2 예측** 패널에 결과가 채워지며,
   왼쪽 원본과 나란히 비교할 수 있습니다.
5. 우클릭 드래그로 밝기/대비(WC/WW) 조정 — 두 패널이 값을 공유합니다.

---

## 문제 해결

**모델 오류가 헤더에 뜨는 경우**

`pillow`/`numpy`/`torch` 설치가 안 됐거나(`pip install -r requirements.txt`),
`mri_rf/checkpoint.88.pt`가 없는 경우입니다. 헤더의 정확한 오류 메시지를
확인하세요.

**포트 충돌 (8765 이미 사용 중)**

`server.py` 상단의 `PORT = 8765`를 다른 값으로 변경 후 재실행.

**브라우저에서 화면이 안 나오는 경우**

`public/index.html` 이 없으면 프론트 빌드가 안 된 것입니다.
[2. 프론트엔드 빌드] 단계를 다시 실행하세요.
