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
├── server.py           # Python 추론 서버 — 별도 서버(로컬 PC/Railway 등)에서 상시 구동 (포트 8765)
├── mri_infer_core.py   # server.py가 쓰는 추론 코어 (모델 로딩 + T1→T2 추론)
├── vercel.json         # Vercel 빌드 설정 (프론트엔드 정적 배포 전용)
├── requirements-local.txt  # Python 패키지 목록 (로컬 server.py 실행용)
├── viewer-src/         # 프론트엔드 소스 (React + Vite) — Vercel에 배포되는 부분
├── public/             # 빌드된 프론트엔드 (server.py가 로컬에서도 서빙 가능)
├── mri_rf/             # 모델 코드 + 체크포인트 (checkpoint.88.pt, git 미포함)
└── patient_mri/        # 환자별 슬라이스 PNG (p1)
```

## 아키텍처: 프론트(Vercel) + 추론 서버(별도)

체크포인트가 200MB, PyTorch(CPU) 설치본이 1GB 이상이라 Vercel Python
서버리스 함수 크기 제한(500MB)을 구조적으로 넘어섭니다. (`uv`가 torch를
설치한 뒤 함수 번들 크기가 1.85GB로 나와 배포 자체가 실패하는 것을
실제로 확인했습니다 — Python 함수로 이 추론을 올리는 건 불가능합니다.)

그래서 이 저장소는 **프론트엔드만 Vercel에 정적 배포**하고, 추론
서버(`server.py`)는 사용자의 PC나 Railway/Render 같은 별도 서버에서
상시 구동하는 구조로 나눴습니다. 프론트는 빌드 시 환경변수
`VITE_API_BASE`에 지정된 URL로 API를 호출합니다 (`viewer-src/src/lib/api.js`).

### Vercel 프로젝트에 설정할 환경변수

- `VITE_API_BASE` = 추론 서버의 공개 URL (예: `https://your-mri-server.example.com`)
  - Vercel 대시보드 → Project Settings → Environment Variables에서 추가 후 재배포.
  - 비워두면(로컬 개발 시) 상대 경로로 요청하며, `vite.config.js`의 프록시가
    `localhost:8765`(`server.py`)로 전달합니다.
- 추론 서버(`server.py`)는 이미 모든 응답에 `Access-Control-Allow-Origin: *`을
  붙이므로, Vercel(다른 origin)에서 호출해도 CORS 문제 없이 동작합니다.

> `pyproject.toml`(Poetry)은 제거했습니다 — Vercel의 Python 빌더가
> `pyproject.toml`을 발견하면 `[project]` 테이블이 있는 PEP 621 형식으로
> 간주하고 `uv lock`을 시도하는데, Poetry 형식과 충돌해 빌드가 실패했습니다.
> 같은 이유로 `requirements.txt`도 저장소 루트에 그대로 두면 Vercel이
> `/api` 함수가 하나도 없는데도 Python 프로젝트로 인식해서 `uv pip install`을
> 실행하려다 실패했습니다 — 그래서 `requirements-local.txt`로 이름을 바꿔
> Vercel의 자동 감지에서 제외했습니다 (로컬 `server.py` 실행에는 그대로 사용).

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
pip install -r requirements-local.txt
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

패키지가 누락된 것입니다. 아래 명령으로 추가 설치:

```bash
source .venv/bin/activate
pip install timm einops einx torchdiffeq networkx hyper-connections scipy lpips
```

**포트 충돌 (8765 이미 사용 중)**

`server.py` 상단의 `PORT = 8765`를 다른 값으로 변경 후 재실행.

**브라우저에서 화면이 안 나오는 경우**

`public/index.html` 이 없으면 프론트 빌드가 안 된 것입니다.
[2. 프론트엔드 빌드] 단계를 다시 실행하세요.
