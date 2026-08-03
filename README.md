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
├── api/[...path].py    # Vercel 서버리스 배포용 함수 (실험적)
├── vercel.json         # Vercel 빌드/라우팅 설정
├── requirements.txt    # Python 패키지 목록
├── viewer-src/         # 프론트엔드 소스 (React + Vite)
├── public/             # 빌드된 프론트엔드 (server.py가 서빙)
├── mri_rf/             # 모델 코드 + 체크포인트 (checkpoint.88.pt, git 미포함)
└── patient_mri/        # 환자별 슬라이스 PNG (p1)
```

## Vercel 배포에 대한 주의

체크포인트가 200MB가 넘고 PyTorch(CPU) 추론까지 필요해서, Vercel
서버리스 함수의 크기/실행시간 제한에 걸릴 가능성이 높은 실험적 구조입니다.
또한 체크포인트 파일은 GitHub 100MB 제한 때문에 저장소에 커밋하지
않았으므로, 배포된 함수에서는 모델이 로드되지 않고 "체크포인트 없음"
상태로 남습니다 (로컬 `server.py`에서는 정상 동작). 실제로 배포에서도
추론이 되게 하려면 Git LFS나 외부 스토리지(S3, Hugging Face Hub 등)에서
체크포인트를 받아오도록 `mri_infer_core.py`를 수정해야 합니다.

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

1. 왼쪽 탐색기 패널에서 **폴더 열기** 또는 드래그&드롭으로 환자 폴더 불러오기
   - 폴더 구조: `루트폴더/환자명/*.png`
2. 슬라이스 탐색: `←` `→` 키 또는 마우스 휠
3. **T1 변환** 버튼 클릭 → 서버에서 T2→T1 추론 시작
   - 헤더에 진행률 표시 (`T1 추론 중... N/전체`)
   - 현재 슬라이스 추론 중이면 캔버스에 스피너 오버레이 표시
4. 추론 완료 후 **T1 변환** 탭 선택 시 변환 이미지 확인

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
