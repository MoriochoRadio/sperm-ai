# 🔬 AI-CASA: AI 기반 정자 종합 분석 시스템

**🌐 Language:** **한국어** | [English](README.en.md)

> 정자 현미경 영상을 입력받아 **운동성 · 운동 패턴 · 형태**를 자동 분석하고
> WHO 기준 기반 설명형 결과를 출력하는 병원 전 단계 보조 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-11-darkgreen)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-3.0-black)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-5%20Complete-brightgreen)]()
[![Demo](https://img.shields.io/badge/Demo-Live-2DD4BF)]()
[![Version](https://img.shields.io/badge/Version-v1.4.0-blueviolet)]()
[![Paper](https://img.shields.io/badge/Paper-KKITS%202026-b31b1b)]()

---

## 📌 프로젝트 소개

이 프로젝트는 **정자 현미경 영상** 하나만 있으면:

1. **YOLO11** 로 정자를 프레임별 자동 탐지
2. **ByteTrack** 으로 정자 이동 경로 추적
3. **앙상블 회귀 모델** 로 운동성 수치 예측
4. **CASA 키네마틱** (VCL, VSL, VAP, LIN, STR, WOB, ALH) 계산
5. **EfficientNet-B3** 로 정자 형태 분류 (머리/첨체/공포/꼬리)
6. **WHO 6판 기준** 으로 종합 해석 및 설명형 보고서 출력
7. **Flask 웹 데모** 로 누구나 영상 업로드 후 인터랙티브 결과 확인

을 자동으로 수행합니다.

> 🧑‍🔬 **1인 개인 프로젝트**입니다. 이 시스템을 베이스라인으로 구축한 뒤,
> 형태 분석의 **도메인 적응(domain adaptation)** 부분을 집중적으로 연구하여
> **학술 논문(KKITS 2026)** 으로 발전시켰습니다. 이후 이 베이스라인은 팀 프로젝트
> [**SEED**](https://github.com/MoriochoRadio/seed-project)로 이어졌습니다.
> 연구·논문 내용은 아래 [**연구 성과 및 논문**](#-연구-성과-및-논문) 참고.

> ⚠️ 본 시스템은 의학적 진단 도구가 아닙니다. 병원 방문 전 참고용 보조 분석 도구입니다.

---

## ⚡ AI-CASA의 차별점

기존 CASA 시스템과 비교했을 때 본 시스템의 핵심 가치:

| 항목 | 기존 CASA | **AI-CASA (본 시스템)** |
|---|---|---|
| **장비** | 전용 장비 필요 (수천만 원) | **일반 현미경 영상만 있으면 OK** |
| **분석 항목** | 운동성 위주 (형태는 별도 검사) | **운동성 + 키네마틱 + 형태 통합** |
| **인력** | 전문 검사 인력 필요 | **AI 자동 분석** |
| **결과 해석** | 수치 위주, 해석 어려움 | **일반인용 / 전문가용 토글** |
| **분석 시간** | 수동 검토 시간 소요 | **즉시 결과 (수분 내)** |
| **접근성** | 병원·연구실 한정 | **웹 브라우저 어디서나** |
| **투명성** | 블랙박스 | **AI 신뢰도 + 판정 근거 명시** |

---

## 🏆 핵심 성과

| 지표 | 본 시스템 | 비교 |
|---|---|---|
| 운동성 MAE (5-Fold CV) | **6.9%p** | motilitAI 7.31%p |
| YOLO11 mAP50 | **0.677** | ~0.65 (YOLOv5l) |
| 형태 분류 AUC (평균, MHSMA) | **0.727** | — |
| **VISEM 정상형태율** (도메인 적응) | **0.5% → 16.3%** (32배 ↑) | 논문 기여 |
| **정규화 후 검출 수 유지율** | **99.3%** | 픽셀 변환 시 −28.6% 회피 |
| 판정 방향 정확도 | **75~100%** | — |

> 운동성 분석은 동일 데이터셋(VISEM)에서 논문 최고 성능을 초과 달성했고,
> 형태 분석의 **도메인 적응**은 별도 논문(KKITS 2026)으로 정리했습니다.

---

## 📄 연구 성과 및 논문

이 프로젝트는 단순 구현에 그치지 않고, **형태 분석의 도메인 갭(domain gap)** 문제를
집중적으로 연구하여 학술 논문으로 발전시켰습니다.

> **Practical Domain Adaptation for Computer-Aided Sperm Analysis:**
> **Quality-Aware Video Normalization and Augmentation-Based Morphology Classification**
> *(컴퓨터 보조 정자 분석을 위한 실용적 도메인 적응: 품질 인식 영상 정규화 및 증강 기반 형태 분류)*
>
> 김태경, 허용도 · 건양대학교 의료IT공학과
> *Proceedings of KKITS (Conference on Knowledge Information Technology and Systems)*, Vol. 20, No. 1, 2026

### 🎯 문제 — 학습셋 ↔ 임상 영상의 도메인 갭

형태 분류 모델은 **MHSMA**(염색 위상차 크롭)로 학습되지만, 실제 적용 대상인
**VISEM**(비염색 현미경 영상)과 시각적 특성이 크게 다릅니다. 벤치마크에서 우수한
모델(v2, MHSMA AUC 0.800)조차 VISEM에서는 정상형태율 **0.5%** 라는 비현실적 결과를 냈습니다.

### 🧩 기여 1 — 품질 인식 영상 정규화

- 6차원 품질 지표(해상도·프레임률·재생시간·안정성·밝기·선명도)로 영상을 S/A/B/C/F 등급화 (F등급 거부)
- **사양 통일(640×480·50fps·30s)만** 적용하고, CLAHE·그레이스케일 같은 **픽셀 단위 변환은 배제**
- 사전학습 YOLO11의 픽셀 분포를 깨뜨리지 않아 검출 수 **99.3% 유지**
  (CLAHE 적용 시 −28.6%, 그레이스케일 −14.9%의 검출 저하를 회피)

### 🧬 기여 2 — VisemStyleAugment (학습 단계 도메인 증강)

- 학습 시 모션 블러·가우시안 노이즈·저해상도·CLAHE 대비·명암 반전을 확률적으로 적용해 VISEM 스타일을 시뮬레이션
- Halo 등 단편적 위치 단서를 제거해 모델이 **형태 자체를 학습**하도록 유도, **추론 시 추가 비용 0**
- VISEM 정상형태율 **0.5% → 16.3% (32배 개선)**, VISEM 참가자 4명 전원 WHO 기준(≥4%) 충족

### 📊 모델 변형 비교 (EfficientNet-B3, MHSMA 1,540장)

| 모델 | 주요 구성 | MHSMA 평균 AUC | VISEM 정상형태율 |
|---|---|---|---|
| v1 | BCE Loss | 0.752 | 0.5% |
| v2 | Focal Loss + WeightedSampler | **0.800** | 0.5% |
| **v3 (채택)** | v2 + **VisemStyleAugment** | 0.727 | **16.3%** |
| v4 | v3 + 강한 증강 | 0.714 | 18.2% |

> 벤치마크 점수(v2)가 실제 적용 유효성을 보장하지 않음을 보이고, WHO 기준을 충족하면서
> 벤치마크 성능이 더 높은 **v3를 최종 채택**했습니다.

### 🔬 시도한 5가지 도메인 적응 기법

의사 라벨링(확증 편향으로 실패) · 기하학적 측정(과대추정으로 실패) · TTA(28.4% 도달하나 추론 11배) ·
앙상블 v2+v3(v2에 끌려가 0.9%) · **VisemStyleAugment(16.3%, 추가 비용 없이 채택)** —
향후 CASA 연구를 위한 실용적 시사점으로 문서화했습니다.

---

## 🏗️ 시스템 아키텍처

```
사용자 (브라우저)
        │
        │  영상 업로드
        ▼
┌─────────────────────────────┐
│       Flask Web Server      │  app.py
│       (webapp/routes.py)    │  → API 엔드포인트
└─────────────────────────────┘
        │
        │  백그라운드 분석 (Threading)
        ▼
┌─────────────────────────────┐
│       SpermDetector         │  YOLO11 정자 탐지
│       (detector.py)         │  → 전체 정자 수 N 추정
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│       SpermTracker          │  ByteTrack 정자 추적
│       (tracker.py)          │  → 이동 경로 + CASA 키네마틱
└─────────────────────────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌──────────────────────┐     ┌──────────────────────────┐
│  MotilityAnalyzer    │     │   MorphologyAnalyzer     │
│  (analyzer.py)       │     │   (morphology.py)        │
│  Ridge+RF 앙상블      │     │   EfficientNet-B3 v3     │
│  → 운동성 % 예측      │     │   → 형태 정상/비정상     │
└──────────────────────┘     └──────────────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
         ┌─────────────────────────┐
         │     interpreter.py      │
         │  WHO 6판 기준 해석       │
         │  → 판정 + 신뢰도 + 권고  │
         └─────────────────────────┘
                       │
                       ▼
         📋 인터랙티브 웹 보고서
         (일반인용 / 전문가용 토글)
```

---

## 🎨 웹 데모 주요 기능

### 메인 페이지
- 영상 드래그 앤 드롭 업로드
- MP4 / AVI / MOV 형식 지원 (최대 500MB)
- 분석 파이프라인 5단계 미리보기

### 분석 진행 화면
- 실시간 진행률 + 5단계 시각화
- 단계별 사용 모델 표시 (YOLO11, ByteTrack, Ridge+RF, EfficientNet-B3, WHO)
- 평균 30초~3분 소요

### 결과 보고서 (핵심)
- **종합 판정 배너** — 정상 / 일부 주의 / 전문의 상담 권고 / 재검사 필요
- **모드 토글** — 일반인용(친절한 설명) ↔ 전문가용(수치 중심)
- **AI 신뢰도 박스** — 각 모델의 검증 결과와 한계점 투명 공개
- **운동성 분석** — 도넛 차트 + 막대 그래프 + WHO 기준 검토
- **CASA 키네마틱** — 7가지 파라미터 (속도 + 운동 패턴)
- **형태 분석** — 4부위(머리/첨체/공포/꼬리) 이상 비율
- **종합 판정 근거** — 왜 이 판정이 나왔는지 항목별 설명

> 🎨 다크 네이비 + 청록 의료 도메인 디자인 / 인쇄 가능

---

## 📊 출력 보고서 예시 (CLI 버전)

```
============================================================
  🔬 AI 정자 분석 보고서
============================================================

【 분석 품질 】
  ✅ 신뢰도: 높음 (100점 / 100점)

────────────────────────────────────────────────────────────
【 정자 운동성 분석 】  (탐지된 정자: 50개)
────────────────────────────────────────────────────────────

  앞으로 잘 나아가는 정자  (전진 운동성):   21.4%  ████
  조금 움직이는 정자      (비전진 운동성): 26.2%  ░░░░░
  움직이지 않는 정자      (비운동성):      52.4%  ··········
  ─────────────────────────────────────────────
  전체 움직이는 정자      (총 운동성):     47.6%

  📋 WHO 기준 검토 결과
     🔴 전진 운동성 21.4% — WHO 정상 기준보다 현저히 낮음
     ✅ 총 운동성 47.6% — 정상 (기준 40% 이상)
     ⚠️  비운동성 52.4% — 움직이지 않는 정자 비율이 높음

────────────────────────────────────────────────────────────
【 정자 형태 분석 】
────────────────────────────────────────────────────────────

  분석된 정자: 1677개  |  정상 형태: 139개 (8.3%)
  ⚠️  머리    (두부)          이상  86.2%  █████████████████
  ⚠️  첨체    (머리 앞부분)    이상  80.7%  ████████████████
  ✅  공포    (머리 내 공간)   이상  31.2%  ██████
  ✅  꼬리    (미부)          이상  18.8%  ███

============================================================
【 종합 판정 】
============================================================
  🔴 최종 판정: [전문의 상담 권고]
  💬 정확한 진단을 위해 비뇨의학과 방문을 강력히 권고합니다.
============================================================
```

웹 버전에서는 동일 정보가 **인터랙티브 차트와 친절한 설명**으로 표시됩니다.

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# conda 가상환경 생성
conda create -n sperm-ai python=3.10
conda activate sperm-ai

# PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 나머지 패키지
pip install -r requirements.txt
```

### 2-A. 웹 데모 실행 (추천)

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속 → 영상 업로드 → 자동 분석

### 2-B. Python 코드로 직접 분석 (3줄)

```python
from src.pipeline import SpermAnalysisPipeline

pipeline = SpermAnalysisPipeline()
result   = pipeline.analyze('video.mp4')
pipeline.print_report(result)
```

---

## 📁 프로젝트 구조

```
sperm-ai/
├── app.py                       # Flask 진입점
├── webapp/                      # 웹 애플리케이션
│   ├── routes.py                #   API 라우트 (/api/analyze, /api/status)
│   ├── tasks.py                 #   백그라운드 분석 작업 관리
│   ├── templates/               #   Jinja2 HTML 템플릿
│   │   ├── base.html
│   │   ├── index.html           #     메인/업로드 페이지
│   │   ├── analyzing.html       #     분석 진행 화면
│   │   └── result.html          #     결과 보고서
│   └── static/                  #   CSS / JS
│       ├── css/style.css
│       └── js/
│           ├── upload.js
│           ├── progress.js
│           └── result.js
│
├── src/                         # AI 분석 코어
│   ├── detector.py              #   SpermDetector (YOLO11)
│   ├── tracker.py               #   SpermTracker (ByteTrack + CASA)
│   ├── analyzer.py              #   MotilityAnalyzer (앙상블 회귀)
│   ├── morphology.py            #   MorphologyAnalyzer (EfficientNet-B3)
│   ├── interpreter.py           #   WHO 해석 + 신뢰도 + 종합 판정
│   └── pipeline.py              #   SpermAnalysisPipeline (통합)
│
├── models/                      # 학습된 모델 (Git 미포함, 별도 다운로드)
│   ├── yolo11_sperm_v2/                  #   YOLO11 탐지 모델
│   ├── motility_ensemble.pkl             #   운동성 회귀 모델
│   └── morphology_efficientnet_b3_v3.pt  #   형태 분류 모델 (v3)
│
├── notebooks/                   # 실험 기록 노트북 (01~16)
├── docs/                        # 문서
│   ├── architecture.md          #   시스템 설계
│   └── performance.md           #   상세 성능 분석
│
├── README.md                    # 한국어
├── README.en.md                 # English
├── requirements.txt
└── bytetrack_custom.yaml
```

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| **백엔드** | Flask 3.0 / Werkzeug / Threading |
| **프론트엔드** | HTML5 / CSS3 / Vanilla JavaScript (Canvas API) |
| **AI 프레임워크** | PyTorch 2.6 / torchvision / scikit-learn |
| **객체 탐지** | YOLO11 (ultralytics) |
| **객체 추적** | ByteTrack (ultralytics 내장) |
| **형태 분류** | EfficientNet-B3 (멀티태스크) |
| **운동성 회귀** | Ridge + RandomForest 앙상블 |
| **영상 처리** | OpenCV / NumPy |
| **GPU 가속** | CUDA 12.4 / cuDNN |
| **버전 관리** | Git / GitHub (main + feature 브랜치 워크플로우) |

---

## 🔬 사용 데이터셋

| 데이터셋 | 참가자 | 용도 | 특징 |
|---|---|---|---|
| VISEM-Tracking | 20명 | YOLO11 학습 | 바운딩박스 + 추적ID |
| VISEM 원본 | 85명 | 회귀 모델 학습 | 영상 + 운동성 CSV |
| MHSMA | 235명 (1,540장) | 형태 모델 학습 | 4부위 정상/비정상 레이블 |

---

## ⚙️ 개발 환경

| 항목 | 스펙 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3080 Ti Laptop (16GB VRAM) |
| RAM | 32GB |
| CPU | Intel i7-12800HX |
| Python | 3.10 |
| PyTorch | 2.6.0 + CUDA 12.4 |

---

## 📈 개발 로드맵

```
Phase 1  ✅ 완료   운동성 분석 (MAE 6.9%p, 논문 초과)              v1.0.0
Phase 2  ✅ 완료   CASA 키네마틱 (VCL/VSL/VAP/LIN/STR/WOB/ALH)      v1.1.0
Phase 3  ✅ 완료   형태 분석 (EfficientNet-B3, AUC 0.727)          v1.2.0
Phase 4  ✅ 완료   AI-CASA 통합 보고서 (일반인 친화적)              v1.3.0
Phase 5  ✅ 완료   Flask 웹 데모 (인터랙티브 보고서)                v1.4.0
Phase 6  ✅ 완료   문서화 (사용자/운영 가이드 · 시나리오)
Phase 7  ✅ 완료   형태 도메인 적응 연구 (품질 인식 정규화 + VisemStyleAugment)
Phase 8  ✅ 완료   학술 논문 작성·발표 (KKITS 2026)  ← 연구 성과
Phase 9  🔜 예정   다기관 임상 데이터 검증 · 일반화 검증
```

---

## 🏷️ Releases

| 버전 | 주요 내용 |
|---|---|
| **v1.4.0** | Flask 웹 데모 + 인터랙티브 보고서 (일반인/전문가 모드) |
| v1.3.0 | 일반인 친화적 보고서 + 종합 판정 시스템 |
| v1.2.0 | Phase 3 형태 분석 (EfficientNet-B3 v3) 통합 |
| v1.1.0 | Phase 2 CASA 키네마틱 7종 추가 |
| v1.0.0 | Phase 1 운동성 분석 (논문 초과 달성) |

---

## ⚠️ 주의사항 및 한계

- 이 시스템은 **의학적 진단 도구가 아닙니다**
- 형태 분석은 MHSMA(염색)↔VISEM(비염색) **도메인 차이**로 비정상 비율이 다소 과대 측정될 수 있음
- VISEM 데이터셋 환경에 최적화되어 있어 다른 현미경 기종에서는 성능이 달라질 수 있음
- 정자 농도가 매우 낮은 샘플(N < 10)은 분석 신뢰도가 낮음
- 최종 판단은 반드시 **전문 의료진**에게 받으시기 바랍니다

---

## 📚 참고 문헌

```bibtex
@inproceedings{kim2026casa,
  author    = {Kim, Tae-Kyoung and Her, Yong-Do},
  title     = {Practical Domain Adaptation for Computer-Aided Sperm Analysis:
               Quality-Aware Video Normalization and Augmentation-Based
               Morphology Classification},
  booktitle = {Proceedings of Conference on Knowledge Information Technology
               and Systems (KKITS)},
  volume    = {20},
  number    = {1},
  year      = {2026},
  publisher = {KKITS}
}

@article{thambawita2023visem,
  author  = {Thambawita, Vajira and Hicks, Steven A. and others},
  title   = {VISEM-Tracking, a human spermatozoa tracking dataset},
  journal = {Scientific Data},
  year    = {2023},
  doi     = {10.1038/s41597-023-02173-4}
}

@article{ottl2022motilitai,
  author  = {Ottl, Sandra and Amiriparian, Shahin and others},
  title   = {motilitAI: A machine learning framework for automatic
             prediction of human sperm motility},
  journal = {iScience},
  year    = {2022},
  doi     = {10.1016/j.isci.2022.104644}
}

@article{javadi2019mhsma,
  author  = {Javadi, Shahin and Mirroshandel, Seyed Abolghasem},
  title   = {A novel deep learning method for automatic assessment
             of human sperm images},
  journal = {Computers in Biology and Medicine},
  year    = {2019},
  doi     = {10.1016/j.compbiomed.2019.04.030}
}

@manual{who2021,
  title  = {WHO laboratory manual for the examination and
            processing of human semen, Sixth Edition},
  author = {{World Health Organization}},
  year   = {2021}
}
```

---

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

*1인 개인 프로젝트 · 2026 · 형태 도메인 적응 연구는 [KKITS 2026 논문](#-연구-성과-및-논문)으로,*
*이 베이스라인은 팀 프로젝트 [SEED](https://github.com/MoriochoRadio/seed-project)로 이어졌습니다.*
*문의: [GitHub Issues](https://github.com/MoriochoRadio/sperm-ai/issues)*
