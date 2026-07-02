# 🔬 AI-CASA: AI-Based Comprehensive Sperm Analysis System

**🌐 Language:** [한국어](README.md) | **English**

> A pre-clinical assistive analysis system that takes a sperm microscopy video and
> automatically analyzes **motility · movement patterns · morphology**, producing
> explainable, WHO-standard-based results.

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

## 📌 Overview

With just a single **sperm microscopy video**, this project automatically performs:

1. Frame-by-frame sperm detection with **YOLO11**
2. Sperm trajectory tracking with **ByteTrack**
3. Motility prediction with an **ensemble regression model**
4. **CASA kinematics** (VCL, VSL, VAP, LIN, STR, WOB, ALH) computation
5. Sperm morphology classification with **EfficientNet-B3** (head / acrosome / vacuole / tail)
6. Comprehensive interpretation and explainable reporting based on the **WHO 6th Edition** criteria
7. A **Flask web demo** where anyone can upload a video and view interactive results

> 🧑‍🔬 This is a **solo personal project**. After building this system as a baseline,
> I focused my research on the **domain adaptation** aspect of morphology analysis and
> developed it into an **academic paper (KKITS 2026)**. This baseline later evolved into
> the team project [**SEED**](https://github.com/MoriochoRadio/seed-project).
> For the research and paper, see [**Research & Publication**](#-research--publication) below.

> ⚠️ This system is **not** a medical diagnostic tool. It is an assistive analysis tool
> for reference before visiting a clinic.

---

## ⚡ What Makes AI-CASA Different

Core value compared to conventional CASA systems:

| Aspect | Conventional CASA | **AI-CASA (this system)** |
|---|---|---|
| **Equipment** | Dedicated hardware (tens of thousands of USD) | **Only a standard microscopy video needed** |
| **Analysis scope** | Mostly motility (morphology tested separately) | **Motility + kinematics + morphology, integrated** |
| **Personnel** | Trained technicians required | **Automated AI analysis** |
| **Interpretation** | Numbers-heavy, hard to interpret | **Layperson / expert mode toggle** |
| **Analysis time** | Time-consuming manual review | **Immediate results (within minutes)** |
| **Accessibility** | Limited to clinics/labs | **Any web browser, anywhere** |
| **Transparency** | Black box | **AI confidence + explicit decision rationale** |

---

## 🏆 Key Results

| Metric | This system | Reference |
|---|---|---|
| Motility MAE (5-Fold CV) | **6.9 %p** | motilitAI 7.31 %p |
| YOLO11 mAP50 | **0.677** | ~0.65 (YOLOv5l) |
| Morphology AUC (mean, MHSMA) | **0.727** | — |
| **VISEM normal morphology rate** (domain adaptation) | **0.5% → 16.3%** (32× ↑) | paper contribution |
| **Detection count retention after normalization** | **99.3%** | avoids −28.6% from pixel transforms |
| Decision-direction accuracy | **75~100%** | — |

> Motility analysis surpasses the best published performance on the same dataset (VISEM),
> and the **domain adaptation** of morphology analysis is documented in a separate paper (KKITS 2026).

---

## 📄 Research & Publication

Beyond implementation, this project investigated the **domain gap** problem in morphology
analysis in depth and developed it into an academic paper.

> **Practical Domain Adaptation for Computer-Aided Sperm Analysis:**
> **Quality-Aware Video Normalization and Augmentation-Based Morphology Classification**
>
> Tae-Kyoung Kim, Yong-Do Her · Department of Medical IT Engineering, Konyang University
> *Proceedings of KKITS (Conference on Knowledge Information Technology and Systems)*, Vol. 20, No. 1, 2026

### 🎯 Problem — Domain gap between training set and clinical video

The morphology classifier is trained on **MHSMA** (stained phase-contrast crops), but the
actual target, **VISEM** (unstained microscopy video), differs substantially in visual
characteristics. Even a model that excels on the benchmark (v2, MHSMA AUC 0.800) produced an
unrealistic **0.5%** normal morphology rate on VISEM.

### 🧩 Contribution 1 — Quality-aware video normalization

- Grades videos into S/A/B/C/F using six quality dimensions (resolution, frame rate,
  duration, stability, brightness, sharpness); F-grade videos are rejected.
- Applies **spec unification only (640×480 · 50 fps · 30 s)** and **excludes pixel-level
  transforms** such as CLAHE and grayscale.
- Because it does not disturb the pixel distribution the pretrained YOLO11 expects, it
  **retains 99.3% of detections** (avoiding the −28.6% detection drop from CLAHE and
  −14.9% from grayscale).

### 🧬 Contribution 2 — VisemStyleAugment (train-time domain augmentation)

- During training, stochastically applies motion blur, Gaussian noise, low resolution,
  CLAHE contrast, and intensity inversion to simulate the VISEM style.
- Removes fragmentary positional cues such as the Halo effect so the model **learns
  morphology itself**, with **zero additional inference cost**.
- Improves the VISEM normal morphology rate **0.5% → 16.3% (32×)**; all 4 VISEM
  participants meet the WHO criterion (≥4%).

### 📊 Model variant comparison (EfficientNet-B3, MHSMA 1,540 images)

| Model | Key configuration | MHSMA mean AUC | VISEM normal rate |
|---|---|---|---|
| v1 | BCE Loss | 0.752 | 0.5% |
| v2 | Focal Loss + WeightedSampler | **0.800** | 0.5% |
| **v3 (adopted)** | v2 + **VisemStyleAugment** | 0.727 | **16.3%** |
| v4 | v3 + strong augmentation | 0.714 | 18.2% |

> This shows that a high benchmark score (v2) does not guarantee real-world validity.
> **v3 was adopted** as the final model: it meets the WHO criterion while retaining the
> higher benchmark performance among the qualifying variants.

### 🔬 Five domain adaptation methods attempted

Pseudo-labeling (failed — confirmation bias) · geometric measurement (failed —
overestimation) · TTA (reaches 28.4% but 11× inference time) · ensemble v2+v3 (pulled
toward v2, 0.9%) · **VisemStyleAugment (16.3%, adopted at no extra cost)** — documented as
practical insights for future CASA research.

---

## 🏗️ System Architecture

```
User (browser)
        │
        │  video upload
        ▼
┌─────────────────────────────┐
│       Flask Web Server      │  app.py
│       (webapp/routes.py)    │  → API endpoints
└─────────────────────────────┘
        │
        │  background analysis (threading)
        ▼
┌─────────────────────────────┐
│       SpermDetector         │  YOLO11 sperm detection
│       (detector.py)         │  → estimates total sperm count N
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│       SpermTracker          │  ByteTrack sperm tracking
│       (tracker.py)          │  → trajectories + CASA kinematics
└─────────────────────────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌──────────────────────┐     ┌──────────────────────────┐
│  MotilityAnalyzer    │     │   MorphologyAnalyzer     │
│  (analyzer.py)       │     │   (morphology.py)        │
│  Ridge+RF ensemble   │     │   EfficientNet-B3 v3     │
│  → motility % pred.  │     │   → normal/abnormal      │
└──────────────────────┘     └──────────────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
         ┌─────────────────────────┐
         │     interpreter.py      │
         │  WHO 6th Ed. reasoning  │
         │  → verdict + confidence │
         └─────────────────────────┘
                       │
                       ▼
         📋 Interactive web report
         (layperson / expert toggle)
```

---

## 🎨 Web Demo Highlights

### Main page
- Drag-and-drop video upload
- MP4 / AVI / MOV formats (up to 500 MB)
- Preview of the 5-stage analysis pipeline

### Analysis progress screen
- Real-time progress + 5-stage visualization
- Per-stage model display (YOLO11, ByteTrack, Ridge+RF, EfficientNet-B3, WHO)
- Typically 30 s – 3 min

### Result report (core)
- **Overall verdict banner** — normal / some caution / specialist consultation advised / re-test needed
- **Mode toggle** — layperson (friendly explanation) ↔ expert (numbers-focused)
- **AI confidence box** — transparent disclosure of each model's validation results and limitations
- **Motility analysis** — donut chart + bar graph + WHO criteria review
- **CASA kinematics** — 7 parameters (velocity + movement patterns)
- **Morphology analysis** — abnormality rates for 4 parts (head / acrosome / vacuole / tail)
- **Decision rationale** — item-by-item explanation of why the verdict was reached

> 🎨 Dark navy + teal medical-domain design / printable

---

## 📊 Sample Output Report (CLI version)

```
============================================================
  🔬 AI Sperm Analysis Report
============================================================

[ Analysis Quality ]
  ✅ Confidence: High (100 / 100)

────────────────────────────────────────────────────────────
[ Sperm Motility Analysis ]  (detected sperm: 50)
────────────────────────────────────────────────────────────

  Progressively motile   (progressive):      21.4%  ████
  Slightly moving        (non-progressive):  26.2%  ░░░░░
  Immotile               (immotile):         52.4%  ··········
  ─────────────────────────────────────────────
  Total motile           (total motility):   47.6%

  📋 WHO Criteria Review
     🔴 Progressive motility 21.4% — well below the WHO normal threshold
     ✅ Total motility 47.6% — normal (threshold ≥ 40%)
     ⚠️  Immotile 52.4% — high proportion of non-moving sperm

────────────────────────────────────────────────────────────
[ Sperm Morphology Analysis ]
────────────────────────────────────────────────────────────

  Analyzed sperm: 1677  |  Normal morphology: 139 (8.3%)
  ⚠️  Head       (head)             abnormal  86.2%  █████████████████
  ⚠️  Acrosome   (front of head)    abnormal  80.7%  ████████████████
  ✅  Vacuole    (space in head)    abnormal  31.2%  ██████
  ✅  Tail       (tail)             abnormal  18.8%  ███

============================================================
[ Overall Verdict ]
============================================================
  🔴 Final verdict: [Specialist consultation advised]
  💬 A urology visit is strongly recommended for an accurate diagnosis.
============================================================
```

In the web version, the same information is shown as **interactive charts with friendly explanations**.

---

## 🚀 Quick Start

### 1. Environment setup

```bash
# create conda virtual environment
conda create -n sperm-ai python=3.10
conda activate sperm-ai

# PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# remaining packages
pip install -r requirements.txt
```

### 2-A. Run the web demo (recommended)

```bash
python app.py
```

Open `http://localhost:5000` in your browser → upload a video → automatic analysis.

### 2-B. Analyze directly from Python (3 lines)

```python
from src.pipeline import SpermAnalysisPipeline

pipeline = SpermAnalysisPipeline()
result   = pipeline.analyze('video.mp4')
pipeline.print_report(result)
```

---

## 📁 Project Structure

```
sperm-ai/
├── app.py                       # Flask entry point
├── webapp/                      # web application
│   ├── routes.py                #   API routes (/api/analyze, /api/status)
│   ├── tasks.py                 #   background analysis task management
│   ├── templates/               #   Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── index.html           #     main/upload page
│   │   ├── analyzing.html       #     analysis progress screen
│   │   └── result.html          #     result report
│   └── static/                  #   CSS / JS
│       ├── css/style.css
│       └── js/
│           ├── upload.js
│           ├── progress.js
│           └── result.js
│
├── src/                         # AI analysis core
│   ├── detector.py              #   SpermDetector (YOLO11)
│   ├── tracker.py               #   SpermTracker (ByteTrack + CASA)
│   ├── analyzer.py              #   MotilityAnalyzer (ensemble regression)
│   ├── morphology.py            #   MorphologyAnalyzer (EfficientNet-B3)
│   ├── interpreter.py           #   WHO interpretation + confidence + verdict
│   └── pipeline.py              #   SpermAnalysisPipeline (integration)
│
├── models/                      # trained models (not in Git, download separately)
│   ├── yolo11_sperm_v2/                  #   YOLO11 detection model
│   ├── motility_ensemble.pkl             #   motility regression model
│   └── morphology_efficientnet_b3_v3.pt  #   morphology classifier (v3)
│
├── notebooks/                   # experiment notebooks (01~16)
├── docs/                        # documentation
│   ├── architecture.md          #   system design
│   └── performance.md           #   detailed performance analysis
│
├── README.md                    # Korean
├── README.en.md                 # English (this file)
├── requirements.txt
└── bytetrack_custom.yaml
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Backend** | Flask 3.0 / Werkzeug / Threading |
| **Frontend** | HTML5 / CSS3 / Vanilla JavaScript (Canvas API) |
| **AI framework** | PyTorch 2.6 / torchvision / scikit-learn |
| **Object detection** | YOLO11 (ultralytics) |
| **Object tracking** | ByteTrack (built into ultralytics) |
| **Morphology classification** | EfficientNet-B3 (multi-task) |
| **Motility regression** | Ridge + RandomForest ensemble |
| **Video processing** | OpenCV / NumPy |
| **GPU acceleration** | CUDA 12.4 / cuDNN |
| **Version control** | Git / GitHub (main + feature branch workflow) |

---

## 🔬 Datasets Used

| Dataset | Participants | Purpose | Notes |
|---|---|---|---|
| VISEM-Tracking | 20 | YOLO11 training | bounding boxes + tracking IDs |
| VISEM (original) | 85 | regression model training | video + motility CSV |
| MHSMA | 235 (1,540 images) | morphology model training | normal/abnormal labels for 4 parts |

---

## ⚙️ Development Environment

| Item | Spec |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3080 Ti Laptop (16GB VRAM) |
| RAM | 32GB |
| CPU | Intel i7-12800HX |
| Python | 3.10 |
| PyTorch | 2.6.0 + CUDA 12.4 |

---

## 📈 Development Roadmap

```
Phase 1  ✅ Done   Motility analysis (MAE 6.9%p, surpasses paper)          v1.0.0
Phase 2  ✅ Done   CASA kinematics (VCL/VSL/VAP/LIN/STR/WOB/ALH)            v1.1.0
Phase 3  ✅ Done   Morphology analysis (EfficientNet-B3, AUC 0.727)        v1.2.0
Phase 4  ✅ Done   AI-CASA integrated report (layperson-friendly)          v1.3.0
Phase 5  ✅ Done   Flask web demo (interactive report)                     v1.4.0
Phase 6  ✅ Done   Documentation (user/admin guides · scenarios)
Phase 7  ✅ Done   Morphology domain adaptation research
                   (quality-aware normalization + VisemStyleAugment)
Phase 8  ✅ Done   Academic paper writing & presentation (KKITS 2026)  ← research outcome
Phase 9  🔜 Planned Multi-center clinical data validation · generalization
```

---

## 🏷️ Releases

| Version | Highlights |
|---|---|
| **v1.4.0** | Flask web demo + interactive report (layperson/expert modes) |
| v1.3.0 | Layperson-friendly report + overall verdict system |
| v1.2.0 | Phase 3 morphology analysis (EfficientNet-B3 v3) integration |
| v1.1.0 | Phase 2 CASA kinematics (7 metrics) added |
| v1.0.0 | Phase 1 motility analysis (surpasses paper) |

---

## ⚠️ Notes & Limitations

- This system is **not a medical diagnostic tool**.
- Morphology analysis may somewhat overestimate abnormality rates due to the **domain gap**
  between MHSMA (stained) and VISEM (unstained).
- It is optimized for the VISEM dataset environment; performance may vary on other microscope models.
- Analysis confidence is low for very low-concentration samples (N < 10).
- Always seek a final judgment from **qualified medical professionals**.

---

## 📚 References

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

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Solo personal project · 2026 · The morphology domain adaptation research became the [KKITS 2026 paper](#-research--publication),*
*and this baseline evolved into the team project [SEED](https://github.com/MoriochoRadio/seed-project).*
*Contact: [GitHub Issues](https://github.com/MoriochoRadio/sperm-ai/issues)*
