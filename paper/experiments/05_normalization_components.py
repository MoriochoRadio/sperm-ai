"""
05_normalization_components.py
─────────────────────────────────
정규화 구성 요소별 효과 분리 측정

가설:
  "사양 변환만" vs "CLAHE 포함" 비교
  → CLAHE가 진짜 효과 있는지 분리 검증
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from ultralytics import YOLO
import time
import json


def evaluate(model, frames, label):
    """프레임 리스트 → detection 평가"""
    counts, confs = [], []
    for frame in frames:
        results = model.predict(
            frame, conf=0.25, iou=0.6,
            verbose=False, device=0)
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                sperm_mask = boxes.cls == 0
                counts.append(int(sperm_mask.sum()))
                if sperm_mask.sum() > 0:
                    confs.extend(
                        boxes.conf[sperm_mask].cpu().numpy().tolist())
            else:
                counts.append(0)

    return {
        'label':       label,
        'avg_count':   float(np.mean(counts)) if counts else 0,
        'std_count':   float(np.std(counts)) if counts else 0,
        'avg_conf':    float(np.mean(confs)) if confs else 0,
    }


def main():
    print("=" * 80)
    print("  실험 5: 정규화 구성 요소별 효과 분리")
    print("=" * 80)

    # 모델 + 영상
    model = YOLO(PROJECT_ROOT + r'\models'
                 r'\yolo11_sperm_v2\weights\best.pt')

    video_path = (PROJECT_ROOT + r'\data\raw'
                  r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                  r'\Train\11\11.mp4')

    # 프레임 50개 샘플링
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // 50)

    frames = []
    idx = 0
    while len(frames) < 50:
        ret, f = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(f.copy())
        idx += 1
    cap.release()
    print(f"\n📹 프레임 추출 완료: {len(frames)}개")

    # ── 4가지 처리 비교 ──────────────────────────────────
    print("\n" + "─" * 80)
    print("  4가지 처리 방식 비교")
    print("─" * 80)

    # 1. 원본
    print("\n[ 1. 원본 (변환 없음) ]")
    r1 = evaluate(model, frames, '원본')
    print(f"   검출: {r1['avg_count']:.1f}±{r1['std_count']:.1f}, "
          f"conf: {r1['avg_conf']:.4f}")

    # 2. 그레이스케일만
    frames_gray = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames_gray.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    print("\n[ 2. 그레이스케일만 (CLAHE 없음) ]")
    r2 = evaluate(model, frames_gray, '그레이스케일')
    print(f"   검출: {r2['avg_count']:.1f}±{r2['std_count']:.1f}, "
          f"conf: {r2['avg_conf']:.4f}")

    # 3. CLAHE만
    frames_clahe = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        frames_clahe.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    print("\n[ 3. CLAHE 적용 ]")
    r3 = evaluate(model, frames_clahe, 'CLAHE')
    print(f"   검출: {r3['avg_count']:.1f}±{r3['std_count']:.1f}, "
          f"conf: {r3['avg_conf']:.4f}")

    # 4. CLAHE 약하게 (clipLimit=1.0)
    frames_weak_clahe = []
    weak_clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = weak_clahe.apply(gray)
        frames_weak_clahe.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    print("\n[ 4. CLAHE 약하게 (clipLimit=1.0) ]")
    r4 = evaluate(model, frames_weak_clahe, 'CLAHE 약')
    print(f"   검출: {r4['avg_count']:.1f}±{r4['std_count']:.1f}, "
          f"conf: {r4['avg_conf']:.4f}")

    # ── 비교 표 ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  📊 정규화 구성 요소별 영향 비교")
    print("=" * 80)
    print(f"{'처리':<25} {'평균 검출':<14} {'Confidence':<14} {'baseline 대비':<15}")
    print("─" * 80)

    baseline = r1['avg_count']
    for r in [r1, r2, r3, r4]:
        delta = r['avg_count'] - baseline
        delta_pct = (delta / baseline * 100) if baseline > 0 else 0
        print(f"{r['label']:<25} "
              f"{r['avg_count']:<14.1f} "
              f"{r['avg_conf']:<14.4f} "
              f"{delta:+.1f} ({delta_pct:+.1f}%)")

    print("\n" + "=" * 80)
    print("  🔍 핵심 분석")
    print("=" * 80)
    print("""
- "원본"이 baseline (가장 좋음 예상)
- "그레이스케일만" vs "원본": 색상 정보의 영향
- "CLAHE 강" vs "CLAHE 약": 강도의 영향
→ 어느 단계에서 성능이 떨어지는지 진단
""")

    # 저장
    output_dir = PROJECT_ROOT + r'\paper\experiments\05_results'
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([r1, r2, r3, r4], f, indent=2, ensure_ascii=False)
    print(f"📁 결과 저장: {json_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()