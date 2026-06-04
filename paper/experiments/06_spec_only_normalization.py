"""
06_spec_only_normalization.py
─────────────────────────────────
사양 변환만 (CLAHE 제외) 효과 검증

가설:
  - 다른 사양 영상 → 정규화 (사양만) → 원본과 동등 성능
  - CLAHE 제거가 더 안전함을 입증
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from ultralytics import YOLO
import json


def evaluate(model, frames, label):
    counts, confs = [], []
    for frame in frames:
        results = model.predict(
            frame, conf=0.25, iou=0.6,
            verbose=False, device=0)
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                m = boxes.cls == 0
                counts.append(int(m.sum()))
                if m.sum() > 0:
                    confs.extend(
                        boxes.conf[m].cpu().numpy().tolist())
            else:
                counts.append(0)
    return {
        'label':     label,
        'avg_count': float(np.mean(counts)) if counts else 0,
        'std_count': float(np.std(counts)) if counts else 0,
        'avg_conf':  float(np.mean(confs)) if confs else 0,
    }


def normalize_spec_only(frame, target_w=640, target_h=480):
    """사양 변환만 - CLAHE 없음"""
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame.copy()
    return cv2.resize(frame, (target_w, target_h),
                      interpolation=cv2.INTER_AREA
                      if w > target_w else cv2.INTER_LINEAR)


def normalize_with_clahe(frame, target_w=640, target_h=480):
    """사양 변환 + CLAHE (현재 normalizer.py와 동일)"""
    h, w = frame.shape[:2]
    if not (w == target_w and h == target_h):
        frame = cv2.resize(frame, (target_w, target_h),
                           interpolation=cv2.INTER_AREA
                           if w > target_w else cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    print("=" * 80)
    print("  실험 6: 사양 변환만 vs CLAHE 포함 정규화")
    print("=" * 80)

    model = YOLO(PROJECT_ROOT + r'\models'
                 r'\yolo11_sperm_v2\weights\best.pt')
    print("✓ 모델 로드 완료")

    video_path = (PROJECT_ROOT + r'\data\raw'
                  r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                  r'\Train\11\11.mp4')

    # 프레임 추출 (50개)
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
    print(f"✓ 프레임 추출: {len(frames)}개\n")

    # ── 시뮬레이션: 사양이 다른 영상 ────────────────────
    # HD 1280×720 → 640×480 변환 시나리오
    print("─" * 80)
    print("시나리오 1: 원본 640×480 (사양 일치)")
    print("─" * 80)
    r1_orig = evaluate(model, frames, '원본 640×480')
    print(f"   {r1_orig['label']:<25}: "
          f"{r1_orig['avg_count']:.1f}±{r1_orig['std_count']:.1f}, "
          f"conf {r1_orig['avg_conf']:.4f}")

    # 1280x720 시뮬레이션 → 640×480 변환
    print("\n─" * 80)
    print("시나리오 2: HD(1280×720) → 정규화")
    print("─" * 80)
    frames_hd = [cv2.resize(f, (1280, 720),
                            interpolation=cv2.INTER_LINEAR)
                 for f in frames]

    # 2-A: HD 그대로 (변환 X) 평가
    r2_raw = evaluate(model, frames_hd, 'HD 변환 없음')
    print(f"   {r2_raw['label']:<25}: "
          f"{r2_raw['avg_count']:.1f}±{r2_raw['std_count']:.1f}, "
          f"conf {r2_raw['avg_conf']:.4f}")

    # 2-B: HD → 사양만 변환 (CLAHE 없음)
    frames_spec_only = [normalize_spec_only(f) for f in frames_hd]
    r2_spec = evaluate(model, frames_spec_only, 'HD→사양만')
    print(f"   {r2_spec['label']:<25}: "
          f"{r2_spec['avg_count']:.1f}±{r2_spec['std_count']:.1f}, "
          f"conf {r2_spec['avg_conf']:.4f}")

    # 2-C: HD → 사양 + CLAHE (기존 normalizer.py)
    frames_with_clahe = [normalize_with_clahe(f) for f in frames_hd]
    r2_clahe = evaluate(model, frames_with_clahe, 'HD→사양+CLAHE')
    print(f"   {r2_clahe['label']:<25}: "
          f"{r2_clahe['avg_count']:.1f}±{r2_clahe['std_count']:.1f}, "
          f"conf {r2_clahe['avg_conf']:.4f}")

    # ── 비교 표 ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  📊 정규화 방식 비교")
    print("=" * 80)
    print(f"{'처리':<25} {'검출 수':<14} {'Confidence':<14} {'기준 대비':<15}")
    print("─" * 80)

    baseline = r1_orig['avg_count']
    for r in [r1_orig, r2_raw, r2_spec, r2_clahe]:
        delta = r['avg_count'] - baseline
        delta_pct = (delta / baseline * 100) if baseline > 0 else 0
        marker = ""
        if r['label'] == 'HD→사양만':
            marker = " ⭐ 추천"
        elif r['label'] == 'HD→사양+CLAHE':
            marker = " (현재 시스템)"
        print(f"{r['label']:<25} "
              f"{r['avg_count']:<14.1f} "
              f"{r['avg_conf']:<14.4f} "
              f"{delta:+.1f} ({delta_pct:+.1f}%){marker}")

    # 분석
    print("\n" + "=" * 80)
    print("  🔍 결론")
    print("=" * 80)

    # CLAHE 제거 시 회복률
    clahe_count = r2_clahe['avg_count']
    spec_only_count = r2_spec['avg_count']
    if clahe_count > 0:
        improvement = (spec_only_count - clahe_count) / clahe_count * 100
        print(f"\nCLAHE 제거 효과:")
        print(f"   {clahe_count:.1f} → {spec_only_count:.1f} "
              f"({improvement:+.1f}%)")

    # 원본 대비 손실
    if baseline > 0:
        spec_only_retention = spec_only_count / baseline * 100
        clahe_retention = clahe_count / baseline * 100
        print(f"\n원본 성능 유지율:")
        print(f"   사양만 변환:    {spec_only_retention:.1f}%")
        print(f"   CLAHE 포함:    {clahe_retention:.1f}%")

    output_dir = PROJECT_ROOT + r'\paper\experiments\06_results'
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([r1_orig, r2_raw, r2_spec, r2_clahe],
                  f, indent=2, ensure_ascii=False)

    print(f"\n📁 결과 저장: {json_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()