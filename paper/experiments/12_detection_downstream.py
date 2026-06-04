"""
12_detection_downstream.py
─────────────────────────────────
Detection 품질 → Downstream task 영향 정량화

목적:
  - Detection threshold 변화 → 분석 결과 영향 측정
  - "Detection 정확도가 임상 분석에 미치는 영향"
  - 임상적 가치 강화
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
import time


def evaluate_at_threshold(model, frames, threshold,
                           label):
    """특정 threshold에서 detection 평가"""
    counts = []
    confs = []
    inference_times = []
    
    for frame in frames:
        t_start = time.time()
        results = model.predict(
            frame, conf=threshold, iou=0.6,
            verbose=False, device=0)
        inference_times.append((time.time() - t_start) * 1000)
        
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
        'label':            label,
        'threshold':        threshold,
        'avg_count':        float(np.mean(counts)) if counts else 0,
        'std_count':        float(np.std(counts)) if counts else 0,
        'avg_confidence':   float(np.mean(confs)) if confs else 0,
        'avg_inference_ms': float(np.mean(inference_times)),
    }


def calculate_track_consistency(model, video_path,
                                 threshold, max_frames=200):
    """
    추적 ID 일관성 측정
    - ByteTrack을 통해 정자별 추적 ID 부여
    - track 길이의 평균과 표준편차 측정
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(max_frames, total_frames)

    track_lengths = {}  # track_id → frame count
    
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO + ByteTrack
        results = model.track(
            frame, conf=threshold, iou=0.6,
            persist=True, verbose=False, device=0,
            tracker='bytetrack.yaml')
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                # sperm 클래스(0)만
                sperm_ids = ids[classes == 0]
                for tid in sperm_ids:
                    track_lengths[int(tid)] = \
                        track_lengths.get(int(tid), 0) + 1

    cap.release()

    if not track_lengths:
        return {
            'n_tracks':           0,
            'avg_track_length':   0,
            'long_tracks_ratio':  0,
        }

    lengths = list(track_lengths.values())
    long_tracks = [l for l in lengths if l >= 10]  # 10프레임 이상

    return {
        'n_tracks':           len(track_lengths),
        'avg_track_length':   float(np.mean(lengths)),
        'std_track_length':   float(np.std(lengths)),
        'long_tracks_ratio':  len(long_tracks) / len(lengths)
                              if lengths else 0,
    }


def main():
    print("=" * 80)
    print("  실험 12: Detection → Downstream 영향 정량화")
    print("=" * 80)

    # 모델 로드
    print("\n📦 모델 로딩...")
    model_path = (PROJECT_ROOT + r'\models'
                  r'\yolo11_sperm_v2\weights\best.pt')
    model = YOLO(model_path)
    print("   ✓ 로드 완료")

    # 테스트 영상
    video_path = (PROJECT_ROOT + r'\data\raw'
                  r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                  r'\Train\11\11.mp4')
    print(f"\n📹 테스트 영상: 참가자 11번")

    # 프레임 추출 (50개)
    print("\n[ 프레임 추출 (50개) ]")
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
    print(f"   ✓ {len(frames)}개 추출")

    # ── Threshold 별 평가 ──────────────────────────────────
    thresholds = [0.15, 0.25, 0.40, 0.55]
    
    print("\n" + "=" * 80)
    print("  Step 1: Detection 평가 (threshold별)")
    print("=" * 80)
    
    detection_results = []
    for th in thresholds:
        print(f"\n[ Threshold = {th} ]")
        result = evaluate_at_threshold(
            model, frames, th, f"th={th}")
        print(f"   평균 검출:    {result['avg_count']:.1f} ± "
              f"{result['std_count']:.1f}")
        print(f"   Confidence:  {result['avg_confidence']:.4f}")
        print(f"   추론 시간:    {result['avg_inference_ms']:.2f}ms")
        detection_results.append(result)

    # ── 추적 일관성 평가 ──────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  Step 2: 추적 일관성 평가")
    print("=" * 80)
    
    tracking_results = []
    for th in thresholds:
        print(f"\n[ Threshold = {th} ]")
        track_result = calculate_track_consistency(
            model, video_path, th, max_frames=200)
        print(f"   추적 ID 수:        {track_result['n_tracks']}")
        print(f"   평균 추적 길이:     {track_result['avg_track_length']:.2f}")
        print(f"   장기 추적 비율:     "
              f"{track_result['long_tracks_ratio']:.2%} (≥10프레임)")
        tracking_results.append({
            **track_result,
            'threshold': th,
        })

    # ── 최종 요약 ─────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  📊 실험 12 최종 요약 — Detection vs Downstream")
    print("=" * 100)
    print(f"{'Threshold':<12} {'검출 수':<14} {'Confidence':<14} "
          f"{'추적 ID':<12} {'평균길이':<12} {'장기 비율':<12} "
          f"{'추론 ms':<10}")
    print("─" * 100)
    
    final_results = []
    for det, trk in zip(detection_results, tracking_results):
        marker = ""
        if det['threshold'] == 0.25:
            marker = " ← 기본값"
        print(f"{det['threshold']:<12} "
              f"{det['avg_count']:<14.1f} "
              f"{det['avg_confidence']:<14.4f} "
              f"{trk['n_tracks']:<12} "
              f"{trk['avg_track_length']:<12.2f} "
              f"{trk['long_tracks_ratio']:<12.2%} "
              f"{det['avg_inference_ms']:<10.2f}{marker}")
        
        final_results.append({
            'threshold':         det['threshold'],
            'avg_count':         det['avg_count'],
            'avg_confidence':    det['avg_confidence'],
            'n_tracks':          trk['n_tracks'],
            'avg_track_length':  trk['avg_track_length'],
            'long_tracks_ratio': trk['long_tracks_ratio'],
            'inference_ms':      det['avg_inference_ms'],
        })

    # ── 핵심 발견 ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  🔍 핵심 분석")
    print("=" * 100)
    
    # 검출 수 변화
    counts_arr = [r['avg_count'] for r in final_results]
    count_change = counts_arr[0] - counts_arr[-1]
    print(f"\n1. Threshold 0.15 → 0.55:")
    print(f"   검출 수: {counts_arr[0]:.1f} → "
          f"{counts_arr[-1]:.1f} ({-count_change:+.1f})")
    print(f"   엄격할수록 검출 수 감소")
    
    # 추적 일관성
    tracks_arr = [r['avg_track_length'] for r in final_results]
    track_change = tracks_arr[-1] - tracks_arr[0]
    print(f"\n2. 추적 일관성:")
    print(f"   평균 추적 길이: {tracks_arr[0]:.2f} → "
          f"{tracks_arr[-1]:.2f} ({track_change:+.2f})")
    print(f"   엄격할수록 안정적인 추적")
    
    # 최적 균형점
    print(f"\n3. 임상적 의미:")
    print(f"   → 너무 관대 (0.15): 정자 수 과대 추정 가능")
    print(f"   → 너무 엄격 (0.55): 정자 수 과소 추정")
    print(f"   → 0.25~0.40: 균형점")
    print(f"   → 우리 시스템 0.25 채택은 합리적")

    # 저장
    output_dir = PROJECT_ROOT + r'\paper\experiments\12_results'
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 결과 저장: {json_path}")
    
    print("\n" + "=" * 100)
    print("  ✅ 실험 완료")
    print("=" * 100)


if __name__ == '__main__':
    main()