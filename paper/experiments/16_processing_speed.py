"""
16_processing_speed.py
─────────────────────────────────
처리 속도 분석 — 실시간 적용 가능성 평가

목적:
  - 영상 1개 처리 시간 측정
  - 단계별 시간 분석
  - FPS 측정
  - 임상 적용 가능성 입증
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import time
import cv2
import numpy as np
from ultralytics import YOLO
import json


def measure_step(name, func, *args, **kwargs):
    """단계별 시간 측정"""
    t_start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t_start
    return result, elapsed


def main():
    print("=" * 80)
    print("  실험 16: 처리 속도 분석 — 실시간 가능성")
    print("=" * 80)

    # ── 모델 로드 시간 ────────────────────────────────────
    print("\n[ 모델 로딩 시간 ]")
    
    t0 = time.time()
    yolo = YOLO(PROJECT_ROOT + r'\models'
                 r'\yolo11_sperm_v2\weights\best.pt')
    yolo_load_time = time.time() - t0
    print(f"   YOLO11m:           {yolo_load_time:.2f}s")
    
    t0 = time.time()
    from src.morphology import MorphologyAnalyzer
    morph = MorphologyAnalyzer(
        model_path=PROJECT_ROOT + r'\models'
                    r'\morphology_efficientnet_b3_v3.pt')
    morph_load_time = time.time() - t0
    print(f"   Morphology v3:     {morph_load_time:.2f}s")
    
    t0 = time.time()
    from src.quality import VideoQualityAnalyzer
    qa = VideoQualityAnalyzer()
    qa_load_time = time.time() - t0
    print(f"   Quality Analyzer:  {qa_load_time:.2f}s")
    
    total_load = yolo_load_time + morph_load_time + qa_load_time
    print(f"   ─────────────────────────")
    print(f"   총 로딩 시간:        {total_load:.2f}s")

    # ── 테스트 영상 ───────────────────────────────────────
    video_path = (PROJECT_ROOT + r'\data\raw'
                  r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                  r'\Train\11\11.mp4')
    
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = n_frames / fps
    cap.release()
    
    print(f"\n[ 테스트 영상: 참가자 11번 ]")
    print(f"   프레임 수:  {n_frames}")
    print(f"   FPS:       {fps:.1f}")
    print(f"   길이:       {duration:.1f}초")

    # ── Step 1: Quality Assessment ─────────────────────────
    print(f"\n{'='*80}")
    print(f"  단계별 처리 시간 측정")
    print(f"{'='*80}")
    
    print(f"\n[ Step 1: Quality Assessment ]")
    quality_result, t_quality = measure_step(
        'quality', qa.analyze, video_path)
    print(f"   처리 시간: {t_quality:.2f}s")
    print(f"   결과:     Q={quality_result['total_score']:.1f} "
          f"({quality_result['grade']})")

    # ── Step 2: Detection (전체 영상) ──────────────────────
    print(f"\n[ Step 2: Detection (전체 영상) ]")
    
    cap = cv2.VideoCapture(video_path)
    detections_per_frame = []
    
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo.predict(
            frame, conf=0.25, iou=0.6,
            verbose=False, device=0)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            n_sperm = int((boxes.cls == 0).sum())
            detections_per_frame.append(n_sperm)
    cap.release()
    
    t_detection = time.time() - t_start
    detection_fps = n_frames / t_detection
    
    print(f"   처리 시간:     {t_detection:.2f}s")
    print(f"   처리 FPS:      {detection_fps:.1f}")
    print(f"   평균 검출:     {np.mean(detections_per_frame):.1f}")
    print(f"   {'✅ 실시간 가능 (>30 FPS)' if detection_fps >= 30 else '⚠️  실시간 미달'}")

    # ── Step 3: Detection + Tracking ───────────────────────
    print(f"\n[ Step 3: Detection + Tracking ]")
    
    cap = cv2.VideoCapture(video_path)
    track_ids = set()
    
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo.track(
            frame, conf=0.25, iou=0.6,
            persist=True, verbose=False, device=0,
            tracker='bytetrack.yaml')
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.int().cpu().numpy()
                track_ids.update(ids.tolist())
    cap.release()
    
    t_tracking = time.time() - t_start
    tracking_fps = n_frames / t_tracking
    
    print(f"   처리 시간:     {t_tracking:.2f}s")
    print(f"   처리 FPS:      {tracking_fps:.1f}")
    print(f"   고유 추적 ID:  {len(track_ids)}")

    # ── Step 4: Morphology (샘플 100개) ────────────────────
    print(f"\n[ Step 4: Morphology Analysis (100 crops) ]")
    
    # 100개 더미 크롭 생성 (실제 영상에서)
    cap = cv2.VideoCapture(video_path)
    crops = []
    while len(crops) < 100:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo.predict(
            frame, conf=0.25, iou=0.6,
            verbose=False, device=0)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy[
                results[0].boxes.cls == 0].cpu().numpy()
            for box in boxes:
                if len(crops) >= 100:
                    break
                x1, y1, x2, y2 = box.astype(int)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 10:
                    crops.append(crop)
    cap.release()
    
    print(f"   크롭 추출:     {len(crops)}개")
    
    t_start = time.time()
    for crop in crops:
        try:
            morph.classify_single(crop)
        except Exception:
            pass
    t_morph = time.time() - t_start
    
    print(f"   처리 시간:     {t_morph:.2f}s")
    print(f"   크롭당 시간:   {(t_morph/len(crops))*1000:.2f}ms")

    # ── 종합 분석 ─────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  📊 처리 속도 종합")
    print(f"{'='*80}")
    
    # 전체 영상 처리 시간 추정
    estimated_total = t_quality + t_tracking + t_morph * 5  # 형태 분석 비율 추정
    
    print(f"\n{'단계':<25} {'시간(s)':<12} {'비율':<10}")
    print("─" * 55)
    print(f"{'Quality Assessment':<25} {t_quality:<12.2f} "
          f"{t_quality/estimated_total*100:<10.1f}%")
    print(f"{'Detection + Tracking':<25} {t_tracking:<12.2f} "
          f"{t_tracking/estimated_total*100:<10.1f}%")
    print(f"{'Morphology (per 100)':<25} {t_morph:<12.2f} "
          f"{t_morph/estimated_total*100:<10.1f}%")
    print("─" * 55)
    print(f"{'총 처리 시간':<25} {estimated_total:<12.2f} "
          f"100.0%")
    
    print(f"\n📊 핵심 지표:")
    print(f"   영상 길이:           {duration:.1f}초")
    print(f"   처리 시간:           {estimated_total:.1f}초")
    print(f"   배속:               {duration/estimated_total:.2f}x")
    
    if estimated_total < duration:
        print(f"   ✅ 실시간 처리 가능 (배속 ≥ 1.0x)")
    elif estimated_total < duration * 2:
        print(f"   ✅ 거의 실시간 (배속 ≥ 0.5x)")
    else:
        print(f"   ⚠️  실시간 미달 (오프라인 처리 권장)")
    
    print(f"\n💡 임상 적용 평가:")
    if detection_fps >= 30:
        print(f"   ✅ Detection만 30 FPS 이상 → 실시간 모니터링 가능")
    if estimated_total < duration * 1.5:
        print(f"   ✅ 전체 분석도 1.5x 이내 → 임상 워크플로우 통합 가능")

    # 저장
    output_dir = PROJECT_ROOT + r'\paper\experiments\16_results'
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'video_info': {
            'n_frames': n_frames,
            'fps':      fps,
            'duration': duration,
        },
        'loading_times': {
            'yolo':       yolo_load_time,
            'morphology': morph_load_time,
            'quality':    qa_load_time,
            'total':      total_load,
        },
        'processing_times': {
            'quality':            t_quality,
            'detection':          t_detection,
            'detection_fps':      detection_fps,
            'tracking':           t_tracking,
            'tracking_fps':       tracking_fps,
            'morphology_per_100': t_morph,
        },
        'speed_ratio':       float(duration / estimated_total),
        'realtime_capable':  bool(estimated_total < duration),
    }
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 결과 저장: {json_path}")
    print(f"\n{'='*80}")
    print(f"  ✅ 처리 속도 분석 완료")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()