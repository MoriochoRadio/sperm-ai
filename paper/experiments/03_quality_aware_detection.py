"""
03_quality_aware_detection.py
─────────────────────────────────
영상 품질 등급별 Detection 성능 평가

목적:
  영상 품질이 detection 성능에 미치는 영향 측정
  정규화 전/후 비교로 Phase 6 정규화 효과 정량화

대상 영상:
  VISEM-Tracking 검증셋 4명 (11, 14, 22, 23)
  + 정규화된 버전 (CLAHE 적용)
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from ultralytics import YOLO
from src.quality import VideoQualityAnalyzer
from src.normalizer import VideoNormalizer


def analyze_video_detection(model, video_path: str,
                             label: str, max_frames: int = 100):
    """
    영상에서 일정 프레임 추출 → detection 실행
    → 평균 검출 수, 평균 confidence, FPS 측정
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // max_frames)

    detection_counts = []
    confidences = []
    inference_times = []

    frame_idx = 0
    sampled = 0

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Detection 실행
            import time
            t_start = time.time()
            results = model.predict(
                frame, conf=0.25, iou=0.6,
                verbose=False, device=0)
            t_end = time.time()

            inference_times.append((t_end - t_start) * 1000)  # ms

            # sperm 클래스(0)만 카운트
            if len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    sperm_mask = boxes.cls == 0
                    sperm_confs = boxes.conf[sperm_mask]
                    detection_counts.append(int(sperm_mask.sum()))
                    if len(sperm_confs) > 0:
                        confidences.extend(
                            sperm_confs.cpu().numpy().tolist())
                else:
                    detection_counts.append(0)

            sampled += 1
        frame_idx += 1

    cap.release()

    return {
        'label':           label,
        'sampled_frames':  sampled,
        'avg_detections':  float(np.mean(detection_counts))
                           if detection_counts else 0,
        'std_detections':  float(np.std(detection_counts))
                           if detection_counts else 0,
        'avg_confidence':  float(np.mean(confidences))
                           if confidences else 0,
        'avg_inference_ms': float(np.mean(inference_times)),
    }


def main():
    print("=" * 70)
    print("  실험 3-A: 영상 품질별 Detection 성능 평가")
    print("=" * 70)

    # 모델 로드
    print("\n📦 YOLO11m v2 모델 로딩...")
    model_path = (PROJECT_ROOT + r'\models'
                  r'\yolo11_sperm_v2\weights\best.pt')
    model = YOLO(model_path)
    print(f"   ✓ 모델 로드 완료")

    # 품질 분석기 + 정규화기
    quality_analyzer = VideoQualityAnalyzer()
    normalizer = VideoNormalizer()

    # 테스트 영상 (VISEM-Tracking 검증셋 4명)
    base = (PROJECT_ROOT + r'\data\raw'
            r'\VISEM-Tracking\VISEM_Tracking_Train_v4\Train')
    test_videos = [
        (f'{base}\\11\\11.mp4', '참가자 11'),
        (f'{base}\\14\\14.mp4', '참가자 14'),
        (f'{base}\\22\\22.mp4', '참가자 22'),
        (f'{base}\\23\\23.mp4', '참가자 23'),
    ]

    output_dir = PROJECT_ROOT + r'\paper\experiments\03_results'
    os.makedirs(output_dir, exist_ok=True)

    results_summary = []

    for video_path, label in test_videos:
        if not os.path.exists(video_path):
            print(f"\n⚠️  파일 없음: {video_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"  📹 {label}")
        print(f"{'=' * 70}")

        # 1. 품질 평가
        quality = quality_analyzer.analyze(video_path)
        print(f"\n[ 품질 평가 ]")
        print(f"   등급: {quality['grade']} ({quality['total_score']}점)")
        print(f"   세부: {quality['scores']}")

        # 2. 원본 영상 detection
        print(f"\n[ 원본 영상 Detection ]")
        orig_result = analyze_video_detection(
            model, video_path, f"{label} (원본)")
        print(f"   평균 검출 정자 수: {orig_result['avg_detections']:.1f}")
        print(f"   평균 confidence:  {orig_result['avg_confidence']:.4f}")
        print(f"   평균 추론 시간:    {orig_result['avg_inference_ms']:.2f}ms")

        # 3. 정규화 후 영상 detection
        normalized_path = os.path.join(
            output_dir,
            f"{os.path.basename(video_path).replace('.mp4', '_norm.mp4')}")

        print(f"\n[ 정규화 진행 (CLAHE 적용) ]")
        norm_type = VideoNormalizer.get_normalization_type(
            quality['metadata'])
        print(f"   정규화 유형: {norm_type}")

        if norm_type == 'none':
            # 정규화 안 해도 됨 → 직접 CLAHE만 적용한 버전 만들기
            print(f"   원본이 표준 사양 → CLAHE만 적용한 비교 버전 생성")
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                normalized_path, fourcc, fps, (w, h), True)
            clahe = cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(8, 8))

            for _ in range(n):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray)
                frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                out.write(frame_out)
            cap.release()
            out.release()
        else:
            # 풀 정규화 또는 빠른 컷팅
            if norm_type == 'trim':
                normalizer.fast_trim(video_path, normalized_path)
            else:
                normalizer.normalize(video_path, normalized_path)

        print(f"   ✓ 정규화 완료: {normalized_path}")

        # 4. 정규화 영상 detection
        print(f"\n[ 정규화 영상 Detection ]")
        norm_result = analyze_video_detection(
            model, normalized_path, f"{label} (정규화)")
        print(f"   평균 검출 정자 수: {norm_result['avg_detections']:.1f}")
        print(f"   평균 confidence:  {norm_result['avg_confidence']:.4f}")
        print(f"   평균 추론 시간:    {norm_result['avg_inference_ms']:.2f}ms")

        # 5. 변화량 계산
        delta_count = (norm_result['avg_detections']
                       - orig_result['avg_detections'])
        delta_conf = (norm_result['avg_confidence']
                      - orig_result['avg_confidence'])

        print(f"\n[ 변화 (정규화 효과) ]")
        print(f"   검출 정자 수 변화:  {delta_count:+.1f}")
        print(f"   Confidence 변화:    {delta_conf:+.4f}")

        results_summary.append({
            'label':              label,
            'grade':              quality['grade'],
            'quality_score':      quality['total_score'],
            'sharpness_score':    quality['scores']['sharpness'],
            'orig_detections':    orig_result['avg_detections'],
            'orig_confidence':    orig_result['avg_confidence'],
            'norm_detections':    norm_result['avg_detections'],
            'norm_confidence':    norm_result['avg_confidence'],
            'delta_count':        delta_count,
            'delta_confidence':   delta_conf,
        })

    # ── 최종 요약 표 ──────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("  📊 실험 3-A 최종 요약")
    print("=" * 90)
    print(f"{'영상':<14} {'등급':<5} {'점수':<5} "
          f"{'원본 검출':<10} {'정규화 검출':<12} "
          f"{'원본 conf':<10} {'정규화 conf':<12} {'Δ검출':<7}")
    print("─" * 90)
    for r in results_summary:
        print(f"{r['label']:<14} {r['grade']:<5} {r['quality_score']:<5} "
              f"{r['orig_detections']:<10.1f} {r['norm_detections']:<12.1f} "
              f"{r['orig_confidence']:<10.4f} {r['norm_confidence']:<12.4f} "
              f"{r['delta_count']:<+7.1f}")

    # CSV 저장
    import json
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n📁 결과 저장: {json_path}")
    print("\n" + "=" * 90)
    print("  ✅ 실험 완료")
    print("=" * 90)


if __name__ == '__main__':
    main()