"""
04_quality_degradation_test.py
─────────────────────────────────
영상 품질 저하 시뮬레이션 + 정규화 효과 측정

목적: 
  - 정규화가 어떤 조건에서 효과적인지 정량화
  - 우리 스마트 분기 로직 정당화
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


# ── 영상 품질 저하 함수들 ─────────────────────────────────

def degrade_resolution(frame: np.ndarray,
                        scale_factor: float = 0.5) -> np.ndarray:
    """저해상도 시뮬레이션: 다운샘플 후 업샘플"""
    h, w = frame.shape[:2]
    small_h, small_w = int(h * scale_factor), int(w * scale_factor)
    # 다운샘플 (정보 손실)
    small = cv2.resize(frame, (small_w, small_h),
                       interpolation=cv2.INTER_AREA)
    # 업샘플 (블러 발생)
    return cv2.resize(small, (w, h),
                      interpolation=cv2.INTER_LINEAR)


def degrade_motion_blur(frame: np.ndarray,
                         kernel_size: int = 7) -> np.ndarray:
    """모션 블러 시뮬레이션 (가로 방향)"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(frame, -1, kernel)


def degrade_noise(frame: np.ndarray,
                   sigma: float = 25) -> np.ndarray:
    """가우시안 노이즈 추가"""
    noise = np.random.normal(0, sigma, frame.shape).astype(np.int16)
    noisy = frame.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def degrade_low_light(frame: np.ndarray,
                       brightness_factor: float = 0.5,
                       blur_kernel: int = 3) -> np.ndarray:
    """저조도 + 흐림 (자가키트 환경)"""
    # 밝기 감소
    darkened = (frame.astype(np.float32) * brightness_factor).astype(np.uint8)
    # 약한 블러
    blurred = cv2.GaussianBlur(darkened, (blur_kernel, blur_kernel), 0)
    return blurred


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """CLAHE 정규화 (Phase 6과 동일)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ── Detection 평가 함수 ──────────────────────────────────

def evaluate_detection(model, frames: list,
                        condition_name: str) -> dict:
    """주어진 프레임들에서 detection 평가"""
    detection_counts = []
    confidences = []
    inference_times = []

    for frame in frames:
        t_start = time.time()
        results = model.predict(
            frame, conf=0.25, iou=0.6,
            verbose=False, device=0)
        inference_times.append((time.time() - t_start) * 1000)

        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                sperm_mask = boxes.cls == 0
                detection_counts.append(int(sperm_mask.sum()))
                if sperm_mask.sum() > 0:
                    sperm_confs = boxes.conf[sperm_mask]
                    confidences.extend(
                        sperm_confs.cpu().numpy().tolist())
            else:
                detection_counts.append(0)

    return {
        'condition':       condition_name,
        'avg_detections':  float(np.mean(detection_counts))
                           if detection_counts else 0,
        'std_detections':  float(np.std(detection_counts))
                           if detection_counts else 0,
        'avg_confidence':  float(np.mean(confidences))
                           if confidences else 0,
        'avg_inference_ms': float(np.mean(inference_times)),
        'n_frames':        len(frames),
    }


# ── 메인 실험 ─────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  실험 4: 영상 품질 저하 시뮬레이션 + 정규화 효과")
    print("=" * 80)

    # 모델 로드
    print("\n📦 모델 로딩...")
    model_path = (PROJECT_ROOT + r'\models'
                  r'\yolo11_sperm_v2\weights\best.pt')
    model = YOLO(model_path)
    print(f"   ✓ 로드 완료")

    # 테스트 영상: 참가자 11번
    video_path = (PROJECT_ROOT + r'\data\raw'
                  r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                  r'\Train\11\11.mp4')

    print(f"\n📹 테스트 영상: 참가자 11번 (S등급 표준 영상)")
    print(f"   {video_path}")

    # 프레임 추출 (50개 균등 샘플링)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = 50
    interval = max(1, total // sample_count)

    frames_original = []
    frame_idx = 0
    while len(frames_original) < sample_count:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frames_original.append(frame.copy())
        frame_idx += 1
    cap.release()

    print(f"   ✓ 프레임 추출 완료: {len(frames_original)}개")

    # ── 5가지 조건 시뮬레이션 ──────────────────────────────
    conditions = {
        '원본 (S등급)':       lambda f: f,
        '저해상도':          lambda f: degrade_resolution(f, 0.5),
        '모션 블러':         lambda f: degrade_motion_blur(f, 7),
        '노이즈':           lambda f: degrade_noise(f, 25),
        '저조도+흐림':       lambda f: degrade_low_light(f, 0.5, 3),
    }

    all_results = []

    for cond_name, degrade_fn in conditions.items():
        print(f"\n{'=' * 80}")
        print(f"  조건: {cond_name}")
        print(f"{'=' * 80}")

        # 1. 품질 저하 적용
        frames_degraded = [degrade_fn(f.copy())
                           for f in frames_original]

        # 2. 정규화 X 평가
        print(f"\n[ 정규화 적용 X ]")
        result_no_norm = evaluate_detection(
            model, frames_degraded,
            f"{cond_name} (No norm)")
        print(f"   평균 검출:   {result_no_norm['avg_detections']:.1f} "
              f"± {result_no_norm['std_detections']:.1f}")
        print(f"   Confidence: {result_no_norm['avg_confidence']:.4f}")

        # 3. 정규화 O 평가 (CLAHE)
        frames_normalized = [apply_clahe(f) for f in frames_degraded]
        print(f"\n[ 정규화 O (CLAHE) ]")
        result_with_norm = evaluate_detection(
            model, frames_normalized,
            f"{cond_name} (With norm)")
        print(f"   평균 검출:   {result_with_norm['avg_detections']:.1f} "
              f"± {result_with_norm['std_detections']:.1f}")
        print(f"   Confidence: {result_with_norm['avg_confidence']:.4f}")

        # 4. 정규화 효과
        delta_count = (result_with_norm['avg_detections']
                       - result_no_norm['avg_detections'])
        delta_conf = (result_with_norm['avg_confidence']
                      - result_no_norm['avg_confidence'])

        if result_no_norm['avg_detections'] > 0:
            recovery_pct = (delta_count
                            / result_no_norm['avg_detections']
                            * 100)
        else:
            recovery_pct = 0

        print(f"\n[ 정규화 효과 ]")
        print(f"   검출 수 변화:    {delta_count:+.1f} ({recovery_pct:+.1f}%)")
        print(f"   Confidence 변화: {delta_conf:+.4f}")

        all_results.append({
            'condition':           cond_name,
            'no_norm_detections':  result_no_norm['avg_detections'],
            'with_norm_detections': result_with_norm['avg_detections'],
            'no_norm_confidence':  result_no_norm['avg_confidence'],
            'with_norm_confidence': result_with_norm['avg_confidence'],
            'delta_detections':    delta_count,
            'delta_confidence':    delta_conf,
            'recovery_percent':    recovery_pct,
        })

    # ── 최종 요약 표 ──────────────────────────────────────
    print("\n\n" + "=" * 95)
    print("  📊 실험 4 최종 요약 — 영상 품질 저하 vs 정규화 효과")
    print("=" * 95)
    print(f"{'조건':<18} "
          f"{'No Norm 검출':<14} {'CLAHE 검출':<14} "
          f"{'Δ 검출':<10} {'회복률':<10}")
    print("─" * 95)
    for r in all_results:
        print(f"{r['condition']:<18} "
              f"{r['no_norm_detections']:<14.1f} "
              f"{r['with_norm_detections']:<14.1f} "
              f"{r['delta_detections']:<+10.1f} "
              f"{r['recovery_percent']:<+10.1f}%")

    # JSON 저장
    output_dir = PROJECT_ROOT + r'\paper\experiments\04_results'
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 결과 저장: {json_path}")

    # 핵심 분석 출력
    print("\n" + "=" * 95)
    print("  🔍 핵심 분석")
    print("=" * 95)
    print("""
1. 원본(S등급)에 정규화 적용: 효과 측정
2. 저품질 시뮬레이션에서: 정규화의 실질 가치
3. 회복률 분석:
   - 양수 = 정규화로 회복됨 (정규화 효과적)
   - 음수 = 정규화로 악화됨 (정규화 불필요)
""")
    print("=" * 95)
    print("  ✅ 실험 완료")
    print("=" * 95)


if __name__ == '__main__':
    main()