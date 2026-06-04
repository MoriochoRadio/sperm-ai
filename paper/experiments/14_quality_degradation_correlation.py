"""
14_quality_degradation_correlation.py
─────────────────────────────────
더 강력한 Quality 검증:
같은 영상에 다양한 저하 → Quality vs Detection 상관관계

실험 4 데이터 + Quality 점수 측정
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from ultralytics import YOLO
from scipy import stats
import json


def degrade_video(input_path, output_path, mode):
    """5가지 품질 저하 적용"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 처음 200프레임만 처리 (빠르게)
    n_process = min(750, n_frames)  # 50fps × 15초 = 750 (Hard Rule 통과)
    
    for _ in range(n_process):
        ret, frame = cap.read()
        if not ret:
            break

        if mode == 'original':
            degraded = frame
        elif mode == 'low_resolution':
            small = cv2.resize(frame, (w//2, h//2))
            degraded = cv2.resize(small, (w, h))
        elif mode == 'motion_blur':
            kernel_size = 9
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1.0 / kernel_size
            degraded = cv2.filter2D(frame, -1, kernel)
        elif mode == 'noise':
            noise = np.random.normal(0, 30, frame.shape).astype(np.int16)
            degraded = np.clip(
                frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif mode == 'low_light_blur':
            dark = (frame.astype(np.float32) * 0.4).astype(np.uint8)
            degraded = cv2.GaussianBlur(dark, (9, 9), 2)
        else:
            degraded = frame

        out.write(degraded)

    cap.release()
    out.release()


def evaluate_detection(model, video_path, n_samples=20):
    """Detection 성능 측정"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // n_samples)
    counts = []
    confs = []
    idx = 0
    sampled = 0

    while sampled < n_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            results = model.predict(
                frame, conf=0.25, iou=0.6,
                verbose=False, device=0)
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    sperm_mask = boxes.cls == 0
                    counts.append(int(sperm_mask.sum()))
                    if sperm_mask.sum() > 0:
                        confs.extend(
                            boxes.conf[sperm_mask]
                            .cpu().numpy().tolist())
            sampled += 1
        idx += 1

    cap.release()
    return {
        'avg_count':      float(np.mean(counts)) if counts else 0,
        'avg_confidence': float(np.mean(confs)) if confs else 0,
    }


def main():
    print("=" * 80)
    print("  실험 14: 동일 영상 품질 저하 → Quality vs Detection")
    print("=" * 80)

    # 모델 로드
    print("\n📦 모델 로딩...")
    yolo = YOLO(PROJECT_ROOT + r'\models'
                 r'\yolo11_sperm_v2\weights\best.pt')
    
    from src.quality import VideoQualityAnalyzer
    qa = VideoQualityAnalyzer()
    print("   ✓ 로드 완료")

    # 원본 영상
    src_video = (PROJECT_ROOT + r'\data\raw'
                 r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                 r'\Train\11\11.mp4')

    # 작업 폴더
    work_dir = PROJECT_ROOT + r'\paper\experiments\14_results'
    os.makedirs(work_dir, exist_ok=True)

    # 5가지 모드
    modes = [
        ('original',       '원본'),
        ('low_resolution', '저해상도'),
        ('motion_blur',    '모션 블러'),
        ('noise',          '노이즈'),
        ('low_light_blur', '저조도+블러'),
    ]

    all_results = []

    for mode, name in modes:
        print(f"\n{'='*80}")
        print(f"  [{name}] 모드 처리 중...")
        print(f"{'='*80}")

        # 1. 영상 저하
        out_path = os.path.join(work_dir, f'video_{mode}.mp4')
        if not os.path.exists(out_path):
            print(f"   영상 생성 중...")
            degrade_video(src_video, out_path, mode)
        print(f"   ✓ 영상 준비 완료")

        # 2. Quality 점수 측정
        print(f"   Quality 평가 중...")
        try:
            quality = qa.analyze(out_path)
        except Exception as e:
            print(f"   ⚠️  Quality 평가 실패: {e}")
            continue
        
        print(f"      Q점수: {quality['total_score']:.1f}")
        print(f"      등급:  {quality['grade']}")

        # 3. Detection 평가
        print(f"   Detection 평가 중...")
        det = evaluate_detection(yolo, out_path)
        print(f"      평균 검출: {det['avg_count']:.1f}")
        print(f"      Confidence: {det['avg_confidence']:.4f}")

        all_results.append({
            'mode':           mode,
            'name':           name,
            'quality_score':  quality['total_score'],
            'grade':          quality['grade'],
            'dim_scores':     quality['scores'],
            'avg_count':      det['avg_count'],
            'avg_confidence': det['avg_confidence'],
        })

    # ── 상관관계 분석 ──────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  📊 결과 요약 (동일 영상, 다른 품질)")
    print("=" * 80)
    
    print(f"\n{'조건':<20} {'Q점수':<10} {'등급':<8} "
          f"{'검출수':<10} {'Conf':<10}")
    print("─" * 70)
    
    for r in all_results:
        print(f"{r['name']:<20} {r['quality_score']:<10.1f} "
              f"{r['grade']:<8} {r['avg_count']:<10.1f} "
              f"{r['avg_confidence']:<10.4f}")

    # 상관관계 (이번엔 의미있을 것)
    if len(all_results) >= 3:
        q_scores = [r['quality_score'] for r in all_results]
        counts = [r['avg_count'] for r in all_results]
        confs = [r['avg_confidence'] for r in all_results]

        r_q_count, p_q_count = stats.pearsonr(q_scores, counts)
        r_q_conf, p_q_conf = stats.pearsonr(q_scores, confs)
        
        print(f"\n[ 상관관계 (동일 영상 기준) ]")
        print(f"   Quality vs 검출수:    r={r_q_count:.4f}, "
              f"p={p_q_count:.4f}")
        print(f"   Quality vs Confidence: r={r_q_conf:.4f}, "
              f"p={p_q_conf:.4f}")
        
        sig = "✅ 유의미" if p_q_count < 0.05 else "⚠️  샘플 부족"
        print(f"\n   {sig}")

    # 저장
    json_path = os.path.join(work_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 결과 저장: {json_path}")

    print("\n" + "=" * 80)
    print("  ✅ 검증 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()