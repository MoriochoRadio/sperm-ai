"""
07_normalizer_validation.py
─────────────────────────────────
수정된 normalizer.py 검증
- CLAHE 제거 후 실제 시스템 성능 확인
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from ultralytics import YOLO
from src.normalizer import VideoNormalizer
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
        'avg_conf':  float(np.mean(confs)) if confs else 0,
    }


def main():
    print("=" * 80)
    print("  실험 7: 수정된 normalizer.py 검증")
    print("=" * 80)

    model = YOLO(PROJECT_ROOT + r'\models'
                 r'\yolo11_sperm_v2\weights\best.pt')
    normalizer = VideoNormalizer()

    # VISEM 원본 14번 영상 (321초, 정규화 필요)
    video_path = (PROJECT_ROOT + r'\tmp_uploads')
    
    # 또는 직접 합성: HD 영상으로 시뮬레이션
    test_video = (PROJECT_ROOT + r'\data\raw'
                  r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                  r'\Train\11\11.mp4')
    
    print(f"\n📹 테스트: 참가자 11번 영상\n")

    # 원본에서 50프레임
    cap = cv2.VideoCapture(test_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // 50)
    frames_orig = []
    idx = 0
    while len(frames_orig) < 50:
        ret, f = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames_orig.append(f.copy())
        idx += 1
    cap.release()

    # 1. 원본 평가
    print("[ 1. 원본 (변환 없음) ]")
    r1 = evaluate(model, frames_orig, '원본')
    print(f"   검출: {r1['avg_count']:.1f}, conf: {r1['avg_conf']:.4f}")

    # 2. 수정된 normalizer로 fast_trim 실행
    print("\n[ 2. 수정된 fast_trim 적용 ]")
    output_path = PROJECT_ROOT + r'\paper\experiments\07_results\normalized.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    norm_result = normalizer.fast_trim(test_video, output_path)
    print(f"   변환사항: {norm_result['changes']}")

    # 정규화된 영상에서 50프레임
    cap = cv2.VideoCapture(output_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // 50)
    frames_norm = []
    idx = 0
    while len(frames_norm) < 50:
        ret, f = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames_norm.append(f.copy())
        idx += 1
    cap.release()

    r2 = evaluate(model, frames_norm, '수정된 정규화')
    print(f"   검출: {r2['avg_count']:.1f}, conf: {r2['avg_conf']:.4f}")

    # 비교
    print("\n" + "=" * 80)
    print("  📊 결과 비교")
    print("=" * 80)
    print(f"{'처리':<30} {'검출 수':<14} {'Confidence':<14}")
    print("─" * 80)
    print(f"{'원본':<30} {r1['avg_count']:<14.1f} {r1['avg_conf']:<14.4f}")
    print(f"{'수정된 normalizer':<30} {r2['avg_count']:<14.1f} {r2['avg_conf']:<14.4f}")

    delta = r2['avg_count'] - r1['avg_count']
    if r1['avg_count'] > 0:
        retention = r2['avg_count'] / r1['avg_count'] * 100
        print(f"\n원본 성능 유지율: {retention:.1f}%")
        print(f"검출 수 변화: {delta:+.1f}")

    print("\n" + "=" * 80)
    print("  ✅ 검증 완료")
    print("=" * 80)
    print("""
🔍 해석:
  - 95% 이상: 수정 성공, CLAHE 제거가 효과적
  - 95% 미만: 추가 검토 필요
""")


if __name__ == '__main__':
    main()