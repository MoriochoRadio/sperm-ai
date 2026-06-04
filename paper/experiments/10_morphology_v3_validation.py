"""
10_morphology_v3_validation.py
─────────────────────────────────
v3 형태 분석 모델의 VISEM 적용 결과 정밀 검증

목적:
  - 4명(11/14/22/23) 영상에서 v3 적용
  - 부위별 정상/비정상 비율 측정
  - 운동성 결과와 일관성 확인
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json


def main():
    print("=" * 80)
    print("  실험 10: v3 형태 분석 VISEM 검증")
    print("=" * 80)

    # ── 모델 로드 ─────────────────────────────────────────
    print("\n📦 모델 로딩...")
    
    # YOLO11 (정자 탐지)
    yolo_path = (PROJECT_ROOT + r'\models'
                 r'\yolo11_sperm_v2\weights\best.pt')
    yolo = YOLO(yolo_path)
    print("   ✓ YOLO11m v2 로드")

    # Morphology v3 분석기 로드
    from src.morphology import MorphologyAnalyzer
    
    morph_path = (PROJECT_ROOT + r'\models'
                  r'\morphology_efficientnet_b3_v3.pt')
    morph = MorphologyAnalyzer(model_path=morph_path)
    print(f"   ✓ Morphology v3 로드")

    # ── 4명 영상 ──────────────────────────────────────────
    base = (PROJECT_ROOT + r'\data\raw'
            r'\VISEM-Tracking\VISEM_Tracking_Train_v4\Train')
    
    test_videos = [
        ('11', f'{base}\\11\\11.mp4'),
        ('14', f'{base}\\14\\14.mp4'),
        ('22', f'{base}\\22\\22.mp4'),
        ('23', f'{base}\\23\\23.mp4'),
    ]

    all_results = []

    for participant, video_path in test_videos:
        if not os.path.exists(video_path):
            print(f"\n⚠️  파일 없음: {video_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"  📹 참가자 {participant}")
        print(f"{'=' * 80}")

        # ── 1. 정자 크롭 추출 (YOLO 사용) ─────────────────
        print(f"\n[ 1단계: 정자 크롭 추출 ]")
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 30 프레임 균등 샘플링
        sample_count = 30
        interval = max(1, total // sample_count)
        
        crops = []
        frame_idx = 0
        sampled = 0
        
        while sampled < sample_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % interval == 0:
                # YOLO 추론
                results = yolo.predict(
                    frame, conf=0.4, iou=0.6,
                    verbose=False, device=0)
                
                if len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        sperm_mask = boxes.cls == 0
                        sperm_boxes = boxes.xyxy[sperm_mask].cpu().numpy()
                        
                        h, w = frame.shape[:2]
                        for box in sperm_boxes:
                            x1, y1, x2, y2 = box.astype(int)
                            # 약간의 여유 추가
                            pad = 5
                            x1 = max(0, x1 - pad)
                            y1 = max(0, y1 - pad)
                            x2 = min(w, x2 + pad)
                            y2 = min(h, y2 + pad)
                            
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0 and crop.shape[0] > 10 \
                               and crop.shape[1] > 10:
                                crops.append(crop)
                
                sampled += 1
            frame_idx += 1
        
        cap.release()
        print(f"   ✓ 추출된 크롭: {len(crops)}개")

        if len(crops) == 0:
            print(f"   ⚠️  크롭 없음, 스킵")
            continue

        # ── 2. 형태 분석 (v3) ─────────────────────────────
        print(f"\n[ 2단계: v3 형태 분석 ]")
        
        # 부위별 결과 누적
        results_by_part = {
            'head':     {'normal': 0, 'abnormal': 0},
            'acrosome': {'normal': 0, 'abnormal': 0},
            'vacuole':  {'normal': 0, 'abnormal': 0},
            'tail':     {'normal': 0, 'abnormal': 0},
        }
        
        # 전체 정상 수 (모든 부위 정상인 정자)
        all_normal_count = 0
        valid_crops = 0
        
        per_crop_results = []
        
        for crop in crops:
            try:
                # MorphologyAnalyzer.analyze_crop은 dict 반환 예상
                # {'head': 0/1, 'acrosome': 0/1, ...}
                result = morph.classify_single(crop)
                
                if result is None:
                    continue
                
                valid_crops += 1
                per_crop_results.append(result)
                
                # 각 부위별 정상/비정상
                # 0=정상, 1=비정상
                all_normal = True
                for part in ['head', 'acrosome', 'vacuole', 'tail']:
                    if part in result:
                        is_normal = (result[part] == 0)
                        if is_normal:
                            results_by_part[part]['normal'] += 1
                        else:
                            results_by_part[part]['abnormal'] += 1
                            all_normal = False
                
                if all_normal:
                    all_normal_count += 1
                    
            except Exception as e:
                print(f"      크롭 분석 실패: {e}")
                continue
        
        print(f"   ✓ 분석된 크롭: {valid_crops}개")
        
        if valid_crops == 0:
            continue

        # ── 3. 통계 계산 ──────────────────────────────────
        print(f"\n[ 3단계: 결과 통계 ]")
        
        normal_morphology_rate = all_normal_count / valid_crops * 100
        
        print(f"   전체 정상 형태율: {normal_morphology_rate:.1f}%")
        print(f"   (모든 부위 정상 / 전체)")
        
        print(f"\n   부위별 정상률:")
        part_rates = {}
        for part, counts in results_by_part.items():
            total = counts['normal'] + counts['abnormal']
            if total > 0:
                rate = counts['normal'] / total * 100
                part_rates[part] = rate
                print(f"      {part:<10}: {rate:>5.1f}% "
                      f"(정상 {counts['normal']}/{total})")

        # WHO 기준 비교
        print(f"\n   WHO 기준 (정상 형태 ≥ 4%):")
        who_pass = "✅ 충족" if normal_morphology_rate >= 4 else "❌ 미달"
        print(f"      → {who_pass}")

        # ── 결과 저장 ─────────────────────────────────────
        all_results.append({
            'participant':     participant,
            'total_crops':     len(crops),
            'valid_crops':     valid_crops,
            'all_normal_count': all_normal_count,
            'normal_morphology_rate': normal_morphology_rate,
            'part_rates':      part_rates,
            'parts_detail':    results_by_part,
            'who_compliance':  normal_morphology_rate >= 4,
        })

    # ── 최종 요약 ─────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("  📊 실험 10 최종 요약 — v3 형태 분석 4명 비교")
    print("=" * 90)
    print(f"{'참가자':<8} {'크롭':<8} {'정상률':<10} "
          f"{'머리':<10} {'첨체':<10} {'공포':<10} {'꼬리':<10} {'WHO':<6}")
    print("─" * 90)
    
    for r in all_results:
        pr = r['part_rates']
        who_mark = "✅" if r['who_compliance'] else "❌"
        print(f"{r['participant']:<8} "
              f"{r['valid_crops']:<8} "
              f"{r['normal_morphology_rate']:<10.1f} "
              f"{pr.get('head', 0):<10.1f} "
              f"{pr.get('acrosome', 0):<10.1f} "
              f"{pr.get('vacuole', 0):<10.1f} "
              f"{pr.get('tail', 0):<10.1f} "
              f"{who_mark:<6}")

    # 평균
    if all_results:
        avg_rate = np.mean([r['normal_morphology_rate']
                            for r in all_results])
        print("─" * 90)
        print(f"{'평균':<8} {'':<8} {avg_rate:<10.1f}")

    # 저장
    output_dir = PROJECT_ROOT + r'\paper\experiments\10_results'
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 결과 저장: {json_path}")
    print("\n" + "=" * 90)
    print("  ✅ 검증 완료")
    print("=" * 90)


if __name__ == '__main__':
    main()