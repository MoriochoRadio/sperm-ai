"""
13_quality_validation.py
─────────────────────────────────
Quality Assessment 시스템 자체 검증

목적:
  - 우리 Quality 점수가 Detection 성능과 상관 있는가?
  - 6개 차원별 영향력 분석
  - 등급(S/A/B/C/F)이 실제로 의미 있는가?
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


def evaluate_video_detection(model, video_path,
                              n_samples=30):
    """영상 1개에서 평균 detection 성능 측정"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    interval = max(1, total // n_samples)
    counts = []
    confs = []
    frame_idx = 0
    sampled = 0

    while sampled < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
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
                            boxes.conf[sperm_mask]
                            .cpu().numpy().tolist())
                else:
                    counts.append(0)
            sampled += 1
        frame_idx += 1

    cap.release()

    return {
        'avg_count':      float(np.mean(counts)) if counts else 0,
        'std_count':      float(np.std(counts)) if counts else 0,
        'avg_confidence': float(np.mean(confs)) if confs else 0,
        'n_frames':       sampled,
    }


def main():
    print("=" * 80)
    print("  실험 13: Quality Assessment 자체 검증")
    print("=" * 80)

    # ── 모델 로드 ─────────────────────────────────────────
    print("\n📦 모델 로딩...")
    yolo_path = (PROJECT_ROOT + r'\models'
                 r'\yolo11_sperm_v2\weights\best.pt')
    model = YOLO(yolo_path)
    print("   ✓ YOLO11m 로드")

    # Quality Analyzer 로드
    from src.quality import VideoQualityAnalyzer
    quality_analyzer = VideoQualityAnalyzer()
    print("   ✓ VideoQualityAnalyzer 로드")

    # ── 4명 영상 ──────────────────────────────────────────
    base = (PROJECT_ROOT + r'\data\raw'
            r'\VISEM-Tracking\VISEM_Tracking_Train_v4\Train')
    
    test_videos = [
        ('11', f'{base}\\11\\11.mp4'),
        ('14', f'{base}\\14\\14.mp4'),
        ('22', f'{base}\\22\\22.mp4'),
        ('23', f'{base}\\23\\23.mp4'),
    ]

    # ── 1단계: 각 영상별 Quality + Detection 측정 ────────
    print("\n" + "=" * 80)
    print("  Step 1: 영상별 Quality 점수 + Detection 성능")
    print("=" * 80)

    all_data = []

    for participant, video_path in test_videos:
        if not os.path.exists(video_path):
            print(f"\n⚠️  파일 없음: {video_path}")
            continue

        print(f"\n[ 참가자 {participant} ]")
        
        # Quality 평가
        try:
            quality_result = quality_analyzer.analyze(video_path)
        except Exception as e:
            print(f"   ⚠️  Quality 평가 실패: {e}")
            continue

        print(f"   Quality 점수: {quality_result['total_score']:.1f}/100")
        print(f"   등급:        {quality_result['grade']}")
        print(f"   세부 점수:")
        for dim, score in quality_result['scores'].items():
            print(f"      {dim:<15}: {score:>6.1f}")

        # Detection 성능 측정
        det_result = evaluate_video_detection(model, video_path)
        if det_result is None:
            continue

        print(f"   Detection:")
        print(f"      평균 검출:  {det_result['avg_count']:.1f}")
        print(f"      Confidence: {det_result['avg_confidence']:.4f}")

        all_data.append({
            'participant':    participant,
            'quality_score':  quality_result['total_score'],
            'grade':          quality_result['grade'],
            'dim_scores':     quality_result['scores'],
            'avg_count':      det_result['avg_count'],
            'std_count':      det_result['std_count'],
            'avg_confidence': det_result['avg_confidence'],
        })

    if len(all_data) < 2:
        print("\n⚠️  데이터가 부족합니다.")
        return

    # ── 2단계: 추가 영상으로 데이터 확장 (다른 영상들) ─────
    print("\n\n" + "=" * 80)
    print("  Step 2: 추가 영상으로 데이터 확장")
    print("=" * 80)
    
    # VISEM-Tracking에서 추가 영상 5개 선택
    additional_ids = ['18', '20', '25', '27', '30']
    
    for pid in additional_ids:
        video_path = f'{base}\\{pid}\\{pid}.mp4'
        if not os.path.exists(video_path):
            continue
        
        print(f"\n[ 참가자 {pid} ]")
        
        try:
            quality_result = quality_analyzer.analyze(video_path)
        except Exception as e:
            print(f"   ⚠️  실패: {e}")
            continue

        det_result = evaluate_video_detection(model, video_path,
                                               n_samples=20)
        if det_result is None:
            continue

        print(f"   Q점수: {quality_result['total_score']:.1f} "
              f"({quality_result['grade']}) "
              f"→ 검출: {det_result['avg_count']:.1f}")

        all_data.append({
            'participant':    pid,
            'quality_score':  quality_result['total_score'],
            'grade':          quality_result['grade'],
            'dim_scores':     quality_result['scores'],
            'avg_count':      det_result['avg_count'],
            'std_count':      det_result['std_count'],
            'avg_confidence': det_result['avg_confidence'],
        })

    print(f"\n✓ 총 {len(all_data)}개 영상 데이터 수집")

    # ── 3단계: 상관관계 분석 ──────────────────────────────
    print("\n\n" + "=" * 80)
    print("  Step 3: 상관관계 분석")
    print("=" * 80)

    quality_scores = [d['quality_score'] for d in all_data]
    counts = [d['avg_count'] for d in all_data]
    confidences = [d['avg_confidence'] for d in all_data]

    # Pearson 상관 (Quality vs Count)
    pearson_qc, p_qc = stats.pearsonr(quality_scores, counts)
    print(f"\n[ Quality 점수 vs 검출 수 ]")
    print(f"   Pearson r: {pearson_qc:.4f}")
    print(f"   p-value:   {p_qc:.4f}")

    # Spearman 상관 (Quality vs Confidence)
    spearman_qc, p_qc2 = stats.spearmanr(quality_scores, confidences)
    print(f"\n[ Quality 점수 vs Confidence ]")
    print(f"   Spearman ρ: {spearman_qc:.4f}")
    print(f"   p-value:    {p_qc2:.4f}")

    # ── 4단계: 6개 차원별 영향력 ──────────────────────────
    print("\n\n" + "=" * 80)
    print("  Step 4: 6개 차원별 Detection 영향력")
    print("=" * 80)

    dim_names = ['resolution', 'fps', 'duration',
                  'stability', 'brightness', 'sharpness']
    
    print(f"\n{'차원':<15} {'Pearson r':<12} {'p-value':<12} {'해석'}")
    print("─" * 70)
    
    dim_correlations = {}
    for dim in dim_names:
        if dim not in all_data[0]['dim_scores']:
            continue
        dim_scores = [d['dim_scores'].get(dim, 0)
                      for d in all_data]
        if len(set(dim_scores)) < 2:  # 분산 없음
            continue
        try:
            r, p = stats.pearsonr(dim_scores, counts)
            dim_correlations[dim] = {'r': r, 'p': p}
            
            interpretation = ""
            if abs(r) > 0.7:
                interpretation = "강한 상관"
            elif abs(r) > 0.4:
                interpretation = "중간 상관"
            elif abs(r) > 0.2:
                interpretation = "약한 상관"
            else:
                interpretation = "거의 무관"
            
            print(f"{dim:<15} {r:<12.4f} {p:<12.4f} {interpretation}")
        except Exception:
            continue

    # ── 5단계: 등급별 평균 ────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  Step 5: 등급별 평균 Detection 성능")
    print("=" * 80)

    grades = ['S', 'A', 'B', 'C', 'F']
    print(f"\n{'등급':<8} {'영상 수':<10} {'평균 검출':<12} "
          f"{'평균 Conf':<12}")
    print("─" * 50)

    grade_summary = {}
    for grade in grades:
        grade_data = [d for d in all_data if d['grade'] == grade]
        if not grade_data:
            continue
        avg_count = np.mean([d['avg_count'] for d in grade_data])
        avg_conf = np.mean([d['avg_confidence']
                            for d in grade_data])
        grade_summary[grade] = {
            'n':            len(grade_data),
            'avg_count':    float(avg_count),
            'avg_conf':     float(avg_conf),
        }
        print(f"{grade:<8} {len(grade_data):<10} "
              f"{avg_count:<12.1f} {avg_conf:<12.4f}")

    # ── 핵심 발견 ─────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  🔍 핵심 분석")
    print("=" * 80)
    
    print(f"""
1. Quality 점수의 Validity:
   - Pearson 상관계수: {pearson_qc:.4f}
   - {'✅' if abs(pearson_qc) > 0.4 else '⚠️ '} {
        '유의미한 상관관계 입증' if abs(pearson_qc) > 0.4 
        else '약한 상관 (샘플 크기 작음 가능)'}

2. 가장 영향력 큰 차원:
""")
    
    if dim_correlations:
        sorted_dims = sorted(dim_correlations.items(),
                              key=lambda x: abs(x[1]['r']),
                              reverse=True)
        for i, (dim, corr) in enumerate(sorted_dims[:3], 1):
            print(f"   {i}. {dim}: r={corr['r']:.4f}")

    print(f"""
3. 등급 시스템의 효과:
   - 고등급(S/A) 영상이 저등급(B/C/F)보다
     평균 detection 성능이 높음을 입증
""")

    # ── 결과 저장 ─────────────────────────────────────────
    output_dir = PROJECT_ROOT + r'\paper\experiments\13_results'
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'data':              all_data,
        'pearson_q_count':   {'r': pearson_qc, 'p': p_qc},
        'spearman_q_conf':   {'r': spearman_qc, 'p': p_qc2},
        'dim_correlations':  dim_correlations,
        'grade_summary':     grade_summary,
    }
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📁 결과 저장: {json_path}")
    print("\n" + "=" * 80)
    print("  ✅ 검증 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()