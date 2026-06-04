"""
08_morphology_ablation.py
─────────────────────────────────
Morphology 모델 v1~v4 정보 수집 + 비교 정리

이미 학습된 4개 모델의:
- MHSMA test set 성능 (AUC)
- VISEM 적용 시 정상 형태율
- 학습 방법론 차이
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import json


def main():
    print("=" * 80)
    print("  실험 8: Morphology 모델 v1~v4 ablation 정리")
    print("=" * 80)

    # 기존 모델 정보 (project-journey.md + 메모리 기반)
    models_info = [
        {
            'version': 'v1',
            'description': '기본 학습 (BCE Loss)',
            'training_method': '기본 train/val/test split (1,000개)',
            'augmentation': '기본 (flip, rotation)',
            'loss': 'BCE Loss',
            'mhsma_auc_avg': 0.752,
            'mhsma_aucs': {
                'head': 0.733, 'acrosome': 0.745,
                'vacuole': 0.778, 'tail': 0.752
            },
            'visem_normal_rate': 0.5,  # %
            'verdict': '벤치마크 기본, VISEM 비현실적',
        },
        {
            'version': 'v2',
            'description': '전체 데이터 + Focal Loss',
            'training_method': '전체 1,540개 + WeightedRandomSampler',
            'augmentation': '기본 + 클래스 균형',
            'loss': 'Focal Loss (alpha tuning)',
            'mhsma_auc_avg': 0.800,
            'mhsma_aucs': {
                'head': 0.785, 'acrosome': 0.812,
                'vacuole': 0.823, 'tail': 0.780
            },
            'visem_normal_rate': 0.5,  # %
            'verdict': '벤치마크 최고, VISEM 여전히 비현실적',
        },
        {
            'version': 'v3',
            'description': 'VisemStyleAugment (도메인 적응)',
            'training_method': '전체 1,540개 + Focal + VISEM 스타일',
            'augmentation': 'VisemStyleAugment (모션블러+노이즈+'
                            '저해상도+CLAHE+명암반전)',
            'loss': 'Focal Loss',
            'mhsma_auc_avg': 0.725,
            'mhsma_aucs': {
                'head': 0.677, 'acrosome': 0.711,
                'vacuole': 0.767, 'tail': 0.752
            },
            'visem_normal_rate': 8.3,  # % (참가자 11번)
            'verdict': '⭐ 채택 - 벤치마크 손해 vs 실용성',
        },
        {
            'version': 'v4',
            'description': '강한 증강',
            'training_method': '전체 + Focal + 강한 VISEM 스타일',
            'augmentation': 'v3보다 공격적 (블러 70%, 저해상도 70%)',
            'loss': 'Focal Loss',
            'mhsma_auc_avg': 0.716,
            'mhsma_aucs': {
                'head': 0.658, 'acrosome': 0.701,
                'vacuole': 0.755, 'tail': 0.749
            },
            'visem_normal_rate': 13.5,  # %
            'verdict': 'VISEM 더 높지만 MHSMA 더 손해, 미채택',
        },
    ]

    # ── 비교 표 출력 ─────────────────────────────────────
    print("\n📊 모델 버전별 비교")
    print("─" * 90)
    print(f"{'버전':<6} {'MHSMA AUC':<12} {'VISEM 정상%':<14} "
          f"{'설명':<35}")
    print("─" * 90)

    for m in models_info:
        marker = " ⭐" if m['version'] == 'v3' else ""
        print(f"{m['version']:<6} "
              f"{m['mhsma_auc_avg']:<12.3f} "
              f"{m['visem_normal_rate']:<14.1f} "
              f"{m['description']:<35}{marker}")

    # ── MHSMA 부위별 AUC ─────────────────────────────────
    print("\n\n📊 MHSMA 부위별 AUC")
    print("─" * 80)
    print(f"{'버전':<6} {'머리':<10} {'첨체':<10} "
          f"{'공포':<10} {'꼬리':<10} {'평균':<10}")
    print("─" * 80)

    for m in models_info:
        a = m['mhsma_aucs']
        marker = " ⭐" if m['version'] == 'v3' else ""
        print(f"{m['version']:<6} "
              f"{a['head']:<10.3f} {a['acrosome']:<10.3f} "
              f"{a['vacuole']:<10.3f} {a['tail']:<10.3f} "
              f"{m['mhsma_auc_avg']:<10.3f}{marker}")

    # ── 핵심 발견 ────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  🔍 핵심 발견 — 벤치마크 vs 실용성 trade-off")
    print("=" * 80)
    print(f"""
1. v2 (벤치마크 최고, AUC 0.800):
   → MHSMA 데이터에는 잘 작동
   → VISEM 적용 시 정상 형태율 0.5% (비현실적)
   → 도메인 mismatch로 실패

2. v3 (도메인 적응, AUC 0.725):
   → MHSMA AUC 0.075 손해
   → BUT: VISEM 정상 형태율 8.3% (WHO 기준 ≥4% 충족)
   → 학습 시 VisemStyleAugment 적용
     · 모션 블러 (40%): 움직이는 정자 시뮬레이션
     · 가우시안 노이즈 (50%): 위상차 노이즈
     · 저해상도 (30%): 128→32→128 다운/업스케일
     · CLAHE (40%): 위상차 대비 특성
     · 명암 반전 (20%): 위상차 현미경 특성

3. v4 (더 강한 증강, AUC 0.716):
   → VISEM 정상률 더 높음 (13.5%)
   → BUT: MHSMA AUC 추가 손해
   → 균형 좋은 v3 채택

4. 핵심 메시지:
   → "벤치마크 성능"과 "실제 적용 성능"의 trade-off
   → 도메인 적응은 학습 단계에서 가장 효과적
   → 추론 단계 변환(예: CLAHE)은 오히려 해로움 (실험 4~6)
""")

    # ── 5가지 도메인 적응 시도 정리 ──────────────────────
    print("=" * 80)
    print("  🔬 5가지 도메인 적응 시도 (실패 사례 포함)")
    print("=" * 80)

    attempts = [
        {
            'method': '1. Pseudo-labeling',
            'description': 'VISEM 크롭에 v2 모델로 예측 → '
                           '높은 확신도 예측을 정답으로 사용',
            'result': '실패',
            'reason': '이미 편향된 예측이 더 강화됨 (확증 편향)',
        },
        {
            'method': '2. 기하학적 측정',
            'description': 'cv2.fitEllipse로 정자 머리 크기 직접 측정',
            'result': '실패',
            'reason': '크롭 스케일 보정 후에도 딥러닝보다 정밀도 낮음',
        },
        {
            'method': '3. TTA (Test-Time Augmentation)',
            'description': '추론 시 여러 번 증강 후 평균',
            'result': '부분 성공',
            'reason': '28.4% 도달 가능하나 크롭당 11회 추론으로 매우 느림',
        },
        {
            'method': '4. 앙상블 (v2 + v3)',
            'description': '두 모델 예측값 가중 평균',
            'result': '실패',
            'reason': 'v2 쪽으로 끌려가서 결국 0.9% (v2와 유사)',
        },
        {
            'method': '5. VisemStyleAugment (v3, 채택)',
            'description': '학습 시 VISEM 도메인 시뮬레이션 증강',
            'result': '✅ 성공',
            'reason': '학습 단계에서 도메인 갭 해결, '
                      '추론 시 추가 변환 불필요',
        },
    ]

    for a in attempts:
        print(f"\n{a['method']}")
        print(f"   설명: {a['description']}")
        print(f"   결과: {a['result']}")
        print(f"   이유: {a['reason']}")

    # ── 저장 ─────────────────────────────────────────────
    output_dir = PROJECT_ROOT + r'\paper\experiments\08_results'
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'models': models_info,
        'domain_adaptation_attempts': attempts,
    }
    json_path = os.path.join(output_dir, 'morphology_ablation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📁 결과 저장: {json_path}")
    print("\n" + "=" * 80)
    print("  ✅ 정리 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()