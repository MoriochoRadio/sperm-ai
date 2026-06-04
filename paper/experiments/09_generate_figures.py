"""
09_generate_figures.py
─────────────────────────────────
논문용 figures 생성
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = PROJECT_ROOT + r'\paper\figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Figure 2: 정규화 구성 요소별 영향 ────────────────────
def figure_2_normalization_components():
    methods = ['Original', 'Grayscale', 'CLAHE\n(Strong)',
               'CLAHE\n(Weak)']
    detections = [45.9, 39.0, 28.3, 35.6]
    confidences = [0.6468, 0.6219, 0.5433, 0.6165]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(methods))
    bars = ax1.bar(x, detections,
                    color=['#2DD4BF', '#60A5FA', '#EF4444', '#F59E0B'],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Normalization Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Sperm Detections', fontsize=12,
                   fontweight='bold', color='#1E293B')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#1E293B')
    ax1.set_ylim(0, 55)

    # 값 표시
    for bar, val in zip(bars, detections):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1,
                 f'{val:.1f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    # 우측 y축: confidence
    ax2 = ax1.twinx()
    ax2.plot(x, confidences, 'o-', color='#7C3AED',
             linewidth=2, markersize=10,
             label='Confidence')
    ax2.set_ylabel('Average Confidence', fontsize=12,
                   fontweight='bold', color='#7C3AED')
    ax2.tick_params(axis='y', labelcolor='#7C3AED')
    ax2.set_ylim(0.5, 0.7)

    # baseline 점선
    ax1.axhline(y=45.9, color='#10B981', linestyle='--',
                alpha=0.5, linewidth=1)
    ax1.text(3.5, 46.5, 'Baseline', color='#10B981',
             fontsize=10, fontweight='bold', ha='right')

    plt.title('Effect of Normalization Components on Detection',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig2_normalization_components.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {path}")


# ── Figure 3: 영상 품질 저하 시뮬레이션 ──────────────────
def figure_3_quality_degradation():
    conditions = ['Original\n(S grade)', 'Low\nResolution',
                  'Motion\nBlur', 'Noise', 'Low Light\n+ Blur']
    no_norm = [45.9, 44.5, 39.4, 8.9, 40.5]
    with_norm = [28.3, 38.0, 33.6, 6.8, 38.4]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_norm, width,
                   label='No Normalization',
                   color='#10B981', alpha=0.85,
                   edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, with_norm, width,
                   label='CLAHE Normalization',
                   color='#EF4444', alpha=0.85,
                   edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Image Quality Condition', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Average Sperm Detections', fontsize=12,
                  fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 55)
    ax.grid(axis='y', alpha=0.3)

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom',
                    fontsize=9)

    plt.title('Detection Performance Across Quality Conditions',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR,
                        'fig3_quality_degradation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {path}")


# ── Figure 4: Morphology 모델 비교 (Trade-off) ────────────
def figure_4_morphology_tradeoff():
    versions = ['v1', 'v2', 'v3', 'v4']
    mhsma_auc = [0.752, 0.800, 0.725, 0.716]
    visem_rate = [0.5, 0.5, 8.3, 13.5]
    colors = ['#94A3B8', '#60A5FA', '#10B981', '#F59E0B']

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter plot with arrows
    for i, ver in enumerate(versions):
        ax.scatter(mhsma_auc[i], visem_rate[i],
                   s=400, color=colors[i],
                   edgecolor='black', linewidth=2,
                   label=f'{ver}', zorder=3)
        # 라벨
        offset_y = 0.8 if visem_rate[i] < 10 else -1.5
        ax.annotate(ver, (mhsma_auc[i], visem_rate[i]),
                    fontsize=14, fontweight='bold',
                    ha='center', va='center', zorder=4,
                    color='white')

    # 채택 모델 강조
    ax.scatter(mhsma_auc[2], visem_rate[2],
               s=600, facecolor='none',
               edgecolor='#EF4444', linewidth=3,
               linestyle='--', zorder=2)
    ax.annotate('Selected', xy=(mhsma_auc[2], visem_rate[2]),
                xytext=(0.74, 12),
                fontsize=12, fontweight='bold', color='#EF4444',
                arrowprops=dict(arrowstyle='->', color='#EF4444',
                                lw=2))

    # WHO 기준선
    ax.axhline(y=4, color='#10B981', linestyle=':',
               linewidth=2, alpha=0.7)
    ax.text(0.81, 4.5, 'WHO Threshold (4%)',
            color='#10B981', fontsize=10, fontweight='bold')

    ax.set_xlabel('MHSMA Test Set AUC', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('VISEM Normal Morphology Rate (%)',
                  fontsize=12, fontweight='bold')
    ax.set_xlim(0.70, 0.82)
    ax.set_ylim(-2, 16)
    ax.grid(True, alpha=0.3)

    plt.title('Benchmark vs Real-World Performance Trade-off',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig4_morphology_tradeoff.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {path}")


# ── 메인 실행 ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  논문용 Figures 생성")
    print("=" * 70)
    print()

    figure_2_normalization_components()
    figure_3_quality_degradation()
    figure_4_morphology_tradeoff()

    print()
    print("=" * 70)
    print(f"  ✅ 모든 figures 저장 완료: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()