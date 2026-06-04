"""
11_system_diagram.py
─────────────────────────────────
논문 Figure 1: 시스템 전체 아키텍처 다이어그램
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def draw_box(ax, x, y, w, h, text, color='#E0F2FE',
              edge='#0284C7', fontsize=10, fontweight='normal'):
    """박스 그리기"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=edge,
        linewidth=2, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            zorder=3, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#475569'):
    """화살표 그리기"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=20,
        color=color, linewidth=2, zorder=1)
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(13, 8))

    # ── 타이틀 ───────────────────────────────────────────
    ax.text(6.5, 7.7, 'AI-CASA System Architecture',
            ha='center', fontsize=15, fontweight='bold')
    ax.text(6.5, 7.3,
            'Quality-Aware Sperm Analysis with Domain Adaptation',
            ha='center', fontsize=11, style='italic',
            color='#64748B')

    # ── 1. 입력 ─────────────────────────────────────────
    draw_box(ax, 0.3, 5.5, 1.8, 1.0,
             'Input\nMicroscopy Video',
             color='#FEF3C7', edge='#F59E0B',
             fontsize=10, fontweight='bold')

    # ── 2. Quality Assessment ⭐ 우리 기여 ───────────────
    draw_box(ax, 2.5, 5.5, 2.2, 1.0,
             'Quality Assessment\n(6-dim scoring)\n★ S/A/B/C/F',
             color='#DBEAFE', edge='#2563EB',
             fontsize=10, fontweight='bold')

    # ── 분기점 ──────────────────────────────────────────
    # F등급 → 거부
    draw_box(ax, 5.2, 6.5, 1.8, 0.8,
             'Reject + Guide\n(F grade)',
             color='#FEE2E2', edge='#DC2626',
             fontsize=9)

    # ── 3. Smart Normalization ⭐ 우리 기여 ──────────────
    draw_box(ax, 5.2, 5.0, 2.2, 1.0,
             'Smart Normalization\n(spec-only,\nno pixel transform)',
             color='#DBEAFE', edge='#2563EB',
             fontsize=10, fontweight='bold')

    # ── 4. Detection ────────────────────────────────────
    draw_box(ax, 7.8, 5.5, 1.8, 1.0,
             'Detection\nYOLO11m\nmAP 0.697',
             color='#E0E7FF', edge='#6366F1',
             fontsize=10)

    # ── 5. Tracking ─────────────────────────────────────
    draw_box(ax, 9.9, 5.5, 1.8, 1.0,
             'Tracking\nByteTrack',
             color='#E0E7FF', edge='#6366F1',
             fontsize=10)

    # ── 분석 영역 (3가지 분석) ─────────────────────────
    # Motility
    draw_box(ax, 0.5, 3.0, 2.3, 1.2,
             'Motility Analysis\nRidge + RF\nMAE 6.9%p',
             color='#D1FAE5', edge='#10B981',
             fontsize=10, fontweight='bold')

    # Kinematics
    draw_box(ax, 3.2, 3.0, 2.3, 1.2,
             'CASA Kinematics\nVCL/VSL/VAP\nLIN/STR/WOB/ALH',
             color='#D1FAE5', edge='#10B981',
             fontsize=10)

    # Morphology ⭐ 우리 기여
    draw_box(ax, 5.9, 3.0, 2.3, 1.2,
             'Morphology v3 ★\nEfficientNet-B3\n+ VisemStyleAugment',
             color='#DBEAFE', edge='#2563EB',
             fontsize=10, fontweight='bold')

    # Quality + Confidence
    draw_box(ax, 8.6, 3.0, 2.3, 1.2,
             'Confidence Score\n(0-100)\nWHO Compliance',
             color='#FEF3C7', edge='#F59E0B',
             fontsize=10)

    # ── 6. Output Report ─────────────────────────────────
    draw_box(ax, 4.5, 0.8, 4.0, 1.2,
             'Comprehensive Report\nNormal / Caution / Rejected\n'
             '(Layperson + Expert modes)',
             color='#FEF3C7', edge='#F59E0B',
             fontsize=11, fontweight='bold')

    # ── 화살표 ──────────────────────────────────────────
    # 1 → 2
    draw_arrow(ax, 2.1, 6.0, 2.5, 6.0)

    # 2 → 거부 (F)
    draw_arrow(ax, 4.4, 6.3, 5.2, 6.8, color='#DC2626')
    ax.text(4.7, 6.7, 'F', color='#DC2626',
            fontsize=9, fontweight='bold')

    # 2 → 3 (정상)
    draw_arrow(ax, 4.4, 5.7, 5.2, 5.5)

    # 3 → 4
    draw_arrow(ax, 7.4, 6.0, 7.8, 6.0)

    # 4 → 5
    draw_arrow(ax, 9.6, 6.0, 9.9, 6.0)

    # 5 → 분석들 (분기)
    draw_arrow(ax, 10.5, 5.5, 1.7, 4.2, color='#475569')  # → motility
    draw_arrow(ax, 10.5, 5.5, 4.4, 4.2, color='#475569')  # → kinematics
    draw_arrow(ax, 10.5, 5.5, 7.0, 4.2, color='#475569')  # → morphology
    draw_arrow(ax, 10.5, 5.5, 9.7, 4.2, color='#475569')  # → confidence

    # 분석 → 보고서
    draw_arrow(ax, 1.6, 3.0, 5.5, 2.0, color='#10B981')
    draw_arrow(ax, 4.3, 3.0, 6.0, 2.0, color='#10B981')
    draw_arrow(ax, 7.0, 3.0, 6.5, 2.0, color='#2563EB')
    draw_arrow(ax, 9.7, 3.0, 7.5, 2.0, color='#F59E0B')

    # ── 범례 ────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor='#DBEAFE', edgecolor='#2563EB',
                       label='★ Our contributions'),
        mpatches.Patch(facecolor='#E0E7FF', edgecolor='#6366F1',
                       label='Detection / Tracking'),
        mpatches.Patch(facecolor='#D1FAE5', edgecolor='#10B981',
                       label='Analysis modules'),
        mpatches.Patch(facecolor='#FEF3C7', edgecolor='#F59E0B',
                       label='I/O + Reporting'),
        mpatches.Patch(facecolor='#FEE2E2', edgecolor='#DC2626',
                       label='Rejection path'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=9, frameon=True, ncol=5,
              bbox_to_anchor=(0.0, -0.02))

    # ── 축 정리 ──────────────────────────────────────────
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # ── 저장 ────────────────────────────────────────────
    output_path = (PROJECT_ROOT + r'\paper\figures'
                   r'\fig1_system_architecture.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()

    print(f"✓ Figure 1 저장: {output_path}")


if __name__ == '__main__':
    main()