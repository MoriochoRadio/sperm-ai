"""
18_visem_augment_figure.py
─────────────────────────────────
Figure 2: VisemStyleAugment 시각화 (MHSMA 원본 사용)

목적:
  - MHSMA (source) → VISEM-style (target) 변환 시각화
  - 도메인 적응의 시각적 증거
  - 논문 핵심 figure
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def apply_motion_blur(img, kernel_size=7):
    """모션 블러 (수평 방향)"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_gaussian_noise(img, std=15):
    """가우시안 노이즈"""
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise,
                    0, 255).astype(np.uint8)


def apply_low_resolution(img, scale=4):
    """저해상도 시뮬레이션 (다운/업스케일)"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // scale, h // scale),
                        interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h),
                       interpolation=cv2.INTER_LINEAR)


def apply_clahe(img):
    """CLAHE 대비 향상"""
    if len(img.shape) == 2 or img.shape[2] == 1:
        clahe = cv2.createCLAHE(clipLimit=3.0,
                                  tileGridSize=(8, 8))
        return clahe.apply(img.squeeze())
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0,
                                  tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_intensity_inversion(img):
    """명암 반전 (위상차 현미경 특성)"""
    return 255 - img


def apply_full_visem_style(img):
    """모든 변환 결합 (v3 학습 시 적용 스타일)"""
    result = img.copy()
    result = apply_motion_blur(result, kernel_size=5)
    result = apply_gaussian_noise(result, std=10)
    result = apply_low_resolution(result, scale=3)
    return result


def main():
    print("=" * 80)
    print("  Figure 2: VisemStyleAugment 시각화 (MHSMA → VISEM)")
    print("=" * 80)

    # ── MHSMA 데이터 로드 ─────────────────────────────────
    mhsma_path = (PROJECT_ROOT + r'\data\raw'
                  r'\mhsma-dataset-master\mhsma-dataset-master'
                  r'\mhsma')
    
    print(f"\n📁 MHSMA 데이터 로딩...")
    print(f"   경로: {mhsma_path}")
    
    # 128×128 이미지 로드 (train set)
    x_train_path = os.path.join(mhsma_path, 'x_128_train.npy')
    y_head_path = os.path.join(mhsma_path, 'y_head_train.npy')
    y_acro_path = os.path.join(mhsma_path, 'y_acrosome_train.npy')
    y_vac_path = os.path.join(mhsma_path, 'y_vacuole_train.npy')
    y_tail_path = os.path.join(mhsma_path, 'y_tail_train.npy')
    
    x_train = np.load(x_train_path)
    y_head = np.load(y_head_path)
    y_acro = np.load(y_acro_path)
    y_vac = np.load(y_vac_path)
    y_tail = np.load(y_tail_path)
    
    print(f"   ✓ 이미지 shape: {x_train.shape}")
    print(f"   ✓ 라벨 shape:   head {y_head.shape}, "
          f"acro {y_acro.shape}")
    print(f"   ✓ 데이터 타입:   {x_train.dtype}")

    # ── 정상 정자 (모든 부위 정상) 샘플 찾기 ──────────────
    # 0=정상, 1=비정상
    normal_mask = (y_head == 0) & (y_acro == 0) & \
                   (y_vac == 0) & (y_tail == 0)
    normal_indices = np.where(normal_mask)[0]
    
    print(f"   ✓ 모든 부위 정상 샘플: {len(normal_indices)}개")
    
    if len(normal_indices) == 0:
        print(f"   ⚠️  정상 샘플 없음, 첫 번째 이미지 사용")
        sample_idx = 0
    else:
        # 첫 번째 정상 샘플 사용
        sample_idx = normal_indices[0]
    
    sample_img = x_train[sample_idx]
    print(f"   ✓ 선택된 샘플: index {sample_idx}")
    print(f"     shape: {sample_img.shape}, "
          f"dtype: {sample_img.dtype}")

    # 채널 처리 (MHSMA는 보통 그레이스케일)
    if len(sample_img.shape) == 2:
        # 그레이스케일 → BGR로 변환
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)
    elif len(sample_img.shape) == 3 and sample_img.shape[2] == 1:
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)
    
    # uint8로 정규화
    if sample_img.dtype != np.uint8:
        if sample_img.max() <= 1.0:
            sample_img = (sample_img * 255).astype(np.uint8)
        else:
            sample_img = sample_img.astype(np.uint8)
    
    print(f"   ✓ 처리 후: {sample_img.shape}, "
          f"{sample_img.dtype}, "
          f"range [{sample_img.min()}-{sample_img.max()}]")

    # ── 변환 적용 ─────────────────────────────────────────
    print("\n📊 VisemStyleAugment 변환 적용...")
    
    # 시드 고정 (재현성)
    np.random.seed(42)
    
    transforms_list = [
        ('Original\n(MHSMA)',          sample_img),
        ('Motion Blur\n(40% prob)',    apply_motion_blur(sample_img, 5)),
        ('Gaussian Noise\n(50% prob)', apply_gaussian_noise(sample_img, 12)),
        ('Low Resolution\n(30% prob)', apply_low_resolution(sample_img, 3)),
        ('CLAHE Contrast\n(40% prob)', apply_clahe(sample_img)),
        ('Intensity Inv.\n(20% prob)', apply_intensity_inversion(sample_img)),
        ('Combined\n(VISEM-style)',    apply_full_visem_style(sample_img)),
    ]
    
    print(f"   ✓ {len(transforms_list)}가지 변환 완료")

    # ── Figure 그리기 ─────────────────────────────────────
    print("\n🎨 Figure 생성...")
    
    fig, axes = plt.subplots(1, 7, figsize=(16, 3.5))
    
    for ax, (title, img) in zip(axes, transforms_list):
        # BGR → RGB
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        ax.imshow(img_rgb,
                   cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # 첫 번째와 마지막에 박스 강조
        if title.startswith('Original'):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#2563EB')
                spine.set_linewidth(3)
        elif title.startswith('Combined'):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#10B981')
                spine.set_linewidth(3)

    # 도메인 라벨
    fig.text(0.075, 0.94, 'Source Domain',
              fontsize=12, fontweight='bold', color='#2563EB')
    fig.text(0.84, 0.94, 'Target Domain (Sim.)',
              fontsize=12, fontweight='bold', color='#10B981')
    
    # 변환 설명
    fig.text(0.5, 0.02,
              'VisemStyleAugment: Domain-aware augmentation '
              'simulating VISEM characteristics during MHSMA training',
              ha='center', fontsize=10, style='italic',
              color='#475569')

    plt.suptitle('Figure 2. VisemStyleAugment for Domain Adaptation '
                  '(MHSMA → VISEM)',
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    # 저장
    output_dir = PROJECT_ROOT + r'\paper\figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig2_visem_augment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                 facecolor='white')
    plt.close()
    
    print(f"\n📁 저장: {output_path}")
    
    # ── 추가: 4개 다른 정상 정자로 더 풍부한 figure ─────
    print("\n🎨 추가 Figure (4개 샘플 비교) 생성...")
    
    if len(normal_indices) >= 4:
        np.random.seed(42)
        sample_indices = np.random.choice(
            normal_indices, 4, replace=False)
        
        fig, axes = plt.subplots(4, 3, figsize=(9, 11))
        
        col_titles = ['Original (MHSMA)',
                      'Combined Augment',
                      'Real VISEM-like']
        
        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(
                title, fontsize=12, fontweight='bold', pad=10)
        
        for row_idx, idx in enumerate(sample_indices):
            img = x_train[idx]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 1. 원본
            axes[row_idx, 0].imshow(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[row_idx, 0].axis('off')
            
            # 2. 결합 증강
            np.random.seed(row_idx)
            combined = apply_full_visem_style(img)
            axes[row_idx, 1].imshow(
                cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            axes[row_idx, 1].axis('off')
            
            # 3. 더 강한 VISEM 시뮬레이션
            heavy = apply_motion_blur(img, 7)
            heavy = apply_gaussian_noise(heavy, 15)
            heavy = apply_low_resolution(heavy, 4)
            axes[row_idx, 2].imshow(
                cv2.cvtColor(heavy, cv2.COLOR_BGR2RGB))
            axes[row_idx, 2].axis('off')
        
        # 화살표
        fig.text(0.36, 0.5, '→', fontsize=30,
                  ha='center', va='center', color='#2563EB')
        fig.text(0.66, 0.5, '→', fontsize=30,
                  ha='center', va='center', color='#10B981')
        
        plt.suptitle(
            'Domain Adaptation: MHSMA → VISEM-style (4 samples)',
            fontsize=13, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        output_path2 = os.path.join(output_dir,
                                      'fig2b_samples_comparison.png')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight',
                     facecolor='white')
        plt.close()
        
        print(f"📁 저장: {output_path2}")
    
    print("\n" + "=" * 80)
    print("  ✅ Figure 2 생성 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()