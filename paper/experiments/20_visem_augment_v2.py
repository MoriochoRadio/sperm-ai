"""
20_visem_augment_v2.py
─────────────────────────────────
Figure 2 v2: 2행 그리드로 재배치
페이지에 적합한 비율로 변경
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
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_gaussian_noise(img, std=15):
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise,
                    0, 255).astype(np.uint8)


def apply_low_resolution(img, scale=4):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // scale, h // scale),
                        interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h),
                       interpolation=cv2.INTER_LINEAR)


def apply_clahe(img):
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


def apply_full_visem_style(img):
    result = img.copy()
    result = apply_motion_blur(result, kernel_size=5)
    result = apply_gaussian_noise(result, std=10)
    result = apply_low_resolution(result, scale=3)
    return result


def main():
    print("=" * 80)
    print("  Figure 2 v2: 2행 그리드 (페이지 적합 비율)")
    print("=" * 80)

    mhsma_path = (PROJECT_ROOT + r'\data\raw'
                  r'\mhsma-dataset-master\mhsma-dataset-master'
                  r'\mhsma')
    
    # 데이터 로드
    x_train = np.load(os.path.join(mhsma_path,
                                     'x_128_train.npy'))
    y_head = np.load(os.path.join(mhsma_path,
                                    'y_head_train.npy'))
    y_acro = np.load(os.path.join(mhsma_path,
                                    'y_acrosome_train.npy'))
    y_vac = np.load(os.path.join(mhsma_path,
                                   'y_vacuole_train.npy'))
    y_tail = np.load(os.path.join(mhsma_path,
                                    'y_tail_train.npy'))
    
    # 정상 정자 선택
    normal_mask = (y_head == 0) & (y_acro == 0) & \
                   (y_vac == 0) & (y_tail == 0)
    normal_indices = np.where(normal_mask)[0]
    
    sample_img = x_train[normal_indices[0]]
    
    if len(sample_img.shape) == 2:
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)
    if sample_img.dtype != np.uint8:
        if sample_img.max() <= 1.0:
            sample_img = (sample_img * 255).astype(np.uint8)
        else:
            sample_img = sample_img.astype(np.uint8)

    # 시드 고정
    np.random.seed(42)
    
    # 6개 변환 (Intensity Inv. 제외 — 효과 작음)
    transforms_list = [
        ('(a) Original (MHSMA)',     sample_img),
        ('(b) Motion Blur',          apply_motion_blur(sample_img, 5)),
        ('(c) Gaussian Noise',       apply_gaussian_noise(sample_img, 12)),
        ('(d) Low Resolution',       apply_low_resolution(sample_img, 3)),
        ('(e) CLAHE Contrast',       apply_clahe(sample_img)),
        ('(f) Combined (VISEM-like)', apply_full_visem_style(sample_img)),
    ]

    # ── Figure 그리기: 2행 3열 ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
    
    for idx, (title, img) in enumerate(transforms_list):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        ax.imshow(img_rgb,
                   cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # 첫 번째와 마지막 박스 강조
        if title.startswith('(a)'):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#2563EB')
                spine.set_linewidth(3)
        elif title.startswith('(f)'):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#10B981')
                spine.set_linewidth(3)

    # 도메인 라벨
    fig.text(0.02, 0.95, 'Source',
              fontsize=11, fontweight='bold', color='#2563EB')
    fig.text(0.88, 0.45, 'Target',
              fontsize=11, fontweight='bold', color='#10B981')
    
    # 설명
    fig.text(0.5, 0.03,
              'VisemStyleAugment: simulated VISEM '
              'characteristics for domain adaptation',
              ha='center', fontsize=10, style='italic',
              color='#475569')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_path = (PROJECT_ROOT + r'\paper\figures'
                   r'\fig2_v2_visem_augment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                 facecolor='white')
    plt.close()
    
    print(f"\n📁 저장: {output_path}")
    print("\n📐 비율: 9:6.5 (페이지에 적합)")
    print("=" * 80)
    print("  ✅ Figure 2 v2 생성 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()