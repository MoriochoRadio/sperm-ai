"""
23_visem_augment_v5.py
─────────────────────────────────
Figure 2 v5: 도메인 라벨 위치 최적화

수정 사항:
  - Source Domain: (a) 위쪽 (좌측)
  - Target Domain: (f) 아래쪽 (우측)
  - 박스 색상 + 라벨 색상 매칭
  - 시각적 흐름 명확
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
    print("  Figure 2 v5: 도메인 라벨 위치 최적화")
    print("=" * 80)

    mhsma_path = (PROJECT_ROOT + r'\data\raw'
                  r'\mhsma-dataset-master\mhsma-dataset-master'
                  r'\mhsma')
    
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

    np.random.seed(42)
    
    transforms_list = [
        ('(a) Original',     sample_img),
        ('(b) Motion Blur',  apply_motion_blur(sample_img, 5)),
        ('(c) Gaussian Noise', apply_gaussian_noise(sample_img, 12)),
        ('(d) Low Resolution', apply_low_resolution(sample_img, 3)),
        ('(e) CLAHE Contrast', apply_clahe(sample_img)),
        ('(f) Combined',     apply_full_visem_style(sample_img)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(7, 9.5))
    
    plt.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.93,
        bottom=0.06,
        wspace=0.08,
        hspace=0.30
    )
    
    for idx, (title, img) in enumerate(transforms_list):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        ax.imshow(img_rgb,
                   cmap='gray' if len(img.shape) == 2 else None)
        
        # 박스 강조 + 제목 색상 매칭
        if title.startswith('(a)'):
            # Source: 파란색
            ax.set_title(title, fontsize=15, fontweight='bold',
                         pad=6, color='#2563EB')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#2563EB')
                spine.set_linewidth(4)
        elif title.startswith('(f)'):
            # Target: 초록색
            ax.set_title(title, fontsize=15, fontweight='bold',
                         pad=6, color='#10B981')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#10B981')
                spine.set_linewidth(4)
        else:
            # 일반: 검은색
            ax.set_title(title, fontsize=15, fontweight='bold',
                         pad=6)
        
        ax.axis('off')

    # ⭐ 핵심 수정: Source/Target 라벨 위치
    
    # Source Domain - (a)의 위쪽 (좌측 상단)
    # (a)는 row=0, col=0이므로 figure의 좌상단 영역
    fig.text(0.27, 0.965, 'Source Domain (MHSMA)',
              fontsize=13, fontweight='bold', color='#2563EB',
              ha='center')
    
    # Target Domain - (f)의 아래쪽 (우측 하단)
    # (f)는 row=2, col=1이므로 figure의 우하단 영역
    fig.text(0.73, 0.025,
              'Target Domain (VISEM-like)',
              fontsize=13, fontweight='bold', color='#10B981',
              ha='center')

    output_path = (PROJECT_ROOT + r'\paper\figures'
                   r'\fig2_v5_visem_augment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                 facecolor='white')
    plt.close()
    
    print(f"\n📁 저장: {output_path}")
    print("\n📐 라벨 위치:")
    print("   Source Domain  → (a) 위쪽 (좌측)")
    print("   Target Domain  → (f) 아래쪽 (우측)")
    print("📐 박스 + 제목 색상 매칭:")
    print("   (a) Original:  파란색 (Source)")
    print("   (f) Combined:  초록색 (Target)")
    print("=" * 80)
    print("  ✅ Figure 2 v5 생성 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()