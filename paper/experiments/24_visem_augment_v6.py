"""
24_visem_augment_v6.py
─────────────────────────────────
Figure 2 v6: 실제 VISEM 추가 + 도메인 비교

수정: VISEM 경로 수정 (VISEM-Tracking 폴더 추가)
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


def apply_full_visem_style(img):
    result = img.copy()
    result = apply_motion_blur(result, kernel_size=5)
    result = apply_gaussian_noise(result, std=10)
    result = apply_low_resolution(result, scale=3)
    return result


def extract_visem_sperm_crop():
    """VISEM 영상에서 실제 정자 크롭 추출"""
    from ultralytics import YOLO
    
    model_path = (PROJECT_ROOT + r'\models'
                  r'\yolo11_sperm_v2\weights\best.pt')
    
    if not os.path.exists(model_path):
        print(f"   ⚠️  YOLO 모델 없음: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # ⭐ 수정된 경로 (VISEM-Tracking 폴더 추가)
    base_dir = (PROJECT_ROOT + r'\data\raw'
                r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                r'\Train')
    
    video_candidates = ['14', '11', '12', '15', '22', '23']
    
    for participant in video_candidates:
        video_path = os.path.join(base_dir, participant,
                                    f'{participant}.mp4')
        
        if not os.path.exists(video_path):
            print(f"   ⚠️  영상 없음: {video_path}")
            continue
        
        print(f"   📹 시도: 참가자 {participant}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"      ⚠️  비디오 열기 실패")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 여러 프레임 시도
        test_positions = [total_frames // 4,
                           total_frames // 2,
                           total_frames * 3 // 4]
        
        for frame_idx in test_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # 다양한 conf 임계값 시도
            for conf_thresh in [0.3, 0.2, 0.15, 0.1]:
                results = model(frame, conf=conf_thresh,
                                verbose=False)
                
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    confs = boxes.conf.cpu().numpy()
                    
                    # 가장 confident한 정자 + 적당한 크기
                    xyxy = boxes.xyxy.cpu().numpy()
                    sizes = ((xyxy[:, 2] - xyxy[:, 0]) *
                              (xyxy[:, 3] - xyxy[:, 1]))
                    
                    # 너무 작거나 큰 것 제외
                    valid = (sizes > 200) & (sizes < 5000)
                    if valid.sum() == 0:
                        continue
                    
                    valid_confs = np.where(valid, confs, -1)
                    best_idx = np.argmax(valid_confs)
                    
                    box = xyxy[best_idx]
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # 정사각형 크롭
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    size = max(x2 - x1, y2 - y1) + 30
                    
                    half = size // 2
                    x1c = max(0, cx - half)
                    y1c = max(0, cy - half)
                    x2c = min(frame.shape[1], cx + half)
                    y2c = min(frame.shape[0], cy + half)
                    
                    crop = frame[y1c:y2c, x1c:x2c]
                    
                    if crop.shape[0] < 30 or crop.shape[1] < 30:
                        continue
                    
                    crop_resized = cv2.resize(
                        crop, (128, 128),
                        interpolation=cv2.INTER_LINEAR)
                    
                    cap.release()
                    print(f"      ✅ 성공! 프레임 {frame_idx}, "
                           f"conf={confs[best_idx]:.3f}")
                    return crop_resized
        
        cap.release()
        print(f"      ⚠️  탐지 실패")
    
    return None


def main():
    print("=" * 80)
    print("  Figure 2 v6: 실제 VISEM 추가 + 도메인 비교")
    print("=" * 80)

    # MHSMA 로드
    print("\n📂 MHSMA 데이터 로드...")
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
    
    print(f"   ✅ MHSMA 정상 정자 크롭 로드")

    # 실제 VISEM 정자 추출
    print("\n📹 실제 VISEM 영상에서 정자 추출...")
    visem_img = extract_visem_sperm_crop()
    
    if visem_img is None:
        print("   ❌ VISEM 크롭 실패")
        print("   → 경로/모델 다시 확인 필요")
        return
    
    print(f"   ✅ VISEM 크롭 성공: {visem_img.shape}")

    np.random.seed(42)
    
    transforms_list = [
        ('(a) MHSMA Original',     sample_img,                                  'source'),
        ('(g) Real VISEM',          visem_img,                                  'target_real'),
        ('(b) Motion Blur',         apply_motion_blur(sample_img, 5),           'normal'),
        ('(c) Gaussian Noise',      apply_gaussian_noise(sample_img, 12),       'normal'),
        ('(d) Low Resolution',      apply_low_resolution(sample_img, 3),        'normal'),
        ('(e) Combined',            apply_full_visem_style(sample_img),         'target_synth'),
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
    
    for idx, (title, img, kind) in enumerate(transforms_list):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        ax.imshow(img_rgb,
                   cmap='gray' if len(img.shape) == 2 else None)
        
        if kind == 'source':
            ax.set_title(title, fontsize=15, fontweight='bold',
                         pad=6, color='#2563EB')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#2563EB')
                spine.set_linewidth(4)
        elif kind == 'target_real':
            ax.set_title(title, fontsize=15, fontweight='bold',
                         pad=6, color='#9333EA')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#9333EA')
                spine.set_linewidth(4)
        elif kind == 'target_synth':
            ax.set_title(title, fontsize=15, fontweight='bold',
                         pad=6, color='#10B981')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#10B981')
                spine.set_linewidth(4)
        else:
            ax.set_title(title, fontsize=15, fontweight='bold',
                         pad=6)
        
        ax.axis('off')

    fig.text(0.27, 0.965, 'Source Domain (MHSMA)',
              fontsize=13, fontweight='bold', color='#2563EB',
              ha='center')
    fig.text(0.73, 0.965, 'Target Domain (Real)',
              fontsize=13, fontweight='bold', color='#9333EA',
              ha='center')
    fig.text(0.73, 0.025,
              'Target (Synthetic)',
              fontsize=13, fontweight='bold', color='#10B981',
              ha='center')

    output_path = (PROJECT_ROOT + r'\paper\figures'
                   r'\fig2_v6_visem_augment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                 facecolor='white')
    plt.close()
    
    print(f"\n📁 저장: {output_path}")
    print("=" * 80)
    print("  ✅ Figure 2 v6 생성 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()