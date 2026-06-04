"""
24b_visem_smart_pick.py
─────────────────────────────────
VISEM 정자 후보 여러 개 추출

목적:
  - 우리 (e) Combined와 시각적으로 유사한 정자 찾기
  - 너무 선명하거나 highlight가 강한 것 제외
  - 후보 8개 모두 저장 → 사용자가 선택
"""

import os
import sys

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from ultralytics import YOLO


def evaluate_crop_quality(crop, target_image=None):
    """크롭의 'VISEM 평균스러움' 평가
    
    높은 점수 = 우리 시뮬레이션과 비슷
    낮은 점수 = 너무 선명하거나 highlight가 강함
    """
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # 1. 밝기 분포 (너무 밝지 않아야 함)
    mean_brightness = gray.mean()
    max_pixel = gray.max()
    
    # 2. 대비 (너무 강하지 않아야 함)
    std_brightness = gray.std()
    
    # 3. Highlight 픽셀 비율 (250 이상 픽셀)
    highlight_ratio = (gray > 240).sum() / gray.size
    
    # 4. 색 채도 (BGR 채널 차이)
    if len(crop.shape) == 3:
        b, g, r = cv2.split(crop)
        color_diff = (np.abs(b.astype(int) - g.astype(int)) +
                      np.abs(g.astype(int) - r.astype(int))).mean()
    else:
        color_diff = 0
    
    # 점수 계산 (높을수록 좋음)
    score = 100.0
    
    # 밝기 너무 높으면 감점
    if mean_brightness > 180:
        score -= 30
    elif mean_brightness > 150:
        score -= 10
    
    # Highlight 너무 많으면 큰 감점
    if highlight_ratio > 0.05:
        score -= 50
    elif highlight_ratio > 0.02:
        score -= 25
    
    # 대비 너무 강하면 감점
    if std_brightness > 70:
        score -= 20
    elif std_brightness > 50:
        score -= 10
    
    # 색 채도 너무 강하면 감점
    if color_diff > 30:
        score -= 25
    elif color_diff > 20:
        score -= 10
    
    return {
        'score': score,
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'highlight_ratio': highlight_ratio,
        'color_diff': color_diff,
    }


def main():
    print("=" * 80)
    print("  VISEM 정자 후보 추출 (스마트 선택)")
    print("=" * 80)

    model_path = (PROJECT_ROOT + r'\models'
                  r'\yolo11_sperm_v2\weights\best.pt')
    model = YOLO(model_path)
    
    base_dir = (PROJECT_ROOT + r'\data\raw'
                r'\VISEM-Tracking\VISEM_Tracking_Train_v4'
                r'\Train')
    
    output_dir = (PROJECT_ROOT + r'\paper\figures'
                  r'\visem_candidates')
    os.makedirs(output_dir, exist_ok=True)
    
    # 여러 참가자 영상 시도
    participants = ['11', '12', '14', '15', '22', '23']
    
    all_candidates = []
    
    for participant in participants:
        video_path = os.path.join(base_dir, participant,
                                    f'{participant}.mp4')
        
        if not os.path.exists(video_path):
            continue
        
        print(f"\n📹 참가자 {participant}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 여러 프레임에서 정자 추출
        test_frames = [
            total_frames // 6,
            total_frames // 3,
            total_frames // 2,
            total_frames * 2 // 3,
            total_frames * 5 // 6,
        ]
        
        for frame_idx in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # 중간 conf 임계값 (너무 높지 않게)
            results = model(frame, conf=0.2, verbose=False)
            
            if len(results[0].boxes) == 0:
                continue
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # 각 정자 평가
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].astype(int)
                
                # 크기 필터
                w, h = x2 - x1, y2 - y1
                if w < 15 or h < 15 or w > 80 or h > 80:
                    continue
                
                # 정사각형 크롭
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                size = max(w, h) + 25
                half = size // 2
                
                x1c = max(0, cx - half)
                y1c = max(0, cy - half)
                x2c = min(frame.shape[1], cx + half)
                y2c = min(frame.shape[0], cy + half)
                
                crop = frame[y1c:y2c, x1c:x2c]
                
                if crop.shape[0] < 30 or crop.shape[1] < 30:
                    continue
                
                # 128x128 리사이즈
                crop_resized = cv2.resize(
                    crop, (128, 128),
                    interpolation=cv2.INTER_LINEAR)
                
                # 품질 평가
                quality = evaluate_crop_quality(crop_resized)
                
                all_candidates.append({
                    'participant': participant,
                    'frame': frame_idx,
                    'box_idx': i,
                    'crop': crop_resized,
                    'conf': confs[i],
                    **quality,
                })
        
        cap.release()
    
    if not all_candidates:
        print("\n❌ 후보 없음")
        return
    
    # 점수 기준 정렬
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 상위 12개 저장
    top_n = min(12, len(all_candidates))
    print(f"\n📊 총 {len(all_candidates)}개 후보 중 상위 {top_n}개:")
    print("-" * 80)
    
    for i, cand in enumerate(all_candidates[:top_n]):
        filename = (f'cand_{i+1:02d}_p{cand["participant"]}'
                     f'_f{cand["frame"]}_score{cand["score"]:.0f}.png')
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cand['crop'])
        
        print(f"  {i+1:2d}. P{cand['participant']:>3} "
               f"frame={cand['frame']:>5} "
               f"conf={cand['conf']:.2f} "
               f"score={cand['score']:>5.0f} "
               f"bright={cand['mean_brightness']:.0f} "
               f"hl={cand['highlight_ratio']*100:.1f}% "
               f"color={cand['color_diff']:.0f}")
    
    # 비교 시트 생성 (모든 후보 한 장에)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    n_cols = 4
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(12, 3 * n_rows))
    
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols] if n_rows > 1 \
            else axes[i % n_cols]
        
        if i < top_n:
            cand = all_candidates[i]
            crop_rgb = cv2.cvtColor(cand['crop'],
                                      cv2.COLOR_BGR2RGB)
            ax.imshow(crop_rgb)
            ax.set_title(
                f"#{i+1} P{cand['participant']} "
                f"score={cand['score']:.0f}",
                fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    sheet_path = os.path.join(output_dir,
                                'all_candidates.png')
    plt.savefig(sheet_path, dpi=200, bbox_inches='tight',
                 facecolor='white')
    plt.close()
    
    print(f"\n📁 후보 저장 위치:")
    print(f"   개별: {output_dir}\\cand_*.png")
    print(f"   비교 시트: {sheet_path}")
    print("\n" + "=" * 80)
    print("  ✅ 완료! all_candidates.png를 보고 마음에 드는 번호 선택")
    print("=" * 80)


if __name__ == '__main__':
    main()