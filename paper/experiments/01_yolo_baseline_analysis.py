"""
01_yolo_baseline_analysis.py
─────────────────────────────────
기존 YOLO11m 학습 결과 분석
- 학습 곡선 + 최종 성능 확인
- 논문용 데이터 정리
"""

import pandas as pd
import os

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 결과 파일 경로
results_csv = PROJECT_ROOT + r'\models\yolo11_sperm_v2\results.csv'
args_yaml = PROJECT_ROOT + r'\models\yolo11_sperm_v2\args.yaml'

print("=" * 70)
print("  기존 YOLO11m 학습 결과 분석")
print("=" * 70)

# 1. 학습 설정 확인
print("\n📋 학습 설정 (args.yaml)")
print("─" * 70)
with open(args_yaml, 'r', encoding='utf-8') as f:
    content = f.read()
# 핵심 항목만 추출
key_items = ['model:', 'data:', 'epochs:', 'batch:', 'imgsz:',
             'lr0:', 'optimizer:', 'augment:']
for line in content.split('\n'):
    for k in key_items:
        if line.strip().startswith(k):
            print(f"  {line.strip()}")
            break

# 2. 학습 결과 분석
print("\n📊 학습 결과 (results.csv)")
print("─" * 70)
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()  # 공백 제거

print(f"  총 학습 epochs: {len(df)}")
print(f"  컬럼: {list(df.columns)}")

# 마지막 epoch (최종 성능)
print("\n🏁 최종 epoch 성능")
print("─" * 70)
last_row = df.iloc[-1]
print(f"  Epoch:      {int(last_row['epoch'])}")

# 주요 지표 (컬럼명이 다를 수 있어 안전하게 처리)
metric_keys = {
    'mAP50':       ['metrics/mAP50(B)', 'metrics/mAP_0.5'],
    'mAP50-95':    ['metrics/mAP50-95(B)', 'metrics/mAP_0.5:0.95'],
    'Precision':   ['metrics/precision(B)', 'metrics/precision'],
    'Recall':      ['metrics/recall(B)', 'metrics/recall'],
    'box_loss':    ['val/box_loss'],
    'cls_loss':    ['val/cls_loss'],
}

for label, possible_cols in metric_keys.items():
    for col in possible_cols:
        if col in df.columns:
            value = last_row[col]
            print(f"  {label:<12}: {value:.4f}")
            break

# 3. 최고 성능 epoch
print("\n🏆 최고 mAP50 epoch")
print("─" * 70)
map50_col = None
for col in df.columns:
    if 'mAP50' in col and 'mAP50-95' not in col:
        map50_col = col
        break

if map50_col:
    best_idx = df[map50_col].idxmax()
    best_row = df.iloc[best_idx]
    print(f"  Best Epoch:  {int(best_row['epoch'])}")
    print(f"  Best mAP50:  {best_row[map50_col]:.4f}")

# 4. 학습 곡선 요약
print("\n📈 학습 곡선 요약")
print("─" * 70)
if map50_col:
    print(f"  Epoch 1 mAP50:    {df[map50_col].iloc[0]:.4f}")
    print(f"  Epoch 50% mAP50:  {df[map50_col].iloc[len(df)//2]:.4f}")
    print(f"  Final mAP50:      {df[map50_col].iloc[-1]:.4f}")

print("\n" + "=" * 70)
print("  ✅ 분석 완료")
print("=" * 70)