"""
28_extract_ci_code.py
─────────────────────────────────
15_statistical_validation.py의 95% CI 계산 코드 정확히 추출

검색 키워드:
  - ci_low, ci_high
  - 95%
  - bootstrap, resample
  - percentile, np.random
  - scipy.stats
  - sem, t.ppf, t.interval
"""

import os

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

target_file = PROJECT_ROOT + r'\paper\experiments\15_statistical_validation.py'

if not os.path.exists(target_file):
    print(f"❌ 파일 없음: {target_file}")
    exit()

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# CI 계산 관련 키워드
keywords = [
    'ci_low', 'ci_high', '95%', 'bootstrap', 'resample',
    'percentile', 'np.random', 'scipy.stats',
    'sem', 't.ppf', 't.interval', 'norm.interval',
    'n_iter', 'n_boot', 'B =', 'np.array',
]

# 키워드 포함된 라인 찾기
matched_lines = []
for i, line in enumerate(lines):
    lower = line.lower()
    for kw in keywords:
        if kw.lower() in lower:
            matched_lines.append(i)
            break

# 주변 ±3줄 포함해서 블록 구성
blocks = []
if matched_lines:
    current_start = matched_lines[0] - 3
    current_end   = matched_lines[0] + 4

    for ln in matched_lines[1:]:
        if ln - 3 <= current_end:
            current_end = ln + 4
        else:
            blocks.append((max(0, current_start), min(len(lines), current_end)))
            current_start = ln - 3
            current_end   = ln + 4

    blocks.append((max(0, current_start), min(len(lines), current_end)))

print("=" * 80)
print("  15_statistical_validation.py")
print("  95% CI 계산 코드 정확 추출")
print("=" * 80)

if not blocks:
    print("\n❌ CI 관련 코드 없음")
else:
    print(f"\n📊 발견된 블록: {len(blocks)}개\n")

    for idx, (start, end) in enumerate(blocks, 1):
        print(f"{'─' * 80}")
        print(f"  블록 {idx} (L{start+1} ~ L{end})")
        print(f"{'─' * 80}")
        for j in range(start, end):
            print(f"L{j+1:>4}: {lines[j].rstrip()}")
        print()

print("=" * 80)
print("  결과 해석 가이드")
print("=" * 80)
print("""
🔍 코드에서 확인할 것:

1. "bootstrap" 또는 "resample" 단어:
   → 있으면 Bootstrap CI ✓ → 본문 표현 OK
   
2. "np.random.choice" 또는 명시적 resampling 루프:
   → 있으면 직접 구현 Bootstrap ✓ → 본문 OK + n_iter 값 확인

3. "scipy.stats.t.interval" 또는 "sem":
   → t-distribution CI → 본문 "Bootstrap" 표현 ❌ 수정 필요
   
4. "scipy.stats.norm.interval":
   → Normal CI → 본문 "Bootstrap" 표현 ❌ 수정 필요

5. "np.percentile([...], [2.5, 97.5])":
   → Percentile CI → Bootstrap 결과로 적용된 경우 OK

→ 실제 코드 보고 정확한 방법 식별!
""")