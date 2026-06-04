"""
31_audit_critical_numbers.py
─────────────────────────────────
의심 수치들의 실제 출처를 직접 확인:
  1. 4.2절 정규화 수치 (41.8, 40.1, 71.4, 62.7)
  2. Table 2 의사 라벨링(0.7%), 기하학적(29.0%) 수치
  3. VisemStyleAugment 확률 (40%, 50%, 30%, 40%, 20%)
"""

import os
import json
import glob

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base = PROJECT_ROOT


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def show_file(path, description=""):
    if not os.path.exists(path):
        print(f"\n   ❌ 파일 없음: {os.path.relpath(path, base)}")
        return False

    print(f"\n📄 {os.path.relpath(path, base)}")
    if description:
        print(f"   ({description})")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"   ⚠️ 읽기 실패: {e}")
        return False

    # JSON이면 예쁘게 출력
    if path.endswith('.json'):
        try:
            data = json.loads(content)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return True
        except Exception:
            pass

    print(content[:3000])  # 처음 3000자
    if len(content) > 3000:
        print(f"\n... (전체 {len(content)}자 중 처음 3000자만)")
    return True


# ── 1. 도메인 적응 실험 결과 (Table 2 출처) ─────────────
print_section("1️⃣ Table 2 도메인 적응 실험 결과 출처 확인")

show_file(
    os.path.join(base, 'paper', 'experiments', '08_results',
                 'morphology_ablation.json'),
    "5가지 도메인 적응 시도 결과"
)

show_file(
    os.path.join(base, 'paper', 'experiments',
                 '08_morphology_ablation.py'),
    "5가지 도메인 적응 시도 코드"
)


# ── 2. 정규화 실험 결과 (4.2절 출처) ────────────────────
print_section("2️⃣ 4.2절 정규화 수치 출처 확인 (41.8/40.1/71.4/62.7)")

# 모든 results.json 검색
results_files = glob.glob(
    os.path.join(base, 'paper', 'experiments', '**', 'results.json'),
    recursive=True
)

print(f"\n발견된 results.json 파일: {len(results_files)}개")
for rf in sorted(results_files):
    print(f"   📄 {os.path.relpath(rf, base)}")

# 정규화 관련 파일들 자세히 보기
target_files = [
    'paper/experiments/05_results/results.json',
    'paper/experiments/06_results/results.json',
    'paper/experiments/07_results/results.json',
]

for tf in target_files:
    show_file(os.path.join(base, tf), "정규화 실험 결과")


# ── 3. VisemStyleAugment 확률 출처 확인 ──────────────────
print_section("3️⃣ VisemStyleAugment 확률 출처 확인 (40%/50%/30%/40%/20%)")

show_file(
    os.path.join(base, 'src', 'morphology.py'),
    "VisemStyleAugment 클래스 정의 가능"
)


# ── 4. 6차원 품질 평가 가중치 전체 확인 ─────────────────
print_section("4️⃣ src/quality.py — 6차원 가중치 전체 확인")

show_file(
    os.path.join(base, 'src', 'quality.py'),
    "품질 평가 가중치"
)


# ── 종합 ─────────────────────────────────────────────────
print_section("📋 검증 후 확인할 것")

print("""
   🎯 4.2절 정규화 수치 (41.8/40.1/71.4/62.7):
      → results.json 또는 실험 코드에 있어야 함
      → 없으면 AI 환각 의심
      
   🎯 Table 2 의사 라벨링(0.7%), 기하학적(29.0%):
      → 08_morphology_ablation.json에 있어야 함
      → 없으면 AI 환각 의심
      
   🎯 VisemStyleAugment 확률 (40%/50%/30%/40%/20%):
      → src/morphology.py의 VisemStyleAugment 클래스에 있어야 함
      → motion_blur_p, noise_p 등의 파라미터로
""")


if __name__ == '__main__':
    pass  # 위의 모든 print 자동 실행