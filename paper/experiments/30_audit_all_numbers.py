"""
30_audit_all_numbers.py
─────────────────────────────────
논문 본문의 모든 핵심 수치 종합 검증
AI 환각 의심 수치 추적
"""

import os
import re
import json
import glob

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 본문의 모든 핵심 수치 (검증 대상)
PAPER_NUMBERS = {
    # 4.2절 정규화 수치
    '41.8':  '4.2절: 원본 검출 수 (개)',
    '40.1':  '4.2절: 사양통일 검출 수 (개)',
    '71.4':  '4.2절: CLAHE 추가 유지율 (%)',
    '62.7':  '4.2절: 그레이스케일 유지율 (%)',

    # Table 2 도메인 적응 수치
    '0.7':   'Table 2: 의사 라벨링 정상률 (%)',
    '29.0':  'Table 2: 기하학적 측정 정상률 (%)',
    '28.4':  'Table 2: TTA 정상률 (%)',
    '0.9':   'Table 2: 앙상블 정상률 (%)',

    # 3.3절 VisemStyleAugment 확률
    '0.40':  '모션 블러/CLAHE 확률 (40%)',
    '0.50':  '가우시안 노이즈 확률 (50%)',
    '0.30':  '저해상도 확률 (30%)',
    '0.20':  '명암 반전 확률 (20%)',

    # 3.2절 품질 평가
    '0.05':  '품질 평가 최소 가중치',
    '0.25':  '품질 평가 최대 가중치',
}

# 검색 경로
search_paths = [
    PROJECT_ROOT + r'\paper\experiments',
    PROJECT_ROOT + r'\src',
    PROJECT_ROOT + r'\notebooks',
]


def search_number_in_file(filepath, number):
    """파일에서 특정 숫자 검색"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []

    matches = []
    lines = content.split('\n')

    # 정확한 숫자 매칭 (앞뒤가 숫자가 아닌 경우)
    pattern = rf'(?<![0-9.]){re.escape(number)}(?![0-9])'

    for i, line in enumerate(lines):
        if re.search(pattern, line):
            matches.append((i + 1, line.strip()))

    return matches


def main():
    print("=" * 80)
    print("  논문 본문 수치 종합 검증")
    print("=" * 80)

    # 모든 파일 수집
    all_files = []
    for path in search_paths:
        if not os.path.exists(path):
            continue
        for ext in ['*.py', '*.json', '*.txt', '*.ipynb']:
            all_files.extend(glob.glob(os.path.join(path, '**', ext),
                                       recursive=True))

    print(f"\n📁 검색 대상 파일: {len(all_files)}개")
    print(f"   경로: paper/experiments, src, notebooks")

    # 각 수치별 검색
    results = {}
    for num, description in PAPER_NUMBERS.items():
        print(f"\n{'─' * 80}")
        print(f"  🔍 검색: {num}")
        print(f"      ({description})")
        print(f"{'─' * 80}")

        found_count = 0
        for fpath in all_files:
            matches = search_number_in_file(fpath, num)
            if matches:
                rel = os.path.relpath(fpath, PROJECT_ROOT)
                print(f"\n   📄 {rel}")
                for lineno, content in matches[:3]:  # 처음 3개만
                    print(f"      L{lineno:>4}: {content[:80]}")
                if len(matches) > 3:
                    print(f"      ... 외 {len(matches) - 3}개")
                found_count += len(matches)

        if found_count == 0:
            print(f"\n   ⚠️ 검색 결과 없음 — AI 환각 가능성!")

        results[num] = found_count

    # 종합 결과
    print(f"\n{'=' * 80}")
    print(f"  📊 종합 결과")
    print(f"{'=' * 80}\n")

    print(f"   {'수치':<10} {'설명':<40} {'발견':>10}")
    print(f"   {'-' * 70}")

    suspicious = []
    for num, desc in PAPER_NUMBERS.items():
        count = results[num]
        status = "✅" if count > 0 else "⚠️"
        print(f"   {num:<10} {desc[:38]:<40} {status} {count:>3}회")
        if count == 0:
            suspicious.append((num, desc))

    if suspicious:
        print(f"\n{'─' * 80}")
        print(f"  🚨 검증 불가 수치 (AI 환각 의심)")
        print(f"{'─' * 80}")
        for num, desc in suspicious:
            print(f"   ⚠️ {num}: {desc}")
        print(f"\n   → 이 수치들의 출처를 별도 확인 필요")
    else:
        print(f"\n   ✅ 모든 수치 검증 완료 — AI 환각 추가 의심 없음")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()