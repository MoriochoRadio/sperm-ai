"""
27_verify_paper_numbers.py
─────────────────────────────────
논문 본문 수치 출처 자동 검증 스크립트

검증 대상:
  1. Bootstrap n_resamples 횟수
  2. 4.2절 41.8개/40.1개 출처
  3. 28.6%, 96.0%, 71.4%, 62.7% 등 정규화 관련 수치 출처

방법:
  - paper/experiments 폴더의 모든 .py 파일 자동 스캔
  - results 폴더의 모든 .json 파일 확인
  - 키워드 + 정규식으로 정확한 출처 추적
"""

import os
import re
import json
import glob

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def search_in_python_files(exp_dir):
    """
    모든 .py 파일에서 Bootstrap 및 본문 수치 검색
    """
    print("=" * 80)
    print("  Part 1: Python 실험 스크립트 검색")
    print("=" * 80)

    # 1. Bootstrap 관련 패턴
    print(f"\n{'─' * 80}")
    print(f"  🔍 Bootstrap n_resamples 검색")
    print(f"{'─' * 80}")

    bootstrap_keywords = ['bootstrap', 'resample', 'n_resamples', 'n_bootstrap']

    py_files = sorted(glob.glob(os.path.join(exp_dir, '*.py')))

    bootstrap_found = False
    for fpath in py_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"   ⚠️  읽기 실패: {fname} ({e})")
            continue

        relevant_lines = []
        for i, line in enumerate(lines):
            lower = line.lower()
            if any(kw in lower for kw in bootstrap_keywords):
                relevant_lines.append((i + 1, line.rstrip()))

        if relevant_lines:
            bootstrap_found = True
            print(f"\n📄 {fname}:")
            for lineno, content in relevant_lines:
                print(f"   L{lineno:>4}: {content}")

    if not bootstrap_found:
        print("\n   ❌ Bootstrap 관련 코드 미발견")

    # 2. 본문 수치 검색
    print(f"\n{'─' * 80}")
    print(f"  🔍 본문 핵심 수치 검색 (41.8, 40.1, 28.6, 96.0, 71.4, 62.7)")
    print(f"{'─' * 80}")

    target_numbers = ['41.8', '40.1', '28.6', '96.0', '71.4', '62.7']

    for fpath in py_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        matches = [num for num in target_numbers if num in content]
        if matches:
            print(f"\n📄 {fname}:")
            print(f"   매칭: {matches}")


def search_in_results_files(exp_dir):
    """
    results.json 등 결과 파일에서 수치 검색
    """
    print(f"\n{'=' * 80}")
    print(f"  Part 2: 결과 파일 (results.json, *.txt 등) 검색")
    print(f"{'=' * 80}")

    target_numbers = ['41.8', '40.1', '28.6', '96.0', '71.4', '62.7', '45.9', '44.0']

    # results 폴더 검색
    result_patterns = [
        os.path.join(exp_dir, '*', 'results.json'),
        os.path.join(exp_dir, '*', '*.json'),
        os.path.join(exp_dir, '*.json'),
        os.path.join(exp_dir, '*.txt'),
    ]

    found_files = set()
    for pattern in result_patterns:
        for fpath in glob.glob(pattern):
            found_files.add(fpath)

    if not found_files:
        print("\n   ℹ️  결과 파일 없음")
        return

    for fpath in sorted(found_files):
        rel_path = fpath.replace(exp_dir, '...')
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        matches = [num for num in target_numbers if num in content]
        if matches:
            print(f"\n📄 {rel_path}:")
            print(f"   매칭: {matches}")

            # JSON 파일이면 파싱 시도
            if fpath.endswith('.json'):
                try:
                    data = json.loads(content)
                    print(f"   📊 JSON 내용 일부:")
                    if isinstance(data, dict):
                        for k, v in list(data.items())[:10]:
                            print(f"      {k}: {v}")
                    elif isinstance(data, list) and len(data) > 0:
                        print(f"      리스트 길이: {len(data)}")
                        print(f"      첫 항목: {data[0]}")
                except Exception:
                    pass


def check_specific_files(exp_dir):
    """
    핵심 파일들의 정확한 내용 확인
    """
    print(f"\n{'=' * 80}")
    print(f"  Part 3: 주요 실험 파일 핵심 부분 확인")
    print(f"{'=' * 80}")

    # 검토 대상 파일
    target_files = [
        '05_normalization_components.py',
        '06_specs_vs_clahe.py',
        '07_normalizer_validation.py',
        '15_statistical_validation.py',
    ]

    for tf in target_files:
        fpath = os.path.join(exp_dir, tf)
        if not os.path.exists(fpath):
            print(f"\n📄 {tf}: ❌ 파일 없음")
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"\n📄 {tf}: ⚠️ 읽기 실패 ({e})")
            continue

        print(f"\n📄 {tf}:")
        print(f"   크기: {len(content)} 글자, {content.count(chr(10))} 줄")

        # 출력 부분 (print) 미리보기
        lines = content.split('\n')
        print_blocks = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('print(') and ('%' in stripped or '검출' in stripped):
                print_blocks.append((i + 1, stripped[:100]))

        if print_blocks:
            print(f"   📝 핵심 print 구문 (처음 8개):")
            for lineno, code in print_blocks[:8]:
                print(f"      L{lineno:>4}: {code}")


def main():
    exp_dir = PROJECT_ROOT + r'\paper\experiments'

    print("=" * 80)
    print("  논문 본문 수치 출처 자동 검증")
    print("=" * 80)
    print(f"\n📁 검색 대상: {exp_dir}")

    if not os.path.exists(exp_dir):
        print(f"\n❌ 경로 없음: {exp_dir}")
        return

    # 단계별 검증
    search_in_python_files(exp_dir)
    search_in_results_files(exp_dir)
    check_specific_files(exp_dir)

    # 종합 안내
    print(f"\n{'=' * 80}")
    print(f"  검증 결과 활용 안내")
    print(f"{'=' * 80}")
    print(f"""
   📋 확인 사항:

   1. Bootstrap n_resamples:
      - L번호와 함께 출력된 코드에서 정확한 값 확인
      - 보통 'n_resamples=1000' 같은 형태

   2. 본문 41.8/40.1 출처:
      - 매칭된 파일 확인
      - 가장 가능성 높은 파일: 06_specs_vs_clahe.py
        또는 07_normalizer_validation.py

   3. 결과 파일에 들어있다면:
      - JSON 내용 일부에서 정확한 수치 맥락 확인

   → 결과를 보고 정확한 출처와 수치를 파악하세요!
""")


if __name__ == '__main__':
    main()