"""
26_video_duration_check.py
─────────────────────────────────
VISEM-Tracking 영상 메타데이터 검증

목적:
  - 논문 3.2절 "180초" 사양 통일 표현의 정확성 검증
  - 모든 영상의 실제 길이, fps, 해상도, 프레임 수 측정
  - 평균/최소/최대/중앙값 통계 산출
  - "180초"가 적절한 표현인지 판단
"""

import os
import sys
import cv2
from collections import defaultdict

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_video_info(video_path):
    """단일 영상의 메타데이터 추출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        'fps':          fps,
        'frame_count':  frame_count,
        'width':        width,
        'height':       height,
        'duration_sec': duration,
    }


def main():
    print("=" * 80)
    print("  VISEM-Tracking 영상 메타데이터 검증")
    print("=" * 80)

    base = (PROJECT_ROOT + r'\data\raw'
            r'\VISEM-Tracking\VISEM_Tracking_Train_v4\Train')

    if not os.path.exists(base):
        print(f"\n❌ 데이터 경로 없음: {base}")
        sys.exit(1)

    # 참가자 폴더 목록 (숫자 폴더만)
    participants = sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.isdigit()
    ], key=lambda x: int(x))

    print(f"\n📁 발견된 참가자 폴더: {len(participants)}개")

    # 영상별 분석
    results = []
    print(f"\n{'=' * 80}")
    print(f"  영상별 메타데이터")
    print(f"{'=' * 80}")
    print(f"\n{'참가자':>6} {'길이(초)':>10} {'fps':>7} "
          f"{'프레임수':>10} {'해상도':>12}")
    print("-" * 55)

    for pid in participants:
        video_path = os.path.join(base, pid, f'{pid}.mp4')

        if not os.path.exists(video_path):
            print(f"{pid:>6} {'영상 없음':>12}")
            continue

        info = get_video_info(video_path)
        if info is None:
            print(f"{pid:>6} {'읽기 실패':>12}")
            continue

        info['participant'] = pid
        results.append(info)

        print(f"{pid:>6} {info['duration_sec']:>9.1f}s "
              f"{info['fps']:>6.1f} {info['frame_count']:>10} "
              f"{info['width']}x{info['height']}")

    if not results:
        print("\n❌ 분석 가능한 영상 없음")
        sys.exit(1)

    # 통계 계산
    durations    = [r['duration_sec'] for r in results]
    fps_values   = [r['fps']          for r in results]
    frame_counts = [r['frame_count']  for r in results]

    avg_dur = sum(durations) / len(durations)
    sorted_d = sorted(durations)
    median = (sorted_d[len(sorted_d) // 2]
              if len(sorted_d) % 2 == 1
              else (sorted_d[len(sorted_d) // 2 - 1]
                    + sorted_d[len(sorted_d) // 2]) / 2)

    print(f"\n{'=' * 80}")
    print(f"  통계 요약 (n={len(results)})")
    print(f"{'=' * 80}")

    print(f"\n📊 영상 길이 (초):")
    print(f"   최소:   {min(durations):>7.1f}s")
    print(f"   최대:   {max(durations):>7.1f}s")
    print(f"   평균:   {avg_dur:>7.1f}s")
    print(f"   중앙값: {median:>7.1f}s")

    print(f"\n📊 fps:")
    print(f"   최소:   {min(fps_values):>7.1f}")
    print(f"   최대:   {max(fps_values):>7.1f}")
    print(f"   평균:   {sum(fps_values)/len(fps_values):>7.1f}")

    print(f"\n📊 프레임 수:")
    print(f"   최소:   {min(frame_counts):>7}")
    print(f"   최대:   {max(frame_counts):>7}")
    print(f"   평균:   {sum(frame_counts)/len(frame_counts):>7.0f}")

    # 해상도 분포
    resolutions = defaultdict(int)
    for r in results:
        key = f"{r['width']}x{r['height']}"
        resolutions[key] += 1

    print(f"\n📊 해상도 분포:")
    for res, count in sorted(resolutions.items(), key=lambda x: -x[1]):
        print(f"   {res}: {count}개")

    # 180초 평가
    print(f"\n{'=' * 80}")
    print(f"  논문 '180초' 표현 검증")
    print(f"{'=' * 80}")

    over_180  = sum(1 for d in durations if d >= 180)
    under_180 = sum(1 for d in durations if d < 180)

    print(f"\n   180초 이상: {over_180}개")
    print(f"   180초 미만: {under_180}개")

    print(f"\n💡 판단:")
    if avg_dur > 200:
        print(f"   평균 {avg_dur:.0f}초 → '180초로 통일(상한 cap)'이 자연스러움")
        print(f"   → 본문 '180초' 표현 OK")
    elif 150 < avg_dur <= 200:
        print(f"   평균 {avg_dur:.0f}초 → '180초'가 자연스러운 사양")
        print(f"   → 본문 '180초' 표현 OK")
    elif 100 <= avg_dur <= 150:
        print(f"   평균 {avg_dur:.0f}초 → '180초'가 약간 어색")
        print(f"   → 실제 길이로 수정 권장 또는 길이 명시 제거")
    else:
        print(f"   평균 {avg_dur:.0f}초 → '180초' 표현 부적절")
        print(f"   → 본문에서 영상 길이 명시 제거 또는 실제 길이로 수정")

    # 결과 파일 저장
    out_dir = PROJECT_ROOT + r'\outputs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'visem_video_metadata.txt')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("VISEM-Tracking 영상 메타데이터 검증\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"참가자 {r['participant']}: "
                    f"{r['duration_sec']:.1f}s, "
                    f"{r['fps']:.1f}fps, "
                    f"{r['frame_count']} frames, "
                    f"{r['width']}x{r['height']}\n")
        f.write(f"\n평균 길이: {avg_dur:.1f}초\n")
        f.write(f"중앙값: {median:.1f}초\n")
        f.write(f"평균 fps: {sum(fps_values)/len(fps_values):.1f}\n")

    print(f"\n📁 결과 저장: {out_path}")
    print(f"\n✅ 완료")


if __name__ == '__main__':
    main()