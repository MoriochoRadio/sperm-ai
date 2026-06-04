"""
29_compute_ci_methods.py
─────────────────────────────────
v3 4명 데이터에 대해 다양한 95% CI 방법 계산
본문의 [13.2%, 19.4%]가 어떤 방법에서 나왔는지 추적
"""

import numpy as np
from scipy import stats


def main():
    # v3 4명 데이터 (코드에서 확인된 정확한 값)
    v3_rates = [14.6, 8.8, 11.9, 29.9]
    mean     = np.mean(v3_rates)
    n        = len(v3_rates)

    print("=" * 80)
    print("  본문 'Bootstrap 95% CI [13.2%, 19.4%]' 출처 추적")
    print("=" * 80)

    print(f"\n📊 데이터:")
    print(f"   v3 참가자별 정상률: {v3_rates}")
    print(f"   평균: {mean:.4f}")
    print(f"   표본 크기: n={n}")
    print(f"\n📋 본문 주장: Bootstrap 95% CI [13.2%, 19.4%]")
    print(f"   폭: 19.4 - 13.2 = 6.2")

    target_low, target_high = 13.2, 19.4
    tolerance = 0.5

    def check_match(low, high, name):
        match_low  = abs(low  - target_low)  < tolerance
        match_high = abs(high - target_high) < tolerance
        if match_low and match_high:
            return "✅ 본문 일치!"
        else:
            return f"❌ 본문과 다름 (차이: low {low-target_low:+.2f}, high {high-target_high:+.2f})"

    # ── 방법 1: t-distribution CI ────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  방법 1: t-distribution CI (코드 실제 구현)")
    print(f"{'─' * 80}")

    sem = stats.sem(v3_rates)
    h   = sem * stats.t.ppf(0.975, n - 1)
    ci  = (mean - h, mean + h)
    print(f"   SEM:        {sem:.4f}")
    print(f"   t(0.975,3): {stats.t.ppf(0.975, n - 1):.4f}")
    print(f"   margin:     {h:.4f}")
    print(f"   CI:         [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"   {check_match(ci[0], ci[1], 't-CI')}")

    # ── 방법 2: Bootstrap (B=1,000, percentile) ──────────
    print(f"\n{'─' * 80}")
    print(f"  방법 2: Bootstrap Percentile CI (B=1,000)")
    print(f"{'─' * 80}")

    for seed in [42, 0, 1, 2025]:
        np.random.seed(seed)
        boot_means = [
            np.mean(np.random.choice(v3_rates, size=n, replace=True))
            for _ in range(1000)
        ]
        ci = np.percentile(boot_means, [2.5, 97.5])
        print(f"   seed={seed:>4}: CI [{ci[0]:.2f}, {ci[1]:.2f}]  "
              f"{check_match(ci[0], ci[1], 'Bootstrap')}")

    # ── 방법 3: Bootstrap (B=10,000) ─────────────────────
    print(f"\n{'─' * 80}")
    print(f"  방법 3: Bootstrap Percentile CI (B=10,000)")
    print(f"{'─' * 80}")

    np.random.seed(42)
    boot_means = [
        np.mean(np.random.choice(v3_rates, size=n, replace=True))
        for _ in range(10000)
    ]
    ci = np.percentile(boot_means, [2.5, 97.5])
    print(f"   seed=42: CI [{ci[0]:.2f}, {ci[1]:.2f}]  "
          f"{check_match(ci[0], ci[1], 'Bootstrap10k')}")

    # ── 방법 4: scipy.stats.bootstrap (BCa) ──────────────
    print(f"\n{'─' * 80}")
    print(f"  방법 4: scipy.stats.bootstrap (BCa method)")
    print(f"{'─' * 80}")

    try:
        for method in ['percentile', 'basic', 'BCa']:
            res = stats.bootstrap(
                (v3_rates,), np.mean,
                confidence_level=0.95,
                n_resamples=9999,
                method=method,
                random_state=42)
            ci = res.confidence_interval
            print(f"   method={method:>11}: CI [{ci.low:.2f}, {ci.high:.2f}]  "
                  f"{check_match(ci.low, ci.high, f'scipy-{method}')}")
    except Exception as e:
        print(f"   ⚠️ 오류: {e}")

    # ── 방법 5: Normal approximation ─────────────────────
    print(f"\n{'─' * 80}")
    print(f"  방법 5: Normal approximation (z=1.96)")
    print(f"{'─' * 80}")

    ci = stats.norm.interval(0.95, loc=mean, scale=sem)
    print(f"   CI: [{ci[0]:.2f}, {ci[1]:.2f}]  "
          f"{check_match(ci[0], ci[1], 'Normal')}")

    # ── 방법 6: 크롭 단위 binomial CI 가정 ──────────────
    print(f"\n{'─' * 80}")
    print(f"  방법 6: 크롭 단위 Wald CI (n=1,824 가정)")
    print(f"{'─' * 80}")

    n_crops = 1824
    p_hat   = 0.163
    se_bin  = np.sqrt(p_hat * (1 - p_hat) / n_crops)
    z       = 1.96
    ci      = (100 * (p_hat - z * se_bin), 100 * (p_hat + z * se_bin))
    print(f"   p_hat:    {p_hat:.4f}")
    print(f"   SE:       {se_bin:.4f}")
    print(f"   CI:       [{ci[0]:.2f}%, {ci[1]:.2f}%]  "
          f"{check_match(ci[0], ci[1], 'Binomial')}")

    # ── 종합 결론 ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  종합 결론")
    print(f"{'=' * 80}")
    print(f"""
   📌 본문 주장 CI: [13.2%, 19.4%]
   
   ✅ 일치한 방법이 있으면:
      → 해당 방법명으로 본문 수정 (정직성)
   
   ❌ 어떤 방법도 일치 안 하면:
      → 본문 CI 값의 출처 불명 (학술적 위험)
      → 실제 코드로 다시 계산해서 본문 업데이트 필요
""")


if __name__ == '__main__':
    main()