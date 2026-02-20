"""
NVDA 실적발표 이벤트 스터디 - Fama-French 3-Factor 모델
==========================================================
- 정상 수익률: α + β₁×Mkt-RF + β₂×SMB + β₃×HML (추정 윈도우)
- 비정상 수익률(AR): 실제 수익률 - FF3 예측 수익률
- 누적 비정상 수익률(CAR): 이벤트 윈도우 내 AR 합산
- 통계 검정: t-test (AR 및 CAR)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

# ── 한글 폰트 설정 (없으면 기본 폰트 사용) ──────────────────────────────────
try:
    plt.rcParams["font.family"] = "NanumGothic"
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════════════════
# 설정값 (여기만 수정하면 다른 종목/이벤트에도 재사용 가능)
# ════════════════════════════════════════════════════════════════════
CONFIG = {
    "ticker"           : "NVDA",
    "event_date"       : "2026-02-26",   # 실적발표일
    "estimation_start" : -250,           # 추정 윈도우 시작 (이벤트일 기준 영업일)
    "estimation_end"   : -11,            # 추정 윈도우 끝
    "event_window_pre" : -5,             # 이벤트 윈도우 시작
    "event_window_post": 5,              # 이벤트 윈도우 끝
    "confidence_level" : 0.95,           # 신뢰수준
    "ff_factor_source" : "famafrench",   # pandas_datareader 소스
    "ff_dataset"       : "F-F_Research_Data_Factors_daily",
}


# ════════════════════════════════════════════════════════════════════
# 1. 데이터 수집
# ════════════════════════════════════════════════════════════════════
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.Series:
    """Yahoo Finance에서 일별 수익률 수집"""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze()
    returns = prices.pct_change().dropna()
    returns.name = ticker
    return returns


def fetch_ff3_factors(start: str, end: str) -> pd.DataFrame:
    """
    Ken French 데이터 라이브러리에서 일별 FF3 팩터 수집
    컬럼: Mkt-RF, SMB, HML, RF  (단위: %)
    """
    ff = web.DataReader(
        CONFIG["ff_dataset"],
        CONFIG["ff_factor_source"],
        start=start,
        end=end,
    )
    ff = ff / 100  # % → 소수 변환
    ff.index = pd.to_datetime(ff.index)
    return ff


def build_dataset(ticker: str, event_date: str,
                  est_start_days: int, evt_post_days: int) -> pd.DataFrame:
    """
    종목 수익률 + FF3 팩터를 합친 분석용 데이터프레임 반환
    """
    evt = pd.Timestamp(event_date)

    # 충분한 데이터 확보를 위해 앞뒤 여유 포함
    dl_start = (evt + timedelta(days=est_start_days * 1.5)).strftime("%Y-%m-%d")
    dl_end   = (evt + timedelta(days=evt_post_days + 30)).strftime("%Y-%m-%d")

    stock = fetch_stock_data(ticker, dl_start, dl_end)
    ff    = fetch_ff3_factors(dl_start, dl_end)

    df = pd.concat([stock, ff], axis=1).dropna()
    df["excess_return"] = df[ticker] - df["RF"]  # 초과 수익률
    return df


# ════════════════════════════════════════════════════════════════════
# 2. 이벤트 인덱스 계산 (영업일 기준)
# ════════════════════════════════════════════════════════════════════
def get_event_indices(df: pd.DataFrame, event_date: str,
                      est_start: int, est_end: int,
                      evt_pre: int, evt_post: int):
    """
    이벤트일을 기준으로 추정 윈도우 / 이벤트 윈도우 인덱스 반환
    """
    evt = pd.Timestamp(event_date)
    dates = df.index

    # 이벤트일 또는 가장 가까운 다음 거래일
    if evt not in dates:
        evt = dates[dates >= evt][0]

    evt_pos = dates.get_loc(evt)

    est_s = max(0, evt_pos + est_start)
    est_e = evt_pos + est_end + 1
    ew_s  = max(0, evt_pos + evt_pre)
    ew_e  = min(len(dates), evt_pos + evt_post + 1)

    return {
        "event_pos"      : evt_pos,
        "event_date_actual": evt,
        "estimation_idx" : slice(est_s, est_e),
        "event_window_idx": slice(ew_s, ew_e),
        "event_window_dates": dates[ew_s:ew_e],
    }


# ════════════════════════════════════════════════════════════════════
# 3. FF3 모델 추정 (OLS 회귀)
# ════════════════════════════════════════════════════════════════════
def estimate_ff3(df: pd.DataFrame, est_idx: slice, ticker: str) -> dict:
    """
    추정 윈도우 데이터로 OLS 회귀:
    excess_return_i = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε
    """
    sub = df.iloc[est_idx]
    y = sub["excess_return"].values
    X = sub[["Mkt-RF", "SMB", "HML"]].values
    X = np.column_stack([np.ones(len(X)), X])  # 상수항 추가

    # OLS: β = (X'X)⁻¹ X'y
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat  = X @ beta
    resid  = y - y_hat
    sse    = np.sum(resid ** 2)
    s2     = sse / (len(y) - len(beta))        # 잔차 분산
    se     = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t_stat = beta / se
    p_val  = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(y) - len(beta)))

    r2 = 1 - sse / np.sum((y - y.mean()) ** 2)

    coef_names = ["α (alpha)", "β₁ Mkt-RF", "β₂ SMB", "β₃ HML"]
    print("\n" + "═" * 60)
    print(f"  FF3 OLS 추정 결과  ({len(sub)}일 추정 윈도우)")
    print("═" * 60)
    print(f"  {'계수':12s}  {'값':>10s}  {'t-stat':>10s}  {'p-value':>10s}")
    print("  " + "-" * 50)
    for name, b, t, p in zip(coef_names, beta, t_stat, p_val):
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {name:12s}  {b:10.6f}  {t:10.4f}  {p:10.4f} {sig}")
    print(f"\n  R² = {r2:.4f}  |  잔차 표준편차 = {np.sqrt(s2):.6f}")
    print("═" * 60)

    return {
        "alpha" : beta[0],
        "betas" : beta[1:],
        "s2"    : s2,
        "r2"    : r2,
        "n_est" : len(sub),
    }


# ════════════════════════════════════════════════════════════════════
# 4. 비정상 수익률(AR) / 누적 비정상 수익률(CAR) 계산
# ════════════════════════════════════════════════════════════════════
def calc_abnormal_returns(df: pd.DataFrame, model: dict,
                          ew_idx: slice, ticker: str) -> pd.DataFrame:
    """
    이벤트 윈도우에서 AR = 실제 초과수익률 - FF3 예측 수익률
    """
    sub = df.iloc[ew_idx].copy()
    factors = sub[["Mkt-RF", "SMB", "HML"]].values
    expected = model["alpha"] + factors @ model["betas"]

    sub["actual"]   = sub["excess_return"].values
    sub["expected"] = expected
    sub["AR"]       = sub["actual"] - sub["expected"]
    sub["CAR"]      = sub["AR"].cumsum()

    # AR 표준오차 (추정 불확실성 + 이벤트 윈도우 분산)
    sub["AR_std"] = np.sqrt(model["s2"])
    sub["CAR_std"] = np.sqrt(
        np.arange(1, len(sub) + 1) * model["s2"]
    )
    return sub


# ════════════════════════════════════════════════════════════════════
# 5. 통계 검정
# ════════════════════════════════════════════════════════════════════
def significance_tests(result: pd.DataFrame, model: dict,
                        confidence: float = 0.95) -> pd.DataFrame:
    """
    각 일자별 AR t-검정 + 이벤트 윈도우 전체 CAR t-검정
    """
    alpha_level = 1 - confidence
    n_est = model["n_est"]
    dof   = n_est - 4  # 파라미터 수(4) 제거

    result = result.copy()
    result["t_AR"]  = result["AR"] / result["AR_std"]
    result["p_AR"]  = 2 * (1 - stats.t.cdf(result["t_AR"].abs(), df=dof))
    result["sig_AR"] = result["p_AR"].apply(
        lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    )

    car_total = result["CAR"].iloc[-1]
    car_std   = result["CAR_std"].iloc[-1]
    t_car     = car_total / car_std
    p_car     = 2 * (1 - stats.t.cdf(abs(t_car), df=dof))

    print("\n" + "═" * 70)
    print("  비정상 수익률(AR) 일별 결과")
    print("═" * 70)
    print(f"  {'날짜':12s}  {'실제(%)':>9s}  {'예측(%)':>9s}  "
          f"{'AR(%)':>9s}  {'t-stat':>8s}  {'p-val':>8s}  {'유의성':>6s}")
    print("  " + "-" * 66)
    for dt, row in result.iterrows():
        marker = " ◀ 이벤트일" if row.name == result.index[result.index.get_loc(
            result.index[result["AR"].abs().argmax()])][0] else ""
        print(
            f"  {str(dt.date()):12s}  {row['actual']*100:9.3f}  "
            f"{row['expected']*100:9.3f}  {row['AR']*100:9.3f}  "
            f"{row['t_AR']:8.4f}  {row['p_AR']:8.4f}  {row['sig_AR']:>6s}"
        )

    print("═" * 70)
    print(f"\n  [ CAR 전체 검정 ]")
    print(f"  CAR ({result.index[0].date()} ~ {result.index[-1].date()}) "
          f"= {car_total*100:.3f}%")
    print(f"  t-stat = {t_car:.4f}  |  p-value = {p_car:.4f}  "
          f"{'(유의)' if p_car < 1-confidence else '(비유의)'}")
    print("═" * 70)

    result.attrs["car_t"]   = t_car
    result.attrs["car_p"]   = p_car
    result.attrs["car_sig"] = p_car < (1 - confidence)
    return result


# ════════════════════════════════════════════════════════════════════
# 6. 시각화
# ════════════════════════════════════════════════════════════════════
def plot_results(result: pd.DataFrame, ticker: str,
                 event_date: str, confidence: float = 0.95) -> None:
    """
    AR 막대그래프 + CAR 누적 그래프를 2패널로 시각화
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"{ticker} 실적발표 이벤트 스터디  (Fama-French 3-Factor)\n"
        f"이벤트일: {event_date}",
        fontsize=14, fontweight="bold", y=0.98
    )

    dates_str = [d.strftime("%m/%d") for d in result.index]
    x = np.arange(len(dates_str))
    evt_idx = len(result) // 2  # 이벤트일 위치 (중앙)

    ci_multiplier = stats.norm.ppf((1 + confidence) / 2)

    # ── 패널 1: AR 막대 ────────────────────────────────────────────
    ax1 = axes[0]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in result["AR"]]
    bars = ax1.bar(x, result["AR"] * 100, color=colors, alpha=0.8, width=0.6)

    # 신뢰구간 에러바
    ci = ci_multiplier * result["AR_std"] * 100
    ax1.errorbar(x, result["AR"] * 100, yerr=ci.values,
                 fmt="none", color="black", capsize=4, linewidth=1)

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(evt_idx, color="navy", linewidth=1.5,
                linestyle=":", label="이벤트일")

    # 유의한 AR에 별표 표시
    for i, (_, row) in enumerate(result.iterrows()):
        if row["sig_AR"]:
            ax1.text(i, row["AR"] * 100 + (0.3 if row["AR"] >= 0 else -0.5),
                     row["sig_AR"], ha="center", fontsize=9, color="black")

    ax1.set_ylabel("비정상 수익률 AR (%)", fontsize=11)
    ax1.set_title("일별 비정상 수익률 (AR)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── 패널 2: CAR 누적 ───────────────────────────────────────────
    ax2 = axes[1]
    car_pct = result["CAR"] * 100
    ci_car  = ci_multiplier * result["CAR_std"] * 100

    ax2.plot(x, car_pct, color="#1f77b4", linewidth=2.5,
             marker="o", markersize=5, label="CAR")
    ax2.fill_between(x,
                     car_pct - ci_car,
                     car_pct + ci_car,
                     alpha=0.15, color="#1f77b4", label=f"{int(confidence*100)}% 신뢰구간")

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axvline(evt_idx, color="navy", linewidth=1.5, linestyle=":", label="이벤트일")

    # 최종 CAR 주석
    final_car = car_pct.iloc[-1]
    ax2.annotate(
        f"최종 CAR\n{final_car:+.2f}%",
        xy=(x[-1], final_car),
        xytext=(x[-1] - 1.5, final_car + (1.5 if final_car >= 0 else -2.5)),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, ha="center"
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(dates_str, rotation=0, fontsize=9)
    ax2.set_ylabel("누적 비정상 수익률 CAR (%)", fontsize=11)
    ax2.set_title("누적 비정상 수익률 (CAR)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = f"{ticker}_event_study_ff3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  차트 저장: {out}")
    plt.show()


# ════════════════════════════════════════════════════════════════════
# 7. 요약 리포트
# ════════════════════════════════════════════════════════════════════
def print_summary(result: pd.DataFrame, model: dict, cfg: dict) -> None:
    evt_row  = result.iloc[len(result) // 2]
    max_ar   = result["AR"].abs().idxmax()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + f"  {cfg['ticker']} 이벤트 스터디 요약 리포트".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  이벤트일       : {cfg['event_date']:<39s}║")
    print(f"║  추정 윈도우    : {cfg['estimation_start']}일 ~ {cfg['estimation_end']}일"
          f"  (총 {model['n_est']}거래일){'':>21s}║")
    print(f"║  이벤트 윈도우  : {cfg['event_window_pre']}일 ~ +{cfg['event_window_post']}일"
          f"  (총 {len(result)}거래일){'':>22s}║")
    print(f"║  FF3 모델 R²    : {model['r2']:.4f}{'':>40s}║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  이벤트일 AR    : {evt_row['AR']*100:+.3f}%  "
          f"(t={evt_row['t_AR']:.3f}, p={evt_row['p_AR']:.4f}) {'':>10s}║")
    print(f"║  전체 CAR       : {result['CAR'].iloc[-1]*100:+.3f}%  "
          f"(t={result.attrs['car_t']:.3f}, p={result.attrs['car_p']:.4f}) {'':>9s}║")
    print(f"║  CAR 유의성     : {'유의 ✓' if result.attrs['car_sig'] else '비유의 ✗'}"
          f"  ({int(cfg['confidence_level']*100)}% 신뢰수준){'':>27s}║")
    print("╚" + "═" * 58 + "╝")


# ════════════════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════════════════
def run_event_study(cfg: dict = CONFIG) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  {cfg['ticker']} 이벤트 스터디 시작")
    print(f"  Fama-French 3-Factor 모델")
    print(f"{'='*60}\n")

    # 1. 데이터 수집
    print("▶ 데이터 수집 중...")
    df = build_dataset(
        cfg["ticker"],
        cfg["event_date"],
        cfg["estimation_start"],
        cfg["event_window_post"],
    )
    print(f"  수집 완료: {len(df)}거래일 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 2. 윈도우 인덱스 계산
    idx = get_event_indices(
        df,
        cfg["event_date"],
        cfg["estimation_start"],
        cfg["estimation_end"],
        cfg["event_window_pre"],
        cfg["event_window_post"],
    )
    print(f"  실제 이벤트일: {idx['event_date_actual'].date()}")

    # 3. FF3 모델 추정
    print("\n▶ FF3 모델 추정 중...")
    model = estimate_ff3(df, idx["estimation_idx"], cfg["ticker"])

    # 4. AR / CAR 계산
    print("\n▶ 비정상 수익률(AR/CAR) 계산 중...")
    result = calc_abnormal_returns(df, model, idx["event_window_idx"], cfg["ticker"])

    # 5. 통계 검정
    print("\n▶ 통계 검정...")
    result = significance_tests(result, model, cfg["confidence_level"])

    # 6. 요약
    print_summary(result, model, cfg)

    # 7. 시각화
    print("\n▶ 차트 생성 중...")
    plot_results(result, cfg["ticker"], cfg["event_date"], cfg["confidence_level"])

    return result


if __name__ == "__main__":
    result = run_event_study(CONFIG)
