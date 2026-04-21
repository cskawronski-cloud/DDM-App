from io import StringIO

import pandas as pd
import requests
import streamlit as st
import yfinance as yf


FRED_SERIES = "DGS10"
MARKET_TICKER = "^GSPC"


def get_fred_latest(series_id: str) -> float:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return float(df[series_id].dropna().iloc[-1]) / 100


def get_close_series(ticker: str, period: str, interval: str = "1mo") -> pd.Series:
    history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    close = history["Close"].dropna()
    if close.empty:
        raise ValueError(f"No price history returned for {ticker}.")
    return close


def get_beta(stock_ticker: str, market_ticker: str, period: str = "5y") -> float:
    stock_returns = get_close_series(stock_ticker, period).pct_change().dropna()
    market_returns = get_close_series(market_ticker, period).pct_change().dropna()
    returns = pd.concat([stock_returns, market_returns], axis=1, join="inner")
    returns.columns = ["stock", "market"]
    if returns.empty:
        raise ValueError("Not enough return history to estimate beta.")
    return float(returns["stock"].cov(returns["market"]) / returns["market"].var())


def get_market_return(market_ticker: str, period: str = "10y") -> float:
    close = get_close_series(market_ticker, period)
    years = (close.index[-1] - close.index[0]).days / 365.25
    return float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1)


def get_annual_dividend(stock: yf.Ticker) -> tuple[float, pd.Series]:
    history = stock.history(period="2y", auto_adjust=False, actions=True)
    dividends = history["Dividends"]
    trailing_four = dividends[dividends > 0].tail(4)
    if trailing_four.empty:
        raise ValueError("No recent dividends found for this company.")
    annual_dividend = float(trailing_four.sum())
    return annual_dividend, trailing_four


def get_dividend_profile(stock: yf.Ticker) -> dict:
    history = stock.history(period="10y", auto_adjust=False, actions=True)
    dividends = history["Dividends"]
    dividends = dividends[dividends > 0]
    annual_dividends = dividends.groupby(dividends.index.year).sum()

    if annual_dividends.empty:
        raise ValueError("No dividend history found for this company.")

    recent_annual_dividends = annual_dividends.tail(5)
    growth_series = recent_annual_dividends.pct_change().dropna()
    growth_volatility = float(growth_series.std()) if not growth_series.empty else 0.0
    dividend_cut_detected = bool((recent_annual_dividends.diff().dropna() < 0).any())

    return {
        "dividend_history": dividends,
        "annual_dividends": annual_dividends,
        "recent_annual_dividends": recent_annual_dividends,
        "dividend_growth_volatility": growth_volatility,
        "dividend_cut_detected": dividend_cut_detected,
        "years_with_dividends": int(len(annual_dividends)),
    }


def get_sustainable_growth_inputs(stock: yf.Ticker) -> dict:
    financials = stock.financials
    balance_sheet = stock.balance_sheet

    annual_dividend, trailing_dividends = get_annual_dividend(stock)

    net_income_series = financials.loc[
        "Net Income From Continuing Operation Net Minority Interest"
    ].dropna().sort_index()
    diluted_eps_series = financials.loc["Diluted EPS"].dropna().sort_index()
    equity_series = balance_sheet.loc["Stockholders Equity"].dropna().sort_index()

    average_equity_series = (equity_series + equity_series.shift(1)) / 2
    average_equity_series = average_equity_series.dropna()
    aligned_net_income = net_income_series.loc[average_equity_series.index]
    roe_series = aligned_net_income / average_equity_series

    diluted_eps = float(diluted_eps_series.iloc[-1])
    latest_roe = float(roe_series.iloc[-1])
    normalized_roe = float(roe_series.tail(3).mean())
    payout_ratio = annual_dividend / diluted_eps
    normalized_payout_ratio = float(min(max(payout_ratio, 0.0), 1.0))
    retention_ratio = 1 - payout_ratio
    normalized_retention_ratio = 1 - normalized_payout_ratio
    sustainable_growth = latest_roe * retention_ratio
    normalized_growth = normalized_roe * normalized_retention_ratio

    return {
        "roe": latest_roe,
        "normalized_roe": normalized_roe,
        "roe_series": roe_series,
        "payout_ratio": payout_ratio,
        "normalized_payout_ratio": normalized_payout_ratio,
        "retention_ratio": retention_ratio,
        "normalized_retention_ratio": normalized_retention_ratio,
        "sustainable_growth": sustainable_growth,
        "normalized_growth": normalized_growth,
        "annual_dividend": annual_dividend,
        "diluted_eps": diluted_eps,
        "trailing_dividends": trailing_dividends,
    }


def classify_company(growth_inputs: dict, dividend_profile: dict) -> dict:
    payout_ratio = growth_inputs["payout_ratio"]
    normalized_growth = growth_inputs["normalized_growth"]
    years_with_dividends = dividend_profile["years_with_dividends"]
    dividend_cut_detected = dividend_profile["dividend_cut_detected"]
    growth_volatility = dividend_profile["dividend_growth_volatility"]

    if years_with_dividends < 5:
        profile = "Limited dividend history"
        description = "The company has too little dividend history for a high-confidence DDM."
    elif payout_ratio > 1.0:
        profile = "Stressed payer"
        description = "Dividends currently exceed diluted EPS, so the app normalizes payout and growth."
    elif dividend_cut_detected or growth_volatility > 0.2:
        profile = "Mature cyclical payer"
        description = "Dividend behavior looks uneven, so the app smooths assumptions and shortens stage 1."
    elif normalized_growth > 0.10:
        profile = "Stable grower"
        description = "Dividend coverage and profitability support a stronger stage-1 growth period."
    else:
        profile = "Mature stable payer"
        description = "Dividend behavior looks steady, so the app uses a balanced two-stage setup."

    return {
        "profile": profile,
        "description": description,
    }


def build_adjustments(
    growth_inputs: dict,
    dividend_profile: dict,
    profile_info: dict,
    requested_stage_1_years: int,
    requested_terminal_growth_cap: float,
    risk_free_rate: float,
) -> dict:
    adjustments = []
    applied_stage_1_years = requested_stage_1_years
    applied_terminal_growth_cap = requested_terminal_growth_cap
    applied_growth_stage_1 = growth_inputs["sustainable_growth"]

    profile = profile_info["profile"]

    if profile == "Limited dividend history":
        applied_stage_1_years = min(requested_stage_1_years, 3)
        applied_growth_stage_1 = min(max(growth_inputs["normalized_growth"], 0.0), 0.06)
        applied_terminal_growth_cap = min(requested_terminal_growth_cap, 0.02)
        adjustments.append("Shortened stage 1 to 3 years because dividend history is limited.")
        adjustments.append("Used normalized growth capped at 6% for a conservative stage-1 forecast.")
        adjustments.append("Reduced terminal growth cap to 2%.")
    elif profile == "Stressed payer":
        applied_stage_1_years = min(requested_stage_1_years, 4)
        applied_growth_stage_1 = min(max(growth_inputs["normalized_growth"], 0.0), 0.05)
        applied_terminal_growth_cap = min(requested_terminal_growth_cap, 0.025)
        adjustments.append("Used normalized ROE and capped payout at 100% because payout exceeds EPS.")
        adjustments.append("Capped stage-1 growth at 5% and shortened stage 1 to 4 years.")
        adjustments.append("Reduced terminal growth cap to 2.5%.")
    elif profile == "Mature cyclical payer":
        applied_stage_1_years = min(requested_stage_1_years, 4)
        applied_growth_stage_1 = min(max(growth_inputs["normalized_growth"], 0.0), 0.08)
        adjustments.append("Used normalized growth to smooth cyclical swings.")
        adjustments.append("Shortened stage 1 to 4 years and capped stage-1 growth at 8%.")
    elif profile == "Stable grower":
        applied_stage_1_years = max(requested_stage_1_years, 5)
        applied_growth_stage_1 = min(growth_inputs["normalized_growth"], 0.12)
        adjustments.append("Used normalized ROE x retention to avoid overreacting to one-year noise.")
        adjustments.append("Allowed a longer stage 1 and capped stage-1 growth at 12%.")
    else:
        applied_growth_stage_1 = min(growth_inputs["normalized_growth"], 0.09)
        adjustments.append("Used normalized ROE x retention for a steadier base-case estimate.")
        adjustments.append("Capped stage-1 growth at 9% for a mature dividend payer.")

    growth_stage_2 = min(applied_terminal_growth_cap, risk_free_rate)
    adjustments.append(
        f"Terminal growth set to the lower of the cap and the risk-free rate, which yields {growth_stage_2:.2%}."
    )

    return {
        "applied_stage_1_years": applied_stage_1_years,
        "applied_terminal_growth_cap": applied_terminal_growth_cap,
        "applied_growth_stage_1": applied_growth_stage_1,
        "applied_growth_stage_2": growth_stage_2,
        "adjustments": adjustments,
    }


def two_stage_ddm(
    dividend_0: float,
    required_return: float,
    growth_stage_1: float,
    growth_stage_2: float,
    years_stage_1: int,
) -> tuple[float, pd.DataFrame, float, float]:
    stage_rows = []
    pv_stage_1 = 0.0

    for year in range(1, years_stage_1 + 1):
        dividend_t = dividend_0 * ((1 + growth_stage_1) ** year)
        pv_dividend_t = dividend_t / ((1 + required_return) ** year)
        pv_stage_1 += pv_dividend_t
        stage_rows.append(
            {
                "Year": year,
                "Dividend": dividend_t,
                "Present Value": pv_dividend_t,
            }
        )

    dividend_n = stage_rows[-1]["Dividend"]
    dividend_n_plus_1 = dividend_n * (1 + growth_stage_2)
    terminal_value = dividend_n_plus_1 / (required_return - growth_stage_2)
    pv_terminal_value = terminal_value / ((1 + required_return) ** years_stage_1)
    intrinsic_value = pv_stage_1 + pv_terminal_value

    return intrinsic_value, pd.DataFrame(stage_rows), terminal_value, pv_terminal_value


def run_valuation(
    ticker: str,
    stage_1_years: int,
    terminal_growth_cap: float,
    auto_adjust: bool,
) -> dict:
    stock = yf.Ticker(ticker)
    price = float(stock.fast_info["lastPrice"])

    risk_free_rate = get_fred_latest(FRED_SERIES)
    beta = get_beta(ticker, MARKET_TICKER, "5y")
    expected_market_return = get_market_return(MARKET_TICKER, "10y")
    required_return = risk_free_rate + beta * (expected_market_return - risk_free_rate)

    growth_inputs = get_sustainable_growth_inputs(stock)
    dividend_profile = get_dividend_profile(stock)
    profile_info = classify_company(growth_inputs, dividend_profile)

    if auto_adjust:
        adjustment_info = build_adjustments(
            growth_inputs=growth_inputs,
            dividend_profile=dividend_profile,
            profile_info=profile_info,
            requested_stage_1_years=stage_1_years,
            requested_terminal_growth_cap=terminal_growth_cap,
            risk_free_rate=risk_free_rate,
        )
        growth_stage_1 = adjustment_info["applied_growth_stage_1"]
        growth_stage_2 = adjustment_info["applied_growth_stage_2"]
        applied_stage_1_years = adjustment_info["applied_stage_1_years"]
    else:
        growth_stage_1 = growth_inputs["sustainable_growth"]
        growth_stage_2 = min(terminal_growth_cap, risk_free_rate)
        applied_stage_1_years = stage_1_years
        adjustment_info = {
            "applied_stage_1_years": applied_stage_1_years,
            "applied_terminal_growth_cap": terminal_growth_cap,
            "applied_growth_stage_1": growth_stage_1,
            "applied_growth_stage_2": growth_stage_2,
            "adjustments": ["Automatic company-specific adjustments are turned off."],
        }

    if required_return <= growth_stage_2:
        raise ValueError("Required return must be greater than terminal growth.")

    intrinsic_value, forecast_df, terminal_value, pv_terminal_value = two_stage_ddm(
        dividend_0=growth_inputs["annual_dividend"],
        required_return=required_return,
        growth_stage_1=growth_stage_1,
        growth_stage_2=growth_stage_2,
        years_stage_1=applied_stage_1_years,
    )

    margin_of_safety = intrinsic_value / price - 1

    summary = pd.DataFrame(
        [
            ("Current price", f"${price:,.2f}"),
            ("Annual dividend (D0)", f"${growth_inputs['annual_dividend']:,.2f}"),
            ("Diluted EPS", f"${growth_inputs['diluted_eps']:,.2f}"),
            ("Risk-free rate (FRED DGS10)", f"{risk_free_rate:.2%}"),
            ("Beta (5Y monthly)", f"{beta:.3f}"),
            ("Expected market return (10Y S&P 500 CAGR)", f"{expected_market_return:.2%}"),
            ("Required return via CAPM", f"{required_return:.2%}"),
            ("ROE", f"{growth_inputs['roe']:.2%}"),
            ("Normalized ROE", f"{growth_inputs['normalized_roe']:.2%}"),
            ("Payout ratio", f"{growth_inputs['payout_ratio']:.2%}"),
            ("Normalized payout ratio", f"{growth_inputs['normalized_payout_ratio']:.2%}"),
            ("Retention ratio (b)", f"{growth_inputs['retention_ratio']:.2%}"),
            ("Normalized retention ratio", f"{growth_inputs['normalized_retention_ratio']:.2%}"),
            ("Raw stage 1 growth (ROE x b)", f"{growth_inputs['sustainable_growth']:.2%}"),
            ("Applied stage 1 growth", f"{growth_stage_1:.2%}"),
            ("Stage 2 terminal growth (g2)", f"{growth_stage_2:.2%}"),
            ("Stage 1 length", f"{applied_stage_1_years} years"),
            ("Terminal value at end of stage 1", f"${terminal_value:,.2f}"),
            ("Present value of terminal value", f"${pv_terminal_value:,.2f}"),
            ("Intrinsic value from two-stage DDM", f"${intrinsic_value:,.2f}"),
            ("Margin of safety", f"{margin_of_safety:.2%}"),
        ],
        columns=["Metric", "Value"],
    )

    return {
        "ticker": ticker,
        "summary": summary,
        "forecast_df": forecast_df,
        "trailing_dividends": growth_inputs["trailing_dividends"],
        "profile_info": profile_info,
        "adjustment_info": adjustment_info,
        "dividend_profile": dividend_profile,
    }


st.set_page_config(page_title="DDM Valuation App", page_icon="📈", layout="wide")

st.title("Dividend Discount Model App")
st.write(
    "Enter a dividend-paying company ticker to run the same two-stage DDM used in the notebooks: "
    "CAPM for required return, `ROE x b` for stage-1 growth, and a capped terminal growth rate."
)

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="IBM").strip().upper()
    stage_1_years = st.number_input("Stage 1 years", min_value=1, max_value=15, value=5, step=1)
    terminal_growth_cap_pct = st.number_input(
        "Terminal growth cap (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.25,
    )
    auto_adjust = st.checkbox("Auto-adjust assumptions by company profile", value=True)
    run_button = st.button("Run valuation", type="primary")


if run_button:
    try:
        result = run_valuation(
            ticker=ticker,
            stage_1_years=int(stage_1_years),
            terminal_growth_cap=terminal_growth_cap_pct / 100,
            auto_adjust=auto_adjust,
        )

        st.subheader("Detected Company Profile")
        st.write(
            f"**{result['profile_info']['profile']}**: {result['profile_info']['description']}"
        )
        st.write("**Adjustments applied:**")
        for adjustment in result["adjustment_info"]["adjustments"]:
            st.write(f"- {adjustment}")

        left, right = st.columns([1, 1])
        with left:
            st.subheader(f"{result['ticker']} Summary")
            st.dataframe(result["summary"], use_container_width=True, hide_index=True)

        with right:
            forecast_display = result["forecast_df"].copy()
            forecast_display["Dividend"] = forecast_display["Dividend"].map(lambda x: f"${x:,.2f}")
            forecast_display["Present Value"] = forecast_display["Present Value"].map(lambda x: f"${x:,.2f}")
            st.subheader("Stage 1 Dividend Forecast")
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)

        st.subheader("Dividend Stability Checks")
        dividend_checks = pd.DataFrame(
            [
                ("Years with dividends", result["dividend_profile"]["years_with_dividends"]),
                (
                    "Dividend cut detected in recent annual data",
                    "Yes" if result["dividend_profile"]["dividend_cut_detected"] else "No",
                ),
                (
                    "Recent dividend growth volatility",
                    f"{result['dividend_profile']['dividend_growth_volatility']:.2%}",
                ),
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(dividend_checks, use_container_width=True, hide_index=True)

        st.subheader("Trailing Four Dividends Used for D0")
        dividend_display = result["trailing_dividends"].reset_index()
        dividend_display.columns = ["Date", "Dividend"]
        dividend_display["Date"] = dividend_display["Date"].astype(str)
        dividend_display["Dividend"] = dividend_display["Dividend"].map(lambda x: f"${x:,.2f}")
        st.dataframe(dividend_display, use_container_width=True, hide_index=True)

        st.caption(
            "Data sources: Yahoo Finance for price, dividends, and financial statements; "
            "FRED DGS10 for the risk-free rate."
        )
    except Exception as exc:
        st.error(f"Could not run valuation for {ticker}: {exc}")
else:
    st.info("Choose a ticker in the sidebar and click `Run valuation`.")
