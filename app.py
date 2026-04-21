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


def get_sustainable_growth_inputs(stock: yf.Ticker) -> dict:
    financials = stock.financials
    balance_sheet = stock.balance_sheet

    net_income = float(
        financials.loc["Net Income From Continuing Operation Net Minority Interest"].dropna().iloc[0]
    )
    diluted_eps = float(financials.loc["Diluted EPS"].dropna().iloc[0])

    equity_series = balance_sheet.loc["Stockholders Equity"].dropna()
    current_equity = float(equity_series.iloc[0])
    prior_equity = float(equity_series.iloc[1])
    average_equity = (current_equity + prior_equity) / 2

    annual_dividend, trailing_dividends = get_annual_dividend(stock)
    roe = net_income / average_equity
    payout_ratio = annual_dividend / diluted_eps
    retention_ratio = 1 - payout_ratio
    sustainable_growth = roe * retention_ratio

    return {
        "roe": roe,
        "payout_ratio": payout_ratio,
        "retention_ratio": retention_ratio,
        "sustainable_growth": sustainable_growth,
        "annual_dividend": annual_dividend,
        "diluted_eps": diluted_eps,
        "trailing_dividends": trailing_dividends,
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


def run_valuation(ticker: str, stage_1_years: int, terminal_growth_cap: float) -> dict:
    stock = yf.Ticker(ticker)
    price = float(stock.fast_info["lastPrice"])

    risk_free_rate = get_fred_latest(FRED_SERIES)
    beta = get_beta(ticker, MARKET_TICKER, "5y")
    expected_market_return = get_market_return(MARKET_TICKER, "10y")
    required_return = risk_free_rate + beta * (expected_market_return - risk_free_rate)

    growth_inputs = get_sustainable_growth_inputs(stock)
    growth_stage_1 = growth_inputs["sustainable_growth"]
    growth_stage_2 = min(terminal_growth_cap, risk_free_rate)

    if required_return <= growth_stage_2:
        raise ValueError("Required return must be greater than terminal growth.")

    intrinsic_value, forecast_df, terminal_value, pv_terminal_value = two_stage_ddm(
        dividend_0=growth_inputs["annual_dividend"],
        required_return=required_return,
        growth_stage_1=growth_stage_1,
        growth_stage_2=growth_stage_2,
        years_stage_1=stage_1_years,
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
            ("Payout ratio", f"{growth_inputs['payout_ratio']:.2%}"),
            ("Retention ratio (b)", f"{growth_inputs['retention_ratio']:.2%}"),
            ("Stage 1 growth (g1 = ROE x b)", f"{growth_stage_1:.2%}"),
            ("Stage 2 terminal growth (g2)", f"{growth_stage_2:.2%}"),
            ("Stage 1 length", f"{stage_1_years} years"),
            ("Terminal value at end of year 5", f"${terminal_value:,.2f}"),
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
    run_button = st.button("Run valuation", type="primary")


if run_button:
    try:
        result = run_valuation(
            ticker=ticker,
            stage_1_years=int(stage_1_years),
            terminal_growth_cap=terminal_growth_cap_pct / 100,
        )

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
