import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
from datetime import date, timedelta
import os
import json
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="AI Stock Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Advisor")
st.caption("Five-Year Historical Stock Analysis powered by Gemini AI")


API_KEY = os.getenv("GEMINI_API_KEY")
print(API_KEY)


def calculate_indicators(df):

    df = df.copy()

    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    df["EMA_20"] = df["Close"].ewm(
        span=20, adjust=False
    ).mean()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    middle = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()

    df["Upper_Band"] = middle + (2 * std)
    df["Lower_Band"] = middle - (2 * std)

    # Historical volatility (annualized)
    daily_returns = df["Close"].pct_change()

    df["Volatility"] = (
        daily_returns.rolling(20).std() * np.sqrt(252) * 100
    )

    return df

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, start_date, end_date):

    stock = yf.Ticker(symbol)

    df = stock.history(
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    if df.empty:
        return None

    df.index = pd.to_datetime(df.index)

    return df


def display_stock_chart(df, symbol):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Stock Price",
            line=dict(color="#00BFFF", width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_50"],
            name="SMA 50",
            line=dict(color="orange")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_200"],
            name="SMA 200",
            line=dict(color="red")
        )
    )

    fig.update_layout(
        title=f"{symbol} - Five Year Stock History",
        xaxis_title="Date",
        yaxis_title="Adjusted Stock Price",
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


def gemini_stock_analysis(symbol, analysis_date, indicators, history):

    if not API_KEY:
        return None, "GEMINI_API_KEY environment variable is missing."

    try:

        client = genai.Client(api_key=API_KEY)

        historical_data = history.tail(252).copy()

        historical_data.index = historical_data.index.strftime(
            "%Y-%m-%d"
        )

        historical_data = historical_data.reset_index()


        historical_data = historical_data.rename(
            columns={"Date": "date"}
        )

        prompt = f"""
You are an experienced stock market research analyst.

Analyze the following stock using its five-year historical
data and technical indicators.

STOCK: {symbol}

ANALYSIS DATE: {analysis_date}

CURRENT INDICATORS:
{json.dumps(indicators, indent=2)}

RECENT HISTORICAL DATA:
{historical_data.to_json(orient="records")}

Your responsibilities:

1. Analyze long-term stock trends.
2. Analyze short-term momentum.
3. Evaluate RSI.
4. Evaluate MACD.
5. Analyze SMA 50 and SMA 200.
6. Evaluate volatility.
7. Identify bullish and bearish signals.
8. Explain the major risks.
9. Suggest potential entry conditions.

Return your answer in this JSON format:

{{
    "recommendation": "BUY / HOLD / SELL / AVOID / INSUFFICIENT_DATA",
    "confidence": 0,
    "trend": "Bullish / Bearish / Sideways",
    "risk_level": "Low / Medium / High",
    "current_price": 0,
    "bullish_factors": [],
    "bearish_factors": [],
    "reason": "",
    "entry_strategy": "",
    "stop_loss_consideration": "",
    "final_advice": ""
}}

IMPORTANT:
- Do not invent prices or indicators.
- Use only information available on or before the analysis date.
- Do not guarantee profits.
- Clearly communicate uncertainty.
- Do not treat AI confidence as probability of profit.
- If data is insufficient, recommend INSUFFICIENT_DATA.
- Return valid JSON only.
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        result = json.loads(response.text)

        return result, None

    except Exception as e:
        return None, str(e)


st.sidebar.header("Stock Analysis")

symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="RELIANCE.NS"
).strip().upper()

analysis_date = st.sidebar.date_input(
    "Select Analysis Date",
    value=date.today(),
    min_value=date(2000, 1, 1),
    max_value=date.today()
)

analyze = st.sidebar.button(
    "🔍 Analyze Stock",
    use_container_width=True
)

st.sidebar.info(
    "Example symbols: RELIANCE.NS, TCS.NS, INFY.NS, "
    "AAPL, MSFT, GOOGL"
)


if analyze:

    if not symbol:
        st.error("Please enter a stock symbol.")

    else:

        with st.spinner("Fetching five years of stock data..."):

            start_date = analysis_date - timedelta(days=365 * 5)

            end_date = analysis_date + timedelta(days=1)

            df = fetch_stock_data(
                symbol,
                start_date,
                end_date
            )

        if df is None or df.empty:

            st.error("No stock data found. Check the symbol and date.")

        else:
            df = df[df.index.date <= analysis_date]

            if len(df) < 200:

                st.warning(
                    "Insufficient historical data for reliable "
                    "long-term technical analysis."
                )

            else:

                df = calculate_indicators(df)

                latest = df.iloc[-1]

                if pd.isna(latest["SMA_200"]):
                    st.error("Insufficient data to calculate indicators.")
                    st.stop()

                indicators = {
                    "Close": round(float(latest["Close"]), 2),
                    "RSI": round(float(latest["RSI"]), 2)
                        if pd.notna(latest["RSI"]) else None,
                    "MACD": round(float(latest["MACD"]), 3),
                    "MACD_Signal": round(float(latest["Signal"]), 3),
                    "SMA_50": round(float(latest["SMA_50"]), 2),
                    "SMA_200": round(float(latest["SMA_200"]), 2),
                    "EMA_20": round(float(latest["EMA_20"]), 2),
                    "Volatility": round(float(latest["Volatility"]), 2)
                        if pd.notna(latest["Volatility"]) else None
                }

                st.subheader(f"📊 {symbol} Analysis")

                st.caption(
                    f"Analysis date: {analysis_date} | "
                    f"Data points: {len(df)}"
                )

                
                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Stock Price",
                    f"{indicators['Close']:.2f}"
                )

                col2.metric(
                    "RSI",
                    indicators["RSI"]
                    if indicators["RSI"] is not None else "N/A"
                )

                col3.metric(
                    "SMA 50",
                    f"{indicators['SMA_50']:.2f}"
                )

                col4.metric(
                    "SMA 200",
                    f"{indicators['SMA_200']:.2f}"
                )

                display_stock_chart(df, symbol)

                st.subheader("Technical Indicators")

                st.dataframe(
                    pd.DataFrame(
                        list(indicators.items()),
                        columns=["Indicator", "Value"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )


                st.subheader("🤖 finance  AI Recommendation")

                if (
                    df.index[-1].date() != analysis_date
                    and analysis_date != date.today()
                ):
                    st.warning(
                        "The selected date is not a trading date. "
                        "Results use the latest available trading session "
                        "on or before that date."
                    )

                with st.spinner(
                    "Gemini AI is analyzing five years of stock history..."
                ):

                    result, error = gemini_stock_analysis(
                        symbol,
                        analysis_date,
                        indicators,
                        df
                    )

                if error:

                    st.error(f"Gemini Error: {error}")

                elif result:

                    recommendation = result.get(
                        "recommendation",
                        "INSUFFICIENT_DATA"
                    ).upper()

                    if recommendation == "BUY":
                        st.success("🟢 BUY")

                    elif recommendation == "HOLD":
                        st.info("🟡 HOLD")

                    elif recommendation == "SELL":
                        st.error("🔴 SELL")

                    elif recommendation == "AVOID":
                        st.warning("⚠️ AVOID")

                    else:
                        st.warning("INSUFFICIENT DATA")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "AI Confidence",
                        f"{result.get('confidence', 0)}%"
                    )

                    col2.metric(
                        "Risk Level",
                        result.get("risk_level", "Unknown")
                    )

                    st.write(
                        "**Market Trend:**",
                        result.get("trend", "Unknown")
                    )

                    st.subheader("AI Explanation")

                    st.write(result.get("reason", "No explanation"))

                    st.subheader("🟢 Bullish Factors")

                    for factor in result.get("bullish_factors", []):
                        st.write(f"• {factor}")

                    st.subheader("🔴 Bearish Factors")

                    for factor in result.get("bearish_factors", []):
                        st.write(f"• {factor}")

                    st.subheader("🎯 Entry Strategy")

                    st.write(
                        result.get("entry_strategy", "Not available")
                    )

                    st.subheader("🛡️ Stop Loss Consideration")

                    st.write(
                        result.get(
                            "stop_loss_consideration",
                            "Not available"
                        )
                    )

                    st.subheader("💡 Final AI Advice")

                    st.info(
                        result.get("final_advice", "No advice available")
                    )

                    st.caption(
                        "This is an AI-generated research opinion, "
                        "not personalized financial advice or a "
                        "guarantee of investment returns."
                    )