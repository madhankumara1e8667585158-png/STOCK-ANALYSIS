# Gemini Stock Market Analyzer

An intelligent, hybrid stock market analysis tool that combines quantitative financial metrics (via `yfinance`) with the reasoning capabilities of Google's **Gemini API** (`google-genai`). This application fetches recent stock trends, calculates basic technical indicators, and leverages Gemini to produce structured, JSON-formatted trading signals and rationales.

---

## Features

- **Automated Data Fetching:** Pulls historical Open, High, Low, Close, and Volume (OHLCV) data directly via `yfinance`.
- **Technical Indicator Computation:** Calculates metrics like Simple Moving Averages (SMA) and percentage changes using `pandas`.
- **Structured AI Insights:** Uses the official `google-genai` SDK to evaluate market data and guarantees clean JSON-formatted outputs (`BUY`, `SELL`, or `HOLD` signals with confidence scores).
- **Secure Configuration:** Uses environment variables (`.env`) to keep API keys safe.

---

## Project Structure

```text
gemini-stock-analyzer/
│
├── .env                  # Environment variables (API keys)
├── requirements.txt      # Project dependencies
├── Stock_an.py               # Core application logic
└── README.md             # Project documentation

## to run

streamlit run C:\stock_analysis\Stock_an.py