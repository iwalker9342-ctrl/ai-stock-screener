import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title='AI Momentum Stock Screener', page_icon='📈', layout='wide')
st.title('📈 AI Momentum Stock Screener')
st.caption('Rule-based explainable scoring • Delayed/provider-dependent data • Research only • No orders')

DEFAULT = 'SOFI, PLTR, NVDA, AMD, INTC, RIVN, LCID, ACHR, JOBY, MARA, RIOT, HIMS, RKLB'
with st.sidebar:
    st.header('Screen settings')
    tickers = st.text_area('Ticker universe (comma-separated)', DEFAULT, height=130)
    min_price = st.number_input('Minimum price ($)', min_value=0.01, value=0.50, step=0.25)
    max_price = st.number_input('Maximum price ($)', min_value=0.02, value=20.0, step=1.0)
    min_rvol = st.slider('Minimum relative volume', 0.0, 5.0, 1.5, 0.1)
    min_score = st.slider('Minimum score', 0, 100, 55, 5)
    account = st.number_input('Illustrative account size ($)', min_value=1.0, value=100.0, step=10.0)
    risk_pct = st.slider('Risk per trade (%)', 0.25, 3.0, 1.0, 0.25)
    st.info('The screener only evaluates symbols you enter. It does not scan every listed stock.')

@st.cache_data(ttl=300, show_spinner=False)
def history(symbol):
    df = yf.Ticker(symbol).history(period='3mo', interval='1d', auto_adjust=True)
    return df.dropna(subset=['Close', 'Volume'])

def evaluate(symbol, account, risk_pct):
    d = history(symbol)
    if len(d) < 25:
        return None
    close = d['Close'].astype(float)
    vol = d['Volume'].astype(float)
    price = float(close.iloc[-1])
    avg_vol = float(vol.iloc[-21:-1].mean())
    rvol = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 0
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = float(gains.iloc[-1] / losses.iloc[-1]) if losses.iloc[-1] > 0 else 100
    rsi = 100 - 100 / (1 + rs)
    prior_high = float(d['High'].iloc[-21:-1].max())
    daily_change = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    # Explainable weighted model; not trained machine learning or predictive probability.
    parts = {
        'volume': min(rvol / 3, 1) * 25,
        'trend': (15 if price > ema20 else 0) + (10 if ema9 > ema20 else 0),
        'breakout': 25 if price > prior_high else (12 if price >= prior_high * .98 else 0),
        'momentum': (15 if 50 <= rsi <= 70 else 7 if 45 <= rsi <= 80 else 0),
        'daily_move': (10 if 2 <= daily_change <= 12 else 5 if daily_change > 0 else 0),
    }
    score = round(sum(parts.values()))
    stop = round(min(ema20, price * .96), 3)
    if stop >= price or stop <= 0:
        stop = round(price * .96, 3)
    risk_per_share = price - stop
    qty = max(0, min(int(account // price), int((account * risk_pct / 100) // risk_per_share))) if risk_per_share > 0 else 0
    return {
        'Ticker': symbol, 'Price': round(price, 3), '1D %': round(daily_change, 2),
        'RVOL': round(rvol, 2), 'RSI': round(rsi, 1), 'EMA9': round(ema9, 3),
        'EMA20': round(ema20, 3), 'Prior 20D high': round(prior_high, 3),
        'Score / 100': score, 'Entry trigger': round(max(price, prior_high) * 1.002, 3),
        'Example stop': stop, 'Example 2R target': round(price + 2 * risk_per_share, 3),
        'Max shares': qty, 'Score drivers': ', '.join(k for k,v in parts.items() if v > 0),
        'Last bar': str(d.index[-1].date()),
    }

if max_price <= min_price:
    st.error('Maximum price must exceed minimum price.')
else:
    symbols = list(dict.fromkeys(s.strip().upper() for s in tickers.replace('\n', ',').split(',') if s.strip()))[:60]
    if st.button('Run screener', type='primary'):
        rows, errors = [], []
        with st.spinner('Fetching historical bars and scoring symbols...'):
            for s in symbols:
                try:
                    result = evaluate(s, account, risk_pct)
                    if result is None:
                        errors.append(f'{s}: insufficient data')
                    else:
                        rows.append(result)
                except Exception as exc:
                    errors.append(f'{s}: data unavailable ({type(exc).__name__})')
        if rows:
            all_df = pd.DataFrame(rows)
            filtered = all_df[(all_df['Price'] >= min_price) & (all_df['Price'] <= max_price) & (all_df['RVOL'] >= min_rvol) & (all_df['Score / 100'] >= min_score)]
            filtered = filtered.sort_values('Score / 100', ascending=False)
            st.metric('Matches', len(filtered))
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button('Download CSV', filtered.to_csv(index=False), 'screener_results.csv', 'text/csv')
            with st.expander('All scored symbols'):
                st.dataframe(all_df.sort_values('Score / 100', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning('No data returned. Try again later or change symbols.')
        if errors:
            st.warning('Some symbols could not be scored: ' + '; '.join(errors[:12]))

st.markdown('''### How to interpret the screen
- **Score** is a transparent technical heuristic, **not AI-trained probability** and not a buy recommendation.
- **RVOL** compares the latest daily bar to the prior 20 daily bars; it is **not intraday time-adjusted RVOL**.
- **Entry trigger** is an illustrative breakout confirmation threshold, not a live quote or executable order.
- **Example stop / 2R target** are simple research scenarios based on the latest closing price; gaps and slippage can cause greater losses.
- **Max shares** is limited by buying power and a 1% (configurable) theoretical risk budget; it may be zero for small accounts.
- **No live news sentiment, earnings classification, intraday VWAP, real-time streaming, broker connection, or auto-trading** is included in this starter version.
- Data may be delayed or unavailable. Scores can change materially during a trading session. Penny stocks carry substantial liquidity, volatility, and manipulation risk.
''')


st.markdown("---")
st.subheader("Advanced Screener Settings")

enable_alerts = st.checkbox(
    "Enable breakout alerts",
    value=False
)

if enable_alerts:
    st.info(
        "Breakout monitoring selected. "
        "Automatic notifications are not yet configured."
    )

