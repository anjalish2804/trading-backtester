import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Backtrackr",
    page_icon="📈",
    layout="wide"
)

# ── Title ───────────────────────────────────────────────────
st.title("📈 Backtrackr")
st.markdown("*Compare trading strategies on any stock, any timeframe.*")
st.divider()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    
    # Stock input
    # Market selector
    market = st.selectbox(
        "Market",
        options=["US", "India (NSE)", "Singapore", "Hong Kong"],
    )

# Suffix mapping
    suffix_map = {
        "US":          "",
        "India (NSE)": ".NS",
        "Singapore":   ".SI",
        "Hong Kong":   ".HK",
    }

# Example tickers per market
    examples = {
        "US":          "e.g. AAPL, TSLA, SPY, GOOGL",
        "India (NSE)": "e.g. RELIANCE, TCS, INFY, HDFCBANK",
        "Singapore":   "e.g. D05 (DBS), O39 (OCBC), U11 (UOB)",
        "Hong Kong":   "e.g. 0700 (Tencent), 9988 (Alibaba)",
    }

    suffix = suffix_map[market]

    raw_ticker = st.text_input(
        "Stock Ticker",
        value="AAPL" if market == "🇺🇸 US" else "",
        help=examples[market]
    ).upper()

    ticker = raw_ticker + suffix

    if ticker != suffix:
        st.caption(f"Full ticker: `{ticker}`")
    
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.Timestamp("2026-01-01"))
        
        # Initial capital
        initial_capital = st.number_input(
            "Initial Capital (USD)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
        st.divider()
        
        # Strategy selection
        st.subheader("Strategies")
        use_ma      = st.checkbox("MA Crossover",    value=True)
        use_rsi     = st.checkbox("RSI",             value=True)
        use_bb      = st.checkbox("Bollinger Bands", value=True)
        use_macd    = st.checkbox("MACD",            value=True)
        
        st.divider()
        
        # Strategy parameters
        st.subheader("Parameters")
        
        if use_ma:
            st.markdown("**MA Crossover**")
            short_window = st.slider("Short MA Window", 10, 100, 50)
            long_window  = st.slider("Long MA Window",  50, 300, 200)
        
        if use_rsi:
            st.markdown("**RSI**")
            rsi_period    = st.slider("RSI Period",         5,  30, 14)
            rsi_oversold  = st.slider("Oversold Threshold", 10, 40, 30)
            rsi_overbought= st.slider("Overbought Threshold", 60, 90, 70)
        
        if use_bb:
            st.markdown("**Bollinger Bands**")
            bb_window = st.slider("BB Window", 5,  50, 20)
            bb_std    = st.slider("BB Std Dev", 1, 3,  2)
        
        if use_macd:
            st.markdown("**MACD**")
            macd_fast   = st.slider("Fast Period",   5,  50, 12)
            macd_slow   = st.slider("Slow Period",  10, 100, 26)
            macd_signal = st.slider("Signal Period", 5,  20,  9)
        
        st.divider()
        
    # Run button
    run = st.button("Run Backtest", type="primary", use_container_width=True)

# ── Data Fetching ───────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    """Download historical price data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']]
    df.columns = ['Close']
    return df

# ── Strategy Functions ──────────────────────────────────────
def strategy_ma(df, short_window, long_window):
    """Moving Average Crossover Strategy."""
    d = df.copy()
    d['SMA_short'] = d['Close'].rolling(short_window).mean()
    d['SMA_long']  = d['Close'].rolling(long_window).mean()
    d['Signal']    = np.where(d['SMA_short'] > d['SMA_long'], 1, 0)
    d['Position']  = d['Signal'].diff()
    return d

def strategy_rsi(df, period, oversold, overbought):
    """RSI Strategy — buy when oversold, sell when overbought."""
    d = df.copy()
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs        = avg_gain / avg_loss
    d['RSI']  = 100 - (100 / (1 + rs))
    d['Signal']   = 0
    d.loc[d['RSI'] < oversold,  'Signal'] = 1
    d.loc[d['RSI'] > overbought,'Signal'] = 0
    d['Signal']   = d['Signal'].ffill()
    d['Position'] = d['Signal'].diff()
    return d

def strategy_bb(df, window, num_std):
    """Bollinger Bands Strategy."""
    d = df.copy()
    d['MA']         = d['Close'].rolling(window).mean()
    d['Upper_Band'] = d['MA'] + (d['Close'].rolling(window).std() * num_std)
    d['Lower_Band'] = d['MA'] - (d['Close'].rolling(window).std() * num_std)
    d['Signal']     = 0
    d.loc[d['Close'] < d['Lower_Band'], 'Signal'] = 1
    d.loc[d['Close'] > d['Upper_Band'], 'Signal'] = 0
    d['Signal']   = d['Signal'].ffill()
    d['Position'] = d['Signal'].diff()
    return d

def strategy_macd(df, fast, slow, signal_period):
    """MACD Strategy."""
    d = df.copy()
    d['EMA_fast']    = d['Close'].ewm(span=fast,   adjust=False).mean()
    d['EMA_slow']    = d['Close'].ewm(span=slow,   adjust=False).mean()
    d['MACD']        = d['EMA_fast'] - d['EMA_slow']
    d['Signal_Line'] = d['MACD'].ewm(span=signal_period, adjust=False).mean()
    d['Signal']      = np.where(d['MACD'] > d['Signal_Line'], 1, 0)
    d['Position']    = d['Signal'].diff()
    return d

# ── Returns Calculator ──────────────────────────────────────
def calculate_returns(d, initial_capital):
    """Calculate portfolio equity curve."""
    d['Daily_Return']   = d['Close'].pct_change()
    d['Strategy_Return']= d['Daily_Return'] * d['Signal'].shift(1)
    d['Buy_Hold_Equity']= initial_capital * (1 + d['Daily_Return']).cumprod()
    d['Strategy_Equity']= initial_capital * (1 + d['Strategy_Return']).cumprod()
    return d

# ── Metrics Calculator ───────────────────────────────────────
def get_metrics(d, initial_capital):
    """Calculate performance metrics."""
    years           = len(d) / 252
    total_return    = (d['Strategy_Equity'].iloc[-1] / initial_capital - 1) * 100
    annual_return   = ((d['Strategy_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100
    sharpe          = (d['Strategy_Return'].mean() / d['Strategy_Return'].std()) * np.sqrt(252)
    rolling_max     = d['Strategy_Equity'].cummax()
    drawdown        = (d['Strategy_Equity'] - rolling_max) / rolling_max
    max_drawdown    = drawdown.min() * 100
    return {
        "Total Return":    f"{total_return:.2f}%",
        "Annual Return":   f"{annual_return:.2f}%",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Max Drawdown":    f"{max_drawdown:.2f}%",
        "Final Value":     f"${d['Strategy_Equity'].iloc[-1]:,.2f}"
    }

# ── Main Page ───────────────────────────────────────────────
if run:
    # Validate inputs
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    # Fetch data
    with st.spinner(f"Fetching data for {ticker}..."):
        data = fetch_data(ticker, start_date, end_date)

    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")
        st.stop()

    # Run selected strategies
    strategies = {}
    if use_ma:
        strategies["MA Crossover"] = calculate_returns(
            strategy_ma(data, short_window, long_window), initial_capital)
    if use_rsi:
        strategies["RSI"] = calculate_returns(
            strategy_rsi(data, rsi_period, rsi_oversold, rsi_overbought), initial_capital)
    if use_bb:
        strategies["Bollinger Bands"] = calculate_returns(
            strategy_bb(data, bb_window, bb_std), initial_capital)
    if use_macd:
        strategies["MACD"] = calculate_returns(
            strategy_macd(data, macd_fast, macd_slow, macd_signal), initial_capital)

    if not strategies:
        st.warning("Please select at least one strategy.")
        st.stop()

    # ── Metrics Table ───────────────────────────────────────
    st.subheader(f"📊 Performance Metrics — {ticker}")

    # Buy & hold metrics
    bh = data.copy()
    bh['Daily_Return']    = bh['Close'].pct_change()
    bh['Strategy_Return'] = bh['Daily_Return']
    bh['Strategy_Equity'] = initial_capital * (1 + bh['Daily_Return']).cumprod()
    bh_metrics = get_metrics(bh, initial_capital)

    metrics_data = {"Buy & Hold": bh_metrics}
    for name, df in strategies.items():
        metrics_data[name] = get_metrics(df, initial_capital)

    metrics_df = pd.DataFrame(metrics_data).T
    st.dataframe(metrics_df, use_container_width=True)

    st.divider()

    # ── Charts ──────────────────────────────────────────────
    st.subheader(f"📈 Strategy Charts — {ticker}")

    COLOURS = {
        "MA Crossover":    "#2196F3",
        "RSI":             "#9C27B0",
        "Bollinger Bands": "#FF9800",
        "MACD":            "#E91E63",
        "Buy & Hold":      "#4CAF50",
    }

    for name, df in strategies.items():
        with st.expander(f"{name} Strategy", expanded=True):

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Price & Signals", "Portfolio Value"),
                row_heights=[0.6, 0.4]
            )

            # Price line
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                name="Price", line=dict(color="#333333", width=1),
                opacity=0.8), row=1, col=1)

            # Strategy-specific indicators
            if name == "MA Crossover":
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['SMA_short'],
                    name=f"SMA {short_window}", line=dict(color="#2196F3", width=1.5)),
                    row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['SMA_long'],
                    name=f"SMA {long_window}", line=dict(color="#FF9800", width=1.5)),
                    row=1, col=1)

            elif name == "Bollinger Bands":
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Upper_Band'],
                    name="Upper Band", line=dict(color="#FF9800", width=1, dash="dash")),
                    row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Lower_Band'],
                    name="Lower Band", line=dict(color="#FF9800", width=1, dash="dash"),
                    fill="tonexty", fillcolor="rgba(255,152,0,0.05)"),
                    row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['MA'],
                    name="MA", line=dict(color="#FF9800", width=1)),
                    row=1, col=1)

            elif name == "RSI":
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['RSI'],
                    name="RSI", line=dict(color="#9C27B0", width=1.5)),
                    row=1, col=1)
                fig.add_hline(y=rsi_oversold,   line_dash="dash",
                              line_color="green", row=1, col=1)
                fig.add_hline(y=rsi_overbought, line_dash="dash",
                              line_color="red",   row=1, col=1)

            elif name == "MACD":
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['MACD'],
                    name="MACD", line=dict(color="#E91E63", width=1.5)),
                    row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Signal_Line'],
                    name="Signal Line", line=dict(color="#FF9800", width=1.5)),
                    row=1, col=1)

            # Buy signals
            buys = df[df['Position'] == 1]
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys['Close'],
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="#4CAF50")),
                row=1, col=1)

            # Sell signals
            sells = df[df['Position'] == -1]
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells['Close'],
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="#F44336")),
                row=1, col=1)

            # Portfolio comparison
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Strategy_Equity'],
                name=name, line=dict(color=COLOURS[name], width=2)),
                row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Buy_Hold_Equity'],
                name="Buy & Hold", line=dict(color="#4CAF50", width=2)),
                row=2, col=1)
            fig.add_hline(y=initial_capital, line_dash="dash",
                          line_color="gray", opacity=0.5, row=2, col=1)

            fig.update_layout(
                height=700,
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", y=1.02, x=0),
                margin=dict(l=0, r=0, t=60, b=0)
            )
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Portfolio Value (USD)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

    # ── Portfolio Comparison Chart ───────────────────────────
    st.divider()
    st.subheader("📉 All Strategies vs Buy & Hold")

    fig_compare = go.Figure()

    # Buy & hold
    bh_equity = initial_capital * (1 + data['Close'].pct_change()).cumprod()
    fig_compare.add_trace(go.Scatter(
        x=data.index, y=bh_equity,
        name="Buy & Hold", line=dict(color="#4CAF50", width=2.5)))

    # Each strategy
    for name, df in strategies.items():
        fig_compare.add_trace(go.Scatter(
            x=df.index, y=df['Strategy_Equity'],
            name=name, line=dict(color=COLOURS[name], width=2)))

    fig_compare.add_hline(y=initial_capital, line_dash="dash",
                          line_color="gray", opacity=0.5)

    fig_compare.update_layout(
        height=500,
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

else:
    # Landing state — before button is clicked
    st.info("👈 Configure your settings in the sidebar and click **Run Backtest** to begin.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategies Available", "4")
    col2.metric("Data Source", "Yahoo Finance")
    col3.metric("Interactive Charts", "Yes")
    col4.metric("Comparison Mode", "Yes")