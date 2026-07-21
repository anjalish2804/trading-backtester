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

    market = st.selectbox(
        "Market",
        options=["US", "India (NSE)", "Singapore", "Hong Kong"],
    )

    suffix_map = {
        "US":          "",
        "India (NSE)": ".NS",
        "Singapore":   ".SI",
        "Hong Kong":   ".HK",
    }

    examples = {
        "US":          "e.g. AAPL, TSLA, SPY, GOOGL",
        "India (NSE)": "e.g. RELIANCE, TCS, INFY, HDFCBANK",
        "Singapore":   "e.g. D05 (DBS), O39 (OCBC), U11 (UOB)",
        "Hong Kong":   "e.g. 0700 (Tencent), 9988 (Alibaba)",
    }

    suffix = suffix_map[market]

    raw_ticker = st.text_input(
        "Stock Ticker",
        value="",
        help=examples[market]
    ).upper()

    ticker = raw_ticker + suffix

    if ticker != suffix:
        st.caption(f"Full ticker: `{ticker}`")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.Timestamp.today().normalize())

    initial_capital = st.number_input(
        "Initial Capital (USD)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )

    st.divider()

    st.subheader("Strategies")
    use_ma   = st.checkbox("MA Crossover",       value=True)
    use_rsi  = st.checkbox("RSI",                value=True)
    use_bb   = st.checkbox("Bollinger Bands",    value=True)
    use_macd = st.checkbox("MACD",               value=True)
    use_ml   = st.checkbox("ML — Random Forest", value=True)

    st.divider()

    st.subheader("Parameters")
    st.markdown("**Global**")
    use_stop_loss = st.checkbox("Enable Stop-Loss", value=True)
    if use_stop_loss:
        stop_loss_pct = st.slider("Stop-Loss %", 1, 20, 5) / 100
    else:
        stop_loss_pct = None
    if use_ma:
        st.markdown("**MA Crossover**")
        short_window = st.slider("Short MA Window", 10, 100, 50)
        long_window  = st.slider("Long MA Window",  50, 300, 200)
    else:
        short_window = 50
        long_window  = 200

    if use_rsi:
        st.markdown("**RSI**")
        rsi_period     = st.slider("RSI Period",            5,  30, 14)
        rsi_oversold   = st.slider("Oversold Threshold",   10,  40, 30)
        rsi_overbought = st.slider("Overbought Threshold", 60,  90, 70)
    else:
        rsi_period     = 14
        rsi_oversold   = 30
        rsi_overbought = 70

    if use_bb:
        st.markdown("**Bollinger Bands**")
        bb_window = st.slider("BB Window",  5,  50, 20)
        bb_std    = st.slider("BB Std Dev", 1,   3,  2)
    else:
        bb_window = 20
        bb_std    = 2

    if use_macd:
        st.markdown("**MACD**")
        macd_fast   = st.slider("Fast Period",   5,  50, 12)
        macd_slow   = st.slider("Slow Period",  10, 100, 26)
        macd_signal = st.slider("Signal Period", 5,  20,  9)
    else:
        macd_fast   = 12
        macd_slow   = 26
        macd_signal = 9

    st.divider()

    run = st.button("Run Backtest", type="primary", use_container_width=True)


# ── Data Fetching ───────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']]
    df.columns = ['Close']
    return df


# ── Strategy Functions ──────────────────────────────────────
def strategy_ma(df, short_window, long_window, stop_loss_pct=0.05):
    d = df.copy()
    d['SMA_short'] = d['Close'].rolling(short_window).mean()
    d['SMA_long']  = d['Close'].rolling(long_window).mean()
    signals = []
    position = 0
    entry_price = 0
    for i in range(len(d)):
        price    = d['Close'].iloc[i]
        short_ma = d['SMA_short'].iloc[i]
        long_ma  = d['SMA_long'].iloc[i]
        if pd.isna(short_ma) or pd.isna(long_ma):
            signals.append(0)
            continue
        if position == 1 and stop_loss_pct is not None and price <= entry_price * (1 - stop_loss_pct):
            position = 0
            signals.append(0)
            continue
        if short_ma > long_ma:
            if position == 0:
                entry_price = price
            position = 1
        else:
            position = 0
        signals.append(position)
    d['Signal']   = signals
    d['Position'] = d['Signal'].diff()
    return d


def strategy_rsi(df, period, oversold, overbought, stop_loss_pct=0.05):
    d = df.copy()
    delta    = d['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss
    d['RSI'] = 100 - (100 / (1 + rs))
    signals = []
    position = 0
    entry_price = 0
    for i in range(len(d)):
        price = d['Close'].iloc[i]
        rsi   = d['RSI'].iloc[i]
        if pd.isna(rsi):
            signals.append(0)
            continue
        if position == 1 and stop_loss_pct is not None and price <= entry_price * (1 - stop_loss_pct):
            position = 0
            signals.append(0)
            continue
        if rsi < oversold:
            if position == 0:
                entry_price = price
            position = 1
        elif rsi > overbought:
            position = 0
        signals.append(position)
    d['Signal']   = signals
    d['Position'] = d['Signal'].diff()
    return d


def strategy_bb(df, window, num_std, stop_loss_pct=0.05):
    d = df.copy()
    d['MA']         = d['Close'].rolling(window).mean()
    d['Upper_Band'] = d['MA'] + (d['Close'].rolling(window).std() * num_std)
    d['Lower_Band'] = d['MA'] - (d['Close'].rolling(window).std() * num_std)
    signals = []
    position = 0
    entry_price = 0
    for i in range(len(d)):
        price = d['Close'].iloc[i]
        upper = d['Upper_Band'].iloc[i]
        lower = d['Lower_Band'].iloc[i]
        if pd.isna(upper) or pd.isna(lower):
            signals.append(0)
            continue
        if position == 1 and stop_loss_pct is not None and price <= entry_price * (1 - stop_loss_pct):
            position = 0
            signals.append(0)
            continue
        if price < lower:
            if position == 0:
                entry_price = price
            position = 1
        elif price > upper:
            position = 0
        signals.append(position)
    d['Signal']   = signals
    d['Position'] = d['Signal'].diff()
    return d


def strategy_macd(df, fast, slow, signal_period, stop_loss_pct=0.05):
    d = df.copy()
    d['EMA_fast']    = d['Close'].ewm(span=fast,          adjust=False).mean()
    d['EMA_slow']    = d['Close'].ewm(span=slow,          adjust=False).mean()
    d['MACD']        = d['EMA_fast'] - d['EMA_slow']
    d['Signal_Line'] = d['MACD'].ewm(span=signal_period,  adjust=False).mean()
    signals = []
    position = 0
    entry_price = 0
    for i in range(len(d)):
        price       = d['Close'].iloc[i]
        macd        = d['MACD'].iloc[i]
        signal_line = d['Signal_Line'].iloc[i]
        if pd.isna(macd) or pd.isna(signal_line):
            signals.append(0)
            continue
        if position == 1 and stop_loss_pct is not None and price <= entry_price * (1 - stop_loss_pct):
            position = 0
            signals.append(0)
            continue
        if macd > signal_line:
            if position == 0:
                entry_price = price
            position = 1
        else:
            position = 0
        signals.append(position)
    d['Signal']   = signals
    d['Position'] = d['Signal'].diff()
    return d


# ── Returns Calculator ──────────────────────────────────────
def calculate_returns(d, initial_capital):
    d['Daily_Return']    = d['Close'].pct_change()
    d['Strategy_Return'] = d['Daily_Return'] * d['Signal'].shift(1)
    d['Buy_Hold_Equity'] = initial_capital * (1 + d['Daily_Return']).cumprod()
    d['Strategy_Equity'] = initial_capital * (1 + d['Strategy_Return']).cumprod()
    return d


# ── Metrics Calculator ───────────────────────────────────────
def get_metrics(d, initial_capital):
    years         = len(d) / 252
    total_return  = (d['Strategy_Equity'].iloc[-1] / initial_capital - 1) * 100
    annual_return = ((d['Strategy_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100
    sharpe        = (d['Strategy_Return'].mean() / d['Strategy_Return'].std()) * np.sqrt(252)
    rolling_max   = d['Strategy_Equity'].cummax()
    drawdown      = (d['Strategy_Equity'] - rolling_max) / rolling_max
    max_drawdown  = drawdown.min() * 100
    return {
        "Total Return":  f"{total_return:.2f}%",
        "Annual Return": f"{annual_return:.2f}%",
        "Sharpe Ratio":  f"{sharpe:.2f}",
        "Max Drawdown":  f"{max_drawdown:.2f}%",
        "Final Value":   f"${d['Strategy_Equity'].iloc[-1]:,.2f}"
    }

# ── Parameter Optimisation ───────────────────────────────────
def optimise_ma(data, short_range, long_range, initial_capital, metric, stop_loss_pct):
    """Grid search over MA window combinations."""
    results = []
    for short in short_range:
        for long in long_range:
            if short >= long:  # short must always be less than long
                continue
            try:
                df = calculate_returns(
                    strategy_ma(data, short, long, stop_loss_pct), initial_capital)
                m = get_metrics(df, initial_capital)
                results.append({
                    "Short Window": short,
                    "Long Window":  long,
                    "Total Return":  float(m["Total Return"].replace("%","")),
                    "Annual Return": float(m["Annual Return"].replace("%","")),
                    "Sharpe Ratio":  float(m["Sharpe Ratio"]),
                    "Max Drawdown":  float(m["Max Drawdown"].replace("%","")),
                })
            except:
                continue
    return pd.DataFrame(results)

# ── ML Strategy ─────────────────────────────────────────────
def strategy_ml(df, stop_loss_pct=0.05):
    """Random Forest ML Strategy."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    d = df.copy()
    
    # ── Feature Engineering ──────────────────────────────────
    # Moving averages
    d['SMA_10']  = d['Close'].rolling(10).mean()
    d['SMA_30']  = d['Close'].rolling(30).mean()
    d['SMA_50']  = d['Close'].rolling(50).mean()
    d['SMA_200'] = d['Close'].rolling(200).mean()
    
    # Price relative to moving averages
    d['Price_SMA10_ratio']  = d['Close'] / d['SMA_10']
    d['Price_SMA50_ratio']  = d['Close'] / d['SMA_50']
    d['Price_SMA200_ratio'] = d['Close'] / d['SMA_200']
    
    # RSI
    delta    = d['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    d['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    d['EMA_12']  = d['Close'].ewm(span=12, adjust=False).mean()
    d['EMA_26']  = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD']    = d['EMA_12'] - d['EMA_26']
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_Hist']   = d['MACD'] - d['MACD_Signal']
    
    # Bollinger Bands
    d['BB_MA']    = d['Close'].rolling(20).mean()
    d['BB_Upper'] = d['BB_MA'] + 2 * d['Close'].rolling(20).std()
    d['BB_Lower'] = d['BB_MA'] - 2 * d['Close'].rolling(20).std()
    d['BB_Position'] = (d['Close'] - d['BB_Lower']) / (d['BB_Upper'] - d['BB_Lower'])
    
    # Returns
    d['Return_1d']  = d['Close'].pct_change(1)
    d['Return_5d']  = d['Close'].pct_change(5)
    d['Return_10d'] = d['Close'].pct_change(10)
    d['Return_20d'] = d['Close'].pct_change(20)
    
    # Volatility
    d['Volatility_10d'] = d['Return_1d'].rolling(10).std()
    d['Volatility_30d'] = d['Return_1d'].rolling(30).std()
    
    # Momentum
    d['Momentum_10d'] = d['Close'] / d['Close'].shift(10) - 1
    d['Momentum_30d'] = d['Close'] / d['Close'].shift(30) - 1
    
    # ── Label ────────────────────────────────────────────────
    # 1 if price is higher tomorrow, 0 if lower
    d['Target'] = (d['Close'].shift(-1) > d['Close']).astype(int)
    
    # ── Features ─────────────────────────────────────────────
    features = [
        'Price_SMA10_ratio', 'Price_SMA50_ratio', 'Price_SMA200_ratio',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Position',
        'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
        'Volatility_10d', 'Volatility_30d',
        'Momentum_10d', 'Momentum_30d',
    ]
    
    # Drop NaN rows
    d = d.dropna()
    
    X = d[features]
    y = d['Target']
    
    # ── Train/Test Split — 70/30 ─────────────────────────────
    split_idx = int(len(d) * 0.70)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # ── Scale Features ───────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # ── Train Model ──────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # ── Evaluate ─────────────────────────────────────────────
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100
    
    # ── Generate Signals on test period only ─────────────────
    predictions = model.predict(X_test)
    
    test_df = d.iloc[split_idx:].copy()
    test_df['Signal'] = predictions
    
    # Apply stop-loss
    signals = []
    position = 0
    entry_price = 0
    for i in range(len(test_df)):
        price = test_df['Close'].iloc[i]
        pred  = predictions[i]
        if position == 1 and stop_loss_pct is not None and price <= entry_price * (1 - stop_loss_pct):
            position = 0
            signals.append(0)
            continue
        if pred == 1:
            if position == 0:
                entry_price = price
            position = 1
        else:
            position = 0
        signals.append(position)
    
    test_df['Signal']   = signals
    test_df['Position'] = test_df['Signal'].diff()
    
    # Store model metadata
    test_df.attrs['train_acc']      = train_acc
    test_df.attrs['test_acc']       = test_acc
    test_df.attrs['feature_names']  = features
    test_df.attrs['importances']    = model.feature_importances_.tolist()
    
    return test_df

# ── Colours ─────────────────────────────────────────────────
COLOURS = {
    "MA Crossover":       "#2196F3",
    "RSI":                "#9C27B0",
    "Bollinger Bands":    "#FF9800",
    "MACD":               "#E91E63",
    "ML — Random Forest": "#C6D400",
    "Buy & Hold":         "#4CAF50",
}

# ── Main Page ───────────────────────────────────────────────
tab1, tab2 = st.tabs(["Backtest", "Optimise"])

with tab1:
    if run:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        if not raw_ticker:
            st.error("Please enter a stock ticker.")
            st.stop()

        with st.spinner(f"Fetching data for {ticker}..."):
            data = fetch_data(ticker, start_date, end_date)

        if data.empty:
            st.error(f"No data found for '{ticker}'. Please check the ticker symbol.")
            st.stop()

        strategies = {}
        if use_ma:
            strategies["MA Crossover"] = calculate_returns(
                strategy_ma(data, short_window, long_window, stop_loss_pct), initial_capital)
        if use_rsi:
            strategies["RSI"] = calculate_returns(
                strategy_rsi(data, rsi_period, rsi_oversold, rsi_overbought, stop_loss_pct), initial_capital)
        if use_bb:
            strategies["Bollinger Bands"] = calculate_returns(
                strategy_bb(data, bb_window, bb_std, stop_loss_pct), initial_capital)
        if use_macd:
            strategies["MACD"] = calculate_returns(
                strategy_macd(data, macd_fast, macd_slow, macd_signal, stop_loss_pct), initial_capital)
        if use_ml:
            with st.spinner("Training Random Forest model..."):
                ml_df = strategy_ml(data, stop_loss_pct)
            if ml_df is not None and not ml_df.empty:
                strategies["ML — Random Forest"] = calculate_returns(ml_df, initial_capital)

        if not strategies:
            st.warning("Please select at least one strategy.")
            st.stop()

        # ── Metrics Table ───────────────────────────────────────
        st.subheader(f"Performance Metrics — {ticker}")

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

        # ── Individual Strategy Charts ───────────────────────────
        st.subheader(f"Strategy Charts — {ticker}")

        for name, df in strategies.items():
            with st.expander(f"{name} Strategy", expanded=True):

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("Price & Signals", "Portfolio Value"),
                    row_heights=[0.6, 0.4]
                )

                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    name="Price", line=dict(color="#333333", width=1),
                    opacity=0.8), row=1, col=1)

                if name == "MA Crossover":
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_short'],
                        name=f"SMA {short_window}",
                        line=dict(color="#2196F3", width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_long'],
                        name=f"SMA {long_window}",
                        line=dict(color="#FF9800", width=1.5)), row=1, col=1)

                elif name == "Bollinger Bands":
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Upper_Band'],
                        name="Upper Band",
                        line=dict(color="#FF9800", width=1, dash="dash")), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Lower_Band'],
                        name="Lower Band",
                        line=dict(color="#FF9800", width=1, dash="dash"),
                        fill="tonexty", fillcolor="rgba(255,152,0,0.05)"), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['MA'],
                        name="MA",
                        line=dict(color="#FF9800", width=1)), row=1, col=1)

                elif name == "RSI":
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['RSI'],
                        name="RSI",
                        line=dict(color="#9C27B0", width=1.5)), row=1, col=1)
                    fig.add_hline(y=rsi_oversold,   line_dash="dash",
                                line_color="green", row=1, col=1)
                    fig.add_hline(y=rsi_overbought, line_dash="dash",
                                line_color="red",   row=1, col=1)

                elif name == "MACD":
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['MACD'],
                        name="MACD",
                        line=dict(color="#E91E63", width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Signal_Line'],
                        name="Signal Line",
                        line=dict(color="#FF9800", width=1.5)), row=1, col=1)

                buys = df[df['Position'] == 1]
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys['Close'],
                    mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=12, color="#4CAF50")),
                    row=1, col=1)

                sells = df[df['Position'] == -1]
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells['Close'],
                    mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="#F44336")),
                    row=1, col=1)

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
                fig.update_yaxes(title_text="Price (USD)",           row=1, col=1)
                fig.update_yaxes(title_text="Portfolio Value (USD)", row=2, col=1)
                fig.update_xaxes(title_text="Date",                  row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

        # ── ML Model Insights ────────────────────────────────────
        if use_ml and "ML — Random Forest" in strategies:
            st.divider()
            st.subheader("ML Model Insights")

            ml_result = strategies["ML — Random Forest"]
            train_acc = ml_result.attrs.get('train_acc', 0)
            test_acc  = ml_result.attrs.get('test_acc',  0)
            features  = ml_result.attrs.get('feature_names', [])
            importances = ml_result.attrs.get('importances', [])

            col1, col2, col3 = st.columns(3)
            col1.metric("Training Accuracy",  f"{train_acc:.1f}%")
            col2.metric("Test Accuracy",      f"{test_acc:.1f}%")
            col3.metric("Features Used",      len(features))

            st.caption("Note: The ML strategy only trades during the test period (last 30% of data) — this is why its equity curve starts later than other strategies.")

            if features and importances:
                importance_df = pd.DataFrame({
                    'Feature':    features,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)

                fig_imp = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#00BCD4',
                ))
                fig_imp.update_layout(
                    height=500,
                    title="Feature Importance — Which indicators does the model rely on most?",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    template="plotly_white",
                    margin=dict(l=0, r=0, t=60, b=0)
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        fig_compare = go.Figure()

        bh_equity = initial_capital * (1 + data['Close'].pct_change()).cumprod()
        fig_compare.add_trace(go.Scatter(
            x=data.index, y=bh_equity,
            name="Buy & Hold", line=dict(color="#4CAF50", width=2.5)))

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
        st.info("Configure your settings in the sidebar and click **Run Backtest** to begin.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategies Available", "4")
        col2.metric("Data Source", "Yahoo Finance")
        col3.metric("Interactive Charts", "Yes")
        col4.metric("Comparison Mode", "Yes")

with tab2:
    st.subheader("Parameter Optimisation — MA Crossover")
    st.markdown("Find the best Short and Long MA windows by testing every combination across your selected date range.")
    st.info("The data is split into **in-sample** (used to optimise) and **out-of-sample** (used to validate) periods to avoid overfitting.")

    col1, col2 = st.columns(2)
    with col1:
        opt_ticker_raw = st.text_input(
            "Ticker to Optimise",
            value="",
            help=examples[market]
        ).upper()
        opt_ticker = opt_ticker_raw + suffix

        opt_metric = st.selectbox(
            "Optimise For",
            options=["Sharpe Ratio", "Annual Return", "Total Return", "Max Drawdown"],
        )

    with col2:
        split_pct = st.slider(
            "In-Sample Split %",
            min_value=50, max_value=80, value=70,
            help="70% means 70% of data used to find best parameters, 30% to validate."
        )
        opt_stop_loss = st.slider("Stop-Loss % (Optimiser)", 1, 20, 5) / 100

    col3, col4 = st.columns(2)
    with col3:
        short_min  = st.number_input("Short Window — Min",  min_value=5,  max_value=50,  value=10)
        short_max  = st.number_input("Short Window — Max",  min_value=10, max_value=100, value=60)
        short_step = st.number_input("Short Window — Step", min_value=1,  max_value=10,  value=5)
    with col4:
        long_min   = st.number_input("Long Window — Min",   min_value=20, max_value=150, value=50)
        long_max   = st.number_input("Long Window — Max",   min_value=50, max_value=400, value=250)
        long_step  = st.number_input("Long Window — Step",  min_value=1,  max_value=20,  value=10)

    run_opt = st.button("Run Optimisation", type="primary", use_container_width=True)

    if run_opt:
        if not opt_ticker_raw:
            st.error("Please enter a ticker.")
            st.stop()

        with st.spinner(f"Fetching data for {opt_ticker}..."):
            opt_data = fetch_data(opt_ticker, start_date, end_date)

        if opt_data.empty:
            st.error(f"No data found for '{opt_ticker}'.")
            st.stop()

        split_idx  = int(len(opt_data) * split_pct / 100)
        in_sample  = opt_data.iloc[:split_idx]
        out_sample = opt_data.iloc[split_idx:]
        split_date = opt_data.index[split_idx].strftime("%Y-%m-%d")

        st.success(f"In-sample: {opt_data.index[0].strftime('%Y-%m-%d')} to {split_date} | Out-of-sample: {split_date} to {opt_data.index[-1].strftime('%Y-%m-%d')}")

        short_range = range(int(short_min), int(short_max)+1, int(short_step))
        long_range  = range(int(long_min),  int(long_max)+1,  int(long_step))
        total_combinations = sum(1 for s in short_range for l in long_range if s < l)

        st.info(f"Testing {total_combinations} parameter combinations...")

        with st.spinner("Running grid search..."):
            results_df = optimise_ma(
                in_sample, short_range, long_range,
                initial_capital, opt_metric, opt_stop_loss)

        if results_df.empty:
            st.error("No valid combinations found. Try adjusting the ranges.")
            st.stop()

        if opt_metric == "Max Drawdown":
            best_idx = results_df["Max Drawdown"].idxmax()
        else:
            best_idx = results_df[opt_metric].idxmax()

        best       = results_df.loc[best_idx]
        best_short = int(best["Short Window"])
        best_long  = int(best["Long Window"])

        st.divider()
        st.subheader("Best Parameters Found")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Short Window", best_short)
        m2.metric("Long Window",  best_long)
        m3.metric(opt_metric, f"{best[opt_metric]:.2f}" + ("%" if opt_metric != "Sharpe Ratio" else ""))
        m4.metric("Combinations Tested", total_combinations)

        st.divider()
        st.subheader("Performance Heatmap — In-Sample")
        pivot = results_df.pivot(
            index="Short Window", columns="Long Window", values=opt_metric)
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            text=pivot.values.round(2),
            texttemplate="%{text}",
            hovertemplate="Short: %{y}<br>Long: %{x}<br>Value: %{z:.2f}<extra></extra>",
        ))
        fig_heat.update_layout(
            height=500,
            xaxis_title="Long Window",
            yaxis_title="Short Window",
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()
        st.subheader("Out-of-Sample Validation")
        st.markdown("Testing the best parameters on unseen data to check for overfitting.")

        oos_optimised = calculate_returns(
            strategy_ma(out_sample, best_short, best_long, opt_stop_loss), initial_capital)
        oos_default   = calculate_returns(
            strategy_ma(out_sample, 50, 200, opt_stop_loss), initial_capital)
        oos_bh = out_sample.copy()
        oos_bh['Daily_Return']    = oos_bh['Close'].pct_change()
        oos_bh['Strategy_Return'] = oos_bh['Daily_Return']
        oos_bh['Strategy_Equity'] = initial_capital * (1 + oos_bh['Daily_Return']).cumprod()

        oos_metrics = {
            f"Optimised ({best_short}/{best_long})": get_metrics(oos_optimised, initial_capital),
            "Default (50/200)":                      get_metrics(oos_default,   initial_capital),
            "Buy & Hold":                            get_metrics(oos_bh,        initial_capital),
        }
        st.dataframe(pd.DataFrame(oos_metrics).T, use_container_width=True)

        fig_oos = go.Figure()
        fig_oos.add_trace(go.Scatter(
            x=oos_optimised.index, y=oos_optimised['Strategy_Equity'],
            name=f"Optimised ({best_short}/{best_long})",
            line=dict(color="#2196F3", width=2)))
        fig_oos.add_trace(go.Scatter(
            x=oos_default.index, y=oos_default['Strategy_Equity'],
            name="Default (50/200)",
            line=dict(color="#FF9800", width=2)))
        fig_oos.add_trace(go.Scatter(
            x=oos_bh.index, y=oos_bh['Strategy_Equity'],
            name="Buy & Hold",
            line=dict(color="#4CAF50", width=2)))
        fig_oos.add_hline(y=initial_capital, line_dash="dash",
                          line_color="gray", opacity=0.5)
        fig_oos.update_layout(
            height=400,
            hovermode="x unified",
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_oos, use_container_width=True)

        st.caption("If the optimised strategy significantly underperforms on out-of-sample data vs in-sample, this suggests overfitting.")