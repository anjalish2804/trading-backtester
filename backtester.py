import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── STEP 1: Fetch Historical Data ──────────────────────────
def fetch_data(ticker, start, end):
    """Download historical price data from Yahoo Finance."""
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]  # we only need closing prices
    print(f"Downloaded {len(df)} trading days of data.")
    return df

# ── STEP 2: Calculate Moving Averages ──────────────────────
def add_moving_averages(df, short_window=50, long_window=200):
    """Add short and long moving averages to the DataFrame."""
    df['SMA_50'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_200'] = df['Close'].rolling(window=long_window).mean()
    return df

# ── STEP 3: Generate Buy/Sell Signals ──────────────────────
def generate_signals(df):
    """Generate buy and sell signals based on moving average crossover."""
    df['Signal'] = 0  # 0 = no position
    
    df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
    
    df['Position'] = df['Signal'].diff()
    
    return df

# ── STEP 4: Calculate Portfolio Returns ────────────────────
def calculate_returns(df, initial_capital=10000):
    """Calculate strategy returns vs buy and hold benchmark."""
    
    # Daily percentage change in price
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Strategy returns — only earn returns on days we're invested
    df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
    
    # Cumulative returns
    df['Buy_Hold_Equity'] = initial_capital * (1 + df['Daily_Return']).cumprod()
    df['Strategy_Equity'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
    
    # Performance metrics
    total_days = len(df)
    years = total_days / 252  # 252 trading days in a year
    
    buy_hold_return = (df['Buy_Hold_Equity'].iloc[-1] / initial_capital - 1) * 100
    strategy_return = (df['Strategy_Equity'].iloc[-1] / initial_capital - 1) * 100
    
    buy_hold_annual = ((df['Buy_Hold_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100
    strategy_annual = ((df['Strategy_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100
    
    sharpe = (df['Strategy_Return'].mean() / df['Strategy_Return'].std()) * np.sqrt(252)
    
    print("\n════════════════════════════════════")
    print("         BACKTEST RESULTS           ")
    print("════════════════════════════════════")
    print(f"Initial Capital:        ${initial_capital:,.2f}")
    print(f"Period:                 2020 - 2026")
    print(f"\nBuy & Hold:")
    print(f"  Final Value:          ${df['Buy_Hold_Equity'].iloc[-1]:,.2f}")
    print(f"  Total Return:         {buy_hold_return:.2f}%")
    print(f"  Annualised Return:    {buy_hold_annual:.2f}%")
    print(f"\nMA Crossover Strategy:")
    print(f"  Final Value:          ${df['Strategy_Equity'].iloc[-1]:,.2f}")
    print(f"  Total Return:         {strategy_return:.2f}%")
    print(f"  Annualised Return:    {strategy_annual:.2f}%")
    print(f"  Sharpe Ratio:         {sharpe:.2f}")
    print("════════════════════════════════════")
    
    return df

# ── STEP 5: Visualise Results ───────────────────────────────
def plot_results(df):
    """Plot stock price, moving averages, signals and portfolio performance."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('AAPL — Moving Average Crossover Backtest (2020–2026)', 
                 fontsize=14, fontweight='bold')
    
    # ── Chart 1: Price + Moving Averages + Signals ──
    ax1.plot(df.index, df['Close'], label='AAPL Price', 
             color='#333333', linewidth=1, alpha=0.8)
    ax1.plot(df.index, df['SMA_50'], label='50-Day MA', 
             color='#2196F3', linewidth=1.5)
    ax1.plot(df.index, df['SMA_200'], label='200-Day MA', 
             color='#FF9800', linewidth=1.5)
    
    # Buy signals
    buy_signals = df[df['Position'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Close'], 
                marker='^', color='#4CAF50', s=150, 
                label='Buy Signal', zorder=5)
    
    # Sell signals
    sell_signals = df[df['Position'] == -1]
    ax1.scatter(sell_signals.index, sell_signals['Close'], 
                marker='v', color='#F44336', s=150, 
                label='Sell Signal', zorder=5)
    
    ax1.set_title('Price & Signals')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # ── Chart 2: Portfolio Value Comparison ──
    ax2.plot(df.index, df['Strategy_Equity'], 
             label='MA Strategy', color='#2196F3', linewidth=2)
    ax2.plot(df.index, df['Buy_Hold_Equity'], 
             label='Buy & Hold', color='#4CAF50', linewidth=2)
    ax2.axhline(y=10000, color='gray', linestyle='--', 
                alpha=0.5, label='Initial Capital')
    
    ax2.set_title('Portfolio Value Comparison')
    ax2.set_ylabel('Portfolio Value (USD)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved as backtest_results.png")

# Test it
if __name__ == "__main__":
    data = fetch_data("AAPL", "2020-01-01", "2026-01-01")
    data = add_moving_averages(data)
    data = generate_signals(data)
    data = calculate_returns(data, initial_capital=10000)
    plot_results(data)