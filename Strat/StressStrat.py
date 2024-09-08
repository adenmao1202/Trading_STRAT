import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_pickle('TXF.pkl')

df
# Preprocess the data if necessary
df['date'] = pd.to_datetime(df['datetime'])
df.set_index('date', inplace=True)




def compute_stress_index(df):
    # Simplified example: using rolling standard deviation of close prices as a proxy for stress
    df['volatility'] = df['Close'].rolling(window=20).std()
    df['stress_index'] = (df['volatility'] - df['volatility'].mean()) / df['volatility'].std()
    return df

df = compute_stress_index(df)




### News Sentiment 
def compute_news_sentiment(df):
    # Placeholder: simplified sentiment analysis using random values
    np.random.seed(42)
    df['news_sentiment'] = np.random.choice([0, 1], size=len(df))  # Replace with actual sentiment analysis
    return df

df = compute_news_sentiment(df)


df['signal'] = np.where((df['stress_index'] < 0.5) & (df['news_sentiment'] == 1), 1, 0)


# Calculate strategy returns
df['strategy_ret'] = df['signal'].shift(1) * df['Close'].pct_change()
df['cum_strategy_ret'] = (1 + df['strategy_ret']).cumprod() - 1
df['cum_ret'] = (1 + df['Close'].pct_change()).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(df['cum_ret'], label='Cumulative Market Return')
plt.plot(df['cum_strategy_ret'], label='Cumulative Strategy Return')
plt.legend()
plt.title('Cumulative Returns')
plt.show()

# Calculate and plot maximum drawdown
MDD_series = - (df['cum_strategy_ret'].cummax() - df['cum_strategy_ret']) / df['cum_strategy_ret'].cummax()
MDD_series.plot(figsize=(16, 6), lw=2)
plt.title('Maximum Drawdown', fontsize=16)
plt.show()

# Define performance metrics calculation function
def calculate_performance_metrics(df):
    metrics = {}
    cumulative_return = df['cum_strategy_ret'].iloc[-1]
    metrics['Cumulative Return'] = cumulative_return
    trading_days = len(df)
    annual_return = (1 + cumulative_return) ** (252 / trading_days) - 1
    metrics['Annual Return'] = annual_return
    daily_return = df['strategy_ret']
    annual_volatility = daily_return.std() * np.sqrt(252)
    metrics['Annual Volatility'] = annual_volatility
    drawdown = (df['cum_strategy_ret'].cummax() - df['cum_strategy_ret']).max()
    max_drawdown = drawdown / df['cum_strategy_ret'].cummax().max()
    metrics['Maximum Drawdown (MDD)'] = max_drawdown
    risk_free_rate = 0.01  # Assume risk-free rate is 1%
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    metrics['Annual Sharpe Ratio'] = sharpe_ratio
    downside_std = daily_return[daily_return < 0].std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std
    metrics['Risk-Reward Ratio'] = sortino_ratio
    return metrics

performance_metrics = calculate_performance_metrics(df)

# Display results
for key, value in performance_metrics.items():
    if key != "Daily Return":  # Do not print daily return
        print(f"{key}: {value}")

# Plot cumulative returns and maximum drawdown
fig, ax = plt.subplots(figsize=(16, 6))
df[['cum_strategy_ret', 'cum_ret']].plot(ax=ax, lw=2)
plt.title('Cumulative Returns', fontsize=16)
plt.show()

MDD_series = - (df['cum_strategy_ret'].cummax() - df['cum_strategy_ret']) / df['cum_strategy_ret'].cummax()
high_index = df['cum_strategy_ret'][df['cum_strategy_ret'].cummax() == df['cum_strategy_ret']].index

MDD_series.plot(figsize=(16, 6), lw=2)
plt.title('Maximum Drawdown', fontsize=16)
plt.show()

# Plot cumulative returns and maximum drawdown (large plot)
fig, ax = plt.subplots(figsize=(12, 6))
df['cum_strategy_ret'].plot(label='Total Return', ax=ax, c='r')
plt.fill_between(MDD_series.index, MDD_series, 0, label='DD', color='gray', alpha=0.3)
plt.scatter(high_index, df['cum_strategy_ret'].loc[high_index], c='#02ff0f', label='High')
plt.legend()
plt.ylabel('Return%')
plt.xlabel('Date')
plt.title('Return & MDD', fontsize=16)
plt.show()
