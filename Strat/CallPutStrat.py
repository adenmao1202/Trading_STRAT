import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from talib import abstract
import warnings
warnings.filterwarnings("ignore")
from FinMind.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D


api = DataLoader()
# api.login_by_token(api_token='token')
# api.login(user_id='user_id',password='password')
df_ii = api.taiwan_futures_institutional_investors(
    data_id='TX',
    start_date='2018-06-05',
)

df_ii.drop(['futures_id'], axis=1, inplace=True)
df_ii.index = pd.to_datetime(df_ii.date)
df_ii.drop(['date'], axis=1, inplace=True)

# 将同一天的多行数据合并为单行
df_ii_pivot = df_ii.pivot_table(index=df_ii.index, columns='institutional_investors', aggfunc='sum').reset_index()
df_ii_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_ii_pivot.columns.values]
df_ii_pivot.rename(columns={'index': 'date'}, inplace=True)
df_ii_pivot.set_index('date', inplace=True)

# 载入主要数据
file_path = 'TXF.pkl'
df = pd.read_pickle(file_path)

# 确保索引日期时间格式正确
df.index = pd.to_datetime(df.index)

# 处理 column 名称
df.drop(['簡稱', '期貨名稱'], axis=1, inplace=True)
df.columns = ['open', 'high', 'low', 'close', 'returns', 'volume', 'open_int', 'basis', 'tot_unsettled']

# 加入技术指标
sma = abstract.MA(df, 20, matype=0)
df['sma'] = sma
ema = abstract.MA(df, 20, matype=1)
df['ema'] = ema
adx = abstract.ADX(df)
df['adx'] = adx
rsi = abstract.RSI(df)
df['rsi'] = rsi

macd_values = abstract.MACD(df)
df = pd.concat([df, macd_values], axis=1)
df.columns
# 截取2018-06-05以后的数据
df = df[df.index >= '2018-06-05']
df
# 合并两个数据框
df = pd.concat([df, df_ii_pivot], axis=1)


# 计算三大法人的总多单和空单
df['large_long_positions'] = df[['long_open_interest_balance_volume_自營商', 'long_open_interest_balance_volume_投信', 'long_open_interest_balance_volume_外資']].sum(axis=1)
df['large_short_positions'] = df[['short_open_interest_balance_volume_自營商', 'short_open_interest_balance_volume_投信', 'short_open_interest_balance_volume_外資']].sum(axis=1)

# 计算散户的多单和空单
df['retail_long_positions'] = df['open_int'] - df['large_long_positions']
df['retail_short_positions'] = df['open_int'] - df['large_short_positions']

# 计算散户多空比
df['retail_long_short_ratio'] = (df['retail_long_positions'] - df['retail_short_positions']) / df['open_int']

# 計算三大法人的多空比
df['FA_long_short_ratio'] = (df['large_long_positions'] - df['large_short_positions']) / df['open_int']

### 切分數據集
train_size = int(len(df) * 0.6)
test_size = int(len(df) * 0.3)
validation_size = len(df) - train_size - test_size

train = df.iloc[:train_size]
test = df.iloc[train_size:train_size + test_size]
validation = df.iloc[train_size + test_size:]

# plot 散戶多空比
train['retail_long_short_ratio'].plot(figsize=(10, 6))
# plot 三大法人多空比
train['FA_long_short_ratio'].plot(figsize=(10, 6))
train
##############################################################################

# 交易策略逻辑
def trading_strategy(df, entry_threshold, exit_threshold):
    position = 0
    entry_ratio = 0

    df['signal'] = 0
    df['position_size'] = 0

    for i in range(1, len(df) - 1):
        if df['retail_long_short_ratio'].iloc[i] < entry_threshold and df.index[i].weekday() != 2:  # 非結算日
            if position == 0:  # 開新倉
                df['signal'].iloc[i + 1] = 1
                df['position_size'].iloc[i + 1] = 1
                position = 1
                entry_ratio = df['retail_long_short_ratio'].iloc[i]
            elif df['retail_long_short_ratio'].iloc[i] < entry_ratio - 0.03:  # 加碼
                df['signal'].iloc[i + 1] = 1
                df['position_size'].iloc[i + 1] += 1

        if df['retail_long_short_ratio'].iloc[i] > exit_threshold or df.index[i].weekday() == 2:  # 出場條件
            if position == 1:
                df['signal'].iloc[i + 1] = -1
                df['position_size'].iloc[i + 1] = 0
                position = 0

    return df



# Grid Search 尋找最佳進場、出場參數
entry_range = np.arange(-1.5, 0, 0.01)
exit_range = np.arange(0, 1.5, 0.01)
performance_matrix = np.zeros((len(entry_range), len(exit_range)))

for i, entry_threshold in enumerate(entry_range):
    for j, exit_threshold in enumerate(exit_range):
        train_copy = train.copy()
        train_copy = trading_strategy(train_copy, entry_threshold, exit_threshold)

        train_copy['strategy_ret'] = train_copy['signal'].shift(1) * train_copy['position_size'].shift(1) * train_copy['close'].pct_change()
        train_copy['cum_strategy_ret'] = (1 + train_copy['strategy_ret']).cumprod() - 1

        cumulative_return = train_copy['cum_strategy_ret'].iloc[-1]
        performance_matrix[i, j] = cumulative_return

X, Y = np.meshgrid(exit_range, entry_range)
Z = performance_matrix

# 找到最佳參數
best_indices = np.unravel_index(np.argmax(performance_matrix), performance_matrix.shape)
best_entry_threshold = entry_range[best_indices[0]]
best_exit_threshold = exit_range[best_indices[1]]

print(f"Best Entry Threshold: {best_entry_threshold}, Best Exit Threshold: {best_exit_threshold}")


df = trading_strategy(df, best_entry_threshold, best_exit_threshold)



# 應用最佳參數到測試數據集
best_entry_threshold = -0.06
best_exit_threshold = 0.06

test = trading_strategy(test, best_entry_threshold, best_exit_threshold)









# 計算策略收益
test['strategy_ret'] = test['signal'].shift(1) * test['position_size'].shift(1) * test['close'].pct_change()
test['cum_strategy_ret'] = (1 + test['strategy_ret']).cumprod() - 1
test['cum_ret'] = (1 + test['close'].pct_change()).cumprod() - 1

# 繪製累積收益率
plt.figure(figsize=(14, 7))
plt.plot(test['cum_ret'], label='Cumulative Market Return')
plt.plot(test['cum_strategy_ret'], label='Cumulative Strategy Return')
plt.legend()
plt.title('Cumulative Returns on Test Data')
plt.show()

# 繪製最大回撤
MDD_series_test = - (test['cum_strategy_ret'].cummax() - test['cum_strategy_ret']) / test['cum_strategy_ret'].cummax()
MDD_series_test.plot(figsize=(16, 6), lw=2)
plt.title('Maximum Drawdown on Test Data', fontsize=16)
plt.show()

# 計算績效指標
def calculate_performance_metrics(df):
    metrics = {}

    # 計算累計報酬
    cumulative_return = df['cum_strategy_ret'].iloc[-1]
    metrics['Cumulative Return'] = cumulative_return

    # 計算年化報酬
    trading_days = len(df)
    annual_return = (1 + cumulative_return) ** (252 / trading_days) - 1
    metrics['Annual Return'] = annual_return

    # 計算年化波動度
    daily_return = df['strategy_ret']
    annual_volatility = daily_return.std() * np.sqrt(252)
    metrics['Annual Volatility'] = annual_volatility

    # 計算最大回撤 (MDD)
    drawdown = (df['cum_strategy_ret'].cummax() - df['cum_strategy_ret']).max()
    max_drawdown = drawdown / df['cum_strategy_ret'].cummax().max()
    metrics['Maximum Drawdown (MDD)'] = max_drawdown

    # 計算年化夏普比率
    risk_free_rate = 0.01  # 假設無風險利率為1%
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    metrics['Annual Sharpe Ratio'] = sharpe_ratio

    # 計算風報比
    downside_std = daily_return[daily_return < 0].std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std
    metrics['Risk-Reward Ratio'] = sortino_ratio

    return metrics

performance_metrics_test = calculate_performance_metrics(test)

# 顯示結果
for key, value in performance_metrics_test.items():
    print(f"{key}: {value}")


############################################################

# 在測試集上細化最佳參數範圍進行優化
entry_range_test = np.arange(best_entry_threshold_train - 0.05, best_entry_threshold_train + 0.05, 0.01)
exit_range_test = np.arange(best_exit_threshold_train - 0.05, best_exit_threshold_train + 0.05, 0.01)
performance_matrix_test = np.zeros((len(entry_range_test), len(exit_range_test)))

for i, entry_threshold in enumerate(entry_range_test):
    for j, exit_threshold in enumerate(exit_range_test):
        test_copy = test.copy()
        test_copy = trading_strategy(test_copy, entry_threshold, exit_threshold)

        test_copy['strategy_ret'] = test_copy['signal'].shift(1) * test_copy['position_size'].shift(1) * test_copy['close'].pct_change()
        test_copy['cum_strategy_ret'] = (1 + test_copy['strategy_ret']).cumprod() - 1

        cumulative_return = test_copy['cum_strategy_ret'].iloc[-1]
        performance_matrix_test[i, j] = cumulative_return

X_test, Y_test = np.meshgrid(exit_range_test, entry_range_test)
Z_test = performance_matrix_test

fig_test = plt.figure(figsize=(12, 8))
ax_test = fig_test.add_subplot(111, projection='3d')
surf_test = ax_test.plot_surface(X_test, Y_test, Z_test, cmap='viridis')

ax_test.set_xlabel('Exit Threshold')
ax_test.set_ylabel('Entry Threshold')
ax_test.set_zlabel('Cumulative Return')
plt.colorbar(surf_test)
plt.title('Parameter Optimization Surface on Test Data')
plt.show()

# 找到測試集上的最佳參數
best_idx_test = np.unravel_index(np.argmax(performance_matrix_test, axis=None), performance_matrix_test.shape)
best_entry_threshold_test = entry_range_test[best_idx_test[0]]
best_exit_threshold_test = exit_range_test[best_idx_test[1]]

print(f"Best Entry Threshold from Test Data: {best_entry_threshold_test}, Best Exit Threshold from Test Data: {best_exit_threshold_test}")