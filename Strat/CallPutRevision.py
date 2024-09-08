# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import talib
# from talib import abstract
# import warnings
# warnings.filterwarnings("ignore")
# from FinMind.data import DataLoader

# api = DataLoader()
# # api.login_by_token(api_token='token')
# # api.login(user_id='user_id',password='password')
# df_ii = api.taiwan_futures_institutional_investors(
#     data_id='TX',
#     start_date='2018-06-05',
# )

# df_ii.drop(['futures_id'], axis=1, inplace=True)
# df_ii.index = pd.to_datetime(df_ii.date)
# df_ii.drop(['date'], axis=1, inplace=True)

# # 将同一天的多行数据合并为单行
# df_ii_pivot = df_ii.pivot_table(index=df_ii.index, columns='institutional_investors', aggfunc='sum').reset_index()
# df_ii_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_ii_pivot.columns.values]
# df_ii_pivot.rename(columns={'index': 'date'}, inplace=True)
# df_ii_pivot.set_index('date', inplace=True)

# # 载入主要数据
# file_path = 'TXF.pkl'
# df = pd.read_pickle(file_path)

# # 确保索引日期时间格式正确
# df.index = pd.to_datetime(df.index)

# # 处理 column 名称
# df.drop(['簡稱', '期貨名稱'], axis=1, inplace=True)
# df.columns = ['open', 'high', 'low', 'close', 'returns', 'volume', 'open_int', 'basis', 'tot_unsettled']

# # 加入技术指标
# sma = abstract.MA(df, 20, matype=0)
# df['sma'] = sma
# ema = abstract.MA(df, 20, matype=1)
# df['ema'] = ema
# adx = abstract.ADX(df)
# df['adx'] = adx
# rsi = abstract.RSI(df)
# df['rsi'] = rsi

# macd_values = abstract.MACD(df)
# df = pd.concat([df, macd_values], axis=1)

# # 截取2018-06-05以后的数据
# df = df[df.index >= '2018-06-05']

# # 合并两个数据框
# df = pd.concat([df, df_ii_pivot], axis=1)

# # 检查合并后的数据框
# print(df.head())
# df.columns
# # 计算三大法人的总多单和空单
# df['large_long_positions'] = df[['long_open_interest_balance_volume_自營商', 'long_open_interest_balance_volume_投信', 'long_open_interest_balance_volume_外資']].sum(axis=1)
# df['large_short_positions'] = df[['short_open_interest_balance_volume_自營商', 'short_open_interest_balance_volume_投信', 'short_open_interest_balance_volume_外資']].sum(axis=1)

# # 计算散户的多单和空单
# df['retail_long_positions'] = df['tot_unsettled'] - df['large_long_positions']
# df['retail_short_positions'] = df['tot_unsettled'] - df['large_short_positions']

# # 计算散户多空比
# df['retail_long_short_ratio'] = (df['retail_long_positions'] - df['retail_short_positions']) / df['tot_unsettled']

# # 显示合并后的数据框的描述性统计
# print(df.describe())

# # 显示合并后的数据框
# print(df.head())
# df.columns

# # 交易策略逻辑
# def trading_strategy(df):
#     position = 0
#     entry_ratio = 0

#     df['signal'] = 0
#     df['position_size'] = 0

#     for i in range(1, len(df) - 1):
#         if df['retail_long_short_ratio'].iloc[i] < -0.06 and df.index[i].weekday() != 4:  # 非结算日
#             if position == 0:  # 开新仓
#                 df['signal'].iloc[i + 1] = 1
#                 df['position_size'].iloc[i + 1] = 1
#                 position = 1
#                 entry_ratio = df['retail_long_short_ratio'].iloc[i]
#             elif df['retail_long_short_ratio'].iloc[i] < entry_ratio + 0.03:  # 加码
#                 df['signal'].iloc[i + 1] = 1
#                 df['position_size'].iloc[i + 1] += 1

#         if df['retail_long_short_ratio'].iloc[i] > 0.06 or df.index[i].weekday() == 4:  # 出场条件
#             if position == 1:
#                 df['signal'].iloc[i + 1] = -1
#                 df['position_size'].iloc[i + 1] = 0
#                 position = 0

#     return df

# # 应用策略
# df = trading_strategy(df)

# # 计算策略收益
# df['strategy_ret'] = df['signal'].shift(1) * df['position_size'].shift(1) * df['close'].pct_change()
# df['cum_strategy_ret'] = (1 + df['strategy_ret']).cumprod() - 1
# df['cum_ret'] = (1 + df['close'].pct_change()).cumprod() - 1

# # 绘制累积收益率
# plt.figure(figsize=(14, 7))
# plt.plot(df['cum_ret'], label='Cumulative Market Return')
# plt.plot(df['cum_strategy_ret'], label='Cumulative Strategy Return')
# plt.legend()
# plt.title('Cumulative Returns')
# plt.show()

# # 绘制最大回撤
# MDD_series = - (df['cum_strategy_ret'].cummax() - df['cum_strategy_ret']) / df['cum_strategy_ret'].cummax()
# MDD_series.plot(figsize=(16, 6), lw=2)
# plt.title('Maximum Drawdown', fontsize=16)
# plt.show()

# # 定义计算绩效指标的函数
# def calculate_performance_metrics(df):
#     metrics = {}

#     # 计算累计报酬
#     cumulative_return = df['cum_strategy_ret'].iloc[-1]
#     metrics['Cumulative Return'] = cumulative_return

#     # 计算年化报酬
#     trading_days = len(df)
#     annual_return = (1 + cumulative_return) ** (252 / trading_days) - 1
#     metrics['Annual Return'] = annual_return

#     # 计算年化波动度
#     daily_return = df['strategy_ret']
#     annual_volatility = daily_return.std() * np.sqrt(252)
#     metrics['Annual Volatility'] = annual_volatility

#     # 计算最大回撤 (MDD)
#     drawdown = (df['cum_strategy_ret'].cummax() - df['cum_strategy_ret']).max()
#     max_drawdown = drawdown / df['cum_strategy_ret'].cummax().max()
#     metrics['Maximum Drawdown (MDD)'] = max_drawdown

#     # 计算年化夏普比率
#     risk_free_rate = 0.01  # 假设无风险利率为1%
#     sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
#     metrics['Annual Sharpe Ratio'] = sharpe_ratio

#     # 计算风报比
#     downside_std = daily_return[daily_return < 0].std() * np.sqrt(252)
#     sortino_ratio = (annual_return - risk_free_rate) / downside_std
#     metrics['Risk-Reward Ratio'] = sortino_ratio

#     return metrics

# performance_metrics = calculate_performance_metrics(df)

# # 显示结果
# for key, value in performance_metrics.items():
#     print(f"{key}: {value}")

# # 绘制累积收益率和最大回撤（大图）
# fig, ax = plt.subplots(figsize=(12, 6))
# df['cum_strategy_ret'].plot(label='Total Return', ax=ax, c='r')
# plt.fill_between(MDD_series.index, MDD_series, 0, label='DD', color='gray', alpha=0.3)
# plt.scatter(df.index, df['cum_strategy_ret'].loc[df.index], c='#02ff0f', label='High')
# plt.legend()
# plt.ylabel('Return%')
# plt.xlabel('Date')
# plt.title('Return & MDD', fontsize=16)
# plt.show()


# ######################################

# ## Revision 
# # Counting the number of trades
# df['trade_count'] = df['signal'].diff().abs().fillna(0)
# trade_count = df['trade_count'].sum()

# print(f"Total number of trades executed: {trade_count}")

# # Plotting trade distribution over time
# plt.figure(figsize=(14, 7))
# df['trade_count'].cumsum().plot()
# plt.title('Cumulative Number of Trades Over Time')
# plt.xlabel('Date')
# plt.ylabel('Cumulative Number of Trades')
# plt.show()

# #############################################

# # Check the frequency of the entry condition being met
# entry_condition_met = (df['retail_long_short_ratio'] < -0.06).sum()
# print(f"Number of times the entry condition was met: {entry_condition_met}")

# # Check the frequency of the exit condition being met
# exit_condition_met = (df['retail_long_short_ratio'] > 0.06).sum()
# print(f"Number of times the exit condition was met: {exit_condition_met}")

# # Check the frequency of the conditions being met on Fridays (settlement day)
# wednesday_condition_met = (df.index.weekday == 2).sum()
# print(f"Number of Wednesdays (potential settlement days): {wednesday_condition_met}")


# ##################################################

# ## Flat period
# # Identify flat periods
# flat_periods = df[df['cum_strategy_ret'].diff() == 0].index

# # Print the flat periods
# print("Flat periods:")
# print(flat_periods)

# # Analyze market conditions during flat periods
# flat_period_analysis = df.loc[flat_periods, ['close', 'retail_long_short_ratio', 'rsi', 'boll_upper', 'boll_lower']]
# print(flat_period_analysis)  ## 1496 days 


""" Analysis: 

The entry condition was met 629 times.
The exit condition was met 402 times.
There are 307 potential settlement days (Wednesdays).
There are significant flat periods (1496 days), 
indicating periods with no trades or no significant changes in strategy returns.
""" 

##############################################
""" Revision : 
1. 調整進場條件：

將散戶多空比的進場閾值從-0.06放寬至-0.05。
增加RSI指標條件:當RSI低於40時進場做多。

2. 調整出場條件：

將散戶多空比的出場閾值從0.06放寬至0.05。
增加RSI指標條件:當RSI高於60時出場。

3. 新增技術指標：

增加布林通道和RSI指標 --> 以幫助更精確地確定進場和出場點。

4. 處理平坦期間：

通過增加交易頻率來減少平坦期間。
""" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from talib import abstract
import warnings
warnings.filterwarnings("ignore")
from FinMind.data import DataLoader

# Data loading
api = DataLoader()
df_ii = api.taiwan_futures_institutional_investors(
    data_id='TX',
    start_date='2018-06-05',
)

df_ii.drop(['futures_id'], axis=1, inplace=True)
df_ii.index = pd.to_datetime(df_ii.date)
df_ii.drop(['date'], axis=1, inplace=True)


# Pivot the institutional investors data
df_ii_pivot = df_ii.pivot_table(index=df_ii.index, columns='institutional_investors', aggfunc='sum').reset_index()
df_ii_pivot.columns = ['_'.join(col).strip() for col in df_ii_pivot.columns.values]
df_ii_pivot.rename(columns={'index_': 'date'}, inplace=True)
# Load main data
file_path = 'TXF.pkl'
df = pd.read_pickle(file_path)
df.index = pd.to_datetime(df.index)
df.drop(['簡稱', '期貨名稱'], axis=1, inplace=True)
df.columns = ['open', 'high', 'low', 'close', 'returns', 'volume', 'open_int', 'basis', 'tot_unsettled']

# Merge dataframes
df = df[df.index >= '2018-06-05']
df = pd.concat([df, df_ii_pivot], axis=1)

# Calculate retail positions
df['retail_long_positions'] = df['tot_unsettled'] - df[['long_open_interest_balance_volume_外資', 'long_open_interest_balance_volume_投信', 'long_open_interest_balance_volume_自營商']].sum(axis=1)
df['retail_short_positions'] = df['tot_unsettled'] - df[['short_open_interest_balance_volume_外資', 'short_open_interest_balance_volume_投信', 'short_open_interest_balance_volume_自營商']].sum(axis=1)
df['retail_long_short_ratio'] = (df['retail_long_positions'] - df['retail_short_positions']) / df['tot_unsettled']

# Add technical indicators
df['rsi'] = talib.RSI(df['close'], timeperiod=14)
df['boll_upper'], df['boll_middle'], df['boll_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)

# Revised trading strategy
def revised_trading_strategy(df):
    position = 0
    entry_ratio = 0

    df['signal'] = 0
    df['position_size'] = 0

    for i in range(1, len(df) - 1):
        if df['retail_long_short_ratio'].iloc[i] < -0.05 and df.index[i].weekday() != 4 and df['rsi'].iloc[i] < 40:
            if position == 0:
                df['signal'].iloc[i + 1] = 1
                df['position_size'].iloc[i + 1] = 1
                position = 1
                entry_ratio = df['retail_long_short_ratio'].iloc[i]
            elif df['retail_long_short_ratio'].iloc[i] < entry_ratio + 0.04:
                df['signal'].iloc[i + 1] = 1
                df['position_size'].iloc[i + 1] += 1

        if df['retail_long_short_ratio'].iloc[i] > 0.05 or df.index[i].weekday() == 4 or df['rsi'].iloc[i] > 60:
            if position == 1:
                df['signal'].iloc[i + 1] = -1
                df['position_size'].iloc[i + 1] = 0
                position = 0

    return df

# Apply revised strategy
df = revised_trading_strategy(df)

# Calculate strategy returns
df['strategy_ret'] = df['signal'].shift(1) * df['position_size'].shift(1) * df['close'].pct_change()
df['cum_strategy_ret'] = (1 + df['strategy_ret']).cumprod() - 1
df['cum_ret'] = (1 + df['close'].pct_change()).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(df['cum_ret'], label='Cumulative Market Return')
plt.plot(df['cum_strategy_ret'], label='Cumulative Strategy Return')
plt.legend()
plt.title('Cumulative Returns')
plt.show()

# Plot maximum drawdown
MDD_series = - (df['cum_strategy_ret'].cummax() - df['cum_strategy_ret']) / df['cum_strategy_ret'].cummax()
MDD_series.plot(figsize=(16, 6), lw=2)
plt.title('Maximum Drawdown', fontsize=16)
plt.show()

# Define performance metrics calculation
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

    risk_free_rate = 0.01
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    metrics['Annual Sharpe Ratio'] = sharpe_ratio

    downside_std = daily_return[daily_return < 0].std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std
    metrics['Risk-Reward Ratio'] = sortino_ratio

    return metrics

# Calculate performance metrics
performance_metrics = calculate_performance_metrics(df)

# Display results
for key, value in performance_metrics.items():
    print(f"{key}: {value}")

# Plot cumulative returns and maximum drawdown
fig, ax = plt.subplots(figsize=(12, 6))
df['cum_strategy_ret'].plot(label='Total Return', ax=ax, c='r')
plt.fill_between(MDD_series.index, MDD_series, 0, label='DD', color='gray', alpha=0.3)
plt.scatter(df.index, df['cum_strategy_ret'].loc[df.index], c='#02ff0f', label='High')
plt.legend()
plt.ylabel('Return%')
plt.xlabel('Date')
plt.title('Return & MDD', fontsize=16)
plt.show()
