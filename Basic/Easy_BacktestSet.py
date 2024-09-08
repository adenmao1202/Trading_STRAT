import yfinance as yf
import pandas as pd
from pandas_datareader import data
from datetime import datetime

### Yahoo finance Quick Search 
# yf.pdr_override() #以pandasreader常用的格式覆寫

# target_stock = '^GSPC'  #股票代號變數

# start_date = datetime(2014, 1, 1)
# end_date = datetime(2024, 7, 20) #設定資料起訖日期

# df = data.get_data_yahoo([target_stock], start_date, end_date) #將資料放到Dataframe裡面
# df.index = pd.to_datetime(df.index) #將索引欄資料轉換成pandas的時間格式，backtesting才有辦法排序

# type(df)

### 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import talib 
from talib import abstract
import pickle
from statsmodels.tsa.stattools import adfuller

# 載入數據
file_path = 'TXF_1m.pkl'
df = pd.read_pickle(file_path)
df.tail()
# 確保索引日期時間格式正確
df.index = pd.to_datetime(df.index)

start_date = datetime(2010, 1, 4)
end_date = datetime(2024, 7, 11) #設定資料起訖日期


## backtesting 

from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from backtesting.test import SMA
import talib


class SmaCross(Strategy): #交易策略命名為SmaClass，使用backtesting.py的Strategy功能
    n1 = 5 #設定第一條均線日數為5日(周線)
    n2 = 20 #設定第二條均線日數為20日(月線)，這邊的日數可自由調整

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1) #定義第一條均線為sma1，使用backtesting.py的SMA功能算繪
        self.sma2 = self.I(SMA, self.data.Close, self.n2) #定義第二條均線為sma2，使用backtesting.py的SMA功能算繪

    def next(self):
        if crossover(self.sma1, self.sma2): #如果周線衝上月線，表示近期是上漲的，則買入
            self.buy()
        elif crossover(self.sma2, self.sma1): #如果周線再與月線交叉，表示開始下跌了，則賣出
            self.sell()
            




df.index = pd.to_datetime(df.index) #將索引欄資料轉換成pandas的時間格式，backtesting才有辦法排序


test = Backtest(df, SmaCross, cash=100000, commission=.002)
# 指定回測程式為test，在Backtest函數中依序放入(資料來源、策略、現金、手續費)

result = test.run()
#執行回測程式並存到result中


print(result) # 直接print文字結果

