import yfinance as yf
import pandas as pd
from pandas_datareader import data
from datetime import datetime

# yf.pdr_override() #以pandasreader常用的格式覆寫

# target_stock = 'TSLA'  #股票代號變數

# start_date = datetime(2010, 1, 1)
# end_date = datetime(2020, 6, 30) #設定資料起訖日期

# df = data.get_data_yahoo([target_stock], start_date, end_date) #將資料放到Dataframe裡面
# df.index = pd.to_datetime(df.index) #將索引欄資料轉換成pandas的時間格式，backtesting才有辦法排序

# type(df)

## backtesting 

from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from backtesting.test import SMA
import talib

class mixedStrat(Strategy):
    # 參數
    n1 = 5
    n2 = 20
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    atr_period = 14
    risk_percent = 10  # 風險百分比
    
    def init(self):
        # 移動平均線
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
    
        # ATR (用於止損)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        self.stop_loss = None
    
    
    def set_stop_loss(self, price, atr, risk_percent=10):
        new_stop = price - 2 * atr
        if self.stop_loss is None or new_stop > self.stop_loss:
            self.stop_loss = new_stop
    

    
    def next(self):
        # 確保有足夠的數據來計算指標
        if len(self.data) < max(self.n2, self.rsi_period, self.atr_period):
            return
        
        # 計算倉位大小
        price = self.data.Close[-1]
        atr = self.atr[-1]
        capital = self.equity  # 當前帳戶價值
        position_size = (capital * self.risk_percent / 100) / (2 * atr)
        position_size = max(1, round(position_size))  # 確保至少交易1單位，並四捨五入

        # 交易信號
        if crossover(self.sma1, self.sma2) and self.rsi[-1] < self.rsi_overbought:
            # 買入信號
            self.buy(size=position_size)
            # 設置移動止損
            self.set_stop_loss(price - 2 * atr)
        
        elif crossover(self.sma2, self.sma1) and self.rsi[-1] > self.rsi_oversold:
            # 賣出信號
            self.sell(size=position_size)
            
         # 檢查是否觸及止損
        if self.position.is_long and price <= self.stop_loss:
            self.position.close()
            self.stop_loss = None    
            
            
        
   
            

## Backtest
test = Backtest(df, mixedStrat, cash=1000000, commission=.002)
# 指定回測程式為test，在Backtest函數中依序放入(資料來源、策略、現金、手續費)

result = test.run()
#執行回測程式並存到result中