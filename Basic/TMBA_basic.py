import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

a = np.array([[1, 2, 3 , 4, 5, 6]])

print(a.sum(), a.max(), a.mean(), a.cumsum(), a.std())

####################################################

b = np.arange(1, 11, 2)
print(b)

c = np.linspace(1, 10, 5)  # linspace 左右端點包含 

print(c)

########################################################

d = np.full((2, 3), 7)
d

e = np.zeros((2, 3))
e

#######################################################

f = np.array([[1, 2], [3, 4], [5, 6]])
f

f[0, 0], f[0] 

f.shape

f.T  # Transpose 
f.T.shape
######################################################

g = np.arange (0, 60, 5) # 一樣不包含尾
g

######################################################

h = np.arange (0, 60, 5).reshape(3, 4)
h
h.sum()
h.sum(axis = 0)  # 整列 column 加起來
h.sum(axis = 1)  # 整行 row 加起來

#####################################################


i = np.arange(24).reshape(2, 3, 4)  
# 分成兩個 3*4 的矩陣
# 注意括號數量
i 
i.sum(axis = 0)
i.sum(axis = 1)
i.sum(axis = 2)

i**2
i**i

# 注意括號數量 stack 
np.hstack((i, i**2))   # horizontal 
np.vstack((i, i**2))   # vertical 
#####################################################
j = np.arange(6).reshape(2, 3)
j
j > 3 
j[j > 3]   # 只顯示 i > 3 == True的值

np.where(j > 3)  # 顯示出 index 



#########################################

df = pd.DataFrame([10, 20, 30, 40], index = ['a', 'b', 'c', 'd'], columns = ['numbers'])
df
df.index, df.columns

# 找特定一比 loc 
df.loc['b']

# 找範圍 iloc 
df.iloc[1:3]   # 一樣不包含最後一列
df.iloc[0:0]   # 出來會是空值，因為不包含最後一列

####################################
df['score'] = [60, 70, 80, 90]
a = df['score']
df
type(df)
type(a)

#####################################
np.random.standard_normal((5, 4))
df = pd.DataFrame(np.random.standard_normal((5, 4)))
df

df.head(2)
df.tail(2)
df.describe()

#####################################
df.columns = ['a', 'b', 'c', 'd']
df

# pd.date_range : 開始月份, 總共奇數, 以月為單位（ freq = 'M', 'D', 'Y' ）
dates1 = pd.date_range('2020-01-01', periods = 5, freq = 'M')
dates1  # 如果是以月份來說，就會從 設定日期月份的最後一天開始算  

dates2 = pd.date_range('2020-01-01', periods = 5, freq = 'D')
dates2


df.index = dates1

df
df.values
df.sum(axis = 0)
df.sum(axis = 1)
df.cumsum()   # 直的看
df.cumsum(axis = 1)    # 橫的看

###########################################
df = pd.DataFrame(np.random.standard_normal((12, 4)))
dates3  = pd.date_range('2020-01-01', periods = 12, freq = 'M')
df.index = dates3
df
df['Quarter'] = df.index.quarter   # 快速三個一組生成 quarter
df

groups = df.groupby('Quarter')
groups.size()

groups.mean() 
groups.max()

df > 0
df[df > 0] # Nan 部分就是 < 0 
df[0] > 0

###########################################
# Concatenation

df1 = pd.DataFrame(np.random.standard_normal((1000, 4)), columns = ['a', 'b', 'c', 'd'])
df1.head()
df2 = pd.DataFrame(np.random.standard_normal((1000, 4)), columns = ['a', 'b', 'c', 'd'])
df2.head()

# Notice how many rows
pd.concat([df1, df2])   # concate by row 

# notice how many columns 
pd.concat([df1, df2], axis = 1)   # concate by column


###########################################

c = pd.Series([10, 20, 30], index = ['a', 'b', 'c'])
c

df1 = pd.DataFrame(np.arange(100, 900, 100).reshape(4, 2), index = ['a', 'b', 'c', 'd'], columns= ['A', 'B'])
df1

df1['C'] = c 
df1    # 注意 df1 中 'd' index 就會顯示 NAN 

df2 = pd.DataFrame(([200, 200], [100, 100], [50, 50]), index = ['e', 'f', 'g'], columns = ['C', 'B'])
df2

pd.merge( df1, df2 ) 


#############################################
# Matplotlib 
import seaborn as sns
plt.style.use('ggplot')


df3 = pd.DataFrame(np.random.standard_normal((100, 10)))   # 100 rows, 10 columns
df3

df3[0].plot(), df3.plot()

df3.cumsum().plot() # 累積權益曲線 

#############################################

np.random.seed(1000)
y = np.random.standard_normal(20)
y 
x = np.arange(len(y))   #  == np.arange(20)
x
plt.plot(x, y)

############################################
# set up space
plt.figure(figsize = (10, 6))
plt.grid(True)    # 是否要網格 grid  ( grid search 網格搜索)
plt.xlim(-1, 20)
plt.ylim(np.min(y)-1, np.max(y)+1)  # 把真實 y 的數據全部包含進來

# draw line and dots 
plt.plot(x, y, 'o')
plt.plot(y, 'b', lw = 2)
plt.xlabel('index')
plt.ylabel('return')

help(plt.plot)  # to see examples 

################################################
# 把圖畫再一起
plt.twinx()

plt.subplot(2, 1, 1)  # 列數，行數，第幾張圖



##################################################
np.random.seed(40)
y = np.random.standard_normal((20,2)).cumsum(axis =0)
y


plt.figure(figsize=(10,6))
plt.plot(y,'b',lw = 1.5) #lw是線寬
plt.plot(y,'ro') #ro 紅點
plt.xlabel('index')
plt.ylabel('return')
plt.title('A simple plot')


#######################################
plt.figure(figsize=(10,6))
# 以下畫兩條線
plt.plot(y[:,0],lw = 1.5,label = '1st') #lw是線寬
plt.plot(y[:,1],lw = 2.5,label = '2nd')

plt.plot(y,'ro') #ro 紅點
plt.legend(loc = 0) #loc = 0 表示選取最佳位置
plt.xlabel('index')
plt.ylabel('return')
plt.title('A simple plot')

#############################################

plt.figure(figsize=(10,6))
np.random.seed(40)
y = np.random.standard_normal((20,2)).cumsum(axis =0)
y
# 以下根據上方 random 產出 y ， 選取所有行的第 0 列 以及第一列
plt.plot(y[:,0],lw = 1.5,label = '1st') #lw是線寬
plt.plot(y[:,1],lw = 1.5,label = '2nd')
plt.plot(y,'ro') #ro 紅點
plt.legend(loc = 0) #loc = 0 表示選取最佳位置
plt.xlabel('index')
plt.ylabel('return')
plt.title('A simple plot')

################################################
# Subplot 
plt.figure(figsize=(10,6))
plt.subplot(2,1,1) #分別代表 列數、欄數、子圖數
plt.plot(y[:,0],'b',lw = 1.5,label = '1st')
plt.plot(y[:,0],'ro')
plt.legend(loc = 0)
plt.xlabel('index')
plt.ylabel('return 1st')
plt.title('A simple plot')
plt.subplot(2,1,2)
plt.plot(y[:,1],'g',lw = 1.5,label = '2nd')
plt.plot(y[:,1],'ro')
plt.legend(loc = 0)
plt.ylabel('return 2nd')


#######################################
plt.figure(figsize=(10,6))
plt.subplot(3,1,1) #分別代表 列數、欄數、子圖數
plt.scatter(y[:,0],y[:,1],label = '1st')
plt.legend(loc = 8)
plt.xlabel('index')
plt.ylabel('return 1st')
plt.title('A simple plot')
plt.subplot(3,1,2)
plt.bar(x,y[:,1],lw = 1.5,label = '2nd')
plt.legend(loc = 0)
plt.ylabel('return 2nd')
plt.subplot(3,1,3)
plt.hist(y,label=['1st','2nd'],bins = 25)
plt.legend(loc = 0)
plt.xlabel('value')
plt.ylabel('frequqncy')
plt.title("histogram")

##########################################
strike = np.linspace(50,150,24).round(1)
ttm = np.linspace(0.5,2.5,24)
strike,ttm = np.meshgrid(strike,ttm)
iv = (strike-100)**2/(100*strike)/ttm
iv[:5,:3]
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(strike,ttm,iv,rstride = 2,cstride=2,cmap=plt.cm.coolwarm,linewidth = 0.5,antialiased = True)
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')
fig.colorbar(surf,shrink=0.5,aspect=5)


###################################### 
import cufflinks as cf
import plotly.offline as plyo
import plotly.io as pio
pio.renderers.default = "colab"
a = np.random.standard_normal((250,5)).cumsum(axis = 0)
index = pd.date_range('2019-1-1',freq = 'B',periods=len(a))
df = pd.DataFrame(100+5*a,columns=list('abcde'),index = index)
df.head()

# 以下無法顯示
plyo.iplot(df.iplot(asFigure=True))
plyo.iplot(df[['a','b']].iplot(asFigure = True,theme='polar',title='A time series plot',
                              xTitle = 'date',yTitle = 'value',
                              mode ={'a':'markers','b':'lines'},
                              symbol={'a':'circle'},
                              size =3.5,colors = {'a':'blue','b':'red'}))

### 抓取資料
import yfinance as yfin 
plt.style.use('fivethirtyeight')

#################################
## 設定變數
company_list =['麗正','聯電','華泰','台積電','旺宏']
code_list = ['2302','2303','2329','2330','2337']

#讀入data( 使用到zip 把資料 zip 在一起)
for company,code in zip(company_list,code_list):
    print(company,code)
    
    # globals()[company]: This dynamically creates variables with the names of the companies to store their respective stock data.
    globals()[company]= yfin.download('{}.tw'.format(code),start='2004-01-01',end='2024-07-20' )
print('End')

台積電.head()  # globals()[company] 讓variable ( 台積電) 可以直接 呼叫



####################################
## 整理資料
## Adjusted close prices 
adj_close_list=[]    # create empty list 
for company in company_list:
    adj_close_list.append(globals()[company]['Adj Close'])
    
adj_close_list

# create a dataframe to store the adjusted close price
close_df=pd.concat(adj_close_list,axis=1,keys=code_list,join='inner') # here specify how to concat, concat by column, keys is the name of columns


# 這邊的 close_df 已經是 adjusted close price
close_df.tail()
close_df.info()
close_df.describe()


#######################################
# Resample: 
## converting a time series from one frequency to another
## Here, converting daily to monthly

# resample對於日頻率以下的資料，內建是選擇左側的index，如果不改成右側的話，會有bias出現
# 日頻率以上的內建是選擇右側
# last表示選最後一筆資料

# 月k : '1M'
close_df_m = close_df.resample('1m',label='right').last()  # 認為一個月最後一筆資料對整個月有代表性
close_df_m.tail()

# 分k : 'T' 
ts_index =pd.date_range('2024-07-20',periods = 12, freq = 'T') # pd.date_range('start',periods = 12, freq = 'T') is a pandas function to create a range of dates 
ts_index # 時間格式指標創建
np.arange(12)
ts = pd.Series(np.arange(12),index =ts_index) # pd.Series 
ts

#########################################
#default在日頻率以內時，label,closed都是left
ts.resample('5min').sum()

#closed = 'right'，表示第一個區間是23:55:01~00:00:00,第二個區間是00:00:01~00:05:00
ts.resample('5min',closed='right').sum()

#label不影響取值，但影響index的結果
## 取值跟 index 都要使用 ｒｉｇｈｔ來貼合我們要的 join right 
ts.resample('5min',closed='right',label='right').sum()

########################################
# Descriptive statistics
plt.figure(figsize=(6,10))
close_df

# 手動畫出
for col in close_df: # 要逐筆分別畫出來
    #plt.subplot(5, 1, n): This creates a subplot in a 5-row, 1-column grid, where n specifies the position of the subplot.
    plt.subplot(5,1,close_df.columns.to_list().index(col)+1) 
    plt.title(col)
    close_df[col].plot()
plt.tight_layout()

# 自動畫出 
close_df.plot(figsize=(10,12),subplots=True)
########################################
## 不同相關係數的算法 
close_df.corr(method='pearson')  # 2303 & 2330 的相關係數 顯著 、 2303 & 2302 也顯著

close_df.corr(method='kendall')  # 相關性不顯著

close_df.corr(method='spearman')  # 2303 & 2302 顯著 

# 會製 heatmap 
import seaborn as sns
plt.figure(figsize=(9,7.5))
plt.title('Correlation Matrix')
sns.heatmap(close_df.corr(),annot=True,annot_kws={"size":15},cmap='coolwarm')
plt.show()    # 這邊顯示出來的是 pearson method 

########################################

## using ADF to test for a unit root in a time series 
# which is a way to check if time series is stationary or not 
from statsmodels.tsa.stattools import adfuller
# H0 : 序列具有單根 --> it is non stationary  ( 單根很突出 --> 不穩定)
## < 0.05 --> H0 is rejected --> stationary, otherwise not stationary
close_df
for col in close_df:
    dftest = adfuller(close_df[col])[1]
    print('p value for {} is :'.format(col),dftest)
    
##conclusion: all the p values are too big to reject H0, so it is non stationary for now 

########################################
# 進行 ACF 
## Autocorrelation Function: measures the correlation between the values of a time series at different lags. 
## It provides insights into the extent of correlation between observations separated by various time steps.


from statsmodels.graphics.tsaplots import plot_acf
for col in close_df:
    plot_acf(close_df[col], lags= 30, alpha=0.05,title = '{}'.format(col))
plt.tight_layout()

## conclusion: 
# Confidence Intervals: The shaded region (determined by alpha=0.05) represents the 95% confidence interval. 
# Autocorrelations outside this region are considered statistically significant.
# Spikes: Each spike represents the autocorrelation at that specific lag. 
# Significant spikes indicate strong correlations at those lags.

#######################################
plt.figure(figsize=(6,20))
for col in close_df:
    plt.subplot(5,1,close_df.columns.to_list().index(col)+1)
    plt.title(col)


    ## This calculates the percentage change of the stock's adjusted close prices:
    ## periods=1: Specifies that the percentage change is calculated over one period (i.e., daily percentage change if the data is daily).
    ## fill_method="ffill": Specifies the method to fill missing values. "ffill" stands for forward fill, which means that missing values are filled with the last valid observation.
    close_df[col].pct_change(periods=1)
    close_df[col].pct_change(periods=1,fill_method="ffill").plot(lw=1)
    
    
plt.tight_layout()

#######################################
# 解決trend，做一階差分
## trend : 持續上漲或下降，在統計中，我們需要去除這樣的趨勢，才可以使數據更加平穩
## 一階差分是一種常見的方法，用來去除數據中的趨勢。具體做法是計算相鄰兩個時點數據之間的差異 --> 差分
## 一階差分可以顯示數據的變化速率，而不是其絕對水平。這樣能夠更好地捕捉到數據中的季節性變化和隨機波動，而不受長期趨勢的影響。

# pct_change() : percentage change ( 後-前 / 前 )
plt.figure(figsize=(9,7.5))
plt.title('Correlation Matrix');
sns.heatmap(close_df.pct_change(periods=1,fill_method="ffill").corr(),annot=True,annot_kws={"size":15},cmap='coolwarm')
plt.show()

########################################
# 累積回報率計算
## 使用  log return ( log(後/前) ) : 數學上有可加性：計算簡單
rets = np.log(close_df/close_df.shift(1))
rets.tail().round(2) # 四捨五入到小數點第二位

rets.cumsum().apply(np.exp).plot(figsize=(12,6))



########################################
# TALIB ： technical analysis library 技術指標庫
import talib
from talib import abstract
# E-Mini S&P 500
df = yfin.download('SPY', start='2012-01-01', end='2024-07-08').round(3)
df.columns   # Check columns
df.Close.plot(lw=1.5)


# Calculate Simple Moving Average
sma = talib.MA(df['Close'],20,matype = 0)  
# talib.MA : writen function to cal the MA 
# matype = 0 : Simple Moving ; matype = 1 : Exponential Moving
sma.plot()

# ADX : 趨勢指標 adjusted directional index
## range between 0 and 100
## Interpretation of ADX Values : 
""" 0-25: A weak trend or a non-trending market.
25-50: A strong trend.
50-75: A very strong trend.
75-100: An extremely strong trend. """ 
adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
adx.plot(lw = 1.5)


#############################################
# all the index we need 
df.columns = ['high','low','open','close','volume','adj_close'] # 改成開頭小寫
df.head()
sma = abstract.MA(df,20,matype = 0)
adx = abstract.ADX(df,14)
rsi = abstract.RSI(df)
macd = abstract.MACD(df)
macd.plot(lw = 1.5),rsi.plot(lw = 1.5)

## 補充 matype 
""" 0 : sma 簡單移動平均線
1 : ema 指數移動平均線
2 : wma 加權移動平均線
3 : dema 雙重指數移動平均線
4 : tema 三重指數移動平均線
5 : trima 三角移動平均線
6 : kama 考夫曼自適應性移動平均線
7 : mama MESA自適應移動平均線
8 : t3 T3移動平均線  """ 

############################################

# Backtesting 

## Creating Signals by diff SMA ( 20 & 120 一快一慢)

df = yfin.download('SPY', start='2012-01-01', end='2024-07-08').round(3)
df.columns = ['high','low','open','close','volume','adj_close']
df

# 快、慢 ｓｍａ
df['sma1'] = abstract.MA(df,20,matype = 0)
df['sma2'] = abstract.MA(df,120,matype = 0)
df # we can see two sma was added 

# ROW SLICING : use row from 2200 to the end 
df[['close','sma1','sma2']][2200:].plot(figsize=(12,8),lw=3)


# create signal itself ( strategy )
signal = np.zeros([len(df)])
condition1 = (df['sma1']>df['sma2'])
condition2 = (df['sma1']<df['sma2'])
condition1  #  is a df of Bool values 

for i in range(len(df)):
    if condition1[i] :
        signal[i] = 1   # True, then set signalto 1 
    elif condition2[i] :
        signal[i] = -1

signal[:1000]  # see how it goes 


# Why using pd.Series: 
""" 1. convenience
    2. 一維數據結構，比df 更輕量
    3. 可以跟df 協作  """ 
    
signal_df = pd.Series(signal,index = df.index)
signal_df
plt.figure(figsize=(12,6))
signal_df[:1000].plot()
#################################

# Simulation for real world trading : cannot make an execution on exactly when the signal was triggered.
# Avoid look- ahead bias


## Shift it down by One Period 
# Position : 部位 
""" If signal is triggered on time t, then the position is set to 1 at time t+1  """ 

position_df=signal_df.shift(1)
position_df.plot(figsize=(12,6))



#################################### 
# Performance calculation  

df['position'] = position_df
df = df[['close','sma1','sma2','position']].dropna()
df['ret'] = df['close'].pct_change()
df['cum_ret'] = df['ret'].cumsum()
df


# two ways to calculate 
# 兩種算法，前者可針對不同報酬如借券利息等細部計算

## 1. 
strategy_ret = np.zeros(len(df))
for i in range(len(df)):
    if df['position'][i] == 1:
        strategy_ret[i] = df['ret'][i]*df['position'][i]
    elif df['position'][i] == -1 :
        strategy_ret[i] = df['ret'][i]*df['position'][i]

## 2. 
strategy_ret = df['ret']*df['position']
df['strategy_ret'] = strategy_ret
df['cum_strategy_ret'] = df['strategy_ret'].cumsum()
df.tail(20)

##########################################
# 績效指標

## 比較單獨持有和執行策略的損益 ( cumulative return  vs. cum strat return )

fig,ax=plt.subplots(figsize=(16,6))
df[['cum_strategy_ret','cum_ret']].plot(label='Total Return',ax=ax)
plt.legend()
plt.title('Stock & Total Return',fontsize=16)


##############################################
## 建累機報酬率的df
## 並且以.cummax()來計算累積報酬創高點

fig=plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
df['cum_strategy_ret'].cummax().plot()
plt.title('Cummax Return')
plt.subplot(1,2,2)
df['cum_strategy_ret'].plot()
plt.title('Cumulative Return')

# Conclusion : 
""" 比較 1. 累積報酬創高點    2. 累積報酬率
    使用cummax 發現自從2015 以後就沒有創新高點了，最後也沒有再cumulate 到前高 """ 
###########################
# Max Drawdown : 出來會是正的值（cummax - 一般值）
MDD_series=df['cum_strategy_ret'].cummax()-df['cum_strategy_ret']
MDD_series.plot(lw = 2)
###########################
# high index : 創新高
## 只返回cum_strategy_ret 中， cum_strategy_ret == cummax 的點 （這個判斷會回傳 Bool) 帶入df 則只會回傳 True 的點
high_index = df['cum_strategy_ret'][df['cum_strategy_ret'].cummax()==df['cum_strategy_ret']].index 


############################
#  最基本 backtesting result 圖
fig,ax=plt.subplots(figsize=(12,6))
df['cum_strategy_ret'].plot(label='Total Return',ax=ax,c='r', lw = 1.5)
plt.fill_between(MDD_series.index,-MDD_series,0,label='DD')
plt.scatter(high_index,df['cum_strategy_ret'].loc[high_index],c='#02ff0f',label='High', s = 100)


plt.legend()
plt.ylabel('Return%')
plt.xlabel('Date')
plt.title('Return & MDD',fontsize=16)

#########################################
# 基本指標計算

def calculate_performance_metrics(df, cum_return_col='cum_strategy_ret'):
    # 計算最大回撤
    MDD_series = (df[cum_return_col].cummax() - df[cum_return_col]) / df[cum_return_col].cummax()
    MDD = round(MDD_series.max(), 2) * 100
    
    # 計算累積回報
    Cumulative_Return = round(df[cum_return_col].iloc[-1], 2) * 100
    
    # 計算回撤回報比
    Return_on_MDD = round(df[cum_return_col].iloc[-1] / MDD_series.max(), 2)
    
    # 計算每日回報
    daily_return = df[cum_return_col].diff(1)
    
    return {
        "Maximum Drawdown (MDD)": MDD,
        "Cumulative Return": Cumulative_Return,
        "Return on MDD": Return_on_MDD,
        "Daily Return": daily_return
    }

# 假設 df 是包含策略回報數據的 DataFrame
performance_metrics = calculate_performance_metrics(df)

# 顯示結果
for key, value in performance_metrics.items():
    if key != "Daily Return":  # 日回報太長不列印
        print(f"{key}: {value}")

####################################
# TMBA 計算方法

MDD=round(MDD_series.max(),2)*100
Cumulative_Return=round(df['cum_strategy_ret'].iloc[-1],2)*100
Return_on_MDD=round(df['cum_strategy_ret'].iloc[-1]/MDD_series.max(),2)
daily_return=df['cum_strategy_ret'].diff(1)

print('Cumulative Return: {}%'.format(Cumulative_Return))
print('MDD: {}%'.format(MDD))
print('Return on MDD: {}'.format(Return_on_MDD))
print('Shapre Ratio: {}'.format(round((daily_return.mean()/daily_return.std())*pow(252,0.5),2)))





###############################################

# Merge 用法補充：類似SQL 中的join
""" 1. inner: 只保留交集
    2. outer: 保留全部, 缺失值會是NaN
    3. left: 保留左邊all,缺失值會是NaN
    4. right: 保留右邊all,缺失值會是NaN """ 
    
import pandas as pd

# 以下兩個重複的部分會是 'B', 'D' 
df1 = pd.DataFrame({
    'key': ['A', 'B', 'C', 'D'],
    'value1': [1, 2, 3, 4]
})

df1 

df2 = pd.DataFrame({      
    'key': ['B', 'D', 'E', 'F'],
    'value2': [5, 6, 7, 8]
})
df2


# 使用 inner join：聯集
merged_df = pd.merge(df1, df2, on='key', how='inner')
merged_df

# outer join ：交集
outer_merge = pd.merge(df1, df2, on='key', how='outer')
outer_merge

# left join: 保留左邊所有 ( a, b, c, d )
left_merge = pd.merge(df1, df2, on='key', how='left')
left_merge

# right join: 保留右邊所有 ( b, d, e, f )
right_merge = pd.merge(df1, df2, on='key', how='right')
right_merge

# Example 
import pandas as pd
import numpy as np
from talib import abstract

# 示例数据
df_stock = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
    'close': [100, 101, 102, 103, 104]
})
df_stock

df_signal = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
    'signal': [0, 1, -1, 1, 0]
})
df_signal


# 合并数据框
df_merged = pd.merge(df_stock, df_signal, on='date', how='inner')
merge = df_merged
merge.columns

merge['return'] = merge['close'].pct_change()
merge['strategy_return'] = merge['signal'].shift(1) * merge['return']

# 计算累积收益率
""" 累积乘积:

.cumprod()：计算累积乘积。对于增长因子序列 1.02, 1.03, 0.97
累积乘积会按顺序计算：1.02, 1.02 * 1.03, 1.02 * 1.03 * 0.97。 """ 

df_merged['cumulative_return'] = (1 + df_merged['return']).cumprod() - 1
df_merged['cumulative_strategy_return'] = (1 + df_merged['strategy_return']).cumprod() - 1

df_merged

## Evaluation Metrics
# 绩效指标计算函数
def calculate_performance_metrics(df, strategy_col='cumulative_strategy_return'):
    # 计算最大回撤
    cumulative = df[strategy_col]
    max_drawdown = (cumulative.cummax() - cumulative).max()
    
    # 计算夏普比率
    sharpe_ratio = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252)
    
    return {
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }

# 计算绩效指标
performance_metrics = calculate_performance_metrics(df_merged)

print("\nPerformance Metrics:")
print(performance_metrics)

# 绘制累积收益率图
plt.figure(figsize=(12, 6))
plt.plot(df_merged['date'], df_merged['cumulative_return'], label='Cumulative Return')
plt.plot(df_merged['date'], df_merged['cumulative_strategy_return'], label='Cumulative Strategy Return')
plt.fill_between(df_merged['date'], 0, (df_merged['cumulative_strategy_return'].cummax() - df_merged['cumulative_strategy_return']), color='gray', alpha=0.3, label='Drawdown')
plt.legend()
plt.title('Cumulative Returns and Drawdown')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()


###################################################

## Shift 的問題 
# shift(1) : 為整體資料往 t+1 移動一格 (會在tail多出1格)
# shift(-1): 為整體資料往 t-1 移動一格 (會在head多出1格)


###################################################
## 常用函式庫
import math 

# math.sqrt(x)：計算平方根
math.sqrt(9) 

# math.exp(x)：計算指數
math.exp(2)

# math.log(x, base)：計算對數 (base 莫認為 2 )
math.log(9, 3)

# math.factorial(x)：計算階乘
math.factorial(5)

import datetime as dt 

# now
dt.datetime.now()

# str --> datetime
dt.datetime.strptime('2023-01-01', '%Y-%m-%d')

# datetime --> str
dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')

# datetime --> timestamp
dt.datetime.timestamp(dt.datetime.now())

# 時間區隔
dt.timedelta(days=1)
dt.timedelta(hours=1)


##################################################
# bokeh 
import bokeh
from bokeh.plotting import figure, output_file, show

p = figure(title = "Example", x_axis_label = "x", y_axis_label = "y")

p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], legend_label="Temp.", line_width=2)

# 指定输出文件 ( 開啟html 網頁呈現)
output_file("test.html")

# 展示图表
show(p)

## e.g.2 ColumnDataSource
from bokeh.models import ColumnDataSource
data = {'x': [1, 2, 3, 4, 5], 'y': [6, 7, 2, 4, 5]}
df = pd.DataFrame(data)

# 将 DataFrame 转换为 ColumnDataSource
source = ColumnDataSource(df)

# 创建图表对象
p = figure(title="Test Figure")

# 使用 ColumnDataSource 添加数据到图表
p.circle(x='x', y='y', source=source, size=10)

# 展示图表
show(p)

#####################################

## pyportfolio  --> cannot use 
import pydantic
import pyportfolio as pf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



######################################

### Scipy 
from scipy import stats, integrate

# define a func 
def f(x):
    return x**2

## calculate integration 
result, error = integrate.quad(f, 0, 1)
print(f"积分结果: {result}, 误差: {error}")


##  optimization ( min, max )
from scipy import optimize

def f(x):
    return x**2 + 10 * np.sin(x)

# 使用 BFGS 算法最小化函数
result = optimize.minimize(f, x0=0)
print(f"最小化结果: {result}")



## ODE: ordinary differential equation
from scipy.integrate import odeint
import numpy as np

# 定义微分方程
def model(y, t):
    dydt = -y
    return dydt

# 初始条件
y0 = 5
# 时间点
t = np.linspace(0, 20, 100)
# 解方程
y = odeint(model, y0, t)

# 绘图
import matplotlib.pyplot as plt
plt.plot(t, y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()

##################################

### time 
import time 
current_time = time.localtime()
current_time

## format time  ( time --> str )
formatted = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
formatted

## str --> time
time.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
