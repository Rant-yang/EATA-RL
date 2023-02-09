import sys
import pandas as pd
import numpy as np
from datetime import datetime
from stockstats import StockDataFrame
from utils import attach_to, landmarks_BB
import torch
import warnings
warnings.filterwarnings('ignore')
from data import DataStorage, BaostockDataWorker 
from globals import indicators, OCLHVA, Normed_OCLHVA, REWARD

class Preprocessor():

    def __init__(self,df:pd.DataFrame=None) -> None:
        # self.ds = DataStorage()
        # self.dw = BaostockDataWorker()
        self.df = self.ds.load_raw() if df is None else df

        self.normalization = 'div_pre_close' # 'div_pre_close' | 'div_self' |'standardization' | 'z-score' 
        # 注：最后进行embedding的是[*tech_indicator_list，*Normed_OCLHVA] 这几个字段
        self.windowsize = 20

    def clean(self,df:pd.DataFrame = None):
        return self.__clean_baostock__(df)
        # return self.__clean_tushare__(df)

    def __clean_tushare__(self,df:pd.DataFrame = None):
        ''' 针对tushare数据进行清洗，确保格式统一
        '''
        data_df = self.df if df is None else df
        # 下面两句似乎互相抵消了？
        # data_df = data_df.set_index("trade_date", drop=True)  # 将 trade_date 列设为索引
        # data_df = data_df.reset_index()
        # 更改列名
        # data_df.columns = [["date", "ticker", *oclhva]] 
        # StockDataFrame requires https://github.com/jealous/stockstats/README.md
        # 但实验过用原来的字段StockDataFrame也可以识别，也不是非改不可
        # 注意这里改成了 'tic', 'date', 'volume'，以后均按这个列名
        data_df.rename(columns={"ts_code": "ticker", "trade_date": "date","vol":"volume"}, inplace = True) 
        # 更改tushare date 列数据格式，原来为“20220122”形如“2022-01-22”。对于baostock有可能不一样
        data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x, "%Y%m%d").strftime("%Y-%m-%d")) 
        # 删除为空的数据行，just IN CASE
        data_df = data_df.dropna()
        # 按照date从小到大进行排序
        data_df = data_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        # https://blog.csdn.net/cainiao_python/article/details/123516004
        # .weekday(),.isoweekday(), .strftime("%w"), .dt.dayofweek(), .dt.weekday(), dt.day_name()
        data_df['day'] = data_df.date.apply(pd.to_datetime).dt.day_of_week
        self.df = data_df # 最終還是複製給了self.df啊，前面的工作都白做了
        self.df = data_df.dropna()
        return self  # return 'self' for currying
    
    def __clean_baostock__(self,df = None):
        '''针对baostock数据进行清洗，确保格式统一
        baostock.columns = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume',
       'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'isST']
        '''
        data_df = self.df if df is None else df
        # 下面两句似乎互相抵消了？
        # data_df = data_df.set_index("trade_date", drop=True)  # 将 trade_date 列设为索引
        # data_df = data_df.reset_index()
        # 更改列名
        # data_df.columns = [["date", "ticker", *oclhva]] 
        # StockDataFrame requires https://github.com/jealous/stockstats/README.md
        # 但实验过用原来的字段StockDataFrame也可以识别，也不是非改不可
        # 注意这里改成了 'ticker', 'date', 'volume'，以后的处理统一按这个列名
        # baostock专用
        data_df.rename(columns={"code": "ticker"},inplace=True) 
        # data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x, "%Y%m%d").strftime("%Y-%m-%d")) 
        # 删除为空的数据行
        data_df = data_df.dropna()
        # 按照date从小到大进行排序
        data_df = data_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        # 添加 day 列，一周的第几天
        # https://blog.csdn.net/cainiao_python/article/details/123516004
        # .weekday(),.isoweekday(), .strftime("%w"), 
        # .dt.day_of_week(), .dt.weekday(), dt.day_name()
        data_df['day'] = data_df.date.apply(pd.to_datetime).dt.day_of_week
        self.df = data_df.dropna()
        return self  # return 'self' for currying

    def add_indicators(self,df:pd.DataFrame = None):
        '''add indicators inplace to the dataframe'''
        df = self.df if df is None else df
        sdf = StockDataFrame(df.copy())  # sdf adds some unneccessary fields inplace, fork a copy for whatever it wants to doodle
        # x = sdf[indicators] # [['close_5_ema', 'close_10_ema','rsi']]
        self.df[indicators] = sdf[indicators] #/100 # d/5 - 10的效果反而不好, 不知道为什么
        self.df = self.df.dropna()    # 一旦dropna()，单行数据的indicators基本上是nan
        return self

    def normalize(self,df= None):
        ''' to normalize the designated fields to [0,1)
            however, remember indicators mostly are around [0,100]?
            in that case, we better unify the scales
        '''
        df = self.df if df is None else df
        if self.normalization == 'div_self':# 将ochlva处理为涨跌幅
            df[Normed_OCLHVA] = df[OCLHVA].pct_change(1).applymap('{0:.06f}'.format)     #volume 出现 前一天极小后一天极大时，pct_change会非常大
        elif self.normalization == 'div_pre_close':# 将ochl处理为相较于前一天close的比例，除volume和amount外
            df['open_'] =(df.open-df.pre_close)/df.pre_close
            df['close_'] =(df.close-df.pre_close)/df.pre_close
            df['low_'] =(df.low-df.pre_close)/df.pre_close
            df['high_'] =(df.high-df.pre_close)/df.pre_close
            df["volume_"] = df.volume.pct_change(1)  # with exception
            df["amount_"] = df.amount.pct_change(1)
        elif self.normalization == 'z-score':# do z-score in a sliding window
            d = df[OCLHVA] 
            r = d.rolling(self.windowsize)
            df[Normed_OCLHVA] = (d-r.mean())/r.std()
        elif self.normalization == 'standardization':# do standardization in a sliding window
            d = df[OCLHVA] 
            r = d.rolling(self.windowsize) 
            df[Normed_OCLHVA] = (d-r.min())/(r.max()-r.min())
        elif self.normalization == 'momentum':# do running smooth in a sliding window, where $x_t = theta*x_{t-1} + (1-theta)&x_t$
            d = df[OCLHVA] 
            r = d.rolling(self.windowsize) 
            df[Normed_OCLHVA] = (d-r.min())/(r.max()-r.min())

        # indicators \in [0,100)，而 oclhva_ \in (-10,10)
        # 注意之所以乘以100是因为和indicators的取值范围保持一致，避免不同scale的影响，当然也可以把indicators除以100
        df[Normed_OCLHVA] = df[Normed_OCLHVA] *100
        self.df = df.round(6).dropna() # 保留小数点后6位
        return self

    def landmark(self,df:pd.DataFrame = None):
        df = df if df else self.df
        self.df = attach_to(df,*landmarks_BB(df))
        return self
    
    def embedding(self,df = None):
        df = self.df if df is None else df
        # note : 'df' must be indexed from 0 on, otherwize the slicing might malfunction
        # df.sort_values(by=['date'], inplace=True)
        d = df[[*indicators,*Normed_OCLHVA]] # v1 只embedd这些字段
        # d = df[[*tech_indicator_list,*after_norm, *mkt_indicators, *mkt_after_norm]] # v2
        matrices = [x.values for x in d.rolling(self.windowsize)][self.windowsize-1:]# don't ask me, try it out youself     # embeding rolling_windows
        # matrices = list(map(lambda x:x.values, d.rolling(self.windowsize)))[self.windowsize-1:]# don't ask me, try it out youself
        # an array of windows each containing a matrix for encoding later on

        input_dim = self.windowsize*len(d.columns)
        vae = VAE(input_dim)
        if torch.cuda.is_available(): vae.cuda()
        vae.load_model() 
        def encode(r):
            mu, logvar = vae.encode(r.reshape(input_dim))
            z = vae.reparametrize(mu, logvar).data.numpy()
            return ','.join(str(i) for i in z[0])  # z is a vector of multiple dimensions in hidden space

        df[self.windowsize-1:,['embedding']] = [encode(r) for r in matrices]
        self.df = df.dropna()
        return self
    
    def add_reward_(self, df:pd.DataFrame = None)->pd.DataFrame:
        '''attach reward to the df once for all, no more calculation on the fly
        requirement: df has a `landmark` column
        '''
        d = df if df else self.df.copy()

        dd = d[(d.landmark.isin(["v","^"]))] 
        d1, d2 = dd.iloc[:-1].index, dd.iloc[1:].index  # 获取索引并配对拼接

        def reward_by_length(length, direction):
            '''
            @direction 1:# from bottom to top ; -1: # from top to bottom
            @return: tuple of np.array
            '''
            assert direction in [1,-1], "direction must be either 1 or -1"
            hold_reward = np.sin(np.linspace(-0.5*np.pi,1.5*np.pi,length))                       # 底部和顶部hold，均为-1分，中间为1分
            buy_reward = np.sin(np.linspace(direction*0.5*np.pi,-direction*0.5*np.pi,length))    # 底部买入 1分，顶部买入 -1分
            sell_reward = np.sin(np.linspace(-direction*0.5*np.pi,direction*0.5*np.pi,length))   # 底部卖出 -1分，顶部卖出1分

            return buy_reward, hold_reward, sell_reward

        def attach_rewards(a,b):
            piece = d.loc[a:b]     # df.loc 闭区间；df.iloc开区间
            buy_reward, hold_reward, sell_reward =  reward_by_length(len(piece), 1 if piece.iloc[0].landmark == "v" else -1 )
            # df.loc[a:b,['buy_reward','hold_reward','sell_reward']] \
            # = pd.DataFrame({'buy_reward':buy_reward, "hold_reward":hold_reward, "sell_reward":sell_reward}
            d.loc[a:b,'buy_reward'] = buy_reward
            d.loc[a:b,'hold_reward'] = hold_reward
            d.loc[a:b,'sell_reward'] = sell_reward 
            

        d[REWARD] = 0
        [attach_rewards(x1,x2) for x1,x2 in zip(d1,d2)]

        self.df = d if df is None else self.df

        return self


    def bundle_process(self, if_market=None):
        if if_market: 
            self.normalize()
        else:
            self.clean().landmark().add_reward_().add_indicators() 
        
        return self.df
    
    def load(self):
        self.df = self.ds.load_raw()
        return self.df

    def save(self):
        return self.ds.save_processed(self.df)

if __name__ == "__main__":
    # MAIN_PATH = './'
    df = DataStorage().load_raw()
    final = Preprocessor(df).bundle_process()
    print("final dataframe\n",final)
    DataStorage().save_processed(final) 