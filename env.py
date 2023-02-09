import pandas as pd
import numpy as np
from data import BaostockDataWorker
from preprocess import Preprocessor
from datetime import datetime
import gym
from globals import BASE_COLUMNS_Minute, BASE_COLUMNS, OCLHVA, indicators, REWARD, WEEKDAY
import pysnooper

from retrying import retry

#%%
class StockmarketEnv(gym.Env):

#%%
    def __init__(self, row, days = 2000, window_size=20):
        ''' set of the stocks, algo, and days to trace back
            the file_name must be a xls file and has fields like .code, .sector, .weight
        '''
        super(StockmarketEnv, self).__init__()
        self.dataworker = BaostockDataWorker()
        self.window_size = 20
        
        def prepare(r, days = 2000, window_size=20):
            '''一年交易日约为244日
            分钟线只需要oclhva，日线以上有indicators应该够用了
            '''
            stock5m = self.dataworker.latest(r.code, ktype="5", days = days) # 5分钟线 [OCLHVA]
            self.stock5m = Preprocessor(stock5m).clean().df[BASE_COLUMNS_Minute+OCLHVA] # `day`字段跑哪儿去了？
            stock = self.dataworker.latest(r.code, ktype="d", days = days) # 股票日线[reward, landmark, indicators]
            self.stock = Preprocessor(stock).bundle_process()[BASE_COLUMNS+OCLHVA+REWARD+indicators+WEEKDAY]
            sector = self.dataworker.latest(r.sector, ktype="d", days = days) # 板块日线 [indicators]
            self.sector = Preprocessor(sector).clean().add_indicators().df[BASE_COLUMNS + indicators]
            market_code = self.get_market(r.code)
            market = self.dataworker.latest(market_code, ktype="d", days = days)
            self.market = Preprocessor(market).clean().add_indicators().df[BASE_COLUMNS + indicators] # 大盘日线 [indicators]
            # self.stock_matrices = [x.values for x in stock.rolling(window_size)][window_size-1:] # 获得rolling的矩阵
            # self.sector_matrices = [x.values for x in sector.rolling(window_size)][window_size-1:] # 获得rolling的矩阵
            # self.market_matrices = [x.values for x in market.rolling(window_size)][window_size-1:] # 获得rolling的矩阵
            # alignment for three dataframes or matrices (a)here or (b) in .step() method.
            date_min = self.stock.date.min()
            date_max = self.stock.date.max()
            print(f"Data ready: {r.code} {r['name']}, from {date_min} to {date_max}, {len(self.stock)} days.")
            trade_days = self.dataworker.calendar# 所有交易日基礎上對該股票作過濾
            date_filter = (date_min <= trade_days.calendar_date) & (trade_days.calendar_date <= date_max)
            self.trade_days = trade_days[date_filter]# 所有交易日基礎上對該股票作過濾

        prepare(row, days)    # prepare the data we need

#%%
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        ''' re-start from the first day
        '''
        self.iter = self.window_size - 1
        return self._state()

    def _reward(self, a:int, df_slice:pd.DataFrame):
        if a is None: return 0
        assert a in [-1, 0, 1], "invalid action " + str(a)
        column = {1:"buy_reward",0:"hold_reward", -1:"sell_reward"}[a]
        return df_slice.iloc[-1][column]
    
    def _info(self, df_slice:pd.DataFrame):
        return df_slice.iloc[-1][['ticker','date','close']].to_dict()
#%%
    def _state(self):
        # 使用days作为指针，获取stock, sector, market的指定范围的数据
        # 注意特定股票的交易日，不一定對應days
        days = self.trade_days.iloc[self.iter +1 - self.window_size : self.iter+1].calendar_date # 警惕:df.loc[] 左闭右闭，df.iloc[]左闭右开
        s0 = self.stock5m[self.stock5m.date.isin(days)]  # 分钟线 oclhva
        s1 = self.stock[self.stock.date.isin(days)]    # 股票日线， landmark, reward, indicators
        s2 = self.sector[self.sector.date.isin(days)]   # 板块日线，仅indicators
        s3 = self.market[self.market.date.isin(days)]   # 股市日线，仅indicators
        # 尽量保持原有信息，工作可以交给bandwagon.choose_action(), 由agent自行决定如何使用。
        # s_ = encode(s_)   # 未来还是将state进行编码后返回吧
        return (s0, s1, s2, s3)  # 拼接后再返回

    def step(self, a:int = None):
        self.iter = self.iter if a else self.window_size - 1    # if no action, go back to the first row
        s_ = self._state()
        r = self._reward(a, s_[1])  # reward的计算基于日线，即s1的最后一行
        info = self._info(s_[1])    # 将ticker,价格和交易日期通过info传递
        done = False if self.iter < len(self.trade_days) -1  else True
        self.iter += 1
        return s_, r, done, info
#%%    
    def get_market(self, ticker:str)->str:
        '''都是上证的股票，都是同一个大盘。因此直接返回sh.000001即可'''
        return "sh.000001"
    # a generator version of step()
    # def next_oclh(self):
    #     self.next_ticker() # for j, t in self.next_ticker(): # make next_ticker() and next_oclh() as a whole
    #     for i, row in self.stock.iterrows():
    #         end_of_stock = True if i == len(self.stock)-1 else False # last row of data
    #         lm = self._rest_oclh(row) # future several oclh from this time
    #         done = yield row, lm, end_of_stock  # after yield, hang here for further invoke, then goes to next line
    #         if done and not end_of_stock : break # to received message from invoker as gen.send(msg)
    