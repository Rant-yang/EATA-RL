#!/usr/bin/env python
# coding=utf-8
# @author  Yin Tang
# @date 2023.07.26

import numpy as np
import pandas as pd
from datetime import datetime
from data import BaostockDataWorker
from preprocess import Preprocessor

class Agent():
    def __init__(self, df: pd.DataFrame):
        # df.columns = ['code', 'name','weight', 'sector']
        # 分别代表：股票代码，股票名称，权重，所属板块的指数代码
        # 对于“所属板块的指数代码”，如果该股票实在找不到对应的板块或者无法获取其代码，可以用大盘的指数来代替。
        self.stock_list = df
        self.dataworker = BaostockDataWorker()
        self.preprcessor = Preprocessor()
        self.window_size = 20

        def __prepare__(s:pd.Series, ktype='5')-> pd.DataFrame:
            # 获取所有股票当天的数据，这样其他函数只需要做计算即可。days取win_size的3倍，应该足够做一些ma,diff,dropna等操作了
            d1 = [self.dataworker.latest(c, ktype=ktype, days = self.window_size * 5) for c in s] # a list of df
            d2 = [self.preprcessor.load(s).bundle_process() for s in d1] # 对每个df做预处理
            return d2
        
        # 准备好stocks, sectors, markets的数据
        self.stocks_datum = __prepare__(self.stock_list.code, ktype='d')
        self.sectors_datum = __prepare__(self.stock_list.sector, ktype='d')
        self.stock_list['market'] = self.stock_list.code.apply(self.get_market) # 多加一个字段为了后面merge
        self.market_codes = self.stock_list.market.drop_duplicates() # 去重，对sz50来说，就剩1个"sh.000001"     
        self.market_datum = __prepare__(self.market_codes, ktype='d') # 准备好大盘数据
        
    @staticmethod
    def market_of(self, ticker:str) -> str:
        '''根据股票代码，返回其所在的大盘指数代码
        http://baostock.com/baostock/index.php/指数数据
        综合指数，例如：sh.000001 上证指数，sz.399106 深证综指 等；
        规模指数，例如：sh.000016 上证50，sh.000300 沪深300，sh.000905 中证500，sz.399001 深证成指等；
        注意指数没有分钟线数据... ...怎么办？
        ie. 'sh.000023' goes to 'sh.000001' # 上证综指 
            'sz.300333' goes to 'sz.399106' # 深圳综指
            'hk.00700' goes to 'HSI'        # 恒生指数
            'us.######' goes to 'NASDAQ' or 'DJX' 
        '''
        market = ticker.split(".")[0]
        # match market:   # requires python 3.10 or higher version
        #     case 'sh': 'sh.000001'
        #     case 'sz':  'sz.399106'
        #     case 'hk':  'HSI'
        #     case 'us':  'DJX'
        if market == 'sh': mkt = 'sh.000001'
        elif market == 'sz': mkt = 'sz.399106'
        elif market == 'hk': mkt = 'HSI'
        else: print("invalid market label")

        return mkt 

    def get_market(self, ticker:str)->str:
        '''都是上证的股票，都是同一个大盘。因此直接返回sh.000001即可'''
        return "sh.000001"

    def stock_momentum(self): # v1.2
        '''股票涨跌惯性，根据昨天的涨跌定义今天的惯性，涨：1，跌：-1
        '''
        # 用diff(1)获得正负号。  
        # self.stocks['momentum'] = 1 if self.stocks.close.diff(1)>0 else -1
        # 当然这里可以sigmoid*2-1函数归到(-1,1)区间，这样就出现了小数
        sig21 = lambda x: 2/(1 + np.exp(-x)) - 1    # sigmod函数
        
        def criteria(d):
          d['date'] = d.date.apply(pd.to_datetime)    # df.resample()要求date字段必须是datetime类型
          d = d.resample("D", on= "date").mean()      # 基于date字段按5日做聚合，求平均，也可以有复杂的计算
          d = d.close.diff(1)                         # 对日线close做差分，>0 则强势，<0则弱势
          return d.iloc[-1]                           # 返回最后一行即可

        # criteria()返回diff(1)的最后一行，sig21的作用是将它投射到[-1,1]
        self.stock_list['stock_momentum'] = [sig21(criteria(s)) for s in self.stocks_datum] 
        return self.stock_list.stock_momentum
        
    def strength(self, w1: float, w2: float, w3: float, w4: float) -> pd.Series:
        ''' 输入股票列表的一行（code 及其对应的sector和market），计算其强弱分数
        record : DataFrame的一行
        具体做法是：
        - 股票量能：根据ticker今天、昨天以致更远的分钟线进行打分，权重30%；
        - 板块量能：根据ticker所在板块的强弱打分，权重30%；
        - 大盘量能：根据大盘的强弱打分，权重20%；
        - 股票惯性：根据股票昨天的涨跌打分，权重20%.
        Args:
            w1 (float): Weight for stock volume score.
            w2 (float): Weight for sector volume score.
            w3 (float): Weight for market volume score.
            w4 (float): Weight for stock momentum score.
        Returns:
            pandas.Series: Strength scores for each stock.

        '''
        self.stock_list['stock_strength'] = [self.criteria(d) for d in self.stocks_datum]   # 股票量能
        self.stock_list['sector_strength'] =[self.criteria(s) for s in self.sectors_datum]  # 板块量能
        x = [self.criteria(m) for m in self.market_datum] # 与self.market_codes 一一对应
        # 计算大盘量能并拼接，这里可以考虑用transform()?
        # 根据指数代码market对应大盘的指数代码code，进行连接
        # y = pd.DataFrame({'market':self.market_codes, 'market_strength':x}) # 拼成一个df
        # self.stock_list = self.stock_list.merge(y, left_on="market", right_on="market", how="left")  # 大盘量能
        self.stock_list['market_strength'] = self.stock_list.market.map({a:b for a,b in zip(self.market_codes, x)})
        self.stock_momentum() # 股票惯性
        # 计算总的strength，权重可以随时调整
        self.stock_list['strength'] = \
                self.stock_list['stock_strength']* w1 \
                + self.stock_list['sector_strength']* w2 \
                + self.stock_list['market_strength']* w3 \
                + self.stock_list['stock_momentum']* w4 

        return self.stock_list['strength'] 
    
# define methods to be overrided in its child
    @staticmethod
    def criteria(d:pd.DataFrame)->int:
        '''
        @input d: window_size的df
        @output : 根据其最后一行的计算返回1/0, simple enough
        示例。一般要求子类自己实现本方法
        '''
        r = d.mean()    # 取20天的平均值试试
        return 1 if r.close_5_ema>r.close_10_ema and r.rsi_24 >50 else -1        

    def vote(self)->int:
        '''输入多个股票代码以及各自的权重，计算etf总的强弱势'''
        s = self.strength(1,0,0,0)
        return np.dot(s, self.stock_list.weight)
    def etf_action(self,score)->int:
        a = 0
        if score > 80:
            a = 1
        elif score < 50:
            a = -1
        return a
    
    def choose_action(cls, s: (pd.DataFrame)) -> int:
        pass