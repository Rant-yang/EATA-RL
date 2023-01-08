#!/usr/bin/env python
# coding=utf-8
# (1)不要炒股票——详解只做ETF一年翻倍的花车理论 - 矩阵的文章 - 知乎 https://zhuanlan.zhihu.com/p/475647897
# (2)赢率 85%的EMA 和 RSI 交易系统 - 威力外汇社区论坛的文章 - 知乎 https://zhuanlan.zhihu.com/p/595923338

import numpy as np
import pandas as pd
from .data import BaostockDataWorker
from .preprocess import Preprocessor

class Bandwagon():
    def __init__(self, df: pd.DataFrame):
        # df.columns = ['code', 'name','weight', 'sector']
        # 分别代表：股票代码，股票名称，权重，所属板块的指数代码
        # 对于“所属板块的指数代码”，如果该股票实在找不到对应的板块或者无法获取其代码，可以用大盘的指数来代替。
        self.dw = BaostockDataWorker()
        self.pre = Preprocessor()
        self.stock_list = df
        self.stocks = self.__prepare__(self.stock_list.code)
        self.sectors = self.__prepare__(self.stock_list.sector)
        markets_code = self.stock_list.code.apply(self.market_of) # 对每个股票获得其大盘指数代码
        markets_code = markets_code.drop_dupulicates() # 去重     
        self.markets = self.__prepare__(markets_code) # 准备好大盘数据
        
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
        
    def __prepare__(self, s:pd.Series)-> pd.DataFrame:
        # 获取所有股票当天的数据，这样其他函数只需要做计算即可
        data = [self.dw.get_data(c) for c in s] # a list of df
        data = [self.pre.bundle_process(s) for s in self.stocks] # 对每个df做预处理
        return data
    
    def stock_strength(self): # v1.1
        '''股票量能的计算，参考上述链接以及Readme.md，入场规则："MA5>MA10 and RSI>50"
        '''
        # 下面代码只是策略之一，以后可以替换为各种其他
        # try apply(), applymap(), other than for-loop
        def criteria(s):
          '''用于apply函数，对每一行做判断
          凡是满足criteria条件的为1，不符合该条件的为0。当然也可以用True/False
          '''
          return 1 if s.ma5>s.ma10 and s.rsi>50 else -1
        
        self.data['stock_strength'] = self.stocks.apply(criteria, axis=1)
                 
    def sector_strength(self, sector): # v1.2
        '''板块量能，注：Baostock目前只能获取板块和指数日线，但不要紧，宏观一点日线也够用
        '''
        criteria = lambda s: 1 if s.ma5>s.ma10 and s.rsi>50 else -1 
                  
        self.stocks['sector_strength'] = self.sectors.apply(criteria, axis=1)

    def market_strength(self): # v1.2
        '''大盘量能，注：Baostock目前只能获取板块和指数日线'''
        criteria = lambda s: 1 if s.ma5>s.ma10 and s.rsi>50 else -1 
          
        # 计算每个market的strength 
        self.markets['strength'] = self.markets.apply(criteria, axis=1)
        # 为stocks增加market字段，填入指数代码
        self.stocks['market'] = self.market_of(self.stocks.code)
        # 根据指数代码market对应大盘的指数代码code，进行连接
        self.stocks.merge(self.markets, left_on="market", right_on="code", how="left")

    def stock_momentum(self, ticker): # v1.2
        '''股票涨跌惯性，根据昨天的涨跌定义今天的惯性，涨：1，跌：-1
        '''
        # 用diff(1)获得正负号。  >0 => 1, <0 => -1
        # self.stocks['momentum'] = 1 if self.stocks.close.diff(1)>0 else -1
        # 当然这里可以sigmoid*2-1函数归到(-1,1)区间，这样就出现了小数
        sig21 = lambda x: 2/(1 + np.exp(-x)) - 1
        self.stocks['momentum'] = sig21(self.stocks.close.diff(1))
        
    def strength(self, record:pd.DataFrame)->int: # v1.2
        ''' 输入一个股票代码，计算其强弱分数
        record : DataFrame的一行
        具体做法是：
        - 股票量能：根据ticker今天、昨天以致更远的分钟线进行打分，权重30%；
        - 板块量能：根据ticker所在板块的强弱打分，权重30%；
        - 大盘量能：根据大盘的强弱打分，权重20%；
        - 股票惯性：根据股票昨天的涨跌打分，权重20%.
        '''
        # 计算从这里开始
        t = record.code[0]
        s = record.sector[0]
        score = self.stock_strength(t)*0.3 + self.sector_strength(s)*0.3 \
            + self.market_strength(t)*0.2 + self.momentum(t)*0.2
        return score 

    # 以上5个函数，可以替换成对个股的预测，然后再进行投票。
    # 预测时可以采用各种手段(的组合)，例如"MA5>MA10 and RSI>50" etc. (2)
    # 但一般原则是:
    # (1) 个股起码去到日内的信息(1分钟线，5分钟线，15分钟线 etc.)，日线信息往往不够用；
    # (2) 携带大盘和板块信息
    # (3) 可增加策略，不同策略之间也可以有投票机制

    def vote(self)->int:
        '''输入多个股票代码以及各自的权重，计算etf总的强弱势'''
        # s = [self.strength(t) for t in self.df.iterrows()]
        s = self.stocks.apply(self.strength, axis = 1)
        return np.dot(s, self.stocks.weight)

    def minute5(self, ticker:str, days:int)-> pd.DataFrame:
        '''输入股票代码和交易日数量，以最新日期为end，返回end-days个交易日的5分钟线'''
        df = self.bao.get(ticker, days)
        return df

if __name__ == 'main':
    df = pd.read_excel("000016closeweight.xls", header = 0)
    bw = Bandwagon(df)
    bw.vote()

