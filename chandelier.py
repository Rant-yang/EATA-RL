#!/usr/bin/env python
# coding=utf-8
# @author  Yin Tang, Xiaotong Luo 
# @date 2023.01.26
'''
Chandelier Exit https://corporatefinanceinstitute.com/resources/equities/chandelier-exit/

The formulas for the two lines are as follows:
- Chandelier Exit Long: n-day Highest High – ATR (n) x Multiplier
- Chandelier Exit Short: n-day Lowest Low + ATR (n) x Multiplier

Where:
- $n$ is the default unit period of 22 or the number that the trader chooses.
- The multiplier is the default 3 Average True Range.
'''

import numpy as np
import pandas as pd
from datetime import datetime
from data import BaostockDataWorker
from preprocess import Preprocessor
import pysnooper

class BaseAgent():
    pass

class Chandelier(BaseAgent):
    def __init__(self, df: pd.DataFrame):
        # df.columns = ['code', 'name','weight', 'sector']
        # 分别代表：股票代码，股票名称，权重，所属板块的指数代码
        # 对于“所属板块的指数代码”，如果该股票实在找不到对应的板块或者无法获取其代码，可以用大盘的指数来代替。
        self.stock_list = df
        self.dataworker = BaostockDataWorker()
        self.window_size = 20

        def __prepare__(s:pd.Series, ktype='5')-> pd.DataFrame:
            # 获取所有股票当天的数据，这样其他函数只需要做计算即可。days取win_size的3倍，应该足够做一些ma,diff,dropna等操作了
            d1 = [self.dataworker.latest(c, ktype=ktype, days = self.window_size * 3) for c in s] # a list of df
            d2 = [Preprocessor(s).bundle_process() for s in d1] # 对每个df做预处理
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
        
    def strength(self): # v1.2
        ''' 输入股票列表的一行（code 及其对应的sector和market），计算其强弱分数
        record : DataFrame的一行
        具体做法是：
        - 股票量能：根据ticker今天、昨天以致更远的分钟线进行打分，权重30%；
        - 板块量能：根据ticker所在板块的强弱打分，权重30%；
        - 大盘量能：根据大盘的强弱打分，权重20%；
        - 股票惯性：根据股票昨天的涨跌打分，权重20%.
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
                self.stock_list['stock_strength']* 0.4 \
                + self.stock_list['sector_strength']*0.3 \
                + self.stock_list['market_strength']*0.2 \
                + self.stock_list['stock_momentum']*0.1 

        return self.stock_list['strength'] 

    # 以上5个函数，可以替换成对个股的预测，然后再进行投票。
    # 预测时可以采用各种手段(的组合)，例如"MA5>MA10 and RSI>50" etc. (2)
    # 但一般原则是:
    # (1) 个股起码去到日内的信息(1分钟线，5分钟线，15分钟线 etc.)，日线信息往往不够用；
    # (2) 携带大盘和板块信息
    # (3) 可增加策略，不同策略之间也可以有投票机制


    def vote(self)->int:
        '''输入多个股票代码以及各自的权重，计算etf总的强弱势'''
        s = self.strength()
        return np.dot(s, self.stock_list.weight)

    def etf_action(self,score):
        a = 0
        if score > 80:
            a = 1
        elif score < 50:
            a = -1
        return a
    
    @staticmethod
    def criteria(d:pd.DataFrame)->int:
        '''
        @input d: window_size的df
        @output : 根据其最后一行的计算返回1/0, simple enough
        '''
        r = d.iloc[-1]
        return 1 if r.close_5_ema>r.close_10_ema and r.rsi >50 else -1

    @classmethod
    def choose_action(cls, d: pd.DataFrame) -> int:
        ''' action for a single stock, RL compatible'''
        s0, s1, s2, s3 = d  # 将d解析为5分钟线、股票日线、板块日线、大盘日线
        # cls.criteria(s0)  # 分钟线用于判断stock_momentum?
        score = cls.criteria(s1) * 0.4 + cls.criteria(s2) * 0.3 + cls.criteria(s3) * 0.3
        if score > 0.15: 
            a = 1
        elif score < 0.5:
            a = -1
        else:
            a = 0
        
        return a
        
    def save(self):
        self.stock_list.to_csv(datetime.now().strftime("%Y-%m-%d.%H.")+"calculated.csv")

if __name__ == "__main__":
    df = pd.read_excel("000016closeweight.xls", dtype={'code':'str'}, header = 0)
    # df['code'] = 'sh.'+df.code
    # df = pd.read_excel("000016closeweight(5).xls", dtype={'code':'str'}, header = 0)
    bw = Chandelier(df)
    score = bw.vote()
    print(f"score = {score}, Buy(1) or Sell(-1)?", bw.etf_action(score))
    print(bw.stock_list)
    print(bw.stocks_datum)

    bw.save()