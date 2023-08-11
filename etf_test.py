#!/usr/bin/env python
# coding=utf-8
# @author  Yin Tang
# @date 2023.07.31
'''
    在agent策略基础上，尝试整个etf去测试改策略。所不同在于，agent或不同的策略面向单个股票做测试。
    而本文件，则是按etf的某个组合来给出action。action针对的是etf本身。即：
    每天下载sz50成分股50支股票的20天数据，组合给出action，
    action 针对sz50的etf给出信号
    持续20年（2000-2022）
    观察50etf的走势+资金曲线。
    注：当前env.py是针对一个股票，所以一日内得循环50次获取数据，或者说fork 50个env才行，这种做法为以后做好了准备。
    或者先挨个获取单个股票数据，计算action，循环50个股票，然后存盘。这样需要做的只是对这些文件，按每天的分数*权重进行投票，这种仅作为验证。
    其余逻辑均已经在Agent中表达。
'''

import numpy as np
import pandas as pd
from datetime import datetime
from data import BaostockDataWorker
from preprocess import Preprocessor
from globals import test_result, summary, etf_action
from pathlib import Path
import os
from functools import reduce
from evaluate import Evaluator
class Etf():
    def __init__(self, df: pd.DataFrame, etf: str):
        # df.columns = ['code', 'name','weight', 'sector']
        # 分别代表：股票代码，股票名称，权重，所属板块的指数代码
        # 对于“所属板块的指数代码”，如果该股票实在找不到对应的板块或者无法获取其代码，可以用大盘的指数来代替。
        self.stock_list = df.sort_values(by='code', ascending=True).reindex()
        self.dataworker = BaostockDataWorker()
        self.preprcessor = Preprocessor()
        self.window_size = 20
        self.etf = etf

        def __prepare__(c:str, ktype='5')-> pd.DataFrame:
            # 获取所有股票当天的数据，这样其他函数只需要做计算即可。days取win_size的3倍，应该足够做一些ma,diff,dropna等操作了
            # c : code of etf, eg. 'sz.510050'
            d1 = self.dataworker.latest(c, ktype=ktype, days = self.window_size * 5) 
            d2 = self.preprcessor.load(d1).bundle_process()
            return d2
        
        # 准备好stocks, sectors, markets的数据
        self.etf_close = __prepare__(etf, ktype='d')
        
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
        
    def strength(self,df:pd.DataFrame): # v1.2
        ''' 输入股票列表的一行（code 及其对应的sector和market），计算其强弱分数
        record : DataFrame的一行
        具体做法是：
        - 股票量能：根据ticker今天、昨天以致更远的分钟线进行打分，权重30%；
        - 板块量能：根据ticker所在板块的强弱打分，权重30%；
        - 大盘量能：根据大盘的强弱打分，权重20%；
        - 股票惯性：根据股票昨天的涨跌打分，权重20%.
        '''
        # df = pd.merge(df[['ticker','action']], self.stock_list[["code", "weight"]], left_on="ticker", right_on="code", how="left")
        df = df[['ticker','action']].merge(self.stock_list[["code", "weight"]], left_on="ticker", right_on="code", how="left")
        # df = df.dropna()    # 将存在空值的行删除。一般不会为空，因为df的ticker原本来自self.stock_list中的code
        score = np.dot(df.action, df.weight)
        action = lambda score: 1 if score > 80 else -1 if score < 50 else 0
        return action(score)   
        # return score
    
    @staticmethod
    def _reward(self, a:int, df_slice:pd.DataFrame):
        ''' copied from env.py, remember to maintain the same
        '''
        if a is None: return 0
        assert a in [-1, 0, 1], "invalid action " + str(a)
        column = {1:"buy_reward",0:"hold_reward", -1:"sell_reward"}[a]
        return df_slice.iloc[-1][column]
    
    def test_etf(self, obj):
        '''所有股票，全周期，并按照日期对齐，按weight组合计算etf的action
        1 读取所有生成的.csv到df中
        2 按日期对齐，并附上weight
        3 用vote()，计算每一天的etf_action
        4 写入sz50etf.csv
        '''
        data_folder = Path(f"{test_result}/{obj}")
        files = os.listdir(str(data_folder))  # 目录下所有文件,
        files = [f for f in files if os.path.splitext(f)[1] == '.csv']  # 只选择 .csv 文件,
        if summary in files: 
            files.remove(summary)
        if etf_action in files: 
            files.remove(etf_action)  # 如果已经有了要去掉
        print(f"Testing ETF strategy {obj} with {files}")
        # 读取所有files，组成一个dataframe或者list of dataframe
        df_list = [pd.read_csv(data_folder/f, index_col=0 ,header=0) for f in files]
        df_list = reduce(lambda a,b: pd.concat([a,b]), df_list)
        df_list = df_list.sort_values(by=['date', 'ticker']).reset_index(drop=True) # 按时间-股票对排序,并且去掉index
        days = df_list.date.drop_duplicates()   # 去掉重复的日期
        ####
        etf = pd.DataFrame()
        for day in days:                        # 按日循环计算action, reward, close
            df = df_list[df_list.date == day]   # 每天的股票列表，包含字段['ticker', 'date', 'close', 'action', 'reward', 'change_wo_short','change_w_short']
            action = self.strength(df)          # 根据每个股票的action和weight计算etf的action
            reward = self._reward(action, df)   
            close = self.etf_close[self.etf_close.date == day].close.iloc[0]
            etf = etf.append([self.etf, day, close, action, reward])    # 'change_wo_short','change_w_short' 置空 

        ev = Evaluator(etf)     # 增加'change_wo_short','change_w_short'字段
        ev.asset_change().df.to_csv(data_folder/f'{etf_action}')  # 保存asset_change()的结果到etf_action
        print(ev.class_perf())
    

if __name__ == '__main__':
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    # df = df.sample(2) # 开发时用，测试时候将此行注释即可
    Etf(df, "sz.510050").test_etf("Kdj")