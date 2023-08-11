#!/usr/bin/env python
# coding=utf-8
# @date 2023.01.26

import pandas as pd
from backtest import test, inference
from agent import Agent
from utils import super_smoother
from globals import WINDOW_SIZE

class Kdj(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'Kdj'

    def kdj_diff(sdf:pd.DataFrame):
        '''按拐点，分段更新为累加值'''
        sdf['rsi_diff'] = sdf['rsi_6'] - sdf['rsi_12']
        sdf['kdj_diff'] = sdf['kdjj'] - sdf['kdjd']
        sdf['kdj_diff'] = (sdf['kdj_diff'] - sdf['kdj_diff'].mean())/sdf['kdj_diff'].std()
        sdf['rsi_diff'] = (sdf['rsi_diff'] - sdf['rsi_diff'].mean())/sdf['rsi_diff'].std()
        kr = sdf['kr'] = sdf['kdj_diff']
        kr1 = kr.fillna(0).shift(1).fillna(0)
        flex = kr[kr*kr1 <0].index # 找出拐点
        flex = pd.DataFrame(flex,columns=['a'])
        flex['b'] = flex.a.shift(-1).dropna().astype("Int32") # 下一个拐点
        flex.dropna(inplace=True)   # 最后一行为空，去掉
        top_row = pd.DataFrame({'a':[0], 'b':[flex.a[0]]})  # 前面插入一行，补齐第一部分
        flex = pd.concat([top_row,flex]).reset_index(drop=True)

        kr = pd.DataFrame(kr).fillna(0)
        def seg_cumsum(x):  
            kr.iloc[x.a:x.b] = kr.iloc[x.a:x.b].cumsum()
            return kr.iloc[x.a:x.b]
       
        flex.apply(seg_cumsum, axis=1)  # 按拐点分段更新kr
        return kr
    
    @staticmethod
    def criteria(df:pd.Series)->int:
        '''
        @input d: a dataframe of window_size
        @output : the score
        '''
        k, d, j = df.kdjk, df.kdjd, df.kdjj
        jd = super_smoother((j-d).fillna(0).to_numpy(), WINDOW_SIZE)
        jd = pd.Series(jd)
        if jd.iloc[-1] > 0:
            a = 1
        elif jd.iloc[-1] <= 0:
            a = -1
        else:
            a = 0
        return a

    @classmethod
    def choose_action(cls, s: (pd.DataFrame)) -> int:
        ''' action for a single stock, RL compatible'''
        s0, s1, s2, s3 = s  # 将s解析为5分钟线、股票日线、板块日线、大盘日线
        # cls.criteria(s0)  # 分钟线用于判断stock_momentum?
        # score = cls.criteria(s1) * 0.7 + cls.criteria(s2) * 0.2 + cls.criteria(s3) * 0.1
        return cls.criteria(s1) 
        
if __name__ == "__main__":
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    df = df.sample(10)
    # df.loc[len(df)] = ['sz.510050', '上证50ETF', 0, 'sz.510050']   # code	name	weight	sector,  sz50etf, baostock 没有指数数据
    test(Kdj, df)
    inference(Kdj, df)