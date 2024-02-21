#!/usr/bin/env python
#coding=utf-8
# @date 2023.01.26

import pandas as pd
from backtest import test, inference
from agent import Agent
import stockstats

class Boll(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'Bolling'
        # if not isinstance(df, (pd.DataFrame,stockstats.StockDataFrame)):
            # raise TypeError("must be pd.DataFrame or stockstats.StockDataFrame")
        # if 'kdjj' not in df.columns or 'kdjd' not in df.columns:
            # raise ValueError("The DataFrame must contain the 'kdjj' and 'kdjd' columns.")
        if df.empty:
            raise ValueError("The DataFrame must not be empty.")
    
    @staticmethod
    def criteria(d:pd.DataFrame)->int:
        '''
        @input d: a dataframe of window_size
        @output : the score
        '''
        if not isinstance(d, (pd.DataFrame,stockstats.StockDataFrame)):
            raise TypeError("must be pd.DataFrame or stockstats.StockDataFrame")
        if 'boll_ub' not in d.columns or 'boll_lb' not in d.columns:
            raise ValueError("The DataFrame must contain the 'boll_ub' and 'boll_lb' columns.")
        if d.empty:
            raise ValueError("The DataFrame must not be empty.")
        
        gap = d.boll_ub - d.boll_lb
        ub_series = (d.low < d.boll_ub.fillna(method="bfill")) & (d.boll_ub.fillna(method="bfill") < d.high)
        lb_series = (d.low < d.boll_lb.fillna(method="bfill")) & (d.boll_lb.fillna(method="bfill") < d.high)

        # 只用到最后一行，要更sophisticated的话，可以考虑整个序列。或者和kdj一起考虑
        if ub_series.iloc[-1]:
            a = -1
        elif lb_series.iloc[-1]:
            a = 1
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
    test(Boll, df)
    inference(Boll, df)