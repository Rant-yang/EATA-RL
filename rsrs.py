#!/usr/bin/env python
# coding=utf-8
# @date 2023.08.09
# 复现网红阻力支撑指标RSRS，手把手教你构建大盘择时策略 - quantkoala的文章 - 知乎
# https://zhuanlan.zhihu.com/p/620876365
# (续)复现网红阻力支撑指标RSRS，手把手教你构建大盘择时策略 - quantkoala的文章 - 知乎
# https://zhuanlan.zhihu.com/p/631688107

import pandas as pd
from backtest import test, inference
from agent import Agent
from globals import WINDOW_SIZE
import numpy as np

class RSRS(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'RSRS'

    @staticmethod
    def criteria(df:pd.DataFrame)->int:
        '''
        @input d: a dataframe of window_size
        @output : the score
        '''
        k, d, j = df.kdjk, df.kdjd, df.kdjj
        if (j.iloc[-1] > d.iloc[-1] and j.iloc[-2]<= d.iloc[-2]) or j.iloc[-1] > 100:
            a = 1
        elif (j.iloc[-1] < d.iloc[-1] and j.iloc[-2] >= d.iloc[-2]) or j.iloc[-1] < 10:
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
        # 滑动窗口window_size，计算(low,high)线性回归后的斜率beta
        # 对beta做z-score标准化 (beta-\mu)/\std
        # 区间[-0.7, 0.7]
        # lr = LinearRegression()
        # lr.fit(s1.low, s1.high)
        # beta = lr.coef_[0]
        # r2 = lr.score(s1.low, s1.high)
        # rsrs = r2* (beta - beta.mean())/beta.std()
        slope = lambda x,y : (len(x)*(x*y).sum()-x.sum()*y.sum())/(len(x)*(x**2).sum()-(x.sum())**2)
        rsrs = [slope(ss.low.values, ss.high.values) for ss in s1.rolling(WINDOW_SIZE)] # 不用sklearn了，直接用公式
        rsrs = np.array(rsrs[1:])
        rsrs = (rsrs-rsrs.mean())/rsrs.std()    # z-score normalization
        score =  rsrs[-1]
        return 1 if score > 0.7 else -1 if score < -0.7 else 0
        
if __name__ == "__main__":
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    df = df.sample(2)
    # df.loc[len(df)] = ['sz.510050', '上证50ETF', 0, 'sz.510050']   # code	name	weight	sector,  sz50etf, baostock 没有指数数据
    test(RSRS, df)
    inference(RSRS, df)