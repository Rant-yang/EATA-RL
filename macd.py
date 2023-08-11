#!/usr/bin/env python
# coding=utf-8
# 什么是MACD？MACD指标详解，四大买入形态和四大卖出形态 - 清风徐来的文章 - 知乎 https://zhuanlan.zhihu.com/p/134077409
# 史上最全“MACD”指标详解及用法诠释，太精辟透彻了 - 糖控的文章 - 知乎 https://zhuanlan.zhihu.com/p/130819105
# @author  Yin Tang, Xiaotong Luo 
# @date 2023.01.26

import pandas as pd
from agent import Agent
import test

class Macd(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'Macd'
        
    @staticmethod
    def criteria(d:pd.DataFrame)->int:
        '''
        @input d: window_size的df
        @output : 根据其最后一行的计算返回1/0, simple enough
        '''
        r = d.mean()    # 取20天的平均值试试
        # r = d.close.ewm(span=len(df)).mean().iloc[-1]   # 20天内的指数平均值的最后一行
        return 1 if r.close_5_ema>r.close_10_ema and r.rsi_24 >50 else -1        
    # 以上5个函数，可以替换成对个股的预测，然后再进行投票。
    # 预测时可以采用各种手段(的组合)，例如"MA5>MA10 and RSI>50" etc. (2)
    # 但一般原则是:
    # (1) 个股起码去到日内的信息(1分钟线，5分钟线，15分钟线 etc.)，日线信息往往不够用；
    # (2) 携带大盘和板块信息
    # (3) 可增加策略，不同策略之间也可以有投票机制

    @classmethod
    def choose_action(cls, s: (pd.DataFrame)) -> int:
        ''' action for a single stock, RL compatible'''
        s0, s1, s2, s3 = s  # 将s解析为5分钟线、股票日线、板块日线、大盘日线
        # cls.criteria(s0)  # 分钟线用于判断stock_momentum?
        score = cls.criteria(s1) # * 0.4 + cls.criteria(s2) * 0.3 + cls.criteria(s3) * 0.3
        if score > 0:
            a = 1
        elif score < 0:
            a = -1
        else:
            a = 0
        
        return a

if __name__ == "__main__":
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    test.test(Macd, df.sample(2))
    test.inference(Macd, df.sample(2))