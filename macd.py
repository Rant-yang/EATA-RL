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