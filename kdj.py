#!/usr/bin/env python
# coding=utf-8
# @date 2023.01.26

import pandas as pd
import test
from agent import Agent

class Kdj(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'Kdj'

    @classmethod
    def choose_action(cls, s: (pd.DataFrame)) -> int:
        ''' action for a single stock, RL compatible'''
        s0, s1, s2, s3 = s  # 将s解析为5分钟线、股票日线、板块日线、大盘日线
        # cls.criteria(s0)  # 分钟线用于判断stock_momentum?
        score = cls.criteria(s1) * 0.4 + cls.criteria(s2) * 0.3 + cls.criteria(s3) * 0.3
        if score > 0:
            a = 1
        elif score < 0:
            a = -1
        else:
            a = 0
        
        return a
        
if __name__ == "__main__":
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    test.test(Kdj, df.sample(2))
    test.inference(Kdj, df.sample(2))