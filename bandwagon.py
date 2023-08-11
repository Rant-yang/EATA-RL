#!/usr/bin/env python
# coding=utf-8
# (1)不要炒股票——详解只做ETF一年翻倍的花车理论 - 矩阵的文章 - 知乎 https://zhuanlan.zhihu.com/p/475647897
# @author  Yin Tang, Xiaotong Luo 
# @date 2023.01.26

import numpy as np
import pandas as pd
from agent import Agent

class Bandwagon(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
    
    @staticmethod
    def kdj_diff(d:pd.DataFrame)->pd.DataFrame:
        kr = d['kdjj'] - d['kdjd']
        kr = (kr-kr.mean())/kr.std()    # z-score normalization
        kr_ = kr.fillna(0).shift(1).fillna(0)
        flex = kr[kr*kr_ <0].index      # 拐点
        flex = pd.DataFrame(flex,columns=['a'])
        flex['b'] = flex.a.shift(-1).dropna().astype("Int32")   # 下一个拐点
        flex.dropna(inplace=True)
        top_row = pd.DataFrame({'a':[0], 'b':[flex.a[0]]})
        flex = pd.concat([top_row,flex]).reset_index(drop=True) # (拐点,下一个拐点)终于有了
        ########
        kr = pd.DataFrame(kr).fillna(0)     # 好像非得是pd.DataFrame，pd.Series都不行
        def seg_cumsum(x):  # 按拐点分段进行update
            kr.iloc[x.a:x.b] = kr.iloc[x.a:x.b].cumsum()
            return kr.iloc[x.a:x.b]
       
        flex.apply(seg_cumsum, axis=1)
        return kr

    @staticmethod
    def criteria(d:pd.DataFrame)->int:
        '''
        @input d: window_size的df
        @output : 根据其最后一行的计算返回1/0, simple enough
        '''

        r = d.mean()    # 取20天的平均值试试
        return 1 if r.close_5_ema>r.close_10_ema and r.rsi_24 >50 else -1

    @classmethod
    def choose_action(cls, s: (pd.DataFrame)) -> int:
        ''' action for a single stock, RL compatible'''
        s1, s2, s3 = s  # 将s解析为5分钟线、股票日线、板块日线、大盘日线
        # cls.criteria(s)  # 分钟线用于判断stock_momentum?
        score = cls.criteria(s) # * 0.7 + cls.criteria(s2) * 0.2 + cls.criteria(s3) * 0.1
        if score > 0.8:
            a = 1
        elif score < -0.8:
            a = -1
        else:
            a = 0
        return a 

if __name__ == "__main__":
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    bw = Bandwagon(df)
    score = bw.vote()
    print(f"score = {score}, Buy(1) or Sell(-1)?", bw.etf_action(score))
    print(bw.stock_list)
    print(bw.stocks_datum)

    # bw.save()