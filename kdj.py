#!/usr/bin/env python
#coding=utf-8
# @date 2023.01.26

import pandas as pd
from backtest import test, inference
from agent import Agent
from utils import super_smoother
from globals import WINDOW_SIZE
import stockstats

def kdj_diff(d:pd.DataFrame)->pd.DataFrame:
    ''' 
        Calculates the difference between the 'kdjj' and 'kdjd' columns of the given DataFrame.

        Parameters:
        d (pd.DataFrame): The DataFrame containing the 'kdjj' and 'kdjd' columns.

        Returns:
        pd.DataFrame: The DataFrame with the calculated difference between 'kdjj' and 'kdjd'.
    '''
    if not isinstance(d, (pd.DataFrame,stockstats.StockDataFrame)):
        raise TypeError("must be pd.DataFrame or stockstats.StockDataFrame")
    if 'kdjj' not in d.columns or 'kdjd' not in d.columns:
        raise ValueError("The DataFrame must contain the 'kdjj' and 'kdjd' columns.")
    if d.empty:
        raise ValueError("The DataFrame must not be empty.")

    kr = d['kdjj'] - d['kdjd']
    # kr = (kr-kr.mean())/kr.std()    # z-score normalization
    ####### 
    kr_ = kr.fillna(0).shift(1).fillna(0)
    flex = kr[kr*kr_ <0].index      # 拐点
    ###
    flex = list(flex)
    flex.insert(0,0)    # flex[:0] = [0]   flex = [0]+flex  # 补回第一个
    flex.append(kr.index[-1])  # flex = flex+[kr.index[-1]] #补回最后那个，flex.loc[len(flex)] = [kr.index[-1]]
    ###
    flex = pd.DataFrame(flex,columns=['a'])
    flex['b'] = flex.a.shift(-1).astype("Int64")
    flex.dropna(inplace=True) #     去掉最后带na的那行。(拐点,下一个拐点)终于有了
    ########
    kr = pd.DataFrame(kr).fillna(0)     # 好像非得是pd.DataFrame，pd.Series都不行?
    def seg_cumsum(x):  # 按拐点分段进行update
        kr.iloc[x.a:x.b] = kr.iloc[x.a:x.b].cumsum()
        return kr.iloc[x.a:x.b]
    
    flex.apply(seg_cumsum, axis=1)
    return kr

def kdj_area(d:pd.DataFrame, knot:int = 10)->pd.DataFrame:
    ''' 
        Calculates the difference between the 'kdjj' and 'kdjd' columns of the given DataFrame.

        Parameters:
        d (pd.DataFrame): The DataFrame containing the 'kdjj' and 'kdjd' columns.

        Returns:
        pd.DataFrame: The DataFrame with the calculated difference between 'kdjj' and 'kdjd'.
    '''
    if not isinstance(d, (pd.DataFrame,stockstats.StockDataFrame)):
        raise TypeError("must be pd.DataFrame or stockstats.StockDataFrame")
    if 'kdjj' not in d.columns or 'kdjd' not in d.columns:
        raise ValueError("The DataFrame must contain the 'kdjj' and 'kdjd' columns.")
    if d.empty:
        raise ValueError("The DataFrame must not be empty.")

    kr = d['kdjj'] - d['kdjd']
    kr = (kr-kr.mean())/kr.std()    # z-score normalization
    kr_ = kr.fillna(0).shift(1).fillna(0)
    flex = kr[kr*kr_ <0].index      # 找出拐点
    if len(flex) < knot:    # 100天都找不到一个拐点的，这股票也不应该考虑的
        print("not enough knots, use the first one")
    knot = min(len(flex),knot)
    kr = kr.loc[flex[-knot]:].cumsum()  # 前knot个拐点进行累加。注意.loc是按索引值，而.iloc按位置，差别很大
    return kr

class Kdj(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'Kdj'
        # if not isinstance(df, (pd.DataFrame,stockstats.StockDataFrame)):
            # raise TypeError("must be pd.DataFrame or stockstats.StockDataFrame")
        # if 'kdjj' not in df.columns or 'kdjd' not in df.columns:
            # raise ValueError("The DataFrame must contain the 'kdjj' and 'kdjd' columns.")
        if df.empty:
            raise ValueError("The DataFrame must not be empty.")
    
    @staticmethod
    def criteria(df:pd.DataFrame)->int:
        '''
        @input d: a dataframe of window_size
        @output : the score
        '''
        kr = kdj_area(df)
        k, d, j = df.kdjk, df.kdjd, df.kdjj
        # jd = super_smoother((j-d).fillna(0).to_numpy(), WINDOW_SIZE)
        jd = j-d
        if jd.iloc[-1] > 0 and jd.iloc[-2]<0 and kr.iloc[-1]<0:
            a = 1
        elif jd.iloc[-1] < 0 and jd.iloc[-2]>0 and kr.iloc[-1]>0:
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