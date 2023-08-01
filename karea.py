#!/usr/bin/env python
# coding=utf-8
# @author  Yin Tang
# @date 2023.07.26
# 唠一个异质化另类量化策略：K线面积交易法（盈亏比1.83） - quantkoala的文章 - 知乎
# https://zhuanlan.zhihu.com/p/408778985
# 多头开仓信号：
# （1）下降趋势的“K线面积”达到阈值，之前成立即可
# （2）KDJ指标值大于80
# （3）价格涨超前一根K线最高价加一倍ATR
# 空头开仓信号：
# （1）上涨趋势的“K线面积”达到阈值，之前成立即可
# （2）KDJ指标值小于20
# （3）价格跌破前一根K线最低价减一倍ATR
# 多头/空头出场：ATR跟踪止损止盈

import pandas as pd
import numpy as np
from agent import Agent


class Karea(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        # super(Reflex,self).__init__(df)
        self.__name__ = 'Karea'
        
    @classmethod
    def choose_action(cls, s: (pd.DataFrame)) -> int:
        ''' action for a single stock, RL compatible
        使用该函数时，同样需要传入价格数据、周期长度和阈值作为参数。
        - `prices`是一个包含价格数据的数组或列表（例如收盘价）。
        - `length`是周期长度，用于计算指标。
        - `threshold`是用于判断买卖信号的阈值。
        函数将返回一个信号数组，其中1表示买入信号，-1表示卖出信号，0表示无信号。
        '''
        s0, s1, s2, s3 = s  # 将s解析为5分钟线、股票日线、板块日线、大盘日线
        prices = s1.close
        threshold = 0.5
        length = 5
        reflex = prc_reflex_ehlers(prices, length)
    
        # 检查 reflex 数组是否为空，如果为空，则设置默认值为0
        if len(reflex) == 0:
            reflex = np.array([0])
    
        # signals 是关于买卖信号的数组，其中1表示买入信号，-1表示卖出信号。
        signals = np.zeros(len(prices))
        for i in range(length, len(prices)):
            if reflex[i] > threshold:
                signals[i] = 1
            elif reflex[i] < -threshold:
                signals[i] = -1
        # 在这里，只需要最后一个信号即可
        return signals[-1]
    
        
if __name__ == "__main__":
    import test
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    test.test(Karea, df.sample(2))
    test.inference(Karea, df.sample(2))