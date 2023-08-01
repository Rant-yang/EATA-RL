#!/usr/bin/env python
# coding=utf-8
# @author  Yin Tang
# @date 2023.07.26
# @reference 又来唠一个另类异质量化策略：20后的Trendflex策略 - quantkoala的文章 - 知乎
# https://zhuanlan.zhihu.com/p/557480350
import pandas as pd
import numpy as np
from agent import Agent
import test
from utils import super_smoother
from globals import WINDOW_SIZE

def prc_reflex_ehlers(data:np.array, length = WINDOW_SIZE):
    # Gently smooth the data in a SuperSmoother
    filt = super_smoother(data, length)

    # Length is assumed cycle period 
    # slope = (filt[-length - 1] - filt[-1]) / length
    # 计算斜率，上面代码有问题 
    slope = (filt - pd.DataFrame(filt).shift(length).values.squeeze(1))/length
    # print(slope)

    # Sum the differences
    from functools import reduce
    sum = np.zeros_like(filt)
    diff = lambda c: filt[i-length] + c * slope[i] - filt[i-length+c]
    for i in range(length, len(filt)):
        diffs = [diff(c) for c in range(0,length)]
        sum[i] = reduce(lambda a,b: a + b, diffs)
    sum /= length
    # print(sum)
        
    # Normalize in terms of Standard Deviations
    # ms = 0.04 * summation * summation + 0.96 * ms[-1]
    ms = np.zeros_like(filt)
    ms[0] = sum[0]**2
    for i in range(1,len(filt)):
        ms[i] = 0.04*sum[i]**2 + 0.96*ms[i-1]

    # if ms != 0:
    reflex = sum / np.sqrt(ms)
    return reflex

class Reflex(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'Reflex'

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
        prices = s1.close.to_numpy()
        length = WINDOW_SIZE * 2
        reflex = prc_reflex_ehlers(prices, length)
    
        # 检查 reflex 数组是否为空，如果为空，则设置默认值为0
        if len(reflex) == 0:
            reflex = np.array([0])
    
        # signals 是关于买卖信号的数组，其中1表示买入信号，-1表示卖出信号。
        signals = np.zeros(len(prices))
        for i in range(length, len(prices)):
            if reflex[i] > 0:
                signals[i] = 1
            elif reflex[i] < 0:
                signals[i] = -1
        # 在这里，只需要最后一个信号即可
        return signals[-1]
    
        
# if __name__ == "__main__":
    #   df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
#     agent = Reflex(df)
#     score = bw.vote()
#     print(f"score = {score}, Buy(1) or Sell(-1)?", agent.etf_action(score))
#     print(agent.stock_list)
#     print(agent.stocks_datum)
#     # agent.save()

if __name__ == "__main__":
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    test.test(Reflex, df.sample(2))
    test.inference(Reflex, df.sample(2))