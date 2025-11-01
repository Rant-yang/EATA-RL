import pandas as pd

class SlidingWindowEnv:
    """
    一个管理时序数据滑动窗口的环境。
    它不是一个标准的 Gym 环境，而是一个数据迭代器。
    """
    def __init__(self, df: pd.DataFrame, lookback_period: int, lookahead_period: int, step: int):
        """
        初始化
        :param df: 完整的、按时间排序的数据集
        :param lookback_period: 回看窗口的大小（用于训练/发现公式）
        :param lookahead_period: 展望窗口的大小（用于评估公式）
        :param step: 每次窗口滑动的步长
        """
        self.df = df
        self.lookback_period = lookback_period
        self.lookahead_period = lookahead_period
        self.step = step
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        """
        滑动到下一个窗口，并返回 lookback 和 lookahead 数据。
        """
        window_start = self.current_index
        lookback_end = window_start + self.lookback_period
        lookahead_end = lookback_end + self.lookahead_period

        # 如果展望窗口超出了数据集的末尾，则停止迭代
        if lookahead_end > len(self.df):
            raise StopIteration

        lookback_df = self.df.iloc[window_start:lookback_end].copy()
        lookahead_df = self.df.iloc[lookback_end:lookahead_end].copy()

        # 将窗口向前滑动
        self.current_index += self.step

        return lookback_df, lookahead_df
