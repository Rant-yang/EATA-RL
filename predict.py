import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs


# 核心改动：直接导入我们改造后的Agent
from agent import Agent
from data import DataStorage # 导入数据存储类
from performance_metrics import TradingMetrics # 导入我们新增的指标计算模块

class Predictor:
    def __init__(self):
        """
        新版预测器，核心职责是初始化和调用Agent。
        """
        self.agent = Agent(df=pd.DataFrame())
        print("🤖 新版 Predictor 初始化完成，内含新版 EATA Agent شه。")

    def predict(self, df: pd.DataFrame, shares_held: int) -> tuple[int, float]:
        """
        使用Agent对单个数据窗口进行预测。
        现在返回一个包含action和rl_reward的元组。
        """
        print("\n[Predictor] -> 调用 Agent.criteria 进行决策...")
        action, rl_reward = self.agent.criteria(df, shares_held=shares_held)
        action_name = {-1: '卖出', 0: '持有', 1: '买入'}[action]
        print(f"[Predictor] <- Agent决策结果: {action} ({action_name}), RL Reward: {rl_reward:.4f}")
        return action, rl_reward


if __name__ == "__main__":
    print("🚀 启动 EATA 项目核心功能测试、回测与评估")
    print("=======================================================")

    try:
        # 1. 初始化Predictor (它会自动创建新的Agent)
        predictor = Predictor()

        # 2. 从 stock.db 加载真实数据
        print("\n[Main] 从 stock.db 加载真实数据...")
        data_storage = DataStorage()
        all_data = data_storage.load_raw()
        
        if all_data.empty:
            raise Exception("数据库中没有找到原始数据(raw_data)。请先运行 import_data.py 导入数据شه。")

        # 3. 选择一支股票进行测试
        if 'code' not in all_data.columns and 'ts_code' in all_data.columns:
            all_data.rename(columns={'ts_code': 'code'}, inplace=True)

        if 'code' not in all_data.columns:
            raise KeyError("数据中既没有找到 'code' 列，也没有找到 'ts_code' 列شه。")
            
        ticker = 'AAPL' # all_data['code'].unique()[0]
        stock_df = all_data[all_data['code'] == ticker].copy()
        stock_df['date'] = pd.to_datetime(stock_df['date']) # 确保date列是datetime类型
        stock_df.sort_values(by='date', inplace=True)
        stock_df.reset_index(drop=True, inplace=True)
        print(f"[Main] 已选择股票 {ticker} 进行测试，共 {len(stock_df)} 条记录شه。")

        # 4. 定义窗口参数和回测参数
        window_len = predictor.agent.lookback + predictor.agent.lookahead + 1
        num_test_windows = 30
        
        if len(stock_df) < window_len + num_test_windows - 1:
            raise Exception(f"股票 {ticker} 的数据不足，无法进行 {num_test_windows} 次窗口测试شه。")

        print(f"\n[Main] 将在最新的数据上运行 {num_test_windows} 个连续的滑动窗口进行回测...")

        # 5. 初始化模拟账户和记录器
        initial_cash = 1_000_000
        cash = initial_cash
        shares = 0
        portfolio_values = [] # 记录每日总资产
        all_trade_dates = [] # 记录所有回测区间的日期
        rl_rewards_history = [] # 新增：记录每个窗口的RL奖励

        # --- 新增：初始持仓逻辑 ---
        # 假设在回测开始时，用一部分现金买入股票
        initial_stock_allocation_ratio = 0.1 # 初始分配10%的现金用于购买股票
        
        # 获取回测期第一天的开盘价
        # 注意：这里需要确保 stock_df 至少有足够的数据来获取第一个交易日的价格
        if len(stock_df) == 0:
            raise Exception("股票数据为空，无法设置初始持仓。")
        
        first_trade_day_price = stock_df.iloc[0]['open']
        
        if first_trade_day_price <= 0:
            print("警告：首个交易日开盘价为0或负数，无法设置初始持仓。将从零持股开始。")
        else:
            initial_stock_value = initial_cash * initial_stock_allocation_ratio
            shares_to_buy_at_start = initial_stock_value // first_trade_day_price
            
            if shares_to_buy_at_start > 0:
                shares = shares_to_buy_at_start
                cash -= shares * first_trade_day_price
                print(f"  [Main] 初始设置：用 {initial_stock_value:.2f} 现金买入 {shares} 股 {ticker} at {first_trade_day_price:.2f}。")
                print(f"  [Main] 初始现金: {cash:.2f}, 初始持股: {shares} 股。")
            else:
                print("  [Main] 初始股票分配比例过低或股价过高，无法买入整数股。将从零持股开始。")
        # --- 结束新增 ---


        # 6. 循环执行回测
        for i in range(num_test_windows):
            window_number = i + 1
            
            # 从数据尾部向前切片，模拟在最新数据上进行的回测
            offset = num_test_windows - 1 - i
            start_index = -(window_len + offset)
            end_index = -offset if offset > 0 else None
            
            window_df = stock_df.iloc[start_index:end_index].copy()
            window_df.reset_index(drop=True, inplace=True)

            print(f"\n[Main] === 第 {window_number}/{num_test_windows} 次预测 ({'冷启动' if i == 0 else '热启动'}) ===")
            
            # 获取Agent的交易决策，并传入当前持仓状态
            action, rl_reward = predictor.predict(df=window_df, shares_held=shares)
            rl_rewards_history.append(rl_reward)
            
            # --- 模拟交易与资产记录 ---
            # 交易发生在lookback期之后的第一天
            trade_day_index = predictor.agent.lookback
            trade_price = window_df.loc[trade_day_index, 'open']

            if action == 1: # 买入
                if cash > 0:
                    shares_to_buy = cash // trade_price
                    shares += shares_to_buy
                    cash -= shares_to_buy * trade_price
                    print(f"  [交易] 买入 {shares_to_buy} 股 at {trade_price:.2f}")
            elif action == -1: # 卖出
                if shares > 0:
                    # 新逻辑：全部卖出 (All-Out)
                    cash += shares * trade_price
                    print(f"  [交易] 全仓卖出 {shares} 股 at {trade_price:.2f}")
                    shares = 0
            
            # 在lookahead期间，逐日更新并记录资产
            lookahead_period_df = window_df.iloc[trade_day_index : trade_day_index + predictor.agent.lookahead]
            for _, day in lookahead_period_df.iterrows():
                daily_value = cash + shares * day['close']
                portfolio_values.append(daily_value)
                all_trade_dates.append(day['date'])
            
            print(f"  [资产] 窗口结束时总资产: {portfolio_values[-1]:.2f}")

        print(f"\n🎉 EATA 项目回测完成 شه！")
        
        # 7. 计算并展示专业指标
        print("\n[Main] 正在计算策略表现指标...")
        portfolio_df = pd.DataFrame({'value': portfolio_values}, index=pd.to_datetime(all_trade_dates))

        # 修复: QuantStats不允许重复的索引。删除重复日期，保留最后一次的记录。
        portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep='last')]

        daily_returns = portfolio_df['value'].pct_change().dropna()

        # 计算基准策略（买入并持有） - 更稳健的方法
        # 1. 确保原始数据以日期为索引，以便高效查找
        stock_df_indexed = stock_df.set_index('date')

        # 2. 从原始数据中，提取与我们策略回测期间完全对应的收盘价
        # portfolio_df.index 包含了回测期间的所有日期，是“事实的唯一来源”
        benchmark_prices = stock_df_indexed.loc[portfolio_df.index, 'close']

        # 3. 计算基准收益率
        buy_and_hold_returns = benchmark_prices.pct_change().dropna()

        metrics = TradingMetrics(returns=daily_returns.values, benchmark_returns=buy_and_hold_returns.values)
        metrics.print_metrics("EATA Agent 策略表现")

        # 8. 绘制并保存资产曲线图
        print("\n[Main] 正在绘制资产曲线图...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(16, 8))

        # 绘制Agent策略曲线
        ax.plot(portfolio_df.index, portfolio_df['value'], label='EATA Agent Strategy', color='royalblue', linewidth=2)

        # 绘制买入并持有基准曲线
        benchmark_value = (1 + buy_and_hold_returns).cumprod() * initial_cash
        ax.plot(benchmark_value.index, benchmark_value.values, label='Buy and Hold Benchmark', color='grey', linestyle='--', linewidth=2)
        
        ax.set_title('EATA Agent vs. Buy and Hold Performance', fontsize=18)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图表
        figure_path = 'asset_curve.png'
        plt.savefig(figure_path)
        print(f"\n📈 资产曲线图已成功保存到: {figure_path}")

        # 9. 生成 QuantStats 报告
        print("\n[Main] 正在生成 QuantStats 详细报告...")
        try:
            # 确保收益率序列的索引是 DatetimeIndex
            daily_returns.index = pd.to_datetime(daily_returns.index)
            buy_and_hold_returns.index = pd.to_datetime(buy_and_hold_returns.index)
            print(buy_and_hold_returns)
            qs.reports.html(daily_returns, benchmark=buy_and_hold_returns, output='EATA_Strategy_Report.html', title=f'{ticker} - EATA Agent Performance')
            print(f"\n📊 QuantStats 报告已成功保存到: EATA_Strategy_Report.html")
        except Exception as e:
            print(f"\n⚠️ 生成 QuantStats 报告失败: {e}")

        # 10. 新增：绘制并保存RL奖励趋势图
        print("\n[Main] 正在绘制RL奖励趋势图...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        
        reward_series = pd.Series(rl_rewards_history)
        moving_avg = reward_series.rolling(window=50).mean()

        ax.plot(reward_series.index, reward_series, label='Raw RL Reward', color='lightsteelblue', alpha=0.7)
        ax.plot(moving_avg.index, moving_avg, label='50-Window Moving Average', color='crimson', linewidth=2)
        
        ax.set_title('RL Reward Trend Over Windows', fontsize=18)
        ax.set_xlabel('Window Number', fontsize=12)
        ax.set_ylabel('RL Reward', fontsize=12)
        ax.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图表
        reward_figure_path = 'rl_reward_trend.png'
        plt.savefig(reward_figure_path)
        print(f"\n📉 RL奖励趋势图已成功保存到: {reward_figure_path}")


    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
