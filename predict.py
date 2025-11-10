import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
import logging

# 屏蔽Numpy数值计算警告 (例如除以0，log(0)等)
np.seterr(all='ignore')
# 屏蔽Matplotlib找不到字体的警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# 核心改动：直接导入我们改造后的Agent
from agent import Agent
from data import DataStorage # 导入数据存储类
from performance_metrics import TradingMetrics # 导入我们新增的指标计算模块

# 核心改动：直接导入我们改造后的Agent
from agent import Agent

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
    print("🚀 启动 EATA 项目核心功能测试、回测与评估 (多股票版)")
    print("=======================================================")

    try:
        # 1. 从 stock.db 加载所有数据
        print("\n[Main] 从 stock.db 加载所有数据...")
        data_storage = DataStorage()
        all_data = data_storage.load_raw()
        
        if all_data.empty:
            raise Exception("数据库中没有找到原始数据(raw_data)。请先运行 import_data.py 导入数据。")

        if 'code' not in all_data.columns and 'ts_code' in all_data.columns:
            all_data.rename(columns={'ts_code': 'code'}, inplace=True)

        # 2. 获取所有唯一的股票代码
        all_tickers = all_data['code'].unique()
        # 用户指定跑10只股票，这里可以根据需要调整
        if len(all_tickers) > 10:
            all_tickers = all_tickers[:10] # 只取前10只股票
        print(f"[Main] 发现 {len(all_tickers)} 支股票，将逐一进行回测: {all_tickers}")

        # 3. 初始化一个列表来存储所有股票的最终指标
        all_results = []

        # 4. 外层循环：遍历每一支股票
        for ticker_idx, ticker in enumerate(all_tickers):
            print(f"\n\n{'='*15} 开始回测股票: {ticker} ({ticker_idx + 1}/{len(all_tickers)}) {'='*15}")
            
            # --- 每个股票都使用全新的Agent ---
            # 重新初始化Predictor，它会自动创建新的Agent
            predictor = Predictor()
            
            stock_df = all_data[all_data['code'] == ticker].copy()
            stock_df['date'] = pd.to_datetime(stock_df['date']) # 确保date列是datetime类型
            stock_df.sort_values(by='date', inplace=True)
            stock_df.reset_index(drop=True, inplace=True)
            
            # 确保数据足够长
            window_len = predictor.agent.lookback + predictor.agent.lookahead + 1
            num_test_windows = 1000 # 默认1000个窗口
            
            if len(stock_df) < window_len + num_test_windows - 1:
                print(f"  [WARN] 股票 {ticker} 的数据不足，无法进行 {num_test_windows} 次窗口测试。跳过。")
                continue

            print(f"[Main] 已选择股票 {ticker} 进行测试，共 {len(stock_df)} 条记录。")
            print(f"\n[Main] 将在最新的数据上运行 {num_test_windows} 个连续的滑动窗口进行回测...")

            # 5. 初始化模拟账户和记录器
            initial_cash = 1_000_000
            cash = initial_cash
            shares = 0
            portfolio_values = [] # 记录每日总资产
            all_trade_dates = [] # 记录所有回测区间的日期
            rl_rewards_history = [] # 记录每个窗口的RL奖励

            # --- 初始持仓逻辑已被移除，回测将从100%现金开始 ---

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

            print(f"\n🎉 EATA 项目回测完成 ({ticker})！")
            
            # 7. 计算并展示专业指标
            print("\n[Main] 正在计算策略表现指标...")
            portfolio_df = pd.DataFrame({'value': portfolio_values}, index=pd.to_datetime(all_trade_dates))

            # 修复: QuantStats不允许重复的索引。删除重复日期，保留最后一次的记录。
            portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep='last')]

            # --- 核心修复：为资产曲线和指标计算增加统一的“第0天”起点 ---
            # 1. 找到回测期开始的前一个交易日
            first_trade_date = portfolio_df.index[0]
            first_date_loc = stock_df.index[stock_df['date'] == first_trade_date][0]
            start_day_minus_one_loc = first_date_loc - 1

            if start_day_minus_one_loc >= 0:
                start_date_t0 = stock_df.loc[start_day_minus_one_loc, 'date']
                
                # 2. 创建一个代表“第0天”的DataFrame
                start_row = pd.DataFrame({'value': [initial_cash]}, index=[start_date_t0])
                
                # 3. 将“第0天”拼接到Agent的资产数据前
                portfolio_df = pd.concat([start_row, portfolio_df])
                print(f"  [绘图修复] 已为资产曲线添加共同起点: {start_date_t0.date()}，初始资产: {initial_cash}")
            else:
                print("  [绘图修复] 警告：无法找到回测前一日，资产曲线可能没有T0起点。")
            # --- 结束修复 ---

            daily_returns = portfolio_df['value'].pct_change().dropna()

            # 计算基准策略（买入并持有） - 更稳健的方法
            # 1. 确保原始数据以日期为索引，以便高效查找
            stock_df_indexed = stock_df.set_index('date')

            # 2. 从原始数据中，提取与我们策略回测期间完全对应的收盘价
            benchmark_prices = stock_df_indexed.loc[portfolio_df.index, 'close']

            # 3. 计算基准收益率
            buy_and_hold_returns = benchmark_prices.pct_change().dropna()

            metrics = TradingMetrics(returns=daily_returns.values, benchmark_returns=buy_and_hold_returns.values)
            metrics.print_metrics(f"EATA Agent 策略表现 ({ticker})") # 打印时带上股票代码

            # 8. 绘制并保存资产曲线图
            print("\n[Main] 正在绘制资产曲线图...")
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(16, 8))

            # --- 核心修复：使用统一起点后的数据进行绘图 ---
            # 1. 绘制Agent策略曲线 (现在包含了T0点)
            ax.plot(portfolio_df.index, portfolio_df['value'], label='EATA Agent Strategy', color='royalblue', linewidth=2)

            # 2. 绘制买入并持有基准曲线 (基于同样包含T0的benchmark_prices)
            #    使用更清晰的归一化方法计算，确保起点一致
            benchmark_value = (benchmark_prices / benchmark_prices.iloc[0]) * initial_cash
            ax.plot(benchmark_value.index, benchmark_value.values, label='Buy and Hold Benchmark', color='grey', linestyle='--', linewidth=2)
            # --- 结束修复 ---
            
            ax.set_title(f'EATA Agent vs. Buy and Hold Performance ({ticker})', fontsize=18)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Portfolio Value', fontsize=12)
            ax.legend(fontsize=12)
            plt.tight_layout()
            
            # 保存图表 (文件名包含股票代码)
            figure_path = f'asset_curve_{ticker}.png'
            plt.savefig(figure_path)
            plt.close(fig) # 关闭图表，释放内存
            print(f"\n📈 资产曲线图已成功保存到: {figure_path}")

            # 9. 生成 QuantStats 报告
            print("\n[Main] 正在生成 QuantStats 详细报告...")
            try:
                # 修复QuantStats频率错误：显式将索引转换为日周期
                daily_returns.index = pd.to_datetime(daily_returns.index).to_period('D')
                buy_and_hold_returns.index = pd.to_datetime(buy_and_hold_returns.index).to_period('D')
                
                report_path = f'EATA_Strategy_Report_{ticker}.html' # 文件名包含股票代码
                qs.reports.html(daily_returns, benchmark=buy_and_hold_returns, output=report_path, title=f'{ticker} - EATA Agent Performance')
                print(f"\n📊 QuantStats 报告已成功保存到: {report_path}")
            except Exception as e:
                print(f"\n⚠️ 生成 QuantStats 报告失败 ({ticker}): {e}")

            # 10. 新增：绘制并保存RL奖励趋势图
            print("\n[Main] 正在绘制RL奖励趋势图...")
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(16, 8))
            
            reward_series = pd.Series(rl_rewards_history)
            moving_avg = reward_series.rolling(window=50).mean()

            ax.plot(reward_series.index, reward_series, label='Raw RL Reward', color='lightsteelblue', alpha=0.7)
            ax.plot(moving_avg.index, moving_avg, label='50-Window Moving Average', color='crimson', linewidth=2)
            
            ax.set_title(f'RL Reward Trend Over Windows ({ticker})', fontsize=18)
            ax.set_xlabel('Window Number', fontsize=12)
            ax.set_ylabel('RL Reward', fontsize=12)
            ax.legend(fontsize=12)
            plt.tight_layout()
            
            # 保存图表 (文件名包含股票代码)
            reward_figure_path = f'rl_reward_trend_{ticker}.png'
            plt.savefig(reward_figure_path)
            plt.close(fig) # 关闭图表，释放内存
            print(f"\n📉 RL奖励趋势图已成功保存到: {reward_figure_path}")

            # 收集当前股票的指标，用于最终汇总
            current_metrics = metrics.get_all_metrics()
            current_metrics['Ticker'] = ticker # 添加股票代码
            all_results.append(current_metrics)

        # 11. 打印最终的汇总结果
        print(f"\n\n{'='*25} 所有股票回测汇总 {'='*25}")
        results_df = pd.DataFrame(all_results)
        # 格式化百分比列
        for col in ['Annual Return (AR)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (MDD)', 'Calmar Ratio', 'Win Rate', 'Volatility (Annual)', 'Alpha', 'IRR']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x*100:.2f}%")
        # 格式化其他数值列
        for col in ['Beta', 'Profit Factor']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}")
        
        print(results_df.to_string()) # 使用to_string()防止截断
        print("="*60)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
