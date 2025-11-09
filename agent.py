import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any, List, Tuple
from scipy.stats import entropy

# 导入新的NEMoTS核心模块
from eata_agent.engine import Engine
from eata_agent.args import Args

class Agent:
    def __init__(self, df: pd.DataFrame, lookback: int = 100, lookahead: int = 20):
        self.stock_list = df
        self.lookback = lookback
        self.lookahead = lookahead
        self.hyperparams = self._create_hyperparams()
        self.engine = Engine(self.hyperparams)
        self.previous_best_tree = None
        self.previous_best_expression = None
        self.is_trained = False
        self.training_history = []
        self.__name__ = 'EATA_Agent_v3.1_fixed_strategy'

        print("EATA Agent (固定策略模式) 初始化完成")
        print(f"   - Lookback={self.lookback}, Lookahead={self.lookahead}")
        print("   - 决策规则: 固定 Q25/Q75 共识规则")

    def _create_hyperparams(self) -> Args:
        """创建超参数配置 - 增强版"""
        args = Args()
        args.device = torch.device("cpu")
        args.seed = 42
        args.seq_in = self.lookback
        args.seq_out = self.lookahead
        args.used_dimension = 1
        args.features = 'M'
        args.symbolic_lib = "NEMoTS"
        args.max_len = 35
        args.max_module_init = 10
        args.num_transplant = 5
        args.num_runs = 5
        args.eta = 1.0
        args.num_aug = 3
        args.exploration_rate = 1 / np.sqrt(2)
        args.transplant_step = 800
        args.norm_threshold = 1e-5
        args.epoch = 10
        args.round = 2
        args.train_size = 64
        args.lr = 1e-5
        args.weight_decay = 0.0001
        args.clip = 5.0
        args.buffer_size = 64
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        return args

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """准备单个滑动窗口的数据"""
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(col in df.columns for col in feature_cols):
            raise ValueError(f"输入数据缺少必要列: 需要 {feature_cols}")
        
        data = df[feature_cols].values
        diff = np.diff(data, axis=0)
        last_row = data[:-1]
        last_row[last_row == 0] = 1e-9
        change_rates = diff / last_row
        
        change_rates[:, :4] = np.clip(change_rates[:, :4], -0.1, 0.1)
        change_rates[:, 4:] = np.clip(change_rates[:, 4:], -0.5, 0.5)

        if len(change_rates) < self.lookback + self.lookahead:
            raise ValueError(f"数据长度不足：需要{self.lookback + self.lookahead}，实际可用{len(change_rates)}")
        
        window_data = change_rates[-(self.lookback + self.lookahead):]
        return window_data

    def _predict_distribution(self, top_10_exps: List[str], lookback_data: np.ndarray) -> np.ndarray:
        """为Top-10表达式生成未来预测分布"""
        all_predictions = []
        lookback_data_transposed = lookback_data.T

        eval_vars = {"np": np}
        for i in range(lookback_data_transposed.shape[0]):
            eval_vars[f'x{i}'] = lookback_data_transposed[i, :]

        for exp in top_10_exps:
            try:
                corrected_expression = exp.replace("exp", "np.exp").replace("cos", "np.cos").replace("sin", "np.sin").replace("sqrt", "np.sqrt").replace("log", "np.log")
                historical_fit = eval(corrected_expression, {"__builtins__": None}, eval_vars)

                if not isinstance(historical_fit, np.ndarray) or historical_fit.ndim == 0:
                    historical_fit = np.repeat(historical_fit, self.lookback)
                
                time_axis = np.arange(self.lookback)
                coeffs = np.polyfit(time_axis, historical_fit, 1)
                trend_line = np.poly1d(coeffs)

                future_time_axis = np.arange(self.lookback, self.lookback + self.lookahead)
                future_predictions = trend_line(future_time_axis)
                all_predictions.extend(future_predictions)

            except Exception as e:
                print(f"表达式 '{exp}' 预测失败: {e}。将填充0。")
                all_predictions.extend([0] * self.lookahead)
        
        return np.array(all_predictions)

    def _calculate_rl_reward_and_signal(self, prediction_distribution: np.ndarray, lookahead_ground_truth: np.ndarray, shares_held: int) -> Tuple[float, int]:
        """
        计算RL奖励和交易信号
        - RL奖励: 基于预测分布与真实分布的KL散度(Kullback-Leibler Divergence)。
        - 交易信号: 基于固定的Q25/Q75规则。
        """
        try:
            if prediction_distribution.size == 0:
                return 0.0, 0

            # --- 交易信号决策 (逻辑保持不变) ---
            strategy = [25, 75]
            q_low, q_high = np.percentile(prediction_distribution, strategy)
            intended_signal = 0
            if q_low > 0:
                intended_signal = 1
                print(f"  [决策] 预测分布的 25% 分位数 > 0，生成意图信号: 买入")
            elif q_high < 0:
                intended_signal = -1
                print(f"  [决策] 预测分布的 75% 分位数 < 0，生成意图信号: 卖出")
            else:
                print("  [决策] 预测分布跨越零点，信号不明确，生成意图信号: 持有")

            # --- RL奖励计算 (新逻辑: KL散度) ---
            # 1. 提取真实的日收益率
            actual_returns = lookahead_ground_truth.T[3, :] 

            # 2. 为两个分布创建共同的区间(bins)
            combined_data = np.concatenate((prediction_distribution, actual_returns))
            min_val, max_val = np.min(combined_data), np.max(combined_data)
            num_bins = 50  # 定义分箱数量
            bins = np.linspace(min_val, max_val, num_bins)

            # 3. 计算两个分布在共同区间上的直方图
            pred_hist, _ = np.histogram(prediction_distribution, bins=bins, density=True)
            actual_hist, _ = np.histogram(actual_returns, bins=bins, density=True)

            # 4. 将频率转换为概率，并添加平滑项防止log(0)
            epsilon = 1e-10
            pred_probs = pred_hist / np.sum(pred_hist) + epsilon
            actual_probs = actual_hist / np.sum(actual_hist) + epsilon

            # 5. 计算KL散度
            # scipy.stats.entropy(pk, qk) 计算 pk 相对于 qk 的KL散度
            kl_divergence = entropy(pred_probs, actual_probs)

            # 6. 将KL散度转换为奖励
            rl_reward = 1 / (1 + kl_divergence)
            
            return rl_reward, intended_signal
        except Exception as e:
            print(f"--- 🚨 在 _calculate_rl_reward_and_signal 中捕获到致命错误 🚨 ---")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            print(f"--- 诊断结束 ---")
            return 0.0, 0

    def criteria(self, d: pd.DataFrame, shares_held: int) -> int:
        """核心决策函数，集成策略学习流程"""
        try:
            if self.previous_best_tree is not None:
                print("检测到已有语法树，切换到热启动参数 (num_runs=1)...")
                self.engine.model.num_runs = 1 # 核心优化：热启动时，只运行1次MCTS
                self.engine.model.num_transplant = 5
                self.engine.model.transplant_step = 300
                self.engine.model.num_aug = 3
            else:
                print("首次运行，使用重量级参数...")
                # 使用更强的冷启动参数
                self.engine.model.num_runs = 5
                self.engine.model.max_len = 35

            full_window_data = self._prepare_data(d)
            lookback_data = full_window_data[:self.lookback, :]
            lookahead_data = full_window_data[-self.lookahead:, :]

            # engine.simulate 现在返回 mcts_records
            best_exp, top_10_exps, top_10_scores, _, mae, mse, corr, _, mcts_score, new_best_tree, mcts_records = self.engine.simulate(
                full_window_data, previous_best_tree=self.previous_best_tree
            )

            self.previous_best_expression = str(best_exp)
            self.previous_best_tree = new_best_tree
            self.is_trained = True
            
            record = {'mae': mae, 'corr': corr, 'mcts_score': mcts_score}
            self.training_history.append(record)
            print(f"NEMoTS运行完成: MAE={mae:.4f}, Corr={corr:.4f}, MCTS Score={mcts_score:.4f}")

            prediction_distribution = self._predict_distribution(top_10_exps, lookback_data)
            print(f"生成了 {len(prediction_distribution)} 个预测点。")

            rl_reward, trading_signal = self._calculate_rl_reward_and_signal(
                prediction_distribution, lookahead_data, shares_held
            )
            print(f"RL奖励 (基于真实信号): {rl_reward:.4f}, 意图交易信号: {trading_signal}")

            # “盖戳”流程：将最终的rl_reward附加到本次窗口产生的所有经验上
            stamped_experiences = []
            for experience in mcts_records:
                # experience 是一个元组 (state, seq, policy, value)
                stamped_experience = experience + (rl_reward,)
                stamped_experiences.append(stamped_experience)
            
            # 将“盖戳”后的经验数据存入引擎，并由引擎决定是否触发训练
            if stamped_experiences:
                self.engine.store_experiences(stamped_experiences)

            return trading_signal, rl_reward

        except Exception as e:
            print(f"NEMoTS Agent 'criteria' 失败: {e}")
            import traceback
            traceback.print_exc()
            return 0

    # choose_action, vote, strength 方法保持不变
    @classmethod
    def choose_action(cls, s: tuple) -> int:
        try:
            _, s1, _, _ = s
            temp_agent = Agent(pd.DataFrame())
            # 注意：这里的静态调用无法知道持仓状态，这是一个简化处理。
            # 在真实的多股票场景中，需要为每个股票维护一个Agent实例。
            return temp_agent.criteria(s1, shares_held=0) # 假设默认是空仓
        except Exception as e:
            print(f"动作选择失败: {e}")
            return 0

    def vote(self) -> int:
        print("'vote' 方法被简化，仅返回中性信号。请在 predict.py 中实现多股票循环。")
        return 50

    def strength(self, w1: float, w2: float, w3: float, w4: float) -> pd.Series:
        print("'strength' 方法被简化，返回固定值。")
        self.stock_list['strength'] = [50] * len(self.stock_list)
        return self.stock_list['strength']
