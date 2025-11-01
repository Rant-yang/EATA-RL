import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


class Tracker:
    """
    增强版跟踪器，支持训练过程指标跟踪和最终预测结果可视化
    同时支持单目标和双目标(Q25/Q75)训练
    """
    def __init__(self, save_dir="logs", dual_target=True):
        self.save_dir = save_dir
        self.eval_dir = os.path.join(save_dir, "evaluation")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.dual_target = dual_target  # 是否为双目标训练
        self.reset()

    def reset(self):
        # 基本训练指标
        self.history = {
            "train_step": [],
            "alpha": [],
            "policy_entropy": [],
            "policy_maxprob": [],
            "value": [],  # 双目标时表示平均MAE
            "reward": [], # 双目标时表示-MAE
            "corr": [],   # 双目标时表示Q25/Q75关系正确率
            "best_score": []
        }
        
        # 双目标特有指标
        if self.dual_target:
            self.dual_history = {
                "train_step": [],
                "mae_q25": [],
                "mae_q75": [],
                "q_violations": [],  # Q75<Q25的比例
                "q_diff": []         # Q75-Q25差值
            }

    def update(self, step, alpha, policy=None, value=None, reward=None, corr=None, best_score=None):
        """更新基本训练指标"""
        self.history["train_step"].append(step)
        self.history["alpha"].append(alpha)
        if policy is not None:
            policy = np.asarray(policy)
            entropy = -np.sum(policy * np.log(policy + 1e-8))
            maxprob = np.max(policy)
        else:
            entropy = None
            maxprob = None
        self.history["policy_entropy"].append(entropy)
        self.history["policy_maxprob"].append(maxprob)
        self.history["value"].append(value)
        self.history["reward"].append(reward)
        self.history["corr"].append(corr)
        self.history["best_score"].append(best_score)
        
    def update_dual_metrics(self, step, mae_q25, mae_q75, pred_q25=None, pred_q75=None):
        """更新双目标特有指标"""
        if not self.dual_target:
            print("\u8b66告: 在非双目标模式下调用update_dual_metrics")
            return
            
        self.dual_history["train_step"].append(step)
        self.dual_history["mae_q25"].append(mae_q25)
        self.dual_history["mae_q75"].append(mae_q75)
        
        # 如果提供了预测值，计算验证指标
        if pred_q25 is not None and pred_q75 is not None:
            q_violations = (pred_q75 < pred_q25).sum() / len(pred_q25)
            q_diff = np.mean(pred_q75 - pred_q25)
            
            self.dual_history["q_violations"].append(float(q_violations))
            self.dual_history["q_diff"].append(float(q_diff))
        else:
            self.dual_history["q_violations"].append(None)
            self.dual_history["q_diff"].append(None)

    def plot(self, keys=None, save_prefix="training_log"):
        """绘制基本训练指标图表"""
        # 输出调试信息
        print(f"\n[调试] plot内部 - 当前保存目录: {self.save_dir}")
        print(f"[调试] 实际目录是否存在: {os.path.exists(self.save_dir)}")
        print(f"[调试] 当前数据点数量: {len(self.history['train_step'])}")
        
        # 如果没有数据点则创建空白图表
        if len(self.history['train_step']) == 0:
            print("[警告] 当前还没有数据点，创建空白图表")
            # 创建一个空的图表标记当前状态
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "训练初始化中...", ha="center", va="center", fontsize=14)
            plt.title(f"No Data Yet - {save_prefix}")
            plt.tight_layout()
            # 确保目录存在
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, f"{save_prefix}_initializing.png"))
            plt.close()
            return

        if keys is None:
            keys = [k for k in self.history if k != "train_step"]
        # 单独曲线
        for key in keys:
            # 跳过含空值的数据
            if all(x is None for x in self.history[key]):
                print(f"[警告] {key} 全部为空值，跳过绘图")
                continue
                
            plt.figure()
            # 筛选非空值
            steps = []
            values = []
            for i, val in enumerate(self.history[key]):
                if val is not None:
                    steps.append(self.history["train_step"][i])
                    values.append(val)
            
            if len(steps) > 0:  # 确保有数据可绘制
                plt.plot(steps, values, marker="o")
                plt.xlabel("Train Step")
                plt.ylabel(key)
                plt.title(key)
                plt.tight_layout()
                # 确保目录存在
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{save_prefix}_{key}.png")
                plt.savefig(save_path)
                print(f"[调试] 保存图表到: {save_path}")
            plt.close()
        # reward vs best_score 双曲线
        if "reward" in self.history and "best_score" in self.history:
            # 筛选两个指标都有效的数据点
            steps = []
            rewards = []
            scores = []
            for i in range(len(self.history["train_step"])):
                if self.history["reward"][i] is not None and self.history["best_score"][i] is not None:
                    steps.append(self.history["train_step"][i])
                    rewards.append(self.history["reward"][i])
                    scores.append(self.history["best_score"][i])
            
            if len(steps) > 0:  # 确保有数据可绘制
                plt.figure()
                plt.plot(steps, rewards, label="Train Best Reward", marker="o")
                plt.plot(steps, scores, label="Test Score", marker="x")
                plt.xlabel("Train Step")
                plt.ylabel("Score")
                plt.title("Train Best Reward vs Test Score")
                plt.legend()
                plt.tight_layout()
                # 确保目录存在
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{save_prefix}_reward_vs_testscore.png")
                plt.savefig(save_path)
                print(f"[调试] 保存对比图表到: {save_path}")
                plt.close()
            else:
                print("[警告] reward 和 best_score 全部为空值，跳过绘图")
            
        # 如果是双目标模式，绘制双目标特有指标
        if self.dual_target:
            if len(self.dual_history["train_step"]) > 0:
                print(f"[调试] 双目标数据点数量: {len(self.dual_history['train_step'])}")
                self.plot_dual_metrics(save_prefix)
            else:
                print("[警告] 双目标数据为空，跳过双目标指标图表绘制")
    
    def plot_dual_metrics(self, save_prefix="dual_target"):
        """绘制双目标特有指标图表"""
        if not self.dual_target:
            return
        
        print(f"[调试] plot_dual_metrics - 数据点数量: {len(self.dual_history['train_step'])}")
        
        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[调试] 确保目录存在: {self.save_dir}")
            
        # 筛选有效数据点
        steps = []
        q25_values = []
        q75_values = []
        for i in range(len(self.dual_history["train_step"])):
            if (self.dual_history["mae_q25"][i] is not None and 
                self.dual_history["mae_q75"][i] is not None):
                steps.append(self.dual_history["train_step"][i])
                q25_values.append(self.dual_history["mae_q25"][i])
                q75_values.append(self.dual_history["mae_q75"][i])
                
        # 绘制MAE对比图
        if len(steps) > 0:  # 确保有数据可绘制
            plt.figure()
            plt.plot(steps, q25_values, label="Q25 MAE", marker="o")
            plt.plot(steps, q75_values, label="Q75 MAE", marker="x")
            plt.xlabel("Train Step")
            plt.ylabel("MAE")
            plt.title("Q25 vs Q75 MAE")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{save_prefix}_mae_comparison.png")
            plt.savefig(save_path)
            print(f"[调试] 保存MAE对比图到: {save_path}")
            plt.close()
        else:
            print("[警告] 没有有效的MAE数据点，跳过绘制")
            # 创建一个空白图表
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "等待MAE数据...", ha="center", va="center", fontsize=14)
            plt.title(f"MAE Data Not Available Yet - {save_prefix}")
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{save_prefix}_mae_comparison_waiting.png")
            plt.savefig(save_path)
            plt.close()
        
        # 绘制Q25/Q75关系指标
        if any(v is not None for v in self.dual_history["q_violations"]):
            print(f"[调试] 绘制Q25/Q75关系指标 - 有效数据点: {sum(1 for v in self.dual_history['q_violations'] if v is not None)}")
            
            plt.figure()
            # 筛选非空值
            steps = []
            violations = []
            for i, v in enumerate(self.dual_history["q_violations"]):
                if v is not None:
                    steps.append(self.dual_history["train_step"][i])
                    violations.append(v * 100)  # 转为百分比
            
            if len(steps) > 0:  # 确保有数据可绘制
                plt.plot(steps, violations, label="Q75<Q25 比例", marker="o", color='r')
                plt.axhline(y=0.0, color='g', linestyle='--', label="目标值(0%)")
                plt.xlabel("Train Step")
                plt.ylabel("违规率 (%)")
                plt.title("Q25/Q75 关系验证")
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(self.save_dir, f"{save_prefix}_q_validation.png")
                plt.savefig(save_path)
                print(f"[调试] 保存Q25/Q75关系图表到: {save_path}")
                plt.close()
            else:
                print("[警告] 没有有效的Q25/Q75关系数据点，跳过绘制")
        else:
            print("[警告] 没有Q25/Q75关系数据，创建等待图表")
            # 创建一个空白图表
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "等待Q25/Q75关系数据...", ha="center", va="center", fontsize=14)
            plt.title(f"Q25/Q75 Relation Not Available Yet - {save_prefix}")
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{save_prefix}_q_validation_waiting.png")
            plt.savefig(save_path)
            plt.close()
            
        # 绘制Q75-Q25差值变化
        if any(d is not None for d in self.dual_history["q_diff"]):
            print(f"[调试] 绘制Q75-Q25差值图表 - 有效数据点: {sum(1 for d in self.dual_history['q_diff'] if d is not None)}")
            
            plt.figure()
            # 筛选非空值
            steps = []
            diffs = []
            for i, d in enumerate(self.dual_history["q_diff"]):
                if d is not None:
                    steps.append(self.dual_history["train_step"][i])
                    diffs.append(d)
            
            if len(steps) > 0:  # 确保有数据可绘制
                plt.plot(steps, diffs, label="Q75-Q25 差值", marker="o", color='b')
                plt.axhline(y=0.0, color='r', linestyle='--', label="最小可接受值(0)")
                plt.xlabel("Train Step")
                plt.ylabel("差值")
                plt.title("Q75-Q25 差值趋势")
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(self.save_dir, f"{save_prefix}_q_diff.png")
                plt.savefig(save_path)
                print(f"[调试] 保存Q75-Q25差值图表到: {save_path}")
                plt.close()
            else:
                print("[警告] 没有有效的Q75-Q25差值数据点，跳过绘制")
        else:
            print("[警告] 没有Q75-Q25差值数据，创建等待图表")
            # 创建一个空白图表
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "等待Q75-Q25差值数据...", ha="center", va="center", fontsize=14)
            plt.title(f"Q75-Q25 Difference Not Available Yet - {save_prefix}")
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{save_prefix}_q_diff_waiting.png")
            plt.savefig(save_path)
            plt.close()

    def save_npz(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, "tracker_history.npz")
        np.savez(filename, **self.history)

    def load_npz(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, "tracker_history.npz")
        data = np.load(filename)
        for key in data:
            self.history[key] = list(data[key])
            
    # 以下是整合自DualTargetVisualizer的最终预测结果可视化方法
    
    def plot_predictions_vs_actual(self, dates, pred_q25, pred_q75, actual, title="价格预测与实际对比", 
                                   filename="prediction_vs_actual.png"):
        """
        绘制预测区间与实际价格的对比图
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制预测区间
        plt.fill_between(range(len(dates)), pred_q25, pred_q75, color='lightblue', alpha=0.5, label='预测价格区间(Q25-Q75)')
        
        # 绘制实际价格曲线
        plt.plot(range(len(dates)), actual, 'r-', label='实际价格', linewidth=2)
        
        # 绘制Q25和Q75预测曲线
        plt.plot(range(len(dates)), pred_q25, 'b--', label='Q25预测', linewidth=1)
        plt.plot(range(len(dates)), pred_q75, 'g--', label='Q75预测', linewidth=1)
        
        # 标记实际价格超出预测区间的点
        below_q25 = actual < pred_q25
        above_q75 = actual > pred_q75
        
        plt.scatter(np.where(below_q25)[0], actual[below_q25], color='purple', marker='v', label='低于Q25(卖出风险)', s=50)
        plt.scatter(np.where(above_q75)[0], actual[above_q75], color='orange', marker='^', label='高于Q75(买入风险)', s=50)
        
        # 添加图例和标题
        plt.title(title)
        plt.xlabel('时间序列')
        plt.ylabel('价格')
        plt.legend()
        
        # 设置x轴刻度和标签
        if len(dates) <= 20:
            plt.xticks(range(len(dates)), dates, rotation=45)
        else:
            # 选择合适的间隔显示日期
            step = max(1, len(dates) // 10)
            indices = range(0, len(dates), step)
            plt.xticks(indices, [dates[i] for i in indices], rotation=45)
            
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.eval_dir, filename), dpi=300)
        plt.close()
        
    def plot_prediction_error(self, dates, pred_q25, pred_q75, actual, title="预测误差分析", 
                             filename="prediction_error.png"):
        """
        绘制预测误差的分析图，包括MAE和区间违规率
        """
        error_q25 = np.abs(pred_q25 - actual)
        error_q75 = np.abs(pred_q75 - actual)
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 预测误差(MAE)随时间变化
        plt.subplot(2, 1, 1)
        plt.plot(range(len(dates)), error_q25, 'b-', label='Q25绝对误差', linewidth=1.5)
        plt.plot(range(len(dates)), error_q75, 'g-', label='Q75绝对误差', linewidth=1.5)
        plt.axhline(y=np.mean(error_q25), color='b', linestyle='--', alpha=0.7, label=f'Q25平均误差: {np.mean(error_q25):.4f}')
        plt.axhline(y=np.mean(error_q75), color='g', linestyle='--', alpha=0.7, label=f'Q75平均误差: {np.mean(error_q75):.4f}')
        
        plt.title("预测绝对误差(MAE)随时间变化")
        plt.ylabel('绝对误差')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 子图2: 预测区间失效率
        plt.subplot(2, 1, 2)
        window_size = min(10, len(dates)) # 移动窗口大小
        
        below_q25 = actual < pred_q25
        above_q75 = actual > pred_q75
        
        # 计算移动窗口内的失效率
        window_failure_rate = []
        x_positions = []
        
        for i in range(len(dates) - window_size + 1):
            window_below = below_q25[i:i+window_size].sum()
            window_above = above_q75[i:i+window_size].sum()
            failure_rate = (window_below + window_above) / window_size * 100
            window_failure_rate.append(failure_rate)
            x_positions.append(i + window_size // 2)  # 窗口中心位置
            
        plt.plot(x_positions, window_failure_rate, 'r-', label=f'区间预测失效率(窗口={window_size})', linewidth=2)
        plt.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='50%基准线')
        
        # 理想的失效率应为50%（对于完美的分位数预测）
        overall_failure_rate = (below_q25.sum() + above_q75.sum()) / len(dates) * 100
        plt.axhline(y=overall_failure_rate, color='r', linestyle='--', alpha=0.7, 
                   label=f'总体失效率: {overall_failure_rate:.2f}%')
        
        plt.title("预测区间失效率分析")
        plt.xlabel('时间序列')
        plt.ylabel('区间失效率 (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, filename), dpi=300)
        plt.close()

    def plot_trading_risk_heatmap(self, dates, pred_q25, pred_q75, actual, title="交易风险热图", 
                               filename="trading_risk_heatmap.png"):
        """
        绘制交易风险热图，展示不同时间点的买入/卖出风险
        """
        # 计算风险指标
        below_q25 = actual < pred_q25  # 卖出风险
        above_q75 = actual > pred_q75  # 买入风险
        
        # 计算风险程度（相对误差）
        sell_risk = np.zeros_like(pred_q25)
        buy_risk = np.zeros_like(pred_q25)
        
        # 卖出风险：实际价格低于Q25多少比例
        sell_mask = below_q25
        if np.any(sell_mask):
            sell_risk[sell_mask] = (pred_q25[sell_mask] - actual[sell_mask]) / pred_q25[sell_mask] * 100
            
        # 买入风险：实际价格高于Q75多少比例
        buy_mask = above_q75
        if np.any(buy_mask):
            buy_risk[buy_mask] = (actual[buy_mask] - pred_q75[buy_mask]) / pred_q75[buy_mask] * 100
            
        plt.figure(figsize=(14, 8))
        
        # 创建数据框用于热图
        risk_data = np.zeros((2, len(dates)))
        risk_data[0, :] = sell_risk  # 卖出风险
        risk_data[1, :] = buy_risk   # 买入风险
        
        # 创建热图
        plt.imshow(risk_data, aspect='auto', cmap='YlOrRd')
        
        # 添加颜色条
        cbar = plt.colorbar(label='风险程度（%）')
        
        # 设置Y轴标签
        plt.yticks([0, 1], ['卖出风险\n(实际<Q25)', '买入风险\n(实际>Q75)'])
        
        # 设置X轴标签
        if len(dates) <= 20:
            plt.xticks(range(len(dates)), dates, rotation=45)
        else:
            step = max(1, len(dates) // 10)
            indices = range(0, len(dates), step)
            plt.xticks(indices, [dates[i] for i in indices], rotation=45)
            
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, filename), dpi=300)
        plt.close()
        
        # 打印风险统计信息
        print(f"\n==== 交易风险统计 ====")
        print(f"卖出风险（实际<Q25）比例: {np.sum(below_q25)/len(dates)*100:.2f}%")
        print(f"买入风险（实际>Q75）比例: {np.sum(above_q75)/len(dates)*100:.2f}%")
        if np.any(sell_mask):
            print(f"卖出风险均值: {np.mean(sell_risk[sell_risk > 0]):.2f}%")
        if np.any(buy_mask):
            print(f"买入风险均值: {np.mean(buy_risk[buy_risk > 0]):.2f}%")

    def generate_final_report(self, pred_data_path=None, pred_q25=None, pred_q75=None, actual=None, dates=None):
        """
        生成最终预测结果评估报告和可视化
        
        Args:
            pred_data_path: 预测数据文件路径（CSV或JSON）
            pred_q25, pred_q75, actual, dates: 预测和实际值数据
        """
        if not self.dual_target:
            print("该方法仅支持双目标模式")
            return
            
        # 从文件加载数据（如果提供了文件路径）
        if pred_data_path is not None:
            if pred_data_path.endswith('.csv'):
                df = pd.read_csv(pred_data_path)
                dates = df['date'].values if 'date' in df.columns else [f"T{i}" for i in range(len(df))]
                pred_q25 = df['pred_q25'].values
                pred_q75 = df['pred_q75'].values
                actual = df['actual'].values if 'actual' in df.columns else df.get('close', df.get('price', None))
            elif pred_data_path.endswith('.json'):
                with open(pred_data_path, 'r') as f:
                    data = json.load(f)
                dates = data.get('dates', [f"T{i}" for i in range(len(data['pred_q25']))])
                pred_q25 = np.array(data['pred_q25'])
                pred_q75 = np.array(data['pred_q75'])
                actual = np.array(data.get('actual', data.get('target', [])))
        
        # 检查数据是否存在
        if pred_q25 is None or pred_q75 is None or actual is None:
            print("预测数据不完整，无法生成报告")
            return
            
        # 确保数据为numpy数组
        pred_q25 = np.array(pred_q25)
        pred_q75 = np.array(pred_q75)
        actual = np.array(actual)
        
        # 如果没有日期，创建默认日期
        if dates is None:
            dates = [f"T{i}" for i in range(len(pred_q25))]
        
        # 生成所有可视化图表
        self.plot_predictions_vs_actual(dates, pred_q25, pred_q75, actual)
        self.plot_prediction_error(dates, pred_q25, pred_q75, actual)
        self.plot_trading_risk_heatmap(dates, pred_q25, pred_q75, actual)
        
        # 计算评估指标
        metrics = self.calculate_evaluation_metrics(pred_q25, pred_q75, actual)
        
        # 保存指标到JSON文件
        with open(os.path.join(self.eval_dir, "evaluation_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # 返回指标便于后续使用
        return metrics
        
    def calculate_evaluation_metrics(self, pred_q25, pred_q75, actual):
        """计算评估指标"""
        # 基本指标
        mae_q25 = np.mean(np.abs(pred_q25 - actual))
        mae_q75 = np.mean(np.abs(pred_q75 - actual))
        mse_q25 = np.mean(np.square(pred_q25 - actual))
        mse_q75 = np.mean(np.square(pred_q75 - actual))
        
        # 区间相关指标
        below_q25 = (actual < pred_q25).sum() / len(actual)
        above_q75 = (actual > pred_q75).sum() / len(actual)
        interval_coverage = 1 - below_q25 - above_q75
        
        # Q25和Q75的相对关系
        q_diff_mean = np.mean(pred_q75 - pred_q25)
        q_violations = (pred_q75 < pred_q25).sum() / len(pred_q25)
        
        # 交易风险指标
        sell_risk = np.zeros_like(pred_q25)
        sell_mask = actual < pred_q25
        if np.any(sell_mask):
            sell_risk[sell_mask] = (pred_q25[sell_mask] - actual[sell_mask]) / pred_q25[sell_mask] * 100
            
        buy_risk = np.zeros_like(pred_q25)
        buy_mask = actual > pred_q75
        if np.any(buy_mask):
            buy_risk[buy_mask] = (actual[buy_mask] - pred_q75[buy_mask]) / pred_q75[buy_mask] * 100
            
        avg_sell_risk = np.mean(sell_risk[sell_risk > 0]) if np.any(sell_risk > 0) else 0
        avg_buy_risk = np.mean(buy_risk[buy_risk > 0]) if np.any(buy_risk > 0) else 0
        
        # 收集所有指标
        metrics = {
            "基本指标": {
                "mae_q25": float(mae_q25),
                "mae_q75": float(mae_q75),
                "avg_mae": float((mae_q25 + mae_q75) / 2),
                "mse_q25": float(mse_q25),
                "mse_q75": float(mse_q75),
                "avg_mse": float((mse_q25 + mse_q75) / 2)
            },
            "区间指标": {
                "below_q25_rate": float(below_q25),
                "above_q75_rate": float(above_q75),
                "interval_coverage": float(interval_coverage),
                "ideal_coverage": 0.5
            },
            "分位数关系": {
                "q_diff_mean": float(q_diff_mean),
                "q_violations_rate": float(q_violations)
            },
            "交易风险": {
                "below_q25_count": int(np.sum(sell_mask)),
                "above_q75_count": int(np.sum(buy_mask)),
                "avg_sell_risk": float(avg_sell_risk),
                "avg_buy_risk": float(avg_buy_risk)
            }
        }
        
        return metrics
        
    def plot_predictions_vs_actual(self, dates, pred_q25, pred_q75, actual, title="价格预测与实际对比", 
                                   filename="prediction_vs_actual.png"):
        """
        绘制预测区间与实际价格的对比图
        
        Args:
            dates: 日期序列
            pred_q25: Q25预测值
            pred_q75: Q75预测值
            actual: 实际价格
            title: 图表标题
            filename: 保存的文件名
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制预测区间
        plt.fill_between(range(len(dates)), pred_q25, pred_q75, color='lightblue', alpha=0.5, label='预测价格区间(Q25-Q75)')
        
        # 绘制实际价格曲线
        plt.plot(range(len(dates)), actual, 'r-', label='实际价格', linewidth=2)
        
        # 绘制Q25和Q75预测曲线
        plt.plot(range(len(dates)), pred_q25, 'b--', label='Q25预测', linewidth=1)
        plt.plot(range(len(dates)), pred_q75, 'g--', label='Q75预测', linewidth=1)
        
        # 标记实际价格超出预测区间的点
        below_q25 = actual < pred_q25
        above_q75 = actual > pred_q75
        
        plt.scatter(np.where(below_q25)[0], actual[below_q25], color='purple', marker='v', label='低于Q25(卖出风险)', s=50)
        plt.scatter(np.where(above_q75)[0], actual[above_q75], color='orange', marker='^', label='高于Q75(买入风险)', s=50)
        
        # 添加图例和标题
        plt.title(title)
        plt.xlabel('时间序列')
        plt.ylabel('价格')
        plt.legend()
        
        # 设置x轴刻度和标签
        if len(dates) <= 20:
            plt.xticks(range(len(dates)), dates, rotation=45)
        else:
            # 选择合适的间隔显示日期
            step = max(1, len(dates) // 10)
            indices = range(0, len(dates), step)
            plt.xticks(indices, [dates[i] for i in indices], rotation=45)
            
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
    def plot_prediction_error(self, dates, pred_q25, pred_q75, actual, title="预测误差分析", 
                             filename="prediction_error.png"):
        """
        绘制预测误差的分析图，包括MAE和区间违规率
        
        Args:
            dates: 日期序列
            pred_q25: Q25预测值
            pred_q75: Q75预测值
            actual: 实际价格
            title: 图表标题
            filename: 保存的文件名
        """
        error_q25 = np.abs(pred_q25 - actual)
        error_q75 = np.abs(pred_q75 - actual)
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 预测误差(MAE)随时间变化
        plt.subplot(2, 1, 1)
        plt.plot(range(len(dates)), error_q25, 'b-', label='Q25绝对误差', linewidth=1.5)
        plt.plot(range(len(dates)), error_q75, 'g-', label='Q75绝对误差', linewidth=1.5)
        plt.axhline(y=np.mean(error_q25), color='b', linestyle='--', alpha=0.7, label=f'Q25平均误差: {np.mean(error_q25):.4f}')
        plt.axhline(y=np.mean(error_q75), color='g', linestyle='--', alpha=0.7, label=f'Q75平均误差: {np.mean(error_q75):.4f}')
        
        plt.title("预测绝对误差(MAE)随时间变化")
        plt.ylabel('绝对误差')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 子图2: 预测区间失效率
        plt.subplot(2, 1, 2)
        window_size = min(10, len(dates)) # 移动窗口大小
        
        below_q25 = actual < pred_q25
        above_q75 = actual > pred_q75
        
        # 计算移动窗口内的失效率
        window_failure_rate = []
        x_positions = []
        
        for i in range(len(dates) - window_size + 1):
            window_below = below_q25[i:i+window_size].sum()
            window_above = above_q75[i:i+window_size].sum()
            failure_rate = (window_below + window_above) / window_size * 100
            window_failure_rate.append(failure_rate)
            x_positions.append(i + window_size // 2)  # 窗口中心位置
            
        plt.plot(x_positions, window_failure_rate, 'r-', label=f'区间预测失效率(窗口={window_size})', linewidth=2)
        plt.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='50%基准线')
        
        # 理想的失效率应为50%（对于完美的分位数预测）
        overall_failure_rate = (below_q25.sum() + above_q75.sum()) / len(dates) * 100
        plt.axhline(y=overall_failure_rate, color='r', linestyle='--', alpha=0.7, 
                   label=f'总体失效率: {overall_failure_rate:.2f}%')
        
        plt.title("预测区间失效率分析")
        plt.xlabel('时间序列')
        plt.ylabel('区间失效率 (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴刻度和标签
        if len(x_positions) <= 20:
            plt.xticks(x_positions, [dates[i] for i in x_positions], rotation=45)
        else:
            step = max(1, len(x_positions) // 10)
            indices = range(0, len(x_positions), step)
            plt.xticks([x_positions[i] for i in indices if i < len(x_positions)], 
                      [dates[x_positions[i]] for i in indices if i < len(x_positions)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_trading_risk_heatmap(self, dates, pred_q25, pred_q75, actual, title="交易风险热图", 
                               filename="trading_risk_heatmap.png"):
        """
        绘制交易风险热图，展示不同时间点的买入/卖出风险
        
        Args:
            dates: 日期序列
            pred_q25: Q25预测值
            pred_q75: Q75预测值
            actual: 实际价格
            title: 图表标题
            filename: 保存的文件名
        """
        # 计算风险指标
        below_q25 = actual < pred_q25  # 卖出风险
        above_q75 = actual > pred_q75  # 买入风险
        
        # 计算风险程度（相对误差）
        sell_risk = np.zeros_like(pred_q25)
        buy_risk = np.zeros_like(pred_q25)
        
        # 卖出风险：实际价格低于Q25多少比例
        sell_mask = below_q25
        if np.any(sell_mask):
            sell_risk[sell_mask] = (pred_q25[sell_mask] - actual[sell_mask]) / pred_q25[sell_mask] * 100
            
        # 买入风险：实际价格高于Q75多少比例
        buy_mask = above_q75
        if np.any(buy_mask):
            buy_risk[buy_mask] = (actual[buy_mask] - pred_q75[buy_mask]) / pred_q75[buy_mask] * 100
            
        plt.figure(figsize=(14, 8))
        
        # 创建数据框用于热图
        risk_data = np.zeros((2, len(dates)))
        risk_data[0, :] = sell_risk  # 卖出风险
        risk_data[1, :] = buy_risk   # 买入风险
        
        # 创建热图
        plt.imshow(risk_data, aspect='auto', cmap='YlOrRd')
        
        # 添加颜色条
        cbar = plt.colorbar(label='风险程度（%）')
        
        # 设置Y轴标签
        plt.yticks([0, 1], ['卖出风险\n(实际<Q25)', '买入风险\n(实际>Q75)'])
        
        # 设置X轴标签
        if len(dates) <= 20:
            plt.xticks(range(len(dates)), dates, rotation=45)
        else:
            step = max(1, len(dates) // 10)
            indices = range(0, len(dates), step)
            plt.xticks(indices, [dates[i] for i in indices], rotation=45)
            
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        
        # 打印风险统计信息
        print(f"\n==== 交易风险统计 ====")
        print(f"卖出风险（实际<Q25）比例: {np.sum(below_q25)/len(dates)*100:.2f}%")
        print(f"卖出风险均值: {np.mean(sell_risk[sell_risk > 0]):.2f}%")
        print(f"买入风险（实际>Q75）比例: {np.sum(above_q75)/len(dates)*100:.2f}%")
        print(f"买入风险均值: {np.mean(buy_risk[buy_risk > 0]):.2f}%")

    def generate_all_visualizations(self, pred_data_path=None, pred_q25=None, pred_q75=None, actual=None, dates=None):
        """
        一键生成所有可视化图表
        
        Args:
            pred_data_path: 预测数据文件路径（CSV或JSON），包含pred_q25, pred_q75, actual, dates列
            pred_q25, pred_q75, actual, dates: 如果直接提供数据，则使用这些数据生成图表
        """
        # 如果提供了文件路径，从文件加载数据
        if pred_data_path is not None:
            if pred_data_path.endswith('.csv'):
                df = pd.read_csv(pred_data_path)
                dates = df['date'].values if 'date' in df.columns else np.arange(len(df))
                pred_q25 = df['pred_q25'].values
                pred_q75 = df['pred_q75'].values
                actual = df['actual'].values
            elif pred_data_path.endswith('.json'):
                import json
                with open(pred_data_path, 'r') as f:
                    data = json.load(f)
                    
                # 假设JSON包含所需的所有键
                dates = data.get('dates', np.arange(len(data['pred_q25'])))
                pred_q25 = np.array(data['pred_q25'])
                pred_q75 = np.array(data['pred_q75'])
                actual = np.array(data['actual'])
            else:
                raise ValueError(f"不支持的文件格式: {pred_data_path}")
        
        # 检查数据是否已提供
        if pred_q25 is None or pred_q75 is None or actual is None:
            raise ValueError("未提供预测数据。请提供pred_data_path或直接提供pred_q25, pred_q75, actual数据")
            
        # 如果未提供日期，创建默认日期
        if dates is None:
            dates = [f"T{i}" for i in range(len(pred_q25))]
            
        # 确保所有数据都是numpy数组
        pred_q25 = np.array(pred_q25)
        pred_q75 = np.array(pred_q75)
        actual = np.array(actual)
        
        # 生成所有图表
        self.plot_predictions_vs_actual(dates, pred_q25, pred_q75, actual)
        self.plot_prediction_error(dates, pred_q25, pred_q75, actual)
        self.plot_trading_risk_heatmap(dates, pred_q25, pred_q75, actual)
        
        # 计算并保存评估指标
        metrics = self.calculate_evaluation_metrics(pred_q25, pred_q75, actual)
        
        import json
        with open(os.path.join(self.save_dir, "evaluation_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"\n所有可视化图表和指标已保存到 {self.save_dir} 目录")
        return metrics
        
    def calculate_evaluation_metrics(self, pred_q25, pred_q75, actual):
        """
        计算评估指标
        
        Args:
            pred_q25: Q25预测值
            pred_q75: Q75预测值
            actual: 实际价格
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        # 基本指标
        mae_q25 = np.mean(np.abs(pred_q25 - actual))
        mae_q75 = np.mean(np.abs(pred_q75 - actual))
        mse_q25 = np.mean(np.square(pred_q25 - actual))
        mse_q75 = np.mean(np.square(pred_q75 - actual))
        
        # 区间相关指标
        below_q25 = (actual < pred_q25).sum() / len(actual)
        above_q75 = (actual > pred_q75).sum() / len(actual)
        interval_coverage = 1 - below_q25 - above_q75  # 区间覆盖率
        
        # Q25和Q75的相对关系
        q_diff_mean = np.mean(pred_q75 - pred_q25)
        q_violations = (pred_q75 < pred_q25).sum() / len(pred_q25)  # Q75小于Q25的比例
        
        # 交易风险指标
        sell_risk = np.zeros_like(pred_q25)
        sell_mask = actual < pred_q25
        if np.any(sell_mask):
            sell_risk[sell_mask] = (pred_q25[sell_mask] - actual[sell_mask]) / pred_q25[sell_mask] * 100
            
        buy_risk = np.zeros_like(pred_q25)
        buy_mask = actual > pred_q75
        if np.any(buy_mask):
            buy_risk[buy_mask] = (actual[buy_mask] - pred_q75[buy_mask]) / pred_q75[buy_mask] * 100
            
        avg_sell_risk = np.mean(sell_risk[sell_risk > 0]) if np.any(sell_risk > 0) else 0
        avg_buy_risk = np.mean(buy_risk[buy_risk > 0]) if np.any(buy_risk > 0) else 0
        
        # 收集所有指标
        metrics = {
            "基本指标": {
                "mae_q25": float(mae_q25),
                "mae_q75": float(mae_q75),
                "avg_mae": float((mae_q25 + mae_q75) / 2),
                "mse_q25": float(mse_q25),
                "mse_q75": float(mse_q75),
                "avg_mse": float((mse_q25 + mse_q75) / 2)
            },
            "区间指标": {
                "below_q25_rate": float(below_q25),
                "above_q75_rate": float(above_q75),
                "interval_coverage": float(interval_coverage),
                "ideal_coverage": 0.5  # 理想覆盖率为50%
            },
            "分位数关系": {
                "q_diff_mean": float(q_diff_mean),
                "q_violations_rate": float(q_violations)
            },
            "交易风险": {
                "avg_sell_risk": float(avg_sell_risk),
                "avg_buy_risk": float(avg_buy_risk)
            }
        }
        
        return metrics
