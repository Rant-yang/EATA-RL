import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_provider.data_factory import data_provider
from engine import Engine
from tracker import Tracker

# 1. 解析参数（与main.py兼容，增加lookAHEAD和目标列支持）
parser = argparse.ArgumentParser(description="NEMoTS Dual Target Arguments")
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='nvda_tech_daily.csv', help='data file')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--freq', type=str, default='d')
parser.add_argument('--lookBACK', type=int, default=84)
parser.add_argument('--lookAHEAD', type=int, default=10)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--round', type=int, default=5)
parser.add_argument('--target_q25', type=str, default='Q25')
parser.add_argument('--target_q75', type=str, default='Q75')
# === main.py参数全量补齐 ===
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--used_dimension', type=int, default=1)
parser.add_argument('--symbolic_lib', type=str, default='NEMoTS')
parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--max_module_init', type=int, default=10)
parser.add_argument('--num_transplant', type=int, default=2)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--num_aug', type=int, default=0)
parser.add_argument('--exploration_rate', type=float, default=1 / np.sqrt(2))
parser.add_argument('--transplant_step', type=int, default=1000)
parser.add_argument('--norm_threshold', type=float, default=1e-5)
parser.add_argument('--seq_in', type=int, default=84, help='length of input seq')
parser.add_argument('--seq_out', type=int, default=12, help='length of output seq')
parser.add_argument('--train_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clip')
parser.add_argument('--recording', action='store_true')
parser.add_argument('--tag', type=str, default='records')
parser.add_argument('--logtag', type=str, default='records_logtag')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
print('Args in experiment:')
print(args)

# 设置日志目录
log_dir = 'logs/dual_target'
os.makedirs(log_dir, exist_ok=True)
os.makedirs('evaluation', exist_ok=True)

# 创建含有3列的q25/q75记录器
tracker = Tracker(save_dir="logs/dual_target", dual_target=True)

# 强制生成初始化图表
def force_create_initial_charts():
    """强制生成初始化图表，确保即使在数据收集前也有图表生成"""
    print("\n[信息] 强制生成初始化图表")
    # 创建目录
    import os
    os.makedirs("logs/dual_target", exist_ok=True)
    
    # 创建一个简单的初始化图表
    import matplotlib.pyplot as plt
    
    # 初始化内容图
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, "模型训练中 - 正在收集数据\n还没有指标数据", ha="center", va="center", fontsize=14)
    plt.title("Training Progress - Initializing")
    plt.tight_layout()
    plt.savefig(os.path.join("logs/dual_target", "training_initializing.png"))
    plt.close()
    
    # Q25/Q75对比图
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, "还没有Q25/Q75指标数据\n等待收集中...", ha="center", va="center", fontsize=14)
    plt.title("Q25/Q75 Metrics - Waiting")
    plt.tight_layout()
    plt.savefig(os.path.join("logs/dual_target", "q25_q75_waiting.png"))
    plt.close()
    
    print(f"[完成] 初始图表已生成到 logs/dual_target/ 目录")

# 立即生成初始化图表
force_create_initial_charts()

# 2. 共享神经网络
from network import PVNetCtx
import pandas as pd
# 使用symbolics中的强化金融文法
import symbolics

def gen_multivar_grammar(n, lookBACK):
    """简化引用symbolics中的金融算子集"""
    # 如果默认生成以前的简化版文法(自带的基础数学运算)
    if False:  # 选项-后续可通过参数切换
        terminals = [f'A->x{i}' for i in range(n*lookBACK)]
        ops = [
            'A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
            'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C', 'A->log(A)', 'A->sqrt(A)'
        ]
        return ops + terminals
    else:  # 使用新强化金融算子集
        return symbolics.gen_enhanced_finance_grammar(n, lookBACK)

# 动态生成grammars
lookBACK = args.lookBACK
csv_path = args.root_path + args.data_path if not args.data_path.startswith('/') else args.data_path

df = pd.read_csv(csv_path)
feature_cols = [c for c in df.columns if c not in ['date', 'Q25', 'Q75']]
n = len(feature_cols)
grammars = gen_multivar_grammar(n, lookBACK)
shared_network = PVNetCtx(grammars=grammars, num_transplant=2, device=args.device)

# 3. 数据加载函数，支持切换目标

def get_data_for_target(target_col):
    class ArgsWrapper:
        def __init__(self, base_args, target):
            self.__dict__.update(vars(base_args))
            self.target = target
    wrapped_args = ArgsWrapper(args, target_col)
    data_set, data_loader = data_provider(wrapped_args, flag='train')
    
    # 从data_loader获取一个batch的数据
    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        return batch_x, batch_y

# 4. 定义ArgsWrapper用于动态添加target和共享network
class ArgsWrapper:
    def __init__(self, base_args, target, shared_network):
        for k, v in vars(base_args).items():
            setattr(self, k, v)
        self.target = target
        self.shared_network = shared_network

# 回到原始方案：维护两个MCTS树，共享神经网络
print("=== 原始方案: 维护两个MCTS树，共享一个神经网络 ===")

# 创建第一个引擎
args_q25 = ArgsWrapper(args, args.target_q25, shared_network)
engine_q25 = Engine(args_q25)

# 创建第二个引擎，共享第一个引擎的神经网络
args_q75 = ArgsWrapper(args, args.target_q75, shared_network)
engine_q75 = Engine(args_q75)

# 让第二个引擎的模型共享第一个引擎的神经网络
engine_q75.model.p_v_net_ctx = engine_q25.model.p_v_net_ctx

print(f"Q25引擎目标列: {engine_q25.args.target}")
print(f"Q75引擎目标列: {engine_q75.args.target}")
print(f"神经网络共享检查: {'SUCCESS' if engine_q25.model.p_v_net_ctx is engine_q75.model.p_v_net_ctx else 'FAILED'}")

# 使用增强MCTS适配器解决维度不匹配问题
print("使用增强MCTS适配器解决维度不匹配问题...")

# 导入适配器
from mcts_adapter import MCTSAdapter

# 直接修补两个引擎
# 使用新的patch_engine方法，保证每次创建MCTS时都会正确适配
# 这比之前的方法更完善和高级
patched_q25 = MCTSAdapter.patch_engine(engine_q25)
patched_q75 = MCTSAdapter.patch_engine(engine_q75)

print(f"Q25引擎适配器补丁状态: {'SUCCESS' if patched_q25 else 'FAILED'}")
print(f"Q75引擎适配器补丁状态: {'SUCCESS' if patched_q75 else 'FAILED'}")

print(f"Q25 grammar vocab size: {len(engine_q25.model.p_v_net_ctx.grammar_vocab)}")
print(f"Q75 grammar vocab size: {len(engine_q75.model.p_v_net_ctx.grammar_vocab)}")
print(f"神经网络对象ID: {id(engine_q25.model.p_v_net_ctx.pv_net)}")

# 对试运行检查，确保MCTS适配器被激活
print("检查是否启用MCTS适配器:", getattr(engine_q25.model, 'use_adapter', False))
print("检查是否启用MCTS适配器:", getattr(engine_q75.model, 'use_adapter', False))

# 辅助函数：用于切换引擎的目标列
def switch_target(engine, target_name):
    """\u5207\u6362\u5f15\u64ce\u7684\u76ee\u6807\u5217"""
    print(f"\u5207\u6362\u76ee\u6807\u5217: {engine.args.target} -> {target_name}")
    engine.args.target = target_name
    return engine
    
# 辅助函数：打印预测结果
def print_predictions(pred_q25, pred_q75):
    """打印Q25和Q75预测结果比较"""
    print("\n=== 预测结果比较 ===")
    print(f"Q25预测: {pred_q25}")
    print(f"Q75预测: {pred_q75}")
    print(f"Q75-Q25差异: {pred_q75 - pred_q25}")
    if (pred_q75 >= pred_q25).all():
        print("结果合理: Q75 大于等于 Q25")
    else:
        print("警告: 有些Q75值小于Q25值")

# 5. 主训练循环
def train_dual_targets():
    # 采用方案一：维护两个MCTS树，共享一个神经网络
    print("\n===== 原始方案：两个MCTS树共享神经网络 =====")
    
    # 确认神经网络共享状态
    print(f"神经网络共享检查: {'SUCCESS' if engine_q25.model.p_v_net_ctx is engine_q75.model.p_v_net_ctx else 'FAILED'}")
    
    # 新方案中我们首先填充经验池（两个引擎各自的经验池）
    print("\n填充经验池...")
    buffer_size = engine_q25.model.data_buffer.maxlen
    warmup_size = int(buffer_size * 0.1)
    
    # 同时填充两个引擎的经验池直到达到阈值
    while len(engine_q25.model.data_buffer) < warmup_size or len(engine_q75.model.data_buffer) < warmup_size:
        # 获取两个目标的数据
        X_q25, _ = get_data_for_target(args.target_q25)
        X_q75, _ = get_data_for_target(args.target_q75)
        
        # 确保数据是正确的形状和类型
        if not isinstance(X_q25, torch.Tensor):
            X_q25 = torch.tensor(X_q25, dtype=torch.float32)
        if X_q25.dim() == 2:  # 如果是[seq_len, feature_dim]
            X_q25 = X_q25.unsqueeze(0)  # 变成[1, seq_len, feature_dim]
            
        if not isinstance(X_q75, torch.Tensor):
            X_q75 = torch.tensor(X_q75, dtype=torch.float32)
        if X_q75.dim() == 2:  # 如果是[seq_len, feature_dim]
            X_q75 = X_q75.unsqueeze(0)  # 变成[1, seq_len, feature_dim]
        
        # 分别填充两个引擎的经验池
        try:
            # 使用MCTS适配器解决维度不匹配问题
            engine_q25.simulate(X_q25)
            engine_q75.simulate(X_q75)
            # 只在进度变化时输出
            buffer_q25 = len(engine_q25.model.data_buffer)
            buffer_q75 = len(engine_q75.model.data_buffer)
            
            # 记录经验池收集当前状态
            if (buffer_q25 % 100 == 0 or buffer_q75 % 100 == 0) and buffer_q25 > 0 and buffer_q75 > 0:
                # 更新跟踪器指标
                tracker.update(
                    step=buffer_q25,  # 使用经验池大小作为步数
                    alpha=0.0,
                    reward=0,
                    value=0,
                    best_score=0
                )
                tracker.update_dual_metrics(
                    step=buffer_q25,
                    mae_q25=0,
                    mae_q75=0
                )
                
                # 每300次生成一次经验池阶段图表
                if buffer_q25 % 300 == 0:
                    print(f"\n[信息] 生成经验池阶段图表 (已收集{buffer_q25}/{warmup_size})")
                    tracker.plot(save_prefix=f"exp_pool_{buffer_q25}")
                    print(f"[完成] 图表已保存到 {tracker.save_dir} 目录")
                    plt.close('all')
            
            if buffer_q25 % 10 == 0 or buffer_q75 % 10 == 0:
                print(f"经验池: Q25={buffer_q25}/{warmup_size}, Q75={buffer_q75}/{warmup_size}", end="\r")
        except Exception as e:
            # 只记录关键错误
            print(f"错误: {e}")
            continue
    
    print("\n经验池填充完成，开始训练")
    print("----------------------------------------")
    
    # 保存经验池填充阶段的情况
    print("\n[信息] 生成经验池阶段图表")
    tracker.plot(save_prefix="experience_pool_phase")
    print(f"[完成] 已保存图表到 {os.path.join(tracker.save_dir, 'experience_pool_phase_*.png')}")
    plt.close('all')
    
    # 开始训练循环
    best_score_q25 = 0
    best_score_q75 = 0
    
    # 确保日志目录存在
    log_dir = os.path.join(tracker.save_dir)  # 确保路径正确
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n图表保存目录: {log_dir}")
    print("\n开始训练 - 方案一: 两个MCTS树共享神经网络\n")
    
    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}/{args.epoch}", end="\r")
        
        # 训练Q25引擎
        loss_q25 = engine_q25.train()
        
        # 训练Q75引擎
        loss_q75 = engine_q75.train()
        
        # 每10个epoch评估一次并生成图表
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epoch - 1:
            # 简单评估 - 获取当前最优表达式
            X_test, y_test_q25 = get_data_for_target(args.target_q25)
            _, y_test_q75 = get_data_for_target(args.target_q75)
            
            pred_q25 = engine_q25.predict(X_test)
            pred_q75 = engine_q75.predict(X_test)
            
            # 计算MAE
            mae_q25 = np.mean(np.abs(pred_q25 - y_test_q25))
            mae_q75 = np.mean(np.abs(pred_q75 - y_test_q75))
            
            # 更新最优分数
            if 1.0/(1.0 + mae_q25) > best_score_q25:
                best_score_q25 = 1.0/(1.0 + mae_q25)
            if 1.0/(1.0 + mae_q75) > best_score_q75:
                best_score_q75 = 1.0/(1.0 + mae_q75)
                
            # 测量Q25/Q75间距的合理性 - 计算Q75小于Q25的比例
            q_violations = (pred_q75 < pred_q25).sum() / len(pred_q25)
            q_diff = np.mean(pred_q75 - pred_q25)
            
            # 记录基本训练指标
            tracker.update(
                step=epoch+1,
                alpha=0.5,  # 代表Q25和Q75的平均α值
                value=float((mae_q25 + mae_q75)/2),  # 平均MAE
                reward=float(-(mae_q25 + mae_q75)),  # 负MAE作为奖励
                corr=float(1 - q_violations),  # Q25/Q75关系正确率
                best_score=float((best_score_q25 + best_score_q75)/2)  # 平均最优分数
            )
            
            # 记录双目标特有指标
            tracker.update_dual_metrics(
                step=epoch+1,
                mae_q25=float(mae_q25),
                mae_q75=float(mae_q75),
                pred_q25=pred_q25,
                pred_q75=pred_q75
            )
            
            # 每10个epoch生成一次训练图表
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epoch:
                print(f"\n[信息] 生成训练中图表 - Epoch {epoch+1}")
                tracker.plot(save_prefix=f"dual_target_epoch_{epoch+1}")
                print(f"[完成] 已保存图表到 {os.path.join(tracker.save_dir, f'dual_target_epoch_{epoch+1}_*.png')}")
                # 每次生成图表后强制刷新缓冲区
                plt.close('all')
            
            if epoch % 50 == 0 or epoch == args.epoch - 1:
                print(f"\nEpoch {epoch+1}: MAE(Q25)={mae_q25:.4f}, MAE(Q75)={mae_q75:.4f}, Q违规率={q_violations*100:.2f}%")
    
    print("\n训练完成！生成最终评估报告...")

    # 评估阶段 - 使用两个引擎分别预测Q25和Q75
    # 获取测试数据
    X_q25, y_q25 = get_data_for_target(args.target_q25)
    X_q75, y_q75 = get_data_for_target(args.target_q75)
    
    # 确保数据格式正确
    if not isinstance(X_q25, torch.Tensor):
        X_q25 = torch.tensor(X_q25, dtype=torch.float32)
    if X_q25.dim() == 2:
        X_q25 = X_q25.unsqueeze(0)
    
    if not isinstance(X_q75, torch.Tensor):
        X_q75 = torch.tensor(X_q75, dtype=torch.float32)
    if X_q75.dim() == 2:
        X_q75 = X_q75.unsqueeze(0)
    
    # 使用两个引擎分别预测
    print("\n正在生成预测...")
    pred_q25 = engine_q25.predict(X_q25)
    pred_q75 = engine_q75.predict(X_q75)
    
    # 生成日期序列用于可视化
    dates = [f"T{i}" for i in range(len(pred_q25))]
    
    # 使用Tracker的generate_final_report生成完整的预测结果分析
    print("\n生成可视化图表和统计报告...")
    metrics = tracker.generate_final_report(
        pred_q25=pred_q25,
        pred_q75=pred_q75,
        actual=y_q25,  # 使用y_q25作为实际值
        dates=dates
    )
    
    # 打印关键指标摘要
    print("\n----- 主要预测指标摘要 -----")
    print(f"MAE (Q25): {metrics['基本指标']['mae_q25']:.4f}")
    print(f"MAE (Q75): {metrics['基本指标']['mae_q75']:.4f}")
    print(f"平均 MAE: {metrics['基本指标']['avg_mae']:.4f}")
    print(f"MSE (Q25): {metrics['基本指标']['mse_q25']:.4f}")
    print(f"MSE (Q75): {metrics['基本指标']['mse_q75']:.4f}")
    
    # 打印交易风险信息
    below_rate = metrics['区间指标']['below_q25_rate'] * 100
    above_rate = metrics['区间指标']['above_q75_rate'] * 100
    print(f"\n----- 交易风险分析 -----")
    print(f"卖出风险（实际<Q25）: {below_rate:.2f}%")
    print(f"买入风险（实际>Q75）: {above_rate:.2f}%")
    print(f"总体预测区间超出率: {below_rate + above_rate:.2f}%")
    
    # 检查分位数关系
    violations = metrics['分位数关系']['q_violations_rate'] * 100
    if violations > 0:
        print(f"\n\u8b66告: 有 {violations:.2f}% 的样本中Q75小于Q25，需要进一步优化模型")
    else:
        print("\n分位数关系均正确: 所有样本的Q75 >= Q25")
    
    # 保存预测结果为CSV
    try:
        import pandas as pd
        results_df = pd.DataFrame({
            'date': dates,
            'pred_q25': pred_q25,
            'pred_q75': pred_q75,
            'actual': y_q25,
            'target_q25': y_q25,
            'target_q75': y_q75
        })
        os.makedirs('evaluation', exist_ok=True)
        results_df.to_csv('evaluation/predictions.csv', index=False)
        print("\n预测结果已保存到 evaluation/predictions.csv")
    except Exception as e:
        print(f"\n保存预测结果时出错: {e}")
        
    print(f"\n图表和评估指标已生成到: {tracker.eval_dir}")
    print("\n===== 完成 =====\n")

if __name__ == '__main__':
    train_dual_targets()
