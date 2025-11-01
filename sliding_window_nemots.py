#!/usr/bin/env python
# coding=utf-8
# 直接调用核心模块：engine.simulate → model.run → mcts + network

import numpy as np
import torch
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 导入NEMoTS核心模块
from eata_agent.engine import Engine
from eata_agent.args import Args


class SlidingWindowNEMoTS:
    
    def __init__(self, lookback: int = 20, lookahead: int = 5):
        """
        初始化滑动窗口NEMoTS
        
        Args:
            lookback: 训练窗口大小（对应原NEMoTS的seq_in）
            lookahead: 预测窗口大小（对应原NEMoTS的seq_out）
        """
        self.lookback = lookback
        self.lookahead = lookahead
        
        # 从main函数迁移的超参数
        self.hyperparams = self._create_hyperparams()
        
        # 初始化引擎
        self.engine = Engine(self.hyperparams)
        
        # 语法树继承
        self.previous_best_tree = None
        self.previous_best_expression = None
        
        # 训练状态
        self.is_trained = False
        self.training_history = []
        
        print(f"滑动窗口NEMoTS初始化完成")
        print(f"   lookback={lookback}, lookahead={lookahead}")
        print(f"   核心模块: engine → model → mcts + network")
    
    def _create_hyperparams(self) -> Args:
        """
        创建超参数配置（从main函数迁移）
        将main函数的超参数设置移到engine模块
        """
        args = Args()
        
        # 设备配置
        args.device = torch.device("cpu")
        args.seed = 42
        
        # 数据配置（适配滑动窗口）
        args.seq_in = self.lookback
        args.seq_out = self.lookahead
        args.used_dimension = 1
        args.features = 'M'  # 多变量预测多变量
        
        # NEMoTS核心参数
        args.symbolic_lib = "NEMoTS"
        args.max_len = 25
        args.max_module_init = 10
        args.num_transplant = 5
        args.num_runs = 3  # 减少运行次数以适应滑动窗口
        args.eta = 1.0
        args.num_aug = 5
        args.exploration_rate = 1 / np.sqrt(2)
        args.transplant_step = 500  # 减少步数以适应滑动窗口
        args.norm_threshold = 1e-5
        
        # 训练参数（适配滑动窗口）
        args.epoch = 10  # 减少epoch以适应实时性
        args.round = 2   # 减少round以适应滑动窗口
        args.train_size = 64  # 减少batch size
        args.lr = 1e-5
        args.weight_decay = 0.0001
        args.clip = 5.0
        args.buffer_size = 64 # 明确设置经验池大小，确保alpha系数能快速增长
        
        # 随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        print(f"超参数配置完成（从main函数迁移）")
        return args
    
    def _prepare_sliding_window_data(self, df: pd.DataFrame) -> torch.Tensor:
        """
        准备滑动窗口数据
        基于RL范式的数据处理，替代全序列拟合
        """
        # 选择特征列
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        data = df[feature_cols].values
        
        # 数据标准化（使用变化率）
        normalized_data = []
        for i in range(1, len(data)):
            row = []
            # 价格变化率
            for j in range(4):  # open, high, low, close
                if data[i-1, j] != 0:
                    change_rate = (data[i, j] - data[i-1, j]) / data[i-1, j]
                    change_rate = np.clip(change_rate, -0.1, 0.1)  # 限制变化率
                else:
                    change_rate = 0.0
                row.append(change_rate)
            
            # 成交量变化率
            for j in [4, 5]:  # volume, amount
                if data[i-1, j] > 0 and data[i, j] > 0:
                    vol_change = (data[i, j] - data[i-1, j]) / data[i-1, j]
                    vol_change = np.clip(vol_change, -0.5, 0.5)
                else:
                    vol_change = 0.0
                row.append(vol_change)
            
            normalized_data.append(row)
        
        normalized_data = np.array(normalized_data)
        
        # 创建滑动窗口
        if len(normalized_data) < self.lookback + self.lookahead:
            raise ValueError(f"数据长度不足：需要{self.lookback + self.lookahead}，实际{len(normalized_data)}")
        
        # 取最后一个窗口的数据
        start_idx = len(normalized_data) - self.lookback - self.lookahead
        window_data = normalized_data[start_idx:start_idx + self.lookback + self.lookahead]
        
        # 转换为tensor格式，添加batch维度
        # tensor_data = torch.FloatTensor(window_data).unsqueeze(0)  # [1, seq_len, features]
        
        print(f"滑动窗口数据准备完成:")
        print(f"   原始数据: {len(data)} → 标准化数据: {len(normalized_data)}")
        # print(f"   窗口数据: {tensor_data.shape}")
        # print(f"   变化率范围: [{tensor_data.min().item():.4f}, {tensor_data.max().item():.4f}]")
        
        return window_data
    
    def _inherit_previous_tree(self):
        """
        语法树继承机制
        支持前一窗口最优解传递
        """
        if self.previous_best_tree is not None:
            print(f"继承前一窗口最优语法树: {self.previous_best_expression}")
            print(f"   继承的表达式类型: {type(self.previous_best_tree)}")
            print(f"   继承机制正常工作 - 将传递给MCTS作为初始解")
            # 这里可以将前一窗口的最优解传递给MCTS作为初始解
            # 具体实现需要在model/mcts模块中添加参数支持
            return self.previous_best_tree
        else:
            print(f"首次训练，无语法树可继承")
            return None
    
    def sliding_fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        滑动窗口训练
        """
        print(f"\n开始滑动窗口训练...")

        # 动态调整参数
        if self.previous_best_tree is not None:
            # 后续窗口，使用轻量参数
            print("检测到已有语法树，切换到轻量化快速迭代参数...")
            # 直接修改Model对象内部的参数以确保生效
            self.engine.model.num_transplant = 2
            self.engine.model.transplant_step = 100
            self.engine.model.num_aug = 2
        else:
            # 首次窗口，使用重量参数
            print("首次运行，使用重量级深度搜索参数...")
            # 确保Model对象使用的是重量级参数
            self.engine.model.num_transplant = 5
            self.engine.model.transplant_step = 500
            self.engine.model.num_aug = 5

        
        try:
            # 1. 准备滑动窗口数据
            window_data = self._prepare_sliding_window_data(df)
            
            # 2. 语法树继承
            inherited_tree = self._inherit_previous_tree()
            
            # 3. 直接调用engine.simulate（简化调用链，传递继承的语法树）
            print(f"调用核心模块: engine.simulate...")
            best_exp, all_times, test_data, loss, mae, mse, corr, policy, reward, new_best_tree = self.engine.simulate(window_data, previous_best_tree=inherited_tree)
            
            # 4. 保存最优解供下次继承
            self.previous_best_expression = str(best_exp)
            self.previous_best_tree = new_best_tree  # 核心修复：保存正确的树节点对象
            
            # 5. 更新训练状态
            self.is_trained = True
            
            # 6. 记录训练历史
            training_record = {
                'timestamp': pd.Timestamp.now(),
                'best_expression': str(best_exp),
                'mae': mae,
                'mse': mse,
                'corr': corr,
                'reward': reward,
                'loss': loss
            }
            self.training_history.append(training_record)
            
            print(f"滑动窗口训练完成")
            print(f"   最优表达式: {best_exp}")
            print(f"   MAE: {mae:.4f}, MSE: {mse:.4f}, Corr: {corr:.4f}")
            print(f"   Reward: {reward:.4f}, Loss: {loss:.4f}")
            
            return {
                'success': True,
                'best_expression': str(best_exp),
                'metrics': {
                    'mae': mae,
                    'mse': mse,
                    'corr': corr,
                    'reward': reward,
                    'loss': loss
                },
                'inherited_tree': inherited_tree is not None
            }
            
        except Exception as e:
            print(f"滑动窗口训练失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'inherited_tree': False
            }
    
    def predict(self, df: pd.DataFrame) -> int:
        """
        基于训练的模型进行预测
        返回交易信号: -1(卖出), 0(持有), 1(买入)
        """
        if not self.is_trained:
            print("模型未训练，返回随机预测")
            return np.random.choice([-1, 0, 1])
        
        try:
            # 使用最优表达式进行预测
            # 这里简化为基于最近的表达式质量判断
            if len(self.training_history) > 0:
                latest_record = self.training_history[-1]
                
                # 基于MAE和相关性判断
                mae = latest_record['mae']
                corr = latest_record['corr']
                
                # 改进的预测逻辑
                reward = latest_record['reward']
                
                print(f"预测分析: MAE={mae:.4f}, Corr={corr}, Reward={reward:.4f}")
                
                # 主要基于reward进行预测，MAE作为辅助
                if mae < 0.01:  # MAE良好
                    if not np.isnan(corr):
                        # 有有效相关性
                        if corr > 0.1:
                            return 1  # 买入
                        elif corr < -0.1:
                            return -1  # 卖出
                        else:
                            return 0  # 持有
                    else:
                        # 相关性为NaN，基于reward判断（调整阈值）
                        if reward > 0.6:  # 降低买入阈值
                            return 1  # 买入
                        elif reward < 0.4:  # 提高卖出阈值
                            return -1  # 卖出
                        else:
                            return 0  # 持有
                else:
                    # MAE较大，保守持有
                    return 0
            
            return 0  # 默认持有
            
        except Exception as e:
            print(f"预测出错: {e}，返回持有")
            return 0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.training_history:
            return {'trained': False}
        
        latest = self.training_history[-1]
        return {
            'trained': True,
            'total_windows': len(self.training_history),
            'latest_expression': latest['best_expression'],
            'latest_metrics': {
                'mae': latest['mae'],
                'mse': latest['mse'],
                'corr': latest['corr'],
                'reward': latest['reward']
            },
            'has_inheritance': self.previous_best_tree is not None
        }


def test_sliding_window_nemots():
    """测试滑动窗口NEMoTS"""
    print("测试滑动窗口NEMoTS")
    print("=" * 60)
    
    # 创建更真实的测试数据（模拟上涨趋势）
    base_price = 100
    trend_data = []
    for i in range(50):
        # 模拟上涨趋势 + 噪声
        trend = i * 0.2  # 上涨趋势
        noise = np.random.randn() * 0.1
        price = base_price + trend + noise
        
        trend_data.append({
            'open': price - 0.1,
            'high': price + 0.2,
            'low': price - 0.2,
            'close': price,
            'volume': 1000 + i * 5
        })
    
    test_data = pd.DataFrame(trend_data)
    test_data['amount'] = test_data['volume'] * test_data['close']
    
    print(f"测试数据: {len(test_data)}行")
    
    # 创建滑动窗口NEMoTS
    sw_nemots = SlidingWindowNEMoTS(lookback=15, lookahead=3)
    
    # 第一个窗口训练
    print(f"\n 第一个滑动窗口训练...")
    result1 = sw_nemots.sliding_fit(test_data[:30])
    print(f"结果1: {result1['success']}")
    
    # 第二个窗口训练（测试语法树继承）
    print(f"\n 第二个滑动窗口训练（测试继承）...")
    result2 = sw_nemots.sliding_fit(test_data[10:40])
    print(f"结果2: {result2['success']}, 继承: {result2.get('inherited_tree', False)}")
    
    # 预测测试
    print(f"\n 预测测试...")
    for i in range(3):
        pred = sw_nemots.predict(test_data[-10:])
        pred_name = {-1: '卖出', 0: '持有', 1: '买入'}[pred]
        print(f"预测 {i+1}: {pred} ({pred_name})")
    
    # 训练摘要
    summary = sw_nemots.get_training_summary()
    print(f"\n 训练摘要:")
    print(f"   训练状态: {summary['trained']}")
    if summary['trained']:
        print(f"   训练窗口数: {summary['total_windows']}")
        print(f"   最新表达式: {summary['latest_expression']}")
        print(f"   最新指标: MAE={summary['latest_metrics']['mae']:.4f}")
        print(f"   语法树继承: {summary['has_inheritance']}")
    
    print(f"\n 滑动窗口NEMoTS测试完成！")


if __name__ == "__main__":
    test_sliding_window_nemots()
