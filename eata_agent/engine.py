import math
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as op
from scipy.stats import pearsonr
from torch.distributions import Categorical

from .model import Model
from .tracker import Tracker

# Define the device for PyTorch for hardware acceleration
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"INFO: Using PyTorch device: {device}")

class Engine(object):
    def __init__(self, args):
        self.args = args
        # Set the device in args to be passed to the Model and subsequent classes
        self.args.device = device
        self.model = Model(args)
        # The model and its sub-modules are now initialized on the correct device via their constructors.
        self.optimizer = op.Adam(self.model.p_v_net_ctx.pv_net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.tracker = Tracker()
        self.global_train_step = 0

    def simulate(self, data, previous_best_tree=None):
        if isinstance(data, torch.Tensor):
            if data.dim() == 3 and data.shape[0] == 1:
                data = data.squeeze(0)
            data = data.cpu().numpy()

        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        X = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)

        # model.run() 现在返回原始的MCTS经验记录 (mcts_records)
        all_eqs, all_times, test_scores, mcts_records, policy, mcts_score, new_best_tree = self.model.run(X, y_tensor, previous_best_tree=previous_best_tree)
        
        # 不再在这里自动存储和训练
        # self.model.data_buffer.extend(supervision_data)
        # if len(self.model.data_buffer) > self.args.train_size:
        #     self.train()

        mae, mse, corr, best_exp, top_10_exps, top_10_scores = OptimizedMetrics.metrics(all_eqs, test_scores, y)

        # 在返回值中增加 mcts_records，以便agent.py可以接收
        return best_exp, top_10_exps, top_10_scores, all_times, mae, mse, corr, policy, mcts_score, new_best_tree, mcts_records

    def store_experiences(self, experiences):
        """接收由Agent处理过的、包含最终rl_reward的完整经验，并触发训练"""
        self.model.data_buffer.extend(experiences)
        print(f"  [经验池] 存入 {len(experiences)} 条新经验。当前经验池大小: {len(self.model.data_buffer)}")
        if len(self.model.data_buffer) > self.args.train_size:
            self.train()




    def train(self):
        self.model.p_v_net_ctx.pv_net.train()
        print("start train neural networks...")
        cumulative_loss = 0
        
        # preprocess_data 现在返回5个批次的数据
        state_batch_full, seq_batch_full, policy_batch_full, value_batch_full, rl_reward_batch_full, _ = self.preprocess_data()

        if not state_batch_full:
            print("[WARN] No data sampled from memory buffer for training.")
            return 0

        mini_batch_size = 64

        for epoch in range(self.args.epoch):
            indices = list(range(len(state_batch_full)))
            random.shuffle(indices)
            
            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(state_batch_full), mini_batch_size):
                self.optimizer.zero_grad()

                mini_batch_indices = indices[i:i + mini_batch_size]

                state_batch = [state_batch_full[j] for j in mini_batch_indices]
                seq_batch = [seq_batch_full[j] for j in mini_batch_indices]
                policy_batch = [policy_batch_full[j] for j in mini_batch_indices]
                value_batch = torch.Tensor([value_batch_full[j] for j in mini_batch_indices]).to(device)
                rl_reward_batch = torch.Tensor([rl_reward_batch_full[j] for j in mini_batch_indices]).to(device) # 新增：盈利奖励批次
                
                length_indices = self.obtain_policy_length(policy_batch)

                if len(state_batch) == 0 or len(seq_batch) == 0:
                    continue

                # 网络现在返回三个头的输出
                raw_dis_out, value_out, profit_out = self.model.p_v_net_ctx.policy_value_batch(seq_batch, state_batch)

                value_batch[torch.isnan(value_batch)] = 0.
                rl_reward_batch[torch.isnan(rl_reward_batch)] = 0.

                # 1. 价值头损失 (V_accuracy)
                value_loss = F.mse_loss(value_out.squeeze(-1), value_batch.to(value_out.device))

                # 2. 新增：盈利预测头损失 (V_profit)
                profit_loss = F.mse_loss(profit_out.squeeze(-1), rl_reward_batch.to(profit_out.device))

                # 3. 策略头损失
                dist_loss = []
                if length_indices:
                    for length, sample_id in length_indices.items():
                        if not sample_id: continue
                        out_policy = F.softmax(torch.stack([raw_dis_out[k] for k in sample_id])[:, :length], dim=-1)
                        gt_policy = torch.Tensor([policy_batch[k] for k in sample_id]).to(device)
                        dist_target = Categorical(probs=gt_policy)
                        dist_out = Categorical(probs=out_policy)
                        dist_loss.append(torch.distributions.kl_divergence(dist_target, dist_out).mean())
                
                # 合并三个损失
                # TODO: 未来可以为不同损失引入权重
                if not dist_loss or not any(dist_loss):
                    total_loss = value_loss + profit_loss
                else:
                    total_loss = value_loss + profit_loss + sum(dist_loss)
                
                epoch_loss += total_loss.item()
                num_batches += 1

                total_loss.backward()
                if self.args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.p_v_net_ctx.pv_net.parameters(), self.args.clip)
                
                self.optimizer.step()

                self.global_train_step += 1
            
            if num_batches > 0:
                cumulative_loss += (epoch_loss / num_batches)

        print("end train neural networks...")
        self.tracker.plot()
        self.tracker.save_npz()
        return cumulative_loss / self.args.epoch if self.args.epoch > 0 else 0

    def obtain_policy_length(self, policy):
        length_indices = defaultdict(list)
        for idx, sublist in enumerate(policy):
            length_indices[len(sublist)].append(idx)
        return dict(length_indices)

    def preprocess_data(self):
        # 经验元组现在有5个元素，最后一个是rl_reward
        non_nan_indices = [index for index, value in enumerate(self.model.data_buffer) if not math.isnan(value[3]) and not math.isnan(value[4])]
        if not non_nan_indices:
            return [], [], [], [], [], {}
            
        sampled_idx = random.sample(non_nan_indices, min(len(non_nan_indices), self.args.train_size))
        raw_mini_batch = [self.model.data_buffer[i] for i in sampled_idx]

        mini_batch = []
        for data in raw_mini_batch:
            # 检查新的5元组格式
            if isinstance(data, (list, tuple)) and len(data) >= 5 and isinstance(data[1], np.ndarray) and data[1].ndim >= 1:
                mini_batch.append(data)
            else:
                print(f"[WARN] Filtering out malformed experience data during preprocessing: {data}")

        if not mini_batch:
            return [], [], [], [], [], {}

        state_batch = [data[0] for data in mini_batch]
        seq_batch = [data[1][1] for data in mini_batch]
        policy_batch = [data[2] for data in mini_batch]
        value_batch = [data[3] for data in mini_batch]
        rl_reward_batch = [data[4] for data in mini_batch] # 新增：解包rl_reward

        length_indices = self.obtain_policy_length(policy_batch)
        return state_batch, seq_batch, policy_batch, value_batch, rl_reward_batch, length_indices

    def eval(self, data):
        pass

class OptimizedMetrics:
    @staticmethod
    def metrics(exps, scores, data, top_k=10):
        if scores is None or len(scores) == 0:
            return 0.0, 0.0, 0.0, "0", [], []

        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        top_exps = [exps[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        best_exp = top_exps[0]

        eval_vars = {"np": np}
        # Assuming data shape is (features, timesteps)
        num_var = data.shape[0]
        for i in range(num_var):
            eval_vars[f'x{i}'] = data[i, :]
        # The ground truth is the target variable, let's assume it's the 4th feature (index 3, e.g., close price)
        gt = data[3, :]

        corrected_expression = best_exp.replace("exp", "np.exp").replace("cos", "np.cos").replace("sin", "np.sin").replace("sqrt", "np.sqrt").replace("log", "np.log")
        
        try:
            prediction = eval(corrected_expression, {"__builtins__": None}, eval_vars)
            if not isinstance(prediction, np.ndarray) or prediction.shape != gt.shape:
                if isinstance(prediction, (int, float)):
                    prediction = np.repeat(prediction, gt.shape)
                else:
                    prediction = np.zeros_like(gt)
        except Exception:
            prediction = np.zeros_like(gt)

        mae = np.mean(np.abs(prediction - gt))
        mse = np.mean((prediction - gt) ** 2)
        
        corr = 0.0
        try:
            if np.any(np.isinf(prediction)) or np.any(np.isnan(prediction)):
                 prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

            if len(prediction) == len(gt) and np.std(prediction) > 0 and np.std(gt) > 0:
                corr, _ = pearsonr(prediction, gt)
            else:
                corr = 0.0
        except (ValueError, TypeError):
            corr = 0.0
        
        if np.isnan(corr):
            corr = 0.0

        return mae, mse, corr, best_exp, top_exps, top_scores

# Example usage (assuming exps, scores, and data are defined)
# metrics = OptimizedMetrics.metrics(exps, scores, data)
#Engine 类是连接高层控制 (main.py) 和底层算法实现 (model.py, network.py, mcts.py) 的核心粘合剂。