import gym
import numpy as np
from fodo_lattice import *
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F
import copy
import time
from lattice_simulator import extract_lattice_parameters, compute_orbit
from fodo_lattice import calculate_orbit_diff

# 添加对gym.spaces的导入
from gym import spaces

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数 - 移除了 state_dim 和 action_dim，将在环境创建后动态填充
HYPERPARAMS = {
    'hidden_dim': 128,         # 神经网络隐藏层维度
    'actor_lr': 3e-4,          # Actor网络学习率，控制策略网络参数更新步长
    'critic_lr': 3e-4,         # Critic网络学习率，控制价值网络参数更新步长
    'gamma': 0.99,             # 折扣因子，决定未来奖励的衰减程度，越接近1越重视长远奖励
    'tau': 0.005,              # 软更新参数，控制目标网络参数更新速度，值越小更新越平缓
    'policy_noise': 0.1,       # 目标策略噪声标准差，用于增加探索性和稳定性
    'noise_clip': 0.25,        # 噪声裁剪范围，限制添加到目标策略的噪声幅度
    'policy_freq': 2,          # 策略更新频率（相对于批评家更新）
    'buffer_size': 500000,     # 经验回放缓冲区大小
    'batch_size': 128,         # 批量大小
    'warmup_steps': 1000,      # 预热步数
    'max_episodes': 1000,      # 最大训练回合数，训练的最大迭代次数
    'max_steps_per_episode': 200,  # 每回合最大步数，每个训练回合的最大执行步数
     'early_stopping_patience': 100,  # 早停耐心度，当性能指标在指定回合内没有改善时停止训练
    'early_stopping_threshold': -1000.0,  # 早停奖励阈值，当平均奖励持续低于此值时停止训练
    'reward_weights': {        # 奖励权重，调节不同类型奖励在总奖励中的比重
        'orbit_improvement': 100.0,   # 轨道改善奖励权重，鼓励减小轨道偏差（增加100倍）
        'action_penalty': 10.0,       # 动作惩罚权重，鼓励使用较小的动作调整，在轨道接近目标时会动态增加（增加10倍）
        'completion_bonus': 5000.0,   # 完成奖励，当任务完成时给予的额外奖励（增加50倍）
        'divergence_penalty': 200.0,  # 轨道发散惩罚权重，当轨道发散时给予的惩罚（增加20倍）
        'stability_penalty': 50.0     # 稳定性惩罚权重，鼓励动作变化平稳，避免剧烈波动
    },
    'target_threshold': 1e-5,  # 目标阈值（当所有BPM读数都小于此值时任务完成）
    'action_range': {          # 动作范围，限制每次动作调整的幅度范围
        'min': 0.0001,          # 最小动作幅度，防止动作过小无效果
        'max': 0.001            # 最大动作幅度，防止动作过大导致不稳定
    },
    'cor_angle_limit': 0.1     # 校正器角度限制，限制校正器角度的物理范围
}

class OrbitCorrectionEnv(gym.Env):
    def __init__(self):
        # 初始化晶格
        self.lattice = MagneticLattice(cell)
        self.tws = twiss(self.lattice, nPoints=1000)
        self.lattice_params = extract_lattice_parameters(self.lattice)
        
        # 动态计算校正器和BPM数量
        self.num_hcors = len([elem for elem in self.lattice.sequence if isinstance(elem, Hcor)])
        self.num_vcors = len([elem for elem in self.lattice.sequence if isinstance(elem, Vcor)])
        self.num_bpms = len([elem for elem in self.lattice.sequence if isinstance(elem, Monitor)])
        
        # 获取初始参数
        self.k1_init = get_k1_array(self.lattice)
        self.hcors_init = get_hcors_angle(self.lattice).astype(np.float64)
        self.vcors_init = get_vcors_angle(self.lattice).astype(np.float64)
        
        # 初始化当前校正器角度
        self.current_hcor_angles = self.hcors_init.copy()
        self.current_vcor_angles = self.vcors_init.copy()
        
        # 历史BPM读数（用于状态构建）
        self.prev_orbit = None
        self.prev_prev_orbit = None
        
        # 初始粒子
        self.p_init = Particle(x=0, y=0, px=1e-5, py=-2e-5)
        
        # 目标轨道（理想情况下为0）
        self.target_orbit = np.zeros(self.num_bpms * 2)  # 根据实际BPM数量确定（水平+垂直）
        
        # 动作空间：HCOR和VCOR校正器角度调整增量
        self.action_space = spaces.Box(
            low=-HYPERPARAMS['action_range']['max'],
            high=HYPERPARAMS['action_range']['max'],
            shape=(self.num_hcors + self.num_vcors,),
            dtype=np.float32
        )
        
        # 观察空间：72维（BPM读数 + 差异特征）
        # 状态维度 = BPM读数 + 与上一步差异 + 与上两步差异 = 3 * BPM数量
        state_dim = 3 * self.num_bpms * 2  # 2表示水平和垂直方向
        self.observation_space = spaces.Box(
            low=-1000, high=1000, shape=(state_dim,), dtype=np.float32)
        
        # 初始化BPM读数统计信息
        self.bpm_mean = 0.0
        self.bpm_std = 1.0
        self.bpm_max = 1.0
        self.bpm_min = -1.0
        self.bpm_stats_initialized = False
        self.bpm_sample_count = 0
        
        # 初始化校正器动态权重（根据实际校正器数量）
        self.hcor_weights = np.ones(self.num_hcors)  # 水平校正器权重
        self.vcor_weights = np.ones(self.num_vcors)  # 垂直校正器权重
        
        self.step_count = 0
        
        # 验证维度一致性
        assert len(self.hcor_weights) == self.num_hcors, \
            f"HCOR权重数组维度({len(self.hcor_weights)})与数量({self.num_hcors})不匹配"
        assert len(self.vcor_weights) == self.num_vcors, \
            f"VCOR权重数组维度({len(self.vcor_weights)})与数量({self.num_vcors})不匹配"
        assert self.action_space.shape[0] == self.num_hcors + self.num_vcors, \
            f"动作空间维度({self.action_space.shape[0]})与校正器总数({self.num_hcors + self.num_vcors})不匹配"

    def reset(self):
        # 重置环境到初始状态
        self.current_hcor_angles = self.hcors_init.copy().astype(np.float64)
        self.current_vcor_angles = self.vcors_init.copy().astype(np.float64)
        self.prev_orbit = None
        self.prev_prev_orbit = None
        self.step_count = 0
        
        # 获取初始轨道
        # orb_x, orb_y = get_ideal_orbit(self.lattice, self.k1_init,
        #                                self.current_hcor_angles, self.current_vcor_angles, self.p_init)
        orb_x, orb_y = compute_orbit(self.lattice_params, self.hcors_init, self.vcors_init, self.p_init, energy=1.0)
        current_orbit = np.concatenate([orb_x, orb_y])
        
        # 更新BPM读数统计信息
        self._update_bpm_stats(current_orbit)
        
        # 更新校正器动态权重
        self._update_corrector_weights(current_orbit)
        
        # 对BPM读数进行归一化处理
        normalized_orbit = self._normalize_bpm_readings(current_orbit)
        
        # 构建状态
        _, diff_to_prev, diff_to_prev_prev = calculate_orbit_diff(
            normalized_orbit, self.prev_orbit, self.prev_prev_orbit)
        
        # 更新历史
        self.prev_prev_orbit = self.prev_orbit
        self.prev_orbit = normalized_orbit
        
        # 构建72维状态向量
        state = np.concatenate([normalized_orbit, diff_to_prev, diff_to_prev_prev])
        return state.astype(np.float32)

    def step(self, action):
        # 应用校正器动态权重到动作上
        weighted_action = action.copy()
        if len(action) == self.num_hcors + self.num_vcors:
            weighted_action[:self.num_hcors] = action[:self.num_hcors] * self.hcor_weights
            weighted_action[self.num_hcors:] = action[self.num_hcors:] * self.vcor_weights
        else:
            # 如果维度不匹配，直接使用原始动作
            weighted_action = action
        
        # 验证加权后的动作范围
        validated_action = validate_action(weighted_action)
        
        # 应用动作（调整校正器角度）
        delta_hcors = validated_action[:self.num_hcors]
        delta_vcors = validated_action[self.num_hcors:]
        
        # 更新校正器角度并限制在范围内
        self.current_hcor_angles = self.current_hcor_angles + delta_hcors
        self.current_vcor_angles = self.current_vcor_angles + delta_vcors
        self.current_hcor_angles = clip_cor_angles(self.current_hcor_angles, HYPERPARAMS['cor_angle_limit'])
        self.current_vcor_angles = clip_cor_angles(self.current_vcor_angles, HYPERPARAMS['cor_angle_limit'])
        
        # 计算新的轨道
        # orb_x, orb_y = get_ideal_orbit(self.lattice, self.k1_init,
        #                                self.current_hcor_angles, self.current_vcor_angles, self.p_init)
        orb_x, orb_y = compute_orbit(self.lattice_params, self.current_hcor_angles, self.current_vcor_angles, self.p_init, energy=1.0)
        current_orbit = np.concatenate([orb_x, orb_y])
        
        # 更新BPM读数统计信息
        self._update_bpm_stats(current_orbit)
        
        # 更新校正器动态权重
        self._update_corrector_weights(current_orbit)
        
        # 对BPM读数进行归一化处理
        normalized_orbit = self._normalize_bpm_readings(current_orbit)
        
        # 构建状态
        _, diff_to_prev, diff_to_prev_prev = calculate_orbit_diff(
            normalized_orbit, self.prev_orbit, self.prev_prev_orbit)
        
        # 构建72维状态向量
        state = np.concatenate([normalized_orbit, diff_to_prev, diff_to_prev_prev])
        
        # 计算奖励
        reward = self._compute_reward(normalized_orbit, validated_action)
        
        # 检查是否完成
        self.step_count += 1
        done = False
        if self.step_count >= HYPERPARAMS['max_steps_per_episode']:
            done = True
        else:
            # 使用反归一化数据检查完成条件
            if self.bpm_stats_initialized and self.bpm_std > 1e-10:
                denormalized_orbit = current_orbit * self.bpm_std + self.bpm_mean
            else:
                denormalized_orbit = current_orbit * 1e-3  # 使用默认因子反归一化
                
            if np.all(np.abs(denormalized_orbit) < HYPERPARAMS['target_threshold']):  # 如果所有BPM读数都小于阈值
                reward += HYPERPARAMS['reward_weights']['completion_bonus']  # 额外奖励
                done = True
        
        # 保存当前轨道用于下一次计算（保存归一化数据）
        self.prev_prev_orbit = self.prev_orbit
        self.prev_orbit = normalized_orbit
        
        # 信息字典
        info = {
            'orbit_error': np.mean(np.abs(current_orbit)),
            'hcor_weights': self.hcor_weights.copy(),
            'vcor_weights': self.vcor_weights.copy()
        }
        
        if done:
            info['completed'] = True
            info['steps'] = self.step_count
        else:
            info['completed'] = False
            
        return state.astype(np.float32), reward, done, info
    
    def _update_bpm_stats(self, bpm_readings):
        """
        更新BPM读数的统计信息
        参数:
            bpm_readings: np.ndarray, BPM读数
        """
        # 如果是第一次，直接初始化
        if not self.bpm_stats_initialized:
            self.bpm_mean = np.mean(bpm_readings)
            self.bpm_std = np.std(bpm_readings)
            self.bpm_max = np.max(bpm_readings)
            self.bpm_min = np.min(bpm_readings)
            self.bpm_stats_initialized = True
            self.bpm_sample_count = len(bpm_readings)
        else:
            # 增量更新统计信息
            n = self.bpm_sample_count
            m = len(bpm_readings)
            new_mean = np.mean(bpm_readings)
            new_std = np.std(bpm_readings)
            
            # 更新总体均值
            combined_mean = (self.bpm_mean * n + new_mean * m) / (n + m)
            
            # 更新总体标准差（使用合并方差公式）
            combined_variance = ((n - 1) * (self.bpm_std ** 2) + (m - 1) * (new_std ** 2) + 
                               n * (self.bpm_mean - combined_mean) ** 2 + 
                               m * (new_mean - combined_mean) ** 2) / (n + m - 1)
            
            self.bpm_mean = combined_mean
            self.bpm_std = np.sqrt(combined_variance)
            self.bpm_max = max(self.bpm_max, np.max(bpm_readings))
            self.bpm_min = min(self.bpm_min, np.min(bpm_readings))
            self.bpm_sample_count += m
    
    def _normalize_bpm_readings(self, bpm_readings):
        """
        基于实际统计信息对BPM读数进行归一化处理
        参数:
            bpm_readings: np.ndarray, BPM读数
        返回:
            np.ndarray: 归一化后的BPM读数
        """
        if not self.bpm_stats_initialized:
            # 如果还没有统计信息，使用默认归一化
            return np.clip(bpm_readings / 1e-3, -1.0, 1.0)
        
        # 使用均值和标准差进行标准化
        if self.bpm_std > 1e-10:  # 避免除零错误
            normalized_readings = (bpm_readings - self.bpm_mean) / self.bpm_std
        else:
            normalized_readings = bpm_readings - self.bpm_mean
            
        # 将数据限制在合理范围内
        normalized_readings = np.clip(normalized_readings, -5.0, 5.0)
        return normalized_readings
    
    def _compute_reward(self, current_orbit, action):
        # 反归一化轨道读数以计算真实的奖励
        if self.bpm_stats_initialized and self.bpm_std > 1e-10:
            denormalized_orbit = current_orbit * self.bpm_std + self.bpm_mean
        else:
            denormalized_orbit = current_orbit * 1e-3  # 使用默认因子反归一化
        
        # 计算当前轨道误差（分别计算x和y方向）
        current_orbit_x = denormalized_orbit[:self.num_bpms]
        current_orbit_y = denormalized_orbit[self.num_bpms:]
        target_orbit_x = self.target_orbit[:self.num_bpms]
        target_orbit_y = self.target_orbit[self.num_bpms:]
        
        # 计算x和y方向的当前误差
        current_error_x = np.abs(current_orbit_x - target_orbit_x)
        current_error_y = np.abs(current_orbit_y - target_orbit_y)
        avg_orbit_error = np.mean(np.abs(denormalized_orbit))
        
        # 第一部分：鼓励BPM数值与目标轨道越接近越好
        # 分别计算水平和垂直方向的接近度奖励
        proximity_reward_x = -np.sum(current_error_x) * HYPERPARAMS['reward_weights']['orbit_improvement']
        proximity_reward_y = -np.sum(current_error_y) * HYPERPARAMS['reward_weights']['orbit_improvement']
        proximity_reward = proximity_reward_x + proximity_reward_y
        
        # 第二部分：根据BPM与目标轨道的差距调整动作奖励
        # 当BPM与目标轨道差别越大时，鼓励使用更大的动作；当BPM与目标轨道越接近时，鼓励使用更小的动作
        # 计算当前误差的总和，用它来调节动作惩罚的权重
        total_current_error = np.sum(current_error_x) + np.sum(current_error_y)
        
        # 动作惩罚权重随着误差增大而减小（鼓励大动作），随着误差减小而增大（鼓励小动作）
        # 我们希望当误差大时，对动作的惩罚较小（允许大动作），当误差小时，对动作的惩罚较大（鼓励精细调整）
        action_penalty_weight = HYPERPARAMS['reward_weights']['action_penalty'] / (1.0 + total_current_error * 1000)
        
        # 对动作幅度采用平方惩罚
        action_magnitude = np.mean(np.abs(action))
        action_penalty = -np.power(action_magnitude, 2) * action_penalty_weight
        
        # 总奖励由两部分组成
        total_reward = proximity_reward + action_penalty
        
        return total_reward

    def render(self, mode='human'):
        pass  # 可视化可以在这里实现
    
    def _update_corrector_weights(self, current_orbit):
        """
        基于当前轨道误差分布更新校正器动态权重
        参数:
            current_orbit: np.ndarray, 当前轨道读数(归一化)
        """
        # 如果统计信息未初始化，使用当前轨道读数进行初始化
        if self.bpm_stats_initialized and self.bpm_std > 1e-10:
            denormalized_orbit = current_orbit * self.bpm_std + self.bpm_mean
        else:
            denormalized_orbit = current_orbit * 1e-3  # 使用默认因子反归一化
            
        # 分离水平和垂直轨道读数
        orbit_x = denormalized_orbit[:self.num_bpms]  # 水平轨道读数
        orbit_y = denormalized_orbit[self.num_bpms:]  # 垂直轨道读数
        
        # 计算水平和垂直方向的误差绝对值
        error_x = np.abs(orbit_x)
        error_y = np.abs(orbit_y)
        
        # 动态映射误差到校正器维度（支持不同数量的BPM和COR）
        def map_error_to_correctors(error, num_cors):
            if len(error) == num_cors:
                return error
            elif num_cors == 0:
                return np.array([])
            else:
                # 使用线性插值扩展或压缩误差数组至目标长度
                return np.interp(
                    np.linspace(0, len(error), num_cors),
                    np.arange(len(error)),
                    error
                )
        
        mapped_error_x = map_error_to_correctors(error_x, self.num_hcors)
        mapped_error_y = map_error_to_correctors(error_y, self.num_vcors)
        
        # 基于误差分布计算权重：误差越大，权重越高
        # 使用指数放大差异，并归一化保证总和一致
        if self.num_hcors > 0:
            exp_x = np.exp(mapped_error_x * 10)
            self.hcor_weights = (exp_x / np.sum(exp_x)) * self.num_hcors
        else:
            self.hcor_weights = np.array([])

        if self.num_vcors > 0:
            exp_y = np.exp(mapped_error_y * 10)
            self.vcor_weights = (exp_y / np.sum(exp_y)) * self.num_vcors
        else:
            self.vcor_weights = np.array([])
    
    def update_reward_weights(self, episode_num=None, avg_error=None, success_rate=None):
        """
        动态更新奖励权重
        
        参数:
            episode_num: 当前训练轮次
            avg_error: 平均轨道误差
            success_rate: 成功率
        """
        if episode_num is not None:
            # 基于训练轮次调整权重
            # 随着训练进行，增加完成奖励的权重，降低动作惩罚权重以允许更精细的调整
            progress_ratio = min(episode_num / (HYPERPARAMS['max_episodes'] * 0.8), 1.0)
            self.reward_weights['completion_bonus'] = HYPERPARAMS['reward_weights']['completion_bonus'] * (1.0 + progress_ratio)
            self.reward_weights['action_penalty'] = HYPERPARAMS['reward_weights']['action_penalty'] * (1.0 - 0.5 * progress_ratio)
        
        if avg_error is not None:
            # 基于当前平均误差调整权重
            # 当误差较小时，增加动作惩罚权重以鼓励精细调整
            error_ratio = min(avg_error / 1e-3, 1.0)  # 假设1e-3为较大误差
            self.reward_weights['action_penalty'] = HYPERPARAMS['reward_weights']['action_penalty'] * (1.0 + (1.0 - error_ratio))
            
        # 确保轨道改善权重不低于最小值
        self.reward_weights['orbit_improvement'] = max(HYPERPARAMS['reward_weights']['orbit_improvement'] * 0.1, 
                                                      HYPERPARAMS['reward_weights']['orbit_improvement'])
    
    def get_reward_weights(self):
        """
        获取当前奖励权重
        返回:
            dict: 当前奖励权重
        """
        return self.reward_weights.copy()
        
    def get_corrector_weights(self):
        """
        获取当前校正器权重
        返回:
            tuple: (hcor_weights, vcor_weights)
        """
        return self.hcor_weights.copy(), self.vcor_weights.copy()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_range):
        super(Actor, self).__init__()
        self.action_range = action_range
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        f1 = 1 / np.sqrt(self.l1.weight.data.size()[0])
        self.l1.weight.data.uniform_(-f1, f1)
        self.l1.bias.data.uniform_(-f1, f1)
        
        f2 = 1 / np.sqrt(self.l2.weight.data.size()[0])
        self.l2.weight.data.uniform_(-f2, f2)
        self.l2.bias.data.uniform_(-f2, f2)
        
        f3 = 3e-3
        self.l3.weight.data.uniform_(-f3, f3)
        self.l3.bias.data.uniform_(-f3, f3)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        action = torch.tanh(a)  # 输出限制在[-1,1]
        if self.action_range is not None:
            action = action * self.action_range['max']
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        
        # Q1架构
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2架构
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        f1 = 1 / np.sqrt(self.l1.weight.data.size()[0])
        self.l1.weight.data.uniform_(-f1, f1)
        self.l1.bias.data.uniform_(-f1, f1)
        
        f2 = 1 / np.sqrt(self.l2.weight.data.size()[0])
        self.l2.weight.data.uniform_(-f2, f2)
        self.l2.bias.data.uniform_(-f2, f2)
        
        f3 = 3e-3
        self.l3.weight.data.uniform_(-f3, f3)
        self.l3.bias.data.uniform_(-f3, f3)
        
        f4 = 1 / np.sqrt(self.l4.weight.data.size()[0])
        self.l4.weight.data.uniform_(-f4, f4)
        self.l4.bias.data.uniform_(-f4, f4)
        
        f5 = 1 / np.sqrt(self.l5.weight.data.size()[0])
        self.l5.weight.data.uniform_(-f5, f5)
        self.l5.bias.data.uniform_(-f5, f5)
        
        f6 = 3e-3
        self.l6.weight.data.uniform_(-f6, f6)
        self.l6.bias.data.uniform_(-f6, f6)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
        
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None

        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        samples = [self.buffer[idx] for idx in indices]
        
        # 解包样本
        state, action, reward, next_state, done = map(np.array, zip(*samples))
        
        return state, action, reward, next_state, done

    def update_priorities(self, batch_indices, td_errors):
        if td_errors is not None:
            priorities = (td_errors + 1e-6)  # 添加小的常数以避免零优先级
            for idx, priority in zip(batch_indices, priorities):
                if idx < len(self.priorities):
                    self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.state_dim = hyperparams['state_dim']
        self.action_dim = hyperparams['action_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.actor_lr = hyperparams['actor_lr']
        self.critic_lr = hyperparams['critic_lr']
        self.gamma = hyperparams['gamma']
        self.tau = hyperparams['tau']
        self.policy_noise = hyperparams['policy_noise']
        self.noise_clip = hyperparams['noise_clip']
        self.policy_freq = hyperparams['policy_freq']
        self.batch_size = hyperparams['batch_size']
        self.action_range = hyperparams['action_range']
        
        # 使用全局设备设置
        self.device = device
        print(f"TD3Agent using device: {self.device}")
        
        # 初始化网络
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_range).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_range).to(self.device)
        self.critic_1 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_2 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_1_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_2_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        
        # 初始化目标网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
        
        # 经验回放缓冲区（使用优先经验回放）
        self.replay_buffer = PrioritizedReplayBuffer(hyperparams['buffer_size'])
        
        self.train_step = 0
        
    def select_action(self, state, evaluate=False):
        # 将状态转换为张量并移动到设备
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # print(f"Select action on device: {state_tensor.device}") if torch.cuda.is_available() else None
        
        # 在评估模式下，直接使用主策略网络
        if evaluate:
            with torch.no_grad():
                action = self.actor(state_tensor).cpu().numpy().flatten()
        else:
            # 训练模式下，使用参数空间噪声
            # 创建主网络的深拷贝
            perturbed_actor = copy.deepcopy(self.actor)
            perturbed_actor = perturbed_actor.to(self.device)
            
            # 为主网络的所有参数添加噪声
            with torch.no_grad():
                for param in perturbed_actor.parameters():
                    param_noise = torch.randn_like(param) * self.policy_noise
                    param.copy_(param + param_noise)  # 使用copy_而不是add_避免in-place操作问题
            
            # 使用扰动后的网络选择动作
            action = perturbed_actor(state_tensor).cpu().data.numpy().flatten()
        
        return action
            
    def update(self):
        # 增加训练步数计数器
        self.train_step += 1
        
        # 从经验回放中采样
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 采样经验
        samples = self.replay_buffer.sample(self.batch_size)
        if samples is None:
            return
            
        # 解包样本
        state, action, reward, next_state, done = samples
        
        # 将数据转换为张量并移动到指定设备
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # 添加目标策略噪声
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-1, 1)  # 假设动作已经被归一化到[-1,1]
            
            # 计算目标Q值
            target_Q1, target_Q2 = self.critic_1_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
            
        # 计算当前Q值
        current_Q1, current_Q2 = self.critic_1(state, action)
        
        # 计算critic损失（简化版本，不使用优先级经验回放的权重）
        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            
        # 优化critic网络
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()
        
        # 延迟策略更新
        if self.train_step % self.policy_freq == 0:
            # 计算actor损失
            actor_action = self.actor(state)
            actor_loss = -self.critic_1.Q1(state, actor_action).mean()  # 使用Q1方法计算actor损失
            
            # 优化actor网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_td3_agent(env, agent, hyperparams=None):
    """
    训练TD3智能体
    
    Args:
        env: 轨道修正环境
        agent: TD3智能体
        hyperparams: 超参数字典
    """
    # 使用默认超参数或传入的超参数
    if hyperparams is None:
        hyperparams = HYPERPARAMS
    
    # 获取超参数
    max_episodes = hyperparams['max_episodes']
    max_steps_per_episode = hyperparams['max_steps_per_episode']
    warmup_steps = hyperparams['warmup_steps']
    target_threshold = hyperparams['target_threshold']
    verbose = hyperparams.get('verbose', True)  # 控制是否输出详细信息
    
    # 检查CUDA是否可用
    device = agent.device if hasattr(agent, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on device: {device}")
    
    # 记录奖励和评估指标
    episode_rewards = []
    completion_rates = []
    gpu_memory_info = []
    
    # 早停机制相关变量
    early_stopping_patience = hyperparams.get('early_stopping_patience', 100)  # 早停耐心度
    early_stopping_threshold = hyperparams.get('early_stopping_threshold', -1000.0)  # 奖励阈值
    best_reward = float('-inf')
    patience_counter = 0
    best_model_state = None
    
    if verbose:
        print("开始训练TD3智能体...")
        print("=" * 80)
        print(f"{'Episode':<8} {'Steps':<8} {'Reward':<12} {'Final Error':<12} {'Completed':<10} {'GPU Mem':<10}")
        print("-" * 80)
    
    # 记录开始训练的时间
    start_time = time.time()
    
    for episode in range(max_episodes):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        done = False
        
        # 记录每回合步数
        steps = 0
        
        for step in range(max_steps_per_episode):
            steps += 1
            
            # 在预热阶段使用随机动作
            if episode * max_steps_per_episode + step < warmup_steps:
                action = env.action_space.sample()
            else:
                # 选择动作（添加噪声）
                action = agent.select_action(state, evaluate=False)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新智能体
            if episode * max_steps_per_episode + step >= warmup_steps:
                agent.update()
            
            # 累积奖励
            episode_reward += reward
            
            # 更新状态
            state = next_state
            
            # 如果回合结束则跳出
            if done:
                break
        
        # 记录回合奖励
        episode_rewards.append(episode_reward)
        
        # 获取回合信息
        final_error = info.get('final_error', np.inf)
        completed = info.get('completed', False)
        completion_rates.append(completed)
        
        # 获取GPU内存信息
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
        else:
            gpu_mem = "N/A"
        gpu_memory_info.append(gpu_mem)
        
        # 早停机制检查
        # 1. 基于奖励的早停
        if episode_reward > best_reward:
            best_reward = episode_reward
            patience_counter = 0
            # 保存最佳模型状态
            best_model_state = {
                'actor': copy.deepcopy(agent.actor.state_dict()),
                'critic_1': copy.deepcopy(agent.critic_1.state_dict()),
                'critic_2': copy.deepcopy(agent.critic_2.state_dict())
            }
        else:
            patience_counter += 1
            
        # 2. 如果奖励持续低于阈值则早停
        if len(episode_rewards) >= 20:  # 至少20回合后才考虑早停
            recent_avg_reward = np.mean(episode_rewards[-20:])
            if recent_avg_reward < early_stopping_threshold:
                if verbose:
                    print(f"最近20回合平均奖励({recent_avg_reward:.2f})持续低于阈值({early_stopping_threshold})，提前停止训练")
                # 恢复最佳模型状态
                if best_model_state:
                    agent.actor.load_state_dict(best_model_state['actor'])
                    agent.critic_1.load_state_dict(best_model_state['critic_1'])
                    agent.critic_2.load_state_dict(best_model_state['critic_2'])
                break
        
        # 3. 如果连续很多回合没有完成任务则早停
        if len(completion_rates) >= early_stopping_patience and sum(completion_rates[-early_stopping_patience:]) == 0:
            if verbose:
                print(f"连续{early_stopping_patience}回合未能完成任务，提前停止训练")
            # 恢复最佳模型状态
            if best_model_state:
                agent.actor.load_state_dict(best_model_state['actor'])
                agent.critic_1.load_state_dict(best_model_state['critic_1'])
                agent.critic_2.load_state_dict(best_model_state['critic_2'])
            break
            
        # 4. 如果耐心度用完则早停
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"奖励值在{early_stopping_patience}回合内没有改善，提前停止训练")
            # 恢复最佳模型状态
            if best_model_state:
                agent.actor.load_state_dict(best_model_state['actor'])
                agent.critic_1.load_state_dict(best_model_state['critic_1'])
                agent.critic_2.load_state_dict(best_model_state['critic_2'])
            break
        
        # 打印详细训练进度（每10个episode）
        if verbose and episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_completion = np.mean(completion_rates[-10:]) if completion_rates else 0
            
            print(f"{episode:<8} {steps:<8} {episode_reward:<12.2f} {final_error:<12.6f} "
                  f"{'Yes' if completed else 'No':<10} {gpu_mem:<10}")
    
    # 记录训练结束时间
    end_time = time.time()
    training_time = end_time - start_time
    
    if verbose:
        # 打印训练总结
        print("=" * 80)
        print("训练完成!")
        print(f"总训练时间: {training_time:.2f} 秒")
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"完成率: {np.mean(completion_rates)*100:.2f}%")
        if torch.cuda.is_available():
            print(f"峰值GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    return episode_rewards, completion_rates


def train_agent(env, agent, episodes=1000):
    """
    训练智能体
    参数:
        env: 轨道校正环境
        agent: TD3智能体
        episodes: 训练回合数
    """
    episode_rewards = []
    success_count = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        orbit_errors = []
        
        # 动态更新奖励权重
        env.update_reward_weights(episode_num=episode)
        
        for step in range(HYPERPARAMS['max_steps_per_episode']):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 训练智能体
            if len(agent.replay_buffer) > HYPERPARAMS['warmup_steps']:
                agent.train(HYPERPARAMS['batch_size'])
            
            episode_reward += reward
            orbit_errors.append(info['orbit_error'])
            
            state = next_state
            
            if done:
                if '完成' in str(info) or reward > 40:  # 粗略判断是否成功完成
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        
        # 每100回合打印一次统计信息
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_error = np.mean(orbit_errors)
            success_rate = success_count / max(1, episode + 1)
            
            # 基于性能更新奖励权重
            env.update_reward_weights(avg_error=avg_error, success_rate=success_rate)
            
            print(f"Episode {episode+1}: 平均奖励 = {avg_reward:.2f}, 平均轨道误差 = {avg_error:.2e}, "
                  f"成功率 = {success_rate:.2%}, 当前奖励权重 = {env.get_reward_weights()}")
    
    print(f"\n训练完成! 总共 {episodes} 回合, 成功 {success_count} 回合, 成功率 {success_count/episodes:.2%}")
    return episode_rewards


# 用于在Jupyter Notebook中使用的便捷函数
def create_env_and_agent():
    """
    创建环境和智能体实例
    """
    # 创建环境
    env = OrbitCorrectionEnv()
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = {
        'min': env.action_space.low[0],
        'max': env.action_space.high[0]
    }
    
    # 更新超参数 - 将动态计算出的维度添加到全局超参数字典的副本中
    hyperparams = HYPERPARAMS.copy()
    hyperparams['state_dim'] = state_dim
    hyperparams['action_dim'] = action_dim
    hyperparams['action_range'] = action_range
    
    # 创建智能体
    agent = TD3Agent(hyperparams)
    
    return env, agent
