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

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
HYPERPARAMS = {
    'state_dim': 72,           # 状态维度 (24个BPM读数 + 24个当前与上次差异 + 24个上次与上上次差异)
    'action_dim': 24,          # 动作维度 (12个水平校正器 + 12个垂直校正器)
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
    'reward_weights': {        # 奖励权重，调节不同类型奖励在总奖励中的比重
        'orbit_improvement': 1.0,   # 轨道改善奖励权重，鼓励减小轨道偏差
        'action_penalty': 1.0,      # 动作惩罚权重，鼓励使用较小的动作调整，在轨道接近目标时会动态增加
        'completion_bonus': 50.0   # 完成奖励，当任务完成时给予的额外奖励
    },
    'target_threshold': 1e-5,  # 目标阈值（当所有BPM读数都小于此值时任务完成）
    'action_range': {          # 动作范围，限制每次动作调整的幅度范围
        'min': 0.0001,          # 最小动作幅度，防止动作过小无效果
        'max': 0.001            # 最大动作幅度，防止动作过大导致不稳定
    },
    'cor_angle_limit': 0.1     # 校正器角度限制，限制校正器角度的物理范围
}

class OrbitCorrectionEnv(gym.Env):
    def __init__(self, target_orbit=None, max_steps=200):
        super(OrbitCorrectionEnv, self).__init__()
        
        # 初始化晶格
        self.lattice = MagneticLattice(cell)
        self.tws = twiss(self.lattice, nPoints=1000)
        
        # 获取初始参数
        self.k1_init = get_k1_array(self.lattice)
        self.hcors_init = get_hcors_angle(self.lattice).astype(np.float64)
        self.vcors_init = get_vcors_angle(self.lattice).astype(np.float64)
        
        # 当前校正器角度
        self.current_hcor_angles = self.hcors_init.copy().astype(np.float64)
        self.current_vcor_angles = self.vcors_init.copy().astype(np.float64)
        
        # 历史BPM读数（用于状态构建）
        self.prev_orbit = None
        self.prev_prev_orbit = None
        
        # 初始粒子
        self.p_init = Particle(x=0, y=0, px=-1e-5, py=2e-5, E=1)
        
        # 目标轨道（理想情况下为0）
        if target_orbit is None:
            self.target_orbit = np.zeros(24)  # 12个水平+12个垂直BPM读数
        else:
            self.target_orbit = target_orbit
            
        # 动作空间：HCOR和VCOR校正器角度调整增量
        self.num_hcors = len([elem for elem in self.lattice.sequence if isinstance(elem, Hcor)])
        self.num_vcors = len([elem for elem in self.lattice.sequence if isinstance(elem, Vcor)])
        
        # 动作范围：每次调整角度的幅度限制
        action_low = np.array([-HYPERPARAMS['action_range']['max']] * (self.num_hcors + self.num_vcors))
        action_high = np.array([HYPERPARAMS['action_range']['max']] * (self.num_hcors + self.num_vcors))
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # 观察空间：当前轨道偏差 + 差异特征
        obs_low = np.array([-np.inf] * HYPERPARAMS['state_dim'])
        obs_high = np.array([np.inf] * HYPERPARAMS['state_dim'])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = max_steps
        
        # 初始化BPM读数统计信息
        self.bpm_mean = 0.0
        self.bpm_std = 1.0
        self.bpm_max = 1.0
        self.bpm_min = -1.0
        self.bpm_stats_initialized = False
        self.bpm_sample_count = 0
        
        # 初始化校正器动态权重
        self.hcor_weights = np.ones(self.num_hcors)  # 水平校正器权重
        self.vcor_weights = np.ones(self.num_vcors)  # 垂直校正器权重
        
        # 初始化奖励权重
        self.reward_weights = HYPERPARAMS['reward_weights'].copy()
        
    def reset(self):
        # 重置环境到初始状态
        self.current_hcor_angles = self.hcors_init.copy().astype(np.float64)
        self.current_vcor_angles = self.vcors_init.copy().astype(np.float64)
        self.prev_orbit = None
        self.prev_prev_orbit = None
        self.step_count = 0
        
        # 获取初始轨道
        orb_x, orb_y = get_ideal_orbit(self.lattice, self.k1_init, 
                                      self.current_hcor_angles, self.current_vcor_angles, 
                                      self.p_init)
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
        weighted_action[:self.num_hcors] = action[:self.num_hcors] * self.hcor_weights
        weighted_action[self.num_hcors:] = action[self.num_hcors:] * self.vcor_weights
        
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
        orb_x, orb_y = get_ideal_orbit(self.lattice, self.k1_init,
                                      self.current_hcor_angles, self.current_vcor_angles,
                                      self.p_init)
        
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
        if self.step_count >= self.max_steps:
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
            
        # 信息（用于调试）
        info = {
            'orbit_error': np.mean(np.abs(current_orbit * self.bpm_std + self.bpm_mean if self.bpm_stats_initialized else current_orbit)),
            'max_error': np.max(np.abs(current_orbit * self.bpm_std + self.bpm_mean if self.bpm_stats_initialized else current_orbit)),
            'action_magnitude': np.mean(np.abs(validated_action)),
            'hcor_weights_mean': np.mean(self.hcor_weights),
            'hcor_weights_std': np.std(self.hcor_weights),
            'vcor_weights_mean': np.mean(self.vcor_weights),
            'vcor_weights_std': np.std(self.vcor_weights)
        }
        
        # 更新历史
        self.prev_prev_orbit = self.prev_orbit
        self.prev_orbit = normalized_orbit
        
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
        current_orbit_x = denormalized_orbit[:12]
        current_orbit_y = denormalized_orbit[12:24]
        target_orbit_x = self.target_orbit[:12]
        target_orbit_y = self.target_orbit[12:24]
        
        # 计算x和y方向的当前误差
        current_error_x = np.abs(current_orbit_x - target_orbit_x)
        current_error_y = np.abs(current_orbit_y - target_orbit_y)
        avg_orbit_error = np.mean(np.abs(denormalized_orbit))
        
        # 如果有历史轨道数据，计算改善程度
        if self.prev_orbit is not None:
            # 反归一化历史轨道数据
            if self.bpm_stats_initialized and self.bpm_std > 1e-10:
                denormalized_prev_orbit = self.prev_orbit * self.bpm_std + self.bpm_mean
            else:
                denormalized_prev_orbit = self.prev_orbit * 1e-3  # 使用默认因子反归一化
                
            prev_orbit_x = denormalized_prev_orbit[:12]
            prev_orbit_y = denormalized_prev_orbit[12:24]
            prev_error_x = np.abs(prev_orbit_x - target_orbit_x)
            prev_error_y = np.abs(prev_orbit_y - target_orbit_y)
            
            # 计算x和y方向的改善程度
            error_improvement_x = prev_error_x - current_error_x
            error_improvement_y = prev_error_y - current_error_y
            
            # 分别计算x和y方向的轨道改善奖励
            orbit_reward_x = np.sum(error_improvement_x) * self.reward_weights['orbit_improvement']
            orbit_reward_y = np.sum(error_improvement_y) * self.reward_weights['orbit_improvement']
            orbit_reward = orbit_reward_x + orbit_reward_y
        else:
            orbit_reward = 0
        
        # 动作惩罚（鼓励更小的动作，且在BPM接近目标时更加严格）
        # 根据当前轨道误差调整动作惩罚权重
        # 当轨道误差较小时，增加对大动作的惩罚
        action_penalty_weight = self.reward_weights['action_penalty'] * (1.0 + 100.0 / (1.0 + avg_orbit_error))
        
        # 对动作幅度采用更强的非线性惩罚（平方惩罚）
        action_magnitude = np.mean(np.abs(action))
        action_penalty = -np.power(action_magnitude, 2) * action_penalty_weight
        
        # 轨道发散惩罚（如果当前误差比之前更大，添加额外惩罚）
        divergence_penalty = 0
        if self.prev_orbit is not None:
            prev_avg_error = np.mean(np.abs(denormalized_prev_orbit))
            if avg_orbit_error > prev_avg_error:
                divergence_penalty = -abs(avg_orbit_error - prev_avg_error) * self.reward_weights['orbit_improvement'] * 10
        
        # 连续改善奖励（如果连续几步轨道误差都在减小，给予额外奖励）
        stability_bonus = 0
        if not hasattr(self, 'error_history'):
            self.error_history = []
        self.error_history.append(avg_orbit_error)
        if len(self.error_history) >= 3:
            # 保持历史记录长度为10
            if len(self.error_history) > 10:
                self.error_history.pop(0)
            # 如果连续3步误差减小，给予奖励
            if (self.error_history[-1] < self.error_history[-2] < self.error_history[-3]):
                stability_bonus = self.reward_weights['orbit_improvement'] * 5
        
        # 检查是否所有BPM读数都接近目标值（分别检查x和y方向）
        done_bonus = 0
        if np.all(np.abs(current_orbit_x) < HYPERPARAMS['target_threshold']) and \
           np.all(np.abs(current_orbit_y) < HYPERPARAMS['target_threshold']):
            done_bonus = self.reward_weights['completion_bonus']
        
        # 保存当前轨道用于下一次计算（保存归一化数据）
        self.prev_orbit = current_orbit.copy()
        
        return orbit_reward + action_penalty + divergence_penalty + done_bonus + stability_bonus

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
        orbit_x = denormalized_orbit[:12]  # 水平轨道读数
        orbit_y = denormalized_orbit[12:]  # 垂直轨道读数
        
        # 计算水平和垂直方向的误差绝对值
        error_x = np.abs(orbit_x)
        error_y = np.abs(orbit_y)
        
        # 基于误差分布计算权重
        # 误差越大，对应的校正器权重越高
        # 使用softmax-like函数确保权重始终为正
        def compute_weights(errors, scale=2.0):
            # 将误差映射到权重，误差越大权重越高
            normalized_errors = errors / (np.max(errors) + 1e-8)  # 归一化到[0,1]
            weights = np.exp(scale * normalized_errors)
            return weights / np.sum(weights) * len(weights)  # 保持权重均值为1
        
        # 更新水平和垂直校正器权重
        self.hcor_weights = compute_weights(error_x)
        self.vcor_weights = compute_weights(error_y)
    
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
    
    # 检查CUDA是否可用
    device = agent.device if hasattr(agent, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 记录奖励和评估指标
    episode_rewards = []
    completion_rates = []
    gpu_memory_info = []
    
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
        
        # 打印详细训练进度（每10个episode）
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_completion = np.mean(completion_rates[-10:]) if completion_rates else 0
            
            print(f"{episode:<8} {steps:<8} {episode_reward:<12.2f} {final_error:<12.6f} "
                  f"{'Yes' if completed else 'No':<10} {gpu_mem:<10}")
        
        # 早停机制：如果连续100回合都没有完成任务，则停止训练
        if len(completion_rates) >= 100 and sum(completion_rates[-100:]) == 0:
            print(f"连续100回合未能完成任务，提前停止训练")
            break
    
    # 记录训练结束时间
    end_time = time.time()
    training_time = end_time - start_time
    
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
    
    # 更新超参数
    hyperparams = HYPERPARAMS.copy()
    hyperparams['state_dim'] = state_dim
    hyperparams['action_dim'] = action_dim
    hyperparams['action_range'] = action_range
    
    # 创建智能体
    agent = TD3Agent(hyperparams)
    
    return env, agent
