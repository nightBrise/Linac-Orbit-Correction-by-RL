import gym
import numpy as np
from fodo_lattice import *
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
HYPERPARAMS = {
    'state_dim': 72,           # 状态维度 (24个BPM读数 + 24个当前与上次差异 + 24个上次与上上次差异)
    'action_dim': 24,          # 动作维度 (12个水平校正器 + 12个垂直校正器)
    'hidden_dim': 256,         # 神经网络隐藏层维度
    'actor_lr': 3e-4,          # Actor网络学习率
    'critic_lr': 3e-4,         # Critic网络学习率
    'gamma': 0.99,             # 折扣因子
    'tau': 0.005,              # 软更新参数
    'policy_noise': 0.2,       # 目标策略噪声标准差
    'noise_clip': 0.5,         # 噪声裁剪范围
    'policy_freq': 2,          # 策略更新频率（相对于批评家更新）
    'buffer_size': 1000000,    # 经验回放缓冲区大小
    'batch_size': 256,         # 批量大小
    'warmup_steps': 1000,      # 预热步数 (随机探索)
    'max_episodes': 1000,      # 最大训练回合数
    'max_steps_per_episode': 200,  # 每回合最大步数
    'reward_weights': {        # 奖励权重
        'orbit_improvement': 5.0,   # 轨道改善奖励权重，增加权重鼓励减小轨道偏差
        'action_penalty': 1.0,      # 动作惩罚权重，增加权重抑制过大动作
        'completion_bonus': 50.0   # 完成奖励，适度降低防止智能体为了奖励而过冲
    },
    'target_threshold': 1e-6,  # 目标阈值（当所有BPM读数都小于此值时任务完成）
    'action_range': {          # 动作范围
        'min': 0.001,          # 最小动作幅度
        'max': 0.01            # 最大动作幅度
    },
    'cor_angle_limit': 0.3     # 校正器角度限制（-0.3到0.3弧度）
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
        
        # 构建状态
        _, diff_to_prev, diff_to_prev_prev = calculate_orbit_diff(
            current_orbit, self.prev_orbit, self.prev_prev_orbit)
        
        # 更新历史
        self.prev_prev_orbit = self.prev_orbit
        self.prev_orbit = current_orbit
        
        # 构建72维状态向量
        state = np.concatenate([current_orbit, diff_to_prev, diff_to_prev_prev])
        return state.astype(np.float32)
    
    def step(self, action):
        # 验证动作范围
        validated_action = validate_action(action)
        
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
        
        # 构建状态
        _, diff_to_prev, diff_to_prev_prev = calculate_orbit_diff(
            current_orbit, self.prev_orbit, self.prev_prev_orbit)
        
        # 构建72维状态向量
        state = np.concatenate([current_orbit, diff_to_prev, diff_to_prev_prev])
        
        # 计算奖励
        reward = self._compute_reward(current_orbit, validated_action)
        
        # 检查是否完成
        self.step_count += 1
        done = False
        if self.step_count >= self.max_steps:
            done = True
        elif np.all(np.abs(current_orbit) < HYPERPARAMS['target_threshold']):  # 如果所有BPM读数都小于阈值
            reward += HYPERPARAMS['reward_weights']['completion_bonus']  # 额外奖励
            done = True
            
        # 信息（用于调试）
        info = {
            'orbit_error': np.mean(np.abs(current_orbit)),
            'max_error': np.max(np.abs(current_orbit)),
            'action_magnitude': np.mean(np.abs(validated_action))
        }
        
        # 更新历史
        self.prev_prev_orbit = self.prev_orbit
        self.prev_orbit = current_orbit
        
        return state.astype(np.float32), reward, done, info
    
    def _compute_reward(self, current_orbit, action):
        # 计算当前轨道误差（分别计算x和y方向）
        current_orbit_x = current_orbit[:12]
        current_orbit_y = current_orbit[12:24]
        target_orbit_x = self.target_orbit[:12]
        target_orbit_y = self.target_orbit[12:24]
        
        # 计算x和y方向的当前误差
        current_error_x = np.abs(current_orbit_x - target_orbit_x)
        current_error_y = np.abs(current_orbit_y - target_orbit_y)
        
        # 如果有历史轨道数据，计算改善程度
        if self.prev_orbit is not None:
            prev_orbit_x = self.prev_orbit[:12]
            prev_orbit_y = self.prev_orbit[12:24]
            prev_error_x = np.abs(prev_orbit_x - target_orbit_x)
            prev_error_y = np.abs(prev_orbit_y - target_orbit_y)
            
            # 计算x和y方向的改善程度
            error_improvement_x = prev_error_x - current_error_x
            error_improvement_y = prev_error_y - current_error_y
            
            # 分别计算x和y方向的轨道改善奖励
            orbit_reward_x = np.sum(error_improvement_x) * HYPERPARAMS['reward_weights']['orbit_improvement']
            orbit_reward_y = np.sum(error_improvement_y) * HYPERPARAMS['reward_weights']['orbit_improvement']
            orbit_reward = orbit_reward_x + orbit_reward_y
        else:
            orbit_reward = 0
        
        # 动作惩罚（鼓励更小的动作，且在BPM接近目标时更加严格）
        # 计算当前轨道误差的平均值
        avg_orbit_error = np.mean(np.abs(current_orbit))
        
        # 根据当前轨道误差调整动作惩罚权重
        # 当轨道误差较小时，增加对大动作的惩罚
        action_penalty_weight = HYPERPARAMS['reward_weights']['action_penalty'] * (1.0 + 10.0 / (1.0 + avg_orbit_error))
        
        # 动作幅度越大，惩罚越重，特别加强对大动作的惩罚
        action_magnitude = np.mean(np.abs(action))
        action_penalty = -action_magnitude * action_penalty_weight
        
        # 检查是否所有BPM读数都接近目标值（分别检查x和y方向）
        done_bonus = 0
        if np.all(np.abs(current_orbit_x) < HYPERPARAMS['target_threshold']) and \
           np.all(np.abs(current_orbit_y) < HYPERPARAMS['target_threshold']):
            done_bonus = HYPERPARAMS['reward_weights']['completion_bonus']
        
        return orbit_reward + action_penalty + done_bonus

    def render(self, mode='human'):
        pass  # 可视化可以在这里实现


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


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
        
        # 初始化网络
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic_1 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic_1_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic_2 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic_2_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        
        # 复制参数到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(hyperparams['buffer_size'], self.state_dim, self.action_dim)
        
        self.total_it = 0
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        
        if not evaluate:
            # 添加噪声用于探索
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            action = action + noise
            
        # 限制动作范围
        action = action.clamp(-self.action_range['max'], self.action_range['max'])
        return action.detach().cpu().numpy()[0]
            
    def update(self):
        self.total_it += 1
        
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 从经验回放缓冲区采样
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)
        
        with torch.no_grad():
            # 选择目标动作并添加噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.action_range['max'], self.action_range['max'])
            
            # 计算目标Q值
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
            
        # 更新Critic网络
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        critic_1_loss = nn.MSELoss()(current_Q1, target_Q)
        critic_2_loss = nn.MSELoss()(current_Q2, target_Q)
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # 延迟更新策略网络
        if self.total_it % self.policy_freq == 0:
            # 计算Actor损失
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            
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


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


def train_td3_agent(env, agent, hyperparams):
    """
    训练TD3智能体
    """
    num_episodes = hyperparams['max_episodes']
    max_steps = hyperparams['max_steps_per_episode']
    warmup_steps = hyperparams['warmup_steps']
    
    total_steps = 0
    episode_rewards = []
    
    # 存储训练历史数据用于最后显示
    training_history = {
        'episodes': [],
        'rewards': [],
        'eval_rewards': [],
        'steps': [],
        'orbit_errors': []
    }
    
    print(f"开始训练TD3智能体，总共 {num_episodes} 回合")
    print("=" * 80)
    print(f"{'Episode':<8} {'Train Reward':<15} {'Eval Reward':<15} {'Steps':<8} {'Orbit Error':<15} {'GPU Mem':<15}")
    print("-" * 80)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 在前几步进行随机探索
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新智能体
            if total_steps > warmup_steps:
                agent.update()
                
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        
        # 每10个episode评估一次
        if episode % 10 == 0:
            eval_reward = 0
            eval_state = env.reset()
            eval_steps = 0
            final_orbit_error = 0
            
            for _ in range(max_steps):
                eval_action = agent.select_action(eval_state, evaluate=True)
                eval_state, reward, done, eval_info = env.step(eval_action)
                eval_reward += reward
                eval_steps += 1
                if done:
                    final_orbit_error = eval_info['orbit_error']
                    break
                    
            # 记录评估数据
            training_history['episodes'].append(episode)
            training_history['rewards'].append(episode_reward)
            training_history['eval_rewards'].append(eval_reward)
            training_history['steps'].append(eval_steps)
            training_history['orbit_errors'].append(final_orbit_error)
            
            # 打印详细训练进度
            gpu_info = "N/A"
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                gpu_info = f"{gpu_mem_used:.2f}GB/{gpu_mem_reserved:.2f}GB"
            
            print(f"{episode:<8} {episode_reward:<15.2f} {eval_reward:<15.2f} {eval_steps:<8} {final_orbit_error:<15.2e} {gpu_info:<15}")
            
            # 检查显存使用情况，如果超过80%则提醒
            if torch.cuda.is_available():
                gpu_mem_pct = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                if gpu_mem_pct > 80:
                    print(f"警告: GPU显存使用率 {gpu_mem_pct:.1f}% 超过80%!")
    
    print("=" * 80)
    print("训练完成!")
    
    # 显示最终的统计信息
    if training_history['episodes']:
        final_eval_reward = training_history['eval_rewards'][-1]
        final_orbit_error = training_history['orbit_errors'][-1]
        avg_reward = np.mean(training_history['eval_rewards'][-10:]) if len(training_history['eval_rewards']) >= 10 else np.mean(training_history['eval_rewards'])
        
        print(f"\n最终评估奖励: {final_eval_reward:.2f}")
        print(f"最终轨道误差: {final_orbit_error:.2e}")
        print(f"最近10次评估平均奖励: {avg_reward:.2f}")
        
        if torch.cuda.is_available():
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_mem_pct = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            print(f"GPU显存使用: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB ({gpu_mem_pct:.1f}%)")
    
    return episode_rewards


# 用于在Jupyter Notebook中使用的便捷函数
def create_env_and_agent():
    """
    创建环境和智能体的便捷函数
    """
    env = OrbitCorrectionEnv()
    agent = TD3Agent(HYPERPARAMS)
    return env, agent