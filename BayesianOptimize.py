import numpy as np
import optuna
import torch
import multiprocessing as mp
import time
# 设置多进程启动方法为spawn以解决CUDA初始化问题
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import os

# 确保在模块级别初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _objective_function_internal(trial_number, trial_params, n_calls=30):
    """
    内部目标函数：在独立进程中执行
    
    Args:
        trial_number: 试验编号
        trial_params: 试验参数字典
        n_calls: 总试验次数
        
    Returns:
        评估奖励
    """
    # 重新导入模块以确保在新进程中正确加载
    import torch
    from td3_agent import HYPERPARAMS, create_env_and_agent, train_td3_agent
    import copy
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用深拷贝创建独立的超参数副本
    hyperparams = copy.deepcopy(HYPERPARAMS)
    hyperparams['hidden_dim'] = trial_params['hidden_dim']
    hyperparams['reward_weights'] = hyperparams['reward_weights'].copy()
    hyperparams['reward_weights']['orbit_improvement'] = trial_params['orbit_improvement_weight']
    hyperparams['reward_weights']['action_penalty'] = trial_params['action_penalty_weight']
    # 移除了completion_bonus，因为新的奖励函数不包含完成奖励
    hyperparams['reward_weights']['divergence_penalty'] = trial_params['divergence_penalty_weight']
    hyperparams['reward_weights']['stability_penalty'] = trial_params['stability_penalty_weight']
    hyperparams['policy_noise'] = trial_params['policy_noise']
    hyperparams['noise_clip'] = trial_params['noise_clip']
    hyperparams['actor_lr'] = trial_params['actor_lr']
    hyperparams['critic_lr'] = trial_params['critic_lr']
    hyperparams['gamma'] = trial_params['gamma']
    hyperparams['tau'] = trial_params['tau']
    hyperparams['buffer_size'] = trial_params['buffer_size']
    hyperparams['batch_size'] = trial_params['batch_size']
    hyperparams['warmup_steps'] = trial_params['warmup_steps']
    hyperparams['target_threshold'] = trial_params['target_threshold']
    hyperparams['action_range'] = {
        'min': trial_params['action_range_min'],
        'max': trial_params['action_range_max']
    }
    hyperparams['cor_angle_limit'] = trial_params['cor_angle_limit']
    hyperparams['verbose'] = False  # 关闭详细输出以减少并行训练时的日志
    
    # 根据优化阶段调整训练回合数
    # 前30%试验使用快速评估
    if trial_number < n_calls * 0.3:
        hyperparams['max_episodes'] = 100  # 快速评估
    # 中间50%试验使用中等评估
    elif trial_number < n_calls * 0.8:
        hyperparams['max_episodes'] = 300  # 中等评估
    # 最后20%试验使用完整评估
    else:
        hyperparams['max_episodes'] = 500  # 完整评估
    
    try:
        # 创建环境和智能体
        env, agent = create_env_and_agent()
        
        # 训练智能体
        episode_rewards, completion_rates = train_td3_agent(env, agent, hyperparams)
        
        # 评估性能（使用最后几次评估的平均奖励）
        eval_rewards = []
        for i in range(len(episode_rewards)-5, len(episode_rewards)):
            if i >= 0:
                eval_rewards.append(episode_rewards[i])
        
        avg_reward = np.mean(eval_rewards) if eval_rewards else -1000
        
        return trial_number, avg_reward
    except Exception as e:
        return trial_number, float('-inf')

class BayesianOptimizer:
    def __init__(self, n_calls=30, random_state=42, fast_eval=False, n_jobs=1):
        """
        初始化贝叶斯优化器
        
        Args:
            n_calls: 优化迭代次数
            random_state: 随机种子
            fast_eval: 是否使用快速评估模式
            n_jobs: 并行任务数，默认为1（串行），>1时启用并行
        """
        self.n_calls = n_calls
        self.random_state = random_state
        self.fast_eval = fast_eval
        self.n_jobs = n_jobs
        
        # 创建Optuna研究对象
        self.study = optuna.create_study(
            direction="maximize",  # 我们要最大化奖励
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        
    def objective_function(self, trial):
        """
        目标函数：训练智能体并返回评估奖励
        
        Args:
            trial: Optuna试验对象
            
        Returns:
            评估奖励（越大越好）
        """
        print(f"Trial {trial.number} started on device: {device}")
        
        # 建议超参数
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        orbit_improvement_weight = trial.suggest_float('orbit_improvement_weight', 0.1, 100.0, log=True)
        action_penalty_weight = trial.suggest_float('action_penalty_weight', 0.1, 100.0, log=True)
        # completion_bonus = trial.suggest_float('completion_bonus', 10.0, 50000.0, log=True)
        divergence_penalty_weight = trial.suggest_float('divergence_penalty_weight', 0.1, 100.0, log=True)
        stability_penalty_weight = trial.suggest_float('stability_penalty_weight', 0.1, 1000.0, log=True)
        policy_noise = trial.suggest_float('policy_noise', 0.005, 0.99, log=True)
        noise_clip = trial.suggest_float('noise_clip', 0.1, 1.0, log=True)
        actor_lr = trial.suggest_float('actor_lr', 1e-6, 1e-2, log=True)
        critic_lr = trial.suggest_float('critic_lr', 1e-6, 1e-2, log=True)
        gamma = trial.suggest_float('gamma', 0.8, 2.0)
        tau = trial.suggest_float('tau', 0.0001, 0.1, log=True)
        buffer_size = trial.suggest_categorical('buffer_size', [5000, 10000, 100000, 250000, 500000])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        warmup_steps = trial.suggest_int('warmup_steps', 100, 10000)
        target_threshold = trial.suggest_float('target_threshold', 1e-6, 1e-4, log=True)
        action_range_min = trial.suggest_float('action_range_min', 1e-5, 1e-3, log=True)
        action_range_max = trial.suggest_float('action_range_max', 1e-3, 1e-2, log=True)
        cor_angle_limit = trial.suggest_float('cor_angle_limit', 0.05, 0.5, log=True)
        
        # 从td3_agent导入超参数（确保每次都是最新状态）
        from td3_agent import HYPERPARAMS
        
        # 使用深拷贝创建独立的超参数副本，防止多线程冲突
        hyperparams = copy.deepcopy(HYPERPARAMS)
        hyperparams['hidden_dim'] = hidden_dim
        hyperparams['reward_weights'] = hyperparams['reward_weights'].copy()
        hyperparams['reward_weights']['orbit_improvement'] = orbit_improvement_weight
        hyperparams['reward_weights']['action_penalty'] = action_penalty_weight
        # 移除了completion_bonus，因为新的奖励函数不包含完成奖励
        hyperparams['reward_weights']['divergence_penalty'] = divergence_penalty_weight
        hyperparams['reward_weights']['stability_penalty'] = stability_penalty_weight
        hyperparams['policy_noise'] = policy_noise
        hyperparams['noise_clip'] = noise_clip
        hyperparams['actor_lr'] = actor_lr
        hyperparams['critic_lr'] = critic_lr
        hyperparams['gamma'] = gamma
        hyperparams['tau'] = tau
        hyperparams['buffer_size'] = buffer_size
        hyperparams['batch_size'] = batch_size
        hyperparams['warmup_steps'] = warmup_steps
        hyperparams['target_threshold'] = target_threshold
        hyperparams['action_range'] = {
            'min': action_range_min,
            'max': action_range_max
        }
        hyperparams['cor_angle_limit'] = cor_angle_limit
        
        # 根据优化阶段调整训练回合数
        # 前30%试验使用快速评估
        if trial.number < self.n_calls * 0.3:
            hyperparams['max_episodes'] = 100  # 快速评估
        # 中间50%试验使用中等评估
        elif trial.number < self.n_calls * 0.8:
            hyperparams['max_episodes'] = 300  # 中等评估
        # 最后20%试验使用完整评估
        else:
            hyperparams['max_episodes'] = 500  # 完整评估
        
        print(f"试验 {trial.number + 1} 使用 {hyperparams['max_episodes']} 回合训练")
        
        # 动态导入，确保每个线程/进程都能正确加载环境和智能体
        from td3_agent import create_env_and_agent
        env, agent = create_env_and_agent()
        
        # 确保智能体在正确的设备上
        print(f"Agent device info - Actor: {next(agent.actor.parameters()).device}, "
              f"Critic1: {next(agent.critic_1.parameters()).device}")
        
        # 训练智能体（实现早期停止机制）
        episode_rewards = self._train_with_early_stopping(env, agent, hyperparams)
        
        # 评估性能（使用最后几次评估的平均奖励）
        eval_rewards = []
        for i in range(len(episode_rewards)-5, len(episode_rewards)):
            if i >= 0:
                eval_rewards.append(episode_rewards[i])
        
        avg_reward = np.mean(eval_rewards) if eval_rewards else -1000
        
        print(f"Trial {trial.number} completed with reward: {avg_reward:.2f}")
        
        return avg_reward
    
    def _train_with_early_stopping(self, env, agent, hyperparams):
        """
        带有早期停止机制的训练函数
        
        Args:
            env: 环境
            agent: 智能体
            hyperparams: 超参数
            
        Returns:
            训练奖励列表
        """
        from td3_agent import train_td3_agent
        # 复用td3_agent中已有的训练逻辑，确保一致性和正确性
        episode_rewards, completion_rates = train_td3_agent(env, agent, hyperparams)
        return episode_rewards
    
    def optimize(self):
        """
        执行贝叶斯优化
        
        Returns:
            best_params: 最佳参数
            best_reward: 最佳奖励
        """
        print("开始贝叶斯优化...")
        print("=" * 50)
        
        # 根据是否启用快速评估模式设置试验次数
        n_trials = self.n_calls
        if self.fast_eval:
            print("使用快速评估模式")
        else:
            print("使用标准评估模式")
            
        print(f"总共计划进行 {n_trials} 次试验")
        print(f"并行任务数: {self.n_jobs}")
        
        # 如果启用并行优化
        if self.n_jobs > 1:
            print(f"使用 {self.n_jobs} 个进程进行并行优化")
            return self._parallel_optimize(n_trials)
        else:
            # 运行优化
            self.study.optimize(self.objective_function, n_trials=n_trials)
        
        # 获取最佳参数和奖励
        best_params = self.study.best_params
        best_reward = self.study.best_value
        
        # 打印优化过程信息
        print("\n优化完成!")
        print(f"最佳奖励: {best_reward:.2f}")
        
        return best_params, best_reward
    
    def _parallel_optimize(self, n_trials):
        """
        并行执行贝叶斯优化
        
        Args:
            n_trials: 试验次数
            
        Returns:
            best_params: 最佳参数
            best_reward: 最佳奖励
        """
        print(f"开始提交 {n_trials} 个并行任务...")
        start_time = time.time()
        
        # 记录已完成的任务数
        completed_tasks = 0
        
        # 使用进程池执行并行优化
        with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=mp.get_context('spawn')) as executor:
            # 提交任务
            futures = []
            
            for i in range(n_trials):
                # 获取试验建议
                trial = self.study.ask()
                
                # 获取试验参数
                trial_params = {
                    'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
                    'orbit_improvement_weight': trial.suggest_float('orbit_improvement_weight', 0.1, 1000.0, log=True),
                    'action_penalty_weight': trial.suggest_float('action_penalty_weight', 0.1, 100.0, log=True),
                    'completion_bonus': trial.suggest_float('completion_bonus', 100.0, 10000.0, log=True),
                    'divergence_penalty_weight': trial.suggest_float('divergence_penalty_weight', 0.1, 1000.0, log=True),
                    'stability_penalty_weight': trial.suggest_float('stability_penalty_weight', 0.1, 500.0, log=True),
                    'policy_noise': trial.suggest_float('policy_noise', 0.05, 0.5, log=True),
                    'noise_clip': trial.suggest_float('noise_clip', 0.1, 1.0, log=True),
                    'actor_lr': trial.suggest_float('actor_lr', 1e-5, 1e-3, log=True),
                    'critic_lr': trial.suggest_float('critic_lr', 1e-5, 1e-3, log=True),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'tau': trial.suggest_float('tau', 0.001, 0.05, log=True),
                    'buffer_size': trial.suggest_categorical('buffer_size', [100000, 250000, 500000, 1000000]),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    'warmup_steps': trial.suggest_int('warmup_steps', 500, 5000),
                    'target_threshold': trial.suggest_float('target_threshold', 1e-6, 1e-4, log=True),
                    'action_range_min': trial.suggest_float('action_range_min', 1e-5, 1e-3, log=True),
                    'action_range_max': trial.suggest_float('action_range_max', 1e-3, 1e-2, log=True),
                    'cor_angle_limit': trial.suggest_float('cor_angle_limit', 0.05, 0.5, log=True)
                }
                
                # 提交任务
                future = executor.submit(_objective_function_internal, trial.number, trial_params, self.n_calls)
                futures.append((future, trial))
            
            # 等待所有任务完成
            results = {}
            for future, trial in futures:
                try:
                    trial_number, value = future.result()
                    results[trial_number] = value
                    self.study.tell(trial, value)
                    completed_tasks += 1
                    elapsed_time = time.time() - start_time
                    print(f"[{time.strftime('%H:%M:%S')}] 任务 {trial_number+1}/{n_trials} 完成，奖励: {value:.2f}，已用时: {elapsed_time:.1f}s")
                except Exception as e:
                    print(f"任务 {trial.number} 失败: {str(e)}")
                    self.study.tell(trial, float('-inf'))  # 失败试验给予负无穷奖励
                    results[trial.number] = float('-inf')
        
        total_time = time.time() - start_time
        print(f"\n所有 {n_trials} 个任务已完成 ({completed_tasks} 成功)，总用时: {total_time:.1f}s")
        
        # 获取最佳参数和奖励
        best_params = self.study.best_params
        best_reward = self.study.best_value
        
        # 打印优化过程信息
        print("\n优化完成!")
        print(f"最佳奖励: {best_reward:.2f}")
        
        return best_params, best_reward

def print_best_params(params):
    """
    打印最佳参数
    
    Args:
        params: 最佳参数字典
    """
    print("最佳超参数:")
    print("=" * 30)
    # 按照重要性排序打印参数
    if isinstance(params, dict):
        # 如果是字典格式（Optuna结果）
        print(f"hidden_dim: {params.get('hidden_dim', 'N/A')}")
        print(f"orbit_improvement_weight: {params.get('orbit_improvement_weight', 'N/A'):.6f}")
        print(f"action_penalty_weight: {params.get('action_penalty_weight', 'N/A'):.6f}")
        print(f"completion_bonus: {params.get('completion_bonus', 'N/A'):.6f}")
        print(f"divergence_penalty_weight: {params.get('divergence_penalty_weight', 'N/A'):.6f}")
        print(f"policy_noise: {params.get('policy_noise', 'N/A'):.6f}")
        print(f"noise_clip: {params.get('noise_clip', 'N/A'):.6f}")
        print(f"actor_lr: {params.get('actor_lr', 'N/A'):.6f}")
        print(f"critic_lr: {params.get('critic_lr', 'N/A'):.6f}")
        print(f"gamma: {params.get('gamma', 'N/A'):.6f}")
        print(f"tau: {params.get('tau', 'N/A'):.6f}")
        print(f"buffer_size: {params.get('buffer_size', 'N/A')}")
        print(f"batch_size: {params.get('batch_size', 'N/A')}")
        print(f"warmup_steps: {params.get('warmup_steps', 'N/A')}")
        print(f"target_threshold: {params.get('target_threshold', 'N/A'):.6e}")
        print(f"action_range_min: {params.get('action_range_min', 'N/A'):.6f}")
        print(f"action_range_max: {params.get('action_range_max', 'N/A'):.6f}")
        print(f"cor_angle_limit: {params.get('cor_angle_limit', 'N/A'):.6f}")
    else:
        # 如果是列表格式（旧版结果）
        param_names = [
            'orbit_improvement_weight', 'action_penalty_weight', 'completion_bonus',
            'policy_noise', 'noise_clip', 'actor_lr', 'critic_lr', 'gamma', 'tau'
        ]
        for name, value in zip(param_names, params):
            if name in ['target_threshold']:
                print(f"{name}: {value:.6e}")
            else:
                print(f"{name}: {value:.6f}")
