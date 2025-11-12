import numpy as np
import optuna
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

# 确保在模块级别初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"BayesianOptimize using device: {device}")

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
        
        # 建议超参数
        orbit_improvement_weight = trial.suggest_float('orbit_improvement_weight', 0.1, 100.0, log=True)
        action_penalty_weight = trial.suggest_float('action_penalty_weight', 0.1, 100.0, log=True)
        completion_bonus = trial.suggest_float('completion_bonus', 10.0, 100.0, log=True)
        policy_noise = trial.suggest_float('policy_noise', 0.05, 0.5, log=True)
        noise_clip = trial.suggest_float('noise_clip', 0.1, 1.0, log=True)
        actor_lr = trial.suggest_float('actor_lr', 1e-5, 1e-3, log=True)
        critic_lr = trial.suggest_float('critic_lr', 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float('gamma', 0.9, 0.999)
        tau = trial.suggest_float('tau', 0.001, 0.05, log=True)
        
        # 从td3_agent导入超参数（确保每次都是最新状态）
        from td3_agent import HYPERPARAMS
        
        # 使用深拷贝创建独立的超参数副本，防止多线程冲突
        hyperparams = copy.deepcopy(HYPERPARAMS)
        hyperparams['reward_weights'] = hyperparams['reward_weights'].copy()
        hyperparams['reward_weights']['orbit_improvement'] = orbit_improvement_weight
        hyperparams['reward_weights']['action_penalty'] = action_penalty_weight
        hyperparams['reward_weights']['completion_bonus'] = completion_bonus
        hyperparams['policy_noise'] = policy_noise
        hyperparams['noise_clip'] = noise_clip
        hyperparams['actor_lr'] = actor_lr
        hyperparams['critic_lr'] = critic_lr
        hyperparams['gamma'] = gamma
        hyperparams['tau'] = tau
        
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
        print(f"Using device: {device}")
        
        # 如果启用并行优化
        if self.n_jobs > 1:
            print(f"使用 {self.n_jobs} 个线程进行并行优化")
            self._parallel_optimize(n_trials)
        else:
            # 运行优化
            self.study.optimize(self.objective_function, n_trials=n_trials)
        
        # 获取最佳参数和奖励
        best_params = self.study.best_params
        best_reward = self.study.best_value
        
        # 打印优化过程信息
        print("\n优化完成!")
        print(f"最佳奖励: {best_reward:.2f}")
        print("\n优化历史:")
        for i, trial in enumerate(self.study.trials):
            if trial.value is not None:
                print(f"试验 {i+1}: 奖励 = {trial.value:.2f}")
            else:
                print(f"试验 {i+1}: 失败")
        
        return best_params, best_reward
    
    def _parallel_optimize(self, n_trials):
        """
        并行执行贝叶斯优化
        
        Args:
            n_trials: 试验次数
        """
        # 计算每个线程的任务数
        trials_per_thread = n_trials // self.n_jobs
        remaining_trials = n_trials % self.n_jobs
        
        def worker(thread_id, num_trials):
            """
            工作线程函数
            
            Args:
                thread_id: 线程ID
                num_trials: 该线程需要执行的试验数
                
            Returns:
                试验结果列表
            """
            results = []
            for i in range(num_trials):
                # 为每个线程设置不同的随机种子
                np.random.seed(self.random_state + thread_id * 1000)
                torch.manual_seed(self.random_state + thread_id * 1000)
                
                # 创建独立的trial对象
                trial = self.study.ask()
                
                try:
                    print(f"\n线程 {thread_id} 开始试验 {i+1}/{num_trials}")
                    
                    # 执行目标函数
                    value = self.objective_function(trial)
                    
                    # 报告试验结果
                    self.study.tell(trial, value)
                    results.append((trial.number, value))
                    
                    print(f"线程 {thread_id} 完成试验 {i+1}/{num_trials}，奖励: {value:.2f}")
                except Exception as e:
                    print(f"线程 {thread_id} 试验 {i+1} 失败: {str(e)}")
                    self.study.tell(trial, float('-inf'))  # 失败试验给予负无穷奖励
                    results.append((trial.number, float('-inf')))
            
            return results
        
        # 使用线程池执行并行优化
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交任务
            futures = []
            for i in range(self.n_jobs):
                # 计算每个线程的试验数
                num_trials = trials_per_thread + (1 if i < remaining_trials else 0)
                if num_trials > 0:
                    future = executor.submit(worker, i, num_trials)
                    futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    results = future.result()
                    print(f"线程完成，返回 {len(results)} 个结果")
                except Exception as e:
                    print(f"线程执行出错: {str(e)}")

def print_best_params(params):
    """
    打印最佳参数
    
    Args:
        params: 最佳参数字典
    """
    print("最佳超参数:")
    print("=" * 30)
    for name, value in params.items():
        print(f"{name}: {value:.6f}")