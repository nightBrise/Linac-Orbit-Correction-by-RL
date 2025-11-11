# 轨道校正强化学习项目

本项目使用TD3（Twin Delayed Deep Deterministic Policy Gradient）强化学习算法来校正粒子加速器中的电子束轨道。通过强化学习智能体控制水平和垂直校正器的角度，使电子束轨道尽可能接近理想轨道。

## 项目结构

- `fodo_lattice.py`: 定义FOFO晶格结构，包括磁铁、漂移段、监测器和校正器等组件
- `td3_agent.py`: 实现TD3强化学习算法，包括环境、智能体和训练过程
- `main.ipynb`: Jupyter笔记本文件，用于运行训练和测试过程，包含可视化结果

## 工作原理

该项目通过强化学习算法控制粒子加速器中的束流轨道。环境中，智能体观察当前的轨道偏差状态，并决定如何调整水平和垂直校正器的角度来减小偏差。奖励函数设计鼓励减小轨道偏差，同时惩罚过大的校正动作。

## 依赖库及版本

- Python 3.10
- OCELOT 24.03.0
- PyTorch 1.12.1
- NumPy 1.21.5
- Gym 0.21.0

## 安装说明

1. 确保已安装Python 3.10
2. 安装OCELOT:
   ```
   conda install -c ocelot-collab ocelot=24.03.0
   ```
3. 安装其他依赖:
   ```
   pip install torch==1.12.1 numpy==1.21.5 gym==0.21.0
   ```

## 使用方法

1. 运行训练:
   打开`main.ipynb`并在Jupyter Notebook中执行所有单元格来训练智能体。

2. 查看结果:
   训练过程中的奖励、轨道误差等指标将在Notebook中显示。