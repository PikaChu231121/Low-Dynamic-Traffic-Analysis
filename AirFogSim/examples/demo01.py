import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

import numpy as np
import random
import torch
import yaml
import matplotlib.pyplot as plt
from data_collector import DataCollector
from context_extractor import ContextExtractor

# 导入AirFogSim相关模块
from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# 设置随机种子
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# 加载配置文件
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# 修改配置，确保有足够的车辆和无人机
config['simulation']['vehicle_count'] = 60  # 至少50辆车
config['traffic']['max_n_UAVs'] = 50        # 至少10架无人机

# 创建环境
env = AirFogSimEnv(config, interactive_mode='graphic')  # 使用图形界面模式

# 创建上下文提取器并提取上下文信息
context_extractor = ContextExtractor(env, config_path)
context_extractor.extract_all_context()

# 创建算法模块
algorithm_module = BaseAlgorithmModule()
algorithm_module.initialize(env)

# 设置奖励模型
RewardScheduler.setModel(env, 'REWARD', '1/max(1e-3, task_delay)')

# 创建数据收集器
data_collector = DataCollector()

# 添加特定区域监控（例如交叉路口）
junctions = env.traffic_manager.getAllJunctionPositions()
if junctions:
    for i, junction in enumerate(junctions[:3]):  # 只监控前3个交叉路口
        # 在交叉路口周围创建监控区域
        area_bounds = [
            junction[0] - 200, junction[1] - 200,  # min_x, min_y
            junction[0] + 200, junction[1] + 200   # max_x, max_y
        ]
        data_collector.area_specific_data[f'junction_{i}'] = area_bounds

# 模拟执行
accumulated_reward = 0
# 设置交通密度采样间隔
density_sampling_interval = 10 # 每10个时间单位采样一次
next_density_sampling = 0  # 从头开始采样

while not env.isDone():
    # 记录任务信息
    prev_tasks = len(env.task_manager.getAllTasks())

    algorithm_module.scheduleStep(env)
    env.step()
    accumulated_reward += algorithm_module.getRewardByTask(env)

    # 检查任务变化
    curr_tasks = len(env.task_manager.getAllTasks())
    done_tasks = TaskScheduler.getDoneTaskNum(env)
    
    # 每10个时间步收集一次数据
    if int(env.simulation_time * 10) % 10 == 0:
        data_collector.collect(env, algorithm_module)
        print(f"Time {env.simulation_time}: Tasks - Previous: {prev_tasks}, Current: {curr_tasks}, Done: {done_tasks}")
    
    # 定期提取交通密度数据
    if env.simulation_time >= next_density_sampling:
        print(f"\n正在提取交通密度数据 (时间点: {env.simulation_time})...")
        context_extractor.extract_traffic_density_areas(env.simulation_time)
        next_density_sampling += density_sampling_interval
    
    print(f"Simulation time: {env.simulation_time:.2f}, Reward: {accumulated_reward:.2f}", end='\r')
    
    env.render()  # 渲染可视化

# 完成模拟后，可视化和导出数据
# 导出带有所有时间点密度数据的上下文信息
context_extractor.export_context_data('output/context_data.json')

# 导出精简版上下文数据，便于AI分析
context_extractor.export_context_data_summary('output/context_summary.json')

# 生成当前时间点的上下文可视化
context_extractor.visualize_context('output/context_visualization.png')

# 生成交通密度随时间变化的可视化
context_extractor.visualize_density_history('output/density_history.png')

# 保存数据并可视化
data_collector.save_to_csv('output/global_data.csv')
data_collector.plot_metrics('output/global_metrics.png')

# 关闭环境
env.close()

print("\nSimulation done.")
