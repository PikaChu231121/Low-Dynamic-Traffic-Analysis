import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

import numpy as np
import random
import torch
import yaml
from data_collector import DataCollector

# 导入AirFogSim相关模块
from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler

# 获取项目根目录 (AirFogSim)，并将其设置为当前工作目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)

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

    print(f"Simulation time: {env.simulation_time:.2f}, Reward: {accumulated_reward:.2f}", end='\r')

    env.render()  # 渲染可视化

# 关闭环境
env.close()

print("\nSimulation done.")

# ==== 新增：批量测试多个实验结果 ====
output_json_path = os.path.join(os.path.dirname(__file__), '../output/runtime/nmae_results.json')
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
formula_json_paths = [
    os.path.join(os.path.dirname(__file__), '../output/runtime/final_formulas.json'),
    # os.path.join(os.path.dirname(__file__), '../../Prediction/results/train/run2.json'),
    # os.path.join(os.path.dirname(__file__), '../../Prediction/results/train/run3.json'),
]
data_collector.predict_and_compare_metrics(
    formula_json_paths=formula_json_paths,
    output_json_path=output_json_path
)
print(f"所有实验的NMAE结果已保存到: {output_json_path}")