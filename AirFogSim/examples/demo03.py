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

# 获取项目根目录 (Low-Dynamic-Traffic-Analysis)，并将其设置为当前工作目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
airfogsim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(airfogsim_root)

# 导入Runtime模块
runtime_path = os.path.join(project_root, 'Runtime')
if runtime_path not in sys.path:
    sys.path.append(runtime_path)

try:
    from runtime_update import init_runtime_updaters, update_runtime_model  # type: ignore
    runtime_available = True
except ImportError as e:
    print(f"警告: Runtime模块无法导入，运行时更新将被禁用 ({e})")
    runtime_available = False

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

# 初始化runtime updaters
updaters = None
if runtime_available:
    print("正在初始化运行时更新模块...")
    try:
        model_path = os.path.join(project_root, "Runtime/predict_model.json")
        data_path = os.path.join(airfogsim_root, "output/global_data.csv")
        updaters = init_runtime_updaters(None, data_path)
        print("运行时更新模块初始化成功")
    except Exception as e:
        print(f"运行时更新模块初始化失败: {e}")
        updaters = None

# 定义输入变量映射
indep_var_map = [
    ['task_success_ratio', 'vehicle_density', 'uav_density', 'junction_0_vehicle_count', 'junction_1_vehicle_count', 'junction_2_vehicle_count'],
    ['vehicle_density', 'uav_density', 'compute_load_avg'],
    ['avg_V2U_rate', 'avg_V2I_rate', 'compute_load_avg'],
]
dep_var_map = [
    'compute_load_avg',
    'avg_V2U_rate',
    'task_success_ratio'
]

# 准备存储每个模式的预测结果
predictions = [[] for _ in range(3)]
actuals = [[] for _ in range(3)]
errors = [[] for _ in range(3)]
timestamps = [[] for _ in range(3)]

# 模拟执行
accumulated_reward = 0
last_values = {}  # 存储上一次的数据值
update_interval = 1.0  # 每隔多少个时间单位更新一次模型

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
        
        # 运行时更新
        if updaters and int(env.simulation_time) % update_interval == 0:
            print(f"\n时间 {env.simulation_time}: 执行预测并更新运行时模型")
            
            # 确保有足够的数据
            for pattern_id in range(3):
                if len(data_collector.data['time']) < 2:
                    continue
                
                # 为模式准备输入和输出数据
                indep_names = indep_var_map[pattern_id]
                dep_name = dep_var_map[pattern_id]
                
                # 获取当前输入
                x_current = []
                valid_inputs = True
                for name in indep_names:
                    if name in data_collector.data and data_collector.data[name]: # 检查列表是否存在且不为空
                        value = data_collector.data[name][-1] # 获取最新的值
                        if value is None:
                            print(f"警告: 模式 {pattern_id} 的输入 '{name}' 在时间 {env.simulation_time:.2f} 为 None。跳过此更新。")
                            valid_inputs = False
                            break
                        x_current.append(value)
                    else:
                        print(f"警告: 模式 {pattern_id} 的输入 '{name}' 在时间 {env.simulation_time:.2f} 不可用或为空。跳过此更新。")
                        valid_inputs = False
                        break
                
                if not valid_inputs:
                    continue # 如果输入无效，则跳过此模式的当前更新
                
                # 如果还没有真实输出，则不更新
                y_current = None
                if dep_name in data_collector.data and data_collector.data[dep_name]: # 检查列表是否存在且不为空
                    y_value = data_collector.data[dep_name][-1]
                    if y_value is None:
                        print(f"警告: 模式 {pattern_id} 的实际输出 '{dep_name}' 在时间 {env.simulation_time:.2f} 为 None。")
                        # y_current 保持为 None
                    else:
                        y_current = y_value
                else:
                    print(f"警告: 模式 {pattern_id} 的实际输出 '{dep_name}' 在时间 {env.simulation_time:.2f} 不可用或为空。")
                    # y_current 保持为 None
                
                # 更新模型并记录预测和误差
                pred, error = update_runtime_model(updaters, pattern_id, x_current, y_current)
                
                if pred is not None:
                    predictions[pattern_id].append(pred)
                    timestamps[pattern_id].append(env.simulation_time)
                    
                    if y_current is not None:
                        actuals[pattern_id].append(y_current)
                        errors[pattern_id].append(error)
                        print(f"模式{pattern_id+1} - 预测: {pred:.4f}, 实际: {y_current:.4f}, 误差: {error:.4f}")
                    else:
                        print(f"模式{pattern_id+1} - 预测: {pred:.4f}, 实际: 未知")

    print(f"Simulation time: {env.simulation_time:.2f}, Reward: {accumulated_reward:.2f}", end='\r')

    env.render()  # 渲染可视化

# 关闭环境
env.close()

# 可视化预测结果
import matplotlib.pyplot as plt

for pattern_id in range(3):
    if len(predictions[pattern_id]) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps[pattern_id], predictions[pattern_id], 'r-', label='预测')
        
        if len(actuals[pattern_id]) > 0:
            plt.plot(timestamps[pattern_id][:len(actuals[pattern_id])], actuals[pattern_id], 'b-', label='实际')
            
            # 计算平均误差
            avg_error = np.mean(errors[pattern_id])
            plt.title(f'模式{pattern_id+1} - {dep_var_map[pattern_id]}预测 (平均误差: {avg_error:.4f})')
        else:
            plt.title(f'模式{pattern_id+1} - {dep_var_map[pattern_id]}预测')
            
        plt.xlabel('时间')
        plt.ylabel(dep_var_map[pattern_id])
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        output_dir = os.path.join(airfogsim_root, 'output/prediction')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'pattern_{pattern_id+1}_prediction.png'))
        plt.close()

# 保存预测结果
results = {
    'timestamps': [t.tolist() if isinstance(t, np.ndarray) else t for t in timestamps],
    'predictions': [p.tolist() if isinstance(p, np.ndarray) else p for p in predictions],
    'actuals': [a.tolist() if isinstance(a, np.ndarray) else a for a in actuals],
    'errors': [e.tolist() if isinstance(e, np.ndarray) else e for e in errors],
    'dep_vars': dep_var_map
}

import json
with open(os.path.join(airfogsim_root, 'output/runtime_predictions.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nSimulation done.")
print(f"预测结果已保存到 {os.path.join(airfogsim_root, 'output/runtime_predictions.json')}")
print(f"预测图表已保存到 {os.path.join(airfogsim_root, 'output')} 目录")