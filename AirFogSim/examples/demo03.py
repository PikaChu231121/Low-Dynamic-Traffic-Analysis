import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

import numpy as np
import random
import torch
import yaml
import json
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
    from runtime_update import init_runtime_updaters  # type: ignore
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
update_points = [[] for _ in range(3)]  # 记录每次update_with_feedback发生的时间点
formula_changes = [[] for _ in range(3)]  # 记录公式变化点
formulas_history = [[] for _ in range(3)]  # 保存公式历史

# 模拟执行
accumulated_reward = 0
last_values = {}  # 存储上一次的数据值
update_interval = 5.0  # 每隔多少个时间单位更新一次模型

last_x = [None] * 3
last_y = [None] * 3

# 保存初始公式
if updaters:
    for pattern_id, updater in enumerate(updaters):
        if updater:
            current_eqs = [{
                "equation": eq["equation"],
                "fitted_params": eq["fitted_params"],
                "complexity": len(eq["equation"]),  # 用公式长度作为复杂度简单估计
                "nmae": eq["last_mae"] if np.isfinite(eq["last_mae"]) else 0.0
            } for eq in updater.top_equations]
            formulas_history[pattern_id].append({
                "time": 0.0,
                "equations": current_eqs,
                "best_idx": updater.best_expression_index
            })

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

            for pattern_id in range(3):
                if pattern_id >= len(updaters) or updaters[pattern_id] is None:
                    continue

                # 记录更新前的表达式和最佳索引
                prev_best_index = updaters[pattern_id].best_expression_index
                prev_equations = [eq["equation"] for eq in updaters[pattern_id].top_equations]

                indep_names = indep_var_map[pattern_id]
                dep_name = dep_var_map[pattern_id]

                # 获取当前输入
                x_current = []
                valid_inputs = True
                for name in indep_names:
                    if name in data_collector.data and data_collector.data[name]:
                        value = data_collector.data[name][-1]
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            print(f"警告: 模式 {pattern_id} 的输入 '{name}' 在时间 {env.simulation_time:.2f} 为 None。跳过此更新。")
                            valid_inputs = False
                            break
                        x_current.append(value)
                    else:
                        print(f"警告: 模式 {pattern_id} 的输入 '{name}' 在时间 {env.simulation_time:.2f} 不可用或为空。跳过此更新。")
                        valid_inputs = False
                        break

                # 获取当前输出
                y_current = None
                if dep_name in data_collector.data and data_collector.data[dep_name]:
                    y_value = data_collector.data[dep_name][-1]
                    if y_value is not None:
                        y_current = y_value

                # 先用上一次的y做update_with_feedback（除了第一次）
                if last_y[pattern_id] is not None:
                    updater = updaters[pattern_id]
                    if updater:
                        updater.update_with_feedback(last_y[pattern_id])
                        update_points[pattern_id].append(env.simulation_time)
                        # print(f"模式{pattern_id+1} - update_with_feedback: {last_y[pattern_id]:.4f}")
                
                # 检查表达式是否发生变化
                curr_equations = [eq["equation"] for eq in updaters[pattern_id].top_equations]
                curr_best_index = updaters[pattern_id].best_expression_index
                
                if prev_equations != curr_equations or prev_best_index != curr_best_index:
                    print(f"模式{pattern_id+1} - 公式发生变化")
                    formula_changes[pattern_id].append(env.simulation_time)
                    
                    # 保存公式变化历史
                    current_eqs = [{
                        "equation": eq["equation"],
                        "fitted_params": eq["fitted_params"],
                        "complexity": len(eq["equation"]),  # 简单估计
                        "nmae": eq["last_mae"] if np.isfinite(eq["last_mae"]) else 0.0
                    } for eq in updaters[pattern_id].top_equations]
                    
                    formulas_history[pattern_id].append({
                        "time": env.simulation_time,
                        "equations": current_eqs,
                        "best_idx": curr_best_index
                    })

                # 再用当前x做预测
                pred = None
                error = float('nan')
                if valid_inputs:
                    updater = updaters[pattern_id]
                    if updater:
                        pred = updater.record_prediction(x_current)
                        if y_current is not None and pred is not None and not np.isnan(pred):
                            error = abs(pred - y_current)

                # 记录
                if pred is not None and not np.isnan(pred):
                    predictions[pattern_id].append(pred)
                    timestamps[pattern_id].append(env.simulation_time)
                    if y_current is not None:
                        actuals[pattern_id].append(y_current)
                        errors[pattern_id].append(error)
                        print(f"模式{pattern_id+1} - 预测: {pred:.4f}, 实际: {y_current:.4f}, 误差: {error:.4f}")
                    else:
                        print(f"模式{pattern_id+1} - 预测: {pred:.4f}, 实际: 未知")

                # 更新last_x, last_y
                last_x[pattern_id] = x_current if valid_inputs else None
                last_y[pattern_id] = y_current

    print(f"Simulation time: {env.simulation_time:.2f}, Reward: {accumulated_reward:.2f}", end='\r')
    env.render()

# 关闭环境
env.close()

# 保存最终的表达式
final_formulas = []
for pattern_id in range(3):
    if updaters and pattern_id < len(updaters) and updaters[pattern_id]:
        pattern_formulas = []
        for eq in updaters[pattern_id].top_equations:
            pattern_formulas.append({
                "equation": eq["equation"],
                "complexity": len(eq["equation"]),  # 简单估计复杂度
                "nmae": eq["last_mae"] if np.isfinite(eq["last_mae"]) else 0.0,
                "fitted_params": eq["fitted_params"]
            })
        final_formulas.append(pattern_formulas)
    else:
        final_formulas.append([])

# 保存最终公式到JSON文件
output_dir = os.path.join(airfogsim_root, 'output/runtime')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'final_formulas.json'), 'w') as f:
    json.dump(final_formulas, f, indent=2)
print(f"最终公式已保存到 {os.path.join(output_dir, 'final_formulas.json')}")

# 可视化预测结果（重设计为散点图）
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

for pattern_id in range(3):
    if len(predictions[pattern_id]) > 0:
        plt.figure(figsize=(12, 8))
        
        # 创建散点图
        plt.scatter(timestamps[pattern_id], predictions[pattern_id], 
                   c='red', marker='o', s=50, alpha=0.7, label='Prediction')
        
        # 实际值散点图
        if len(actuals[pattern_id]) > 0:
            plt.scatter(timestamps[pattern_id][:len(actuals[pattern_id])], 
                       actuals[pattern_id], c='blue', marker='x', s=50, 
                       alpha=0.7, label='Actual')
        
        # 添加误差条
        if len(errors[pattern_id]) > 0:
            for i, (t, p, a) in enumerate(zip(
                timestamps[pattern_id][:len(errors[pattern_id])], 
                predictions[pattern_id][:len(errors[pattern_id])], 
                actuals[pattern_id][:len(errors[pattern_id])]
            )):
                plt.plot([t, t], [p, a], 'k-', alpha=0.3, linewidth=1)
        
        # 添加公式变化点标注
        for idx, change_time in enumerate(formula_changes[pattern_id]):
            if change_time in timestamps[pattern_id]:
                time_idx = timestamps[pattern_id].index(change_time)
                if time_idx < len(predictions[pattern_id]):
                    pred_val = predictions[pattern_id][time_idx]
                    plt.scatter(change_time, pred_val, c='yellow', marker='*', s=200, 
                               edgecolors='black', zorder=5, label='Formula Change' if idx == 0 else "")
                    
                    # 添加公式变化注释
                    for pattern_history in formulas_history[pattern_id]:
                        if pattern_history["time"] == change_time:
                            best_idx = pattern_history["best_idx"]
                            if best_idx < len(pattern_history["equations"]):
                                best_eq = pattern_history["equations"][best_idx]["equation"]
                                # 截断过长的公式
                                if len(best_eq) > 30:
                                    best_eq = best_eq[:27] + "..."
                                plt.annotate(f"Formula: {best_eq}", 
                                           xy=(change_time, pred_val),
                                           xytext=(10, 10), textcoords='offset points',
                                           bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        # 标题和图例
        plt.title(f'Pattern {pattern_id+1} - {dep_var_map[pattern_id]} Prediction', fontsize=16)
        plt.xlabel('Simulation Time', fontsize=14)
        plt.ylabel(dep_var_map[pattern_id], fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f'pattern_{pattern_id+1}_prediction_scatter.png'))
        plt.close()

# 保存预测结果
results = {
    'timestamps': [t.tolist() if isinstance(t, np.ndarray) else t for t in timestamps],
    'predictions': [p.tolist() if isinstance(p, np.ndarray) else p for p in predictions],
    'actuals': [a.tolist() if isinstance(a, np.ndarray) else a for a in actuals],
    'errors': [e.tolist() if isinstance(e, np.ndarray) else e for e in errors],
    'dep_vars': dep_var_map,
    'update_points': update_points,
    'formula_changes': formula_changes,
    'formulas_history': formulas_history
}

import json
with open(os.path.join(output_dir, 'runtime_predictions.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nSimulation done.")
print(f"Prediction results saved to {os.path.join(output_dir, 'runtime_predictions.json')}")
print(f"Prediction charts saved to {output_dir} directory")