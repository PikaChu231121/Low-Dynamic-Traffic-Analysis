import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from airfogsim.scheduler import TaskScheduler
from airfogsim import AirFogSimEvaluation
import json

class DataCollector:
    def __init__(self):
        self.data = {
            'time': [],
            'task_success_ratio': [],
            'vehicle_count': [],
            'uav_count': [],
            'avg_V2U_rate': [],
            'avg_V2I_rate': [],
            'avg_U2I_rate': [],
            'compute_load_avg': [],
            'vehicle_density': [],
            'uav_density': []
        }
        
        # 可以添加特定区域的数据收集
        self.area_specific_data = {}
    
    def collect(self, env, algorithm_module):
        """收集当前时间点的数据"""
        # 更新评估指标
        evaluation_module = AirFogSimEvaluation()
        evaluation_module.updateEvaluationIndicators(env, algorithm_module)
        
        # 收集基本指标
        self.data['time'].append(env.simulation_time)
        
        # 任务成功率
        task_num = TaskScheduler.getDoneTaskNum(env)
        total_task_num = TaskScheduler.getTotalTaskNum(env)
        success_ratio = task_num / max(1, total_task_num)
        self.data['task_success_ratio'].append(success_ratio)
        
        # 使用实体数量
        n_vehicles = len(env.vehicles)
        n_UAVs = len(env.UAVs)
        self.data['vehicle_count'].append(n_vehicles)
        self.data['uav_count'].append(n_UAVs)
        
        # 通信指标
        try:
            self.data['avg_V2U_rate'].append(env.getChannelAvgRate(channel_type='V2U'))
            self.data['avg_V2I_rate'].append(env.getChannelAvgRate(channel_type='V2I'))
            self.data['avg_U2I_rate'].append(env.getChannelAvgRate(channel_type='U2I'))
        except:
            # 如果获取通信速率失败，添加默认值
            self.data['avg_V2U_rate'].append(0)
            self.data['avg_V2I_rate'].append(0)
            self.data['avg_U2I_rate'].append(0)
        
        # 分析任务分配情况（每10个时间步执行一次，避免输出过多）
        # if int(env.simulation_time * 10) % 100 == 0:
        #     self.analyze_task_allocation(env)
        
        # 计算负载指标
        compute_loads = self.calculate_compute_load(env)
        load_values = list(compute_loads.values())
        self.data['compute_load_avg'].append(np.mean(load_values) if load_values else 0)
        
        # 计算交通流密度
        x_range = env.traffic_manager.getConfig('x_range')
        y_range = env.traffic_manager.getConfig('y_range')
        if x_range is not None and y_range is not None:
            area_width = x_range[1] - x_range[0]
            area_height = y_range[1] - y_range[0]
            total_area = area_width * area_height
            # 防止面积为0的异常情况
            if total_area <= 0:
                total_area = 1.0
        else:
            total_area = 1.0

        current_vehicle_num = env.traffic_manager.getNumberOfVehicles()
        current_uav_num = env.traffic_manager.getNumberOfUAVs()
        
        # 将密度值转换为每平方公里的数量
        # 假设坐标系统单位是米，转换为每平方公里
        conversion_factor = 1000000  # 1平方公里 = 1,000,000平方米
        self.data['vehicle_density'].append((current_vehicle_num / total_area) * conversion_factor)
        self.data['uav_density'].append((current_uav_num / total_area) * conversion_factor)

        # 收集特定区域数据
        for area_name, area_bounds in self.area_specific_data.items():
            area_key = f'{area_name}_vehicle_count'
            if area_key not in self.data:
                self.data[area_key] = []
            
            count = self.count_entities_in_area(env, area_bounds, 'vehicle')
            self.data[area_key].append(count)

    def analyze_task_allocation(self, env):
        """分析任务分配情况"""
        all_tasks = env.task_manager.getAllTasks()
        
        # 统计任务状态
        task_states = {}
        allocated_tasks = 0
        unallocated_tasks = 0
        
        for task in all_tasks:
            if hasattr(task, 'getTaskState'):
                state = task.getTaskState()
                task_states[state] = task_states.get(state, 0) + 1
            
            if hasattr(task, 'getAllocatedNodeId'):
                node_id = task.getAllocatedNodeId()
                if node_id:
                    allocated_tasks += 1
                else:
                    unallocated_tasks += 1
        
        print(f"Task state statistics: {task_states}")
        print(f"Allocated tasks: {allocated_tasks}, Unallocated tasks: {unallocated_tasks}")
        
        # 检查任务调度器配置
        if hasattr(env, 'task_scheduler'):
            print(f"Task scheduler type: {type(env.task_scheduler)}")
    
    def calculate_compute_load(self, env):
        """计算各计算节点的负载情况"""
        compute_loads = {}
        total_tasks = 0
        
        # 打印任务信息
        all_tasks = env.task_manager.getAllTasks()
        # print(f"Total tasks in system: {len(all_tasks)}")
        
        # 只打印部分任务状态，避免输出过多
        task_states = {}
        sample_tasks = all_tasks[:5] if len(all_tasks) >= 5 else all_tasks  # 只显示前5个任务
        for task in sample_tasks:
            state = "unknown"
            if task.isComputed():
                state = "computed"
            elif task.isComputing():
                state = "computing"
            elif task.isTransmitting():
                state = "transmitting"
            task_states[task.getTaskId()] = state
        # print(f"Sample task states: {task_states}")
        
        # 获取所有计算中的任务
        computing_tasks = env.task_manager.getComputingTasks()
        
        # 遍历节点上正在计算的任务
        for node_id, tasks in computing_tasks.items():
            # 获取节点信息
            node_info = None
            if node_id in env.vehicles:
                node_info = env.vehicles[node_id]
            elif node_id in env.UAVs:
                node_info = env.UAVs[node_id]
            elif hasattr(env, 'RSUs') and node_id in env.RSUs:
                node_info = env.RSUs[node_id]
            
            if node_info is None:
                continue
                
            try:
                # 获取节点计算资源信息
                total_cpu = 0
                if hasattr(node_info, 'fog_profile') and node_info.fog_profile:
                    total_cpu = node_info.fog_profile.get('cpu', 0)
                else:
                    # 尝试直接获取fog_profile
                    node_attributes = node_info.to_dict() if hasattr(node_info, 'to_dict') else {}
                    fog_profile = node_attributes.get('fog_profile', {})
                    total_cpu = fog_profile.get('cpu', 0)
                
                if total_cpu <= 0:
                    continue
                
                # 计算已使用的CPU
                used_cpu = sum([task.getComputedSize() for task in tasks])
                total_tasks += len(tasks)
                
                # 计算负载比例
                load_ratio = used_cpu / total_cpu
                compute_loads[node_id] = load_ratio
                
            except Exception as e:
                print(f"Error processing node {node_id}: {str(e)}")
        
        # 打印统计信息
        # print(f"Total computing tasks found: {total_tasks}")
        # print(f"Nodes with compute load: {len(compute_loads)}")
        
        if len(compute_loads) > 0:
            avg_load = sum(compute_loads.values()) / len(compute_loads)
            # print(f"Average compute load: {avg_load:.4f}")
        
        return compute_loads
    
    def count_entities_in_area(self, env, area_bounds, entity_type='vehicle'):
        """计算特定区域内的实体数量
        area_bounds: [min_x, min_y, max_x, max_y]
        entity_type: 'vehicle' 或 'UAV'
        """
        count = 0
        
        try:
            if entity_type == 'vehicle':
                entities = env.vehicles
            else:
                entities = env.UAVs
            
            for entity_id, entity in entities.items():
                pos = entity.getPosition()
                if (area_bounds[0] <= pos[0] <= area_bounds[2] and 
                    area_bounds[1] <= pos[1] <= area_bounds[3]):
                    count += 1
        except Exception as e:
            print(f"Error counting entities in area: {e}")
            # 如果获取位置失败，返回0
            pass
        
        return count
    
    def save_to_csv(self, filename):
        """保存数据为CSV文件"""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
    
    def plot_metrics(self, ouput_path=None):
        """绘制指标变化图"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # 任务成功率
        axes[0, 0].plot(self.data['time'], self.data['task_success_ratio'])
        axes[0, 0].set_title('Task Success Ratio')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Success Ratio')
        
        # 实体数量
        axes[0, 1].plot(self.data['time'], self.data['vehicle_count'], label='Vehicles')
        axes[0, 1].plot(self.data['time'], self.data['uav_count'], label='UAVs')
        axes[0, 1].set_title('Entity Count')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        
        # 通信速率
        axes[1, 0].plot(self.data['time'], self.data['avg_V2U_rate'], label='V2U')
        axes[1, 0].plot(self.data['time'], self.data['avg_V2I_rate'], label='V2I')
        axes[1, 0].plot(self.data['time'], self.data['avg_U2I_rate'], label='U2I')
        axes[1, 0].set_title('Communication Rates')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Rate (Mbps)')
        axes[1, 0].legend()
        
        # 计算负载
        axes[1, 1].plot(self.data['time'], self.data['compute_load_avg'])
        axes[1, 1].set_title('Average Compute Load')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Load Ratio')

        # 绘制交通流密度
        axes[2, 0].plot(self.data['time'], self.data['vehicle_density'], label='Vehicle Density')
        axes[2, 0].plot(self.data['time'], self.data['uav_density'], label='UAV Density')
        axes[2, 0].set_title('Traffic Density')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Density')
        axes[2, 0].legend()
        
        # 检查是否有特定区域数据需要绘制
        area_metrics = [k for k in self.data.keys() if k.endswith('_vehicle_count') and 'junction' in k]
        if area_metrics and len(area_metrics) > 0:
            axes[2, 1].set_title('Area Specific Vehicle Count')
            axes[2, 1].set_xlabel('Time')
            axes[2, 1].set_ylabel('Count')
            
            for metric in area_metrics[:3]:  # 最多显示3个区域的数据
                axes[2, 1].plot(self.data['time'], self.data[metric], label=metric.replace('_vehicle_count', ''))
            
            axes[2, 1].legend()
        else:
            # 如果没有特定区域数据，则显示密度和计数的比较
            vehicle_density = np.array(self.data['vehicle_density'])
            vehicle_count = np.array(self.data['vehicle_count'])
            
            # 为了使两个指标在同一图中显示得更清楚，进行归一化处理
            max_density = max(vehicle_density) if len(vehicle_density) > 0 else 1
            max_count = max(vehicle_count) if len(vehicle_count) > 0 else 1
            
            norm_density = vehicle_density / max_density if max_density > 0 else vehicle_density
            norm_count = vehicle_count / max_count if max_count > 0 else vehicle_count
            
            ax1 = axes[2, 1]
            ax1.set_title('Vehicle Count vs Density')
            ax1.set_xlabel('Time')
            ax1.plot(self.data['time'], norm_count, 'b-', label='Vehicle Count (normalized)')
            ax1.set_ylabel('Vehicle Count (normalized)', color='b')
            ax1.tick_params('y', colors='b')
            
            ax2 = ax1.twinx()
            ax2.plot(self.data['time'], norm_density, 'r-', label='Vehicle Density (normalized)')
            ax2.set_ylabel('Vehicle Density (normalized)', color='r')
            ax2.tick_params('y', colors='r')
            
            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        if ouput_path:
            plt.savefig(ouput_path, dpi=300, bbox_inches='tight')
            print(f"全局指标可视化结果已保存到: {ouput_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def predict_and_compare_metrics(self, formula_json_paths, output_json_path=None):
        """
        批量测试多个实验结果，支持每个 pattern 使用不同自变量，误差为 NMAE，支持 min/max/movavg。
        formula_json_paths: list of json file paths (每个实验一个)
        output_json_path: 保存所有实验结果的 json 路径
        """
        import numpy as np
        import json
        import sys
        import os
        from sklearn.metrics import mean_absolute_error
        import matplotlib.pyplot as plt

        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录的路径（假设是 Low-Dynamic-Traffic-Analysis）
        project_root = os.path.abspath(os.path.join(current_dir, '../../'))
        # 将 Prediction 目录添加到 Python 路径
        sys.path.append(os.path.join(project_root, 'Prediction'))
        from helper import movavg  # type: ignore

        indep_var_map = [
            ['task_success_ratio', 'vehicle_density', 'uav_density', 'junction0_vehicle_count', 'junction1_vehicle_count', 'junction2_vehicle_count'],
            ['vehicle_density', 'uav_density', 'compute_load_avg'],
            ['avg_V2U_rate', 'avg_V2I_rate', 'compute_load_avg'],
        ]
        dep_var_map = [
            'compute_load_avg',
            'avg_V2U_rate',
            'task_success_ratio'
        ]

        for junc in [0, 1, 2]:
            key = f'junction{junc}_vehicle_count'
            if key not in self.data:
                self.data[key] = [0] * len(self.data['time'])

        all_runs_results = []

        # 可视化每个实验的预测与真实对比
        for run_idx, formula_json_path in enumerate(formula_json_paths):
            with open(formula_json_path, 'r') as f:
                formulas = json.load(f)
            if isinstance(formulas, dict) and 'CombResults' in formulas:
                CombResults = formulas['CombResults']
            else:
                CombResults = formulas

            run_result = []
            fig, axes = plt.subplots(1, 3, figsize=(18, 7))
            for pattern_idx in range(3):
                indep_names = indep_var_map[pattern_idx]
                dep_name = dep_var_map[pattern_idx]
                min_len = min([len(self.data[n]) for n in indep_names + [dep_name]])
                indep_vars = [np.array(self.data[n][:min_len]) for n in indep_names]
                dep_var = np.array(self.data[dep_name][1:min_len])  # 预测下一时隙
                time = np.array(self.data['time'][1:min_len])

                pattern_results = []
                if isinstance(CombResults, dict) and str(pattern_idx) in CombResults:
                    formulas_list = CombResults[str(pattern_idx)]
                elif isinstance(CombResults, list) and len(CombResults) > pattern_idx:
                    formulas_list = CombResults[pattern_idx]
                else:
                    formulas_list = []

                ax = axes[pattern_idx]
                ax.plot(time, dep_var, label='Actual', color='black', linewidth=2)
                for eq_idx, formula in enumerate(formulas_list):
                    eq = formula['equation']
                    params = formula['fitted_params']
                    preds = []
                    for i in range(min_len-1):
                        local_vars = {f'x{j+1}': indep_vars[j][i] for j in range(len(indep_vars))}
                        local_vars['c'] = params
                        local_vars['np'] = np
                        local_vars['min'] = np.minimum
                        local_vars['max'] = np.maximum
                        local_vars['movavg'] = lambda p, k: movavg(indep_vars[p-1][:i], k)
                        local_vars['log'] = np.log
                        local_vars['exp'] = np.exp
                        local_vars['sqrt'] = np.sqrt
                        local_vars['cbrt'] = np.cbrt
                        try:
                            pred = eval(eq, {}, local_vars)
                        except Exception:
                            pred = np.nan
                        preds.append(pred)
                    arr_pred = np.array(preds)
                    arr_actual = dep_var
                    mask = ~np.isnan(arr_pred)
                    if np.sum(mask) > 0:
                        # 使用 min-max 范围计算 NMAE
                        actual_min = np.min(arr_actual[mask])
                        actual_max = np.max(arr_actual[mask])
                        actual_range = actual_max - actual_min
                        if actual_range > 0:
                            mae = mean_absolute_error(arr_actual[mask], arr_pred[mask])
                            nmae = mae / actual_range
                        else:
                            nmae = float('inf')  # 如果范围为 0，则设置为无穷大
                    else:
                        nmae = float('inf')
                    # nmae为inf或nan时，写为None
                    nmae_json = None if (np.isnan(nmae) or np.isinf(nmae)) else nmae
                    pattern_results.append({
                        'equation': eq,
                        'fitted_params': params,
                        'nmae': nmae_json
                    })
                    # 绘制预测曲线
                    ax.plot(time, arr_pred, label=f'Pred {eq_idx+1} (NMAE={nmae:.3g})', alpha=0.7)
                ax.set_title(f'Pattern {pattern_idx+1}')
                ax.set_xlabel('Time')
                ax.set_ylabel(dep_name)
                ax.legend()
                ax.grid(True)
                run_result.append(pattern_results)
            # 调整布局，留出顶部空间以显示标题
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.suptitle(f'Run {run_idx+1} - Prediction vs Actual', y=1.00)
            plt.show()
            all_runs_results.append(run_result)

        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(all_runs_results, f, indent=2)
        return all_runs_results

