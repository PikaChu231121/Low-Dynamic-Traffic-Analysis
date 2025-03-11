import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from airfogsim.scheduler import TaskScheduler
from airfogsim import AirFogSimEvaluation

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
            'compute_load_avg': []
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
        if int(env.simulation_time * 10) % 100 == 0:
            self.analyze_task_allocation(env)
        
        # 计算负载指标
        compute_loads = self.calculate_compute_load(env)
        load_values = list(compute_loads.values())
        self.data['compute_load_avg'].append(np.mean(load_values) if load_values else 0)
        
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
        print(f"Total tasks in system: {len(all_tasks)}")
        
        # 只打印部分任务状态，避免输出过多
        task_states = {}
        for task in all_tasks[:5]:  # 只显示前5个任务
            if hasattr(task, 'getTaskState'):
                state = task.getTaskState()
                task_states[state] = task_states.get(state, 0) + 1
        print(f"Sample task states: {task_states}")
        
        # 直接获取节点信息的替代方法
        # 对于车辆
        for vehicle_id, vehicle in env.vehicles.items():
            try:
                # 获取车辆计算资源信息
                if hasattr(vehicle, 'fog_profile') and vehicle.fog_profile:
                    total_cpu = vehicle.fog_profile.get('cpu', 0)
                    used_cpu = 0
                    
                    # 获取当前正在该节点上执行的任务
                    computing_tasks = env.task_manager.getComputingTasksByNodeId(vehicle_id)
                    for task in computing_tasks:
                        used_cpu += task.getAllocatedCpu()
                        total_tasks += 1
                    
                    # 计算负载比例
                    if total_cpu > 0:
                        load_ratio = used_cpu / total_cpu
                        compute_loads[vehicle_id] = load_ratio
            except Exception as e:
                print(f"Error processing vehicle {vehicle_id}: {str(e)}")
        
        # 对于无人机
        for uav_id, uav in env.UAVs.items():
            try:
                # 获取无人机计算资源信息
                if hasattr(uav, 'fog_profile') and uav.fog_profile:
                    total_cpu = uav.fog_profile.get('cpu', 0)
                    used_cpu = 0
                    
                    # 获取当前正在该节点上执行的任务
                    computing_tasks = env.task_manager.getComputingTasksByNodeId(uav_id)
                    for task in computing_tasks:
                        used_cpu += task.getAllocatedCpu()
                        total_tasks += 1
                    
                    # 计算负载比例
                    if total_cpu > 0:
                        load_ratio = used_cpu / total_cpu
                        compute_loads[uav_id] = load_ratio
            except Exception as e:
                print(f"Error processing UAV {uav_id}: {str(e)}")
        
        # 对于RSU
        if hasattr(env, 'RSUs'):
            for rsu_id, rsu in env.RSUs.items():
                try:
                    # 获取RSU计算资源信息
                    if hasattr(rsu, 'fog_profile') and rsu.fog_profile:
                        total_cpu = rsu.fog_profile.get('cpu', 0)
                        used_cpu = 0
                        
                        # 获取当前正在该节点上执行的任务
                        computing_tasks = env.task_manager.getComputingTasksByNodeId(rsu_id)
                        for task in computing_tasks:
                            used_cpu += task.getAllocatedCpu()
                            total_tasks += 1
                        
                        # 计算负载比例
                        if total_cpu > 0:
                            load_ratio = used_cpu / total_cpu
                            compute_loads[rsu_id] = load_ratio
                except Exception as e:
                    print(f"Error processing RSU {rsu_id}: {str(e)}")
        
        # 打印统计信息
        print(f"Total computing tasks found: {total_tasks}")
        print(f"Nodes with compute load: {len(compute_loads)}")
        
        if len(compute_loads) > 0:
            avg_load = sum(compute_loads.values()) / len(compute_loads)
            print(f"Average compute load: {avg_load:.4f}")
        
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
    
    def plot_metrics(self):
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
        axes[1, 0].set_ylabel('Rate (bps)')
        axes[1, 0].legend()
        
        # 计算负载
        axes[1, 1].plot(self.data['time'], self.data['compute_load_avg'])
        axes[1, 1].set_title('Average Compute Load')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Load Ratio')
        
        # 特定区域数量
        area_metrics = [k for k in self.data.keys() if 'junction' in k]
        if area_metrics:
            for i, metric in enumerate(area_metrics[:2]):
                if i < 2:  # 只显示前两个交叉路口数据
                    axes[2, i].plot(self.data['time'], self.data[metric])
                    axes[2, i].set_title(f'{metric} Change')
                    axes[2, i].set_xlabel('Time')
                    axes[2, i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
