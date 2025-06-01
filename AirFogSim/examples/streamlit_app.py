# streamlit_app.py
import os
import sys
import time

import streamlit as st

st.set_page_config(layout="wide")

# --- 行1：表单与 GUI ---
col_left, _, col_right = st.columns([5, 1, 4])  # 中间是空隙

with col_left:
    st.title("AirFogSim 模拟控制台")

    with st.form("param_form"):
        num_drones = st.number_input("请输入无人机数量", min_value=1, max_value=100, value=60)
        num_vehicles = st.number_input("请输入车辆数量", min_value=1, max_value=100, value=50)
        submitted = st.form_submit_button("开始模拟")

    status_placeholder = st.empty()

# 右半 GUI 画面区域
with col_right:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    airfogsim_placeholder = st.empty()

# --- 第二行：曲线图展示区 ---
curve_placeholder = st.empty()

# --- 第三行：模拟完成提示 ---
final_status_placeholder = st.empty()

if submitted:
    status_placeholder.info("模拟开始，请稍等...")

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
    config['simulation']['vehicle_count'] = num_drones  # 至少50辆车
    config['traffic']['max_n_UAVs'] = num_vehicles  # 至少10架无人机

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
                junction[0] + 200, junction[1] + 200  # max_x, max_y
            ]
            data_collector.area_specific_data[f'junction_{i}'] = area_bounds

    formula_json_paths = [
        os.path.join(os.path.dirname(__file__), '../output/runtime/final_formulas.json'),
        # os.path.join(os.path.dirname(__file__), '../../Prediction/results/train/run1.json'),
        # os.path.join(os.path.dirname(__file__), '../../Prediction/results/train/run2.json'),
        # os.path.join(os.path.dirname(__file__), '../../Prediction/results/train/run3.json')
    ]
    output_json_path = os.path.join(os.path.dirname(__file__), '../output/runtime/final_nmae.json')
    output_image_path = os.path.join(os.path.dirname(__file__), '../output/runtime')
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 模拟执行
    accumulated_reward = 0

    plot_placeholder = st.empty()

    while not env.isDone():
        # 记录任务信息
        prev_tasks = len(env.task_manager.getAllTasks())

        algorithm_module.scheduleStep(env)
        env.step()

        accumulated_reward += algorithm_module.getRewardByTask(env)

        # 检查任务变化
        curr_tasks = len(env.task_manager.getAllTasks())
        done_tasks = TaskScheduler.getDoneTaskNum(env)

        # 每10个时间步收集一次数据并绘图
        if int(env.simulation_time * 10) % 10 == 0:
            data_collector.collect(env, algorithm_module)
            # 获取 AirFogSim GUI 当前图像
            fig_gui_buf = env.get_gui_image()
            if fig_gui_buf:
                airfogsim_placeholder.image(fig_gui_buf, caption="AirFogSim GUI", width=400)
            # 实时绘图
            fig_buf = data_collector.predict_and_compare_metrics_live(
                formula_json_paths=formula_json_paths
            )
            curve_placeholder.image(fig_buf, caption="预测 vs 实际曲线", use_container_width=False, width=1400)
            time.sleep(0.2)  # 避免过快刷新

        # print(f"Simulation time: {env.simulation_time:.2f}, Reward: {accumulated_reward:.2f}", end='\r')

        env.render()  # 渲染可视化

    # 关闭环境
    env.close()

    print("\nSimulation done.")

    # 批量测试多个实验结果
    # ../output/runtime/nmae_results.json
    # ../output/prediction/all_runs_nmae.json

    data_collector.predict_and_compare_metrics(
        formula_json_paths=formula_json_paths,
        output_json_path=output_json_path,
        output_image_path=output_image_path,
    )
    print(f"所有实验的NMAE结果已保存到: {output_json_path}")

    final_status_placeholder.success("模拟完成！")

