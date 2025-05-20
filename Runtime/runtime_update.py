import json
import os
import sys
import numpy as np

os.environ['OPENAI_API_KEY'] = 'sk-5T6oMdbJnNP23WEC5psT4I8sZyEd90Nve1YquVKdj9coHpIy'
os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.org/v1'

# 把Prediction添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
prediction_dir = os.path.join(project_root, 'Prediction')
sys.path.append(prediction_dir)

if current_dir not in sys.path:
    sys.path.append(current_dir)

from airfog_updater import AirFogRuntimeUpdater
from build_runtime_llm_chain import build_runtime_llm_chain
from export_formulas import extract_best_formulas
from fittingClass_airfog import FittingOptimizerAirFog  # type: ignore
from load_data import load_data  # type: ignore

def init_runtime_updaters(model_path=None, data_path=None):
    """
    初始化运行时模型更新器
    
    Args:
        model_path: 预测模型JSON路径，如果为None则尝试导出新模型
        data_path: 训练数据CSV路径
    
    Returns:
        updaters: 包含3个模式的updater列表
    """
    # 如果没有指定模型路径，则尝试导出最佳表达式
    if model_path is None:
        test_results_path = os.path.join(project_root, "AirFogSim/output/prediction/all_runs_nmae.json")
        model_path = os.path.join(project_root, "Runtime/predict_model.json")
        if os.path.exists(test_results_path):
            print(f"从{test_results_path}导出最佳表达式")
            extract_best_formulas(test_results_path, model_path)
        else:
            print(f"无法找到测试结果文件: {test_results_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"无法找到预测模型文件: {model_path}")
            
    # 加载导出的表达式
    with open(model_path, "r") as f:
        model_data = json.load(f)
    print(f"已加载预测模型: {model_path}")
    
    # 用于训练的原始输入输出
    if data_path is None:
        data_path = os.path.join(project_root, "AirFogSim/output/global_data.csv")
    
    if not os.path.exists(data_path):
        print(f"警告：找不到训练数据文件 {data_path}，使用空数据初始化")
        indep_vars_train = [[] for _ in range(3)]
        dep_vars_train = [[] for _ in range(3)]
    else:
        print(f"加载训练数据: {data_path}")
        indep_vars_train, dep_vars_train = load_data(data_path)
    
    optimizer = FittingOptimizerAirFog()
    
    # 初始化 3 个模式的 updater
    updaters = []
    pattern_keys = ["pattern_1", "pattern_2", "pattern_3"]
    
    for i, pat_key in enumerate(pattern_keys):
        if pat_key not in model_data:
            print(f"警告：模型数据中没有 {pat_key}，跳过初始化")
            updaters.append(None)
            continue
            
        try:
            updater = AirFogRuntimeUpdater(
                pattern_id=i,
                exprs=[eq["equation"] for eq in model_data[pat_key]],
                fitted_params=[eq["fitted_params"] for eq in model_data[pat_key]],
                optimizer=optimizer,
                llm_chain=build_runtime_llm_chain(i),
                indep_vars=indep_vars_train[i] if i < len(indep_vars_train) else [],
                dep_vars=dep_vars_train[i] if i < len(dep_vars_train) else [],
                error_threshold=0.15,
                n_cached_expressions=len(model_data[pat_key]),
            )
            updaters.append(updater)
            print(f"初始化 {pat_key} 更新器成功")
        except Exception as e:
            print(f"初始化 {pat_key} 更新器失败: {e}")
            updaters.append(None)
    
    return updaters

# 加载导出的表达式
with open("predict_model.json", "r") as f:
    model_data = json.load(f)

# 用于训练的原始输入输出（airFogSim生成）
indep_vars_train, dep_vars_train = load_data('../AirFogSim/output/global_data.csv')

optimizer = FittingOptimizerAirFog()

# 初始化 3 个模式的 updater
updaters = []
for i in range(3):
    pat_key = f"pattern_{i+1}"
    updater = AirFogRuntimeUpdater(
        pattern_id=i,
        exprs=[eq["equation"] for eq in model_data[pat_key]],
        fitted_params=[eq["fitted_params"] for eq in model_data[pat_key]],
        optimizer=optimizer,
        llm_chain=build_runtime_llm_chain(i),
        indep_vars=indep_vars_train,
        dep_vars=dep_vars_train,
        error_threshold=0.15,
        n_cached_expressions=len(model_data[pat_key]),
    )
    updaters.append(updater)

# 示例运行：t 时间片都传入新的 x，t+1 时间片传入新的 y
for t in range(10):
    pattern_id = 2   # 例如 Pattern 3，对应 index=2
    
    # t 时间片
    x = [...]        # 当前时间片输入（例如来自 AirFogSim）
    updaters[pattern_id].record_prediction(x)
    
    # t+1 时间片
    y = ...          # 下一时间片真实输出
    updaters[pattern_id].update_with_feedback(y)
