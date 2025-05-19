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

def update_runtime_model(updaters, pattern_id, x, y):
    """
    使用新的观测数据更新运行时模型
    
    Args:
        updaters: updater列表
        pattern_id: 要更新的模式ID (0-2)
        x: 当前时间片的输入特征列表
        y: 当前时间片的真实输出值
    
    Returns:
        prediction: 模型对当前输入的预测值
        error: 预测误差
    """
    if pattern_id < 0 or pattern_id >= len(updaters) or updaters[pattern_id] is None:
        print(f"警告: Updater for pattern {pattern_id} 不可用")
        return None, None
    
    # 转换为numpy数组
    try:
        # 确保 x 中的所有元素都是数值类型，如果不是，np.array可能会创建 object 类型的数组
        # 如果 x 可能包含 None，应该在 demo03.py 中处理掉
        x_array = np.array(x, dtype=float) # 显式指定 dtype，如果x包含非数值会报错
    except ValueError as e:
        print(f"错误: 转换输入 x 到 numpy 数组失败 for pattern {pattern_id}: {x}. Error: {e}")
        return None, None
    
    # 预测当前值
    prediction = updaters[pattern_id].predict(x_array)
    
    # 计算错误
    error = None
    if y is not None:
        if prediction is not None:
            # 确保 y 也是数值类型
            try:
                y_val = float(y)
                error = abs(prediction - y_val) / (abs(y_val) + 1e-8)
            except (TypeError, ValueError):
                print(f"警告: 无法计算误差，因为 y ({y}) 不是有效的数值。")
                error = None # 保持 error 为 None
        else:
            # prediction 为 None，无法计算 error
            print(f"信息: pattern {pattern_id} 的 prediction 为 None，无法计算误差。")
            error = None # 保持 error 为 None
    
    # 更新模型
    if y is not None:
        updaters[pattern_id].update_with_feedback(x_array, y)
    
    return prediction, error
