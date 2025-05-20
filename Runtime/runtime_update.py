import json
from airfog_updater import AirFogRuntimeUpdater
from build_runtime_llm_chain import build_runtime_llm_chain
from export_formulas import extract_best_formulas
from Prediction.fittingClass_airfog import FittingOptimizerAirFog
from Prediction.load_data import load_data

# 导出最优表达式
extract_best_formulas("../AirFogSim/output/test_results.json")

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
