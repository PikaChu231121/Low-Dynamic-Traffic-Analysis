import json
from airfog_updater import AirFogRuntimeUpdater
from build_runtime_llm_chain import build_runtime_llm_chain
from Prediction.fittingClass_airfog import FittingOptimizerAirFog
from Prediction.load_data import load_data

# 加载导出的表达式
with open("predict_model.json", "r") as f:
    model_data = json.load(f)

# 你用于训练的原始输入输出（airFogSim生成）
indep_vars_all, dep_vars_all = load_data('../AirFogSim/output/global_data.csv')

optimizer = FittingOptimizerAirFog()

# 初始化 4 个模式的 updater
updaters = []
for i in range(4):
    pat_key = f"pattern_{i+1}"
    updater = AirFogRuntimeUpdater(
        pattern_id=i,
        expr_str=model_data[pat_key]["equation"],
        fitted_params=model_data[pat_key]["fitted_params"],
        optimizer=optimizer,
        llm_chain=build_runtime_llm_chain(i),
        indep_vars_all=indep_vars_all,
        dep_vars_all=dep_vars_all,
        error_threshold=0.03
    )
    updaters.append(updater)

# 示例运行：每个时间片都传入新的 (x, y)
for t in range(10):
    x = [...]        # 当前时间片输入（例如来自 AirFogSim）
    y = ...          # 当前真实输出
    pattern_id = 2   # 例如 Pattern 3，对应 index=2
    updaters[pattern_id].update_with_feedback(x, y)
