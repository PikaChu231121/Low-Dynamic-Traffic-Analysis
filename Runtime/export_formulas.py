import json
import os

def extract_best_formulas(json_path: str, save_path: str = "predict_model.json", n: int = 3) -> None:
    # 检查文件路径和格式
    if not json_path.endswith('.json'):
        raise ValueError(f"Invalid input file: {json_path}. Expected a JSON file.")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        all_runs_data = json.load(f)

    if not all_runs_data:
        raise ValueError(f"Input file {json_path} is empty or does not contain any run data.")

    # 默认使用第一次运行的数据
    # all_runs_data 是一个运行列表 (List[RunData])
    # RunData 是一个模式列表 (List[PatternData]) -> e.g., [pattern1_results, pattern2_results, pattern3_results]
    # PatternData 是一个公式字典列表 (List[EquationDict]) -> e.g., [{"equation": ..., "nmae": ...}, ...]
    first_run_data = all_runs_data[0]

    # 确认第一次运行的数据包含至少3个模式的结果
    assert len(first_run_data) >= 3, f"The first run data in {json_path} must contain results for at least 3 patterns."

    result = {}

    # 遍历第一次运行的前3个模式
    for pattern_idx, pattern_equations_list in enumerate(first_run_data[:3]):
        # pattern_equations_list 是一个包含多个公式详情的列表 (List[EquationDict])
        # e 是一个公式详情字典 (EquationDict)
        # 过滤掉 "nmae" 不是数字或为 None 的条目
        valid_equations = [
            e for e in pattern_equations_list 
            if isinstance(e.get("nmae"), (float, int)) and e.get("nmae") is not None
        ]
        bests = sorted(valid_equations, key=lambda x: x["nmae"])[:n]
        
        result[f"pattern_{pattern_idx+1}"] = [{
            "equation": best["equation"],
            "fitted_params": best["fitted_params"]
        } for best in bests]

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"已导出每个模式最优表达式至 {save_path}")
