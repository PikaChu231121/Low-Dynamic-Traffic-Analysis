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

    # aggregated_formulas_by_pattern 将存储每个模式（前3个）从所有运行中收集到的所有有效公式
    # 键是模式索引 (0, 1, 2)，值是公式字典的列表
    aggregated_formulas_by_pattern = {i: [] for i in range(3)}

    for run_data in all_runs_data:
        # RunData 是一个模式列表 (List[PatternData])
        # PatternData 是一个公式字典列表 (List[EquationDict])
        for pattern_idx in range(3): # 我们只关心前3个模式
            if pattern_idx < len(run_data):
                pattern_equations_list = run_data[pattern_idx]
                # 过滤掉 "nmae" 不是数字或为 None 的条目
                valid_equations = [
                    e for e in pattern_equations_list
                    if isinstance(e.get("nmae"), (float, int)) and e.get("nmae") is not None
                ]
                aggregated_formulas_by_pattern[pattern_idx].extend(valid_equations)

    result = {}

    # 遍历前3个模式，从聚合数据中选择最优公式
    for pattern_idx in range(3):
        all_formulas_for_this_pattern = aggregated_formulas_by_pattern[pattern_idx]
        
        if not all_formulas_for_this_pattern:
            bests = []
        else:
            # 按 nmae 排序并选择前 n 个
            bests = sorted(all_formulas_for_this_pattern, key=lambda x: x["nmae"])[:n]
        
        result[f"pattern_{pattern_idx+1}"] = [{
            "equation": best["equation"],
            "fitted_params": best["fitted_params"]
        } for best in bests]

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"已导出每个模式最优表达式至 {save_path}")
