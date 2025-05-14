import json
import os

def extract_best_formulas(json_path: str, save_path: str = "predict_model.json"):
    # 检查文件路径和格式
    if not json_path.endswith('.json'):
        raise ValueError(f"Invalid input file: {json_path}. Expected a JSON file.")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        final_results = json.load(f)

    # 从 JSON 文件中提取公式列表
    assert len(final_results) >= 3, "需要找到 3 个表达式列表（Pattern 1~3）"

    result = {}

    for i, equations in enumerate(final_results[:3]):  # 只取前 3 个 pattern
        best = sorted([e for e in equations if isinstance(e["nmae"], (float, int))], key=lambda x: x["nmae"])[0]
        result[f"pattern_{i+1}"] = {
            "equation": best["equation"],
            "fitted_params": best["fitted_params"]
        }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"已导出每个模式最优表达式至 {save_path}")
