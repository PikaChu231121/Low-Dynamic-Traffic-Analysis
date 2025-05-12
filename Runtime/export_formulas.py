import json
import re


def extract_best_formulas(txt_path: str, save_path: str = "predict_model.json"):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 提取所有 JSON 列表块（每个 pattern 对应一个）
    pattern_blocks = re.findall(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
    assert len(pattern_blocks) >= 4, "需要找到 4 个表达式列表（Pattern 1~4）"

    result = {}

    for i, block in enumerate(pattern_blocks[:4]):  # 只取前 4 个 pattern
        equations = json.loads(block)
        best = sorted([e for e in equations if isinstance(e["mae"], (float, int))], key=lambda x: x["mae"])[0]
        result[f"pattern_{i+1}"] = {
            "equation": best["equation"],
            "fitted_params": best["fitted_params"]
        }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"已导出每个模式最优表达式至 {save_path}")
