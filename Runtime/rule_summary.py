import os

def build_rule_summary_prompt(trace_log, pattern_id: int, indep_var_names: list, dep_var_name: str):
    header = f"""You are an expert in collaborative vehicular-UAV systems.
Analyze the logs of a symbolic regression system predicting `{dep_var_name}` (Pattern {pattern_id}).

Inputs:
{chr(10).join([f"x{i+1}: {name}" for i, name in enumerate(indep_var_names)])}

Log format:
- x_input: current indicators
- y_true: actual target (next-slot)
- y_pred: predicted target
- mae: error
- nmae: normalized error (if available)
- equation: used symbolic model

Goal: Extract time-dependent rules, threshold effects, correlations, and prediction failure cases.

Write rules like:
- "If x3 > 5 for 3+ steps → y > 0.8"
- "When x1 drops quickly, y spikes"

Avoid simply restating equations. Focus on **semantic behavioral patterns**.
---
"""

    log_lines = []
    for i, entry in enumerate(trace_log):
        line = f"{i+1}. x={entry['x_input']} | y_true={entry['y_true']:.3f} | y_pred={entry['y_pred']:.3f} | MAE={entry['mae']:.4f} | NMAE={entry.get('nmae', 'n/a')} | Eq={entry['equation']}"
        log_lines.append(line)

    return f"{header}\n" + "\n".join(log_lines) + "\n\nNow summarize the learned rules:"


def summarize_rules_for_pattern(updater, indep_var_names, dep_var_name, save_dir="summaries"):
    os.makedirs(save_dir, exist_ok=True)

    prompt = build_rule_summary_prompt(
        trace_log=updater.rule_trace_log,
        pattern_id=updater.pattern_id,
        indep_var_names=indep_var_names,
        dep_var_name=dep_var_name
    )

    print(f"\n[Pattern {updater.pattern_id}] Requesting rule summary...")
    result = updater.llm_chain.run(prompt)

    filename = os.path.join(save_dir, f"pattern_{updater.pattern_id}_rules.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Rule Summary for Pattern {updater.pattern_id}\n\n")
        f.write(f"**Target:** {dep_var_name}\n\n")
        f.write(f"**Inputs:**\n")
        for i, name in enumerate(indep_var_names):
            f.write(f"- x{i+1}: {name}\n")
        f.write("\n---\n\n")
        f.write(result.strip())

    print(f"✔ Summary saved to {filename}")
    return result

