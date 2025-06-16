import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


def build_rule_summarizer_llm_chain(model):
    prompt = PromptTemplate(
        input_variables=["log", "pattern_id", "indep_list", "dep_name"],
        template="""
You are an expert in vehicular-UAV collaborative systems.

You will be given data collected from a low-altitude vehicular network where vehicles and UAVs collaborate to complete computational tasks under limited bandwidth and computational resources.
Key indicators—such as task success ratio, V2U rate, and UAV density—are recorded at each time slot. Predictive patterns describe the relationship between next-slot outcomes and current-slot indicators.

**Pattern {pattern_id}:** Predict `{dep_name}` from inputs:
{indep_list}

Each log entry includes:
- x_input: current input indicators
- y_true: actual outcome
- y_pred: predicted outcome
- mae: prediction error
- equation: symbolic expression used
- nmae: normalized error (if available)

Now here are the logs:

{log}

---

Your task is to summarize **behavioral rules** and **indicator-outcome relationships** observed in this pattern.

Focus on:

- High-level causal patterns (e.g. “If x3 remains high for several time steps, then y tends to increase.”)
- Threshold-based triggers (e.g. “When vehicle density > 5 and task success ratio < 0.6, compute load spikes.”)
- Stable vs unstable regions of the system
- Local patterns: brief intervals where indicators show strong influence on outcomes
- Ignore the internal structure of symbolic equations

Avoid:
- Describing symbolic expressions or constants
- Overfitting to one-off fluctuations

Respond in clear and concise markdown bullet points, using natural language. Do NOT list equations.
"""
    )

    llm = ChatOpenAI(
        model_name=model,
        temperature=0.4,
        max_tokens=1000
    )

    return LLMChain(llm=llm, prompt=prompt)


def summarize_rules_for_pattern(updater, indep_var_names, dep_var_name, model="gpt-4o", save_dir="Summaries"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_dir = os.path.join(project_root, save_dir, f"pattern_{updater.pattern_id}")
    os.makedirs(save_dir, exist_ok=True)

    rule_llm_chain = build_rule_summarizer_llm_chain(model)

    log_lines = []
    for i, entry in enumerate(updater.rule_trace_log):
        line = f"{i+1}. x={entry['x_input']} | y_true={entry['y_true']:.3f} | y_pred={entry['y_pred']:.3f} | MAE={entry['mae']:.4f} | NMAE={entry.get('nmae', 'n/a')} | Eq={entry['equation']}"
        log_lines.append(line)

    log_text = "\n".join(log_lines)

    result = rule_llm_chain.run(
        log=log_text,
        pattern_id=updater.pattern_id,
        indep_list={'\n'.join([f"x{i+1}: {name}" for i, name in enumerate(indep_var_names)])},
        dep_name=dep_var_name
    )

    filename = os.path.join(save_dir, f"pattern_{updater.pattern_id}_rules.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Rule Summary for Pattern {updater.pattern_id}\n\n")
        f.write(result.strip())

    print(f"✔ Rule summary saved to {filename}")
