# airfog_updater.py
from Prediction.fittingClass_airfog import FittingOptimizerAirFog
from langchain.chains import LLMChain
from Prediction.helper import format_and_parse_expressions, format_expressions, custom_sorting


class AirFogRuntimeUpdater:
    def __init__(self, pattern_id: int, expr_str: str, fitted_params: list, optimizer: FittingOptimizerAirFog,
                 llm_chain: LLMChain, indep_vars_all: list, dep_vars_all: list,
                 error_threshold: float = 0.05):
        self.pattern_id = pattern_id
        self.expr_template = expr_str
        self.params = fitted_params
        self.optimizer = optimizer
        self.llm_chain = llm_chain
        self.error_threshold = error_threshold

        self.indep_vars_all = indep_vars_all  # full matrix of X
        self.dep_vars_all = dep_vars_all      # full vector of y

        # Incremental data for online updates
        self.history_X = []
        self.history_y = []
        
        # Current prediction, ground truth and error
        self.current_pred = None
        self.current_true = None
        self.current_mae = None

    def predict(self, x_input: list) -> float:
        local_vars = {f'c{i}': self.params[i] for i in range(len(self.params))}
        local_vars.update({f'x{j+1}': x_input[j] for j in range(len(x_input))})
        return eval(self.expr_template, {}, local_vars)

    def update_with_feedback(self, x_input: list, y_true: float):
        self.history_X.append(x_input)
        self.history_y.append(y_true)

        self.current_pred = self.predict(x_input)
        self.current_true = y_true
        self.current_mae = abs(self.current_pred - self.current_true)
        print(f"Pattern {self.pattern_id}: Pred={self.current_pred:.4f}, True={self.current_true:.4f}, Error={self.current_mae:.4f}")

        if self.current_mae > self.error_threshold:
            print(f"Error exceeds threshold, triggering re-optimization for pattern {self.pattern_id}")
            self.retrain_expression()

    def retrain_expression(self):
        # 使用历史样本再训练
        X = self.indep_vars_all[self.pattern_id] + self.history_X
        y = self.dep_vars_all[self.pattern_id] + self.history_y

        # 使用运行时 prompt 生成表达式
        response = self.llm_chain.run(current_formula=self.expr_template, 
                                      predicted=self.current_pred, 
                                      actual=self.current_true, 
                                      mae=self.current_mae,
                                      dep=y,
                                      indep=X,
                                      Neq=5)

        # 提取并解析表达式
        parts = response.split("<EXP>")
        LLMthoughts = ''
        equationsStr = ''
        equations = []
        if len(parts) < 2:
            print("Warning: <EXP> not found or incorrectly placed in the response.")
            LLMthoughts = response.strip()
        else:
            LLMthoughts = parts[0].strip()
            equationsStr = parts[1].strip()
            equations = format_and_parse_expressions(equationsStr)  # Parse the list of equations
        new_exprs = format_expressions(equations)
        print("LLM thoughts:")
        print(LLMthoughts)
        
        # 参数拟合与优化
        print(f"Fitting constants for new expressions...")
        new_results = self.optimizer.fitting_constants(X, y, new_exprs)

        # 最优表达式更新
        best = custom_sorting(new_results)[-1]
        self.expr_template = best["equation"]
        self.params = best["fitted_params"]
        print(f"Updated expression for Pattern {self.pattern_id+1}: {self.expr_template} (MAE={best['mae']:.4f})")

