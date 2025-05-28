from collections import deque
import numpy as np

from helper import format_expressions, format_and_parse_expressions, custom_sorting, movavg  # type: ignore

class AirFogRuntimeUpdater:
    def __init__(self, pattern_id: int, exprs: list, fitted_params: list, optimizer, llm_chain,
                 indep_vars: list, dep_vars: str, error_threshold: float = 0.1,
                 n_cached_expressions: int = 3, window_size: int = 5):
        
        self.pattern_id = pattern_id
        self.optimizer = optimizer
        self.llm_chain = llm_chain
        self.error_threshold = error_threshold
        self.n_cached_expressions = n_cached_expressions
        self.window_size = window_size
        
        # 原始训练数据
        self.dep_vars = np.round(np.array(eval(dep_vars)), 3)
        self.indep_vars = [np.round(np.array(eval(v)), 3) for v in indep_vars]

        # 动态历史
        self.history_X = []
        self.history_y = []
        self.recent_y_true = deque(maxlen=window_size)
        self.recent_y_pred = deque(maxlen=window_size)

        # 表达式池初始化
        self.top_equations = [{
            "equation": exprs[i],
            "fitted_params": fitted_params[i],
            "last_mae": float('inf'),   # 初始设置为 inf
        } for i in range(min(n_cached_expressions, len(exprs)))]

        self.best_expression_index = 0  # 当前预测使用哪个表达式
        self.pending_input = None       # 上轮 x_input
        self.pending_prediction = None  # 上轮预测值

    def record_prediction(self, x_input: list) -> float:
        """ 使用上次最优表达式预测，并保存 x_input 与预测值 """
        eq = self.top_equations[self.best_expression_index]
        equation, params = eq["equation"], eq["fitted_params"]

        global_vars = {
            'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp,
            'min': np.minimum, 'max': np.maximum,
            'movavg': lambda i, k: movavg([x[i-1] for x in (self.history_X + [x_input])], k)
        }
        local_vars = {'c': params, **{f'x{j+1}': x_input[j] for j in range(len(x_input))}}

        try:
            pred = eval(equation, global_vars, local_vars)
        except Exception as e:
            print(f"[Pattern {self.pattern_id}] Prediction failed: {e}")
            pred = float('nan')

        self.pending_input = x_input
        self.pending_prediction = pred
        return pred

    def update_with_feedback(self, y_true: float):
        """ 根据上次记录的 x_input 和预测值进行评估并触发更新 """
        x_input = self.pending_input
        y_pred = self.pending_prediction
        if x_input is None or y_pred is None:
            print("Warning: No pending prediction to update.")
            return

        self.history_X.append(x_input)
        self.history_y.append(y_true)
        self.recent_y_true.append(y_true)
        self.recent_y_pred.append(y_pred)

        current_mae = abs(y_pred - y_true)
        print(f"[Pattern {self.pattern_id}] Pred={y_pred:.4f}, True={y_true:.4f}, MAE={current_mae:.4f}")

        # 更新当前表达式的 MAE 记录
        self.top_equations[self.best_expression_index]["last_mae"] = current_mae

        # 选择新的最优表达式用于下次预测
        self.select_best_expression()

        # 滑动窗口 NMAE 判断是否需要更新
        if len(self.recent_y_true) == self.window_size:
            errors = [abs(p - t) for p, t in zip(self.recent_y_pred, self.recent_y_true)]
            y_range = np.max(self.recent_y_true) - np.min(self.recent_y_true)
            # 避免分母为0
            nmae = np.mean(errors) / (y_range + 1e-8)
            print(f"[Pattern {self.pattern_id}] Window NMAE={nmae:.4f}")
            if nmae > self.error_threshold:
                print(f"[Pattern {self.pattern_id}] Triggering update (NMAE={nmae:.4f})")
                self.retrain_expression()

    def select_best_expression(self):
        """ 选择 MAE 最小的表达式用于下次预测 """
        valid_eqs = [i for i, eq in enumerate(self.top_equations) if np.isfinite(eq["last_mae"])]
        if valid_eqs:
            self.best_expression_index = min(valid_eqs, key=lambda i: self.top_equations[i]["last_mae"])
        else:
            self.best_expression_index = -1

    def retrain_expression(self):
        """ 使用历史数据 + LLM + optimizer 重建表达式 """
        X_matrix = np.array(list(zip(*self.indep_vars)) + self.history_X)
        X = [X_matrix[:, i] for i in range(X_matrix.shape[1])]
        y = np.array(list(self.dep_vars) + self.history_y)
        current_eq = self.top_equations[self.best_expression_index]

        response = self.llm_chain.run(
            current_formula=current_eq["equation"],
            predicted=self.pending_prediction,
            actual=self.history_y[-1],
            mae=current_eq["last_mae"],
            dep=y,
            indep=X,
            Neq=5
        )

        parts = response.split("<EXP>")
        if len(parts) < 2:
            print("Warning: <EXP> not found. Skipping update.")
            return

        thoughts, expr_str = parts[0].strip(), parts[1].strip()
        try:
            new_exprs = format_expressions(format_and_parse_expressions(expr_str))
        except Exception as e:
            print(f"Error parsing new expressions: {e}\nSkipping update.")
            return
        print("LLM thoughts:")
        print(thoughts)

        print("Fitting constants for new expressions...")
        new_results = self.optimizer.fitting_constants(X, y, new_exprs)
        sorted_results = custom_sorting(new_results)[-self.n_cached_expressions:]

        self.top_equations = [{
            "equation": r["equation"],
            "fitted_params": r["fitted_params"],
            "last_mae": r["nmae"],  # 使用拟合NMAE作为初始MAE
        } for r in sorted_results]

        print(f"Updated expressions:")
        for r in sorted_results:
            print(f"{r['equation']} (NMAE={r['nmae']:.4f})")

        # 重新选择最优表达式
        self.best_expression_index = 0
