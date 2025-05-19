# airfog_updater.py
import numpy as np
from fittingClass_airfog import FittingOptimizerAirFog  # type: ignore
from langchain.chains import LLMChain
from helper import format_and_parse_expressions, format_expressions, custom_sorting, movavg  # type: ignore
from collections import deque


class AirFogRuntimeUpdater:
    def __init__(self, pattern_id: int, exprs: list, fitted_params: list, optimizer: FittingOptimizerAirFog,
                 llm_chain: LLMChain, indep_vars: list, dep_vars: str,
                 error_threshold: float = 0.1, n_cached_expressions: int = 3, window_size: int = 5):
        self.pattern_id = pattern_id
        self.optimizer = optimizer
        self.llm_chain = llm_chain
        self.error_threshold = error_threshold
        self.n_cached_expressions = n_cached_expressions
        
        self.window_size = window_size
        self.recent_y_true = deque(maxlen=self.window_size)
        self.recent_y_pred = deque(maxlen=self.window_size)
        self.recent_window_nmae = 0.0
        
        dep_vars = np.array(eval(dep_vars))
        indep_vars = [np.array(eval(indep_var)) for indep_var in indep_vars]
        
        self.dep_vars = np.round(dep_vars, 3)
        self.indep_vars = [np.round(indep_var, 3) for indep_var in indep_vars]

        # Incremental data for online updates
        self.history_X = []
        self.history_y = []
        
        # Current prediction, ground truth and error
        self.current_true = None
        
        # Cached expressions
        self.top_equations = [{
            "equation": exprs[i],
            "fitted_params": fitted_params[i],
            "current_pred": None,
            "current_mae": None
        } for i in range(min(self.n_cached_expressions, len(exprs)))]

    def predict(self, x_input: list):
        global_vars = {'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: movavg([history_x[i-1] for history_x in self.history_X], k)}
        local_vars = {**{f'x{j+1}': x_input[j] for j in range(len(x_input))}}
        
        for eq in self.top_equations:
            local_vars['c'] = eq["fitted_params"]
            try:
                pred = eval(eq["equation"], global_vars, local_vars)
                eq["current_pred"] = pred
                eq["current_mae"] = abs(pred - self.current_true)
            except Exception as e:
                print(f"Error evaluating equation {eq['equation']}: {e}")
                eq["current_pred"] = float('nan')
                eq["current_mae"] = float('inf')

    def update_with_feedback(self, x_input: list, y_true: float):
        self.current_true = y_true
        self.history_X.append(x_input)
        self.history_y.append(y_true)

        self.predict(x_input)
        self.top_equations = sorted(self.top_equations, key=lambda x: x["current_mae"])
        best_eq = self.top_equations[0]

        self.recent_y_true.append(y_true)
        self.recent_y_pred.append(best_eq["current_pred"])

        print(f"[Pattern {self.pattern_id}] Best Pred={best_eq['current_pred']:.4f}, True={y_true:.4f}, MAE={best_eq['current_mae']:.4f}")

        # 滑动窗口 NMAE 计算
        if len(self.recent_y_true) == self.window_size:
            errors = [abs(p - t) for p, t in zip(self.recent_y_pred, self.recent_y_true)]
            nmae = np.mean(errors) / (np.max(self.recent_y_true) - np.min(self.recent_y_true) + 1e-8)
            print(f"[Pattern {self.pattern_id}] Window NMAE={nmae:.4f}")
            if nmae > self.error_threshold:
                print(f"Triggering update: window NMAE {nmae:.4f} > {self.error_threshold}")
                self.retrain_expression(best_eq['equation'], best_eq['current_pred'], y_true)

    def retrain_expression(self, current_expr: str, current_pred: float, current_true: float):
        X = list(map(np.array, list(zip(*self.indep_vars)) + self.history_X))
        y = np.array(list(self.dep_vars) + self.history_y)

        # 使用运行时 prompt 生成表达式
        response = self.llm_chain.run(
            current_formula=current_expr,
            predicted=current_pred,
            actual=current_true,
            mae=abs(current_pred - current_true),
            dep=y,
            indep=X,
            Neq=5
        )

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
            equations = format_and_parse_expressions(equationsStr)
        new_exprs = format_expressions(equations)
        print("LLM thoughts:")
        print(LLMthoughts)
        
        # 参数拟合与优化
        print(f"Fitting constants for new expressions...")
        new_results = self.optimizer.fitting_constants(X, y, new_exprs)

        # Top N 公式更新
        sorted_results = custom_sorting(new_results)[-self.n_cached_expressions:]
        self.top_equations = [{
            "equation": r["equation"],
            "fitted_params": r["fitted_params"],
            "current_pred": None,
            "current_mae": None
        } for r in sorted_results]
        print(f"Updated top-{self.n_cached_expressions} expressions:")
        for r in sorted_results:
            print(f"{r['equation']} (NMAE={r['nmae']:.4f})")

