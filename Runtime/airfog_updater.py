from collections import deque
import numpy as np

from helper import calculate_normalized_mae, calculate_complexity, format_expressions, format_and_parse_expressions, custom_sorting, movavg  # type: ignore

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

        # 历史运行记录
        self.history_X = []
        self.history_y = []

        # 滑动窗口
        self.recent_y_true = deque(maxlen=window_size)
        self.recent_y_pred = deque(maxlen=window_size)

        # 初始化表达式池：每条表达式维护自己的误差历史
        self.top_equations = [{
            "equation": exprs[i],
            "fitted_params": fitted_params[i],
            "mae_history": deque(maxlen=window_size),
            "nmae": float('inf')
        } for i in range(min(n_cached_expressions, len(exprs)))]
        self.initialize_mae_history()

        self.best_expression_index = 0
        self.pending_input = None
        self.pending_prediction = None
        
    def initialize_mae_history(self):
        """基于训练数据的最后 window_size 个样本，为表达式池初始化滑动 MAE 历史"""
        X_train_matrix = list(zip(*self.indep_vars))
        X_matrix = np.array(X_train_matrix + self.history_X)
        y_true = np.array(self.dep_vars.tolist() + self.history_y)
        window_k = min(len(y_true), self.window_size)

        for eq in self.top_equations:
            y_pred = []
            for i in range(len(y_true)):
                global_vars = {
                    'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp,
                    'min': np.minimum, 'max': np.maximum,
                    'movavg': lambda i, k: movavg([x[i-1] for x in (self.history_X if self.history_X else X_train_matrix)], k)
                }
                local_vars = {f'x{j+1}': X_matrix[i, j] for j in range(X_matrix.shape[1])}
                local_vars['c'] = eq['fitted_params']
                try:
                    pred = eval(eq['equation'], global_vars, local_vars)
                    y_pred.append(pred)
                except:
                    y_pred.append(np.nan)

            # 保留最后 window_k 个残差
            residuals = [abs(p - t) for p, t in zip(y_pred[-window_k:], y_true[-window_k:]) if np.isfinite(p)]
            eq["mae_history"].extend(residuals)
            
            # 计算滑动窗口 NMAE
            if len(residuals) > 0:
                y_range = np.max(y_true[-window_k:]) - np.min(y_true[-window_k:])
                nmae = np.mean(residuals) / (y_range + 1e-8)
                eq["nmae"] = nmae

    def record_prediction(self, x_input: list) -> float:
        """ 使用当前最优表达式进行预测 """
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
        """ 基于真实值反馈误差并维护历史 """
        x_input = self.pending_input
        y_pred = self.pending_prediction

        if x_input is None or y_pred is None:
            print("Warning: No pending prediction.")
            return

        self.history_X.append(x_input)
        self.history_y.append(y_true)

        self.recent_y_true.append(y_true)
        self.recent_y_pred.append(y_pred)

        current_mae = abs(y_pred - y_true)
        self.top_equations[self.best_expression_index]["mae_history"].append(current_mae)

        print(f"[Pattern {self.pattern_id}] Pred={y_pred:.4f}, True={y_true:.4f}, MAE={current_mae:.4f}")
        self.select_best_expression()

        # 触发更新判断
        if len(self.recent_y_true) == self.window_size:
            errors = [abs(p - t) for p, t in zip(self.recent_y_pred, self.recent_y_true)]
            y_range = np.max(self.recent_y_true) - np.min(self.recent_y_true)
            nmae = np.mean(errors) / (y_range + 1e-8)
            print(f"[Pattern {self.pattern_id}] Window NMAE={nmae:.4f}")
            if nmae > self.error_threshold:
                print(f"[Pattern {self.pattern_id}] Triggering update (NMAE={nmae:.4f})")
                self.retrain_expression()

    def select_best_expression(self):
        """ 基于平均 MAE 选出最优表达式 """
        best_idx = 0
        best_score = float('inf')
        for i, eq in enumerate(self.top_equations):
            if eq["mae_history"]:
                avg_mae = np.mean(eq["mae_history"])
                if avg_mae < best_score:
                    best_score = avg_mae
                    best_idx = i
        self.best_expression_index = best_idx

    def retrain_expression(self):
        """ 使用历史数据 + LLM 更新表达式池 """
        X_matrix = np.array(list(zip(*self.indep_vars)) + self.history_X)
        X = [X_matrix[:, i] for i in range(X_matrix.shape[1])]
        y = np.array(list(self.dep_vars) + self.history_y)

        current_eq = self.top_equations[self.best_expression_index]

        response = self.llm_chain.run(
            current_formula=current_eq["equation"],
            predicted=self.pending_prediction,
            actual=self.history_y[-1],
            window_size=self.window_size,
            mae=np.mean(current_eq["mae_history"]),
            y_range=(float(np.min(y)), float(np.max(y))),
            dep=y,
            indep=X,
            Neq=5
        )

        parts = response.split("<EXP>")
        if len(parts) < 2:
            print("Warning: <EXP> not found.")
            return

        try:
            new_exprs = format_expressions(format_and_parse_expressions(parts[1].strip()))
        except Exception as e:
            print(f"Error parsing expressions: {e}")
            return

        print("LLM thoughts:")
        print(parts[0].strip())

        print("Fitting new expressions...")
        new_results = self.optimizer.fitting_constants(X, y, new_exprs)

        # 当前表达式池旧结果重新评估
        old_results = []
        for eq in self.top_equations:
            nmae = calculate_normalized_mae(eq["equation"], np.column_stack([*X, y]), eq["fitted_params"])
            complexity = calculate_complexity(eq["equation"])
            old_results.append({
                "equation": eq["equation"],
                "fitted_params": eq["fitted_params"],
                "nmae": nmae,
                "complexity": complexity
            })

        merged = custom_sorting(new_results + old_results)[:self.n_cached_expressions]

        self.top_equations = [{
            "equation": r["equation"],
            "fitted_params": r["fitted_params"],
            "mae_history": deque(maxlen=self.window_size),
            "nmae": float('inf')
        } for r in merged]
        self.initialize_mae_history()
