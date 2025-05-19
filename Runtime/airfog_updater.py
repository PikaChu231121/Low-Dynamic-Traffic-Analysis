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
            "current_mae": None,
            "source": "initial"  # Source of the equation
        } for i in range(min(self.n_cached_expressions, len(exprs)))]

    def predict(self, x_input: list):
        global_vars = {'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: movavg([history_x[i-1] for history_x in self.history_X], k)}
        local_vars_base = {f'x{j+1}': x_input[j] for j in range(len(x_input))}
        
        predictions_made = False
        for eq_data in self.top_equations:
            local_vars = {**local_vars_base, 'c': eq_data["fitted_params"]}
            try:
                pred_eval = eval(eq_data["equation"], global_vars, local_vars)
                
                if not isinstance(pred_eval, (int, float, np.number)): # Broader check for numpy numbers
                    # Handle cases where eval might return non-scalar if not careful with expressions
                    print(f"Warning: Eval result for {eq_data['equation']} is not a scalar: {pred_eval} (type: {type(pred_eval)}). Treating as NaN.")
                    pred = float('nan')
                else:
                    pred = float(pred_eval) # Ensure it's a Python float

                eq_data["current_pred"] = pred

                if self.current_true is not None and isinstance(self.current_true, (int, float, np.number)) and \
                   not np.isnan(pred) and not np.isinf(pred) and \
                   not np.isnan(self.current_true) and not np.isinf(self.current_true):
                    eq_data["current_mae"] = abs(pred - float(self.current_true))
                else:
                    eq_data["current_mae"] = float('inf')
                predictions_made = True
            except Exception as e:
                print(f"Error evaluating equation {eq_data['equation']} or calculating MAE: {e}")
                eq_data["current_pred"] = float('nan') 
                eq_data["current_mae"] = float('inf')

        if not predictions_made and self.top_equations:
            return None # Or float('nan') if preferred for no predictions
        if not self.top_equations:
            return None # Or float('nan')

        sorted_equations = sorted(self.top_equations, key=lambda x: x.get("current_mae", float('inf')))
        
        if sorted_equations:
            best_eq_data = sorted_equations[0]
            best_pred_val = best_eq_data.get("current_pred")
            if best_pred_val is not None and not np.isnan(best_pred_val) and not np.isinf(best_pred_val):
                return float(best_pred_val)
        
        return None # Default if no valid prediction found

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
        # self.indep_vars is a list of arrays, e.g., [x1_history_array, x2_history_array, ...]
        # self.history_X is a list of lists, e.g., [[x1_t1, x2_t1], [x1_t2, x2_t2], ...]

        # Combine historical and runtime data correctly
        # First, ensure history_X is a numpy array for easier manipulation
        history_X_np = np.array(self.history_X) # Shape: (num_runtime_points, num_features)

        # Prepare X for fitting_constants: a list of 1D arrays, each array is a feature over all time points
        combined_X_features = []
        num_features = 0
        if self.indep_vars: # If there's historical data
            num_features = len(self.indep_vars)
            for i in range(num_features):
                feature_history = self.indep_vars[i] # This is already a 1D array for this feature from history
                if history_X_np.size > 0 and history_X_np.shape[1] > i : # If there's runtime data for this feature
                    feature_runtime = history_X_np[:, i]
                    combined_feature = np.concatenate((feature_history, feature_runtime))
                else:
                    combined_feature = feature_history
                combined_X_features.append(combined_feature)
        elif history_X_np.size > 0: # Only runtime data
            num_features = history_X_np.shape[1]
            for i in range(num_features):
                combined_X_features.append(history_X_np[:, i])
        
        # Ensure all combined features have the same length if data was sparse or inconsistent
        # This part might need more sophisticated handling if feature counts differ or data is missing
        if combined_X_features:
            expected_len = len(combined_X_features[0])
            for i in range(len(combined_X_features)):
                if len(combined_X_features[i]) != expected_len:
                    # This case indicates a problem with data alignment or missing data.
                    # For now, we'll raise an error or log, as fitting would be problematic.
                    # A more robust solution might involve padding or imputation if appropriate.
                    print(f"Error: Feature {i} has length {len(combined_X_features[i])}, expected {expected_len}. Data inconsistency.")
                    # Potentially, you might need to skip retraining or handle this scenario.
                    # For now, let's assume data is consistent or the error will propagate.
                    pass # Or raise ValueError("Feature length mismatch during data combination")


        # Combine historical and runtime dependent variable
        y = np.array(list(self.dep_vars) + self.history_y)
        
        # Check if combined_X_features is empty (e.g., no historical and no runtime data)
        if not combined_X_features:
            print("Warning: No data available for retraining. Skipping.")
            return

        # Ensure y has the same number of data points as the features in X
        if len(y) != len(combined_X_features[0]):
            print(f"Warning: Length mismatch between y ({len(y)}) and X features ({len(combined_X_features[0])}). Skipping retraining.")
            # This can happen if dep_vars or history_y collection is out of sync with indep_vars/history_X
            return

        # 使用运行时 prompt 生成表达式
        response = self.llm_chain.run(
            current_formula=current_expr,
            predicted=current_pred,
            actual=current_true,
            mae=abs(current_pred - current_true),
            dep=y, # Pass the combined y
            indep=combined_X_features, # Pass the correctly structured X
            Neq=5
        )

        # 解析LLM响应
        try:
            new_exprs = format_and_parse_expressions(response)
            if not new_exprs:
                print("LLM did not return valid new expressions. Keeping existing equations.")
                return
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Keeping existing equations.")
            return
            
        print(f"LLM generated {len(new_exprs)} new candidate expressions.")
        for i, expr_str in enumerate(new_exprs):
            print(f"  Candidate {i+1}: {expr_str}")
        
        # 参数拟合与优化
        print(f"Fitting constants for new expressions...")
        new_results = self.optimizer.fitting_constants(combined_X_features, y, new_exprs)

       # Filter out results where fitting failed (fitted_params is None or key missing)
        # and ensure 'nmae' is present for sorting.
        successfully_fitted_results = []
        for r in new_results:
            if r.get("fitted_params") is not None and r.get("nmae") is not None:
                successfully_fitted_results.append(r)
            else:
                print(f"  Skipping expression due to fitting failure or missing NMAE/params: {r.get('equation', 'Unknown Equation')}")

        if not successfully_fitted_results:
            print("LLM generated expressions, but none could be successfully fitted with parameters and NMAE. Keeping existing equations.")
            return

        # Sort the successfully fitted new expressions by NMAE
        sorted_new_llm_expressions = sorted(successfully_fitted_results, key=lambda x: x["nmae"]) # Now 'nmae' and 'fitted_params' are known to exist
        
        print(f"Successfully fitted {len(sorted_new_llm_expressions)} new expressions.")

        # Prepare the new equations in the standard format for caching
        llm_generated_equations_to_cache = [{
            "equation": r["equation"],
            "current_pred": None,  # Will be calculated on next predict() call
            "current_mae": r["nmae"], # Use NMAE from fitting as the initial MAE for the cache
            "fitted_params": r["fitted_params"],
            "source": "llm_generated" 
        } for r in sorted_new_llm_expressions]

        # Combine current top equations with newly generated and fitted ones
        all_candidate_equations = self.top_equations + llm_generated_equations_to_cache
        
        # Deduplicate by equation string, keeping the one with the best 'current_mae'
        # (which for new equations is their fitting NMAE, for old ones their runtime MAE)
        unique_equations_map = {}
        for eq_data in all_candidate_equations:
            eq_str = eq_data["equation"]
            # Use current_mae as the primary sorting/selection metric from the cache
            mae_to_compare = eq_data.get("current_mae", float('inf')) 
            
            if eq_str not in unique_equations_map or mae_to_compare < unique_equations_map[eq_str].get("current_mae", float('inf')):
                unique_equations_map[eq_str] = eq_data
                
        # Sort all unique candidates by their 'current_mae'
        final_sorted_equations = sorted(unique_equations_map.values(), key=lambda x: x.get("current_mae", float('inf')))
        
        # Update the cache with the top N expressions
        self.top_equations = final_sorted_equations[:self.n_cached_expressions]

        print("Updated top expressions after LLM generation and fitting:")
        for i, eq_data in enumerate(self.top_equations):
            # Ensure 'current_mae' exists or provide a default for printing
            mae_val_print = eq_data.get('current_mae')
            mae_str = f"{mae_val_print:.4f}" if isinstance(mae_val_print, (int, float)) else "N/A"
            print(f"  {i+1}. {eq_data['equation']} (MAE from fit/cache: {mae_str}, Source: {eq_data.get('source', 'unknown')})")

