import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Hardcoded indep_var_map based on demo03.py and the structure of dep_vars in runtime_predictions.json
INDEPVARMAP = [
    ['task_success_ratio', 'vehicle_density', 'avg_V2U_rate', 'compute_load_avg'], # For compute_load_avg (x1, x2, x3, x4)
    ['vehicle_density', 'uav_density', 'compute_load_avg'],                        # For avg_V2U_rate
    ['avg_V2U_rate', 'avg_V2I_rate', 'compute_load_avg'],                         # For task_success_ratio
]

# Define paths
base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
airfogsim_output_path = os.path.join(base_project_dir, 'AirFogSim/output')
runtime_output_path = os.path.join(airfogsim_output_path, 'runtime')
json_path = os.path.join(runtime_output_path, 'runtime_predictions_ds.json')
csv_path = os.path.join(airfogsim_output_path, 'global_data.csv')
plot_output_dir = os.path.join(runtime_output_path, 'formula_history_plots_ds')

os.makedirs(plot_output_dir, exist_ok=True)

def replace_params_in_formula(equation_str, params_list):
    if not isinstance(equation_str, str): return None
    if not isinstance(params_list, list): return equation_str
    def repl(match):
        idx = int(match.group(1))
        return f"{params_list[idx]:.6f}" if idx < len(params_list) else match.group(0)
    return re.sub(r"c\[(\d+)]", repl, equation_str)

def evaluate_formula(formula_str_with_coeffs, data_df, pattern_id, indep_var_map_global, dep_vars_from_json_list):
    if formula_str_with_coeffs is None: return None

    eval_namespace = {}
    current_pattern_indep_vars = indep_var_map_global[pattern_id]

    for i, var_name_actual in enumerate(current_pattern_indep_vars):
        if var_name_actual in data_df.columns:
            eval_namespace[f'x{i+1}'] = data_df[var_name_actual]
        else:
            print(f"Warning: Indep var '{var_name_actual}' for x{i+1} (pattern {pattern_id}) not in data_df. Cols: {data_df.columns.tolist()}")
            return None

    for col in data_df.columns:
        if col not in eval_namespace: eval_namespace[col] = data_df[col]
    
    eval_namespace['np'] = np
    eval_namespace['sqrt'] = np.sqrt
    eval_namespace['exp'] = np.exp
    eval_namespace['min'] = np.minimum
    eval_namespace['max'] = np.maximum
    # Safer log function
    eval_namespace['log'] = lambda val: np.log(np.maximum(np.asarray(val, dtype=float), 1e-9))


    current_formula_str = formula_str_with_coeffs
    
    movavg_pattern = re.compile(r"movavg\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
    computed_movavgs = {}

    def replace_movavg_match(m):
        var_idx_str, window_str = m.group(1), m.group(2)
        window = int(window_str)
        var_list_idx = int(var_idx_str) - 1
        if not (0 <= var_list_idx < len(current_pattern_indep_vars)):
            raise ValueError(f"movavg index {var_idx_str} out of bounds.")
        actual_var_name = current_pattern_indep_vars[var_list_idx]
        placeholder_name = f"__MA_{actual_var_name.replace('_','')}_{window}__"
        if placeholder_name not in computed_movavgs:
            if actual_var_name in data_df.columns:
                series = data_df[actual_var_name].rolling(window=window, min_periods=1).mean()
                eval_namespace[placeholder_name] = series
                computed_movavgs[placeholder_name] = series
            else:
                raise ValueError(f"Var '{actual_var_name}' for movavg not in data_df.")
        return placeholder_name

    processed_formula_str = current_formula_str
    try:
        # Pre-calculate all movavg series first
        for match in movavg_pattern.finditer(current_formula_str):
            replace_movavg_match(match) # Populates eval_namespace
        processed_formula_str = movavg_pattern.sub(replace_movavg_match, current_formula_str)
    except ValueError as e:
        print(f"Error processing movavg terms in '{current_formula_str}': {e}")
        return None

    # Special handling for x5 in pattern 0 (compute_load_avg lagged)
    if pattern_id == 0 and 'x5' in processed_formula_str:
        dep_var_name_for_pattern0 = dep_vars_from_json_list[0] # Should be 'compute_load_avg'
        if dep_var_name_for_pattern0 in data_df.columns:
            dep_col_for_x5 = data_df[dep_var_name_for_pattern0]
            x5_series = dep_col_for_x5.shift(1)
            if not x5_series.empty:
                fill_value_for_first_nan = dep_col_for_x5.iloc[0] if not dep_col_for_x5.empty else 0.0
                if pd.isna(x5_series.iloc[0]):
                    x5_series.iloc[0] = fill_value_for_first_nan
            x5_series.fillna(0.0, inplace=True) # Fill any other NaNs with 0.0
            eval_namespace['x5'] = x5_series
        else:
            print(f"Warning: Cannot create x5 for pattern 0. Dep var '{dep_var_name_for_pattern0}' not in data_df.")
            # If x5 is crucial and missing, evaluation might fail or be incorrect.
            # Depending on formula, might not need to return None if x5 part is 0*x5 etc.

    try:
        predicted_series = pd.eval(processed_formula_str, local_dict=eval_namespace, global_dict={})
        return predicted_series
    except Exception as e:
        print(f"Error evaluating formula '{processed_formula_str}' (original: '{current_formula_str}'): {e}")
        return None

def main():
    try:
        with open(json_path, 'r') as f: runtime_data = json.load(f)
    except Exception as e: print(f"Error loading/parsing JSON {json_path}: {e}"); return

    try:
        global_data_df = pd.read_csv(csv_path)
    except Exception as e: print(f"Error loading CSV {csv_path}: {e}"); return
    
    if 'time' not in global_data_df.columns: print("Error: 'time' col missing in CSV"); return
    global_data_df['time'] = global_data_df['time'].astype(float)

    formulas_history_all = runtime_data.get("formulas_history", [])
    dep_vars_json = runtime_data.get("dep_vars", [])

    if not dep_vars_json: print("Error: 'dep_vars' missing/empty in JSON."); return
    if not formulas_history_all or len(formulas_history_all) != len(dep_vars_json):
        print(f"Warning: 'formulas_history' (len {len(formulas_history_all)}) mismatch with 'dep_vars' (len {len(dep_vars_json)}).")

    for pattern_id, dep_var_name in enumerate(dep_vars_json):
        if pattern_id >= len(formulas_history_all):
            print(f"Skipping pattern {pattern_id+1} ({dep_var_name}): No history array entry."); continue
        
        current_pattern_hist = formulas_history_all[pattern_id]
        if not current_pattern_hist:
            print(f"No history entries for pattern {pattern_id+1} ({dep_var_name})."); continue
        
        if dep_var_name not in global_data_df.columns:
            print(f"Warning: Dep var '{dep_var_name}' (pattern {pattern_id+1}) not in CSV. Skipping."); continue

        num_snaps = len(current_pattern_hist)
        if num_snaps == 0: continue

        max_cols = 3
        num_rows = (num_snaps + max_cols - 1) // max_cols
        num_cols = min(num_snaps, max_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols, 6*num_rows), squeeze=False, constrained_layout=True)
        fig.suptitle(f'Pattern {pattern_id+1}: {dep_var_name} - Formula History Predictions vs Actual', fontsize=16)
        axes_flat = axes.flatten()

        for snap_idx, snap_data in enumerate(current_pattern_hist):
            if snap_idx >= len(axes_flat): break
            ax = axes_flat[snap_idx]
            
            snap_time = snap_data.get("time", "N/A")
            equations_snap = snap_data.get("equations", [])

            ax.plot(global_data_df['time'], global_data_df[dep_var_name], label=f'Actual {dep_var_name}', color='black', lw=2, alpha=0.7)

            if not equations_snap:
                ax.text(0.5, 0.5, 'No formulas in snapshot', ha='center', va='center', transform=ax.transAxes)
            
            plotted_preds = False
            for eq_idx, eq_details in enumerate(equations_snap):
                if not isinstance(eq_details, dict): continue

                eq_str, eq_params = eq_details.get("equation"), eq_details.get("fitted_params")

                if eq_str and eq_params is not None: # eq_params can be an empty list
                    formula_coeffs = replace_params_in_formula(eq_str, eq_params)
                    if formula_coeffs:
                        pred_vals = evaluate_formula(formula_coeffs, global_data_df, pattern_id, INDEPVARMAP, dep_vars_json)
                        if pred_vals is not None and len(pred_vals) == len(global_data_df['time']):
                            leg_eq_str = formula_coeffs[:57] + "..." if len(formula_coeffs) > 60 else formula_coeffs
                            ax.plot(global_data_df['time'], pred_vals, label=f'F{eq_idx+1}: {leg_eq_str}', linestyle='--')
                            plotted_preds = True
            
            if not plotted_preds and equations_snap:
                 ax.text(0.5, 0.5, 'Formulas present,\nbut evaluation failed for all.', ha='center', va='center', transform=ax.transAxes, color='red')

            ax.set_title(f'Formula Gen Time: {snap_time:.1f}s', fontsize=10)
            ax.set_xlabel('Global Time (s)', fontsize=9); ax.set_ylabel('Value', fontsize=9)
            ax.legend(fontsize=7, loc='best'); ax.grid(True, linestyle=':', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=8)

        for i in range(num_snaps, len(axes_flat)): fig.delaxes(axes_flat[i])
        
        plot_fname = os.path.join(plot_output_dir, f'pattern_{pattern_id+1}_{dep_var_name}_formula_history.png')
        try: plt.savefig(plot_fname, dpi=150); print(f"Saved plot: {plot_fname}")
        except Exception as e: print(f"Error saving plot {plot_fname}: {e}")
        plt.close(fig)

    print(f"All plots saved to: {plot_output_dir}")

if __name__ == '__main__':
    main()