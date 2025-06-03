import numpy as np
import re
import json
from scipy import optimize as opt
from helper import calculate_complexity, movavg


class FittingOptimizerAirFog:
    def __init__(self):
        self.results = []
        self.equation_indices_pattern = re.compile(r'c\[(\d+)\]')

    def get_equation_indices(self, equation):
        return sorted([int(index) for index in self.equation_indices_pattern.findall(equation)], reverse=True)

    def equation_error(self, c, equation, data):
        num_indep_vars = data.shape[1] - 1
        x = data[:, :num_indep_vars]
        y = data[:, num_indep_vars]
        
        return np.mean(np.abs(eval(equation, {'c': c, 'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: np.array([movavg(x[:j+1, i-1].reshape(-1), k) for j in range(len(x))]), **{f'x{i+1}': x[:, i] for i in range(num_indep_vars)}}) - y)) / (np.max(y) - np.min(y))

    def is_valid_equation(self, equation, data, c):
        try:
            # 检查是否为有效的 Python 表达式
            compile(equation, "<string>", "eval")

            # 检查是否包含不允许的操作符
            undesired_patterns = [r"sin", r"cos", r"tan"]
            if any(re.search(pattern, equation) for pattern in undesired_patterns):
                return False
        
            # 检查常量索引是否有效
            equation_indices = self.get_equation_indices(equation)
            if any(int(index) >= len(c) for index in equation_indices):
                return False
        
            # 检查是否可以执行
            num_indep_vars = data.shape[1] - 1
            x = data[:, :num_indep_vars]
            eval(equation, {'c': c, 'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: np.array([movavg(x[:j+1, i-1].reshape(-1), k) for j in range(len(x))]), **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})
            return True
        except Exception as e:
            print(f"Invalid equation: {equation}. Error: {e}")
            return False
        
    def fitting_constants(self, indep_vars, dep_var, expressions):
        results = []

        # Prepare data: each column is one variable
        try:
            data = np.column_stack([var.reshape(-1, 1) for var in indep_vars] + [dep_var.reshape(-1, 1)])
        except Exception as e:
            print("Data column stack failed:", e)
            return []

        # Parse expressions if in string form
        if isinstance(expressions, str):
            expressions = json.loads(expressions)

        for equation in expressions:
            equation_indices = self.get_equation_indices(equation)
            num_constants = len(equation_indices)

            # Get all used constant indices, e.g., [0, 1, 2]
            equation_indices = self.get_equation_indices(equation)
            num_constants = len(equation_indices)
            
            if num_constants == 0:
                nmae = self.equation_error([], equation, data)
                complexity = calculate_complexity(equation)
                results.append({
                    'equation': equation,
                    'complexity': complexity,
                    'nmae': nmae,
                    'fitted_params': []  # 没有参数
                })
                continue

            # Improved bound logic: infer per constant index
            bounds = []
            for idx in equation_indices:
                # 查找每个 c[idx] 周围是否有敏感操作
                local_expr = re.findall(rf"(exp|log|sqrt)\s*\([^\)]*c\[{idx}\][^\)]*\)", equation)
                if local_expr:
                    bounds.append((-5.0, 5.0))
                else:
                    bounds.append((-10.0, 10.0))

            # Fallback for safety
            if len(bounds) != num_constants or any(len(b) != 2 for b in bounds):
                print(f"[Warning] Invalid bounds for equation: {equation}, fallback to default")
                bounds = [(-10.0, 10.0)] * num_constants

            # Check validity
            initial_val = [1.0] * num_constants
            if not self.is_valid_equation(equation, data, initial_val):
                continue

            # Perform optimization
            try:
                result = opt.differential_evolution(
                    func=lambda c: self.equation_error(c, equation, data),
                    bounds=bounds,
                    strategy='best1bin',
                    maxiter=500,
                    tol=1e-6,
                    polish=True
                )
                fitted_params = result.x
                nmae = self.equation_error(fitted_params, equation, data)
                complexity = calculate_complexity(equation)

                results.append({
                    'equation': equation,
                    'complexity': complexity,
                    'nmae': nmae,
                    'fitted_params': fitted_params.tolist()
                })

            except Exception as e:
                print(f"Optimization failed for equation: {equation}, error: {e}")
                results.append({
                    'equation': equation,
                    'complexity': float('inf'),
                    'nmae': float('inf'),
                    'fitted_params': []
                })

        # Sort by NMAE then complexity
        results.sort(key=lambda x: (x['nmae'], x['complexity']))
        return results