import numpy as np
import re
import json
from scipy import optimize as opt
from helper import calculate_complexity


class FittingOptimizerAirFog:
    def __init__(self):
        self.results = []
        self.equation_indices_pattern = re.compile(r'c\[(\d+)\]')

    def get_equation_indices(self, equation):
        return sorted([int(index) for index in self.equation_indices_pattern.findall(equation)], reverse=True)

    def equation_error(self, c, equation, data):
        num_indep_vars = data.shape[1] - 1
        x = data[:, :num_indep_vars]
        
        return np.mean(np.abs(eval(equation, {'c': c, 'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, **{f'x{i+1}': x[:, i] for i in range(num_indep_vars)}}) - data[:, num_indep_vars]))

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
            eval(equation, {'c': c, 'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})
            return True
        except Exception as e:
            print(f"Invalid equation: {equation}. Error: {e}")
            return False
        
    def fitting_constants(self, indep_vars, dep_var, expressions):
        results = []
        data = np.column_stack([var.reshape(-1, 1) for var in indep_vars] + [dep_var.reshape(-1, 1)])

        # Parse expressions as JSON if it's a string
        if isinstance(expressions, str):
            expressions = json.loads(expressions)
        
        for equation in expressions:
            equation_indices = self.get_equation_indices(equation)
            initial_val = [1] * len(equation_indices)
            if not self.is_valid_equation(equation, data, initial_val):
                result_dict = {'equation': equation, 'complexity': calculate_complexity(equation), 'mae': float('inf')}
                results.append(result_dict)
                continue
            try:
                result = opt.basinhopping(func=self.equation_error, x0=initial_val,
                                          minimizer_kwargs={"method": "Nelder-Mead", "args": (equation, data)})
                fitted_params = result.x
                mae = self.equation_error(fitted_params, equation, data)
                complexity = calculate_complexity(equation)
                results.append({'equation': equation, 'complexity': complexity, 'mae': mae, 'fitted_params': fitted_params.tolist()})
            except Exception as e:
                results.append({'equation': equation, 'complexity': float('inf'), 'mae': float('inf'), 'fitted_params': []})
        
        results.sort(key=lambda x: (x['mae'], x['complexity']))
        return results