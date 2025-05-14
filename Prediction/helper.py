import ast
import json
import re
from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_mse(equation, data, fitted_params):
    num_indep_vars = data.shape[1] - 1
    x = data[:, :num_indep_vars]
    y = data[:, num_indep_vars]
    predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: np.array([movavg(x[:j+1, i-1].reshape(-1), k) for j in range(len(x))]), **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

    if np.isscalar(predicted_y):
        predicted_y = np.full(y.shape, predicted_y)
    mse = mean_squared_error(y, predicted_y)
    return round(mse, 8)

def calculate_mae(equation, data, fitted_params):
    num_indep_vars = data.shape[1] - 1
    x = data[:, :num_indep_vars]
    y = data[:, num_indep_vars]
    predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: np.array([movavg(x[:j+1, i-1].reshape(-1), k) for j in range(len(x))]), **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

    if np.isscalar(predicted_y):
        predicted_y = np.full(y.shape, predicted_y)
    mae = mean_absolute_error(y, predicted_y)
    return round(mae, 8)

def calculate_normalized_mse(equation, data, fitted_params):
    num_indep_vars = data.shape[1] - 1
    x = data[:, :num_indep_vars]
    y = data[:, num_indep_vars]
    predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: np.array([movavg(x[:j+1, i-1].reshape(-1), k) for j in range(len(x))]), **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

    if np.isscalar(predicted_y):
        predicted_y = np.full(y.shape, predicted_y)
    mse = mean_squared_error(y, predicted_y)
    normalized_mse = mse / (np.std(y) ** 2)
    return round(normalized_mse, 8)

def calculate_normalized_mae(equation, data, fitted_params):
    num_indep_vars = data.shape[1] - 1
    x = data[:, :num_indep_vars]
    y = data[:, num_indep_vars]
    predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: np.array([movavg(x[:j+1, i-1].reshape(-1), k) for j in range(len(x))]), **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

    if np.isscalar(predicted_y):
        predicted_y = np.full(y.shape, predicted_y)
    mae = mean_absolute_error(y, predicted_y)
    normalized_mae = mae / (np.std(y))
    return round(normalized_mae, 8)

def calculate_complexity(equation):
    binary_operator_pattern = re.compile(r'\*{2}|\*{1}|[+]|[-]|[/]')
    unary_function_pattern = re.compile(r'\b(log|exp|sqrt|sin|cos|tan|abs)\b')

    binary_operators = len(binary_operator_pattern.findall(equation))
    unary_functions = len(unary_function_pattern.findall(equation))
    
    complexity = (binary_operators * 2) + (unary_functions * 1) + 1
    
    return complexity

def custom_sorting(data):
    if isinstance(data, str):
        data = json.loads(data)
    
    if len(data) < 6:
        return data        
    else:
        new_data = data.copy()
        error_name = ''
        for err in 'nmse', 'nmae', 'mse', 'mae':
            if err in new_data[0]:
                error_name = err
                break
        new_data = sorted(new_data,key=lambda x: (x[error_name], x['complexity']), reverse=True)
        drop = 0
        i = 0
        while drop < 3 and i < len(new_data):
            current_entry = new_data[i]
            for j in range(i + 1, len(new_data)):
                if new_data[j]['complexity'] == current_entry['complexity']:
                    new_data.pop(i)
                    drop = drop + 1
                    break
            else:
                i += 1
        return new_data
        
# def is_valid_equation(equation):
#     try:
#         compile(equation, "<string>", "eval")
#         return True
#     except SyntaxError:
#         return False

def movavg(x_val_list, k):
    if len(x_val_list) < k:
        return sum(x_val_list) / len(x_val_list)
    return sum(x_val_list[-k:]) / k

def format_and_parse_expressions(expression_string: str):
    expression_string = expression_string.strip()

    # Remove wrapping brackets if present
    if expression_string.startswith('[') and expression_string.endswith(']'):
        expression_string = expression_string[1:-1].strip()

    parsed_expressions = []

    # Case 1: Try JSON-style list parsing first (if expressions are quoted)
    try:
        json_compatible = "[" + expression_string + "]"
        expressions = json.loads(json_compatible)
        lines = [str(e).strip() for e in expressions]
    except:
        # Case 2: If not JSON, try line-by-line parsing
        if '\n' in expression_string:
            lines = [line.strip() for line in expression_string.split('\n') if line.strip()]
        else:
            # As last resort, split using semicolons (preferred over commas)
            lines = [line.strip() for line in expression_string.split(';') if line.strip()]

    for line in lines:
        # Remove leading numbers (like '1.', '2.') and parentheses
        line = re.sub(r'^\d+\.\s*', '', line).strip()
        line = re.sub(r'^\\\(|\\\)$', '', line).strip()
        line = line.strip('"').strip("'").rstrip(',')

        # Convert symbolic constants to Pythonic notation: c_0 → c[0], c_{1} → c[1]
        line = re.sub(r'c_\{(\d+)\}', r'c[\1]', line)
        line = re.sub(r'c_(\d+)', r'c[\1]', line)

        # Clean up input variable notation: x_1 → x1
        line = re.sub(r'([a-zA-Z])_(\d+)', r'\1\2', line)
        line = re.sub(r'x\d+_\((\d+)\)', r'x\1', line)

        if line:
            parsed_expressions.append(line)

    return parsed_expressions


def format_expressions(expressions):
    formatted_expressions = []

    for expression in expressions:
        # Split the expression into left (variable) and right (formula) parts
        if "=" in expression:
            _, formula = expression.split("=")  # We are discarding the left side
            formula = formula.strip()
        else:
            formula = expression.strip()

        # Process the formula (right side)
        formula = formula.replace(r"\(", "(").replace(r"\)", ")")
        formula = formula.replace(r"\\(", "(").replace(r"\\)", ")")
        formula = formula.replace(r"\*", "*").replace(r"\\*", "*")
        formula = re.sub(r"\\sqrt\{([^}]+)\}", r"(\1)**0.5", formula)  # Replace \sqrt{content} with content**0.5
        formula = re.sub(r"\\cbrt\{([^}]+)\}", r"(\1)**(1/3)", formula)  # Replace \cbrt{content} with content**(1/3)
        formula = formula.replace(r"cube\_root", "**(1/3)")
        formula = formula.replace(r"\log", "log").replace(r"\exp", "exp")
        formula = formula.replace(r"\\log", "log").replace(r"\\exp", "exp")
        formula = formula.replace(r"log10", "log")
        formula = re.sub(r'cube_root\(([^)]+)\)', r'\1**(1/3)', formula)
        formula = re.sub(r'cubert\(([^)]+)\)', r'\1**(1/3)', formula)
        formula = re.sub(r'cube\(([^)]+)\)', r'\1**3', formula)
        formula = re.sub(r'square\(([^)]+)\)', r'\1**2', formula)
        formula = formula.replace("log10*", "log")
        formula = formula.replace("e**", "exp")
        formula = formula.replace("\\cdot", "*")
        formula = re.sub(r"c(\d+)", r"c[\1]", formula)  # Replace c0, c1, etc. with c[0], c[1], etc.
        formula = re.sub(r"\{([^}]+)\}", r"(\1)", formula)  # Replace { } with ( )
        formula = formula.replace("^", "**")  # Replace ^ with **
        formula = formula.replace(" ", "")  # Remove white space
        formula = re.sub(r"(?<![a-zA-Z])x(?![a-zA-Z0-9])", "x1", formula)  # Replace x with x1 if it's not followed by a digit
        formula = re.sub(r"(?<![a-zA-Z])y(?![a-zA-Z0-9])", "x2", formula)  # Replace y with x2 if it's not followed by a digit
        formula = re.sub(r"(?<![a-zA-Z])z(?![a-zA-Z0-9])", "x3", formula)  # Replace z with x3 if it's not followed by a digit
        formula = formula.replace("$", "")  # Replace $ signs if present
        formula = formula.replace('\\frac', '').replace(')(', ')/(') if 'frac' in formula else formula
        formula = formula.replace('frac', '').replace(')(', ')/(') if 'frac' in formula else formula
        formula = formula.replace(')(', ')*(')

        # Fix missing multiplication signs between constants, variables, and parentheses
        formula = re.sub(r"(\d)([a-zA-Z\(])", r"\1*\2", formula)  # Add * between number and variable/parenthesis
        formula = re.sub(r"(\))([a-zA-Z\(])", r"\1*\2", formula)  # Add * between closing parenthesis and variable/parenthesis
        formula = re.sub(r"(c\[\d+\])([a-zA-Z\(])", r"\1*\2", formula)  # Add * between c[i] and opening parenthesis
        
        # Rewrite movavg expressions
        formula = re.sub(r'movavg\(x(\d+),\s*(\d+)\)', r'movavg(\1, \2)', formula)

        formatted_expressions.append(formula)

    return formatted_expressions


def format_and_parse_expression_matrix(expression_matrix_string: str) -> List[List[str]]:
    # Step 1: 清洗 Markdown 包裹
    expression_matrix_string = expression_matrix_string.strip()
    if expression_matrix_string.startswith("```"):
        expression_matrix_string = re.sub(r"^```[a-zA-Z]*\n?", "", expression_matrix_string)
        expression_matrix_string = re.sub(r"```$", "", expression_matrix_string.strip())

    # Step 2: 替换转义字符，恢复为多行文本
    expression_matrix_string = expression_matrix_string.encode().decode('unicode_escape')

    # Step 3: 使用 ast.literal_eval 解析成嵌套列表（更安全替代 eval）
    try:
        parsed_matrix = ast.literal_eval(expression_matrix_string)
        if isinstance(parsed_matrix, list) and all(isinstance(row, list) for row in parsed_matrix):
            formatted_matrix = []
            for row in parsed_matrix:
                formatted_row = []
                for expr in row:
                    if isinstance(expr, str):
                        formatted_row.extend(format_and_parse_expressions(expr))
                formatted_matrix.append(formatted_row)
            return formatted_matrix
    except Exception as e:
        raise ValueError(f"Failed to parse expression matrix string: {e}")
