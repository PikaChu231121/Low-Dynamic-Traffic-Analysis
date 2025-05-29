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
    normalized_mae = mae / (y.max() - y.min())
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

    # Case 1: Try Python-style list parsing first (if expressions are quoted)
    try:
        safe_expr_string = re.sub(r'\\', r'\\\\', expression_string)
        expressions = ast.literal_eval(safe_expr_string)
        lines = [str(e).strip() for e in expressions]
    except:
        # Case 2: If not Python expression, try line-by-line parsing
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

        # Clean up input variable notation: x_1 → x1, x_{1} → x1
        line = re.sub(r'([a-zA-Z])_(\d+)', r'\1\2', line)
        line = re.sub(r'x\d+_\((\d+)\)', r'x\1', line)
        line = re.sub(r'x_\{(\d+)\}', r'x\1', line)
        line = re.sub(r'x_(\d+)', r'x\1', line)

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
        formula = formula.replace(r'\,', '')
        formula = re.sub(r"\\sqrt\{([^}]+)\}", r"(\1)**0.5", formula)  # Replace \sqrt{content} with content**0.5
        formula = re.sub(r"\\cbrt\{([^}]+)\}", r"(\1)**(1/3)", formula)  # Replace \cbrt{content} with content**(1/3)
        formula = formula.replace(r"cube\_root", "**(1/3)")
        formula = formula.replace(r"\log", "log").replace(r"\exp", "exp").replace(r"\min", "min").replace(r"\max", "max")
        formula = formula.replace(r"\\log", "log").replace(r"\\exp", "exp").replace(r"\\min", "min").replace(r"\\max", "max")
        formula = re.sub(r'\\Bigl\s*\(', '(', formula)
        formula = re.sub(r'\\Bigr\s*\)', ')', formula)
        formula = re.sub(r'\\bigl\s*\(', '(', formula)
        formula = re.sub(r'\\bigr\s*\)', ')', formula)
        formula = formula.replace(r"log10", "log")
        formula = re.sub(r'cube_root\(([^)]+)\)', r'\1**(1/3)', formula)
        formula = re.sub(r'cubert\(([^)]+)\)', r'\1**(1/3)', formula)
        formula = re.sub(r'cube\(([^)]+)\)', r'\1**3', formula)
        formula = re.sub(r'square\(([^)]+)\)', r'\1**2', formula)
        formula = formula.replace("log10*", "log")
        formula = formula.replace("e**", "exp")
        formula = formula.replace("\\cdot", "*")
        formula = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', formula)
        formula = re.sub(r"c(\d+)", r"c[\1]", formula)  # Replace c0, c1, etc. with c[0], c[1], etc.
        formula = re.sub(r"\{([^}]+)\}", r"(\1)", formula)  # Replace { } with ( )
        formula = formula.replace("^", "**")  # Replace ^ with **
        formula = re.sub(r"(?<![a-zA-Z])x(?![a-zA-Z0-9])", "x1", formula)  # Replace x with x1 if it's not followed by a digit
        formula = re.sub(r"(?<![a-zA-Z])y(?![a-zA-Z0-9])", "x2", formula)  # Replace y with x2 if it's not followed by a digit
        formula = re.sub(r"(?<![a-zA-Z])z(?![a-zA-Z0-9])", "x3", formula)  # Replace z with x3 if it's not followed by a digit
        formula = formula.replace(" ", "")  # Remove white space
        formula = formula.replace("$", "")  # Replace $ signs if present
        formula = formula.replace('\\tfrac', '').replace(')(', ')/(') if 'tfrac' in formula else formula
        formula = formula.replace('\\frac', '').replace(')(', ')/(') if 'frac' in formula else formula
        formula = formula.replace('frac', '').replace(')(', ')/(') if 'frac' in formula else formula
        formula = formula.replace(')(', ')*(')
        formula = re.sub(r'x\d+_\{(\d+)\}', r'x\1', formula)

        # Fix missing multiplication signs between constants, variables, and parentheses
        formula = re.sub(r"(\d)([a-zA-Z\(])", r"\1*\2", formula)  # Add * between number and variable/parenthesis
        formula = re.sub(r"(\))([a-zA-Z\(])", r"\1*\2", formula)  # Add * between closing parenthesis and variable/parenthesis
        formula = re.sub(r"(c\[\d+\])([a-zA-Z\(])", r"\1*\2", formula)  # Add * between c[i] and opening parenthesis
        formula = re.sub(r'(x\d)([a-zA-Z])', r'\1*\2', formula) # Add * between variable and another variable
        formula = re.sub(r'([a-zA-Z])(x\d)', r'\1*\2', formula) # Add * between variable and variable
        
        # Rewrite movavg expressions
        formula = formula.replace(r"\movavg", "movavg").replace(r"\\movavg", "movavg")
        formula = re.sub(r'movavg\(x(\d+),\s*(\d+)\)', r'movavg(\1, \2)', formula)

        formatted_expressions.append(formula)

    return formatted_expressions

def format_and_parse_expression_matrix(expression_matrix_string: str):
    expression_matrix_string = expression_matrix_string.strip()

    # 清除 Markdown 包裹
    if expression_matrix_string.startswith("```"):
        expression_matrix_string = re.sub(r"^```[a-zA-Z]*\n?", "", expression_matrix_string)
        expression_matrix_string = re.sub(r"```$", "", expression_matrix_string.strip())

    # 判断是否为 LaTeX array 结构
    if r'\begin{array}' in expression_matrix_string:
        return extract_latex_arrays(expression_matrix_string)
    
    # 判断是否为 LaTeX bmatrix 结构
    if r'\begin{bmatrix}' in expression_matrix_string:
        return extract_latex_bmatrix(expression_matrix_string)
    
    # 判断是否为 LaTeX align 结构
    if r'\begin{align' in expression_matrix_string:
        return extract_latex_align_patterns(expression_matrix_string)

    # 尝试解析为 JSON / Python 格式
    try:
        parsed = ast.literal_eval(expression_matrix_string)
        if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
            return [
                [parsed_expr for expr in row for parsed_expr in format_and_parse_expressions(expr)]
                for row in parsed
            ]
    except Exception:
        pass

    raise ValueError(f"Unsupported expression format:\n{expression_matrix_string}")

def extract_latex_arrays(expr_str: str) -> List[List[str]]:
    expr_str = expr_str.strip()

    # 替换 LaTeX 换行符为真实换行
    expr_str = expr_str.replace('\\\\', '\n')

    # 匹配每个 \begin{array} ... \end{array}
    array_blocks = re.findall(
        r'\\left\[\s*\\begin\{array\}\{[lcr]+\}(.*?)\\end\{array\}\s*\\right\]',
        expr_str,
        re.DOTALL
    )

    matrix = []
    for block in array_blocks:
        parsed_row = format_and_parse_expressions(block)
        matrix.append(parsed_row)

    return matrix

def extract_latex_bmatrix(expr_str: str) -> List[List[str]]:
    expr_str = expr_str.strip()

    # 清洗多余 LaTeX 包裹
    expr_str = expr_str.replace('\\\\', '\n')
    expr_str = expr_str.replace(r'\,', ',')  # 恢复逗号
    expr_str = re.sub(r'\\begin\{bmatrix\}|\s*\\end\{bmatrix\}', '', expr_str)
    expr_str = re.sub(r'\\\[|\\\]', '', expr_str)

    # 匹配每行：包含 Pattern N 与 \left[ ... \right]
    rows = re.findall(r'\\text\{Pattern\s+\d+:[^}]*\}\s*&\s*\\left\[([^\]]+)\\right\]', expr_str, re.DOTALL)

    matrix = []
    for row in rows:
        expressions = []
        # 提取每个 \text{"..."} 内容
        raw_items = re.findall(r'\\text\{"([^"]+)"\}', row)
        expressions = format_and_parse_expressions(str(raw_items).replace('\\\\', '\\'))
        matrix.append(expressions)

    return matrix

def extract_latex_align_patterns(expr_str: str) -> List[List[str]]:
    expr_str = expr_str.strip()

    # 清洗换行符等
    expr_str = expr_str.replace('\\\\', '\n')
    expr_str = expr_str.replace(r'\,', ',')  # LaTeX spacing comma

    # 替换 \frac{a}{b} → (a)/(b)
    expr_str = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1) / (\2)', expr_str)

    # 匹配每个 Pattern 组
    pattern_blocks = re.split(r'\\text\{Pattern\s+\d+:}', expr_str)
    patterns = []

    for block in pattern_blocks:
        if not block.strip():
            continue
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        row = []

        for line in lines:
            # 提取 `y = ...` 右边内容
            match = re.match(r'&\s*y\d+\s*=\s*(.+)', line)
            if match:
                expr = match.group(1).strip().rstrip(';').strip()
                row.extend(format_and_parse_expressions(expr))
        if row:
            patterns.append(row)

    return patterns
