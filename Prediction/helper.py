import json
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_mse(equation, data, fitted_params):
        num_indep_vars = data.shape[1] - 1
        x = data[:, :num_indep_vars]
        y = data[:, num_indep_vars]
        predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

        if np.isscalar(predicted_y):
            predicted_y = np.full(y.shape, predicted_y)
        mse = mean_squared_error(y, predicted_y)
        return round(mse, 8)

def calculate_mae(equation, data, fitted_params):
        num_indep_vars = data.shape[1] - 1
        x = data[:, :num_indep_vars]
        y = data[:, num_indep_vars]
        predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

        if np.isscalar(predicted_y):
            predicted_y = np.full(y.shape, predicted_y)
        mae = mean_absolute_error(y, predicted_y)
        return round(mae, 8)

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
            new_data = sorted(new_data,key=lambda x: (x['mse' if 'mse' in x else 'mae'], x['complexity']), reverse=True)
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
        

def format_and_parse_expressions(expression_string):
        expression_string = expression_string.strip()
        if expression_string.startswith('[') and expression_string.endswith(']'):
            expression_string = expression_string[1:-1].strip()

        if '\n' in expression_string:
            lines = [line.strip() for line in expression_string.split('\n') if line.strip()]
        else:
            lines = [line.strip() for line in expression_string.split(',') if line.strip()]

        parsed_expressions = []

        for line in lines:
            # Remove leading numbers (like '1.', '2.') from the line
            line = re.sub(r'^\d+\.\s*', '', line).strip()

            # Remove '\\(' and '\\)' from the start and end of the line
            line = re.sub(r'^\\\(|\\\)$', '', line).strip()
            
            line = line.strip('"').strip('\'').strip('"')

            line = line.strip().strip('"').strip('\'').strip('"').strip()
            line = line.rstrip(',').strip('"').strip('\'').strip('"')
            if not line:
                continue

            # Replace c_{0}, c_0, c_{1}, c_1, etc. with c[0], c[1], etc.
            line = re.sub(r'c_\{(\d+)\}', r'c[\1]', line)  # Handle c_{0} notation
            line = re.sub(r'c_(\d+)', r'c[\1]', line)      # Handle c_0 notation

            # Handle x1_1, x2_2, x_1, x_2, etc. by converting them to x1, x2, etc.
            line = re.sub(r'([a-zA-Z])_(\d+)', r'\1\2', line)  # Replace x1_1 with x1, x_1 with x1, etc.

            # Replace xi_(j) with xj, regardless of the value of i
            line = re.sub(r'x\d+_\((\d+)\)', r'x\1', line)  # Handle xi_(j) -> xj

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

        formatted_expressions.append(formula)

    return formatted_expressions


def format_and_parse_expression_matrix(expression_matrix_string):
    # 检查输入是否是 Python 二维列表形式
    try:
        parsed_matrix = eval(expression_matrix_string)
        if isinstance(parsed_matrix, list) and all(isinstance(row, list) for row in parsed_matrix):
            # 如果是有效的二维列表，直接格式化每个公式
            formatted_matrix = []
            for row in parsed_matrix:
                formatted_row = []
                for expression in row:
                    formatted_row.extend(format_and_parse_expressions(expression))
                formatted_matrix.append(formatted_row)
            return formatted_matrix
    except Exception:
        pass  # 如果解析失败，继续处理为 LaTeX 格式

    # 处理 LaTeX 格式的公式矩阵
    expression_matrix_string = re.sub(r'\\begin\{[a-zA-Z\*]*\}|\s*\\end\{[a-zA-Z\*]*\}', '', expression_matrix_string)  # 删除 \begin 和 \end
    expression_matrix_string = re.sub(r'\\\\', '\n', expression_matrix_string)  # 将 \\ 替换为换行符
    expression_matrix_string = expression_matrix_string.strip()

    # 替换Latex中括号为Python列表样式
    expression_matrix_string = expression_matrix_string.replace(r'\[', '[').replace(r'\]', ']')
    expression_matrix_string = expression_matrix_string.replace(r'\\[', '[').replace(r'\\]', ']')

    # 按行分割公式
    lines = [line.strip() for line in expression_matrix_string.split('\n') if line.strip()]

    # 提取公式部分，按 Pattern 分组
    expression_matrix = []
    current_row = []
    for line in lines:
        if '\\text{' in line and 'pattern' in line.lower():  # 检测包含 Pattern 的行
            if current_row:
                expression_matrix.append(current_row)
            current_row = []
        elif '=' in line:  # 仅保留包含等号的行
            if '&' in line:  # 处理 align 环境中的公式
                line = line.split('&')[-1].strip()  # 取 & 后的部分
            current_row.append(line)
    if current_row:  # 添加最后一行
        expression_matrix.append(current_row)

    # 删除 \text{...} 内容
    expression_matrix = [
        [re.sub(r'\\text\{[^}]*\}', '', expression).strip() for expression in row]
        for row in expression_matrix
    ]

    # 格式化每个公式
    formatted_matrix = []
    for row in expression_matrix:
        formatted_row = []
        for expression in row:
            formatted_row.extend(format_and_parse_expressions(expression))
        formatted_matrix.append(formatted_row)

    return formatted_matrix
