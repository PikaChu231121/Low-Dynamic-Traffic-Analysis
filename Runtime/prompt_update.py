SYS_MSG = """
You are an Intelligent Symbolic Regressor that predicts non-linear equations from patterns in a dataset.
The data is collected from a low-altitude vehicular network where vehicles and UAVs collaborate to complete computational tasks under limited bandwidth and computational resources.
Key indicators—such as task success ratio, V2U rate, and UAV density—are recorded at each time slot. Predictive patterns describe the relationship between next-slot outcomes and current-slot indicators.

Please provide your response in two parts. First, your analysis of the dataset should be written on a scratch pad. 
Remember, while we want better-fitted expressions, they must also be short.

The second part should consist only of your suggested expressions in LaTeX format and NO other text. 
Suppose if expressions are y1 and y2, the output is a list like this: ["y1", "y2"]

Separate the two parts with this exact string: “<EXP>”. 
The first part (analysis) must appear before “<EXP>”, and the second part (expressions) must appear after “<EXP>”.
Do NOT include any text before “<EXP>” or after the list of expressions.

"""

PAT_MSG = [
"""
We are predicting the average compute load (Pattern 1).

Inputs:
- x1 = task success ratio
- x2 = vehicle density
- x3 = UAV density
- x4 = Junction 0 vehicle count
- x5 = Junction 1 vehicle count
- x6 = Junction 2 vehicle count

""", 
"""
We are predicting the average V2U rate (Pattern 2).

Inputs:
- x1 = vehicle density
- x2 = UAV density
- x3 = average compute load

""", 
"""
We are predicting the task success ratio (Pattern 3).

Inputs:
- x1 = average V2U rate
- x2 = average V2I rate
- x3 = average compute load

""", 
"""
We are predicting the UAV density (Pattern 4).

Inputs:
- x1 = average U2I rate
- x2 = average compute load
- x3 = non-fly zone area ratio

"""
]

PAT_TPL = """
Current formula:
y = {current_formula}

At time t:
- Predicted: {predicted}
- Actual: {actual}
- MAE: {mae}

Current data:
- Dependent variable: {dep}
- Independent variables: {indep}

Your task is to generate {Neq} expressions to improve the above formula to better fit the current data.

Expressions must satisfy the following restrictions:
    - Only acceptable binary operators are limited to these four: +, -, *, and /.
    - Only acceptable unary operators are limited to these five: square, cube, sqrt, log, and exp.
    - Do not fit constants, but use c0, c1, etc.
    - Only include accessible independent variables from data. This dataset has only one, x1.

YOUR RESPONSE:

"""
