SYS_MSG = """
You are an Intelligent Symbolic Regressor that predicts and improves symbolic equations based on recent feedback from a real-time vehicular edge computing system.

The data is collected from a low-altitude vehicular network where vehicles and UAVs collaborate to complete computational tasks under limited bandwidth and computational resources. 
Key indicators—such as task success ratio, V2U rate, and UAV density—are recorded at each time slot. 
Each predictive pattern captures the relationship between current-slot inputs and next-slot outputs.

You are now asked to revise and improve an existing symbolic expression using recent prediction feedback. 
You will be shown the current expression, the predicted value, the actual value, and their absolute error. 
When possible, try to improve the expression by modifying or extending the current structure. 
Avoid unnecessary drastic rewrites. The new expressions must be short, symbolic, and better fitted.

Please provide your response in two parts:
1. First, write your analysis of the dataset and your reasoning process on a scratch pad.
2. Second, provide only your suggested expressions in LaTeX format and NO other text. Suppose if expressions are y1 and y2, the output is a list like this: ["y1", "y2"]

Separate the two parts using the exact string: <EXP>
Do NOT include any explanation after <EXP>, and do NOT include any text outside the expression list.

"""


PAT_TPL = [
"""
We are predicting the average compute load.

Inputs:
- x1 = task success ratio
- x2 = vehicle density
- x3 = UAV density
- x4 = Junction 0 vehicle count
- x5 = Junction 1 vehicle count
- x6 = Junction 2 vehicle count

Current formula:
y = {current_formula}

At time t:
- Predicted: {predicted}
- Actual: {actual}
- MAE: {mae}

Recent samples:
- Dependent variable: {dep}
- Independent variables: {indep}

Your task is to generate {Neq} expressions to improve the current formula to better fit the current data.

Expressions must satisfy the following restrictions:
    - Only acceptable binary operators are limited to these four: +, -, *, and /.
    - Only acceptable unary operators are limited to these five: square, cube, sqrt, log, and exp.
    - Additionally, you may use the following compound functions:
        - movavg(x, k): moving average of variable x over the past k time steps (x can only be one variable instead of an expression)
    - Do not fit constants, but use c0, c1, etc.
    - Only include accessible independent variables from data. This dataset has only one, x1.

Note: The target y shows complex and oscillating behavior.

You may consider using:
- Polynomial terms (square(x), cube(x))
- Logarithmic and exponential structures
- Variable differences: (x1 - x2)
- Local trend smoothing: movavg(x3, 3)

YOUR RESPONSE:

""", 
"""
We are predicting the average V2U rate.

Inputs:
- x1 = vehicle density
- x2 = UAV density
- x3 = average compute load

Current formula:
y = {current_formula}

At time t:
- Predicted: {predicted}
- Actual: {actual}
- MAE: {mae}

Recent samples:
- Independent variables: {indep}
- Dependent variable: {dep}

Your task is to generate {Neq} expressions to improve the current formula to better fit the current data.

Expressions must satisfy the following restrictions:
    - Only acceptable binary operators are limited to these four: +, -, *, and /.
    - Only acceptable unary operators are limited to these five: square, cube, sqrt, log, and exp.
    - Do not fit constants, but use c0, c1, etc.
    - Only include accessible independent variables from data. This dataset has only one, x1.

YOUR RESPONSE:

""", 
"""
We are predicting the task success ratio.

Inputs:
- x1 = average V2U rate
- x2 = average V2I rate
- x3 = average compute load

Current formula:
y = {current_formula}

At time t:
- Predicted: {predicted}
- Actual: {actual}
- MAE: {mae}

Recent samples:
- Independent variables: {indep}
- Dependent variable: {dep}

Your task is to generate {Neq} expressions to improve the current formula to better fit the current data.

Expressions must satisfy the following restrictions:
    - Only acceptable binary operators are limited to these four: +, -, *, and /.
    - Only acceptable unary operators are limited to these five: square, cube, sqrt, log, and exp.
    - Additionally, you may use the following compound functions:
        - min(x, y, ...): minimum of x, y, ...
        - max(x, y, ...): maximum of x, y, ...
    - Do not fit constants, but use c0, c1, etc.
    - Only include accessible independent variables from data. This dataset has only one, x1.

Note: The target y is a success ratio and must be in the range [0, 1].

You are encouraged to use structures that naturally output bounded values, such as:
- Sigmoid: y = 1 / (1 + exp(-(…)))
- Bounded min/max: y = min(1, max(0, …))

Your task is to revise the current formula to reduce error while ensuring y ∈ [0, 1].

YOUR RESPONSE:

"""
]
