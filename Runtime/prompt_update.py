SYS_MSG = """
You are an Intelligent Symbolic Regressor that predicts and improves symbolic equations based on recent feedback from a real-time vehicular edge computing system.

The data is collected from a low-altitude vehicular network where vehicles and UAVs collaborate to complete computational tasks under limited bandwidth and computational resources. 
Key indicators—such as task success ratio, V2U rate, and UAV density—are recorded at each time slot. 
Each predictive pattern captures the relationship between current-slot inputs and next-slot outputs.

You are now asked to revise and improve an existing symbolic expression using recent prediction feedback. 
You will be shown the current expression, the predicted value, the actual value, and their mean absolute error. You are also provided with the approximate value range of the predicted variable to help you judge the severity of this MAE.
When possible, try to improve the expression by modifying or extending the current structure. 
Avoid unnecessary drastic rewrites. The new expressions must be short, symbolic, and better fitted.

Please provide your response in two parts:
1. First, write your analysis of the dataset and your reasoning process on a scratch pad.
2. Second, provide only your suggested expressions and NO other text. Suppose if expressions are y1 and y2, the output is a list like this: ["y1", "y2"]

Separate the two parts using the exact string: <EXP>
Do NOT include any explanation after <EXP>, and do NOT include any text outside the expression list.

"""


PAT_TPL = [
"""
We are predicting the average compute load of the next time slot.

Inputs:
- x1 = current task success ratio
- x2 = current vehicle density
- x3 = current average V2U rate
- x4 = current average compute load (i.e., y_t)
- x5 = previous average compute load (i.e., y_(t-1))

Current formula:
y = {current_formula}

At time t:
- Predicted: {predicted}
- Actual: {actual}
- MAE of last {window_size} predictions: {mae}
- Range of y: {y_range}

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
    - Only include accessible independent variables from data, which are x1, x2, x3, x4 and x5.

Note: The target y shows complex and oscillating behavior.

You may consider using the follwing, but always AVOID UNNECESSARY DRASTIC REWRITES:
- Polynomial terms (e.g., x1², x2³)
- Exponential and logarithmic transformations
- Cross-variable interactions (e.g., x1 * x3, x2 / x5)
- Temporal modeling elements:
  - Moving average: movavg(x1, 3)
  - Residual dynamics: x1 - x2
  - State memory: x4, x5, (x4 - x5)

YOUR RESPONSE:

""", 
"""
We are predicting the average V2U rate of the next time slot.

Inputs:
- x1 = vehicle density
- x2 = UAV density
- x3 = average compute load

Current formula:
y = {current_formula}

At time t:
- Predicted: {predicted}
- Actual: {actual}
- MAE of last {window_size} predictions: {mae}
- Range of y: {y_range}

Recent samples:
- Independent variables: {indep}
- Dependent variable: {dep}

Your task is to generate {Neq} expressions to improve the current formula to better fit the current data.

Expressions must satisfy the following restrictions:
    - Only acceptable binary operators are limited to these four: +, -, *, and /.
    - Only acceptable unary operators are limited to these five: square, cube, sqrt, log, and exp.
    - Do not fit constants, but use c0, c1, etc.
    - Only include accessible independent variables from data, which are x1, x2 and x3.

YOUR RESPONSE:

""", 
"""
We are predicting the task success ratio of the next time slot.

Inputs:
- x1 = average V2U rate
- x2 = average V2I rate
- x3 = average compute load

Current formula:
y = {current_formula}

At time t:
- Predicted: {predicted}
- Actual: {actual}
- MAE of last {window_size} predictions: {mae}
- Range of y: {y_range}

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
    - Only include accessible independent variables from data, which are x1, x2 and x3.

Note: The target y is a success ratio and must be in the range [0, 1]. Empirical observations show:
- When bandwidth is insufficient, task success sharply declines.
- When compute load nears saturation, task success drops abruptly.
- The y-value (success ratio) non-linearly changes with respect to x1–x3.

You may consider using the follwing, but always AVOID UNNECESSARY DRASTIC REWRITES:
- Sigmoid-like structures to model saturating effects:  
  `y = 1 / (1 + exp(-...))`  
  `y = min(1, max(0, ...))`
- Load penalty: include inverse terms like `1 / (x3 + c)` or log-shifted forms like `log(x3 + 1)`
- Bandwidth interplay: try cross-variable interactions such as `x1 * x2` or `x1 / (x2 + 1)`
- Saturation indicators: consider residuals or thresholding functions for compute load
- Simpler alternatives: linear combinations, squares or square roots of x1/x2 if it captures the trend
- Avoid always returning expressions that collapse to ~0.5.

Your task is to revise the current formula to reduce error while ensuring y ∈ [0, 1].

YOUR RESPONSE:

"""
]
