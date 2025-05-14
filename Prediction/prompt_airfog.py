SYS_MSG = """
You are an Intelligent Symbolic Regressor that predicts non-linear equations from patterns in a dataset.

Please provide your response in two parts. First, your analysis of the dataset should be written on a scratch pad. 
Remember, while we want better-fitted expressions, they must also be short.

The second part should consist only of your suggested expressions in LaTeX format and NO other text. 
Suppose there're 2 patterns, where expressions of the first pattern are y11 and y12, and expressions of the second pattern are y21 and y22, the output is a 2-dimensional list like this: [["y11", "y12"], ["y21", "y22"]]

Separate the two parts with this exact string: “<EXP>”. 
The first part (analysis) must appear before “<EXP>”, and the second part (expressions) must appear after “<EXP>”.
Do NOT include any text before “<EXP>” or after the list of expressions.

"""

IGNITE = """
Your job is to find expressions for 3 patterns that approximately describe the dataset. For each pattern, find {Neq} expressions that approximately describe the pattern. 
For the first pattern, the dependent variable is y: {dep1} and independent variables are (x1, x2, x3, x4, x5, x6): {indep1}. 
For the second pattern, the dependent variable is y: {dep2} and independent variables are (x1, x2, x3): {indep2}. 
For the third pattern, the dependent variable is y: {dep3} and independent variables are (x1, x2, x3): {indep3}. 

{context}

Expressions must satisfy the following restrictions:
    - Only acceptable binary operators are limited to these four: +, -, *, and /.
    - Only acceptable unary operators are limited to these five: square, cube, sqrt, log, and exp.
    - Do not fit constants, but use c0, c1, etc.
    - Only include accessible independent variables from data.

YOUR RESPONSE:
"""

ITER = """
Based on your previous suggestions, here is an analysis of the accuracy (measured by Normalized Mean Absolute Error) and complexity Pareto front:

{ResultsAnalysis}

Suggest {Neq} new equations for each 3 patterns minimizing both complexity and loss. Diverse ones are likely to be helpful. 
Here's the dataset:
Dependent variable of the first pattern is y: {dep1} 
Independent variables of the first pattern are (x1, x2, x3, x4, x5, x6): {indep1}.
Dependent variable of the second pattern is y: {dep2}
Independent variables of the second pattern are (x1, x2, x3): {indep2}.
Dependent variable of the third pattern is y: {dep3}
Independent variables of the third pattern are (x1, x2, x3): {indep3}.

{context}

Expressions must satisfy the following restrictions:
    - Only acceptable binary operators are limited to these four: +, -, *, and /.
    - Only acceptable unary operators are limited to these five: square, cube, sqrt, log, and exp.
    - Additionally, you may use the following compound functions:
        - min(x, y, ...): minimum of x, y, ...
        - max(x, y, ...): maximum of x, y, ...
        - movavg(x, k): moving average of variable x over the past k time steps (e.g., k = 3)
    - Do not fit constants, but use c0, c1, etc.
    - Only include accessible independent variables from data.
    - Do not suggest SR-similar expressions to avoid redundant expressions.

Note: We handle fitted constants differently than variables to avoid redundant expressions. 
Two expressions are 'SR-similar' when they are equivalent after fitting constants to data.  
For example: - c0/(x1-c1) & c0/(x1+c1) are SR-similar because sign of a constant can be absorbed after fitting
             - x1*(c0+c1) & x1*c0 are SR-similar because c0 and c1 can be consolidated into one fitted constant
             - c0/(x1*c1) & c0/x1 are SR-similar because c0 and c1 can be consolidated into one fitted constant


YOUR RESPONSE:
"""