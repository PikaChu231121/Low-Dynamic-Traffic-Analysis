import time
import numpy as np
from helper import *
from fittingClass_airfog import FittingOptimizerAirFog
import matplotlib.pyplot as plt 

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.callbacks import get_openai_callback

class airFog:
    def __init__(self, dep_vars, indep_vars, N, temp, context, sys_msg, ignite_msg, iter_msg, model):
        self.dep_vars = [np.array(eval(dep_var)) for dep_var in dep_vars]
        self.indep_vars = [[np.array(eval(var)) for var in vars_group] for vars_group in indep_vars]
        
        self.dep_vars_rounded = [np.round(dep_var, 3) for dep_var in self.dep_vars]
        self.indep_vars_rounded = [[np.round(var, 3) for var in vars_group] for vars_group in self.indep_vars]

        self.results_list = [[] for _ in range(3)]
        self.N = N
        self.temp = temp
        self.model = model
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.usage_list = []
        self.all_expressions = [[] for _ in range(3)]
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizerAirFog()

        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dep1", "indep1", "dep2", "indep2", "dep3", "indep3", "Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dep1", "indep1", "dep2", "indep2", "dep3", "indep3", "ResultsAnalysis", "Neq", "context"])
        self.equation_iteration_chain = LLMChain(llm=self.llm, prompt=self.iteration_promt)

    def add_usage_data(self, cb):
        self.usage_list.append({
            'tokens': cb.total_tokens,
            'prompt-tokens': cb.prompt_tokens,
            'completion-tokens': cb.completion_tokens,
            'cost': round(cb.total_cost, 12)
        })

    def run(self, total_iterations):
        LLMrawExpressions = []
        total_chain_run_time = 0

        # Initial generation
        start_time = time.time()
        with get_openai_callback() as cb:
            StartupOutput = self.equation_generation_chain.run(
                dep1=self.dep_vars_rounded[0], indep1=self.indep_vars_rounded[0],
                dep2=self.dep_vars_rounded[1], indep2=self.indep_vars_rounded[1],
                dep3=self.dep_vars_rounded[2], indep3=self.indep_vars_rounded[2],
                Neq=self.N, context=self.context)
        end_time = time.time()           
        total_chain_run_time += end_time - start_time
        self.add_usage_data(cb)

        parts = StartupOutput.split("<EXP>")
        LLMithoughts = ''
        startupEquationsStr = ''
        startupEquations = [[]] * 3
        if len(parts) < 2:
            print("Warning: <EXP> not found or incorrectly placed in the response.")
            LLMithoughts = StartupOutput.strip()
        else:
            LLMithoughts = parts[0].strip()
            startupEquationsStr = parts[1].strip()
            startupEquations = format_and_parse_expression_matrix(startupEquationsStr)  # Parse the list of equations
        
        LLMrawExpressions.append(startupEquationsStr)
        self.all_LLMthoughts.append(LLMithoughts)
        
        valid_startupEquations = []
        for i in range(3):
            # Format and parse the nested list of equations
            formatted_equations = format_expressions(startupEquations[i] if i < len(startupEquations) else [])
            self.all_expressions[i].extend(formatted_equations)
            startup_equation_analysis = self.optimizer.fitting_constants(
                self.indep_vars[i], self.dep_vars[i], formatted_equations)
            # Prune the results whose nmae > 1
            valid_startupEquations.append([analysis for analysis in startup_equation_analysis if analysis['nmae'] < 1])
        
        # Plot the initial results
        self.plot_predictions(self.indep_vars, self.dep_vars, valid_startupEquations)
        self.results_list = [custom_sorting(results + valid_startupEquations[i]) for i, results in enumerate(self.results_list)]

        print(f"Iteration:" "Seed")
        print("SciPy feedback used for this iteration:")
        print("None")
        print("LLM thoughts:")
        print(LLMithoughts)
        print("New equations generated:")
        for i in range(3):
            print(f"- Equation {i+1}:", end=' ')
            print(self.all_expressions[i][:self.N])
        print()

        self.iteration_info.append({
            'Iteration number': 'Seed',
            'LLM Initial Thoughts': LLMithoughts,
            'New equations generated': startupEquationsStr
        })

        # Iterative refinement
        for iter_num in range(total_iterations):
            start_time = time.time()
            with get_openai_callback() as cb:
                IterOutput = self.equation_iteration_chain.run(
                    dep1=self.dep_vars_rounded[0], indep1=self.indep_vars_rounded[0],
                    dep2=self.dep_vars_rounded[1], indep2=self.indep_vars_rounded[1],
                    dep3=self.dep_vars_rounded[2], indep3=self.indep_vars_rounded[2],
                    ResultsAnalysis=self.results_list, Neq=self.N, context=self.context)
            end_time = time.time()
            total_chain_run_time += end_time - start_time
            self.add_usage_data(cb)

            parts = IterOutput.split("<EXP>")
            LLMthoughts = ''
            IterEquationsStr = ''
            IterEquations = [[]] * 3
            if len(parts) < 2:
                print("Warning: <EXP> not found or incorrectly placed in the response.")
                LLMthoughts = IterOutput.strip()
            else:
                LLMthoughts = parts[0].strip()
                IterEquationsStr = parts[1].strip()
                IterEquations = format_and_parse_expression_matrix(IterEquationsStr)  # Parse the list of equations
            
            LLMrawExpressions.append(IterEquationsStr)
            self.all_LLMthoughts.append(LLMthoughts)

            valid_iterEquations = []
            for i in range(3):
                # Format and parse the nested list of equations
                formatted_equations = format_expressions(IterEquations[i] if i < len(IterEquations) else [])
                self.all_expressions[i].extend(formatted_equations)
                iter_equation_analysis = self.optimizer.fitting_constants(
                    self.indep_vars[i], self.dep_vars[i], formatted_equations)
                # Prune the results whose nmae > 1
                valid_iterEquations.append([analysis for analysis in iter_equation_analysis if analysis['nmae'] < 1])

            # Plot the results
            self.plot_predictions(self.indep_vars, self.dep_vars, valid_iterEquations, iter_num)
            self.results_list = [custom_sorting(results + valid_iterEquations[i]) for i, results in enumerate(self.results_list)]
            
            print(f"Iteration:{iter_num+1}")
            print("SciPy feedback used for this iteration:")
            print(self.results_list)
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            for i in range(3):
                print(f"- Equation {i+1}:", end=' ')
                print(self.all_expressions[i][self.N*(iter_num+1):self.N*(iter_num+2)])
            print()
            
            self.iteration_info.append({
                'Iteration number': iter_num + 1,
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterEquationsStr
            })

        self.visualize_results(self.results_list)

        return self.results_list, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

    def visualize_results(self, final_results):
        """可视化最终结果，绘制 NMAE 与复杂度的散点图"""
        num_equations = len(final_results)
        fig, axes = plt.subplots(1, num_equations, figsize=(5 * num_equations, 5), sharey=True)
        if num_equations == 1: # 处理只有一个子图的情况
             axes = [axes]

        for i, results in enumerate(final_results):
            if not results: # 如果结果列表为空，则跳过
                axes[i].set_title(f'Equation Set {i+1} (No data)')
                axes[i].set_xlabel('Complexity')
                if i == 0:
                    axes[i].set_ylabel('NMAE')
                continue

            complexities = [r['complexity'] for r in results if r['nmae'] != float('inf')]
            maes = [r['nmae'] for r in results if r['nmae'] != float('inf')]

            if not complexities or not maes: # 如果过滤后列表为空，则跳过
                axes[i].set_title(f'Equation Set {i+1} (No valid data)')
                axes[i].set_xlabel('Complexity')
                if i == 0:
                    axes[i].set_ylabel('NMAE')
                continue

            axes[i].scatter(complexities, maes, alpha=0.6)
            axes[i].set_title(f'Equation Set {i+1}')
            axes[i].set_xlabel('Complexity')
            if i == 0:
                axes[i].set_ylabel('NMAE')
            axes[i].grid(True)

            # 添加最佳点标注 (最低 NMAE)
            if maes:
                min_mae_idx = np.argmin(maes)
                best_complexity = complexities[min_mae_idx]
                best_mae = maes[min_mae_idx]
                axes[i].scatter(best_complexity, best_mae, color='red', s=100, label=f'Best (NMAE={best_mae:.4f})', zorder=5)
                axes[i].legend()

        plt.suptitle('NMAE vs. Complexity for Fitted Equations (AirFog)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止标题重叠
        plt.show() # 显示图表

    def predict(self, indep_vars, equation, params):
        """
        根据独立变量和公式计算预测值
        :param indep_vars: 独立变量列表
        :param equation: 表达式字符串
        :param params: 拟合参数列表
        :return: 预测值列表
        """
        global_vars = {'c': params, 'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, 'min': np.minimum, 'max': np.maximum, 'movavg': lambda i, k: np.array([movavg(indep_vars[i-1][:j+1], k) for j in range(len(indep_vars[i-1]))]), **{f'x{i+1}': indep_vars[i] for i in range(len(indep_vars))}}
        local_vars = {f'x{i+1}': indep_vars_i for i, indep_vars_i in enumerate(indep_vars)}
        
        try:
            return eval(equation, global_vars, local_vars)
        except Exception as e:
            print(f"Error in evaluating equation: {equation} with params: {params}. Error: {e}")
            return np.zeros_like(indep_vars[0])  # 返回与输入大小相同的零数组以避免中断

    def plot_predictions(self, indep_vars_list, dep_vars_list, equations_list, iteration=None):
        """
        绘制每轮生成的公式的曲线图，包含 3 个子图，每个子图对应一个 pattern
        """
        num_patterns = len(equations_list)
        fig, axes = plt.subplots(num_patterns, 1, figsize=(12, 6 * num_patterns))  # 调整子图大小
        if num_patterns == 1:
            axes = [axes]  # 确保 axes 是可迭代的

        for i, ax in enumerate(axes):
            dep_var = dep_vars_list[i]
            indep_vars = indep_vars_list[i]
            equations = equations_list[i]

            ax.plot(dep_var, label="True y", color="black", linewidth=2)

            for eq_idx, equation_data in enumerate(equations):
                equation = equation_data["equation"]
                if "fitted_params" not in equation_data:
                    print(f"Warning: No fitted_params found for equation {eq_idx+1} in pattern {i+1}. Skipping.")
                    continue
                params = equation_data["fitted_params"]
                try:
                    # 计算预测值
                    y_pred = self.predict(indep_vars, equation, params)
                    ax.plot(y_pred, label=f"Eq {eq_idx+1}", linestyle="--", alpha=0.7)
                except Exception as e:
                    print(f"Error in plotting predictions for equation {eq_idx+1} in pattern {i+1}: {e}")

            ax.set_title(f"Pattern {i+1} - Iteration {iteration+1 if iteration is not None else 'Seed'}", fontsize=14)
            ax.set_xlabel("Data Index", fontsize=12)
            ax.set_ylabel("y Value", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True)

        plt.tight_layout()
        plt.suptitle(f"Predictions for Iteration {iteration+1 if iteration is not None else 'Seed'}", fontsize=18, y=1.02)
        plt.show()

    def cost(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for token in self.usage_list:
            total_prompt_tokens += token['prompt-tokens']
            total_completion_tokens += token['completion-tokens']

        if self.model == "gpt3.5-turbo":
            total_cost = (total_prompt_tokens * 0.0015 + total_completion_tokens * 0.002) / 1000
        elif self.model == "gpt-4-0613":
            total_cost = (total_prompt_tokens * 0.03 + total_completion_tokens * 0.06) / 1000
        elif self.model == "gpt-4o":
            total_cost = (total_prompt_tokens * 0.0175 + total_completion_tokens * 0.07) / 1000
        elif self.model == "chatgpt-4o-latest":
            total_cost = (total_prompt_tokens * 0.035 + total_completion_tokens * 0.105) / 1000
        elif self.model == "o3-mini":
            total_cost = (total_prompt_tokens * 0.0088 + total_completion_tokens * 0.0352) / 1000
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost
