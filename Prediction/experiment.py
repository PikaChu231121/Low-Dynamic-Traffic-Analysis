import time
import numpy as np
from helper import *
from fittingClass_airfog import FittingOptimizerAirFog
from fittingClass_nmse import FittingOptimizerNMSE
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

        self.results_list = [[] for _ in range(4)]
        self.N = N
        self.temp = temp
        self.model = model
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.usage_list = []
        self.all_expressions = [[] for _ in range(4)]
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizerAirFog()

        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dep1", "indep1", "dep2", "indep2", "dep3", "indep3", "dep4", "indep4", "Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dep1", "indep1", "dep2", "indep2", "dep3", "indep3", "dep4", "indep4", "ResultsAnalysis", "Neq", "context"])
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
                dep4=self.dep_vars_rounded[3], indep4=self.indep_vars_rounded[3],
                Neq=self.N, context=self.context)
        end_time = time.time()           
        total_chain_run_time += end_time - start_time
        self.add_usage_data(cb)

        parts = StartupOutput.split("<EXP>")
        LLMithoughts = ''
        startupEquationsStr = ''
        startupEquations = [[]] * 4
        if len(parts) < 2:
            print("Warning: <EXP> not found or incorrectly placed in the response.")
            LLMithoughts = StartupOutput.strip()
        else:
            LLMithoughts = parts[0].strip()
            startupEquationsStr = parts[1].strip()
            startupEquations = format_and_parse_expression_matrix(startupEquationsStr)  # Parse the list of equations
        
        LLMrawExpressions.append(startupEquationsStr)
        self.all_LLMthoughts.append(LLMithoughts)

        for i in range(4):
            # Format and parse the nested list of equations
            formatted_equations = format_expressions(startupEquations[i] if i < len(startupEquations) else [])
            self.all_expressions[i].extend(formatted_equations)
            startup_equation_analysis = self.optimizer.fitting_constants(
                self.indep_vars[i], self.dep_vars[i], formatted_equations)
            self.results_list[i].extend(startup_equation_analysis)
            self.results_list[i] = custom_sorting(self.results_list[i])

        print(f"Iteration:" "Seed")
        print("SciPy feedback used for this iteration:")
        print("None")
        print("LLM thoughts:")
        print(LLMithoughts)
        print("New equations generated:")
        for i in range(4):
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
                    dep4=self.dep_vars_rounded[3], indep4=self.indep_vars_rounded[3],
                    ResultsAnalysis=self.results_list, Neq=self.N, context=self.context)
            end_time = time.time()
            total_chain_run_time += end_time - start_time
            self.add_usage_data(cb)

            parts = IterOutput.split("<EXP>")
            LLMthoughts = ''
            IterEquationsStr = ''
            IterEquations = [[]] * 4
            if len(parts) < 2:
                print("Warning: <EXP> not found or incorrectly placed in the response.")
                LLMthoughts = IterOutput.strip()
            else:
                LLMthoughts = parts[0].strip()
                IterEquationsStr = parts[1].strip()
                IterEquations = format_and_parse_expression_matrix(IterEquationsStr)  # Parse the list of equations
            
            LLMrawExpressions.append(IterEquationsStr)
            self.all_LLMthoughts.append(LLMthoughts)

            for i in range(4):
                # Format and parse the nested list of equations
                formatted_equations = format_expressions(IterEquations[i] if i < len(IterEquations) else [])
                self.all_expressions[i].extend(formatted_equations)
                iter_equation_analysis = self.optimizer.fitting_constants(
                    self.indep_vars[i], self.dep_vars[i], formatted_equations)
                self.results_list[i].extend(iter_equation_analysis)
                self.results_list[i] = custom_sorting(self.results_list[i])
            
            print(f"Iteration:{iter_num+1}")
            print("SciPy feedback used for this iteration:")
            print(self.results_list)
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            for i in range(4):
                print(f"- Equation {i+1}:", end=' ')
                print(self.all_expressions[i][self.N*(iter_num+1):self.N*(iter_num+2)])
            print()
            
            self.iteration_info.append({
                'Iteration number': iter_num + 1,
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterEquationsStr
            })

        # 在 run 方法结束前调用可视化
        self.visualize_results(self.results_list)

        return self.results_list, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

    def visualize_results(self, final_results):
        """可视化最终结果，绘制 MAE 与复杂度的散点图"""
        num_equations = len(final_results)
        fig, axes = plt.subplots(1, num_equations, figsize=(5 * num_equations, 5), sharey=True)
        if num_equations == 1: # 处理只有一个子图的情况
             axes = [axes]

        for i, results in enumerate(final_results):
            if not results: # 如果结果列表为空，则跳过
                axes[i].set_title(f'Equation Set {i+1} (No data)')
                axes[i].set_xlabel('Complexity')
                if i == 0:
                    axes[i].set_ylabel('MAE')
                continue

            complexities = [r['complexity'] for r in results if r['mae'] != float('inf')]
            maes = [r['mae'] for r in results if r['mae'] != float('inf')]

            if not complexities or not maes: # 如果过滤后列表为空，则跳过
                axes[i].set_title(f'Equation Set {i+1} (No valid data)')
                axes[i].set_xlabel('Complexity')
                if i == 0:
                    axes[i].set_ylabel('MAE')
                continue

            axes[i].scatter(complexities, maes, alpha=0.6)
            axes[i].set_title(f'Equation Set {i+1}')
            axes[i].set_xlabel('Complexity')
            if i == 0:
                axes[i].set_ylabel('MAE')
            axes[i].grid(True)

            # 添加最佳点标注 (最低 MAE)
            if maes:
                min_mae_idx = np.argmin(maes)
                best_complexity = complexities[min_mae_idx]
                best_mae = maes[min_mae_idx]
                axes[i].scatter(best_complexity, best_mae, color='red', s=100, label=f'Best (MAE={best_mae:.4f})', zorder=5)
                axes[i].legend()


        plt.suptitle('MAE vs. Complexity for Fitted Equations (AirFog)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止标题重叠
        plt.show() # 显示图表

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
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost


class airFogWithNMSE:
    def __init__(self, dep_vars, indep_vars, N, temp, context, sys_msg, ignite_msg, iter_msg, model):
        self.dep_vars = [np.array(eval(dep_var)) for dep_var in dep_vars]
        self.indep_vars = [[np.array(eval(var)) for var in vars_group] for vars_group in indep_vars]
        
        self.dep_vars_rounded = [np.round(dep_var, 3) for dep_var in self.dep_vars]
        self.indep_vars_rounded = [[np.round(var, 3) for var in vars_group] for vars_group in self.indep_vars]

        self.results_list = [[] for _ in range(4)]
        self.N = N
        self.temp = temp
        self.model = model
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.usage_list = []
        self.all_expressions = [[] for _ in range(4)]
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizerNMSE()

        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dep1", "indep1", "dep2", "indep2", "dep3", "indep3", "dep4", "indep4", "Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dep1", "indep1", "dep2", "indep2", "dep3", "indep3", "dep4", "indep4", "ResultsAnalysis", "Neq", "context"])
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
                dep4=self.dep_vars_rounded[3], indep4=self.indep_vars_rounded[3],
                Neq=self.N, context=self.context)
        end_time = time.time()
        total_chain_run_time += end_time - start_time
        self.add_usage_data(cb)

        parts = StartupOutput.split("<EXP>")
        LLMithoughts = ''
        startupEquationsStr = ''
        startupEquations = [[]] * 4
        if len(parts) < 2:
            print("Warning: <EXP> not found or incorrectly placed in the response.")
            LLMithoughts = StartupOutput.strip()
        else:
            LLMithoughts = parts[0].strip()
            startupEquationsStr = parts[1].strip()
            startupEquations = format_and_parse_expression_matrix(startupEquationsStr)  # Parse the list of equations
        
        LLMrawExpressions.append(startupEquationsStr)
        self.all_LLMthoughts.append(LLMithoughts)

        for i in range(4):
            # Format and parse the nested list of equations
            formatted_equations = format_expressions(startupEquations[i] if i < len(startupEquations) else [])
            self.all_expressions[i].extend(formatted_equations)
            startup_equation_analysis = self.optimizer.fitting_constants(
                self.indep_vars[i], self.dep_vars[i], formatted_equations)
            self.results_list[i].extend(startup_equation_analysis)
            self.results_list[i] = custom_sorting(self.results_list[i])

        print(f"Iteration:" "Seed")
        print("SciPy feedback used for this iteration:")
        print("None")
        print("LLM thoughts:")
        print(LLMithoughts)
        print("New equations generated:")
        for i in range(4):
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
                    dep4=self.dep_vars_rounded[3], indep4=self.indep_vars_rounded[3],
                    ResultsAnalysis=self.results_list, Neq=self.N, context=self.context)
            end_time = time.time()
            total_chain_run_time += end_time - start_time
            self.add_usage_data(cb)

            parts = IterOutput.split("<EXP>")
            LLMthoughts = ''
            IterEquationsStr = ''
            IterEquations = [[]] * 4
            if len(parts) < 2:
                print("Warning: <EXP> not found or incorrectly placed in the response.")
                LLMthoughts = IterOutput.strip()
            else:
                LLMthoughts = parts[0].strip()
                IterEquationsStr = parts[1].strip()
                IterEquations = format_and_parse_expression_matrix(IterEquationsStr)  # Parse the list of equations
            
            LLMrawExpressions.append(IterEquationsStr)
            self.all_LLMthoughts.append(LLMthoughts)

            for i in range(4):
                # Format and parse the nested list of equations
                formatted_equations = format_expressions(IterEquations[i] if i < len(IterEquations) else [])
                self.all_expressions[i].extend(formatted_equations)
                iter_equation_analysis = self.optimizer.fitting_constants(
                    self.indep_vars[i], self.dep_vars[i], formatted_equations)
                self.results_list[i].extend(iter_equation_analysis)
                self.results_list[i] = custom_sorting(self.results_list[i])
            
            print(f"Iteration:{iter_num+1}")
            print("SciPy feedback used for this iteration:")
            print(self.results_list)
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            for i in range(4):
                print(f"- Equation {i+1}:", end=' ')
                print(self.all_expressions[i][self.N*(iter_num+1):self.N*(iter_num+2)])
            print()
            
            self.iteration_info.append({
                'Iteration number': iter_num + 1,
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterEquationsStr
            })

        return self.results_list, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

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
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost
