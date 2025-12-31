import copy
import random
import re
import time

import numpy as np

from ...llm.interface_LLM import InterfaceLLM

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode,prompts, **kwargs):
        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking

        # -------------------- RZ: use local LLM --------------------
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_prompt_i1(self):
        
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    def get_prompt_o_index(self, indiv1, operator=0):
        prompt_content = self.prompt_task + "\n" \
"I have one algorithm with its code as follows. \
Algorithm description: " + indiv1['algorithm'] + "\n \
Code:\n \
" + indiv1['code'] + "\n" \
+ self.get_opts(operator) + "\n\
First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def _get_alg(self,prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)
                
            if n_retry > 3:
                break
            n_retry +=1

        algorithm = algorithm[0]
        code = code[0] 

        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 


        return [code_all, algorithm]

    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    # ############# code space operators
    def get_opts(self, operator):

        return "Modify the provided algorithm to improve its performance, where you can determine the degree of modification needed."

    def delete_lines(self, code, number_of_delete):
        """
        参数:
            code: 原始代码字符串
            number_of_delete: 删除比例（0-1之间的浮点数）或具体删除行数

        返回:
            new_code: 删除部分行后的代码
            number_of_delete: 实际删除的行数
        """
        lines = copy.deepcopy(code.split('\n'))
        to_be_deleted_lines_index = []  # 待删除的行索引：包括注释行和随机选择的内容行
        content_lines_index = []  # 有效内容行索引：排除函数定义、import语句、空行和注释

        # 遍历所有代码行，分类标记
        for index, line in enumerate(lines):
            stripped_line = line.strip()
            # 跳过函数定义、import语句和空行（这些是代码结构的关键部分）
            if stripped_line.startswith('def ') or stripped_line.startswith('import') or stripped_line == "":
                continue
            if stripped_line.startswith('#'):
                # 注释行标记为待删除
                to_be_deleted_lines_index.append(index)
            else:
                if "#" in line:
                    line.split('#', 1)[0].rstrip()
                # 记录有效内容行索引
                content_lines_index.append(index)

        # 根据删除比例计算实际删除的行数
        number_of_delete = np.ceil(number_of_delete * len(content_lines_index)).astype(int)
        # 从有效内容行中随机选择要删除的行
        content_delete_lines = random.choices(content_lines_index, k=min([number_of_delete, len(content_lines_index)]))
        if len(content_delete_lines) == 0:
            raise Exception("No content lines to delete")
        to_be_deleted_lines_index.extend(content_delete_lines)

        # 重构新代码，保留未被标记删除的行
        new_code = ""
        for indexI, i in enumerate(lines):
            if indexI not in to_be_deleted_lines_index:
                new_code += i
                if i != lines[-1]:
                    new_code += '\n'

        return new_code, number_of_delete

    def ts_merge_features(self, code, code_feature):
        lines = copy.deepcopy(code.split('\n'))
        to_be_deleted_lines_index = []  # "#" and selected lines
        content_lines_index = []  # not "#", "" and import and def

        for index, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith('def ') or stripped_line.startswith('import') or stripped_line == "":
                continue
            if stripped_line.startswith('#'):
                to_be_deleted_lines_index.append(index)
            else:
                if "#" in line:
                    line.split('#', 1)[0].rstrip()
                content_lines_index.append(index)

        new_code = ""
        for indexI, i in enumerate(lines[:-1]):
            if indexI not in to_be_deleted_lines_index:
                new_code += i
                new_code += '\n'

        # insert code feature
        if code_feature is not None:
            for i in code_feature:
                new_code += i
                new_code += '\n'

        new_code += lines[-1]

        return new_code

    def get_rr_opts(self, number_of_delete):

        opt = str(number_of_delete) + "lines have been removed from the provided code. Please review the code, add the necessary lines to get a better result."

        return opt

    def get_prompt_rr(self, indiv1, number_of_delete):
        """
        参数:
            indiv1: 当前个体（包含算法描述和代码）
            number_of_delete: 删除比例（0-1）

        返回:
            prompt_content: 完整的LLM提示词
        """
        # 破坏阶段：删除部分代码行
        deleted_code, deleted_lines = self.delete_lines(indiv1['code'], number_of_delete)

        # 构造提示词，包含：任务描述、原算法描述、破坏后的代码、重构指令
        prompt_content = self.prompt_task + "\n" \
        "I have one algorithm with its code as follows. \
        Algorithm description: " + indiv1['algorithm'] + "\n \
        Code:\n \
        " + deleted_code + "\n" \
        + self.get_rr_opts(deleted_lines) + "\n\
        First, describe your new algorithm and main steps in one sentence. \
        The description must be inside a brace. Next, implement it in Python as a function named \
        " + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_ts_renew(self, indiv1, ts_element):
        renew_code = self.ts_merge_features(indiv1['code'], ts_element['code_feature'])

        prompt_content = self.prompt_task + "\n" \
"I have algorithm A with its code, which inserts key lines from algorithm B's code, inserted just before the 'return' statement. \
Algorithm A description: " + indiv1['algorithm'] + "\n \
Code:\n \
" + renew_code + "\n" + \
"Algorithm B description: " + ts_element['algorithm'] + "\n \
Please review the given code, integrating two algorithm descriptions provided to rearrange it to get a better result." + "\n\
First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    # ############# proceed operators
    def o_index(self, indiv1, operator):
        prompt_content = self.get_prompt_o_index(indiv1, operator)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def rr(self, indiv1, number_of_delete):
        """

        参数:
            indiv1: 当前个体（包含算法描述和代码）
            number_of_delete: 删除比例，控制破坏程度（0-1之间）

        返回:
            code_all: LLM生成的新算法代码
            algorithm: LLM生成的新算法描述
            code_features: 新生成的代码特征（与原代码的差异行）
        """
        # 生成包含破坏代码的提示词
        prompt_content = self.get_prompt_rr(indiv1, number_of_delete)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ rr ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        # 让LLM基于破坏后的代码重新生成完整算法
        [code_all, algorithm] = self._get_alg(prompt_content)
        # 提取LLM新生成的代码行（与原代码的差异）
        code_features = self._find_llm_generate_lines(indiv1['code'], code_all)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, code_features]

    def p1(self, indiv1):
        prompt_content = self.get_prompt_o_index(indiv1, 0)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        code_features = self._find_llm_generate_lines(indiv1['code'], code_all)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, code_features]

    def p2(self, indiv1, ts_element):
        prompt_content = self.get_prompt_ts_renew(indiv1, ts_element)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        code_features = self._find_llm_generate_lines(indiv1['code'], code_all)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, code_features]

    # ##################### other tools

    def _find_llm_generate_lines(self, prev_ind_code, new_ind_code):
        p_code = prev_ind_code
        n_code = new_ind_code
        p_lines = copy.deepcopy(p_code.split('\n'))
        n_lines = copy.deepcopy(n_code.split('\n'))
        prev_content_lines = []  # not "#", "" and import and def
        new_content_lines = []  # not "#", "" and import and def

        code_features = []

        # extract effective lines
        for index, line in enumerate(p_lines):
            stripped_line = line.strip()
            if stripped_line.startswith('def ') or stripped_line.startswith(
                    'import') or stripped_line == "" or stripped_line.startswith('#'):
                continue
            else:
                if "#" in line:
                    line.split('#', 1)[0].rstrip()
                prev_content_lines.append(line.strip())

        for index, line in enumerate(n_lines):
            stripped_line = line.strip()
            if stripped_line.startswith('def ') or stripped_line.startswith(
                    'import') or stripped_line == "" or stripped_line.startswith('#'):
                continue
            else:
                if "#" in line:
                    line.split('#', 1)[0].rstrip()
                new_content_lines.append(line.rstrip())

        # compare
        for line in new_content_lines:
            if line.strip() not in prev_content_lines:
                code_features.append(line)
        return code_features
