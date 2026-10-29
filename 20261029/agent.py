import json
from typing import List, Dict, Tuple

from toolset import ToolRegistry

class WeeklyReportAgent:
    def __init__(self, llm, tool_registry: ToolRegistry):
        self.llm = llm
        self.tools = tool_registry
        self.conversation_history = []
        self.short_term_memory = []

    def system_prompt(self) -> str:
        tools_desc = self.tools.list_tools()
        return f"""你是一个专业的周报撰写助手。

## 你的能力
你可以通过以下工具完成复杂任务：
{tools_desc}

## 工作流程
1. 首先收集信息：读取工作记录文件
2. 分析整理：分类本周工作内容
3. 生成周报：按照标准格式撰写

## 输出格式
周报需要包含：
- 本周完成（3-5条）
- 进行中（1-3条）
- 下周计划（3-5条）
- 心得总结（1条）

## 注意事项
- 数据必须来自实际文件，不要编造
- 语言简洁专业，突出价值
- 遇到不确定的信息，主动搜索核实"""

    def think(self, user_input: str) -> str:
        """生成思考和行动"""
        context = self.build_context()
        prompt = f"""{self.system_prompt()}

当前任务：{user_input}

{context}

请进行THINK，分析当前状态，然后决定ACT（使用工具或完成）。
"""

        response = self.llm.generate(prompt)
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        return response

    def act(self, thought: str) -> Tuple[str, any]:
        """执行行动"""
        # 解析行动
        if "read_file" in thought:
            # 提取文件路径
            import re
            match = re.search(r'read_file[([^]]+)]', thought)
            if match:
                file_path = match.group(1)
                result = self.tools.get_tool("read_file").function(file_path)
                return f"已读取文件 {file_path}", result

        elif "save_file" in thought:
            import re
            match = re.search(r'save_file[([^]]+)]', thought)
            if match:
                content = match.group(1)
                # 实际实现需要从上下文中获取内容
                return "保存文件", "success"

        elif "finish" in thought or "完成" in thought:
            return "FINISH", None

        return "继续执行", None

    def run(self, initial_task: str, max_iterations: int = 10):
        """运行Agent主循环"""
        current_input = initial_task
        iteration = 0

        while iteration < max_iterations:
            # 思考
            thought = self.think(current_input)

            # 提取思考内容
            think_content = self.extract_think(thought)

            # 执行行动
            action_result = self.act(thought)

            # 记录到记忆
            self.short_term_memory.append({
                "iteration": iteration,
                "thought": think_content,
                "action_result": action_result
            })

            # 检查是否完成
            if action_result[0] == "FINISH":
                return self.generate_final_report()

            # 更新输入
            current_input = f"观察结果：{action_result[0]}"
            iteration += 1

        return "执行超时，请检查任务设置"

    def extract_think(self, response: str) -> str:
        """提取THINK部分"""
        if "THINK:" in response:
            return response.split("THINK:")[1].split("ACT:")[0]
        return response

    def generate_final_report(self) -> str:
        """生成最终周报"""
        # 汇总所有记忆
        context = "\n".join([
            f"步骤{m['iteration']}: {m['thought']}"
            for m in self.short_term_memory
        ])

        prompt = f"""基于以下执行记录，生成最终周报：

{context}

请按照标准格式输出周报内容。
"""

        return self.llm.generate(prompt)
