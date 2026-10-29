from dataclasses import dataclass
from typing import List,Dict,Callable
@dataclass
class Tool:
    name:str
    descriptions:str
    function:Callable
    parameters:Dict
class ToolRegistry:
    def __init__(self):
        self.tools:List[Tool] = []
    def register(self,name:str,description:str,func:Callable,params:List[str]):
        self.tools.append(Tool(name,description,func,params))
    def get_tool(self,name:str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ValueError(f'Tool{name} not found')
    def list_tools(self) -> str:
        return "\n".join([
            f"-{t.name}:{t.descriptions}(参数:{','.join(t.parameters)})"
            for t in self.tools
        ])
registry = ToolRegistry()
registry.register(
    name="read_file",
    description='读取本地文件内容',
    func=read_file_content,
    params=['file_path']
)
registry.register(
    name='search_web',
    description='搜索互联网获取信息',
    func=search_intent,
    params=['query']
)
registry.register(
    name='save_file',
    description='保存内容到本地文件',
    func=save_to_file,
    params=['content','file_path']
)