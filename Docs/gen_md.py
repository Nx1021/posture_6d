import os
import yaml

from typing import Any

# 生成md文件
# 生成yml文件的导航栏

def load_yaml(path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as file:
        # 使用Loader=FullLoader来保持键的顺序
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def dump_yaml(path, data:dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as file:
        # 使用default_flow_style=False和sort_keys=False来保持键的顺序
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

def is_subdirectory(directory, potential_parent):
    relative_path = os.path.relpath(directory, potential_parent)
    return not relative_path.startswith('..') and not os.path.isabs(relative_path)


class MkdocsCreator:
    def __init__(self, package_path) -> None:
        self.package_path = package_path  

        self.Docs = os.path.join(self.package_path, 'Docs') 
        self.Docs_css = os.path.join(self.Docs, 'css')
        self.config_yml = os.path.join(self.Docs, 'mkdocs.yml')

        self.config = load_yaml(self.config_yml)

    def add_one_file(self, root, file):
        if not file.endswith('.py'):
            return
        file_name = file.split('.')[0]
        modules = os.path.relpath(root, self.package_path).split(os.sep)
        print(file_name, modules)

    def create_mkdocs(self):
        for r, d, f in os.walk(self.package_path):
            if r == self.Docs or is_subdirectory(r, self.Docs):
                continue
            for file in f:
                self.add_one_file(r, file)

if __name__ == "__main__":
    package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creator = MkdocsCreator(r"D:\python3_9_6\Lib\site-packages\MyLib\posture_6d")
    creator.create_mkdocs()