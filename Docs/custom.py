import os
import re
import argparse

from typing import Optional
from bs4 import BeautifulSoup
from MyLib.packagetree import PackageTreeBuilder, BlockNode, ChainStr
from tqdm import tqdm


def modify_main_js(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'function formatResult\s{0,1}\(.*?\) *\{.*?\}', re.DOTALL)

    # 修改 formatResult 函数
    content = re.sub(
        pattern,  # 正则匹配 formatResult 函数
        r'''function formatResult(location, title, summary) {
  return '<article><h3><a href="' + joinUrl(base_url, location) + '">' + escapeHtml(title) + '</a></h3><p class="search-location"><a href="' + joinUrl(base_url, location) + '">Location: ' + escapeHtml(location) + '</a></p><p>' + escapeHtml(summary) + '</p></article>';
}''',
        content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def find_recent_serve_dir(directory, pattern):
    # 匹配符号条件的最新的目录
    matching_dirs = []
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if re.match(pattern, d):
                matching_dirs.append((os.path.join(root, d), os.path.getmtime(os.path.join(root, d))))
    
    if matching_dirs:
        latest_dir = max(matching_dirs, key=lambda x: x[1])
        return latest_dir[0]

def find_and_modify_latest_file(directory, pattern, file_name):
    recent_dir = find_recent_serve_dir(directory, pattern)

    if recent_dir:
        latest_file = os.path.join(recent_dir, file_name)
        modify_main_js(latest_file)

class HTMLBlockNode(BlockNode):
    def __init__(self, name: str, parent: Optional[BlockNode] = None, node_type=BlockNode.UNKNOWN, lazy=False) -> None:
        super().__init__(name, parent, node_type, lazy)

        self.start_line_idx = 0
        self.end_line_idx = 0

        self.doc_start_line_idx = 0
        self.doc_end_line_idx = 0

        self.html_name = ""

    @classmethod
    def from_block_node_tree(cls, block_node:BlockNode):
        def build(block_node:BlockNode):
            block_node.__setattr__("start_line_idx", 0)
            block_node.__setattr__("end_line_idx", 0)
            block_node.__setattr__("doc_start_line_idx", 0)
            block_node.__setattr__("doc_end_line_idx", 0)
            block_node.__setattr__("html_name", "")
            for child in block_node.children:
                build(child)
            return block_node
        return build(block_node)


class _InsertLink:
    def __init__(self, pt:PackageTreeBuilder, directory:str) -> None:
        self.pt = pt
        self.pt.build()

        self.html_tree:HTMLBlockNode = HTMLBlockNode.from_block_node_tree(self.pt.package_tree)

        self.directory = directory
    
    def _gen_link(self, this_node:BlockNode, ref_node:BlockNode):
        # 计算link
        this_node_module = this_node.get_module()
        ref_node_module = ref_node.get_module()
        if this_node_module == ref_node_module:
            # 在同一module下，直接使用chain
            chain = ref_node.chain.get_to_end_from(1) # 
            link = f'"#{chain}"'
        else:
            # 不在同一module下，chain前需要添加路径
            cur_file_path = this_node_module.chain.replace('.', '/')
            ref_file_path = ref_node_module.chain.replace('.', '/')
            relative_path = os.path.relpath(ref_file_path, cur_file_path)
            relative_path.replace('\\', '/')
            relative_path = os.path.join(relative_path, "index.html")
            chain = ref_node.chain.get_to_end_from(1)
            link = f'"{relative_path}#{chain}"'
        
        return link

    def gen_link_str(self, this_node:BlockNode, ref_node:BlockNode, tag_id:str, type_word:str):
        pattern = "<a href={link} class=\"{tag_id}-link\">{indent}Go to {type}: {ref_chain}</a><br>\n"

        # 计算link
        link = self._gen_link(this_node, ref_node)
        
        # 计算indent
        indent = ""#this_node.chain.length - this_node_module.chain.length

        # ref_chain
        ref_chain = ref_node.chain.get_to_end_from(1)

        return pattern.format(link=link, tag_id = tag_id, indent=indent, type=type_word, ref_chain=ref_chain)

    def gen_self_link_str(self, this_node:BlockNode):
        pattern = "<a href={link}>"

        # 计算link
        chain = this_node.chain.get_to_end_from(1) # 
        link = f'"#{chain}"'

        return pattern.format(link=link)

    def _get_doclink_replace(self, node:BlockNode, html_content_list:list[str], tag_line_idx:int, tag_line_endidx:int):
        pattern_whole = re.compile(r':link:<code>[^<]+</code>')
        pattern_part = re.compile(r':link:<code>([^<]+)</code>')

        dict_ = {}

        for line_n in range(tag_line_idx, tag_line_endidx+1):
            replaced = False
            if line_n >= len(html_content_list):
                break
            line = html_content_list[line_n]

            # 匹配所有
            all_matched = pattern_whole.findall(line)

            # while (matched := pattern_whole.search(line)) is not None: # TODO: 重复的
            # matched = matched.group(0)
            for matched in all_matched:
                core_str = pattern_part.findall(matched)[0]
                core_str = ChainStr(core_str)
                # 尝试在树中寻找
                get_from = node.get_parent(core_str.length)
                try:
                    rlt:HTMLBlockNode = get_from.get(core_str)
                except:
                    rlt = None
                if rlt is None:
                    # 尝试在module下查找
                    module_node = node.get_module()
                    try:
                        rlt:HTMLBlockNode = module_node.get(core_str)
                    except:
                        rlt = None
                if rlt:
                    # 找到了
                    link_str = self._gen_link(node, rlt)
                    replace_str = "<a href={link}><code>{core_str}</code></a>"
                    line = line.replace(matched, replace_str.format(link=link_str, core_str=core_str))
                    # 在被链接的doc中添加返回的链接
                    insert_idx = rlt.doc_end_line_idx
                    # 确保不重复添加
                    link_str = self._gen_link(rlt, node)
                    insert_str = "<a href={link}><p>see <code>{core_str}</code></p></a>".format(link=link_str, core_str=node.chain.get_to_end_from(1))

                    if insert_idx not in dict_ or (insert_idx in dict_ and insert_str not in dict_[insert_idx]):
                        self.update_to_insert_dict(dict_, {insert_idx: insert_str})
                    replaced = True
            if replaced:
                html_content_list[line_n] = line

        return dict_

    def _get_selflink_insert(self, node:BlockNode, html_content_list:list[str], tag_line_idx:int):
        # 自身链接
        dict_ = {}

        selflink_to_insert_str = self.gen_self_link_str(node)

        front_idx = tag_line_idx

        span_start_pattern = re.compile(r'<\s*/?\s*span')
        span_end_pattern = re.compile(r'<\s*/?\s*/span\s*\\?\s*>')

        while True:
            line = html_content_list[front_idx]
            # 匹配 
            if span_start_pattern.search(line):
                break
            front_idx += 1

        end_idx = front_idx
        while True:
            line = html_content_list[end_idx]
            # 匹配 
            if span_end_pattern.search(line):
                break
            end_idx += 1
        end_idx += 1

        self.update_to_insert_dict(dict_, {front_idx: selflink_to_insert_str})
        self.update_to_insert_dict(dict_, {end_idx: "</a>"})

        return dict_

    def _get_overwriting_insert(self, node:BlockNode, html_content_list:list[str], tag_line_idx:int, h_tag_name:str):
        dict_ = {}

        to_insert_str = ""
        for n in node.overwriting_base:
            to_insert_str += self.gen_link_str(node, n, h_tag_name, "Overwriting:")
        for n in node.overwriting_root_base:
            to_insert_str += self.gen_link_str(node, n, h_tag_name, "Overwriting Root:")
        pattern = re.compile(r'<\s*/?\s*' + h_tag_name + r'\s*\\?\s*>')
        end_line_idx = tag_line_idx
        while True:
            line = html_content_list[end_line_idx]
            # 匹配 
            if pattern.search(line):
                break
            end_line_idx += 1

        if to_insert_str:
            self.update_to_insert_dict(dict_, {end_line_idx + 1: to_insert_str})
        return dict_



    def _get_insert(self, module_node:HTMLBlockNode, html_content:str):
        html_content_list = html_content.split('\n')

        to_insert:dict[int, str] = {} # line_number: tag

        for node in module_node.walk():
            node:HTMLBlockNode
            if node.type & (BlockNode.MODULE| BlockNode.IMPORT):
                continue
            if node.html_name == "":
                continue
            # 自身链接
            _dict = self._get_selflink_insert(node, html_content_list, node.start_line_idx)
            self.update_to_insert_dict(to_insert, _dict)

            # 重载链接
            _dict = self._get_overwriting_insert(node, html_content_list, node.start_line_idx, node.html_name)
            self.update_to_insert_dict(to_insert, _dict)

            _dict = self._get_doclink_replace(node, html_content_list, node.doc_start_line_idx, node.doc_end_line_idx)
            self.update_to_insert_dict(to_insert, _dict)

        return to_insert, html_content_list

    def _replace(self, original_text: str, replace_dict: dict[int, str]):
        lines = original_text.splitlines()

        for line_number, replacement_text in replace_dict.items():
            if line_number < len(lines):
                lines[line_number] = replacement_text

        modified_text = '\n'.join(lines)
        return modified_text

    def _insert(self, html_content_list:str, insert_dict:dict[int, str]):
        html_content_list

        # 将插入操作按照行号进行排序
        sorted_insertions = sorted(insert_dict.items())

        offset = 0  # 用于跟踪插入操作引起的行号偏移

        for line_number, insertion_text in sorted_insertions:
            insertion_text_list = insertion_text.splitlines()
            for text in insertion_text_list:
                # 插入字符串到行号对应的位置
                html_content_list.insert(line_number + offset, text)
                offset += 1  # 每次插入后，更新偏移

        # 重新组合文本字符串
        modified_text = '\n'.join(html_content_list)
        return modified_text

    @staticmethod
    def update_to_insert_dict(to_insert:dict[int, str], dict_:dict):
        for key, value in dict_.items():
            if key in to_insert:
                to_insert[key] += value + '\n'
            else:
                to_insert[key] = value + '\n'


    def replenish_html_tree(self, html_content:str):
        """
        补充html_tree的信息
        """
        def match_endtag(tag, start):
            nonlocal html_content_list
            stack = []  # 用于跟踪打开的标签
            for i in range(start, len(html_content_list)):
                line = html_content_list[i]

                # 查找开始标签
                start_tags = [m.group(0) for m in re.finditer(fr"<{tag}(?:\s[^>]*)?>", line)]
                for start_tag in start_tags:
                    stack.append((i, start_tag))

                # 查找结束标签
                end_tags = [m.group(0) for m in re.finditer(fr"</{tag}>", line)]
                for end_tag in end_tags:
                    if stack:
                        stack.pop()
                        if len(stack) == 0:
                            return i
                    else:
                        return i  # 返回结束标签所在的行号
            if len(stack) == 0:
                return i  # 返回结束标签所在的行号

        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        html_content_list = html_content.split('\n')

        to_insert:dict[int, str] = {} # line_number: tag
        
        h_tag_list = soup.find_all(re.compile(r'^h[2-6]'))
        doc_tag_list = soup.find_all('div', class_="doc doc-contents")

        tags_list = h_tag_list + doc_tag_list
        # 重新根据sourceline排序
        tags_list = sorted(tags_list, key=lambda x: x.sourceline) # 每个title对应1个或0个doc

        for tag_i, tag in enumerate(tags_list):
            if tag.name == "div":
                pass
            else:
                tag_id = tag.get('id')

                rlt:HTMLBlockNode = self.html_tree.get(tag_id)
                if not rlt:
                    # 查找文档
                    continue

                start_idx = tag.sourceline - 1
                end_idx = match_endtag(tag.name, start_idx)

                # 尝试获取下个doctagt
                next_tag = tags_list[tag_i+1] if tag_i < len(tags_list) - 1 else None
                    
                if next_tag and next_tag.name == "div":
                    doc_start_idx = next_tag.sourceline - 1
                    if rlt.type & BlockNode.CLASS:
                        doc_end_idx = doc_start_idx
                        while True:
                            if doc_end_idx >= len(html_content_list):
                                break
                            line = html_content_list[doc_end_idx]
                            if "doc doc-children" in line:
                                break
                            doc_end_idx += 1
                    else:
                        doc_end_idx = match_endtag(next_tag.name, doc_start_idx)
                else:
                    doc_start_idx = doc_end_idx = None

                # 更新节点
                rlt.start_line_idx = start_idx
                rlt.end_line_idx = end_idx
                rlt.doc_start_line_idx = doc_start_idx
                rlt.doc_end_line_idx = doc_end_idx
                rlt.html_name = tag.name

        return to_insert, html_content_list
    

    def modify_html(self):
        for r,d, files in tqdm(os.walk(self.directory)):
            for file in files:
                with open(os.path.join(r, file), 'r', encoding='utf-8') as f:
                    html_content = f.read()
                self.replenish_html_tree(html_content)

        for r, d, files in tqdm(os.walk(self.directory)):
            for file in files:
                with open(os.path.join(r, file), 'r', encoding='utf-8') as f:
                    html_content = f.read()
                module_dir = os.path.relpath(r, self.directory)
                module_dir = module_dir.replace(os.sep, '.')

                module_node = self.html_tree.get(module_dir)

                insert_dict, html_content_list = self._get_insert(module_node, html_content)
                modified_text = self._insert(html_content_list, insert_dict)
                with open(os.path.join(r, file), 'w', encoding='utf-8') as f:
                    f.write(modified_text)

        # for r,d, files in tqdm(os.walk(self.directory)):
        #     for file in files:
        #         with open(os.path.join(r, file), 'r', encoding='utf-8') as f:
        #             html_content = f.read()
        #         insert_dict, html_content_list = self._get_insert(html_content)
        #         modified_text = self._insert(html_content_list, insert_dict)
        #         with open(os.path.join(r, file), 'w', encoding='utf-8') as f:
        #             f.write(modified_text)
 
if __name__ == "__main__":
    search_js_file_name = r"search\main.js"

    file_directory = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='An example script with command line arguments.')

    # 添加命令行参数
    parser.add_argument('--treedir', type=str, default = os.path.normpath(os.path.join(file_directory, "../")), help='First argument')
    parser.add_argument('--mt', type=bool, default = True, help='Second argument (optional)') # modify temp

    args = parser.parse_args()

    tree_dir:str = args.treedir
    mt:bool = args.mt

    if not mt:
        directory = file_directory
        directory = os.path.join(directory, "site")
    else:
        search_files_pattern = r"mkdocs_.*?"
        temp_directory = r"C:\Users\nx\AppData\Local\Temp"

        directory = find_recent_serve_dir(temp_directory, search_files_pattern)

    if directory:
        latest_file = os.path.join(directory, search_js_file_name)
        modify_main_js(latest_file)

    pt = PackageTreeBuilder(tree_dir)
    pt.build()

    inserter = _InsertLink(pt, os.path.join(directory, "md"))
    inserter.modify_html()

