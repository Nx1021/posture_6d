import os
import re


import re
import os
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

    @classmethod
    def from_block_node_tree(cls, block_node:BlockNode):
        def build(htmlparent, block_node:BlockNode):
            htmlnode = HTMLBlockNode(block_node.name, htmlparent, block_node.type)
            for child in block_node.children:
                htmlchild = build(htmlnode, child)
            return htmlnode
        return build(None, block_node)


class _InsertLink:
    def __init__(self, pt:PackageTreeBuilder, directory:str) -> None:
        self.pt = pt
        self.pt.build()

        self.html_tree = HTMLBlockNode.from_block_node_tree(self.pt.package_tree)

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
        pattern = "<a href={link}>\n"

        # 计算link
        chain = this_node.chain.get_to_end_from(1) # 
        link = f'"#{chain}"'

        return pattern.format(link=link)

    def _get_doclink_replace(self, node:BlockNode, html_content_list:list[str], tag_line_number:int, tag_line_endnumber:int):
        pattern_whole = re.compile(r':link:<code>[^<]+</code>')
        pattern_part = re.compile(r':link:<code>([^<]+)</code>')

        for line_n in range(tag_line_number-1, tag_line_endnumber-1):
            replaced = False
            line = html_content_list[line_n]

            # 匹配所有
            all_matched = pattern_whole.findall(line)

            for matched in all_matched:
                core_str = pattern_part.findall(matched)[0]
                core_str = ChainStr(core_str)
                # 尝试在树中寻找
                get_from = node.get_parent(core_str.length)
                try:
                    rlt = get_from.get(core_str)
                except:
                    rlt = None
                if rlt:
                    # 找到了
                    link_str = self._gen_link(node, rlt)
                    replace_str = "<a href={link}><code>{core_str}</code></a>"
                    line = line.replace(matched, replace_str.format(link=link_str, core_str=core_str))
                    replaced = True
            if replaced:
                html_content_list[line_n] = line

    def _get_selflink_insert(self, node:BlockNode, html_content_list:list[str], tag_line_number:int):
        # 自身链接
        dict_ = {}

        selflink_to_insert_str = self.gen_self_link_str(node)

        front_idx = tag_line_number - 1

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

        dict_[front_idx] = selflink_to_insert_str
        dict_[end_idx] = "</a>"

        return dict_

    def _get_overwriting_insert(self, node:BlockNode, html_content_list:list[str], tag_line_number:int, h_tag_name:str):
        dict_ = {}

        to_insert_str = ""
        for n in node.overwriting_base:
            to_insert_str += self.gen_link_str(node, n, h_tag_name, "Overwriting:")
        for n in node.overwriting_root_base:
            to_insert_str += self.gen_link_str(node, n, h_tag_name, "Overwriting Root:")
        pattern = re.compile(r'<\s*/?\s*' + h_tag_name + r'\s*\\?\s*>')
        end_line_idx = tag_line_number - 1
        while True:
            line = html_content_list[end_line_idx]
            # 匹配 
            if pattern.search(line):
                break
            end_line_idx += 1

        if to_insert_str:
            dict_[end_line_idx + 1] = to_insert_str
        return dict_



    def _get_insert(self, html_content:str):
        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        html_content_list = html_content.split('\n')

        to_insert:dict[int, str] = {} # line_number: tag

        h_tag_list = soup.find_all(re.compile(r'^h[2-6]'))

        for tag_i, h_tag in enumerate(h_tag_list):
            tag_id = h_tag.get('id')
            if "--" in tag_id:
                truncate_idx = tag_id.index("--")
                tag_id = tag_id[:truncate_idx]
            tag_class = h_tag.get('class')
            tag_text = h_tag.get_text(strip=True)
            tag_line_number = h_tag.sourceline # html_content.count('\n', 0, h_tag.sourcepos[0]) + 1
            tag_line_endnumber = h_tag_list[tag_i+1].sourceline if tag_i < len(h_tag_list) - 1 else len(html_content_list) # html_content.count('\n', 0, h_tag.sourcepos[0]) + 1

            rlt = self.html_tree.get(tag_id)
            if rlt:
                # 自身链接
                _dict = self._get_selflink_insert(rlt, html_content_list, tag_line_number)
                to_insert.update(_dict)

                # 重载链接
                _dict = self._get_overwriting_insert(rlt, html_content_list, tag_line_number, h_tag.name)
                to_insert.update(_dict)

                self._get_doclink_replace(rlt, html_content_list, tag_line_number, tag_line_endnumber)

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

    def replenish_html_tree(self, html_content:str):
        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        html_content_list = html_content.split('\n')

        to_insert:dict[int, str] = {} # line_number: tag

        h_tag_list = soup.find_all(re.compile(r'^h[2-6]'))

        for tag_i, h_tag in enumerate(h_tag_list):
            tag_id = h_tag.get('id')
            if "--" in tag_id:
                truncate_idx = tag_id.index("--")
                tag_id = tag_id[:truncate_idx]
            tag_line_number = h_tag.sourceline # html_content.count('\n', 0, h_tag.sourcepos[0]) + 1
            tag_line_endnumber = h_tag_list[tag_i+1].sourceline if tag_i < len(h_tag_list) - 1 else len(html_content_list) # html_content.count('\n', 0, h_tag.sourcepos[0]) + 1

            rlt = self.html_tree.get(tag_id)
            if rlt:
                # 查找文档
                pass

        return to_insert, html_content_list
    

    def modify_html(self):
        for r,d, files in tqdm(os.walk(self.directory)):
            for file in files:
                with open(os.path.join(r, file), 'r', encoding='utf-8') as f:
                    html_content = f.read()
                self.replenish_html_tree(html_content)

        # for r,d, files in tqdm(os.walk(self.directory)):
        #     for file in files:
        #         with open(os.path.join(r, file), 'r', encoding='utf-8') as f:
        #             html_content = f.read()
        #         insert_dict, html_content_list = self._get_insert(html_content)
        #         modified_text = self._insert(html_content_list, insert_dict)
        #         with open(os.path.join(r, file), 'w', encoding='utf-8') as f:
        #             f.write(modified_text)
 

if __name__ == "__main__":
    file_name = r"search\main.js"
    search_files_pattern = r"mkdocs_.*?"
    temp_directory = r"C:\Users\nx\AppData\Local\Temp"

    recent_dir = find_recent_serve_dir(temp_directory, search_files_pattern)

    if recent_dir:
        latest_file = os.path.join(recent_dir, file_name)
        modify_main_js(latest_file)

    pt = PackageTreeBuilder(r"D:\python3_9_6\Lib\site-packages\MyLib\posture_6d")
    pt.build()

    inserter = _InsertLink(pt, os.path.join(recent_dir, "md"))
    inserter.modify_html()

