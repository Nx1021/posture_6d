#!/bin/bash

# 1. 构建文档
mkdocs build

# 2. 运行custom.py
python custom.py

# 3. 进入子目录site
cd site

# 将子目录下的修改都提交
git add .
git commit -m "Update documentation"

# 4. 推送master分支到远端的gh-pages分支
git push origin master:gh-pages