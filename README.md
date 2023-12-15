# posture_6d
version 0.0.0

## Api docs
Detailed code documentation (under update) <br>
详细的代码文档（更新中）<br>
https://nx1021.github.io/posture_6d/

## sub modules

### create_6d_posture
创建6d姿态估计的数据集

### core
姿态运算
- intr.py 相机内参
- posture.py SE(3)的运算
- utils.py 相关的一些库函数、数据结构

### data
数据集管理器，可以实现对多类型数据的批量化读取。
- dataCluster.py 数据簇，管理同种类型的数据
- dataset.py 数据集，管理多个数据簇
- spliter.py 分割器，实现数据集的多种分割方式
- IOAbstract.py 抽象基类
- mesh_manager.py 三角网格文件的管理器

- dataset_example.py 一些数据集示例
- viewmeta.py 单帧视角元数据，包含色彩图、深度图、内外参、掩膜等数据

## usage

1. dataCluster

读数据<br>
<code>
from posture_6d.data.dataCluster import UnifiedFileCluster<br>
dc = UnifiedFileCluster("your/dataCluster/path")<br>
x = dc[0] # 支持使用索引<br>
x = dc.read(0) #使用read<br>
for x in dc[:10]:<br>
...# 逐个读取<br>
</code>

写数据
<code>
with dc.get_writer() as writer:<br>
    dc.write(0, ...)<br>
    dc.append(...)<br>
with dc.get_writer().allow_overwriting() as writer:<br>
    dc.write(0, ...) # 覆盖已有数据<br>
</code>

2. dataset
    ...


## AUTHORS
nx