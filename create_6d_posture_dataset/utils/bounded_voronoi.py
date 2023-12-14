import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import cv2
import skimage

def bounded_voronoi(bnd, pnts, plot = False):
    """
    计算有界的voronoi要绘制的函数。
    """

    # 添加了 3 个虚拟矩阵点，使所有矩阵点的 Voronoi 区域有界。
    bbox_size = (np.max(pnts, axis=0) - np.min(pnts, axis=0))*1000
    bbox_center = np.mean(pnts, axis=0)
    gn_pnts = np.concatenate([pnts, np.array([  [bbox_center[0] + bbox_size[0], bbox_center[1] + bbox_size[1]], 
                                                [bbox_center[0] + bbox_size[0], bbox_center[1] - bbox_size[1]], 
                                                [bbox_center[0] - bbox_size[0], 0]])])
    # voronoi計算
    vor = Voronoi(gn_pnts)

    # 将区域划分为多边形
    bnd_poly = Polygon(bnd)

    # 用于存储每个voronoi区域的列表
    vor_polys = []

    for i in range(len(gn_pnts) - 3):
        # Voronoi 区域，不考虑封闭空间
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # 计算要划分的Voronoi 区域的交点
        i_cell = bnd_poly.intersection(Polygon(vor_poly))
        # 存储考虑封闭空间的Voronoi区域的顶点坐标
        vor_polys.append(list(i_cell.exterior.coords[:-1]))
    if plot:
        # ボロノイ図の描画
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)

        # 母点
        ax.scatter(pnts[:,0], pnts[:,1])

        # ボロノイ領域
        poly_vor = PolyCollection(vor_polys, edgecolor="black",
                                facecolors="None", linewidth = 1.0)
        ax.add_collection(poly_vor)

        xmin = np.min(bnd[:,0])
        xmax = np.max(bnd[:,0])
        ymin = np.min(bnd[:,1])
        ymax = np.max(bnd[:,1])

        ax.set_xlim(xmin-0.1, xmax+0.1)
        ax.set_ylim(ymin-0.1, ymax+0.1)
        ax.set_aspect('equal')

        plt.show()
    return vor_polys

def get_seg_maps(floor_slice_map, restore_mat, scale = 1000, model_num = 9):
    filter_k = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(floor_slice_map, cv2.MORPH_CLOSE, filter_k, iterations=1)
    ### 根据第一层提取中心点，完成vorinoi图的分割
    not_image = np.logical_not(image).astype(np.uint8)
    labels_num = 0
    while np.max(skimage.measure.label(not_image)) > model_num:
        not_image = cv2.morphologyEx(not_image, cv2.MORPH_DILATE, filter_k, iterations=1)
    labels = skimage.measure.label(not_image) #分割孤立实体
    labels_num = np.max(labels)
    ptns = []
    for i in range(1, labels.max() + 1):
        area = np.sum(labels == i)
        if area < 9:
            continue
        else:
            coords = np.array(np.where(labels == i)).T
            ptns.append(np.mean(coords, axis=0))
    ptns = np.array(ptns)        
    bnd = np.array([[0, 0], [image.shape[0], 0], image.shape, [0, image.shape[1]]])
    vor_polys = bounded_voronoi(bnd, ptns)
    vor_polys_restore = []
    for vp in vor_polys:
        vp = np.array(vp)
        vp = np.hstack((vp, np.zeros((vp.shape[0], 1))))
        vp = np.hstack((vp, np.ones((vp.shape[0], 1))))
        vp_coord = restore_mat.dot(vp.T).T[:, :3] / scale
        vor_polys_restore.append(vp_coord)
    return vor_polys_restore

if __name__ == "__main__":
    # ボロノイ分割する領域
    bnd = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # 母点の個数
    n = 30
    # 母点座標
    pnts = np.random.rand(n, 2)

    # ボロノイ図の計算?描画
    vor_polys = bounded_voronoi(bnd, pnts)
    print()