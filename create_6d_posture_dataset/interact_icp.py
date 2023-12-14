
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d import geometry, utility, io
import os
import cv2
import json
import scipy.spatial as spt
from sko.GA import GA
import sys
import glob
import re
import json
from scipy.spatial import ConvexHull
import shapely.geometry

from typing import Union
from . import Posture, SphereAngle, MeshMeta, Voxelized, CameraIntr, CALI_INTR_FILE, calc_masks
from .data_manager import DataRecorder, FrameMeta, ModelManager, ProcessData
from .utils import homo_pad
from .pcd_creator import ProcessData

class MyGA(GA):
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=1, lb=-1, ub=1, constraint_eq=[], constraint_ueq=[], precision=1e-7, early_stop=None, result_precision = 0.01):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, lb, ub, constraint_eq, constraint_ueq, precision, early_stop)
        self.result_precision = result_precision

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        best = []
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    deltas = np.abs(np.array(best) - min(best))
                    if np.all(deltas < self.result_precision):
                        break
                    else:
                        best.pop(0)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

    fit = run

class ModelInfo(MeshMeta):
    def __init__(self, mesh, name="", class_id=-1, barycenter:np.ndarray = None) -> None:
        super().__init__(mesh, None, None, None, None, name, class_id)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = mesh.vertices
        distances = self.pcd.compute_nearest_neighbor_distance()
        self.avg_dist = np.mean(distances)  # 点云密度
        self.mesh.compute_vertex_normals()
        # 重心
        if barycenter is None:
            self.barycenter = self.calc_barycenter(self.avg_dist*3)
        else:
            self.barycenter = barycenter

    def calc_barycenter(self, voxel_size):
        voxelized = Voxelized.from_mesh(self.mesh, voxel_size)
        body_points = voxelized.entity_indices
        body_points = np.hstack((body_points, np.ones((body_points.shape[0], 1))))
        body_points = voxelized.restore_mat.dot(body_points.T).T
        barycenter = np.mean(body_points[:, :3], axis=0)
        return barycenter
    
    def transform(self, posture: Union[Posture, np.ndarray], copy=True):
        return super().transform(posture, copy)

class ModelBalencePosture():
    POINT_BALENCE_TOL = 0.5 * np.pi / 180
    POINT_STABLE_TOL = 0.1 * np.pi / 180
    def __init__(self) -> None:
        # self.std_meshes_dir = os.path.join(os.path.abspath(os.path.join(directory, "..")), "dense_models")
        # self.std_models_list = [x for x in os.listdir(self.std_meshes_dir) if os.path.splitext(x)[-1] == ".ply"]

        self.current_mi:ModelInfo = None
        # self.model_list = [] # 模型点云列表
        # for name in std_models_list:
        #     print(name)
        #     path = os.path.join(std_meshes_dir, name)
        #     mi = ModelInfo(path)
        #     self.model_list.append(mi)
    
    def get_bary_vec(self, mi:ModelInfo, index):
        point = mi.points_array[index]
        bary_vec = point - mi.barycenter
        bary_vec = (bary_vec.T/np.linalg.norm(bary_vec, axis=-1)).T
        return bary_vec

    def get_normal(self, mi:ModelInfo, index):
        normal = mi.normals_array[index].copy()
        return normal

    @staticmethod
    def _angle(vce1, vec2):
        vce1 = np.array(vce1).squeeze()
        vec2 = np.array(vec2).squeeze()
        angle = np.arccos(np.sum(vce1 * vec2, axis=-1))
        return angle

    @staticmethod
    def proj_vec(vec, n, normal = True):
        if len(vec.shape) == 1:
            vec = np.expand_dims(vec, 0) #[1, 3]
        n_expand = np.tile(np.expand_dims(n, 0), [vec.shape[0],1])
        proj = vec - (n_expand * np.sum(vec * n_expand, axis = -1, keepdims=True))
        if normal:
            proj = (proj.T/np.linalg.norm(proj, axis=-1)).T
        return proj.squeeze()
    
    @staticmethod
    def get_drop_foot(point, line_p1, line_p2):
        vector1 = point - line_p1
        vector2 = point - line_p2
        vector3 = line_p2 - line_p1

        k = np.dot(line_p1 - point, line_p2 - line_p1)
        k /= -np.square(np.linalg.norm(vector3))
        dropFoot = k * (vector3) + line_p1
        return dropFoot

    @staticmethod
    def find_all_connected(tri, seed_idx, valid_idx_array):
        connected = np.array([], dtype=np.int32)
        new = np.array([seed_idx], dtype=np.int32)
        while new.size != 0:
            tem_tri = np.tile(np.expand_dims(tri, -1), [1, 1, new.size])
            tri_idx = np.where(np.any(tem_tri == new, axis=(1,2)))[0] #相邻的三角面的序号
            tmp_new = np.unique(np.hstack(tri[tri_idx]))
            new = np.intersect1d(tmp_new, valid_idx_array) #必须是有效点
            new = np.setdiff1d(new, connected) #减去已有点
            connected = np.append(connected, new)
        return connected

    @staticmethod
    def point_distance_line(points, lines_point1, lines_point2):
        '''
        points : [P, 3]         -> [P, L, 3]
        lines_point1: [L, 3]    -> [P, L, 3]
        '''
        points = np.array(points)
        lines_point1 = np.array(lines_point1)
        lines_point2 = np.array(lines_point2)
        if len(points.shape) == 1:
            P = 1
            points = np.expand_dims(points, axis=0)
        else:
            P = points.shape[0]
        if len(lines_point1.shape) == 1:
            L = 1
            lines_point1 = np.expand_dims(lines_point1, axis=0)
            lines_point2 = np.expand_dims(lines_point2, axis=0)
        else:
            L = lines_point1.shape[0]
        points = np.tile(np.expand_dims(points, 1), [1, L, 1])
        lines_point1 = np.tile(np.expand_dims(lines_point1, 0), [P, 1, 1])
        lines_point2 = np.tile(np.expand_dims(lines_point2, 0), [P, 1, 1])
        #计算向量
        vec1 = lines_point1 - points
        vec2 = lines_point2 - points
        distance =  np.linalg.norm(np.cross(vec1,vec2), axis=-1) /\
                    np.linalg.norm(lines_point1-lines_point2, axis=-1)
        return distance #[P, L]

    def fall(self, mi:ModelInfo, contact_dict:dict, step = 1.0/180*np.pi):
        '''
        step 步长
        '''
        line_contact = False
        points = []
        lines = []
        contact_index = [int(x) for x in list(contact_dict.keys())]
        contact_neighbor_index = list(contact_dict.values())
        in_poly = False
        if len(contact_index) == 1:
            index = contact_index[0]
            points = mi.points_array[index]
            points = np.expand_dims(points, 0)
            lines = np.zeros((0,2,3))
        elif len(contact_index) == 2:
            points = mi.points_array[(contact_index[0], contact_index[1]), :]
            line = (mi.points_array[contact_index[0]] , mi.points_array[contact_index[1]])
            lines = np.array([line])
        else:
            contact_points = mi.points_array[contact_index]
            if contact_points.shape[0] >= 4:
            # 三个以及以上
                hull = ConvexHull(contact_points[:, :2]) # 求凸包
                hull_points = contact_points[hull.vertices][:, :2] #[x,y]
            else:
                hull_points = contact_points[:,:2] #[HP, 2]
            # 给点排序，确保是正续
            hull_center = np.mean(hull_points, axis=0)
            angles = np.arctan2(hull_points[:,1] - hull_center[1], 
                                hull_points[:,0] - hull_center[0])
            angles[angles<0] += 2*np.pi
            points = hull_points[np.argsort(angles)]
            points = np.hstack((points, np.full((points.shape[0],1), contact_points[:,2].min())))
            k, b = np.polyfit(points[:,0], points[:,1], 1)   
            A, B, C = k, -1, b
            distance = np.abs(A * points[:, 0] + B * points[:, 1] + C) / (np.sqrt(A**2 + B**2))        
            if distance.max() < 1e-3:
                line_contact = True
            poly_context = {'type': 'MULTIPOLYGON',
                                        'coordinates': [[points.tolist()]]}
            poly_shape = shapely.geometry.shape(poly_context)
            bary = mi.barycenter[:2]
            bary_point = shapely.geometry.Point(bary[0], bary[1])
            if poly_shape.intersects(bary_point):
                in_poly = True
                return np.eye(4), np.array([0.0, 0, 0]) #稳定
            else:
                lines = []
                for pi in range(len(points)):
                    next_pi = (pi + 1) % len(points)
                    lines.append((points[pi], points[next_pi]))
                lines = np.array(lines)

        ### 首先检查重心是否在连线中间，如有，还要判断该线是否在近侧
        barycenter_on_floor = np.array([mi.barycenter[0], mi.barycenter[1], points[:,2].min()])
        norm_lines =  lines[:,0,:] - lines[:,1,:] # [N,3]
        norm_lines = (norm_lines.T / np.linalg.norm(norm_lines, axis=-1)).T
        points_bary_distance = ModelBalencePosture.point_distance_line( points, 
                                                                        mi.barycenter, 
                                                                        mi.barycenter + np.array([0,0,-1])).squeeze(axis=1)
        lines_bary_distance = \
            ModelBalencePosture.point_distance_line(barycenter_on_floor, 
                                                    lines[:,0,:], 
                                                    lines[:,1,:]).squeeze(axis=0)
        # 判断是否是线接触
        rot_lines_idx = []
        if line_contact:
            end1_idx = np.argmin(points[:,0])
            end2_idx = np.argmax(points[:,0])
            rot_line = points[(end1_idx, end2_idx), :]
        else:            
            for line_i in range(len(lines)):
                norm_line = norm_lines[line_i]
                dir0 = np.sum(self.get_bary_vec(mi, contact_index[0]) * norm_line)
                dir1 = np.sum(self.get_bary_vec(mi, contact_index[1]) * norm_line)
                if dir0 * dir1 > 0:
                    continue #同号，重心不在范围内   
                if lines_bary_distance[line_i] > np.min(points_bary_distance):
                    continue
                else:
                    rot_lines_idx.append(line_i)
            if len(rot_lines_idx) > 1:
                rot_line = lines[rot_lines_idx[np.argmin(lines_bary_distance[rot_lines_idx])]]
            elif len(rot_lines_idx) == 0:
                rot_point = points[np.argmin(points_bary_distance)]
                # 绕点旋转
                p2b = mi.barycenter - rot_point
                rot_line = np.cross(p2b, np.array([0,0,-1.0]))
                rot_norm = rot_line/np.linalg.norm(rot_line)
                rot_line = np.array([rot_point, rot_point + rot_norm])
            else:
                rot_line = lines[rot_lines_idx[0]]
        # 转轴方向校验
        vec1 = rot_line[1] - rot_line[0]
        vec2 = barycenter_on_floor - rot_line[1]
        cross = np.cross(vec1, vec2)
        if cross[2] > 0:
            rot_line = rot_line[::-1]
        # 绕直线旋转
        rot_norm = rot_line[1,:] - rot_line[0,:]
        rot_norm = rot_norm/np.linalg.norm(rot_norm)
        # bary_center_proj = ModelBalencePosture.proj_vec(mi.barycenter, rot_line[0], rot_line[1])
        drop_foot = ModelBalencePosture.get_drop_foot(mi.barycenter, rot_line[0], rot_line[1])
        distance = ModelBalencePosture.point_distance_line(drop_foot, 
                                                            mi.barycenter, 
                                                            mi.barycenter + np.array([0,0,-1])).squeeze()
        rot_angle = distance/0.01 * step
        posture_r = Posture(rvec=rot_angle * rot_norm)
        posture_t = Posture(tvec=drop_foot)
        T = np.linalg.multi_dot((posture_t.trans_mat, posture_r.trans_mat, posture_t.inv_transmat))
        # 试旋转，防止另一侧触点超过平面
        homo_points = np.hstack((mi.points_array, np.ones((mi.points_array.shape[0], 1))))
        while True:
            try_fall_points = T.dot(homo_points.T).T[:, :3]
            min_index = np.argmin(try_fall_points[:, 2])
            min_z_point_org = mi.points_array[min_index]
            min_z_point_new = try_fall_points[min_index]
            drop_foot_z = drop_foot[2]            
            if min_z_point_new[2] - drop_foot_z >= -5e-5:
                break
            # 判断是否在垂足的对侧
            drop_foot_porj = ModelBalencePosture.proj_vec(drop_foot, rot_norm, normal = False)
            baryline_porj = ModelBalencePosture.proj_vec(   barycenter_on_floor,
                                                            rot_norm, normal = False)
            min_z_point_org_porj = ModelBalencePosture.proj_vec(min_z_point_org, rot_norm, normal = False)
            min_z_point_new_porj = ModelBalencePosture.proj_vec(min_z_point_new, rot_norm, normal = False)
            if (min_z_point_org_porj[0] - baryline_porj[0]) * (drop_foot_porj[0] - baryline_porj[0]) > 0:
                break
            R = np.linalg.norm(min_z_point_org_porj - drop_foot_porj)
            h = min_z_point_org[2] - drop_foot_z
            rot_angle = np.arcsin(h/R)
            posture_r = Posture(rvec=rot_angle * rot_norm)
            posture_t = Posture(tvec=drop_foot)
            T = np.linalg.multi_dot((posture_t.trans_mat, posture_r.trans_mat, posture_t.inv_transmat))
        # T = posture_t.inv_transmat
        return T, rot_norm

    def place(self, mi:ModelInfo):
        stable = False
        iter = 0
        max_iter = 100
        step = 1
        last_rot_norm = np.array([0,0,1])
        while iter < max_iter and not stable:
            contact_dict = self.get_contact_point(mi)
            T, rot_norm = self.fall(mi, contact_dict, step/180*np.pi)
            if np.all(T == np.eye(4)):
                stable = True    
            step = max((np.sum(last_rot_norm * rot_norm) + 1) / 2, 0.02)
            last_rot_norm = rot_norm
            iter += 1
            yield T

    def get_contact_point(self, mi:ModelInfo):
        z_thre = 5e-5
        all_points = mi.points_array.copy()
        all_tri = np.array(mi.mesh.triangles) #[TRI, 3]
        z = all_points[:,2]
        low_points_global_idx = np.where(z-z.min() < z_thre)[0]
        low_points = all_points[low_points_global_idx].copy()
        # 进行连续片分割，每个连续片至少有一个最小值
        contact_dict = {}        
        for p in low_points_global_idx:
            contact_dict.update({int(p): []})
        return contact_dict
        lowest_local_idx = np.argmin(low_points[:,2])
        lowest_point = low_points[lowest_local_idx] #最低点坐标
        lowest_global_idx = low_points_global_idx[lowest_local_idx]   
        # # low_points_idx = np.setdiff1d(low_points_idx, lowest_global_idx)
        # upper = z.max()
        # neighbor_threshold = 0
        # while np.sum(low_points[:,2] < upper) > neighbor_threshold:
        #     lower_local_idx = np.argmin(low_points[:,2])
        #     lower_point = low_points[lower_local_idx] #最低点坐标
        #     lower_global_idx = low_points_global_idx[lower_local_idx]
        #     neighbor_local_idx = self.get_neighbors(lower_point, low_points, max(mi.avg_dist*1.5, 0.002)) #附近点坐标
        #     if neighbor_local_idx.size > neighbor_threshold:
        #         neighbor_global_idx = low_points_global_idx[neighbor_local_idx]
        #         neighbor_points = all_points[neighbor_global_idx]
        #         neighbor_global_idx = np.setdiff1d(neighbor_global_idx, lower_global_idx)
        #         contact_dict.update({lower_global_idx: neighbor_global_idx})  
        #         upper = min(upper, max(np.min(neighbor_points[:, 2])+1e-5, np.max(neighbor_points[:, 2])))
        #         low_points[low_points[:,2] > upper] += 100 # 超过上界的点也不再计算
        #         # neighbor_global_idx = connected_global[neighbor_local_idx]
        #     low_points[neighbor_local_idx, 2] += 100 # 这些点不再参与后续计算

        return contact_dict

    def get_neighbors(self, point:np.ndarray, point_list:np.ndarray, r):
        dist = np.linalg.norm(point_list - point, axis=-1)
        return np.where(dist < r)[0]

    # def normal_filter(self, normal_angle):
    #     mean = np.mean(normal_angle)
    #     std = np.std(normal_angle)
    #     ok_index = np.where((normal_angle < mean + 3*std) * (normal_angle > mean - 3*std))[0]
    #     return ok_index

    def if_point_stable(self, mi:ModelInfo, contact_dict:dict):
        '''
        点是否稳定，计算该点的夹角、附近点的夹角
        '''
        contact_index = list(contact_dict.keys())
        contact_neighbor_index = list(contact_dict.values())
        # if len(contact_index) > 3:
        #     colors = np.zeros(mi.points_array.shape)
        #     colors[np.hstack(contact_neighbor_index)] = np.array([255, 0, 0])
        #     colors[contact_index] = np.array([0, 0, 255])
        #     mi.pcd.colors = utility.Vector3dVector(colors)
        #     o3d.visualization.draw_geometries([mi.pcd], width=1280, height=720) 
        if len(contact_index) == 1:
            ### 首先判断该点是否平衡
            index = contact_index[0]
            neighbor_index = np.array(contact_neighbor_index[0]).squeeze()
            bary_vec = self.get_bary_vec(mi, index) #重心到该点的向量
            normal = self.get_normal(mi, index) #该点法向
            angle = np.arccos(np.sum(bary_vec * normal))
            if angle > self.POINT_BALENCE_TOL:
                return False
            ### 判断该位置是否是稳定的
            bary_angle = self._angle(self.get_bary_vec(mi, neighbor_index), [0,0,-1])
            normal_angle = self._angle(self.get_normal(mi, neighbor_index), [0,0,-1])
            # ok_idx = np.argsort(normal_angle)[:int( (normal_angle.size-1)*0.8)]
            # bary_angle = bary_angle[ok_idx]
            # normal_angle = normal_angle[ok_idx]
            if np.any(normal_angle > bary_angle + self.POINT_STABLE_TOL):
                return False #确保是稳定平衡
        elif len(contact_index) == 2:
            line = mi.points_array[contact_index[0]] - mi.points_array[contact_index[1]]
            line = line / np.linalg.norm(line)
            neighbor_index = np.hstack(contact_neighbor_index).squeeze()
            ### 两点连线
            # 先判断重心是否在两点之间，两点的重心线在连线上的投影应该反向
            dir0 = np.sum(self.get_bary_vec(mi, contact_index[0]) * line)
            dir1 = np.sum(self.get_bary_vec(mi, contact_index[1]) * line)
            if dir0 * dir1 > 0:
                return False #同号，重心不在范围内
            # 判断稳定性
            # 向以连线为法向的平面投影
            proj_bary_vec = self.proj_vec(self.get_bary_vec(mi, neighbor_index), line)
            proj_normal = self.proj_vec(self.get_normal(mi, neighbor_index), line)
            bary_angle = self._angle(proj_bary_vec, [0,0,-1])
            normal_angle = self._angle(proj_normal, [0,0,-1])
            min_normal_idx = np.argmin(normal_angle)
            angle = np.arccos(np.sum(proj_bary_vec[min_normal_idx] * proj_normal[min_normal_idx]))
            if angle > self.POINT_BALENCE_TOL:
                return False
            # ok_idx = np.argsort(normal_angle)[:int( (normal_angle.size-1)*0.8)]
            # bary_angle = bary_angle[ok_idx]
            # normal_angle = normal_angle[ok_idx]
            if np.any(normal_angle > bary_angle + self.POINT_STABLE_TOL):
                return False #确保是稳定平衡
        else:
            points = mi.points_array[contact_index]
            if points.shape[0] >= 4:
            # 三个以及以上
                hull = ConvexHull(points[:, :2]) # 求凸包
                hull_points = points[hull.vertices][:, :2] #[x,y]
            else:
                hull_points = points[:,:2]

            poly_context = {'type': 'MULTIPOLYGON',
                                        'coordinates': [[hull_points.tolist()]]}
            poly_shape = shapely.geometry.shape(poly_context)
            bary = mi.barycenter[:2]
            bary_point = shapely.geometry.Point(bary[0], bary[1])
            if not poly_shape.intersects(bary_point):
                return False
        # colors = np.zeros(mi.points_array.shape)
        # colors[np.hstack(contact_neighbor_index)] = np.array([255, 0, 0])
        # colors[contact_index] = np.array([0, 0, 255])
        # mi.pcd.colors = utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([mi.pcd], width=1280, height=720)   
        return True

    # def run(self):
    #     self.model_list = [] # 模型点云列表
    #     sphere_angle = SphereAngle()
    #     stable_dict = {}
    #     for name in self.std_models_list:
    #         # if name == "ape_dense.ply":
    #         #     continue
    #         print(name)
    #         path = os.path.join(self.std_meshes_dir, name)
    #         mi = ModelInfo(path)
    #         stable_dict.update({name: []})
    #         rot_N = len(sphere_angle.rvec)
    #         skip = 0
    #         start_time = time.time()
    #         for i, rvec in enumerate(sphere_angle.rvec):
    #             if i < skip:
    #                 continue
    #             time_left = (time.time() - start_time)/(i+1-skip)*(rot_N-i)/60
    #             print("\r{:>4d}/{:>4d} time left:{:>4.2f}min".format(i, rot_N, time_left), end="")
    #             posture = Posture(rvec=rvec)
    #             r_mi = mi.transform(posture.trans_mat)
    #             contact_dict = self.get_contact_point(r_mi)
    #             is_stable = self.if_point_stable(r_mi, contact_dict)
    #             if is_stable:
    #                 stable_dict[name].append(rvec.tolist())
    #         print()
    #         break
    #     with open(os.path.join(self.std_meshes_dir, "stable.json"), 'w') as jf:
    #         json.dump(stable_dict, jf)

class MaskBilter():
    SHOW_NUM = 6
    def __init__(self, data_recorder:DataRecorder) -> None:
        self.data_recorder = data_recorder
        self.data_path = data_recorder.data_path
        # 视角变换矩阵
        # trans_mat_directory = os.path.join(self.directory, "transforms.npy")
        # self.transforms = np.load(trans_mat_directory) 
        # 类别范围
        self.model_index_range = self.data_recorder.category_idx_range
        # 当前类别名
        self.__current_name = ""
        # 选取的视角
        self.selected_view_trans_mats = []
        self.selected_frame_indecies = []
        self.selected_rgbs = []
        self.selected_depth_masks = []
        self.__view_pointer = 0

        # 相机内参
        self.camera_intr = CameraIntr.from_json(os.path.join(self.data_path, CALI_INTR_FILE))

        plt.ion()
    
    @property
    def cur_range(self):
        '''
        当前的帧范围
        '''
        try:
            range_ = self.model_index_range[self.__current_name]
        except:
            range_ = range(0,0)
        return range_

    @property
    def this_view_trans_mat(self):
        if len(self.selected_view_trans_mats) > 0:
            return self.selected_view_trans_mats[self.__view_pointer]
        else:
            return None

    @property
    def this_frame_rgb(self):
        if len(self.selected_view_trans_mats) > 0:
            return self.selected_rgbs[self.__view_pointer]
        else:
            return None

    @staticmethod
    def farthest_point_sample_angle(xyz, npoint): 

        """
        Input:
            xyz: pointcloud data, [N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        def angle(vec1, vec2):
            inner_product = np.sum(vec1 * vec2, axis=-1)/(np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1))
            inner_product = np.clip(inner_product, -1, 1)
            angles = np.arccos(inner_product)
            return angles
        N, C = xyz.shape
        
        centroids = np.zeros(npoint, dtype=np.int32)     # 采样点矩阵（B, npoint）
        distance = np.ones(N) * 1e10                       # 采样点到所有点距离（B, N）

        farthest = np.random.randint(0, N, dtype=np.int32)  # 初始时随机选择一点
        
        # barycenter = np.sum((xyz), 1)                                    #计算重心坐标 及 距离重心最远的点
        # barycenter = barycenter/xyz.shape[1]
        # barycenter = barycenter.view(1, 3)

        # dist = np.sum((xyz - barycenter) ** 2, -1)
        # farthest = np.max(dist,1)[1]                                     #将距离重心最远的点作为第一个点
        farthest_indeices = []        
        for i in range(npoint):
            # print("-------------------------------------------------------")
            # print("The %d farthest pts %s " % (i, farthest))
            centroids[i] = farthest                                      # 更新第i个最远点
            centroid = xyz[farthest, :]       # 取出这个最远点的xyz坐标
            dist = angle(xyz, centroid)                     # 计算点集中的所有点到这个最远点的欧式距离
            mask = dist < distance
            distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
            farthest = np.argmax(distance)                           # 返回最远点索引
            farthest_indeices.append(farthest)
    
        return centroids, np.array(farthest_indeices, dtype=np.int32)

    def set_current_name(self, name):
        self.__current_name = name
        self.select_views(MaskBilter.SHOW_NUM)
        self.read_rgbs()

    def next_view(self):
        '''
        下一个视角
        '''
        if len(self.selected_view_trans_mats)  > 0:
            self.__view_pointer += 1
            self.__view_pointer = self.__view_pointer % len(self.selected_view_trans_mats)

    def select_views(self, num):
        '''
        选择视角
        '''
        vectors = []
        cur_view_trans = []
        frame_indecies = []
        view_idx = []        
        for data_i in self.cur_range:
            T = self.data_recorder.trans_elements[data_i]
            end_points = T.dot(np.array([[0,0,0,0], [0,0,1,0]]).T).T #[2,4]
            vec = end_points[1] - end_points[0]
            vectors.append(vec)
            cur_view_trans.append(T)
        cur_view_trans = np.array(cur_view_trans)
        vectors = np.array(vectors)[:,:3]
        view_wanted = np.array([[1,0,0], [-1, 0, 0],[0, 1, 0],[0, -1, 0], [-1.732/3, 1.732/3, -1.732/2], [0,0, -1]])
        for vw in view_wanted:
            vec1 = vectors
            vec2 = vw
            inner_product = np.sum(vec1 * vec2, axis=-1)/(np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1))
            inner_product = np.clip(inner_product, -1, 1)
            angles = np.arccos(inner_product)
            view_idx.append(np.argmin(angles))
            frame_indecies.append(self.cur_range[np.argmin(angles)])
        # # 最远点采样，选角度差距最大的点
        # _, view_index = MaskBilter.farthest_point_sample_angle(vectors, num)
        view_idx = np.array(view_idx, dtype=np.int32)
        self.selected_view_trans_mats = cur_view_trans[view_idx, :, :]
        self.selected_frame_indecies = frame_indecies

    def read_rgbs(self):
        self.selected_rgbs.clear()
        for index in self.selected_frame_indecies:
            rgb = self.data_recorder.rgb_elements.read(index)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            self.selected_rgbs.append(rgb)

    def show(self, std_model_meta:ModelInfo, is_mask_visible):
        impos_list = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
        plt.clf()
        for i in range(MaskBilter.SHOW_NUM):
            impos = impos_list[i]
            trans_mat = self.selected_view_trans_mats[i]
            rgb = self.selected_rgbs[i]
            masks, _ = calc_masks([std_model_meta],
                                   [Posture(homomat=trans_mat).inv()],
                                   self.camera_intr,
                                    ignore_depth= True,
                                    tri_mode= False)
            mask = masks[0]
            if np.max(mask) > 0:
                bbox_max = np.array(np.where(mask)).max(axis=-1) + 5
                bbox_max[0] = np.clip(bbox_max[0], 0, rgb.shape[0])
                bbox_max[1] = np.clip(bbox_max[1], 0, rgb.shape[1])
                bbox_min = np.array(np.where(mask)).min(axis=-1) - 5
                bbox_min[0] = np.clip(bbox_min[0], 0, rgb.shape[0])
                bbox_min[1] = np.clip(bbox_min[1], 0, rgb.shape[1])
            else:
                bbox_max = np.array([rgb.shape[0], rgb.shape[1]])
                bbox_min = np.array([0, 0])
            
            mask = 255 - masks[0] #反
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            if is_mask_visible:
                show_image = mask * 0.4 + rgb * 0.6
                show_image = show_image.astype(np.uint8)
            else:
                show_image = rgb.copy()
            show_image = show_image[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1]]
            if i == self.__view_pointer:
                # 加一个红框
                show_image = cv2.rectangle(show_image, (0,0), (show_image.shape[1], show_image.shape[0]), (255, 0, 0), 10)
            plt.subplot(3, 2, i+1)
            plt.imshow(show_image)
        # plt.show()

class InteractIcp():
    def __init__(self, data_recorder: DataRecorder, model_manager: ModelManager) -> None:
        self.data_recorder = data_recorder
        self.model_manager = model_manager

        self.std_meshes_dir = self.model_manager.std_meshes_dir
        self.std_meshes_enums = self.model_manager.std_meshes_enums
        self.icp_trans_dir      = self.model_manager.icp_trans.data_path
        self.icp_unf_pcd_dir    = self.model_manager.icp_unf_pcd.data_path

        # self.segmesh_dir = os.path.join(directory, SEGMESH_DIR)
        # self.segmesh_list = os.listdir(self.segmesh_dir)
        # self.icp_dir = os.path.join(directory, ICP_DIR)
        # self.regis_dir = os.path.join(directory, REGIS_DIR)
        ### scs
        ###

        # 读取预先计算的各个物体的平衡姿态
        self.place_gen = None
        self.current_index = 0
        self.pcd_visible = True
        self.is_refine_mode = True
        self.step = 1.0
        self.refine_mode(None) #非精确模式
        self.color_g = InteractIcp.color_generator()
        self.modelbalenceposture = ModelBalencePosture()
        self.std_model_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        self.std_model_baryline = \
            o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cone_radius=0.004, 
                                                    cylinder_height=0.1,  cone_height=0.02).transform(
                                                    Posture(rvec=np.array([0, np.pi, 0])).trans_mat)
        self.baryline_visible = False
        self.contact_pcd = o3d.geometry.PointCloud()
        self.view_directions = [((1,0,0),(0,0,1)),((-1,0,0),(0,0,1)),
                                ((0,1,0),(0,0,1)),((0,-1,0),(0,0,1)),
                                ((0,0,1),(1,0,0)),((0,0,-1),(1,0,0))] #[(front, up)]
        self.view_switch:int = 0 #int
        self.ckt = None
        self.cover_data_confirm = False
        # 相机视角
        self.is_camera_mode = False
        self.is_mask_visible = True
        self.trans_with_CCS = False


    def load_data(self):
        # 读取所有的std和unf
        self.std_mi_list = []
        self.pcd_list = []  
        self.pcd_info_list = []
        with self.model_manager.icp_unf_pcd.get_writer().allow_overwriting():
            for name in self.std_meshes_enums:
                mesh = self.model_manager.std_meshes.read(name)
                barycenter = self.data_recorder.barycenter_dict[name]
                std_mi = ModelInfo(mesh, 
                                name = name, 
                                class_id=self.model_manager.std_meshes.dulmap_id_name(name),
                                barycenter=barycenter)
                if name not in self.data_recorder.barycenter_dict:
                    self.data_recorder.barycenter_dict[name] = std_mi.barycenter
                self.std_mi_list.append(std_mi)

                icp_unf_pcd = self.model_manager.icp_unf_pcd.read(name)
                if icp_unf_pcd is None:
                    extracted_mesh = self.model_manager.extracted_mesh.read(name)
                    if extracted_mesh is not None:
                        icp_unf_pcd = extracted_mesh.sample_points_uniformly(int(np.asarray(std_mi.mesh.vertices).shape[0] * 1.2))
                    else:
                        icp_unf_pcd = self.model_manager.registerd_pcd.read(name)
                        # trans_mat = build_sceneCS(aruco_centers, plane_equation, pcd)
                        # pcd.transform(np.linalg.inv(trans_mat))
                    print("创建:" + name)
                    self.model_manager.icp_unf_pcd.write(name, icp_unf_pcd)
                # 缩小
                bbox_3d = np.array(icp_unf_pcd.get_axis_aligned_bounding_box().get_box_points())
                org_z_max = np.max(bbox_3d[:,2])       
                self.pcd_list.append(icp_unf_pcd)
                self.pcd_info_list.append({"z_max": org_z_max})                

    @staticmethod
    def color_generator():
        color_list = np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255]])
        color_index = 0
        while True:
            yield color_list[color_index]
            color_index += 1
            color_index = color_index % int(color_list.shape[0])

    @property
    def current_std_mi(self) ->ModelInfo:
        try:
            return self.std_mi_list[self.current_index]
        except:
            return None

    @property
    def current_pcd(self):
        try:
            return self.pcd_list[self.current_index]
        except:
            return None

    def switch_to_trans_with_CCS(self, vis):
        if self.trans_with_CCS == True:
            print("以世界坐标系运动")
            self.trans_with_CCS = False
        else:
            print("以相机坐标系运动")
            self.trans_with_CCS = True

    def switch_to_camera_view_mode(self, vis):
        if self.is_camera_mode == True:
            print("退出相机模式")
            self.is_camera_mode = False
        else:
            print("进入相机模式")
            self.is_camera_mode = True
            self.next_view(vis)

    def set_is_mask_visible(self, vis):
        if self.is_mask_visible == True:
            print("掩膜显示关闭")
            self.is_mask_visible = False
        else:
            print("掩膜显示开启")
            self.is_mask_visible = True 
        self.maskbilter.show(self.gen_cur_std_model_meta(), self.is_mask_visible)     

    def next_view(self, vis):
        ctr = vis.get_view_control()
        # ctr.reset_camera_local_rotate()
        if self.is_camera_mode == False:
            ctr.set_lookat(self.current_std_mi.mesh.get_center())
            ctr.set_front(  self.view_directions[self.view_switch][0])  # set the positive direction of the x-axis toward you
            ctr.set_up(     self.view_directions[self.view_switch][1])  # set the positive direction of the x-axis as the up direction
            self.view_switch += 1
            self.view_switch = (self.view_switch + 1) % 6
        else:
            self.maskbilter.next_view()
            phcp = o3d.camera.PinholeCameraParameters()
            phcp.extrinsic = np.linalg.inv(self.maskbilter.this_view_trans_mat) #外参
            phcp.intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic  
            # phcp.intrinsic = o3d.camera.PinholeCameraIntrinsic( self.maskbilter.camera_intr.CAM_WID, 
            #                                                     self.maskbilter.camera_intr.CAM_HGT, 
            #                                                     self.maskbilter.camera_intr.CAM_FX, 
            #                                                     self.maskbilter.camera_intr.CAM_FY, 
            #                                                     self.maskbilter.camera_intr.CAM_CX, 
            #                                                     self.maskbilter.camera_intr.CAM_CY) #内参
            ctr.convert_from_pinhole_camera_parameters(phcp)
            self.maskbilter.show(self.gen_cur_std_model_meta(), self.is_mask_visible)

    def show_contact(self, vis):
        contact_dict = self.modelbalenceposture.get_contact_point(self.current_std_mi)
        contact_index = list(contact_dict.keys())
        contact_neighbor_index = np.hstack(list(contact_dict.values())).astype(np.int32)
        
        points = np.array(self.current_std_mi.points_array)
        index = np.union1d(contact_index, contact_neighbor_index).astype(np.int32)
        points = points[index]
        colors = np.zeros(self.current_std_mi.points_array.shape)
        colors[contact_neighbor_index] = np.array([255, 0, 0])
        colors[contact_index] = np.array([0, 0, 255])
        colors = colors[index]
        self.contact_pcd.points = utility.Vector3dVector(points)
        self.contact_pcd.colors = utility.Vector3dVector(colors)
        vis.update_geometry(self.contact_pcd)
        # o3d.visualization.draw_geometries([mi.pcd], width=1280, height=720) 

    def show_bary_arrow(self, vis):
        if self.baryline_visible:
            self.baryline_visible = False
        else:
            self.baryline_visible = True

    def change_background_color(self, vis):        
        opt = vis.get_render_option()
        opt.background_color = next(self.color_g)
        return False

    def set_unfpcd_visible(self, vis):        
        # opt = vis.get_render_option()
        # opt.point_size = self.point_size
        # self.point_size += 1.0
        # if self.point_size > 5.0:
        #     self.point_size -= 5
        if self.pcd_visible:
            self.current_pcd.scale(1/2**10, np.array([0.0,0,0]))    
            self.pcd_visible = False
        else:
            self.current_pcd.scale(2**10, np.array([0.0,0,0]))    
            self.pcd_visible = True
        vis.update_geometry(self.current_pcd)
        
    def skip(self, vis):
        vis.remove_geometry(self.current_pcd)
        vis.remove_geometry(self.current_std_mi.mesh)
        vis.remove_geometry(self.std_model_frame)
        self.current_index += 1
        if self.current_std_mi is None:
            self.model_manager.icp_std_mesh.close()
            self.model_manager.icp_unf_pcd.close()
            self.model_manager.icp_trans.close()
            print("已经结束，请手动关闭")
            return
        self.std_model_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        vis.add_geometry(self.current_pcd)
        vis.add_geometry(self.current_std_mi.mesh)
        vis.add_geometry(self.std_model_frame)
        self.pre_progress()        

    def confirm(self, vis):
        saved = self.post_progress()
        if saved:
            self.skip(vis)

    def get_O2C0(self):
        # 根据当前坐标反求变换矩阵
        # transed_std_mesh = self.current_std_mi.mesh #标准点云，找标准点云和估计点云最匹配的姿态
        # path = os.path.join(self.std_meshes_dir, self.std_models_list[self.current_index])
        # org_std_pcd = io.read_point_cloud(path) #标准点云，找标准点云和估计点云最匹配的姿态
        transed_std_mesh = self.current_std_mi.mesh #transformed standard mesh
        org_std_mesh = self.model_manager.std_meshes.read(self.current_index)
        # org_std_pcd.scale(0.001, np.array([0.0, 0, 0]))
        transed_std_points = np.array(transed_std_mesh.vertices) #点坐标
        org_std_points = np.array(org_std_mesh.vertices)
        
        # pad 1 to the last dimension
        N = org_std_points.shape[0]
        transed_std_points =  np.hstack((transed_std_points, np.ones((N, 1)))) #[N, 4]
        org_std_points = np.hstack((org_std_points, np.ones((N, 1)))) #[N, 4]
        # sample 20 points to compute the transformation matrix
        idx = np.linspace(0, org_std_points.shape[0], 20, endpoint=False, dtype= np.int32)
        org_std_points_set = org_std_points[idx] #[N, 4]
        transed_std_points_set = transed_std_points[idx] #[N, 4]

        trans_mat = transed_std_points_set.T.dot(np.linalg.pinv(org_std_points_set.T))
        return trans_mat

    def gen_cur_std_model_meta(self):
        return self.std_mi_list[self.current_index]
        # name = self.std_meshes_enums[self.current_index]
        # trans_mat_O2C0 = self.get_O2C0()
        # points = np.array(self.current_std_mi.mesh.vertices)
        # points = np.linalg.inv(trans_mat_O2C0).dot(homo_pad(points).T).T[:, :3]
        # tris = np.array(self.current_std_mi.mesh.triangles)
        # return ModelInfo(name, 0, trans_mat_O2C0, points, tris)

    def current_std_mi_transform(self, vis, T):
        self.current_std_mi.transform(T, False)  
        self.std_model_frame.transform(T)  
        self.cover_data_confirm = False
        if vis is not None:
            vis.update_geometry(self.current_std_mi.mesh)
            vis.update_geometry(self.std_model_frame)
            self.show_contact(vis)

            top = np.array(self.std_model_baryline.vertices)[:, 2].max()
            center = self.std_model_baryline.get_center()
            local_org = np.array([center[0], center[1], top])
            dest = self.current_std_mi.barycenter
            dest[2] += int(self.baryline_visible)
            self.std_model_baryline.transform(Posture(tvec=dest - local_org).trans_mat)
            vis.update_geometry(self.std_model_baryline)
            if self.is_camera_mode:
                self.maskbilter.show(self.gen_cur_std_model_meta(), self.is_mask_visible)

    def trans_with_obj_center(self, T_center:Posture, vis):
        '''
        以物体中心的旋转
        '''
        # if self.is_camera_mode:
        if self.trans_with_CCS:
            ctr = vis.get_view_control()
            phcp = ctr.convert_to_pinhole_camera_parameters()   
            extr = phcp.extrinsic
            camera_rot = Posture(rmat=extr[:3,:3])
            rotate_T = np.linalg.multi_dot((camera_rot.trans_mat, T_center.trans_mat, camera_rot.inv_transmat))     
        else:
            rotate_T = T_center.trans_mat
        ####
        center = self.current_std_mi.mesh.get_center()
        translate = Posture(tvec=center)
        T = np.linalg.multi_dot((translate.trans_mat, rotate_T, translate.inv_transmat))      
        self.current_std_mi_transform(vis, T)
    
    def rotate_X_inc(self, vis):
        rotate = Posture(rvec=np.array([self.step * np.pi/180, 0, 0]))
        self.trans_with_obj_center(rotate, vis)
    
    def rotate_Y_inc(self, vis):
        rotate = Posture(rvec=np.array([0, self.step * np.pi/180, 0]))
        self.trans_with_obj_center(rotate, vis)

    def rotate_Z_inc(self, vis):
        rotate = Posture(rvec=np.array([0, 0, self.step * np.pi/180]))
        self.trans_with_obj_center(rotate, vis)
    
    def rotate_X_dec(self, vis):
        rotate = Posture(rvec=np.array([-self.step * np.pi/180, 0, 0]))
        self.trans_with_obj_center(rotate, vis)
    
    def rotate_Y_dec(self, vis):
        rotate = Posture(rvec=np.array([0, -self.step * np.pi/180, 0]))
        self.trans_with_obj_center(rotate, vis)

    def rotate_Z_dec(self, vis):
        rotate = Posture(rvec=np.array([0, 0, -self.step * np.pi/180]))
        self.trans_with_obj_center(rotate, vis)

    def translate_X_inc(self, vis):
        posture = Posture(tvec=np.array([self.step * 1, 0, 0]))
        self.trans_with_obj_center(posture, vis)

    def translate_Y_inc(self, vis):
        posture = Posture(tvec=np.array([0, self.step * 1, 0]))
        self.trans_with_obj_center(posture, vis)
    
    def translate_Z_inc(self, vis):
        posture = Posture(tvec=np.array([0, 0, self.step * 1]))
        self.trans_with_obj_center(posture, vis)

    def translate_X_dec(self, vis):
        posture = Posture(tvec=np.array([-self.step * 1, 0, 0]))
        self.trans_with_obj_center(posture, vis)

    def translate_Y_dec(self, vis):
        posture = Posture(tvec=np.array([0, -self.step * 1, 0]))
        self.trans_with_obj_center(posture, vis)
    
    def translate_Z_dec(self, vis):
        posture = Posture(tvec=np.array([0, 0, -self.step * 1]))
        self.trans_with_obj_center(posture, vis)

    def refine_mode(self, vis):
        if not self.is_refine_mode:
            self.step = 0.1
            self.is_refine_mode = True
        else:
            self.step = 1.0
            self.is_refine_mode = False

    def auto_icp(self, vis):
        '''
        根据用户定义初始姿态进行配准
        '''
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(np.asarray(self.current_std_mi.mesh.vertices))
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(np.asarray(self.current_pcd.points))
        threshold = 2
        icp = o3d.pipelines.registration.registration_icp(
                source, target, threshold, np.eye(4, dtype = np.float32),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # print(icp)
        posture = Posture(homomat=icp.transformation)
        posture_t = Posture(tvec = posture.tvec)
        self.current_std_mi_transform(vis, posture.trans_mat)
        # std_pcd_o3d_geometry = o3d.io.read_triangle_mesh(os.path.join(std_meshes_dir, std_models_list[std_model_index]))
        # std_pcd_o3d_geometry.compute_vertex_normals()
    
    def nearest_points_mean_d(self):
        if self.ckt is not None:
            find_point = self.current_std_mi.points_array
            d, _ = self.ckt.query(find_point)  # 返回最近邻点的距离d和在数组中的顺序x
            # 允许对于高度较低的点存在误差
            # 对于超出平面的点给与惩罚
            z = find_point[:, 2]
            z_tol = self.pcd_info_list[self.current_index]["z_max"]*0.2
            low = z < z_tol
            decay = np.ones(z.shape)
            decay[low] = 0
            d = d * decay
            score = np.mean(d)*1000
            return score
        else:
            return -1.0

    def print_nearest_points_mean_d(self, vis):
        print("距离：{}".format(self.nearest_points_mean_d()))

    def GA_registration(self, vis):
        # 使用遗传算法，只优化3个坐标：x,y,z转
        print("start GA", end="")
        call_num = 0
        size_pop=500
        max_iter=30
        def func(posture):
            nonlocal call_num
            call_num += 1
            x,y = posture[0], posture[1]
            z_angle = posture[2]
            local_rot = np.array([  [np.cos(z_angle),     -np.sin(z_angle),   0],
                                    [np.sin(z_angle),     np.cos(z_angle),    0],
                                    [0,                   0,                  1]])
            std_center = self.current_std_mi.mesh.get_center()
            t1_std_points = self.current_std_mi.points_array - std_center
            find_point = np.dot(local_rot, t1_std_points.T).T # 原点
            find_point = find_point + std_center + np.array([posture[0], posture[1], 0])
            if call_num % (size_pop) == 0:
                print("-", end="")
            return self.nearest_points_mean_d()
        ga = MyGA(func=func, n_dim=3, size_pop=size_pop, max_iter=max_iter, lb=[-2, -2, -np.pi/36], 
                                                                            ub=[ 2,  2,  np.pi/36], 
                                                                            precision=[ 0.5, 0.5, np.pi/360], 
                                                                            early_stop=4,
                                                                            result_precision=1)
        best_x, best_y = ga.run()
        posture_local = Posture(rvec=np.array([0,0,best_x[2]]), tvec=np.array([best_x[0], best_x[1], 0]))
        self.trans_with_obj_center(posture_local, vis)
        print()

    def place_step(self, vis):
        if self.place_gen is None:
            self.place_gen = self.modelbalenceposture.place(self.current_std_mi)
        else:
            pass
        try: 
            T = next(self.place_gen)
        except StopIteration:
            self.place_gen = None
            return
        self.current_std_mi_transform(vis, T)
        min_z = np.min(self.current_std_mi.points_array[:,2])
        up_move_posture = Posture(tvec=np.array([0,0,-min_z]))
        self.current_std_mi_transform(vis, up_move_posture.trans_mat)

    def pre_progress(self):
        '''
        预处理:
        1) 修改为平衡姿态
        2) 平移到中心
        '''
        # 尝试读取变换矩阵，如果没有则仅平移至中心
        T = self.model_manager.icp_trans.read(self.current_index)
        if T is None:
            pcd_center = self.current_pcd.get_center()
            std_pcd_center = self.current_std_mi.mesh.get_center()
            T = Posture(tvec=pcd_center - std_pcd_center).trans_mat
        self.current_std_mi_transform(None, T)

        self.ckt = spt.cKDTree(np.array(self.current_pcd.points))  # 用C写的查找类，执行速度更快
        self.maskbilter.set_current_name(self.std_meshes_enums[self.current_index])
        points = np.array(self.current_pcd.points)
        # self.maskbilter.read_depths_as_mask(points)

    def post_progress(self):
        '''
        完成配准后的后处理
        '''
        # 根据当前坐标反求变换矩阵
        trans_mat = self.get_O2C0()
        # points_ = trans_mat.dot(org_std_points_set.T).T

        # try:
        #     transform_path = self.model_manager.icp_trans.auto_path(self.current_index)
        # except:
        #     pass
        # else:
        if self.current_index in self.model_manager.icp_std_mesh and self.cover_data_confirm == False:
            print("文件已经存在，再次按下确认覆盖")
            self.cover_data_confirm = True
            return False                   
        self.cover_data_confirm = False
        self.model_manager.icp_trans.write(self.current_index, trans_mat, force=True)
        self.model_manager.icp_std_mesh.write(self.current_index, self.current_std_mi.mesh, force=True)
        return True

    def start(self):
        self.load_data()
        
        self.maskbilter = MaskBilter(self.data_recorder)

        self.pre_progress()

        key_to_callback = {}
        key_to_callback[ord("Q")] = self.rotate_X_inc
        key_to_callback[ord("W")] = self.rotate_X_dec
        key_to_callback[ord("A")] = self.rotate_Y_inc
        key_to_callback[ord("S")] = self.rotate_Y_dec
        key_to_callback[ord("Z")] = self.rotate_Z_inc
        key_to_callback[ord("X")] = self.rotate_Z_dec
        key_to_callback[ord("E")] = self.translate_X_inc
        key_to_callback[ord("R")] = self.translate_X_dec       
        key_to_callback[ord("D")] = self.translate_Y_inc
        key_to_callback[ord("F")] = self.translate_Y_dec       
        key_to_callback[ord("C")] = self.translate_Z_inc
        key_to_callback[ord("V")] = self.translate_Z_dec
        
        key_to_callback[ord("T")] = self.change_background_color
        key_to_callback[ord("Y")] = self.set_unfpcd_visible
        key_to_callback[ord("U")] = self.print_nearest_points_mean_d
        key_to_callback[ord("O")] = self.next_view
        
        key_to_callback[ord("1")] = self.refine_mode
        key_to_callback[ord("2")] = self.switch_to_camera_view_mode
        key_to_callback[ord("3")] = self.switch_to_trans_with_CCS
        key_to_callback[ord("4")] = self.set_is_mask_visible

        key_to_callback[ord("J")] = self.skip
        key_to_callback[ord("N")] = self.confirm
        # key_to_callback[ord("K")] = self.GA_registration
        key_to_callback[ord("M")] = self.auto_icp
        # key_to_callback[ord(",")] = self.place_step
        key_to_callback[ord("O")] = self.show_bary_arrow
        key_to_callback[ord("L")] = self.next_view

        for key, func in key_to_callback.items():
            print(f"press {chr(key)} : {func.__name__}")

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        table = o3d.geometry.TriangleMesh.create_box(700, 700 ,1.0).paint_uniform_color(np.array([0.0, 1.0, 1.0]))
        table.transform(Posture(tvec=np.array([0,0,-1.0])).trans_mat)
        o3d.visualization.draw_geometries_with_key_callbacks([self.current_pcd, self.current_std_mi.mesh, 
                                                              self.std_model_frame, self.contact_pcd,
                                                              self.std_model_baryline,
                                                              frame, table], key_to_callback ,width =640, height=480)

# if __name__ == "__main__":
#     argv = sys.argv
#     argv = ("", "LINEMOD\\{}".format(DATADIR)) 
#     try:
#         if argv[1] == "all":
#             folders = glob.glob("LINEMOD/*/")
#         elif argv[1] +"\\" in glob.glob("LINEMOD/*/"):
#             folders = [argv[1] +"\\"]
#         else:
#             exit()
#     except:
#         exit()

#     for directory in folders:
#     # pcd = io.read_point_cloud(os.path.join(r"LINEMOD\models", "ape.ply"))
#         interact_icp = Interact_icp(directory)
#         interact_icp.start()
#     # m = ModelBalencePosture(r"LINEMOD\000010")
#     # m.run()