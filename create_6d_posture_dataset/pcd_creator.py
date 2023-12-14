"""
register_scene.py
---------------
Create registered scene pointcloud with ambient noise removal
The registered pointcloud includes the table top, markers, and some noise
This mesh needs to be processed in a mesh processing tool to remove the artifact

"""
from tqdm import tqdm, trange
import open3d as o3d
import numpy as np
import cv2
import colorsys
import os
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm import tqdm

from .data_manager import DataRecorder, EnumElements, ModelManager, ProcessData
from .utils.aruco_detector import ArucoDetector
from .utils.camera_sys import convert_depth_frame_to_pointcloud
from .utils.bounded_voronoi import bounded_voronoi, get_seg_maps
from .utils.plane import fitplane
from .utils.pc_voxelize import pc_voxelize, pc_voxelize_reture
from . import Posture, JsonIO, FRAMETYPE_DATA, CameraIntr



class PcdCreator():
    '''
    1.register the pointclouds of the scene
    2.segment the scene into different parts
    3.match the name of the segment with the name of the object
    '''
    def __init__(self, data_recorder:DataRecorder, aruco_detector:ArucoDetector, model_manager:ModelManager, voxel_size = 0.5) -> None:
        
        self.data_recorder = data_recorder
        self.aruco_detector = aruco_detector
        self.model_manager = model_manager

        self.voxel_size = voxel_size
        self.radius = self.voxel_size * 2
        self.max_correspondence_distance_coarse     = voxel_size * 15
        self.max_correspondence_distance_fine       = voxel_size * 1.5
        
        self.process_data = self.model_manager.process_data

        # assert self.data_recorder.intr_0_file.all_exist, "intr_0_file must exist"
        # assert self.data_recorder.intr_1_file.all_exist, "intr_1_file must exist"
        # self.intr_0 = self.data_recorder.intr_0_file.read(0)
        # self.intr_1 = self.data_recorder.intr_1_file.read(0)
        
    def _register_post_process(self, originals:list[o3d.geometry.PointCloud]):
        """
        Merge segments so that new points will not be add to the merged
        model if within voxel_Radius to the existing points, and keep a vote
        for if the point is issolated outside the radius of inlier_Radius at 
        the timeof the merge

        Parameters
        ----------
        originals : List of open3d.Pointcloud classe
            6D pontcloud of the segments transformed into the world frame
        voxel_Radius : float
            Reject duplicate point if the new point lies within the voxel radius
            of the existing point
        inlier_Radius : float
            Point considered an outlier if more than inlier_Radius away from any 
            other points

        Returns
        ----------
        points : (n,3) float
            The (x,y,z) of the processed and filtered pointcloud
        colors : (n,3) float
            The (r,g,b) color information corresponding to the points
        vote : (n, ) int
            The number of vote (seen duplicate points within the voxel_radius) each 
            processed point has reveived
        """
        singlemerged_pcd = o3d.geometry.PointCloud()
        for pcd in originals:
            singlemerged_pcd += pcd
        # o3d.io.write_point_cloud("singlemerged_pcd.ply", singlemerged_pcd)
        singlemerged_pcd = singlemerged_pcd.remove_radius_outlier(nb_points=16, radius=self.radius)
        return singlemerged_pcd

    def _merge_meshes(self):
        '''
        generate the merged mesh of the scene
        '''
        scenemerged_pcd = o3d.geometry.PointCloud()
        for pcd in self.model_manager.registerd_pcd:
            scenemerged_pcd += pcd
        # 降采样合并结果
        # mesh_pcd, _ = mesh_pcd.remove_statistical_outlier(20, 0.8)
        scenemerged_pcd = scenemerged_pcd.voxel_down_sample(voxel_size=self.voxel_size*2)
        return scenemerged_pcd

    def register(self, downsample = True, update = False):
        if update or len(self.model_manager.registerd_pcd) != len(self.model_manager.std_meshes):
            raw_pcd_dict = {}
            aruco_used_times:dict[str, dict[int, int]] = {}
            for category_idx, framemeta in tqdm(self.data_recorder.read_in_category_range(0, -1), total=self.data_recorder.num):
                category_name = self.data_recorder.category_names[category_idx]

                color   = framemeta.color
                depth   = framemeta.depth
                T       = framemeta.trans_mat_Cn2C0
                intr_M  = framemeta.intr_M
                
                mask = np.zeros((color.shape[0], color.shape[1])).astype(np.bool_)
                crop_rect_tl = (np.array(color.shape) * 0.25).astype(np.int32)
                crop_rect_br = (np.array(color.shape) * 0.75).astype(np.int32)
                mask[crop_rect_tl[0]: crop_rect_br[0], crop_rect_tl[1]: crop_rect_br[1]] = 1
                mask = mask * depth.copy().astype(np.bool_)
                depth = convert_depth_frame_to_pointcloud(depth, intr_M)

                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(depth[mask>0])
                source.colors = o3d.utility.Vector3dVector(color[mask>0][:, ::-1] / 255)
                source = source.transform(T)
                # o3d.alization.draw_geometries([source], width=1280, height=720)
                if downsample == True:
                    source = source.voxel_down_sample(voxel_size = self.voxel_size)
                    # source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = self.voxel_size * 3, max_nn = 50))
                # source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = self.voxel_size * 3, max_nn = 50))
                raw_pcd_dict.setdefault(
                    category_name, 
                    []).append(source)
                
                # statistic the used times of each aruco for each category
                _, ids, _ = self.aruco_detector.detect_aruco_2d(color)
                aruco_used_times.setdefault(category_name, {})
                for id_ in ids:
                    aruco_used_times[category_name].setdefault(id_, 0)
                    aruco_used_times[category_name][id_] += 1
            
            self.process_data.write_info(ProcessData.ARUCO_USED_TIMES, aruco_used_times)

            # post process

            with self.model_manager.registerd_pcd.get_writer().allow_overwriting():
                for k, pcd_list in tqdm(raw_pcd_dict.items()):
                    singlemerged_pcd = self._register_post_process(pcd_list)[0]
                    # downsample
                    singlemerged_pcd = singlemerged_pcd.voxel_down_sample(voxel_size = self.voxel_size *2)
                    # write
                    name = k
                    self.model_manager.registerd_pcd.write(name, singlemerged_pcd)
        if update or not self.model_manager.merged_regist_pcd_file.all_files_exist:
            merged = self._merge_meshes()
            self.model_manager.merged_regist_pcd_file.write(0, merged, force=True)

        # self.model_manager.close()
        # self.data_recorder.close()
    
    def match_segmesh_name(self, seg_polygons, SCStrans_mat):
        '''
        根据voronoi分割图和各个模型的ply，将他们按正确的类别进行分割，返回字典{name: box}

        seg_polygons: N*[p, 3]
        '''
        aruco_voronoi = Aruco_Voronoi(self, SCStrans_mat)

        # regis_dir = os.path.join(directory, REGIS_DIR)

        # files = os.listdir(regis_dir)
        # files = list(filter(lambda x: os.path.splitext(x)[1] == ".ply" and (not "merged" in x), files))
        # model_names = [os.path.splitext(x)[0] for x in files]
        # M = len(files)
        # seged_pcds = {}

        # # 尝试读取结果
        # voronoi_seg_dir = os.path.join(directory, VORONOI_SEGPCD_DIR)        
        # for name in model_names:
        #     pcd = o3d.io.read_point_cloud(os.path.join(voronoi_seg_dir, name+".ply"))
        #     if np.array(pcd.points).shape[0] == 0:
        #         seged_pcds = {}
        #         break
        #     else:
        #         seged_pcds.update({name: pcd})
        # if len(seged_pcds.keys()) == M:
        #     return
        #     # return seged_pcds
        ### 创建用于分割的凸包
        o3d_vols = []
        for polygon in seg_polygons:
            # 将二维多边形扩展为三维
            polygon = np.array(polygon)
            bounding_polygon = polygon.copy()
            bounding_polygon[:, -1] = 0.000 # 从3mm以上开始分割
            vol = o3d.visualization.SelectionPolygonVolume()
            vol.orthogonal_axis = "Z"
            vol.axis_max = 500
            vol.axis_min = 6
            vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
            o3d_vols.append(vol)

        ### 原始点云的前处理
        model_pcds = [] #点云列表[M]
        model_pcds_dense = [] #点云密度列表[M] 用于估算面积
        for i, pcd in tqdm(enumerate(self.model_manager.registerd_pcd), desc="original pointcloud preprocess", total=self.model_manager.registerd_pcd.num):
            pcd.transform(SCStrans_mat)
            pcd = pcd.voxel_down_sample(voxel_size=1) #降采样，使得点云密度相等
            name = self.model_manager.std_meshes_names[i]
            # pcd = aruco_voronoi.crop_by_aruco_num(pcd, name)
            # pcd = aruoc_voronoi.crop(pcd, [0, 11, 6, 3])
            # o3d.visualization.draw_geometries([pcd], width=1280, height=720)   
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances) 
            model_pcds_dense.append(avg_dist)
            model_pcds.append(pcd)
            # o3d.visualization.draw_geometries([pcd], width=1280, height=720)

        assert len(model_pcds) == len(o3d_vols)
        M = len(model_pcds)

        region_scores = np.zeros((M, M))
        # calculate score, use linear_sum_assignment to match
        for i, vol in enumerate(o3d_vols):
            region_score = []
            # pcd_num = 0
            for j, (dense, pcd) in enumerate(zip(model_pcds_dense, model_pcds)):
                comp = vol.crop_point_cloud(pcd) #裁剪
                distances = comp.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances) 
                comp_num = int(np.array(comp.points).shape[0])
                if avg_dist > dense * 2 or np.isnan(avg_dist):
                    score = 0
                else:
                    score = comp_num * avg_dist**2
                region_score.append(score)
                # pcd_num += 1
                # o3d.visualization.draw_geometries([comp], width=1280, height=720)
                region_scores[i, j] = score
        row_ind, top_rank_slice = linear_sum_assignment(region_scores, maximize=True)
        # region_scores_mean = np.mean(region_scores, axis=-1) # 每个区域的得分均值
        # region_scores_rank = np.argsort(region_scores, axis=-1) #从小到大排序
        # top_rank_slice = region_scores_rank[:, -1]
        # top_score_slice = region_scores[np.arange(0,M, dtype=np.uint), top_rank_slice]
        # while len(np.unique(top_rank_slice)) != M:
        #     # 每个区域的得分最高必须各不相同
        #     for i in range(M):
        #         same_region = np.where(top_rank_slice == i)[0] # 行号
        #         if len(same_region) > 1:
        #             campare_adv = top_score_slice[same_region] / region_scores_mean[same_region] # 比较优势
        #             eliminated_region = same_region[np.argsort(campare_adv)[:-1]]
        #             eliminated = (eliminated_region, top_rank_slice[eliminated_region])
        #             region_scores[eliminated] = 0
        #     region_scores_mean = np.mean(region_scores, axis=-1) # 每个区域的得分均值
        #     region_scores_rank = np.argsort(region_scores, axis=-1) #从小到大排序
        #     top_rank_slice = region_scores_rank[:, -1]
        #     top_score_slice = region_scores[np.arange(0,M, dtype=np.uint), top_rank_slice]
        # return model_pcds, model_names, top_rank_slice, o3d_vols
        interact = Interact_ChechSeg(model_pcds, self.model_manager.std_meshes_names, top_rank_slice, o3d_vols)
        interact.start()
        # print("请检查是否匹配正确，如有错误请在{}目录下手动修改名称".format(VORONOI_SEGPCD_DIR))


        with self.model_manager.voronoi_segpcd.get_writer().allow_overwriting():
            for model_id, vol in zip(interact.top_rank_slice, interact.o3d_vols):
                pcd = interact.model_pcds[model_id]
                name = interact.model_names[model_id]

                # o3d.visualization.draw_geometries([pcd], width=1280, height=720)    
                vol.axis_min = 0.000
                comp = vol.crop_point_cloud(pcd) #裁剪
                # o3d.visualization.draw_geometries([comp], width=1280, height=720)   
                self.model_manager.voronoi_segpcd.write(model_id, comp)
                # o3d.io.write_point_cloud(os.path.join(voronoi_seg_dir, name+".ply"), comp)
        # y = input("检查完毕请输入任意字符")
        return 
    
    def extract_uniform_pcd(self, SCStrans_mat, floor_color = None):
        '''
        提取均匀点云

        seg_polygons: N*[p, 3]
        '''
        # if len(os.listdir(os.path.join(directory, SEGMESH_DIR))) > 0:
        #     return
        # voronoi_segpcd_dir = os.path.join(directory, VORONOI_SEGPCD_DIR)
        # seged_pcds_dict = {}
        # for name in os.listdir(voronoi_segpcd_dir):
        #     if os.path.splitext(name)[-1] != '.ply':
        #         continue
        #     mainname = os.path.splitext(name)[0]
        #     pcd = o3d.io.read_point_cloud(os.path.join(voronoi_segpcd_dir, name))
        #     seged_pcds_dict.update({mainname: pcd})



        with self.model_manager.extracted_mesh.get_writer().allow_overwriting():
            # for name, comp in zip(self.model_manager.std_meshes.enums , self.model_manager.voronoi_segpcd):
            for fh in self.model_manager.voronoi_segpcd.query_all_fileshandle():
                print(fh.get_name())      
                model_idx = self.model_manager.voronoi_segpcd.deformat_corename(fh.corename)
                comp = self.model_manager.voronoi_segpcd[model_idx]
                # o3d.visualization.draw_geometries([comp], width=1280, height=720)
                # name:str = model_names[pcd_index]
                offset = 5
                floor_points, floor_point_colors, refine_vol = self._get_floor_points(comp)
                if floor_color is not None:
                    comp = self._filter_by_color(comp, floor_color, 20/255)
                refine_vol.axis_min = offset
                comp = refine_vol.crop_point_cloud(comp) #裁剪
                comp.points = o3d.utility.Vector3dVector(np.vstack((np.array(comp.points), floor_points)))
                comp.colors = o3d.utility.Vector3dVector(np.vstack((np.array(comp.colors), floor_point_colors)))

                # o3d.visualization.draw_geometries([comp], width=1280, height=720)       
                comp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=10))
                comp.orient_normals_towards_camera_location(np.mean(np.array(comp.points), axis = 0))
                normals = np.negative(np.array(comp.normals))
                comp.normals = o3d.utility.Vector3dVector(normals)          
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(comp, depth = 8, linear_fit = True) 
                mesh.transform(np.linalg.inv(SCStrans_mat))

                self.model_manager.extracted_mesh.write(model_idx, mesh)


    def auto_seg(self, update = False):
        '''
        自动分割：
        1、根据aruco建立场景坐标系SCS，aruco必须由12个，均布在矩形的一周
        2、初步分割点云，只保留aruco包围的矩形范围
        3、场景体素化，开运算提取孤立实体
        4、保存
        '''
        org_pcd = self.model_manager.merged_regist_pcd_file.read(0)        
        ### 1 ###
        if update or \
                not self.process_data.has_info(self.process_data.ARUCO_CENTERS) or \
                not self.process_data.has_info(self.process_data.PLANE_EQUATION) or \
                not self.process_data.has_info(self.process_data.TRANS_MAT_C0_2_SCS):
            aruco_centers, plane_equation, trans_mat = self.build_build_sceneCS(org_pcd)
            # self.process_data.write(self.process_data.ARUCO_CENTERS, aruco_centers)
            self.process_data.write_info(self.process_data.ARUCO_CENTERS,        aruco_centers)
            self.process_data.write_info(self.process_data.PLANE_EQUATION,       plane_equation)
            self.process_data.write_info(self.process_data.TRANS_MAT_C0_2_SCS,   trans_mat)
        else:
            aruco_centers   = self.process_data.read_info(self.process_data.ARUCO_CENTERS)
            plane_equation  = self.process_data.read_info(self.process_data.PLANE_EQUATION)
            trans_mat       = self.process_data.read_info(self.process_data.TRANS_MAT_C0_2_SCS)
        ### 2 ###
        arucos_SCS = Posture(homomat=trans_mat) * aruco_centers
        pointcloud_SCS = Posture(homomat=trans_mat) * np.array(org_pcd.points)
        # 裁除多余点
        xmin, xmax = np.min(arucos_SCS[:, 0]), np.max(arucos_SCS[:, 0])
        ymin, ymax = np.min(arucos_SCS[:, 1]), np.max(arucos_SCS[:, 1])
        _mask = (pointcloud_SCS[:, 2] > 0) *\
                (pointcloud_SCS[:, 0] > xmin) *\
                (pointcloud_SCS[:, 0] < xmax) *\
                (pointcloud_SCS[:, 1] > ymin) *\
                (pointcloud_SCS[:, 1] < ymax)
        pointcloud_SCS = pointcloud_SCS[_mask]
        colors_SCS = np.array(org_pcd.colors)[_mask]

        # ax = plt.axes(projection='3d')  # 设置三维轴
        # ax.scatter(arucos_SCS[:, 0], arucos_SCS[:, 1], arucos_SCS[:, 2])
        # plt.show()
        if update or not self.process_data.has_info(self.process_data.VOR_POLYS_COORD):
            box, box_color, restore_mat = pc_voxelize(pointcloud_SCS, 3, pcd_color = colors_SCS) #???
            vor_polys_coord = get_seg_maps(box[:,:,0], restore_mat, scale = 1) #???
            self.process_data.write_info(self.process_data.VOR_POLYS_COORD, vor_polys_coord)
        else:
            vor_polys_coord = self.process_data[self.process_data.VOR_POLYS_COORD]
        ### 3 ###
        if update or not self.process_data.has_info(self.process_data.FLOOR_COLOR):
            floor_color = self.get_floor_color(pointcloud_SCS, colors_SCS)
            self.process_data.write_info(self.process_data.FLOOR_COLOR, floor_color)
        else:
            floor_color = self.process_data.read_info(self.process_data.FLOOR_COLOR)

        if update or self.model_manager.voronoi_segpcd.num != self.model_manager.std_meshes.num:
            seged_pcds = self.match_segmesh_name(vor_polys_coord, trans_mat)
        if update or self.model_manager.extracted_mesh.num != self.model_manager.std_meshes.num:
            self.extract_uniform_pcd(trans_mat, floor_color)

    def build_build_sceneCS(self, test_pcd):
        _C0_aruco_3d_dict = self.aruco_detector.C0_aruco_3d_dict
        arucos = np.array(list(_C0_aruco_3d_dict.values()))
        aruco_centers = np.mean(arucos, axis=1)
        # for v in _C0_aruco_3d_dict.values():
        #     aruco_centers.append(np.mean(v, axis=0))
        # aruco_centers = np.array(aruco_centers)
        sol = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]
        plane_equation = fitplane(sol, aruco_centers)
        plane_equation = plane_equation/ np.linalg.norm(plane_equation[:3])


        # 建立临时坐标系
        arucos = np.reshape(arucos, (-1, 3))
        p_x = arucos[0]
        center = np.mean(arucos, axis=0)
        x_axis = p_x - center
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = fitplane(plane_equation, arucos)[:3]
        z_axis = z_axis/np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        frame_C0 = np.array([   [0,0,0,1], 
                                [1,0,0,1], 
                                [0,1,0,1], 
                                [0,0,1,1]], np.float32)
        frame_scene_temp =  np.array([   center, 
                                center + x_axis, 
                                center + y_axis, 
                                center + z_axis], np.float32)  
        frame_scene_temp = np.hstack((frame_scene_temp, np.ones((4,1))))
        trans_mat_C0_2_temp:np.ndarray = np.dot(frame_C0.T, np.linalg.inv(frame_scene_temp.T))
        # 排序
        arucos = np.hstack((arucos, np.ones((arucos.shape[0],1))))
        # center = np.append(center, [1.0])
        arucos_in_temp = trans_mat_C0_2_temp.dot(arucos.T).T[:,:3]
        # # ax = plt.axes(projection='3d')  # 设置三维轴
        # # ax.scatter(arucos_in_temp[:, 0], arucos_in_temp[:, 1], arucos_in_temp[:, 2])
        # # plt.show()

        # center_in_temp = trans_mat_C0_2_temp.dot(center)[:3]
        # angles = np.arctan2(arucos_in_temp[:, 0], arucos_in_temp[:, 1])
        # sorted_index = np.argsort(angles)
        # arucos_in_temp = arucos_in_temp[sorted_index]
        # 找对角4个点
        distance = np.linalg.norm(arucos_in_temp, axis = -1)
        max_index = np.argsort(distance)[-4:]
        arucos_at_corner = arucos[max_index, :3] # in C0
        # 定义新矩形，长边平行于x轴，长宽为各点坐标极差
        vec_1 = arucos_at_corner[3] - arucos_at_corner[0]
        vec_2 = arucos_at_corner[2] - arucos_at_corner[1]
        vec_3 = arucos_at_corner[1] - arucos_at_corner[0]
        vec_4 = arucos_at_corner[2] - arucos_at_corner[3]
        parallel_length_1 = (np.linalg.norm(vec_1) + np.linalg.norm(vec_2))/2
        parallel_length_2 = (np.linalg.norm(vec_3) + np.linalg.norm(vec_4))/2

        if parallel_length_1 > parallel_length_2:
            x_vec = vec_1 + vec_2
        else:
            x_vec = vec_3 + vec_4
        new_x_axis = x_vec / np.linalg.norm(x_vec)
        new_y_axis = np.cross(z_axis, new_x_axis)
        frame_scene =  np.array([   center, 
                            center + new_x_axis, 
                            center + new_y_axis, 
                            center + z_axis], np.float32) 
        frame_scene = np.hstack((frame_scene, np.ones((4,1))))
        trans_mat_C0_2_SCS:np.ndarray = np.dot(frame_C0.T, np.linalg.inv(frame_scene.T))
        # 检查平面方向
        # pcd = test_pcd.transform(trans_mat_C0_2_SCS)
        # o3d.visualization.SelectionPolygonVolume(test_pcd)
        points = np.array(test_pcd.points)
        points = trans_mat_C0_2_SCS.dot(np.hstack((points, np.ones((points.shape[0],1)))).T).T
        z_minus = np.sum(points[:, 2] < -10)
        z_plus  = np.sum(points[:, 2] > 10)
        if z_plus < z_minus:
            r_mat = Posture(rvec=np.array([np.pi, 0, 0])).trans_mat
            trans_mat_C0_2_SCS = r_mat.dot(trans_mat_C0_2_SCS)
        points = np.array(test_pcd.points)
        points = trans_mat_C0_2_SCS.dot(np.hstack((points, np.ones((points.shape[0],1)))).T).T
        z_minus = np.sum(points[:, 2] < -5)
        z_plus  = np.sum(points[:, 2] > 5)

        return aruco_centers, plane_equation, trans_mat_C0_2_SCS

    def crop_sence(self, trans_mat, arucos_SCS):
        # pointcloud_path = os.path.join(directory, REGIS_DIR, "merged.ply")
        # org_pcd = o3d.io.read_point_cloud(pointcloud_path)  
        # org_pcd = org_pcd.voxel_down_sample(voxel_size=0.001)
        org_pcd = self.model_manager.merged_regist_pcd_file.read(0)
        org_pcd.transform(trans_mat)
        w,h,_ = np.max(arucos_SCS, axis=0) - np.min(arucos_SCS, axis=0)

        ###
        bbp = [[-w/2,-h/2,0],[w/2,-h/2,0],[w/2,h/2,0],[-w/2,h/2,0]]
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Z"
        vol.axis_max = 500.0
        vol.axis_min = 0.0
        vol.bounding_polygon = o3d.utility.Vector3dVector(bbp)
        org_pcd = vol.crop_point_cloud(org_pcd)
        return org_pcd

    def get_floor_color(self, points:np.ndarray, colors:np.ndarray):
        '''
        get the mean color in hsv space
        '''
        assert points.shape[0] == colors.shape[0], "points and colors must have the same shape"
        in_index = np.where((points[:, 2] < 3) * (points[:, 2] > 0))
        floor_colors = colors[in_index]
        # cvt from rgb to hsv
        rgb_np = np.array(floor_colors)
        hsv_np = self._cvt_rgb_2_hsv(rgb_np)
        floor_color = np.mean(hsv_np, axis=0)
        return floor_color

    # @staticmethod
    # def _cvt_rgb_2_hsv(rgb_np:np.ndarray):
    #     # 如果颜色是0-1之间的浮点数，转化为0-255的uint8
    #     if rgb_np.max() <= 1:
    #         rgb_np = rgb_np * 255
    #     rgb_np = rgb_np.astype(np.uint8)
        hsv_np = np.apply_along_axis(colorsys.rgb_to_hsv, 1, rgb_np[:,0], rgb_np[:,1], rgb_np[:,2])
    #     return hsv_np

    @staticmethod
    def _cvt_rgb_2_hsv(rgb_array):
        # 将RGB数组的值映射到0到1的范围内
        # rgb_normalized = rgb_array / 255.0
        rgb_normalized = rgb_array
        # 找到最大值和最小值
        max_values = np.max(rgb_normalized, axis=1)
        min_values = np.min(rgb_normalized, axis=1)
        
        # 计算色调（H）
        delta = max_values - min_values
        hue = np.zeros(rgb_normalized.shape[0])
        r,g,b = rgb_array[:,0], rgb_array[:,1], rgb_array[:,2]
        np.seterr(divide='ignore', invalid='ignore')
        rc = (max_values-r) / delta
        gc = (max_values-g) / delta
        bc = (max_values-b) / delta
        np.seterr(divide='warn', invalid='warn')

        _rm = r == max_values
        _gm = g == max_values
        _bm = b == max_values
        _invaildm =  delta == 0

        hue[_rm] = (bc-gc)[_rm]
        hue[_gm] = (2.0+rc-bc)[_gm]
        hue[_bm] = (4.0+gc-rc)[_bm]
        hue[_invaildm] = 0.0
        hue = (hue/6.0) % 1.0

        # delta = max_values - min_values
        # hue = np.zeros(rgb_normalized.shape[0])
        
        # nonzero_indices = delta != 0
        # hue[nonzero_indices] = np.select(
        #     [max_values == rgb_normalized[nonzero_indices, 0],
        #     max_values == rgb_normalized[nonzero_indices, 1]],
        #     [60 * ((rgb_normalized[nonzero_indices, 1] - rgb_normalized[nonzero_indices, 2]) / delta[nonzero_indices]),
        #     60 * ((rgb_normalized[nonzero_indices, 2] - rgb_normalized[nonzero_indices, 0]) / delta[nonzero_indices]) + 120],
        #     60 * ((rgb_normalized[nonzero_indices, 0] - rgb_normalized[nonzero_indices, 1]) / delta[nonzero_indices]) + 240
        # )
        # hue[hue < 0] += 360
        
        # 计算饱和度（S）
        saturation = np.where(max_values != 0, (delta / max_values), 0)
        
        # 计算值（V）
        value = max_values
        
        # 组合HSV颜色数组
        hsv_array = np.column_stack((hue, saturation, value))
        
        return hsv_array

    @staticmethod
    def _filter_by_color(pcd, remove_color, tol, remove = True):
        '''
        按颜色过滤, RGB

        remove_color: hsv
        '''
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        hsv_colors = PcdCreator._cvt_rgb_2_hsv(colors)
        in_array =  (hsv_colors < remove_color + tol) * \
                        (hsv_colors > remove_color - tol)
        if remove:
            reserve_index = np.where(np.logical_not(in_array))[0]
        else:
            reserve_index = np.where(in_array)[0]
        points = points[reserve_index]
        colors = colors[reserve_index]
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points)
        new_pcd.colors = o3d.utility.Vector3dVector(colors)
        return new_pcd

    @staticmethod
    def _get_floor_points(pcd, band_height = 2, offset = 5):
        points = np.array(pcd.points)
        body = points[points[:, 2] > points[:, 2].max()*0.2]
        rect_bias = 10 #矩形裁剪的偏移
        bounding_box = [[body[:,0].min() - rect_bias, body[:,1].min() - rect_bias, 0], 
                        [body[:,0].min() - rect_bias, body[:,1].max() + rect_bias, 0], 
                        [body[:,0].max() + rect_bias, body[:,1].max() + rect_bias, 0], 
                        [body[:,0].max() + rect_bias, body[:,1].min() - rect_bias, 0]]
        refine_vol = o3d.visualization.SelectionPolygonVolume()
        refine_vol.orthogonal_axis = "Z"
        refine_vol.axis_max = 500
        refine_vol.axis_min = 0
        refine_vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_box)

        comp = refine_vol.crop_point_cloud(pcd) #裁剪
        points = np.array(comp.points)
        distances = comp.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances) 
        nearfloor_points = points[(points[:,2]< (band_height*0.5)) * (points[:,2]>- band_height*0.5)]
        box, box_color, restore_mat = pc_voxelize(nearfloor_points, band_height)
        image = box[:,:,0]
        not_image = np.logical_not(image).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        image = cv2.morphologyEx(not_image, cv2.MORPH_OPEN, kernel, iterations=3)
        dilate_times = 0
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_center = (np.array(image.shape) - 1) / 2
        valid_center_r = (np.min(np.array(image.shape)) - (2*rect_bias/band_height))/2 # 允许的边缘最大半径
        for c in contours:
            sqz_c = np.squeeze(c, axis=1)
            c_center = np.mean(sqz_c, axis=0)
            center_distance = np.linalg.norm(image_center - c_center, axis=-1)
            if center_distance > valid_center_r:
                image = cv2.drawContours(image, [c], -1, 0, -1)
        while len(cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]) > 1:
            image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
            dilate_times += 1
        image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=dilate_times)
        contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image = cv2.drawContours(image, contour, -1, 1, -1)
        # 根据avg_dist缩放，确保填充的底面点云密度和其他区域相等
        resize_r = 1/avg_dist
        image = cv2.resize(image, (0, 0), fx=resize_r, fy=resize_r)
        floor_points = np.array(np.where(image)).T #[N，3]
        floor_points = np.hstack((  floor_points, 
                                    np.zeros((floor_points.shape[0], 1)), 
                                    np.ones((floor_points.shape[0], 1)) ))
        floor_points[:, :3] = floor_points[:, :3] / resize_r
        floor_points = restore_mat.dot(floor_points.T).T [:, :3]
        floor_points = floor_points# / 1000
        thickened = []
        z_num = int(np.ceil(offset/ avg_dist))
        for z in np.linspace(0, z_num * avg_dist, z_num, endpoint=False):
            fp = floor_points.copy()
            fp[:, 2] = z
            thickened.append(fp)
        floor_points = np.vstack(thickened)
        colors = np.mean(box_color, (0,1,2))
        floor_point_colors = np.zeros(floor_points.shape)
        floor_point_colors[:] = colors
        return floor_points, floor_point_colors, refine_vol


class Aruco_Voronoi():
    def __init__(self, pcd_creator:PcdCreator, trans_mat) -> None:
        self.pcd_creator = pcd_creator
        process_data = self.pcd_creator.process_data
        self.aruco_used_times = process_data[process_data.ARUCO_USED_TIMES]

        C0_aruco_3d_dict = self.pcd_creator.aruco_detector.C0_aruco_3d_dict
        self.arcuo_centers = {}
        centers = []
        for id, points in C0_aruco_3d_dict.items():
            center = np.mean(points, axis=0)
            center = np.append(center, 1)
            center_SCS = trans_mat.dot(center)
            self.arcuo_centers.update({int(id): center_SCS})
            centers.append(center_SCS)
        centers = np.array(centers)
        max_x = np.max(centers[:,0])
        max_y = np.max(centers[:,1])
        min_x = np.min(centers[:,0])
        min_y = np.min(centers[:,1])
        bbox = np.array([[max_x, max_y], [max_x, min_y], [min_x, min_y], [min_x, max_y]])
        self.voronoi_poly = bounded_voronoi(bbox, centers[:, :2])
        # for p,c  in zip(self.voronoi_poly, centers):
        #     p = np.array(p)
        #     plt.scatter(p[:,0], p[:,1], c = 'g')
        #     plt.scatter(c[0], c[1], c = 'r')
        #     plt.show()
        self.o3d_vols = {}
        for polygon, id in zip(self.voronoi_poly, self.arcuo_centers.keys()):
            # 将二维多边形扩展为三维
            polygon = np.array(polygon)
            bounding_polygon = polygon.copy()
            bounding_polygon = np.hstack((bounding_polygon, np.zeros((bounding_polygon.shape[0], 1))))
            vol = o3d.visualization.SelectionPolygonVolume()
            vol.orthogonal_axis = "Z"
            vol.axis_max = 500
            vol.axis_min = 0
            vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
            self.o3d_vols.update({id: vol})

    def crop(self, pcd, ids):
        in_index = []
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        for id in ids:
            vol = self.o3d_vols[id]
            ii = vol.crop_in_polygon(pcd)
            in_index += ii
        # out_index = np.setdiff1d(np.arange(points.shape[0]), in_index)
        points = points[in_index]
        colors = colors[in_index]
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points)
        new_pcd.colors = o3d.utility.Vector3dVector(colors)
        return new_pcd
    
    def crop_by_aruco_num(self, pcd, name:str):
        aruco_used_num = self.aruco_used_times[name]
        ids = np.array(list(aruco_used_num.keys()), np.int32)
        used_num = np.array(list(aruco_used_num.values()), np.int32)
        rank = np.argsort(used_num)
        used_num = used_num[rank]
        ids = ids[rank]
        mean_used_num = np.mean(used_num)
        ok_ids = ids[np.where(used_num > mean_used_num)[0]]
        if len(ok_ids) > 4:
            ok_ids = ok_ids[-4:]
        elif len(ok_ids) < 3:
            ok_ids = ids[-3:]
        return self.crop(pcd, ok_ids)

class Interact_ChechSeg():
    def __init__(self, model_pcds, model_names, top_rank_slice, o3d_vols) -> None:
        self.model_pcds     = model_pcds    
        self.model_names    = model_names   
        self.top_rank_slice = top_rank_slice
        self.o3d_vols       = o3d_vols      
        self.vol_idx        = -1
        self.vol            = None
        self.model_idx      = -1
        self.comp           = None

    def change_model(self, vis, inc = True):
        if inc:
            self.model_idx += 1
            self.model_idx = self.model_idx % len(self.model_pcds)
        name = self.model_names[self.model_idx]
        print(name, end = '')
        self.crop(vis)
        
    def crop(self, vis):
        if self.comp is not None:
            vis.clear_geometries()
        self.vol = self.o3d_vols[self.vol_idx]
        self.vol.axis_min = 0.000
        self.comp = self.vol.crop_point_cloud(self.model_pcds[self.model_idx]) #裁剪
        vis.add_geometry(self.comp)
    
    def confirm(self, vis = None):
        # 上一个
        if self.model_idx > -1:
            to_exchange = int(np.where(self.top_rank_slice == self.model_idx)[0])
            if to_exchange != self.vol_idx:
                print("交换：{} / {}".format(to_exchange, self.vol_idx))
            else:
                print(" √")                
            self.top_rank_slice[self.vol_idx], self.top_rank_slice[to_exchange] = \
                self.top_rank_slice[to_exchange], self.top_rank_slice[self.vol_idx] 

        self.vol_idx += 1
        try:
            self.vol = self.o3d_vols[self.vol_idx]
        except IndexError:
            # vis.destroy_window()
            # vis.close()
            print("已经结束，请手动关闭界面")
            return
        self.vol.axis_min = 0.000
        if vis is not None:
            self.model_idx = self.top_rank_slice[self.vol_idx]
            self.change_model(vis, False)
    
    def check_end(self, vis):
        if self.vol_idx == len(self.o3d_vols):
            vis.close()
            vis.destory()

    def start(self):
        '''
        press V to change model
        press N to confirm
        '''
        key_to_callback = {}
        key_to_callback[ord("V")] = self.change_model
        key_to_callback[ord("N")] = self.confirm
        # confirm()
        print("press N to start to check segmentation".center(50, "-"))
        print(self.start.__doc__)
        o3d.visualization.draw_geometries_with_key_callbacks([], key_to_callback, width=1080, height=720)
    
