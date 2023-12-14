import os
import numpy as np
import open3d as o3d
import cv2
import scipy.ndimage as ndimage
from . import Posture, JsonIO, modify_class_id, get_meta_dict

from typing import Union

class MeshMeta:
    def __init__(self,
                 mesh, 
                 bbox_3d: np.ndarray = None, 
                 symmetries:dict = None, 
                 diameter:float = None,  
                 ldmk_3d: np.ndarray = None,
                 name = "",
                 class_id = -1) -> None:
        self.mesh = mesh
        self.bbox_3d: np.ndarray    = bbox_3d
        self.symmetries:dict        = symmetries #"symmetries_continuous": "symmetries_discrete": 
        self.diameter: float        = diameter
        self.ldmk_3d: np.ndarray    = ldmk_3d

        self.name = name
        self.class_id = class_id

    @property
    def points_array(self):
        return np.asarray(self.mesh.vertices)
    
    @property
    def normals_array(self):
        return np.asarray(self.mesh.vertex_normals)

    @property
    def tris_array(self):
        return np.asarray(self.mesh.triangles)

    def transform(self, posture:Union[Posture, np.ndarray], copy = True):
        assert isinstance(posture, (Posture, np.ndarray)), "posture must be Posture or np.ndarray"
        if isinstance(posture, np.ndarray):
            posture = Posture(homomat=posture)

        new_bbox = posture * self.bbox_3d if self.bbox_3d is not None else None
        new_ldmk = posture * self.ldmk_3d if self.ldmk_3d is not None else None

        if copy:
            new_mesh = o3d.geometry.TriangleMesh(self.mesh)
            new_mesh = new_mesh.transform(posture.trans_mat)

            return MeshMeta(new_mesh, new_bbox, self.symmetries, self.diameter, new_ldmk, self.name, self.class_id)
        else:
            self.mesh.transform(posture.trans_mat)
            if new_bbox is not None:
                self.bbox_3d[:] = new_bbox
            else:
                self.bbox_3d = None
            if new_ldmk is not None:
                self.ldmk_3d[:] = new_ldmk
            else:
                self.ldmk_3d = None
            return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.name} at {id(self)}"

class MeshManager:
    _instance = None

    def __new__(cls, *arg, **kw):
        if not cls._instance:
            cls._instance = super(MeshManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, root, model_names: dict[int, str] = {}, load_all = False, modify_class_id_pairs:list[tuple[int]]=[]) -> None:
        self.root = root
        self.model_names: dict[int, str] = model_names
        self.__model_name_json = os.path.join(self.root, "models_name.json")
        self.model_dirs : dict[int, str] = {}
        if len(model_names) > 0:
            model_src = self.model_names.items()
            JsonIO.dump_json(self.__model_name_json, self.model_names)
        elif os.path.exists(self.__model_name_json):
            self.model_names = JsonIO.load_json(self.__model_name_json)
            model_src = self.model_names.items()
        else:
            model_src = enumerate([x for x in os.listdir(self.root) if os.path.splitext(x)[-1] == ".ply"])
        for id_, name in model_src:
            # 模型名称
            path = os.path.join(self.root, name)
            self.model_dirs.update({id_: path}) 
            name = os.path.splitext((os.path.split(name)[-1]))[0]
            self.model_names[id_] = name

        self.landmark_info_path = os.path.join(self.root, "landmarks.json")
        self.models_info_path = os.path.join(self.root, "models_info.json")
        
        self.model_meshes       = {}
        self.model_bbox_3d          = {}
        self.model_symmetries = {} #"symmetries_continuous": "symmetries_discrete": 
        self.model_diameter = {}
        self.model_ldmk_3d = {}
        self.load_landmarks(self.landmark_info_path)
        self.load_models_info(self.models_info_path)
        if load_all or len(modify_class_id_pairs)>0:
            for key in self.model_dirs:
                if key not in self.model_meshes:
                    self.load_model(key)
        if len(modify_class_id_pairs)>0:
            self.modify_class_id(modify_class_id_pairs)

    @property
    def class_num(self):
        return len(self.model_dirs)

    def modify_class_id(self, modify_class_id_pairs):
        orig_dict_list = get_meta_dict(self)
        modify_class_id(orig_dict_list, modify_class_id_pairs)

    @staticmethod
    def farthest_point_sample(point_cloud, npoint): 
        """
        Input:
            point_cloud: pointcloud data, [N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """

        N = point_cloud.shape[0]
        centroids = np.zeros((npoint, 3))    # 采样点矩阵
        farthest_idx_array = np.zeros(npoint)
        distance = np.ones((N)) * 1e10    # 采样点到所有点距离（npoint, N）    
        
        #计算重心
        center = np.mean(point_cloud, 0) # [3]
        # center = np.array([0,0,0])
        # 计算距离重心最远的点
        dist = np.sum((point_cloud - center) ** 2, -1)
        farthest_idx = np.argmax(dist)                                     #将距离重心最远的点作为第一个点，这里跟torch.max不一样
        for i in range(npoint):
            # print("-------------------------------------------------------")
            # print("The %d farthest point %s " % (i, farthest_idx))
            centroid = point_cloud[farthest_idx, :]             # 取出这个最远点的xyz坐标
            centroids[i, :] = centroid                          # 更新第i个最远点
            farthest_idx_array[i] = farthest_idx
            dist = np.sum((point_cloud - centroid) ** 2, -1)                     # 计算点集中的所有点到这个最远点的欧式距离，-1消掉了xyz那个维度
            # print("dist    : ", dist)
            mask = dist < distance
            # print("mask %i : %s" % (i,mask))
            distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点（已采样集合中的点）的最小距离
            # print("distance: ", distance)

            farthest_idx = np.argmax(distance)                           # 返回最远点索引

        return centroids, farthest_idx_array

    def load_model(self, model_id:int):
        # 模型路径下的名称目录
        mesh = o3d.io.read_triangle_mesh(self.model_dirs[model_id])
        mesh.normalize_normals()
  
        self.model_meshes.update({model_id: mesh})   

    def load_models_info(self, models_info_path: str):
        '''
          1_______7
         /|      /|         Z
        3_______5 |         |__Y 
        | 0_____|_6        /
        |/      |/        X
        2_______4        
        '''
        if os.path.exists(models_info_path):
            self.model_diameter:dict[int,float] = {} ### 模型直径
            self.model_bbox_3d:dict[int,np.ndarray] = {} ### 模型包围盒
            self.model_symmetries:dict[int, dict] = {}
            # with open(models_info_path, 'r') as MI:
            #     info = json.load(MI)
            info = JsonIO.load_json(models_info_path)
            for k,v in info.items():
                self.model_diameter.update({k: v["diameter"]})
                min_x =  info[k]["min_x"]
                min_y =  info[k]["min_y"]
                min_z =  info[k]["min_z"]
                size_x = info[k]["size_x"]
                size_y = info[k]["size_y"]
                size_z = info[k]["size_z"]
                # 计算顶点坐标并以ndarray返回
                max_x = min_x + size_x
                max_y = min_y + size_y
                max_z = min_z + size_z
                # 计算顶点坐标并以ndarray返回
                #   0,  1,  2,  3,  4,  5,  6,  7
                x =np.array([-1,-1, 1, 1, 1, 1,-1,-1]) * max_x
                y =np.array([-1,-1,-1,-1, 1, 1, 1, 1]) * max_y
                z =np.array([-1, 1,-1, 1,-1, 1,-1, 1]) * max_z
                vertex = np.vstack((x, y, z)).T
                self.model_bbox_3d.update({k: vertex}) #[8, 3]
                # 对称属性
                for symm in ["symmetries_continuous", "symmetries_discrete"]:
                    if symm in info[k]:
                        self.model_symmetries.update({k: info[k][symm]})
            self.models_info_path = models_info_path
        else:
            print("Warning: models_info_path doesn't exist")

    def load_landmarks(self, landmark_info_path: str):
        recalc = True
        if os.path.exists(landmark_info_path):
            self.landmark_info_path = landmark_info_path
            self.model_ldmk_3d:dict[int,np.ndarray] = JsonIO.load_json(landmark_info_path)
            if set(self.model_ldmk_3d.keys()) == set(self.model_dirs.keys()):
                recalc = False
        if recalc:
            for class_id in self.model_dirs:
                points = self.get_model_pcd(class_id)
                centroids, farthest_idx_array = self.farthest_point_sample(points, 24)
                self.model_ldmk_3d.update({class_id: centroids})
            JsonIO.dump_json(landmark_info_path, self.model_ldmk_3d)
            print("Warning: landmark_info_path doesn't exist, calculated")

    def get_bbox_3d(self, class_id:int):
        bbox_3d = self.model_bbox_3d[class_id].copy()
        return bbox_3d

    def get_ldmk_3d(self, class_id:int):
        ldmk_3d = self.model_ldmk_3d[class_id].copy()
        return ldmk_3d

    def get_model_name(self, class_id:int) -> str:
        if class_id not in self.model_names:
            self.load_model(class_id)
        name = self.model_names[class_id]
        return name

    def get_model_pcd(self, class_id:int) -> np.ndarray:
        if class_id not in self.model_meshes:
            self.load_model(class_id)
        pcd = np.asarray(self.model_meshes[class_id].vertices)
        return pcd

    def get_model_normal(self, class_id:int) -> np.ndarray:
        if class_id not in self.model_meshes:
            self.load_model(class_id)
        normal = np.asarray(self.model_meshes[class_id].vertex_normals)
        return normal

    def get_model_mesh(self, class_id:int) -> np.ndarray:
        if class_id not in self.model_meshes:
            self.load_model(class_id)
        mesh = o3d.geometry.TriangleMesh(self.model_meshes[class_id])
        return mesh

    def get_model_diameter(self, class_id:int):
        diameter = self.model_diameter[class_id]
        return diameter

    def get_model_symmetries(self, class_id:int):
        if class_id not in self.model_symmetries:
            return None
        else:
            return self.model_symmetries[class_id].copy()
    
    def export_meta(self, class_id:int):
        mesh                = self.get_model_mesh(class_id)
        bbox_3d     = self.get_bbox_3d(class_id)
        symmetries  = self.get_model_symmetries(class_id) #"symmetries_continuous": "symmetries_discrete": 
        diameter: float    = self.get_model_diameter(class_id)
        ldmk_3d     = self.get_ldmk_3d(class_id)

        name = self.get_model_name(class_id)

        return MeshMeta(mesh, bbox_3d, symmetries, diameter, ldmk_3d, name = name, class_id = class_id)
    
    def get_meta_dict(self):
        meta_dict = {}
        for key in self.model_dirs:
            meta = self.export_meta(key)
            meta_dict[key] = meta
        return meta_dict

def get_bbox_connections(bbox_3d_proj:np.ndarray):
    '''
      1_______7
     /|      /|         Z
    3_______5 |         |__Y 
    | 0_____|_6        /
    |/      |/        X
    2_______4        

    params
    -----
    bbox_3d_proj: [..., B, (x,y)]

    return
    -----
    lines: [..., ((x1,x2), (y1,y2)), 12]
    '''
    b = bbox_3d_proj
    lines = [
    ([b[...,0,0], b[...,1,0]], [b[...,0,1], b[...,1,1]]),
    ([b[...,0,0], b[...,6,0]], [b[...,0,1], b[...,6,1]]),
    ([b[...,6,0], b[...,7,0]], [b[...,6,1], b[...,7,1]]),
    ([b[...,1,0], b[...,7,0]], [b[...,1,1], b[...,7,1]]),

    ([b[...,2,0], b[...,3,0]], [b[...,2,1], b[...,3,1]]),
    ([b[...,2,0], b[...,4,0]], [b[...,2,1], b[...,4,1]]),
    ([b[...,4,0], b[...,5,0]], [b[...,4,1], b[...,5,1]]),
    ([b[...,3,0], b[...,5,0]], [b[...,3,1], b[...,5,1]]),

    ([b[...,0,0], b[...,2,0]], [b[...,0,1], b[...,2,1]]),
    ([b[...,1,0], b[...,3,0]], [b[...,1,1], b[...,3,1]]),
    ([b[...,7,0], b[...,5,0]], [b[...,7,1], b[...,5,1]]),
    ([b[...,6,0], b[...,4,0]], [b[...,6,1], b[...,4,1]]),
    ]
    lines = np.stack(lines)
    return lines #[12, ..., ((x1,x2), (y1,y2))]

class Voxelized():
    def __init__(self, 
                 entity_cube:np.ndarray,
                 restore_mat:np.ndarray,
                 orig_geometry = None) -> None:
        self.entity_cube         = entity_cube.astype(np.uint8)
        self.restore_mat = restore_mat          
        self.orig_geometry = orig_geometry
      
        self.surf_cube, self.erode_entity_cube = self.split_entity_cube(entity_cube)
        
        self.entity_indices =\
              np.array(np.where(self.entity_cube)).T #[N, 3]
        self.surf_indices =\
              np.array(np.where(self.surf_cube)).T
        
        self.surf_points =\
              self.restore_mat[:3, :3].dot(self.surf_indices.T).T + self.restore_mat[:3, 3]
        self.surf_normals = self.calc_surf_normal()

        self.entity_query = np.full(self.entity_cube.shape, -1, np.int64)
        self.entity_query[self.entity_indices[:,0], 
                        self.entity_indices[:,1], 
                        self.entity_indices[:,2]] = np.array(range(self.entity_indices.shape[0]), np.int64)
        
        self.surf_query = np.full(self.surf_cube.shape, -1, np.int64)
        self.surf_query[self.surf_indices[:,0], 
                        self.surf_indices[:,1], 
                        self.surf_indices[:,2]] = np.array(range(self.surf_indices.shape[0]), np.int64)

    @property
    def shape(self):
        return self.entity_cube.shape
 
    def split_entity_cube(self, cube:np.ndarray):
        N = int(np.sum(cube.shape) / 20)
        surf_cube = np.zeros(cube.shape, np.uint8)
        for d in range(3):
            cube = np.swapaxes(cube, 0, d)
            surf_cube = np.swapaxes(surf_cube, 0, d)
            for i in range(cube.shape[0]):
                layer = cube[i]
                # 找外圈轮廓，并排除长度较小的轮廓
                # 查找轮廓
                contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # 绘制轮廓
                image = surf_cube[i].copy()
                for contour in contours:
                    # 排除长度小于N的轮廓
                    if len(contour) >= N:
                        cv2.drawContours(image, [contour], -1, 1, thickness=0)
                surf_cube[i] = image
            cube = np.swapaxes(cube, 0, d)
            surf_cube = np.swapaxes(surf_cube, 0, d)
        surf_cube = np.clip(surf_cube, 0, 1)
        eroded_body = (cube - surf_cube).astype(np.uint8)
        return surf_cube, eroded_body

    def calc_surf_normal(self):
        if isinstance(self.orig_geometry, o3d.geometry.TriangleMesh) and len(self.orig_geometry.vertex_normals) > 0:
            surf = o3d.geometry.PointCloud()
            surf.points = o3d.utility.Vector3dVector(self.surf_indices)

            ref_pcd = o3d.geometry.PointCloud()
            inv_restore_mat = np.linalg.inv(self.restore_mat)
            ref_pcd.points = self.orig_geometry.vertices
            ref_pcd.normals = self.orig_geometry.vertex_normals
            ref_pcd.transform(inv_restore_mat)

            # 构建 KD 树
            kdtree = o3d.geometry.KDTreeFlann()
            kdtree.set_geometry(ref_pcd)

            # 对每个点在 A 中进行最近邻搜索并插值法向
            k = 1  # 最近邻点的数量
            interpolated_normals = np.zeros(self.surf_indices.shape)

            ref_pcd_normals = np.asarray(ref_pcd.normals)
            for i in range(len(surf.points)):
                _, indices, _ = kdtree.search_knn_vector_3d(surf.points[i], k)
                interpolated_normals[i] = ref_pcd_normals[indices]

            # 归一化
            interpolated_normals = interpolated_normals / np.linalg.norm(interpolated_normals, axis=-1, keepdims=True)
            surf.normals = o3d.utility.Vector3dVector(interpolated_normals)

            return interpolated_normals
        else:
            raise NotImplementedError("Only support triangle mesh with vertex normals")
    
    def query_surf_normal(self, indices):
        idx = self.surf_query[indices[..., 0], indices[..., 1], indices[..., 2]]
        normals = self.surf_normals[idx]
        normals[np.any(idx == -1, -1)] = 0.0
        return normals
    
    def query_surf_points(self, indices):
        idx = self.surf_query[indices[..., 0], indices[..., 1], indices[..., 2]]
        points = self.surf_points[idx]
        points[np.any(idx == -1, -1)] = 0.0
        return points

    @staticmethod
    def _process_raw_voxel_grid(raw_voxel_grid: o3d.geometry.VoxelGrid):
        '''
        raw_voxel_grid: o3d.geometry.VoxelGrid
        '''
        _voxel_indices = np.asarray([x.grid_index for x in raw_voxel_grid.get_voxels()])

        ### 创建cube
        # 计算体素网格的尺寸
        voxel_dims = np.max(_voxel_indices, 0) + 1
        # 创建全0的三维矩阵
        _entity = np.zeros(voxel_dims, dtype=np.uint8)
        _entity[_voxel_indices[:, 0], _voxel_indices[:, 1], _voxel_indices[:, 2]] = 1

        ### 开运算，填充大部分缝隙
        _entity = np.pad(_entity, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        kernel = ndimage.generate_binary_structure(3,2)
        _entity = ndimage.binary_closing(_entity, kernel, iterations=1).astype(np.uint8)
        _entity = _entity[1:-1, 1:-1, 1:-1]

        ### 填充实体       
        entity = Voxelized.fill_3d(_entity)

        ### 过滤部分离群点
        idx = np.array(np.where(entity)).T
        pcd = o3d.geometry.PointCloud()
        points = idx.astype(np.float32)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, _ = pcd.remove_statistical_outlier(60, 2)

        ### 
        final_indices = np.asarray(pcd.points, np.int64)

        entity = np.zeros(voxel_dims, dtype=np.uint8)
        entity[final_indices[:,0], final_indices[:,1], final_indices[:,2]] = 1

        return entity

    @staticmethod
    def _get_restore_mat(origin:np.ndarray, voxel_size:float):
        restore_mat = np.eye(4)*voxel_size
        restore_mat[3, 3] = 1
        restore_mat[:3, 3] = origin
        return restore_mat

    @staticmethod
    def auto_voxel_size(geometry) -> float:
        max_bound = geometry.get_max_bound()
        min_bound = geometry.get_min_bound()
        return max(max_bound - min_bound) / 30

    @staticmethod
    def from_mesh(o3d_mesh, voxel_size = None):
        if voxel_size is None:
            voxel_size = Voxelized.auto_voxel_size(o3d_mesh)
        ### 进行体素化
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size)
        
        ###retore_mat
        origin = voxel_grid.origin
        restore_mat = Voxelized._get_restore_mat(origin, voxel_size)
        entity = Voxelized._process_raw_voxel_grid(voxel_grid)

        return Voxelized(entity, restore_mat, orig_geometry = o3d_mesh)

    @staticmethod
    def from_pcd(o3d_pcd, voxel_size = None):
        if voxel_size is None:
            voxel_size = Voxelized.auto_voxel_size(o3d_pcd)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=voxel_size)
        
        ###retore_mat
        origin = voxel_grid.origin
        restore_mat = Voxelized._get_restore_mat(origin, voxel_size)
        
        ###entity
        entity = Voxelized._process_raw_voxel_grid(voxel_grid)

        return Voxelized(entity, restore_mat, orig_geometry = o3d_pcd)

    @staticmethod
    def fill_3d(voxel_array:np.ndarray):
        '''
        沿不同方向累加，累加值为奇数的是内部
        '''
        padded_array = np.pad(voxel_array, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        # padded_array = padded_array.astype(np.int8)
        cum_list = []
        for i in range(3):
            padding = tuple([(1,0) if p == i else (0,0) for p in range(3)])
            padded_array = np.pad(voxel_array, padding, mode='constant', constant_values=0)
            diff = np.diff(padded_array, axis = i)
            # diff = np.swapaxes(diff, 0, i)[:-1]
            # diff = np.swapaxes(diff, 0, i)
            diff = diff > 0
            cum = (np.cumsum(diff, axis=i) / 2).astype(np.uint16)
            cum_list.append(cum)
        cum = np.stack(cum_list) # [3, W, H, D]
        odd = np.mod(cum, 2) == 1 # [3, W, H, D]
        in_voting = np.sum(odd, axis=0) > 2

        entity = voxel_array.copy()
        entity[in_voting] = 1
        return entity
