# from . import DatasetFormat, DST, ClusterNotRecommendWarning, FRAMETYPE_DATA
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import open3d as o3d
from typing import Union, Callable, Generator, Any, TypeVar, Any, Generic
from warnings import warn
from functools import partial

from . import RGB_DIR, DEPTH_DIR, TRANS_DIR, ARUCO_FLOOR, FRAMETYPE_DATA
from . import DatasetNode, UnifiedFileCluster, DisunifiedFileCluster, DictLikeCluster, UnifiedFilesHandle, DisunifiedFilesHandle, DictLikeHandle, JsonIO, Dataset, DictFile, FilesCluster
from ..data.dataCluster import FHT, UFC, DFC, DLC, UFH, DFH, DLFH, VDMT 

VDMT = TypeVar("VDMT", bound=Any)

class FrameMeta():
    def __init__(self, trans_mat_Cn2C0, rgb = None, depth = None, intr_M = None) -> None:
        self.trans_mat_Cn2C0:np.ndarray = trans_mat_Cn2C0
        self.color:np.ndarray = rgb
        self.depth:np.ndarray = depth
        self.intr_M:np.ndarray = intr_M


class CommonData(DatasetNode[FilesCluster, "CommonData", VDMT], Generic[VDMT]):

    def init_clusters_hook(self):
        super().init_clusters_hook()


        self.std_models_corenames = [os.path.splitext(x)[0] for x in os.listdir(os.path.join(self.data_path, "../std_meshes")) if x.endswith(".ply")]
        
        self.aruco_floor_json       = DisunifiedFileCluster(self, "../", flag_name="aruco_floor_json") # 如何rebuild？跳过？
        json_file = DisunifiedFilesHandle.from_name(
            self.aruco_floor_json, ARUCO_FLOOR + ".json", read_func=JsonIO.load_json, write_func=JsonIO.dump_json)
        self.aruco_floor_json._set_fileshandle(0, json_file)
        self.aruco_floor_json.cache_priority = False

        self.aruco_floor_png        = DisunifiedFileCluster(self, "../", flag_name="aruco_floor_png")
        img_file        = DisunifiedFilesHandle.from_name(self.aruco_floor_png, ARUCO_FLOOR + ".png", read_func=cv2.imread, write_func=cv2.imwrite)
        longside_file   = DisunifiedFilesHandle.from_name(self.aruco_floor_png, ARUCO_FLOOR + "_long_side.txt", read_func=np.loadtxt, write_func=np.savetxt)
        self.aruco_floor_png._set_fileshandle(0, img_file)
        self.aruco_floor_png._set_fileshandle(1, longside_file)
        self.aruco_floor_png.cache_priority = False

        self.imu_calibration        = DisunifiedFileCluster[DisunifiedFilesHandle, "DisunifiedFileCluster", CommonData, dict](self, "../", True, flag_name="imu_calibration")
        imu_calibration_paras       = DisunifiedFilesHandle.from_name(self.imu_calibration, "imu_calibration.json", read_func=JsonIO.load_json, write_func=JsonIO.dump_json)
        self.imu_calibration._set_fileshandle(0, imu_calibration_paras)
        self.imu_calibration.cache_priority = False
        
        self.barycenter_dict        = DictFile(self, "../std_meshes", flag_name="barycenter_dict", file_name="barycenter.json")

class EnumElements(UnifiedFileCluster[UnifiedFilesHandle, "EnumElements", "ModelManager", VDMT]):
    @property
    def enums(self):
        return [x[(self.filllen + 1):] for x in self.dataset_node.std_meshes_names]

    def format_corename(self, data_i: int):
        if isinstance(data_i, (np.intc, np.integer)): # type: ignore
            data_i = int(data_i)

        if isinstance(data_i, int):
            enum_name:str = self.dulmap_id_name(data_i)
            data_i = data_i
        elif isinstance(data_i, str):
            # get key by value
            enum_name:str = data_i
            data_i = self.dulmap_id_name(data_i)
        else:
            raise TypeError("data_i must be int or str")

        idx_name = super().format_corename(data_i)

        return idx_name + '_' + enum_name
    
    def deformat_corename(self, corename: str) -> int:
        idx_name, enum_name = corename.split('_', maxsplit=1)
        data_i = super().deformat_corename(idx_name)
        # try:
        #     self.dataset_node.std_meshes
        # except AttributeError:
        #     pass
        # else:
        #     verify_data_i = self.dulmap_id_name(enum_name)
        #     assert verify_data_i == data_i, "data_i must be equal to idx_name"
        return data_i
    
    def dulmap_id_name(self, enum:Union[str, int]):
        if isinstance(enum, int):
            return self.enums[enum]
        elif isinstance(enum, str):
            return self.enums.index(enum)

    def cvt_key(self, key):
        key = super().cvt_key(key)
        if isinstance(key, str):
            return self.dulmap_id_name(key)
        else:
            return key

class DataRecorder(CommonData[FrameMeta], Dataset):
    SPLIT_PARA = {"default": []}

    @property
    def category_idx_range(self):
        return self.active_spliter.get_idx_dict()
            
    @property
    def current_category_index(self):
        return self.__categroy_index
    
    @property
    def current_category_name(self):
        return self.category_names[self.__categroy_index]
    
    @property
    def current_categroy_num(self):
        return len(self.category_idx_range[self.current_category_name])

    @property
    def is_all_recorded(self):
        return self.__categroy_index == len(self.category_names)

    # def _update_dataset(self, data_i = None):
    #     super()._update_dataset(data_i)
    #     if data_i is None:
    #         self.category_idx_range.clear()
    #         for name in self.category_names:
    #             self.category_idx_range.setdefault(name, [])
    #         for data_i in self.rgb_elements.keys():
    #             cate_i = self.get_category_idx(data_i)
    #             self.category_idx_range.setdefault(self.category_names[cate_i], []).append(data_i)
    #     else:
    #         self.category_idx_range[self.current_category_name].append(data_i)

    def update_overview(self, log_type, src, dst, value, cluster):
        super().update_overview(log_type, src, dst, value, cluster)
        if log_type == self.LOG_ADD:
            self.active_spliter.set_one(dst, self.current_category_name, True)
        elif log_type == self.LOG_REMOVE:
            self.active_spliter.set_one(dst, self.current_category_name, False)

    def rebuild(self, force = False):
        super().rebuild(force = force)
        # if self.num != self.spliter_group.num:
        with self.spliter_group.get_writer(True).allow_overwriting(True):
            for fh in self.rgb_elements.query_all_fileshandle():
                self.active_spliter.set_one(fh.get_key(), fh.sub_dir, True)

    def init_clusters_hook(self):
        super().init_clusters_hook()
        Dataset.init_clusters_hook(self)
        
        self.rgb_elements   = ElementsWithCategory(self, RGB_DIR, suffix='.png',
                                       read_func=cv2.imread,                                    
                                       write_func=cv2.imwrite)

        self.depth_elements = ElementsWithCategory(self,      DEPTH_DIR,    
                                       read_func= partial(cv2.imread, flags = cv2.IMREAD_ANYDEPTH),   
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.trans_elements = ElementsWithCategory(self,    TRANS_DIR,
                                       read_func=np.load,
                                       write_func=np.save,
                                       suffix='.npy')

        self.category_names:list[str] = list([x[(self.rgb_elements.filllen + 1):] for x in self.std_models_corenames])
        self.category_names.insert(0, "global_base_frames") # 在标准模型列表的第一位插入"global_base_frames"
        self.category_names.insert(1, "local_base_frames") # 在标准模型列表的第二位插入"local_base_frames"
        self.category_names.append("dataset_frames") # 在标准模型列表的最后一位插入"dataset_frames"

        for name in self.category_names:
            self.active_spliter.add_subset(name)
        self.active_spliter.exclusive = True

        self.intr_file:DisunifiedFileCluster[DisunifiedFilesHandle, DisunifiedFileCluster, DataRecorder, dict] =\
              DisunifiedFileCluster(self, "", flag_name="intr_file")
        intr_0_file         = DisunifiedFilesHandle.from_name(self.intr_file, "intrinsics_0.json", read_func=JsonIO.load_json, write_func=JsonIO.dump_json)
        intr_1_file         = DisunifiedFilesHandle.from_name(self.intr_file, "intrinsics_1.json", read_func=JsonIO.load_json, write_func=JsonIO.dump_json)
        self.intr_file._set_fileshandle(0, intr_0_file)
        self.intr_file._set_fileshandle(1, intr_1_file)
        self.intr_file.cache_priority = False

    def init_dataset_attr_hook(self):
        super().init_dataset_attr_hook()
        self.__categroy_index = 0
        self.AddNum = 0 # 当前标准模型已采集的增量帧数
        self.skip_segs = []

    def inc_idx(self):
        self.__categroy_index += 1
        self.__categroy_index = min(self.__categroy_index, len(self.category_names))
        self.AddNum = 0

    def dec_idx(self):
        self.__categroy_index -= 1
        self.__categroy_index = max(self.__categroy_index, 0)
        self.AddNum = 0

    def clear_skip_segs(self):
        self.skip_segs.clear()

    def add_skip_seg(self, seg):
        assert isinstance(seg, int), "seg must be int"
        if seg > 0 and seg <= len(self.category_names):
            self.skip_segs.append(seg)
        if seg < 0 and seg > -len(self.category_names):
            self.skip_segs.append(len(self.category_names) + seg)

    def skip_to_seg(self):
        '''
        跳过，直到分段点
        '''
        try:
            skip_to = self.skip_segs.pop(0)
            skip_to = min(skip_to, len(self.category_names))
            skip_to = max(skip_to, self.__categroy_index)
        except:
            skip_to = len(self.category_names)
        while self.__categroy_index < skip_to:
            self.inc_idx()
            
    def get_category_idx(self, data_i):
        idx_range = self.category_idx_range
        for subset_name in idx_range:
            if data_i in idx_range[subset_name]:
                return self.category_names.index(subset_name)
            self.rgb_elements.num
        raise IndexError("data_i not in category_idx_range")

    def read(self, data_i, *, force = False, **other_paras) -> FrameMeta:
        read_dict = self.raw_read(data_i, force=force, **other_paras)

        rgb   = read_dict[self.rgb_elements.identity_name()]
        depth = read_dict[self.depth_elements.identity_name()]
        trans = read_dict[self.trans_elements.identity_name()]

        if data_i in self.category_idx_range[FRAMETYPE_DATA]:
            intr_M = self.intr_file.read(1)
        else:
            intr_M = self.intr_file.read(0)

        return FrameMeta(trans_mat_Cn2C0=trans, rgb=rgb, depth=depth, intr_M=intr_M)
    
    def read_in_category_range(self, start, end):
        valid_category = list(range(len(self.category_names)))[start:end]
        for i in range(self.num):
            category_idx:int = self.get_category_idx(i)
            if category_idx in valid_category:
                framemeta:FrameMeta = self.read(i) #TODO
                yield category_idx, framemeta

    def write(self, dst:int, value:FrameMeta, *, force = False, **other_paras) -> None:
        with self.get_writer(force).allow_overwriting(force):
            sub_dir = self.category_names[self.__categroy_index]
            self.rgb_elements.write(dst,    value.color,            sub_dir=sub_dir)
            self.depth_elements.write(dst,  value.depth,            sub_dir=sub_dir)
            self.trans_elements.write(dst,  value.trans_mat_Cn2C0,  sub_dir=sub_dir)
            # self.active_spliter.set_one(dst, sub_dir, True)
            self.AddNum += 1

    def save_frames(self, c, d, t):
        framemeta = FrameMeta(t, c, d)
        data_i = self.i_upper
        self.write(data_i, framemeta, force=True)

    # def remove(self, remove_list:list, change_file = True):
    #     pass

    # def insert(self, insert_list:list, change_file = True):
    #     pass

    # def rename_all(self, exchange_pair = []):
    #     pass

    # def make_directories(self):
    #     pass
    #     # if os.path.exists(self.rgb_dir):
    #     #     return
    #     # else:
    #     #     for d in [self.rgb_dir, self.depth_dir, self.trans_dir]:
    #     #         try:
    #     #             shutil.rmtree(d)
    #     #         except:
    #     #             pass
    #     #         os.makedirs(d)
    #     #     with open(self.directory+'category_idx_range.json', 'w') as fp:
    #     #         json.dump(self.model_index_dict, fp)

class ElementsWithCategory(UnifiedFileCluster[UnifiedFilesHandle, "ElementsWithCategory", DataRecorder, np.ndarray]):
    @property
    def current_category_range(self):
        return self.dataset_node.category_idx_range[self.dataset_node.current_category_name]

    def in_current_category(self):
        try:
            _range = self.current_category_range
        except:
            _range = [] 
        for data_i in _range:
            yield self.read(data_i)

class ModelManager(CommonData):

    ARUCO_USED_TIMES = "aruco_used_times"
    ARUCO_CENTERS = "aruco_centers"
    PLANE_EQUATION = "plane_equation"
    TRANS_MAT_C0_2_SCS = "trans_mat_C0_2_SCS"
    VOR_POLYS_COORD = "vor_polys_coord"
    FLOOR_COLOR = "floor_color"

    @property
    def std_meshes_dir(self):
        return self.std_meshes.data_path
    
    @property
    def std_meshes_names(self) -> tuple[str]:
        std_meshes_names = tuple([fh.corename for fh in self.std_meshes.query_all_fileshandle()])
        return std_meshes_names
    
    @property
    def std_meshes_enums(self):
        return self.std_meshes.enums

    def init_clusters_hook(self):
        super().init_clusters_hook()

        self.std_meshes         = EnumElements(self, "../std_meshes", suffix='.ply', flag_name="std_meshes",
                                    read_func=o3d.io.read_triangle_mesh,
                                    write_func=o3d.io.write_triangle_mesh,
                                    )
        self.std_meshes_image   = EnumElements(self, "../std_meshes", suffix='.jpg', flag_name="std_meshes_image",
                                    read_func=cv2.imread,
                                    write_func=cv2.imwrite)

        self.registerd_pcd  = EnumElements(self, "registerd_pcd", suffix='.ply',
                                        read_func=o3d.io.read_point_cloud,
                                        write_func=o3d.io.write_point_cloud)
        self.voronoi_segpcd = EnumElements(self, "voronoi_segpcd", suffix='.ply',
                                        read_func=o3d.io.read_point_cloud,
                                        write_func=o3d.io.write_point_cloud)
        self.extracted_mesh = EnumElements(self, "extracted_mesh", suffix='.ply',
                                        read_func=o3d.io.read_triangle_mesh,
                                        write_func=o3d.io.write_triangle_mesh)
        self.icp_trans      = EnumElements(self, "icp_trans", suffix='.npy',
                                        read_func=np.load,
                                        write_func=np.save)
        self.icp_std_mesh   = EnumElements(self, "icp_std_mesh", suffix='.ply',
                                        read_func = o3d.io.read_triangle_mesh,
                                        write_func= o3d.io.write_triangle_mesh)
        self.icp_unf_pcd    = EnumElements(self, "icp_unf_pcd", suffix='.ply',
                                        read_func=o3d.io.read_point_cloud,
                                        write_func=o3d.io.write_point_cloud)        
        
        self.merged_regist_pcd_file = DisunifiedFileCluster(self, "", flag_name="merged_regist_pcd_file")
        merged_file = DisunifiedFilesHandle.from_name(self.merged_regist_pcd_file, "merged.ply", read_func = o3d.io.read_point_cloud, write_func= o3d.io.write_point_cloud)
        self.merged_regist_pcd_file._set_fileshandle(0, merged_file)

        for cluster in self.clusters:
            cluster.write_synchronous = False
            cluster.cache_priority = False

        self.process_data = ProcessData(self, "", file_name="process_data.json")

class ProcessData(DictFile[DisunifiedFilesHandle, "ProcessData", ModelManager]):

    ARUCO_USED_TIMES = "aruco_used_times"
    ARUCO_CENTERS = "aruco_centers"
    PLANE_EQUATION = "plane_equation"
    TRANS_MAT_C0_2_SCS = "trans_mat_C0_2_SCS"
    VOR_POLYS_COORD = "vor_polys_coord"
    FLOOR_COLOR = "floor_color"

    
