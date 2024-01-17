from typing import Callable

from .dataset import Dataset, DatasetNode, Mix_Dataset, Spliter, FCT, DST, VDST
from .dataCluster import UnifiedFileCluster, DisunifiedFileCluster, DictLikeCluster, UnifiedFilesHandle, DisunifiedFilesHandle, DictLikeHandle,\
    IntArrayDictAsTxtCluster, NdarrayAsTxtCluster
from .IOAbstract import ClusterWarning, FilesCluster, get_with_priority, IO_CTRL_STRATEGY
from .viewmeta import ViewMeta
from ..core.utils import deserialize_object, serialize_object, rebind_methods
from ..core.posture import Posture

import numpy as np
import cv2
import warnings
from functools import partial

class PostureDataset(Mix_Dataset[FCT, DST, VDST]):
    POSTURE_SPLITER_NAME = "posture"
    POSTURE_SUBSETS = ["train", "val"]


    SPLIT_PARA = Mix_Dataset.SPLIT_PARA.copy()
    SPLIT_PARA.update(
        {
            POSTURE_SPLITER_NAME: POSTURE_SUBSETS,
            "aug_posture": POSTURE_SUBSETS,
        }
    )

    def split_for_posture(self, ratio = 0.15, source:list[int] = None):
        """
        Only take ratio of the real data as the verification set
        """
        # assert self.reality_spliter.total_num == self.data_num, "reality_spliter.total_num != data_num"
        # assert self.basis_spliter.total_num == self.data_num, "basis_spliter.total_num != data_num"
        
        real_idx = self.reality_spliter.get_idx_list(0)
        sim_idx  = self.reality_spliter.get_idx_list(1)

        real_base_idx = list(set(real_idx).intersection(self.basis_spliter.get_idx_list(self.BASIS_SUBSETS[0])))

        if source is None:
            pass
        else:
            real_idx = list(set(real_idx).intersection(source))
            real_base_idx = list(set(real_base_idx).intersection(source))
            sim_idx  = list(set(sim_idx).intersection(source))

        train_real, val_real = Spliter.gen_split(real_base_idx, ratio, 2)

        self.posture_spliter.set_one_subset(self.POSTURE_SUBSETS[1], val_real)

        posture_train_idx_list = np.setdiff1d(
            np.union1d(real_idx, sim_idx),
            self.posture_val_idx_array
            ).astype(np.int32).tolist()

        self.posture_spliter.set_one_subset(self.POSTURE_SUBSETS[0], posture_train_idx_list)

        self.posture_spliter.split_table.sort()

    @property
    def posture_spliter(self):
        return self.spliter_group.get_cluster(self.POSTURE_SPLITER_NAME)
    
    @property
    def posture_train_idx_array(self):
        return self.posture_spliter.get_idx_list(self.POSTURE_SUBSETS[0])
    
    @property
    def posture_val_idx_array(self):
        return self.posture_spliter.get_idx_list(self.POSTURE_SUBSETS[1])


class MutilMaskFilesHandle(UnifiedFilesHandle):
    pass

class MutilMaskCluster(UnifiedFileCluster[MutilMaskFilesHandle, "MutilMaskCluster", "BopFormat", dict[int, np.ndarray]]):
    
    MULTI_FILES = True

    DEFAULT_APPENDNAMES_JOINER = "_"
    DEFAULT_READ_FUNC = cv2.imread
    DEFAULT_WRITE_FUNC = cv2.imwrite
    DEFAULT_SUFFIX = ".png"

    KW_ID_SEQ = "id_seq"
    
    def _set_rely(self, relied: DictLikeHandle, rlt:dict):
        '''
        rlt = {
            "scene_camera": {KW_CAM_K: array, KW_cam_R_w2c: array, KW_cam_t_w2c: array, KW_CAM_DS:float}
            "scene_gt": [{KW_GT_R: array, KW_GT_t: array, KW_GT_ID: int}, ...]
        }
        '''
        ids = [x[BopFormat.KW_GT_ID] for x in rlt[1]]
        self._update_rely(self.KW_ID_SEQ, ids)

    class _read(UnifiedFileCluster._read["MutilMaskCluster", dict[int, np.ndarray], MutilMaskFilesHandle]):
        def gather_mutil_results(self, results:list):
            id_seq = self.files_cluster._get_rely(self.files_cluster.KW_ID_SEQ)
            argsort = np.argsort(id_seq)
            id_seq  = (np.array(id_seq, dtype=np.int32)[argsort]).tolist()
            results = np.array(results)[argsort]
            dict_rlt:dict[int, np.ndarray] = dict(zip(id_seq, results))
            return dict_rlt
        
        def cvt_to_core_paras(self, 
                              src_file_handle, 
                              dst_file_handle, 
                              value, 
                              **other_paras) -> tuple:
            paths, *paras = super().cvt_to_core_paras(src_file_handle, 
                                                    dst_file_handle,
                                                    value,
                                                    **other_paras)
            return sorted(paths), *paras
    
    class _write(UnifiedFileCluster._write["MutilMaskCluster", dict[int, np.ndarray], MutilMaskFilesHandle]):
        def split_value_as_mutil(self, core_values:dict[int, np.ndarray]):
            id_seq = self.files_cluster._get_rely(self.files_cluster.KW_ID_SEQ)
            value = [core_values[x] for x in id_seq]
            return value

class BopFormat(PostureDataset[UnifiedFileCluster, "BopFormat", ViewMeta]):
    '''
    info:
    rlt = {
        "scene_camera": {KW_CAM_K: array, KW_cam_R_w2c: array, KW_cam_t_w2c: array, KW_CAM_DS:float}
        "scene_gt": [{KW_GT_R: array, KW_GT_t: array, KW_GT_ID: int}, ...]
    }
    '''
    GT_CAM_FILE_NAME    = "scene_camera"
    GT_CAM_FILE     = GT_CAM_FILE_NAME + ".json"
    KW_cam_R_w2c = "cam_R_w2c"
    KW_cam_t_w2c = "cam_t_w2c"
    KW_CAM_K = "cam_K"
    KW_CAM_DS = "depth_scale"
    KW_CAM_VL = "view_level"

    GT_FILE_NAME        = "scene_gt"
    GT_FILE         = GT_FILE_NAME + ".json"
    KW_GT_R = "cam_R_m2c"
    KW_GT_t = "cam_t_m2c"
    KW_GT_ID = "obj_id"

    GT_INFO_FILE_NAME   = "scene_gt_info"
    GT_INFO_FILE    = GT_INFO_FILE_NAME + ".json"
    KW_GT_INFO_BBOX_OBJ = "bbox_obj"
    KW_GT_INFO_BBOX_VIS = "bbox_visib"
    KW_GT_INFO_PX_COUNT_ALL = "px_count_all"
    KW_GT_INFO_PX_COUNT_VLD = "px_count_valid"
    KW_GT_INFO_PX_COUNT_VIS = "px_count_visib" 
    KW_GT_INFO_VISIB_FRACT = "visib_fract"

    RGB_DIR = "rgb"
    DEPTH_DIR = "depth"
    MASK_DIR = "mask"
    INFO = "info"

    def init_clusters_hook(self):
        super().init_clusters_hook()

        self.info_dictlike = DictLikeCluster(self, "", flag_name=self.INFO)
        scene_camera    = DictLikeHandle.from_name(self.info_dictlike, self.GT_CAM_FILE)
        scene_gt        = DictLikeHandle.from_name(self.info_dictlike, self.GT_FILE)
        scene_gt_info   = DictLikeHandle.from_name(self.info_dictlike, self.GT_INFO_FILE)
        self.info_dictlike._set_fileshandle(0, scene_camera)
        self.info_dictlike._set_fileshandle(1, scene_gt)
        self.info_dictlike._set_fileshandle(2, scene_gt_info)

        self.rgb_cluster = UnifiedFileCluster(self,     self.RGB_DIR,      suffix = ".jpg", read_func=cv2.imread, write_func=cv2.imwrite, flag_name=ViewMeta.COLOR, alternate_suffix=[".png"])
        self.depth_cluster = UnifiedFileCluster(self,   self.DEPTH_DIR,    suffix = ".png", read_func=partial(cv2.imread, flags = cv2.IMREAD_ANYDEPTH), write_func=cv2.imwrite, flag_name=ViewMeta.DEPTH)
        self.mask_cluster = MutilMaskCluster(self,      self.MASK_DIR,     suffix = ".png", read_func=partial(cv2.imread, flags = cv2.IMREAD_GRAYSCALE), write_func=cv2.imwrite, flag_name=ViewMeta.MASKS)
        self.mask_cluster.link_rely_on(self.info_dictlike) # set rely, the obj_id is recorded in scene_gt.json

    def read(self, src: int, *, force = False, **other_paras)-> ViewMeta:
        raw_read_rlt = super().raw_read(src, force = force, **other_paras)
        rgb     = raw_read_rlt[ViewMeta.COLOR]
        depth   = raw_read_rlt[ViewMeta.DEPTH]
        masks   = raw_read_rlt[ViewMeta.MASKS]
        extr_vecs = {}
        visib_fracts = {}
        labels = {}
        intr        = raw_read_rlt[self.INFO][0][self.KW_CAM_K]
        depth_scale = raw_read_rlt[self.INFO][0][self.KW_CAM_DS]
        infos   = raw_read_rlt[self.INFO]
        
        id_seq = []

        for obj_info in infos[1]:
            posture = Posture(rmat = np.reshape(obj_info[self.KW_GT_R], (3,3)), tvec = np.reshape(obj_info[self.KW_GT_t], (3,1)))
            extr_vecs.update({obj_info[self.KW_GT_ID]: np.array([posture.rvec, posture.tvec])})
            id_seq.append(obj_info[self.KW_GT_ID])

        for id_, obj_gt_info in zip(id_seq, infos[2]):
            visib = obj_gt_info[self.KW_GT_INFO_VISIB_FRACT]
            visib_fracts.update({id_: visib})
            bbox = obj_gt_info[self.KW_GT_INFO_BBOX_VIS]
            labels.update({id_: bbox})
        
        return ViewMeta(rgb, depth, masks, extr_vecs, intr, depth_scale, None, None, visib_fracts, labels)

    def write(self, data_i: int, value: ViewMeta, *, force = False, **other_paras):
        raise NotImplementedError


class cxcywhLabelCluster(IntArrayDictAsTxtCluster[UnifiedFilesHandle, "cxcywhLabelCluster", "VocFormat_6dPosture"]):
    KW_IO_RAW = "raw"
    KW_IMAGE_SIZE = "image_size"

    def init_attrs(self):
        super().init_attrs()
        self.default_image_size = None
    
    class _read(IntArrayDictAsTxtCluster._read["cxcywhLabelCluster", dict[int, np.ndarray], UnifiedFilesHandle]):
        def postprogress_value(self, value:np.ndarray, *, image_size = None, **other_paras):
            value = super().postprogress_value(value)
            image_size = get_with_priority(image_size, self.files_cluster._get_rely(cxcywhLabelCluster.KW_IMAGE_SIZE), self.files_cluster.default_image_size)
            if image_size is not None:
                for id_, array in value.items():
                    bbox_2d = array.astype(np.float32) #[cx, cy, w, h]
                    bbox_2d = cxcywhLabelCluster._normedcxcywh_2_x1y1x2y2(bbox_2d, image_size)
                    value[id_] = bbox_2d
            else:
                if image_size != self.files_cluster.KW_IO_RAW:
                    warnings.warn("image_size is None, bbox_2d will not be converted from normed cxcywh to x1x2y1y2",
                    ClusterWarning)
            return value
        
    class _write(IntArrayDictAsTxtCluster._write["cxcywhLabelCluster", dict[int, np.ndarray], UnifiedFilesHandle]):
        def preprogress_value(self, value:dict[int, np.ndarray], *, image_size = None, **other_paras):
            value = super().preprogress_value(value)
            if value is None:
                return None
            image_size = get_with_priority(image_size, self.files_cluster._get_rely(cxcywhLabelCluster.KW_IMAGE_SIZE), self.files_cluster.default_image_size)
            if image_size is not None:
                bbox_2d = {}
                for k, v in value.items():
                    bbox_2d[k] = cxcywhLabelCluster._x1y1x2y2_2_normedcxcywh(v, image_size)
                value = bbox_2d
            else:
                if image_size != self.files_cluster.KW_IO_RAW:
                    warnings.warn("image_size is None, bbox_2d will not be converted from x1x2y1y2 to normed cxcywh",
                    ClusterWarning)
            return value

    def _set_rely(self, relied: FilesCluster, rlt):
        if relied.flag_name == ViewMeta.COLOR:
            rlt:np.ndarray = rlt
            self._update_rely(self.KW_IMAGE_SIZE, rlt.shape[:2][::-1])

    def read(self, src: int, *, sub_dir = None, image_size = None, force = False,  **other_paras) -> dict[int, np.ndarray]:
        return super().read(src, sub_dir=sub_dir, image_size = image_size, force=force, **other_paras)
    
    def write(self, data_i: int, value: dict[int, np.ndarray], *, sub_dir = None, image_size = None, force = False, **other_paras):
        return super().write(data_i, value, sub_dir=sub_dir, image_size = image_size, force=force, **other_paras)

    @staticmethod
    def _normedcxcywh_2_x1y1x2y2(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (cx, cy, w, h)]
        img_size: (w, h)
        '''

        # Unpack the normalized bounding box coordinates
        cx, cy, w, h = np.split(bbox_2d, 4, axis=-1)

        # Denormalize the center coordinates and width-height by image size
        w_img, h_img = img_size
        x1 = (cx - w / 2) * w_img
        y1 = (cy - h / 2) * h_img
        x2 = x1 + w * w_img
        y2 = y1 + h * h_img

        # Return the bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_2d = np.concatenate([x1, y1, x2, y2], axis=-1)
        return bbox_2d 
    

    @staticmethod
    def _x1y1x2y2_2_normedcxcywh(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (x1, x2, y1, y2)]
        img_size: (w, h)
        '''

        # Calculate center coordinates (cx, cy) and width-height (w, h) of the bounding boxes
        x1, y1, x2, y2 = np.split(bbox_2d, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Normalize center coordinates and width-height by image size
        w_img, h_img = img_size
        cx_normed = cx / w_img
        cy_normed = cy / h_img
        w_normed = w / w_img
        h_normed = h / h_img

        # Return the normalized bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_normed = np.concatenate([cx_normed, cy_normed, w_normed, h_normed], axis=-1)
        return bbox_normed

class VocFormat_6dPosture(PostureDataset[UnifiedFileCluster, "VocFormat_6dPosture", ViewMeta]):
    KW_IMGAE_DIR = "images"

    def __init__(self, directory, *, flag_name="", parent: DatasetNode = None) -> None:
        super().__init__(directory, flag_name=flag_name, parent=parent)

    def init_clusters_hook(self):
        super().init_clusters_hook()
        self.images_elements =\
              UnifiedFileCluster[UnifiedFilesHandle, UnifiedFileCluster, VocFormat_6dPosture, np.ndarray](self, self.KW_IMGAE_DIR, suffix = ".jpg" ,
                    read_func=cv2.imread, 
                    write_func=cv2.imwrite, 
                    flag_name=ViewMeta.COLOR)
        
        self.depth_elements      =\
              UnifiedFileCluster[UnifiedFilesHandle, UnifiedFileCluster, VocFormat_6dPosture, np.ndarray](self, "depths",  suffix = '.png',
                    read_func = partial(cv2.imread, flags = cv2.IMREAD_ANYDEPTH), 
                    write_func =cv2.imwrite,
                    flag_name=ViewMeta.DEPTH)
        
        self.masks_elements      =\
              UnifiedFileCluster[UnifiedFilesHandle, UnifiedFileCluster, VocFormat_6dPosture, dict[int, np.ndarray]](self, "masks", suffix = ".pkl",
                    read_func = deserialize_object,
                    write_func = serialize_object,
                    flag_name=ViewMeta.MASKS)
        rebind_methods(self.masks_elements.read_meta, self.masks_elements.read_meta.inv_format_value, self.deserialize_mask_dict) ### TODO
        rebind_methods(self.masks_elements.write_meta, self.masks_elements.write_meta.format_value, self.serialize_mask_dict) ### TODO

        self.extr_vecs_elements  = IntArrayDictAsTxtCluster(self, "trans_vecs",     array_shape=(2, 3), write_func_kwargs={"fmt":"%8.8f"},  
                                                            flag_name=ViewMeta.EXTR_VECS)

        self.intr_elements       = NdarrayAsTxtCluster(self,        "intr",         array_shape=(3,3),  write_func_kwargs={"fmt":"%8.8f", "delimiter":'\t'},
                                                            flag_name=ViewMeta.INTR)

        self.depth_scale_elements= NdarrayAsTxtCluster(self,        "depth_scale",  array_shape=(-1,),  write_func_kwargs={"fmt":"%8.8f"}, 
                                                            flag_name=ViewMeta.DEPTH_SCALE)

        self.bbox_3ds_elements   = IntArrayDictAsTxtCluster(self,   "bbox_3ds",     array_shape=(-1, 2), write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.BBOX_3DS)

        self.landmarks_elements  = IntArrayDictAsTxtCluster(self,   "landmarks",    array_shape=(-1, 2), write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.LANDMARKS)

        self.visib_fracts_element = IntArrayDictAsTxtCluster(self,   "visib_fracts", array_shape=(-1,),  write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.VISIB_FRACTS)

        self.labels_elements      = cxcywhLabelCluster(self,         "labels",       array_shape=(-1,),  write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.LABELS)

        self.labels_elements.default_image_size = (640, 480)
        self.labels_elements.link_rely_on(self.images_elements)

    @staticmethod
    def serialize_mask_dict(obj, mask_ndarray_dict:dict[int, np.ndarray]):
        def serialize_image(image:np.ndarray):  
            # 将NumPy数组编码为png格式的图像
            retval, buffer = cv2.imencode('.png', image)
            # 将图像数据转换为字节字符串
            image_bytes = buffer.tobytes()
            image.tobytes()
            return image_bytes

        new_value = dict(zip(mask_ndarray_dict.keys(), [serialize_image(x) for x in mask_ndarray_dict.values()]))
        return new_value

    @staticmethod
    def deserialize_mask_dict(obj, mask_bytes_dict:dict[int, bytes]):
        def deserialize_image(image_bytes):  
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_ANYDEPTH)# 将numpy数组解码为图像
            return image
        new_value = dict(zip(mask_bytes_dict.keys(), [deserialize_image(x) for x in mask_bytes_dict.values()]))
        return new_value

    @property
    def default_spliter(self):
        return self.spliter_group.get_cluster("default")

    def init_dataset_attr_hook(self):
        super().init_dataset_attr_hook()
        self.default_split_rate = 0.75

    def get_default_set_of(self, data_i):
        default_split_dict = self.default_spliter.get_idx_dict()
        for sub_set, idx_array in default_split_dict.items():
            if data_i in idx_array:
                return sub_set
        # didn't find data_i in default_split_dict
        sub_set = self.default_spliter.set_one_by_rate(data_i, self.default_split_rate)
        return sub_set
    
    def read(self, src: int, *, force = False, **other_paras)-> ViewMeta:
        raw_read_rlt = super().raw_read(src, force = force, **other_paras)
        return ViewMeta(**raw_read_rlt)
    
    def write(self, data_i: int, value: ViewMeta, *, force = False, **other_paras):
        sub_dir = self.get_default_set_of(data_i)
        super().raw_write(data_i, value.as_dict(), sub_dir = sub_dir, force = force, **other_paras)

    def file_to_cache_simple(self, *, force = False):
        self.file_to_cache(force = force)
        # self.masks_elements.file_to_cache(force = force, auto_decide=False)

    def copy_from_simplified(self, src_dataset:"VocFormat_6dPosture", *, cover = False, force = False):
        
        # masks_elements_strategy         = self.masks_elements.get_io_ctrl_strategy()
        extr_vecs_elements_strategy     = self.extr_vecs_elements.get_io_ctrl_strategy()
        intr_elements_strategy          = self.intr_elements.get_io_ctrl_strategy()
        depth_scale_elements_strategy   = self.depth_scale_elements.get_io_ctrl_strategy()
        bbox_3ds_elements_strategy      = self.bbox_3ds_elements.get_io_ctrl_strategy()
        landmarks_elements_strategy     = self.landmarks_elements.get_io_ctrl_strategy()
        visib_fracts_element_strategy   = self.visib_fracts_element.get_io_ctrl_strategy()
        # labels_elements_strategy        = self.labels_elements.get_io_ctrl_strategy()

        # self.masks_elements.set_io_ctrl_strategy(       IO_CTRL_STRATEGY.CACHE_IDPNDT)
        self.extr_vecs_elements.set_io_ctrl_strategy(   IO_CTRL_STRATEGY.CACHE_IDPNDT)
        self.intr_elements.set_io_ctrl_strategy(        IO_CTRL_STRATEGY.CACHE_IDPNDT)
        self.depth_scale_elements.set_io_ctrl_strategy( IO_CTRL_STRATEGY.CACHE_IDPNDT)
        self.bbox_3ds_elements.set_io_ctrl_strategy(    IO_CTRL_STRATEGY.CACHE_IDPNDT)
        self.landmarks_elements.set_io_ctrl_strategy(   IO_CTRL_STRATEGY.CACHE_IDPNDT)
        self.visib_fracts_element.set_io_ctrl_strategy( IO_CTRL_STRATEGY.CACHE_IDPNDT)
        # self.labels_elements.set_io_ctrl_strategy(      IO_CTRL_STRATEGY.CACHE_IDPNDT)

        src_dataset.extr_vecs_elements.file_to_cache(force = force)
        src_dataset.intr_elements.file_to_cache(force = force)
        src_dataset.depth_scale_elements.file_to_cache(force = force)
        src_dataset.bbox_3ds_elements.file_to_cache(force = force)
        src_dataset.landmarks_elements.file_to_cache(force = force)
        src_dataset.visib_fracts_element.file_to_cache(force = force)

        self.copy_from(src_dataset, cover = cover, force = force)

        # self.masks_elements.set_io_ctrl_strategy(masks_elements_strategy)
        self.extr_vecs_elements.set_io_ctrl_strategy(extr_vecs_elements_strategy)
        self.intr_elements.set_io_ctrl_strategy(intr_elements_strategy)
        self.depth_scale_elements.set_io_ctrl_strategy(depth_scale_elements_strategy)
        self.bbox_3ds_elements.set_io_ctrl_strategy(bbox_3ds_elements_strategy)
        self.landmarks_elements.set_io_ctrl_strategy(landmarks_elements_strategy)
        self.visib_fracts_element.set_io_ctrl_strategy(visib_fracts_element_strategy)
        # self.labels_elements.set_io_ctrl_strategy(labels_elements_strategy)