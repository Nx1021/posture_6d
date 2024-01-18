# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_items, dict_keys, dict_values
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from open3d import geometry, utility, io
import sys
import os
import glob
import shutil
import pickle
import cv2
import time
from tqdm import tqdm
import types
import warnings
import copy

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Generator
from functools import partial


# from posture_6d.data.IOAbstract import DatasetNode


from .IOAbstract import DataMapping, DatasetNode, IOMeta, FhBinDict, _KT, _VT, DMT, VDMT, DSNT,\
    FilesHandle, CacheProxy, \
    AmbiguousError, IOMetaParameterError, KeyNotFoundError, ClusterDataIOError, DataMapExistError, \
        IOStatusWarning, ClusterIONotExecutedWarning, ClusterNotRecommendWarning,\
    FilesCluster,\
    parse_kw, get_with_priority, method_exit_hook_decorator
from ..core.utils import JsonIO, deserialize_object, serialize_object, rebind_methods

FHT = TypeVar('FHT', bound=FilesHandle) # type of the files handle
UFC = TypeVar('UFC', bound="UnifiedFileCluster") # type of the unified file cluster
DFC = TypeVar('DFC', bound="DisunifiedFileCluster") # type of the disunified file cluster
DLC = TypeVar('DLC', bound="DictLikeCluster") # type of the dict-like cluster
UFH = TypeVar('UFH', bound="UnifiedFilesHandle") # type of the unified files handle
DFH = TypeVar('DFH', bound="DisunifiedFilesHandle") # type of the disunified files handle
DLFH = TypeVar('DLFH', bound="DictLikeHandle") # type of the dict-like files handle
VDLT = TypeVar('VDLT', bound=dict[int, Any]) # type of the value of data cluster
VDMT = TypeVar('VDMT') # type of the value of data cluster

NDAC = TypeVar('NDAC', bound="NdarrayAsTxtCluster") # type of the number dict cluster
VNDAC = TypeVar('VNDAC', bound=np.ndarray) # type of the value of number dict cluster
INDADC = TypeVar('INDADC', bound="IntArrayDictAsTxtCluster") # type of the input argument data cluster
VINDADC = TypeVar('VINDADC', bound=dict[int, np.ndarray]) # type of the value of input argument data cluster
DF = TypeVar('DF', bound="DictFile") # type of the data format

class UnifiedFilesHandle(FilesHandle[UFC, VDMT], Generic[UFC, VDMT]):
    def init_input_hook(self, *,
                        read_func = None, write_func = None, value_type = None, #type: ignore
                        **kwargs):
        read_func  = self.cluster.read_func 
        write_func = self.cluster.write_func
        value_type = self.cluster.value_type if value_type is None else value_type
        return super().init_input_hook(read_func=read_func, write_func=write_func, value_type=value_type, **kwargs)

class UnifiedFileCluster(FilesCluster[UFH, UFC, DSNT, VDMT], Generic[UFH, UFC, DSNT, VDMT]):
    _IS_ELEM = True
    FILESHANDLE_TYPE = UnifiedFilesHandle

    DEFAULT_SUFFIX = None
    DEFAULT_READ_FUNC = None
    DEFAULT_WRITE_FUNC = None
    DEFAULT_VALUE_TYPE = None

    def __init__(self, dataset_node:DSNT, mapping_name: str,
                 suffix:str = None, *,
                 flag_name = "", 
                 read_func:Callable[[str], VDMT] = None, 
                 write_func:Callable[[str, VDMT], None] = None, 
                 value_type:Callable = None,
                 filllen = 6, 
                 fillchar = '0',
                 alternate_suffix:list[str] = None,
                 read_func_args = None,
                 read_func_kwargs = None,
                 write_func_args = None,
                 write_func_kwargs = None,
                 lazy = False,
                 **kwargs
                 ) -> None:
        self._lazy = lazy # lazy mode, only load the fileshandle when needed

        suffix = suffix if suffix is not None else self.DEFAULT_SUFFIX
        assert suffix is not None, "suffix must be specified"
        self.suffix = suffix
        
        read_func, write_func, value_type = self.try_get_default(suffix, read_func, write_func, value_type)
        self.read_func  = read_func  
        self.write_func = write_func 
        self.value_type = value_type 

        read_func_args = tuple() if read_func_args is None else read_func_args
        read_func_kwargs = dict() if read_func_kwargs is None else read_func_kwargs
        write_func_args = tuple() if write_func_args is None else write_func_args
        write_func_kwargs = dict() if write_func_kwargs is None else write_func_kwargs
        assert isinstance(read_func_args, tuple), "read_func_args must be tuple"
        assert isinstance(read_func_kwargs, dict), "read_func_kwargs must be dict"
        assert isinstance(write_func_args, tuple), "write_func_args must be tuple"
        assert isinstance(write_func_kwargs, dict), "write_func_kwargs must be dict"

        if len(read_func_args) > 0 or len(read_func_kwargs) > 0:
            self.read_func = partial(self.read_func, *read_func_args, **read_func_kwargs)
        if len(write_func_args) > 0 or len(write_func_kwargs) > 0:
            self.write_func = partial(self.write_func, *write_func_args, **write_func_kwargs)

        self.filllen = filllen
        self.fillchar = fillchar
        alternate_suffix = [alternate_suffix] if isinstance(alternate_suffix, str) else alternate_suffix
        self.alternate_suffix = alternate_suffix if alternate_suffix is not None else []  
        super().__init__(dataset_node, mapping_name, flag_name=flag_name)
        self.cache_priority = False 

    def __init_subclass__(cls, **kwargs):
        cls.build_partly = method_exit_hook_decorator(cls, cls.build_partly, cls._rebuild_done)
        super().__init_subclass__(**kwargs)

    def data_keys(self):
        if self._lazy:
            return self.MemoryData._lazy_data.keys()
        else:
            return self.MemoryData.keys()

    def load_postprocess(self, data:dict):
        if not data:
            return self.MEMORY_DATA_TYPE({})  # 返回一个空的 data_info_map

        if not self._lazy:
            new_dict = {int(k): None for k in data.keys()}
            for k, v in tqdm(data.items(), desc=f"loading {self}", total=len(new_dict), leave=False):
                new_dict[int(k)] = self.FILESHANDLE_TYPE.from_dict(self, v)
            data_info_map = self.MEMORY_DATA_TYPE(new_dict)
        else:
            new_dict = {}
            data_info_map = self.MEMORY_DATA_TYPE({})
            data_info_map._lazy_data = data
            data_info_map.fileshandle_init_func = partial(self.FILESHANDLE_TYPE.from_dict, self)

        return data_info_map

    def save_preprecess(self, MemoryData:FhBinDict[FHT] = None ):
        MemoryData = self.MemoryData if MemoryData is None else MemoryData
        to_save_dict = {item[0]: item[1].as_dict() for item in MemoryData.items()}
        if MemoryData._lazy_data is not None:
            MemoryData._lazy_data.update(to_save_dict)
            to_save_dict = MemoryData._lazy_data
        return to_save_dict

    def query_fileshandle(self, data_i:int) -> FHT:
        """
        Retrieve the fileshandle at the specified index from the MemoryData.

        Args:
            data_i (int): The index of the fileshandle to retrieve.

        Returns:
            FHT: The fileshandle at the specified index.
        """
        return self.MemoryData[data_i]

    def build_partly(self, build_data_i_list:Union[int, Iterable[int]]):
        ### input check ###
        if isinstance(build_data_i_list, int):
            build_data_i_list = [build_data_i_list]
        assert isinstance(build_data_i_list, (Iterable)), f"the type of data_i {type(build_data_i_list)} is not Iterable"
        if len(build_data_i_list) == 0:
            return
        else:
            assert all([isinstance(i, int) for i in build_data_i_list]), f"the type of data_i {type(build_data_i_list)} is not Iterable[int]"
        
        ### build ###

        to_build_paths:dict[int, list[str]] = {}
        exist_files_i = []

        if len(build_data_i_list) < 20:
            # if the number of data_i is small, search one by one
            for data_i in tqdm(build_data_i_list, leave=False, desc = f"searching {self.data_path}"):
                search_name = self.format_name(data_i, "")
                paths = glob.glob(os.path.join(self.data_path, f"**/{search_name}*.*")) # match only with prefix + corename, to compatible with the mutil-files mode
                if len(paths) > 0:
                    exist_files_i.append(data_i)
                    if data_i not in self.keys():
                        to_build_paths[data_i] = paths
        else:
            # if the number of data_i is large, search all files and then match the data_i
            print("\r", f"matching paths in {self.data_path}", end = "")
            paths = glob.glob(os.path.join(self.data_path, f"**/*.*")) # match all files
            for path in tqdm(paths, leave=False, desc = f"scanning {self.data_path}"):
                paras = self.FILESHANDLE_TYPE._parse_one_path_to_paras(
                                self, path, self.DEFAULT_PREFIX_JOINER, self.DEFAULT_APPENDNAMES_JOINER)
                data_i = self.deformat_corename(paras[1]) # paras[1] is the corename
                # check if the data_i is in the build_data_i_list
                if data_i in build_data_i_list:
                    exist_files_i.append(data_i)
                    if data_i not in self.keys():
                        to_build_paths.setdefault(data_i, []).append(path)
        
        not_exist_files_i = set(self.keys()).intersection(set(build_data_i_list)).difference(exist_files_i)
                
        for data_i, bp in tqdm(to_build_paths.items(), leave=False, desc = f"creating fileshandles for {self}"):
            if isinstance(bp, list):
                for _p in bp:
                    fh:FilesHandle = self.FILESHANDLE_TYPE.create_new_and_cover().from_path(self, _p)
            else:
                fh:FilesHandle = self.FILESHANDLE_TYPE.create_new_and_cover().from_path(self, bp)
            if fh.all_file_exist:
                self._set_fileshandle(data_i, fh)
                # self.MemoryData[data_i] = fh
            else:
                self.paste_file(data_i, fh)

        for data_i in tqdm(not_exist_files_i, leave=False, desc = f"deleting not exist fileshandles for {self}"):
            self._pop_fileshandle(data_i)
        
        self.save()

    def activate(self, build_data_i_list:Union[int, Iterable[int]] = None):
        if not self._lazy:
            return
        ### input check ###
        self.MemoryData.activate(build_data_i_list)

    #####
    def create_fileshandle_in_iometa(self, src, dst, value, * ,sub_dir = "", **other_paras):
        if not self.MULTI_FILES:
            corename = self.format_corename(dst)
            fh = self.FILESHANDLE_TYPE(self, sub_dir, corename, self.suffix)
            return fh
        else:
            raise NotImplementedError
    
    # @classmethod
    # def from_cluster(cls:type[UFC], cluster:UFC, dataset_node:DSNT = None, mapping_name = None, *args, **kwargs) -> UFC:
    #     dataset_node    = cluster.dataset_node if dataset_node is None else dataset_node
    #     mapping_name    = cluster.mapping_name if mapping_name is None else mapping_name
    #     new_cluster:UFC = cls(dataset_node, mapping_name)
    #     return new_cluster

    #####
    def format_corename(self, data_i: int):
        filllen = self.filllen
        fillchar = self.fillchar
        return f"{str(data_i).rjust(filllen, fillchar)}"
    
    def format_name(self, data_i:int, suffix:str = None, prefix:str=None, appendnames:Union[str, list[str]]=None, prefix_joiner:str=None, appendnames_joiner:str=None):
        corename = self.format_corename(data_i)

        suffix                  = get_with_priority(suffix, self.DEFAULT_SUFFIX) # type: ignore

        prefix:str              = get_with_priority(prefix,             self.DEFAULT_PREFIX,             self.FILESHANDLE_TYPE.DEFAULT_PREFIX,             '') # type: ignore
        appendnames:list[str]   = get_with_priority(appendnames,        self.DEFAULT_APPENDNAMES,        self.FILESHANDLE_TYPE.DEFAULT_APPENDNAMES,        ['']) # type: ignore
        if isinstance(appendnames, str):
            appendnames = [appendnames]

        prefix_joiner:str       = get_with_priority(prefix_joiner,      self.DEFAULT_PREFIX_JOINER,      self.FILESHANDLE_TYPE.DEFAULT_PREFIX_JOINER,      '') # type: ignore
        appendnames_joiner:str  = get_with_priority(appendnames_joiner, self.DEFAULT_APPENDNAMES_JOINER, self.FILESHANDLE_TYPE.DEFAULT_APPENDNAMES_JOINER, '') # type: ignore

        if len(prefix) > 0:
            prefix = prefix + prefix_joiner
        if len(suffix) > 0 and suffix[0] != '.': 
            suffix = '.' + suffix
        
        names = [prefix + corename + appendnames_joiner + apn + suffix for apn in appendnames]
        if len(names) == 1:
            return names[0]
        else:
            return names

    def format_file_name(self, data_i: int):
        core_name = self.format_corename(data_i)

    def deformat_corename(self, corename: str):
        return int(corename)

    def matching_path(self):
        paths = []
        for suffix in self.alternate_suffix + [self.suffix]:
            paths.extend(glob.glob(os.path.join(self.data_path, "**/*" + suffix), recursive=True))
        return paths
    
    @classmethod
    def try_get_default(cls, file:str,
                            read_func:Callable = None, 
                            write_func:Callable = None, 
                            value_type:Callable = None):
        if file.startswith('.'):
            suffix = file
        else:
            suffix = os.path.splitext(file)[1]
        if suffix in cls.FILESHANDLE_TYPE.DEFAULT_FILE_TYPE:
            _read_func, _write_func, _value_type = cls.FILESHANDLE_TYPE.DEFAULT_FILE_TYPE[suffix]
            read_func =  get_with_priority(read_func,  cls.DEFAULT_READ_FUNC,  cls.FILESHANDLE_TYPE.DEFAULT_READ_FUNC,   _read_func)
            write_func = get_with_priority(write_func, cls.DEFAULT_WRITE_FUNC, cls.FILESHANDLE_TYPE.DEFAULT_WRITE_FUNC, _write_func)
            value_type = get_with_priority(value_type, cls.DEFAULT_VALUE_TYPE, cls.FILESHANDLE_TYPE.DEFAULT_VALUE_TYPE, _value_type)
        # else:
        #     warnings.warn(f"can't find default file type for {suffix}", ClusterNotRecommendWarning)

        return read_func, write_func, value_type
        
    ############################
    _FCT = TypeVar('_FCT', bound="UnifiedFileCluster")
    _VDMT = TypeVar('_VDMT')
    _FHT = TypeVar('_FHT', bound=UnifiedFilesHandle)

    def update_read_func_binded_paras(self, paras:dict[str, Any]):
        self.read_meta.core_func_binded_paras.update(paras)

    def update_write_func_binded_paras(self, paras:dict[str, Any]):
        self.write_meta.core_func_binded_paras.update(paras)

    def init_io_metas(self):
        '''
        init the io_metas of the data cluster
        '''
        super().init_io_metas()
        self.change_dir_meta:IOMeta[UFC, VDMT, UFH] = self._change_dir(self)

    class _read(FilesCluster._read[_FCT, _VDMT, _FHT]):
        @property
        def core_func(self):
            return self.files_cluster.read_func
        
        @core_func.setter
        def core_func(self, func):
            pass

    class _write(FilesCluster._write[_FCT, _VDMT, _FHT]):
        def get_FilesHandle(self, src, dst, value, *, sub_dir = "", **other_paras):
            return super().get_FilesHandle(src, dst, value, sub_dir = sub_dir, **other_paras)
        
        @property
        def core_func(self):
            return self.files_cluster.write_func
        
        @core_func.setter
        def core_func(self, func):
            pass      

    class _paste_file(FilesCluster._paste_file[_FCT, _VDMT, _FHT]):
        def get_FilesHandle(self, src, dst, value:UnifiedFilesHandle, **other_paras):
            sub_dir = value.sub_dir
            return super().get_FilesHandle(src, dst, value, sub_dir = sub_dir, **other_paras)

    def clear(self, * ,force=False, clear_both=True):
        super().clear(force = force, clear_both=clear_both)

    def write(self, data_i:int, value:VDMT, *, sub_dir = "", force = False, **other_paras) -> None:
        return super().write(data_i, value, sub_dir=sub_dir, force=force, **other_paras)
    
    def append(self, value: VDMT, *, sub_dir = "", force=False, **other_paras):
        return super().append(value, sub_dir = sub_dir, force = force,  **other_paras)
    
    def copy_from(self, src:UFC, *, cover=False, force=False, only_cache = None, **other_paras):
        if only_cache is None:
            # decide whether to copy cache by the size of the files
            sizes = [os.path.getsize(fh.get_path()) for fh in src.query_all_fileshandle()] # TODO
            mean_size = sum(sizes) / len(sizes)
            only_cache = mean_size > 1e5
        return super().copy_from(src, cover=cover, force=force, **other_paras)
    #####################

    #######################
    ########################

    @staticmethod
    def ElementsTest():
        top_dir = os.path.join(os.getcwd(), "ElementsTest")
        cv2.imwrite(os.path.join(top_dir, "000000.png"), np.random.random((640, 480, 3))*255)
        e0 = UnifiedFileCluster(top_dir, "000000", suffix=".jpg", alternate_suffix=[".png"])
        e1 = UnifiedFileCluster(top_dir, "000001", suffix=".jpg", alternate_suffix=[".png"])

        e0.clear(force=True)
        e1.clear(force=True)

        start = time.time()
        with e0.get_writer():
            for i in range(5):    
                e0.append((np.random.random((640, 480, 3))*255).astype(np.uint8), sub_dir="sub0")
        print(time.time() - start)

        e0.cache_priority = False
        with e0.get_writer():
            for i in range(5):    
                e0.append((np.random.random((640, 480, 3))*255).astype(np.uint8), sub_dir="sub1")

        for fh in e0.query_all_fileshandle():
            print(fh)

        with e0.get_writer().allow_overwriting():
            e0.cache_to_file()
            e0.file_to_cache()

        os.remove(e0.MemoryData_path)
        e0.close()
        e0.open()

        print()
        for fh in e0.query_all_fileshandle():
            print(fh)

        start = time.time()
        with e1.get_writer():
            for array in e0:
                e1.append(array)
        print(time.time() - start)
        print()
        for fh in e1.query_all_fileshandle():
            print(fh)

        e1.file_to_cache()
        e1.clear(force=True)

        start = time.time()
        with e1.get_writer():
            e1.copy_from(e0, cover=True)
        print(time.time() - start)

        with e1.get_writer():
            e1.remove(0, remove_both=True)
            e1.remove(5, remove_both=True)
        e1.make_continuous(force = True)

        print()
        for fh in e1.query_all_fileshandle():
            print(fh)

        e0.clear(force=True)
        e1.clear(force=True)

class DisunifiedFilesHandle(FilesHandle[DFC, VDMT], Generic[DFC, VDMT]):
    # _instances = {}
    # def __new__(cls, *args, **kwargs) -> None:
    #     instance = super().__new__(cls)
    #     super().__init__(instance, *args, **kwargs)
    #     file_id_str = instance.__repr__()
    #     if file_id_str not in cls._instances:
    #         cls._instances[file_id_str] = instance
    #     return cls._instances[file_id_str]
        
    # def init_input_hook(self, *, suffix:str, read_func, write_func, value_type, **kw):
    #     read_func, write_func, value_type = self.try_get_default(suffix, read_func, write_func, value_type)
    #     return super().init_input_hook(suffix = suffix, read_func=read_func, write_func=write_func, value_type=value_type, **kw)

    @property
    def file(self):
        return self.get_name()
    
    @file.setter
    def file(self, file):
        corename, suffix = os.path.splitext(os.path.basename(file))
        self.corename = corename
        self.suffix = suffix

class DisunifiedFileCluster(FilesCluster[DFH, DFC, DSNT, VDMT], Generic[DFH, DFC, DSNT, VDMT]):
    FILESHANDLE_TYPE = DisunifiedFilesHandle

    def __init__(self, dataset_node: Union[str, DatasetNode], mapping_name: str, *args, flag_name = "", fileshandle_list = None, **kwargs) -> None:
        super().__init__(dataset_node, mapping_name, *args, flag_name=flag_name, **kwargs)
        fileshandle_list = [] if fileshandle_list is None else fileshandle_list
        for fh in fileshandle_list:
            self._set_fileshandle(self.next_valid_data_i, fh)

    #####
    def create_fileshandle_in_iometa(self, src:int, dst:int, value:Any, **other_paras):
        if not self.MULTI_FILES:
            if isinstance(value, DisunifiedFilesHandle):
                fh = self.FILESHANDLE_TYPE.from_fileshandle(self, value)
                return fh
            else:
                fh = self.FILESHANDLE_TYPE.from_name(self, "_t.dfhtemp")
                return fh
        else:
            raise NotImplementedError
    #####
    @property
    def all_files_exist(self):
        return all([fh.all_file_exist for fh in self.MemoryData.values()])

    @property
    def file_names(self):
        return [fh.file for fh in self.MemoryData.values()]

    def add_default_file(self, filename):
        suffix = os.path.splitext(filename)[-1]
        assert suffix in FilesHandle.DEFAULT_FILE_TYPE, f"suffix {suffix} is not supported"
        fh = self.FILESHANDLE_TYPE.from_name(self, filename)
        self._set_fileshandle(self.next_valid_data_i, fh)

    def cvt_key(self, key:Union[int, str, DisunifiedFilesHandle]):
        key = super().cvt_key(key)
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            if '.' in key:
                idx = list([f.file for f in self.MemoryData.values()]).index(key)
            else:
                idx = list([f.file for f in self.MemoryData.values()]).index(key)
            return list(self.keys())[idx]
        elif isinstance(key, DisunifiedFilesHandle):
            idx = list(self.MemoryData.values()).index(key)
            return list(self.keys())[idx]
        
    _FCT = TypeVar('_FCT', bound="DisunifiedFileCluster")
    _VDMT = TypeVar('_VDMT')
    _FHT = TypeVar('_FHT', bound=DisunifiedFilesHandle)

    class _read(FilesCluster._read[_FCT, _VDMT, _FHT]):
        def get_file_core_func(self, src_file_handle: DisunifiedFilesHandle, dst_file_handle: DisunifiedFilesHandle, value) -> Callable[..., Any]:
            return src_file_handle.read_func
        
    class _write(FilesCluster._write[_FCT, _VDMT, _FHT]):
        def get_file_core_func(self, src_file_handle: DisunifiedFilesHandle, dst_file_handle: DisunifiedFilesHandle, value) -> Callable[..., Any]:
            return dst_file_handle.write_func
        
    def rebuild(self, force = False):
        run_rebuild = True
        for dm in self._registry.values():
            if dm.data_path == self.data_path:
                # to avoid add some unexcepted fileshandle
                # can only be setted mannully
                run_rebuild = False
                break
        if run_rebuild:
            super().rebuild(force = force)

    @staticmethod
    def Test():
        def print_all_fh(fc:DisunifiedFileCluster):
            print()
            for fh in fc.query_all_fileshandle():
                print(fh)

        top_dir = os.path.join(os.getcwd(), "DisunifiedFileClusterTest")

        d0 = DisunifiedFileCluster(top_dir, "000000")
        d1 = DisunifiedFileCluster(top_dir, "000001")

        d0.clear(force=True)
        d1.clear(force=True)

        d0.add_default_file("f0.npy")
        d0.add_default_file("f1.json")
        d0.add_default_file("f2.txt")
        d0.add_default_file("f3.pkl")
        d0.add_default_file("f4.png")
        d0.add_default_file("f0.npy")

        print_all_fh(d0)

        with d0.get_writer():
            d0.write(0, np.array([1, 2, 3]))
            d0.write(1, {"a": 1, "b": 2})
            d0.write(2, "hello world")
            d0.write(3, {"a": 1, "b": 2})
            d0.write(4, np.random.random((100, 100, 3)).astype(np.uint8))
        
        print_all_fh(d0)
        
        # TODO synced 控制逻辑？ copy_from 为什么不能拷贝cache
        d1.copy_from(d0, cover=True, force=True)
        print_all_fh(d1)

        d1.file_to_cache(force=True)
        print_all_fh(d1)
        d1.cache_to_file(force=True)
        print_all_fh(d1)

        d1.rebuild()

        with d1.get_writer().allow_overwriting():
            d1.remove(0, remove_both=True)
        d1.make_continuous(force = True)

        print_all_fh(d1)

        d0.clear(force=True)
        d1.clear(force=True)    

class DictLikeHandle(DisunifiedFilesHandle[DLC, dict[int, Any]], Generic[DLC]):
    LOAD_CACHE_ON_INIT = True

    def init_input_hook(self, *, value_type, **kw):
        return super().init_input_hook(value_type=dict, **kw)

    # def io_cache_at_wrapper(self, func:Callable, *, elem_i = None, value=None, is_read = False):
    #     io_error = False
    #     if self.is_closed:
    #         warnings.warn(f"can't set or get cache at {elem_i}", IOStatusWarning)
    #         io_error = True
    #     if  not is_read and self.is_readonly:
    #         warnings.warn(f"can't set cache at {elem_i}", IOStatusWarning)
    #         io_error = True
    #     if elem_i in self.cache and not is_read and self.overwrite_forbidden:
    #         warnings.warn(f"overwrite forbidden, can't set cache at {elem_i}", IOStatusWarning)
    #         io_error = True

    #     if func.__name__ == "__getitem__":
    #         return func(elem_i) if not io_error else None
    #     elif func.__name__ == "__setitem__":
    #         if io_error:
    #             return False
    #         else:
    #             func(elem_i, value)
    #             return True
    #     elif func.__name__ == "pop":
    #         return func(elem_i) if not io_error else None

    def get_cache_at(self, elem_i) -> Union[Any, None]:
        return self.cache[elem_i]
    
    def set_cache_at(self, elem_i, value) -> bool:
        return self.cache.__setitem__(elem_i, value)

    def pop_cache_at(self, elem_i) -> Union[Any, None]:
        return self.cache.pop(elem_i)
    
    def sort_cache(self):
        self.cache = dict(sorted(self.cache.items()))

    @property
    def elem_num(self):
        return len(self.cache)
    
    @property
    def elem_i_upper(self):
        return max(self.cache.keys(), default=-1) + 1

    @property
    def elem_continuous(self):
        return self.elem_num == self.elem_i_upper
        
    def erase_cache(self):
        if not self.is_closed and not self.is_readonly:
            self.cache.clear()

    @property
    def has_cache(self):
        return len(self.cache) > 0

class DictLikeCluster(DisunifiedFileCluster[DLFH, DLC, DSNT, VDLT], Generic[DLFH, DLC, DSNT, VDLT]):
    _IS_ELEM = True
    _ELEM_BY_CACHE = True
    FILESHANDLE_TYPE:type[FilesHandle] = DictLikeHandle

    SAVE_IMMIDIATELY = 0
    SAVE_AFTER_CLOSE = 1
    SAVE_STREAMLY = 2

    class StreamlyWriter(DisunifiedFileCluster._Writer):
        def __init__(self, cluster:"DictLikeCluster") -> None:
            super().__init__(cluster)
            self.streams:list[JsonIO.Stream] = []
            self.obj:DictLikeCluster = self.obj

        def enter_hook(self):
            self.obj.save_mode = self.obj.SAVE_STREAMLY
            streams = []
            for fh in self.obj.query_all_fileshandle():
                fh: DictLikeHandle
                streams.append(JsonIO.Stream(fh.get_path(), True))
            self.streams.extend(streams)
            for stream in self.streams:
                stream.open()            
            super().enter_hook()


        def exit_hook(self):
            rlt = super().exit_hook()
            for stream in self.streams:
                stream.close()
            self.streams.clear()
            self.obj.save_mode = self.obj.SAVE_AFTER_CLOSE
            return rlt

        def write(self, data_i:int, elem_i, value):
            stream = self.streams[data_i]
            stream.write({elem_i: value})

    def __init__(self, dataset_node: Union[str, DatasetNode], mapping_name: str, *args, flag_name = "", **kwargs) -> None:
        super().__init__(dataset_node, mapping_name, *args, flag_name = flag_name, **kwargs)
        self.__save_mode = self.SAVE_AFTER_CLOSE
        self.stream_writer = self.StreamlyWriter(self)
        
    @property
    def caches(self):
        return [fh.cache for fh in self.query_all_fileshandle()]

    @property
    def save_mode(self):
        return self.__save_mode
    
    @save_mode.setter
    def save_mode(self, mode):
        assert mode in [self.SAVE_IMMIDIATELY, self.SAVE_AFTER_CLOSE, self.SAVE_STREAMLY]
        if self.is_writing and mode != self.SAVE_STREAMLY:
            warnings.warn("can't change save_mode from SAVE_STREAMLY to the others while writing streamly")
        if self.__save_mode == self.SAVE_AFTER_CLOSE and mode != self.SAVE_AFTER_CLOSE and self.opened:
            self.save()
        self.__save_mode = mode

    @property
    def write_streamly(self):
        return self.save_mode == self.SAVE_STREAMLY

    def elem_keys(self):
        if len(self.MemoryData) == 0:
            return tuple()
        else:
            first_fh = next(iter(self.MemoryData.values()))
            return first_fh.cache.keys()

    ### IO ###
    def init_io_metas(self):
        super().init_io_metas()
        self.read_elem_meta = self._read_elem(self)
        self.write_elem_meta = self._write_elem(self)
        self.modify_elem_key_meta = self._modify_elem_key(self)
        self.remove_elem_meta = self._remove_elem(self)

    def __cvt_elem_i_input(self, elem_i:Union[int, Iterable[int]]):
        if isinstance(elem_i, int):
            return_list = False
            elem_i = [elem_i]
        elif isinstance(elem_i, Iterable):
            return_list = True
        else:
            raise TypeError(f"elem_i should be int or Iterable[int], not {type(elem_i)}")
        return return_list, elem_i

    class _read_elem(DisunifiedFileCluster._read["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, value, **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():    
                fh = self._query_fileshandle(data_i)
                rlt_dict[data_i] = fh.get_cache_at(src)
            return rlt_dict

    class _write_elem(DisunifiedFileCluster._write["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, values:dict[int, Any], **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():    
                fh = self._query_fileshandle(data_i)
                success = fh.set_cache_at(dst, values[data_i])
                if success and self.files_cluster.write_streamly:
                    self.files_cluster.stream_writer.write(data_i, dst, values[data_i])
                rlt_dict[data_i] = success
            return rlt_dict

    class _modify_elem_key(DisunifiedFileCluster._modify_key["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, values:dict, **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():
                elem = self._query_fileshandle(data_i).pop_cache_at(src)
                success = self._query_fileshandle(data_i).set_cache_at(dst, elem)
                rlt_dict[data_i] = success
            return rlt_dict

    class _remove_elem(DisunifiedFileCluster._remove["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, value, **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():
                rlt_dict[data_i] = self._query_fileshandle(data_i).pop_cache_at(dst)
            return rlt_dict        

    def read_elem(self, src:Union[int, Iterable[int]], *, force = False, **other_paras) -> Union[dict, list[dict]]:
        rlt = self.io_decorator(self.read_elem_meta, force=force)(src=src, **other_paras)
        return rlt

    def write_elem(self, dst:int, value:Union[Any, dict[int, Any]], *, force = False, **other_paras) -> dict:
        assert len(value) == self.data_num, f"values length {len(value)} != cluster length {self.data_num}"
        
        rlt = self.io_decorator(self.write_elem_meta, force=force)(dst=dst, value=value, **other_paras)

        if self.save_mode == self.SAVE_IMMIDIATELY:
            self.cache_to_file(force=True)
            self.save()
        return rlt

    def modify_elem_key(self, src:int, dst:int, *, force = False, **other_paras) -> dict:
        if self.write_streamly:
            raise ValueError("can't modify item while writing streamly")
        
        rlt = self.io_decorator(self.modify_elem_key_meta, force=force)(src=src, dst=dst, **other_paras)

        return rlt

    def remove_elem(self, dst:int, remove_both = False, *, force = False, **other_paras) -> dict:
        if self.write_streamly:
            raise ValueError("can't pop item while writing streamly")
        
        rlt = self.io_decorator(self.remove_elem_meta, force=force)(dst=dst, **other_paras)

        return rlt

    def sort_elem(self):
        if self.write_streamly:
            raise ValueError("can't pop item while writing streamly")
        for dict_fh in self.query_all_fileshandle():
            dict_fh:DictLikeHandle
            dict_fh.sort_cache()
    ##########

    def _set_fileshandle(self, data_i, fh:DictLikeHandle):
        if (fh.elem_num == self.elem_num and fh.elem_continuous and self.elem_continuous) or (self.elem_num == 0 and self.data_num == 0):
            super()._set_fileshandle(data_i, fh)     
        else:       
            raise ValueError("can't add fileshandle while there are items or writing streamly")
        
    def add_default_file(self, filename):
        suffix = os.path.splitext(filename)[-1]
        assert suffix in FilesHandle.DEFAULT_FILE_TYPE, f"suffix {suffix} is not supported"
        assert issubclass(FilesHandle.DEFAULT_FILE_TYPE[suffix][2], dict), f"suffix {suffix} is not supported"
        fh = self.FILESHANDLE_TYPE.from_name(self, filename)
        self._set_fileshandle(self.next_valid_data_i, fh)

    def save_without_cache(self):
        '''
        to spare memory, save without cache
        '''
        dict_wo_cache = self.save_preprecess()
        for v in dict_wo_cache.values():
            
            v[FilesHandle.KW_cache][CacheProxy.KW_cache] = {}
        self.__class__.save_memory_func(self.MemoryData_path, dict_wo_cache)

    @classmethod
    def from_cluster(cls:type[DLC], cluster:DLC, dataset_node:DSNT = None, mapping_name = None, *args, flag_name = "",  **kwargs) -> DLC:
        new_cluster = super().from_cluster(cluster, dataset_node=dataset_node, mapping_name=mapping_name, *args, flag_name = flag_name, **kwargs)
        new_cluster.open()
        for fh in cluster.query_all_fileshandle():
            new_fh = cls.FILESHANDLE_TYPE.from_fileshandle(new_cluster, fh, cache={})
            new_cluster._set_fileshandle(new_cluster.next_valid_data_i, new_fh)
        return new_cluster

    @staticmethod
    def Test():
        def print_all_fh(fc:DictLikeCluster):
            print()
            for fh in fc.query_all_fileshandle():
                print(fh)

        top_dir = os.path.join(os.getcwd(), "DictLikeClusterTest")

        d0 = DictLikeCluster(top_dir, "000000")
        d1 = DictLikeCluster(top_dir, "000001")

        d0.clear(force=True)
        d1.clear(force=True)

        d0.add_default_file("f0.json")
        d0.add_default_file("f1.json")
        d0.add_default_file("f2.json")
        print_all_fh(d0)

        with d0.get_writer():
            d0.write(0, [np.array([1,2]), 2, None])
            d0.write(1, [np.array([3,4]), 4, -1])
        print_all_fh(d0)

        try:
            d0.add_default_file("f3.json")
        except:
            print("can not add fileshandle while there are items")
        
        d0.save_mode = d0.SAVE_IMMIDIATELY
        with d0.get_writer():
            d0.write(2, [np.array([5,6]), 6, -2])
        print_all_fh(d0)

        with d0.stream_writer:
            d0.write(2, [np.array([7,8]), 8, -3])
            d0.write(4, [np.array([9,10]), 10, -4])

        print("d0 elem_continuous:", d0.elem_continuous)
        d0.make_elem_continuous(True)
        print("d0 elem_continuous:", d0.elem_continuous)
        d0.save_without_cache()
        print_all_fh(d0)
        print(len(d0.elem_keys()))

        d1.copy_from(d0, cover=True, force=True)
        print_all_fh(d1)

        d1.file_to_cache(force=True)
        print_all_fh(d1)
        d1.cache_to_file(force=True)
        print_all_fh(d1)

        d1.rebuild()

        with d1.get_writer().allow_overwriting():
            d1.remove(0, remove_both=True)
        d1.make_continuous(force = True)

        print_all_fh(d1)

        d0.clear(force=True)
        d1.clear(force=True)      

def cache_to_file_decorator(func):
    def wrapper(self:"DictFile", *args, **kwargs):
        rlt = func(self, *args, **kwargs)
        self.cache_to_file(force=True)
        return rlt
    return wrapper

class SingleFile:
    pass
    # TODO 仅 包含一个文件，不保存.memorydata 而是直接保存到文件。每启动时重建memorydata


class DictFile(DisunifiedFileCluster[DFH, DFC, DSNT, dict], Generic[DFH, DFC, DSNT]):
    def _set_fileshandle(self, data_i, fileshandle: DFH):
        if len(self.MemoryData) > 0:
            raise ValueError("DictFile can only have one fileshandle")
        super()._set_fileshandle(data_i, fileshandle)
        
    def __init__(self, dataset_node: Union[str, DatasetNode], mapping_name: str, *args, flag_name="", file_name = ".json", **kwargs) -> None:
        super().__init__(dataset_node, mapping_name, *args, flag_name=flag_name, fileshandle_list=[], **kwargs)
        json_file = DictLikeHandle.from_name(self, file_name, read_func=JsonIO.load_json, write_func=JsonIO.dump_json)
        self._set_fileshandle(0, json_file)
        self.write_synchronous = True
        self.cache_priority = True
        
    def __unsafe_get_cache(self) -> dict:
        return self.query_fileshandle(0)._unsafe_get_cache()

    @cache_to_file_decorator
    def write_info(self, key, value):
        self.__unsafe_get_cache()[key] = value
        self.cache_to_file(force=True)
        
    def read_info(self, key):
        return self.read(0)[key]
    
    def has_info(self, key):
        return key in self.read(0)
    
    def __getitem__(self, key: str):
        return self.read_info(key)
    
    def __setitem__(self, key: str, value):
        self.write_info(key, value)

    def __contains__(self, i):
        return self.has_info(i)
    
    @cache_to_file_decorator
    def update(self, *arg, **kw):
        return self.__unsafe_get_cache().update(*arg, **kw)
    
    @cache_to_file_decorator
    def pop(self, *arg, **kw):
        return self.__unsafe_get_cache().pop(*arg, **kw)
    
    @cache_to_file_decorator
    def popitem(self, *arg, **kw):
        return self.__unsafe_get_cache().popitem(*arg, **kw)
    
    @cache_to_file_decorator
    def clear(self, *arg, **kw):
        return self.__unsafe_get_cache().clear(*arg, **kw)
    
    def get(self, *arg, **kw):
        return self.__unsafe_get_cache().get(*arg, **kw)
    
    @cache_to_file_decorator
    def setdefault(self, *arg, **kw):
        return self.__unsafe_get_cache().setdefault(*arg, **kw)
    
    def __len__(self):
        return len(self.read(0))
    
    def __iter__(self):
        return self.read(0)

class NdarrayAsTxtCluster(UnifiedFileCluster[UFH, NDAC, DSNT, VNDAC], Generic[UFH, NDAC, DSNT, VNDAC]):
    
    DEFAULT_SUFFIX = ".txt"
    DEFAULT_READ_FUNC = np.loadtxt
    DEFAULT_WRITE_FUNC = np.savetxt
    DEFAULT_VALUE_TYPE = np.ndarray
    
    def __init__(self, dataset_node:DSNT, mapping_name: str,
                 suffix:str = None, *,
                 flag_name = "",
                 read_func:Callable[[str], VDMT] = None, 
                 write_func:Callable[[str, VDMT], None] = None, 
                 value_type:Callable = None,
                 filllen = 6, 
                 fillchar = '0',
                 alternate_suffix:list = None, 
                 read_func_args = None,
                 read_func_kwargs = None,
                 write_func_args = None,
                 write_func_kwargs = None,
                 array_shape:tuple[int] = None, 
                 **kwargs
                 ) -> None:
        super().__init__(dataset_node, mapping_name, 
                         suffix=suffix, 
                         flag_name = flag_name,
                         read_func=read_func, 
                         write_func=write_func, 
                         value_type=value_type, 
                         filllen=filllen, 
                         fillchar=fillchar, 
                         alternate_suffix=alternate_suffix, 
                         read_func_args = read_func_args,
                         read_func_kwargs = read_func_kwargs,
                         write_func_args = write_func_args,
                         write_func_kwargs = write_func_kwargs,
                         **kwargs)
        self.array_shape:tuple[int]     = array_shape if array_shape is not None else (-1,)

        rebind_methods(self.read_meta, self.read_meta.inv_format_value, self.read_inv_format)
        rebind_methods(self.write_meta, self.write_meta.format_value, self.write_format)

    @staticmethod
    def read_inv_format(self:IOMeta["NdarrayAsTxtCluster", np.ndarray, UnifiedFilesHandle], array:np.ndarray):
        '''
        array: np.ndarray [N, 5]
        '''
        array = array.reshape(self.files_cluster.array_shape)
        return array
    
    @staticmethod
    def write_format(self:IOMeta, value:np.ndarray) -> Any:
        array = np.array(value)
        if len(array.shape) == 0:
            array = array.reshape((1, 1))
        array = array.reshape((array.shape[0], -1))
        return array

IntArrayDict = dict[int, np.ndarray]
class IntArrayDictAsTxtCluster(NdarrayAsTxtCluster[UFH, INDADC, DSNT, IntArrayDict], Generic[UFH, INDADC, DSNT]):

    DEFAULT_VALUE_TYPE = dict

    @staticmethod
    def read_inv_format(self:IOMeta["IntArrayDictAsTxtCluster", IntArrayDict, UnifiedFilesHandle], array:np.ndarray):
        '''
        array: np.ndarray [N, 5]
        '''
        dict_ = {}
        array = np.reshape(array, (-1, array.shape[-1]))
        for i in range(array.shape[0]):
            dict_[int(array[i, 0])] = array[i, 1:].reshape(self.files_cluster.array_shape)
        return dict_
    
    @staticmethod
    def write_format(self:IOMeta["IntArrayDictAsTxtCluster", IntArrayDict, UnifiedFilesHandle], value:IntArrayDict) -> Any:
        if value is None:
            return None
        array = []
        for i, (k, v) in enumerate(value.items()):
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            array.append(np.concatenate([np.array([k]).astype(v.dtype), v.reshape(-1)]))
        array = np.stack(array)
        return array