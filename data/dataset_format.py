# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_items, dict_keys, dict_values
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
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

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Generator
from functools import partial

from . import Posture, JsonIO, JSONDecodeError, Table, extract_doc, search_in_dict, int_str_cocvt
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta

DCT  = TypeVar('DCT',  bound="_DataCluster") # type of the cluster
DSNT = TypeVar('DSNT', bound='DatasetNode') # dataset node type
VDCT = TypeVar('VDCT') # type of the value of data cluster
VDST = TypeVar('VDST') # type of the value of dataset
from numpy import ndarray

def as_dict(ids, objs):
    if objs is None:
        return None
    else:
        return dict(zip(ids, objs))

def savetxt_func(fmt=...):
    return lambda path, x: np.savetxt(path, x, fmt=fmt, delimiter='\t')

def loadtxt_func(shape:tuple[int]):
    return lambda path: np.loadtxt(path).reshape(shape)

class WriteController(ABC):
    '''
    control the write operation.
    the subclass of WriteController must implement :
    * start_writing
    * stop_writing
    '''
    WRITING_MARK = '.writing'

    LOG_READ = -1
    LOG_APPEND = 0
    LOG_REMOVE = 1
    LOG_CHANGE = 2
    LOG_MOVE   = 3
    LOG_OPERATION = 4
    LOG_KN = [LOG_APPEND, LOG_REMOVE, LOG_CHANGE, LOG_MOVE, LOG_OPERATION]
    class __Writer():
        def __init__(self, writecontroller:"WriteController") -> None:
            self.writecontroller = writecontroller
            self.__overwrite_allowed = False

        def allow_overwriting(self):
            self.__overwrite_allowed = True
            return self

        def __enter__(self):
            self.writecontroller.start_writing(self.__overwrite_allowed)
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                self.writecontroller.stop_writing()
                self.__overwrite_allowed = False
                return True
    
    def __init__(self) -> None:
        self.__writer = self.__Writer(self)
        self.is_writing = False

    @property
    def writer(self):
        return self.__writer

    @abstractmethod
    def get_writing_mark_file(self) -> str:
        pass

    def start_writing(self, overwrite_allowed = False):
        if self.is_writing:
            return False
        else:
            self.is_writing = True
            with open(self.get_writing_mark_file(), 'w'):
                pass
            return True

    def stop_writing(self):
        if self.is_writing:
            if os.path.exists(self.get_writing_mark_file()):
                os.remove(self.get_writing_mark_file())
            self.is_writing = False
            return True
        else:
            return False

    def mark_exist(self):
        return os.path.exists(self.get_writing_mark_file())

    def remove_mark(self):
        if self.mark_exist():
            os.remove(self.get_writing_mark_file())

    def load_from_mark_file(self):
        file_path = self.get_writing_mark_file()
        if os.path.exists(file_path):
            result = []
            with open(file_path, 'r') as file:
                for line in file:
                    # 使用strip()函数移除行末尾的换行符，并使用split()函数分割列
                    columns = line.strip().split(', ')
                    assert len(columns) == 3, f"the format of {file_path} is wrong"
                    log_type, key, value_str = columns
                    # 尝试将第二列的值转换为整数
                    log_type = int(log_type)
                    assert log_type in self.LOG_KN, f"the format of {file_path} is wrong"
                    try: key = int(key)
                    except ValueError: pass
                    if value_str == 'None':
                        value = None
                    else:
                        try: value = int(value_str)
                        except: value = value_str
                    result.append([log_type, key, value])
            return result
        else:
            return None

    def log_to_mark_file(self, log_type, data_i, value=None):
        ## TODO
        if data_i is None or value is None:
            return 
        assert log_type in self.LOG_KN, f"log_type must be in {self.LOG_KN}"
        assert isinstance(data_i, int), f"data_i must be int"

        file_path = self.get_writing_mark_file()
        with open(file_path, 'a') as file:
            line = f"{log_type}, {data_i}, {type(value)}\n"
            file.write(line)

#### Warning Types ####
class ClusterDataIOError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ElementsAmbiguousError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterWarning(Warning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterParaWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterIONotExecutedWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterNotRecommendWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class InstanceRegistry(ABC):
    _cluster_registry = {}
    _dataset_registry = {}
    _registry:dict = None

    def __init_subclass__(cls, **kwargs) -> None:
        if cls.__name__ == "_DataCluster" or ("_DataCluster" in globals() and issubclass(cls, _DataCluster)):
            cls._registry = cls._cluster_registry
        elif cls.__name__ == "DatasetNode" or ("DatasetNode" in globals() and issubclass(cls, DatasetNode)):
            cls._registry = cls._dataset_registry
        else:
            raise TypeError(f"invalid subclass {cls}")
        cls._org_init__ = cls.__init__

        def decorated_init(obj:InstanceRegistry, *args, **kwargs):
            if obj in obj._registry.values():
                try:    obj._InstanceRegistry_inited
                except: return cls._org_init__(obj, *args, **kwargs)
            else:
                return cls._org_init__(obj, *args, **kwargs)
            
        cls.__init__ = decorated_init
        return super().__init_subclass__()

    def __new__(cls, *args, **kwargs):
        instance = super(InstanceRegistry, cls).__new__(cls)
        instance._init_identity_paramenter(*args, **kwargs)
        identity_string = instance.identity_string()
        if identity_string in cls._registry:
            return cls._registry[identity_string]
        # , dataset_node, sub_dir, register, name, *args, **kwargs)
        cls._registry[identity_string] = instance
        return instance
    
    def __init__(self) -> None:
        super().__init__()
        self._InstanceRegistry_inited = True

    @abstractmethod
    def _init_identity_paramenter(self):
        pass

    @abstractmethod
    def identity_string(self):
        pass

    @classmethod
    @abstractmethod
    def gen_identity_string(cls, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def parse_identity_string(identity_string):
        pass

class _IOMeta(Generic[DCT, VDCT]):
    '''
    the position parameter of 'record', '_call' , '__call__' must be the same
    '''
    IO_TYPE = WriteController.LOG_READ
    def __init__(self, cluster:DCT) -> None:
        super().__init__()
        self.cluster:DCT = cluster
        self.warning_info:str = "no description"

        self._kwargs = {}
        self.__once = False

    def record(self, **kwargs):
        pass

    def check_key(self, key, **kwargs) -> bool:
        return True
    
    def check_value(self, value, **kwargs) -> bool:
        return True

class _ReadMeta(_IOMeta[DCT, VDCT], ABC):
    @abstractmethod
    def _call(self, key, **kwargs: Any) -> VDCT:
        pass

    def __call__(self, key, *args, force = False, **kwargs) -> VDCT:
        rlt = self.cluster._read_decorator(self)(key, *args, force = force, **kwargs)
        return rlt
    
    def record(self, key, *args, **kwargs):
        pass

class _WriteMeta(_IOMeta[DCT, VDCT], ABC):
    '''
    abstractmethod
    --
    * _call: the function to call when write
    * is_overwriting: check if the data is overwriting

    recommend to implement
    --
    * check_key: check if the key is valid
    * check_value: check if the value is valid
    '''
    @abstractmethod
    def _call(self, key, value, **kwargs: Any):
        pass
    
    @abstractmethod
    def is_overwriting(self, key, value, **kwargs):
        pass

    def __call__(self, key, value, *args, force = False, **kwargs) -> Any:
        rlt = self.cluster._write_decorator(self)(key, value, *args, force = force, **kwargs)
        return rlt    
    
    def record(self, key, value, **kwargs):
        pass

#### _DataCluster ####

class _DataCluster(WriteController, InstanceRegistry, ABC, Generic[DSNT, VDCT, DCT]):
    '''
    This is a private class representing a data cluster used for managing datasets with a specific format.

    # attr
    ----
    * self.dataset_node: DatasetNode
    * self.closed: bool, Control the shielding of reading and writing, 
        if it is true, the instance will not write, and the read will get None
    * register: bool, whether to register to dataset_node
    * _unfinished: bool, whether the data is unfinished
    * _closed: bool, Indicates whether the cluster is closed or open.
    * _readonly: bool, Indicates whether the cluster is read-only or write-enabled.
    * changes_unsaved: bool, Indicates if any changes have been made to the cluster.
    * directory: str, Directory path for the cluster.


    # property
    -----
    * overwrite_allowed: bool, Control the shielding of writing,
    * cluster_data_num: int, the number of data in the cluster
    * cluster_data_i_upper: int, the upper of the iterator, it is the max index of the iterator + 1
    * changed_since_opening: bool, Indicates whether the cluster has been modified since last opening.

    # method
    -----
    abstract method:
    -----
    - __len__: return the number of data in the cluster
    - keys: return the keys of the cluster
    - values: return the values of the cluster
    - items: return the items of the cluster(key and value)
    - _read: read data from the cluster
    - _write: write data to the cluster
    - _clear: clear all data of the cluster
    - _copyto: copy the cluster to dst
    - _merge: merge the cluster to self, the cluster must has the same type as self

    recommend to implement:
    -----
    - _init_attr: initialize additional attributes specified by subclasses.
    - _update_cluster_inc: update the incremental modification of the cluster after writing
    - _update_cluster_all: update the state of the cluster after writing
    - __getitem__: return the value of the key
    - __setitem__: set the value of the key    
    - _open: operation when open the cluster
    - _close: operation when close the cluster
    - _start_writing: operation when start writing
    - _stop_writing: operation when stop writing
    - check_key: check if the key is valid

    not need to implement:
    -----
    - __iter__: return the iterator of the cluster
    - open: open the cluster for operation.
    - close: close the cluster, preventing further operations.
    - is_close: check if the cluster is closed.
    - set_readonly: set the cluster as read-only or write-enabled.
    - set_writable: set the cluster as writable or read-only.
    - set_overwrite_allowed: set the cluster as writable or read-only.
    - is_readonly: check if the cluster is read-only.
    - is_writeable: check if the cluster is writable.
    - is_overwrite_allowed: check if the cluster is writable.
    - _read_decorator: decorator function to handle reading operations when the cluster is closed.
    - _write_decorator: decorator function to handle writing operations when the cluster is closed or read-only.
    - clear: clear any data in the cluster. Subclasses may implement _clear.
    - read: read data from the cluster. Subclasses must implement _read.
    - write: write data to the cluster. Subclasses must implement _write.
    - copyto: copy the cluster to dst
    - merge: merge the cluster to self, the cluster must has the same type as self
    - start_writing: start writing
    - stop_writing: stop writing
    - __repr__: return the representation of the cluster
    - register_to_format: register the cluster to dataset_node
    '''
    class _Force():
        def __init__(self, obj, force = False, writing = False) -> None:
            self.obj:_DataCluster = obj
            self.orig_closed = None
            self.orig_readonly = None
            self.orig_overwrite_allowed = None
            self.force = force
            self.writing = writing

        def __enter__(self):
            if self.force:
                self.orig_closed = self.obj.is_closed()
                self.orig_readonly = self.obj.is_readonly()
                self.orig_overwrite_allowed = self.obj.overwrite_allowed
                self.obj.open()
                if self.writing:
                    self.obj.set_writable()
                    self.obj.set_overwrite_allowed(True)
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                if self.force:
                    if self.orig_closed:
                        self.obj.close()
                    if self.orig_readonly:
                        self.obj.set_readonly()
                    self.obj.set_overwrite_allowed(self.orig_overwrite_allowed)
                return True

    _DCT = TypeVar('_DCT', bound="_DataCluster")
    _VDCT = TypeVar('_VDCT')

    class _read(_ReadMeta[_DCT, _VDCT]):
        def check_key(self, key, **kwargs) -> bool:
            return key in self.cluster.keys()

    class _write(_WriteMeta[_DCT, _VDCT]):
        IO_TYPE = WriteController.LOG_APPEND

        def is_overwriting(self, data_i, value, **kwargs):
            return data_i in self.cluster.keys()
        
    class _remove(_WriteMeta[_DCT, _VDCT]):
        IO_TYPE = WriteController.LOG_REMOVE

        def check_key(self, key, **kwargs) -> bool:
            return key in self.cluster.keys()

        def is_overwriting(self, data_i, value, **kwargs):
            return True
            
        def __call__(self, key, *args, force=False, **kwargs) -> Any:
            return super().__call__(key, None, force=force, **kwargs)
    
    class _move(_WriteMeta[_DCT, _VDCT]):
        IO_TYPE = WriteController.LOG_MOVE

        def check_key(self, key, **kwargs) -> bool:
            return key in self.cluster.keys()
        
        def check_value(self, value, **kwargs) -> bool:
            return value not in self.cluster.keys()

        def is_overwriting(self, data_i, value, **kwargs):
            return value in self.cluster.keys()
        
        def __call__(self, src, dst, *args, force=False, **kwargs) -> Any:
            return super().__call__(src, dst, force=force, **kwargs)

    class _load_in(_WriteMeta[_DCT, _VDCT]):
        IO_TYPE = WriteController.LOG_APPEND
        
        def __call__(self, src, dst, *args, force=False, **kwargs) -> Any:
            return super().__call__(src, dst, *args, force=force, **kwargs)

        def check_value(self, dst, **kwargs) -> bool:
            return dst not in self.cluster.keys()
        
        def is_overwriting(self, src, dst, **kwargs):
            return dst in self.cluster.keys()
        
    def __init__(self, dataset_node:DSNT, sub_dir: str, register=True, name = "", *args, **kwargs) -> None:
        '''
        Initialize the data cluster with the provided dataset_node, sub_dir, and registration flag.
        '''
        WriteController.__init__(self)    
        InstanceRegistry.__init__(self)
        # self._init_identity_paramenter(dataset_node, sub_dir, register, name, *args, **kwargs)
        self.dataset_node:DSNT = dataset_node
        self.sub_dir = os.path.normpath(sub_dir) 
        self.name = name       
        self.directory = os.path.normpath(os.path.join(self.dataset_node.directory, self.sub_dir))  # Directory path for the cluster.        
        self.register = register
        self._unfinished = self.mark_exist()
        self._unfinished_operation = self.dataset_node._unfinished_operation
        self._error_to_load = False
        self._changed = False  # Indicates if any changes have been made to the cluster. 
        self._data_i_upper = 0      

        self.__closed = True  # Indicates whether the cluster is closed or open.
        self.__readonly = True  # Indicates whether the cluster is read-only or write-enabled.
        self.__overwrite_allowed = False

        self._init_attr(*args, **kwargs)  # Initializes additional attributes specified by subclasses.
     
        if os.path.exists(self.directory) and not self._error_to_load:
            self.open()  # Opens the cluster for operation.
        else:
            self.close()
    
        self.process_unfinished()   
        self.register_to_dataset()

    def _init_attr(self, *args, **kwargs):
        '''Method to initialize additional attributes specified by subclasses.'''
        self.read:_ReadMeta[DCT, VDCT]          = self._read(self)
        self.write:_WriteMeta[DCT, VDCT]        = self._write(self)
        self.remove:_WriteMeta[DCT, VDCT]       = self._remove(self)
        self.move:_WriteMeta[DCT, VDCT]         = self._move(self)
        self.load_in:_WriteMeta[DCT, VDCT]      = self._load_in(self)   

    ### implement InstanceRegistry ###
    def _init_identity_paramenter(self, dataset_node:DSNT, sub_dir: str, register=True, name = "", *args, **kwargs):
        self.dataset_node = dataset_node
        self.sub_dir = sub_dir
        self.name = name
        self.directory = os.path.normpath(os.path.join(self.dataset_node.directory, self.sub_dir))

    @classmethod
    def gen_identity_string(cls, dataset_node:"DatasetNode", sub_dir, name, *args, **kwargs):
        return f"{cls.__name__}({dataset_node.__class__.__name__}, {dataset_node.directory}, {sub_dir}, {name})"

    @staticmethod
    def parse_identity_string(identity_string):
        import re
        pattern = re.compile(r"(\w*)\((\w*), (.*), (.*), (.*)\)")
        match = pattern.match(identity_string)
        if match:
            cls_name, dataset_node_cls_name, dataset_node_dir, sub_dir, name = match.groups()
            return cls_name, dataset_node_cls_name, dataset_node_dir, sub_dir, name
        else:
            raise ValueError(f"invalid identity string {identity_string}")

    def identity_string(self):
        return self.gen_identity_string(self.dataset_node, self.sub_dir, self.name)

    def key_identity_string(self):
        if self.name != "":
            return self.sub_dir + ":" + self.name
        else:
            return self.sub_dir
    ###

    ### implement WriteController ###
    def get_writing_mark_file(self):
        path = os.path.join(self.directory, self.name)
        if '.' in path:
            return path + self.WRITING_MARK
        else:
            return os.path.join(path, self.WRITING_MARK)

    def _start_writing(self):
        pass

    def _stop_writing(self):
        self._changed = False  # Resets the updated flag to false.
        # self.__closed = True  # Marks the cluster as closed.

    def start_writing(self, overwrite_allowed = False):
        '''
        rewrite the method of WriteController
        '''
        if super().start_writing(overwrite_allowed):
            self.open()
            self.set_writable()
            self._start_writing()
            if not self.overwrite_allowed:
                self.set_overwrite_allowed(overwrite_allowed)
            return True
        else:
            return False

    def stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        if not self.is_writing:
            return False
        else:
            self.set_overwrite_allowed(False)
            self._stop_writing()
            # self._update_cluster_all()
            return super().stop_writing()
    ### 

    @property
    def overwrite_allowed(self):
        '''Property that returns whether the cluster format allows write operations.'''
        return self.__overwrite_allowed

    @property
    def cluster_data_num(self):
        return len(self)

    @property
    def cluster_data_i_upper(self):
        return max(self.keys()) + 1 if len(self) > 0 else 0

    @property
    def changed_since_opening(self):
        '''Indicates whether the cluster has been modified since last opening.'''
        return self._changed
    
    @changed_since_opening.setter
    def changed_since_opening(self, value:bool):
        self._changed = bool(value)
        self.dataset_node.updated = True

    @property
    def continuous(self):
        return self.cluster_data_num == self.cluster_data_i_upper

    @abstractmethod
    def __len__(self):
        '''Returns the number of data in the cluster.'''
        pass     

    @abstractmethod
    def keys(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def values(self) -> Iterable[VDCT]:
        pass

    @abstractmethod
    def items(self) -> Iterable[tuple[Any, VDCT]]:
        pass

    @abstractmethod
    def get_file_path(self, data_i, **kwargs):
        pass

    def _update_cluster_inc(self, iometa:_IOMeta, data_i, value, **kwargs):
        '''
        update the state of the cluster after writing
        '''
        pass

    def _update_cluster_all(self, **kwargs):
        pass
   
    def __getitem__(self, data_i:Union[int, slice]):
        if isinstance(data_i, slice):
            # 处理切片操作
            start, stop, step = data_i.start, data_i.stop, data_i.step
            if start is None:
                start = 0
            if step is None:
                step = 1
            def g():
                for i in range(start, stop, step):
                    yield self.read(i)
            return g()
        elif isinstance(data_i, int):
            # 处理单个索引操作
            return self.read(data_i)
        else:
            raise TypeError("Unsupported data_i type")
    
    def __setitem__(self, data_i, value:VDCT):
        return self.write(data_i, value)

    def __iter__(self) -> Iterable[VDCT]:
        return self.values()

    def choose_unfinished_operation(obj):
        '''
            0. skip
            1. clear the unfinished data
            2. try to rollback the unfinished data
            3. exit"))
        '''
        if isinstance(obj, _DataCluster):
            tip_0 = "skip"
        elif isinstance(obj, DatasetFormat):
            tip_0 = "decide one by one"
        choice = int(input(f"please choose an operation to continue:\n\
                    0. {tip_0}\n\
                    1. clear the unfinished data\n\
                    2. try to rollback the unfinished data\n\
                    3. exit\n"))
        if choice not in [0, 1, 2, 3]:
            raise ValueError(f"invalid choice {choice}")
        return choice

    def check_key(self, key) -> bool:
        return True

    ### IO mode ###
    def _open(self):
        '''Method to open the cluster for operation.''' 
        return True

    def _close(self):
        return True    
    
    def open(self):
        if self.__closed == True:
            self.__closed = not self._open()

    def close(self):
        '''Method to close the cluster, preventing further operations.'''
        if self.__closed == False:
            self.set_readonly()
            self.__closed = self._close()

    def is_closed(self, with_warning = False):
        '''Method to check if the cluster is closed.'''
        if with_warning and self.__closed:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is closed, any I/O operation will not be executed.",
                            ClusterIONotExecutedWarning)
        return self.__closed

    def is_opened(self):
        return not self.is_closed()

    def set_readonly(self, readonly=True):
        '''Method to set the cluster as read-only or write-enabled.'''
        if self.is_writable() and readonly == True:
            self.stop_writing()
        self.__readonly = readonly

    def set_writable(self, writable=True):
        '''Method to set the cluster as writable or read-only.'''
        self.set_readonly(not writable)

    def set_overwrite_allowed(self, overwrite_allowed=True):
        '''Method to set the cluster as writable or read-only.'''
        self.__overwrite_allowed = overwrite_allowed

    def is_readonly(self, with_warning = False):
        '''Method to check if the cluster is read-only.'''
        if with_warning and self.__readonly:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is read-only, any write operation will not be executed.",
                ClusterIONotExecutedWarning)
        return self.__readonly
    
    def is_writable(self):
        return not self.is_readonly()

    def is_overwrite_allowed(self):
        return self.__overwrite_allowed
    ########################

    ### IO operation ###
    def _read_decorator(self, iometa:_ReadMeta):
        '''
        brief
        -----
        Decorator function to handle reading operations when the cluster is closed. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        
        parameter
        -----
        func: Callable, the decorated function
        '''
        func = iometa._call
        warning_info:str = iometa.warning_info
        def wrapper(data_i, *args, force = False, **kwargs):
            nonlocal self, warning_info
            with self._Force(self, force, False): 
                read_error = False
                if self.is_closed(with_warning=True):
                    return None
                elif not iometa.check_key(data_i, **kwargs):
                    warning_info = f"key:{data_i} is not valid"
                    read_error = True
                
                try:
                    rlt = func(data_i, *args, **kwargs)  # Calls the original function.
                except ClusterDataIOError as e:
                    rlt = None
                    if str(e) == "skip":
                        pass
                    else:
                        read_error = True
                
                if read_error:
                    warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                                read {data_i} failed:\
                                    {warning_info}", ClusterIONotExecutedWarning)
            return rlt
        return wrapper

    def _write_decorator(self, iometa:_WriteMeta):
        '''
        brief
        -----
        Decorator function to handle writing operations when the cluster is closed or read-only. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        if the cluster is read-only and the decorated function is a writing operation, the decorated function will not be executed and return None.\n
        
        parameter
        -----
        func: Callable, the decorated function
        '''
        func = iometa._call
        log_type:int = iometa.IO_TYPE
        warning_info:str = iometa.warning_info

        def wrapper(data_i, value = None, *args, force = False, **kwargs):
            nonlocal self, log_type, warning_info
            with self._Force(self, force): 
                write_error = False
                overwrited = False

                if self.is_closed(with_warning=True) or self.is_readonly(with_warning=True):
                    return None
                elif not iometa.check_key(data_i, **kwargs):
                    warning_info = f"key:{data_i} is not valid"
                    write_error = True
                elif not iometa.check_value(value, **kwargs):
                    warning_info = f"value:{value} is not valid"
                    write_error = True              
                elif iometa.is_overwriting(data_i, value, **kwargs):
                    if not self.overwrite_allowed and not force:
                        warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                                    is not allowed to overwitre, any write operation will not be executed.",
                                        ClusterIONotExecutedWarning)
                        write_error = True
                        return None
                    overwrited = True 

                if not write_error:                
                    try:
                        if not self.is_writing:
                            self.start_writing()
                        rlt = func(data_i, value, *args, **kwargs)  # Calls the original function.
                    except ClusterDataIOError as e:
                        rlt = None
                        if str(e) == "skip":
                            pass
                        else:
                            write_error = True     
                    else:
                        self.changed_since_opening = True  # Marks the cluster as updated after writing operations.
                        if overwrited and log_type == self.LOG_APPEND:
                            log_type = self.LOG_CHANGE
                        self._update_cluster_inc(iometa, data_i, value, *args, **kwargs)
                        self.log_to_mark_file(log_type, data_i, value)
                
                if write_error:
                    warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                        write {data_i}, {value} failed:\
                        {warning_info}", ClusterIONotExecutedWarning)

            return rlt
        return wrapper
    #####################

    # complex io #
    def clear(self, *, force = False, ignore_warning = False):
        '''
        Method to clear any data in the cluster. Subclasses may implement this method.
        * it is dargerous
        '''
        if not ignore_warning:
            y = input("All files in {} will be removed, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            with self._Force(self, force, True): 
                for data_i in tqdm(list(self.keys()), desc=f"clear {self.sub_dir}"):
                    self.remove(data_i)
            return True
        else:
            return False

    def make_continuous(self, *args, **kwargs):
        if self.continuous:
            return
        for new_i, data_i in tqdm(enumerate(self.keys()), desc=f"make {self.sub_dir} continuous"):
            if data_i != new_i:
                self.move(data_i, new_i)

    def merge_from(self, src:"_DataCluster", *args, **kwargs):
        assert type(src) == type(self), f"can't merge {type(src)} to {type(self)}"
        assert self.is_opened() and self.is_writable(), f"{self.__class__.__name__}:{self.sub_dir} is not writable"
        assert src.is_opened(), f"{src.__class__.__name__}:{src.sub_dir} is not opened"
        for data_i in tqdm(src.keys(), desc=f"merge {src.sub_dir} to {self.sub_dir}", total=len(src)):
            self.load_in(src.get_file_path(data_i), self.cluster_data_i_upper, *args, **kwargs)

    @abstractmethod
    def copyto(self, dst:str, *args, **kwargs):
        pass

    ###############

    def __repr__(self):
        return f"{self.identity_string()} at {hex(id(self))}"

    def register_to_dataset(self):
        if self.register:
            self.dataset_node.cluster_map.add_cluster(self)

    def unregister_from_dataset(self):
        if self.identity_string() in self.dataset_node.cluster_map:
            self.dataset_node.cluster_map.remove_cluster(self)

    def process_unfinished(self):
        ### TODO
        pass
        if self._unfinished:
            if self._unfinished_operation == 0:
                self._unfinished_operation = self.choose_unfinished_operation()
            if self._unfinished_operation == 0:
                return False
            self._update_cluster_all()
            if self._unfinished_operation == 1:
                self.clear(force=True, ignore_warning=True)
                self._unfinished = False
                self.remove_mark()
                self._update_cluster_all()
                return True
            elif self._unfinished_operation == 2:
                # try roll back
                log = self.load_from_mark_file()
                with self._Force(self, True, True):
                    for log_i, data_i, _ in log:
                        if log_i == self.LOG_APPEND:
                            if data_i in self.keys():
                                self.remove(data_i)
                                print(f"try to rollback, {data_i} in {self.identity_string()} is removed.")
                        else:
                            raise ValueError("can not rollback")
                self.remove_mark()
                return True
            elif self._unfinished_operation == 3:
                # reinit
                self.remove_mark()
                self._update_cluster_all()
            else:
                raise ValueError(f"invalid operation {self._unfinished_operation}")
        if self._unfinished_operation == 3:
            # reinit
            self._update_cluster_all()

class JsonDict(_DataCluster[DSNT, VDCT, "JsonDict"], dict):
    '''
    dict for base json
    ----
    it is a subclass of dict, so it can be used as a dict \n
    returns None if accessing an key that does not exist

    attr
    ----
    see _DataCluster

    method
    ----
    see _DataCluster
    * clear: clear all data of the dict and clear the json file
    '''
    SAVE_IMMIDIATELY = 0
    SAVE_AFTER_CLOSE = 1
    SAVE_STREAMLY = 2

    class _Placeholder():
        def __init__(self) -> None:
            pass

    def __init__(self, dataset_node:DSNT, sub_dir, register = True, name = "", *args, **kwargs):       
        dict.__init__(self)
        super().__init__(dataset_node, sub_dir, register, name, *args, **kwargs)

    def _init_attr(self, *args, **kwargs):
        _DataCluster._init_attr(self, *args, **kwargs)
        self.reload()
        self.__save_mode = self.SAVE_AFTER_CLOSE
        self.stream = JsonIO.Stream(self.directory)

    @property
    def save_mode(self):
        return self.__save_mode
    
    @save_mode.setter
    def save_mode(self, mode):
        assert mode in [self.SAVE_IMMIDIATELY, self.SAVE_AFTER_CLOSE, self.SAVE_STREAMLY]
        if self.is_writing and mode != self.SAVE_STREAMLY:
            warnings.warn("can't change save_mode from SAVE_STREAMLY to the others while writing streamly", 
                          ClusterParaWarning)
        if self.__save_mode == self.SAVE_AFTER_CLOSE and mode != self.SAVE_AFTER_CLOSE and self.is_opened():
            self.save()
        self.__save_mode = mode

    @property
    def write_streamly(self):
        return self.save_mode == self.SAVE_STREAMLY

    def __len__(self) -> int:
        super().__len__()
        return dict.__len__(self)

    def keys(self):
        super().keys()
        return dict.keys(self)

    def values(self):
        super().values()
        return dict.values(self)

    def items(self):
        super().items()
        return dict.items(self)
    
    def _read(self, data_i, **kwargs):
        super()._read(data_i, **kwargs)
        try:
            return dict.__getitem__(self, data_i)
        except KeyError:
            raise ClusterDataIOError(f"key {data_i} does not exist")

    def _write(self, data_i, value, **kwargs):
        super()._write(data_i, value, **kwargs)

        if self.save_mode == self.SAVE_IMMIDIATELY:
            dict.__setitem__(self, data_i, value)
            self.save()
        elif self.save_mode == self.SAVE_AFTER_CLOSE:
            dict.__setitem__(self, data_i, value)
        elif self.save_mode == self.SAVE_STREAMLY:
            set_value = self._Placeholder()
            if not self.is_writing:
                self.start_writing() # auto start writing if save_mode is SAVE_STREAMLY
            self.stream.write({data_i: value})
            dict.__setitem__(self, data_i, set_value)
        else:
            raise ValueError(f"invalid save_mode {self.save_mode}")
        return self.LOG_APPEND

    def _remove(self, data_i, *args, **kwargs):
        super()._remove(data_i, *args, **kwargs)
        if self.save_mode == self.SAVE_IMMIDIATELY:
            dict.__delitem__(self, data_i)
            self.save()
        elif self.save_mode == self.SAVE_AFTER_CLOSE:
            dict.__delitem__(self, data_i)
        elif self.save_mode == self.SAVE_STREAMLY:
            raise RuntimeError("can't remove while save_mode is SAVE_STREAMLY")
        return self.LOG_REMOVE

    def _clear(self, *args, **kwargs):
        with open(self.directory, 'w'):
            pass
        dict.clear(self)

    def _make_continuous(self, *args, **kwargs):
        assert self.save_mode != self.SAVE_STREAMLY, "can't make_continuous while save_mode is SAVE_STREAMLY"
        super()._make_continuous(*args, **kwargs)
        new_dict = {}
        for i, v in enumerate(self.values()):
            new_dict[i] = v
        self.clear(force=True, ignore_warning=True)
        self.update(new_dict)
        self.save()

    def _copyto(self, dst: str, *args, **kwargs):
        super()._copyto(dst, *args, **kwargs)
        os.makedirs(os.path.split(dst)[0], exist_ok=True)
        shutil.copy(self.directory, dst)

    def _merge_one(self, src: "JsonDict", k, *args, **kwargs):
        ### TODO
        v = src[k]
        if isinstance(k, int):
            self[self.cluster_data_i_upper] = v
        elif isinstance(k, str):
            new_k = k
            _count = 1
            while new_k in self.keys():
                warnings.warn(f"key {new_k} exists, it will be renamed to {new_k}.{str(_count)}", ClusterNotRecommendWarning)
                new_k = k + f".{str(_count)}"
                _count += 1
            self[new_k] = v
        else:
            raise TypeError(f"unsupported type: {type(k)} in src")
        self.log_to_mark_file(self.cluster_data_i_upper - 1, self.LOG_APPEND)

    def __iter__(self) -> Iterator:
        return _DataCluster.__iter__(self)

    def _stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        if self.is_writing:
            self.stop_writing()
        if self.save_mode == self.SAVE_AFTER_CLOSE:
            self.sort()
            self.save()
        super()._stop_writing()

    def update(self, _m:Union[dict, Iterable[tuple]] = None, **kwargs):
        if _m is None:
            warnings.warn(f"It's not recommended to use update() for {self.__class__.__name__} \
                for it will call {self._update_cluster_all.__name__} and spend more time. \
                nothine is done. \
                use {self.__setitem__.__name__} or {self.write.__name__} or {self.__setitem__.__name__} instead", ClusterNotRecommendWarning)
        elif isinstance(_m, dict):
            for k, v in _m.items():
                assert self.check_key(k), f"key {k} is not allowed"
                self[k] = v
        elif isinstance(_m, Iterable):
            assert all([len(x) == 2 for x in _m])
            for k, v in _m:
                assert self.check_key(k), f"key {k} is not allowed"
                self[k] = v
        else:
            raise TypeError(f"unsupported type: {type(_m)}")

    def reload(self, value = {}):
        dict.clear(self)
        if isinstance(value, dict) and len(value) > 0:
            pass
        elif os.path.exists(self.directory):
            try:
                value = JsonIO.load_json(self.directory)
            except JSONDecodeError:
                self._error_to_load = True
                value = {}
        else:
            value = {}
        dict.update(self, value)

    def sort(self, *args, **kwargs):
        '''
        sort by keys
        '''
        raise NotImplementedError
        ### TODO
        if self.is_writing:
            warnings.warn(f"It's no effect to call {self.sort.__name__} while writing", ClusterNotRecommendWarning)
            return 
        new_dict = dict(sorted(self.items(), key=lambda x: x[0]))
        self.reload(new_dict)
        if self.save_mode == self.SAVE_IMMIDIATELY:
            self.save()

    def start_writing(self, overwrite_allowed = False):
        '''
        rewrite the method of WriteController
        '''
        if _DataCluster.start_writing(self, overwrite_allowed):
            self.save_mode = self.SAVE_STREAMLY
            self.stream.open()
            return True
        else:
            return False

    def stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        if not self.is_writing:
            return False
        else:
            self.stream.close()
            self.reload()
            _DataCluster.stop_writing(self)
            return 

    def save(self):
        JsonIO.dump_json(self.directory, self)

class Elements(_DataCluster[DSNT, VDCT, "Elements"]):
    '''
    elements manager
    ----
    Returns None if accessing an data_id that does not exist \n
    will not write if the element is None \n 
    it can be used as an iterator, the iterator will return (data_id, element) \n

    attr
    ----
    see _DataCluster
    * readfunc:  Callable, how to read one element from disk, the parameter must be (path)
    * writefunc: Callable, how to write one element to disk, the parameter must be (path, element)
    * suffix: str, the suffix of the file
    * filllen: int, to generate the store path
    * fillchar: str, to generate the store path
    * _data_i_dir_map: dict[int, str], the map of data_id and directory name
    * data_i_upper: int, the max index of the iterator

    method
    ----
    * __len__: return the number of elements
    * __iter__: return the iterator
    * __next__: return the next element
    * read: read one element from disk with the logic of self.readfunc
    * write: write one element to disk with the logic of self.writefunc
    * format_path: format the path of the element
    * clear: clear all data of the dict and clear directory
    '''

    WRITING_MARK = ".elmsw" # the mark file of writing process, Elements streamly writing

    DIR_MAP_FILE = "data_i_dir_map.elmm"
    APPNAMES_MAP_FILE = "data_i_appnames.elmm"
    DATA_INFO_MAP_FILE = "data_info_map.elmm"
    CACHE_FILE = "cache.elmcache"

    class DataInfoDict(dict[str, Union[str, list, bool]]):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.update({
                "dir": None,
                "appnames": [],
                "file_exist": [],
                "cache": None
            })            
            self.update(*args, **kwargs)
        
        # dir
        @property
        def dir(self)->str:
            return self["dir"]
        
        @dir.setter
        def dir(self, value):
            assert isinstance(value, str), f"dir must be str, not {type(value)}"
            self["dir"] = value

        # appnames
        @property
        def appnames(self) -> tuple[str]:
            return tuple(self["appnames"])

        @appnames.setter
        def appnames(self, value:Iterable[str]):
            assert isinstance(value, Iterable), f"appnames must be list, not {type(value)}"
            assert all([isinstance(x, str) for x in value]), f"appnames must be list[str], not {type(value)}"
            self["appnames"].clear()
            self["appnames"].extend(value)

        @property
        def file_exist(self):
            return tuple(self["file_exist"])

        @file_exist.setter
        def file_exist(self, value:Iterable[bool]):
            assert isinstance(value, Iterable), f"file_exist must be list, not {type(value)}"
            assert all([isinstance(x, bool) for x in value]), f"file_exist must be list[bool], not {type(value)}"
            assert len(value) == len(self["appnames"]), f"file_exist must have the same length as appnames"
            self["file_exist"].clear()
            self["file_exist"].extend(value)

        @property
        def has_notfound(self):
            return any([not e for e in self["file_exist"]])

        # has_cache

        # has_file
        @property
        def has_file(self) -> bool:
            return any(self["file_exist"])

        # cache
        @property
        def cache(self):
            return self.get_cache()

        @property
        def has_cache(self):
            return self.cache is not None

        @property
        def empty(self):
            return not (self.has_file or self.has_cache)

        def add_file(self, appname):
            if appname not in self.appnames:
                self["appnames"].append(appname)
                self["file_exist"].append(True)

        def remove_file(self, appname):
            if appname in self.appnames:
                idx = self.appnames.index(appname)
                self["file_exist"][idx] = False

        def get_file_exist(self, appname):
            if appname in self.appnames:
                idx = self.appnames.index(appname)
                return self.file_exist[idx]
            else:
                return False

        def get_cache(self):
            return self["cache"]
            
        def set_cache(self, cache):
            self["cache"] = cache
            
        def erase_cache(self):
            self["cache"] = None

        def clear_notfound(self, idx):
            new_appnames = []
            for i, e in enumerate(self["file_exist"]):
                if e:
                    new_appnames.append(self["appnames"][i])
            self.appnames = new_appnames
            self["file_exist"] = [True] * len(self.appnames)

        def sort(self):
            new_appnames = []
            new_file_exist = []
            for i in sorted(self.appnames):
                idx = self.appnames.index(i)
                new_appnames.append(i)
                new_file_exist.append(self["file_exist"][idx])
            self["appnames"] = new_appnames
            self["file_exist"] = new_file_exist
        
        def refresh_has_file(self, elem:"Elements", data_i:int):
            for idx, appname in enumerate(self.appnames):
                self["file_exist"][idx] = elem.has_file_of(data_i, self.dir, appname)

    class DataInfoMap(dict[int, DataInfoDict]):
        def __init__(self, elmm:"Elements", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.elements = elmm
            self.directory = self.elements.directory
            self.path = os.path.join(self.elements.directory, self.elements.DATA_INFO_MAP_FILE)
        
        def add_info(self, data_i, subdir, appnames:list[str] = None):
            appnames = [] if appnames is None else appnames
            if data_i not in self.keys():
                self[data_i] = Elements.DataInfoDict()
                self[data_i].dir = subdir
                self[data_i].appnames = appnames
                self[data_i].file_exist = [True] * len(appnames)

        def remove_info(self, data_i):
            if data_i in self.keys():
                self.pop(data_i)

        def add_appname(self, data_i, appname):
            self[data_i].add_file(appname)

        def rebuild(self, ignore_notfound = False):
            paths = glob.glob(os.path.join(self.directory, "**/*" + self.elements.suffix), recursive=True)
            for path in paths:
                data_i, subdir, appname = Elements.parse_path(self.directory, path)
                self.add_info(data_i, subdir)
                self.add_appname(data_i, appname)
            for data_i, info in dict(self).items():
                info.refresh_has_file(self.elements, data_i)
                if info.empty:
                    self.pop(data_i)
                if info.has_notfound:
                    if ignore_notfound:
                        info.clear_notfound()
                    else:
                        notfound_names = [n for n, e in zip(info.appnames, info.file_exist) if e == False]
                        raise ValueError(f"notfound data_i: {data_i}: {notfound_names}")
            # sort by data_i
            new_dict = dict(sorted(self.items(), key=lambda x:x[0]))
            self.clear()
            self.update(new_dict)

        @staticmethod
        def load(elmm:"Elements"):
            path = os.path.join(elmm.directory, elmm.DATA_INFO_MAP_FILE)
            if os.path.exists(path):
                data_info_dict:dict = deserialize_object(path)
                data_info_map = Elements.DataInfoMap(elmm, {int(k): Elements.DataInfoDict(v) for k, v in data_info_dict.items()})
            else:
                data_info_map = Elements.DataInfoMap(elmm)
                data_info_map.rebuild()
                data_info_map.save()

            return data_info_map

        def save(self):
            to_save_dict = {item[0]: dict(item[1]) for item in self.items()}
            if self.elements.is_opened():
                serialize_object(self.path, to_save_dict)

        # def __setitem__(self, __key: int, __value) -> None:
        #     raise NotImplementedError

    _DCT = TypeVar('_DCT', bound="Elements")
    _VDCT = TypeVar('_VDCT')
    class _read(_DataCluster._read[_DCT, _VDCT]):
        def _read_from_cache(self, data_i, **kwargs):
            if data_i in self.cluster.data_info_map:
                value = self.cluster.data_info_map[data_i].get_cache()
                return value
            else:
                return None
        
        def _read_from_file(self, data_i, **kwargs):
            subdir, appnames = self.cluster.auto_path(data_i, return_app=True)
            paths = [self.cluster.format_path(data_i, subdir, appname) for appname in appnames]
            single_values = [self.cluster.read_func(path, **kwargs) for path in paths]
            value = self.cluster.construct_value_func(appnames, single_values)
            return value

        def _call(self, data_i, **kwargs: Any):
            def _read_with_priority(func1, func2):
                value = func1(data_i, **kwargs)
                if value is None:
                    value = func2(data_i, **kwargs)
                if value is None:
                    raise ClusterDataIOError(f"can't find {data_i}")
                return value

            cache_priority = self.cluster.cache_priority

            if cache_priority:
                value = _read_with_priority(self._read_from_cache, self._read_from_file)
            else:
                value = _read_with_priority(self._read_from_file, self._read_from_cache)
            return value

    class _write(_DataCluster._write[_DCT, _VDCT]):
        def check_key(self, data_i, **kwargs) -> bool:
            value = isinstance(data_i, int) and data_i >= 0 
            return value and super().check_key(data_i, **kwargs)

        def _call(self, data_i, value:VDCT, subdir = "", **kwargs: Any):
            path = self.cluster.format_path(data_i, subdir=subdir)
            dir_ = os.path.split(path)[0]
            os.makedirs(dir_, exist_ok=True)
            appnames, single_values = self.cluster.parse_appnames_func(value)
            for n, v in zip(appnames, single_values):
                path = self.cluster.format_path(data_i, subdir, n)
                if value is not None:
                    self.cluster.write_func(path, v, **kwargs)
            
            self.cluster._data_info_map.add_info(data_i, subdir, appnames)

        def __call__(self, data_i, value:VDCT, subdir = "", *args, force=False, **kwargs) -> Any:
            return super().__call__(data_i, value, subdir, *args, force=force, **kwargs)

    class _remove(_DataCluster._remove[_DCT, _VDCT]):
        def _call(self, data_i, value, **kwargs: Any):
            paths = self.cluster.auto_path(data_i, return_app=False, allow_mutil_appendname=True)
            for path in paths:
                if os.path.exists(path):
                    os.remove(path)
            # record
            self.cluster._data_info_map.remove_info(data_i)

    class _move(_DataCluster._move[_DCT, _VDCT]):
        def check_key(self, key, **kwargs) -> bool:
            rlt = isinstance(key, int) and key >= 0
            rlt = rlt and super().check_key(key, **kwargs)
            return rlt

        def check_value(self, value, **kwargs) -> bool:
            rlt = isinstance(value, int) and value >= 0
            rlt = rlt and super().check_value(value, **kwargs)
            return rlt

        def _call(self, src:int, dst:int, *args, **kwargs):
            src_dir, src_appnames = self.cluster.auto_path(src, return_app=True, allow_mutil_appendname=True)
            for name in src_appnames:
                src_path = self.cluster.format_path(src, src_dir, name)
                dst_path = self.cluster.format_path(dst, src_dir, name)
                shutil.move(src_path, dst_path)
            info = self.cluster._data_info_map.pop(src)
            self.cluster._data_info_map.update({dst: info})
            
    class _load_in(_DataCluster._load_in[_DCT, _VDCT]):
        def check_key(self, src:list[str], **kwargs) -> bool:
            rlt = isinstance(src, Iterable)
            rlt = rlt and all([isinstance(x, str) for x in src])
            rlt = rlt and super().check_key(src, **kwargs)

        def check_value(self, dst:int, **kwargs) -> bool:
            rlt = isinstance(dst, int) and dst >= 0
            rlt = rlt and super().check_value(dst, **kwargs)
            return rlt
        
        def _call(self, src:list[str], dst:int, subdir = "", **kwargs: Any):
            src_appnames = []
            for sp in src:
                name = os.path.split(sp)[1] 
                main_name = os.path.splitext(name)[0]
                src_appnames.append(main_name.split('_')[-1])
            dst_paths = []
            for appname in src_appnames:
                dst_paths.append(self.cluster.format_path(dst, subdir, appname))
            for sp, dp in zip(src, dst_paths):
                shutil.copy(sp, dp)
            self.cluster._data_info_map.add_info(dst, subdir, src_appnames)

        def __call__(self, data_i, value:VDCT, subdir = "", *args, force=False, **kwargs) -> Any:
            return super().__call__(data_i, value, subdir, *args, force=force, **kwargs)
            
    class _change_dir(_WriteMeta[_DCT, _VDCT]):
        IO_TYPE = WriteController.LOG_OPERATION

        def is_overwriting(self, data_i, value, **kwargs):
            return False
        
        def check_key(self, key, **kwargs) -> bool:
            rlt = isinstance(key, int) and key >= 0
            rlt = rlt and key in self.cluster.keys()
            rlt = rlt and super().check_key(key, **kwargs)
            return rlt

        def check_value(self, value:str, **kwargs) -> bool:
            rlt = isinstance(value, str)
            rlt = rlt and super().check_value(value, **kwargs)
            return rlt

        def _call(self, data_i:int, dst_dir:str, **kwargs: Any):
            dir_, appnames = self.cluster.auto_path(data_i, return_app=True)
            for appname in appnames:
                src_path = self.cluster.format_path(data_i, dir_, appname)
                dst_path = self.cluster.format_path(data_i, dst_dir, appname)
                os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
                shutil.move(src_path, dst_path)
            # record
            info = self.cluster._data_info_map[data_i]
            info.dir = dst_dir
        
        def __call__(self, data_i:int, dst_dir:str, *args, force=False, **kwargs) -> Any:
            return super().__call__(data_i, dst_dir, *args, force=force, **kwargs)
        
    def __init__(self, 
                dataset_node:DSNT,
                sub_dir,
                register = True,
                name = "",
                read_func:Callable = lambda x: None, 
                write_func:Callable = lambda x,y: None, 
                suffix:str = '.txt', 
                filllen = 6, 
                fillchar = '0', *,
                alternative_suffix:list[str] = []) -> None:
        _cvt_old_element_map(dataset_node, sub_dir, suffix)
        super().__init__(dataset_node, sub_dir, register, name, 
                         read_func, write_func, suffix, filllen, fillchar, alternative_suffix = alternative_suffix)
        # self-check

    def _init_attr(self, read_func, write_func, suffix, filllen, fillchar, alternative_suffix = [], *args, **kwargs):
        def _check_dot(suffix:str):
            if not suffix.startswith('.'):
                suffix = '.' + suffix
            return suffix

        super()._init_attr(*args, **kwargs)
        assert isinstance(suffix, str), f"suffix must be str, not {type(suffix)}"
        assert isinstance(filllen, int), f"filllen must be int, not {type(filllen)}"
        assert isinstance(fillchar, str), f"fillchar must be str, not {type(fillchar)}"
        assert isinstance(alternative_suffix, (list, tuple)), f"alternative_suffix must be list or tuple, not {type(alternative_suffix)}"
        # self.DIR_MAP_NAME = "data_i_dir_map.elmm"
        # self.APPNAMES_MAP_NAME = "data_i_appnames.elmm"
        # self.CACHE_NAME = "cache.elmcache"

        self.filllen    = filllen
        self.fillchar   = fillchar
        self.suffix     = suffix
        self.__alternative_suffix = alternative_suffix
        self.suffix = _check_dot(self.suffix)
        self.__alternative_suffix = [_check_dot(x) for x in self.__alternative_suffix]

        self.read_func  = read_func
        self.write_func = write_func

        self.read: Elements._read[DCT, VDCT]                 = self.read
        self.write:Elements._write[DCT, VDCT]                = self.write
        self.remove:Elements._write[DCT, VDCT]               = self.remove
        self.move:Elements._write[DCT, VDCT]                 = self.move
        self.load_in:Elements._write[DCT, VDCT]              = self.load_in
        self.change_dir:Elements._write["Elements", VDCT]    = self._change_dir(self)
        
        self._data_info_map = self.DataInfoMap.load(self)

        self.cache_priority = True

    @property
    def data_info_map(self):
        return types.MappingProxyType(self._data_info_map)

    def has_file_of(self, data_i, subdir, appname):
        return os.path.exists(self.format_path(data_i, subdir=subdir, appname=appname))

    def __len__(self):
        '''
        Count the total number of files in the directory
        '''
        super().__len__()
        return len(self.data_info_map)

    def keys(self):
        '''
        brief
        -----
        return a generator of data_i
        * Elements is not a dict, so it can't be used as a dict.
        '''
        super().keys()
        return self.data_info_map.keys()
    
    def values(self):
        super().values()
        def value_generator():
            for i in self.data_info_map.keys():
                yield self.read(i)
        return value_generator()
    
    def items(self):
        super().items()
        def items_generator():
            for i in self.data_info_map.keys():
                yield i, self.read(i)
        return items_generator()
     
    def get_file_path(self, data_i, *args, **kwargs):
        return self.auto_path(data_i)

    def copyto(self, dst: str, *args, **kwargs):
        if not self.cache_priority:
            shutil.copytree(self.directory, dst)
        else:
            os.makedirs(dst, exist_ok=True)
            shutil.copy(os.path.join(self.directory, self.DATA_INFO_MAP_FILE),
                        os.path.join(dst, self.DATA_INFO_MAP_FILE))
            for data_i, info in self._data_info_map.items():
                if not info.has_cache:
                    for appname in info.appnames:
                        src_path = self.format_path(data_i, info.dir, appname)
                        dst_path = self.format_path(data_i, info.dir, appname, dst)
                        os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
                        shutil.copy(src_path, dst_path)

    def _update_cluster_inc(self, iometa:_IOMeta, data_i, value, subdir = "", *args, **kwargs):
        pass

    def _update_cluster_all(self, *args, **kwargs):
        self._data_info_map.rebuild()
        self._data_info_map.save()

    def __getitem__(self, idx):
        return self.read(idx)
        
    def __setitem__(self, idx, value):
        if idx in self.data_info_map:
            subdir, appnames = self.auto_path(idx, return_app=True, allow_mutil_appendname=False)
            self.write(idx, value, subdir=subdir, appname = appnames[0])
        else:
            raise KeyError(f'idx {idx} not in {self.directory}, if you want to add new data, \
                           please use method:write to specify the subdir and appname')

    def __iter__(self):
        return self.values()
    
    def _open(self):
        open_allowed = super()._open()
        if open_allowed and not os.path.exists(self.directory):
            print(f"Elements: {self.directory} is new, it will be created")
            os.makedirs(self.directory, exist_ok=True)
        return open_allowed

    def _stop_writing(self):
        pass
        # if self.changed_since_opening:
        #     # if not self.check_storage():
        #     self._update_cluster_all()  
        if self.changed_since_opening:
            self.save_data_info_map()
        return super()._stop_writing()

    def process_unfinished(self):
        if super().process_unfinished():
            self.save_data_info_map() # TODO

    def format_base_name(self, data_i):
        return "{}".format(str(data_i).rjust(self.filllen, "0"))

    def format_path(self, data_i, subdir = "", appname = "", directory=None, **kwargs):
        '''
        format the path of data_i
        '''
        if appname and appname[0] != '_':
            appname = '_' + appname # add '_' before appname
        directory = self.directory if directory is None else directory
        return os.path.join(directory, subdir, 
                            "{}{}{}".format(
                                self.format_base_name(data_i), 
                                appname, 
                                self.suffix))

    @staticmethod
    def parse_path(directory, path:str):
        '''
        parse the path to get data_i, subdir, appname, it is the reverse operation of format_path
        '''
        subdir, file = os.path.split(os.path.relpath(path, directory))
        filename = os.path.splitext(file)[0]
        split_filename = filename.split('_')
        mainname = split_filename[0]
        try:
            appname  = "_".join(split_filename[1:])
        except IndexError:
            appname = ""
        data_i = int(mainname)
        return data_i, subdir, appname

    def parse_appnames_func(self, value):
        return [""], [value]
    
    def construct_value_func(self, appnames:list[str], values:list) -> VDCT:
        return values[0]

    def auto_path(self, data_i, _subdir = None, _appname=None, return_app = False, allow_mutil_appendname = True) -> Union[list[str], tuple[str, list[str]]]:
        '''
        auto find the path of data_i. \n
        * if data_i has multiple appnames, raise IndexError

        if return_app is True, return subdir, appname, else return path

        if allow_mutil_appendname is True, the type of appname will be list[str], else str; 
        and the type of path will be list[str], else str
        '''
        if data_i in self.data_info_map:
            subdir      = self.data_info_map[data_i].dir
            appnames    = self.data_info_map[data_i].appnames
        else:
            raise ClusterDataIOError(f'idx {data_i} not in {self.directory}')
        
        if _subdir is not None and subdir != _subdir:
            raise ClusterDataIOError(f"can't find {data_i} in {_subdir}")
        if _appname is not None:
            if _appname not in appnames:
                raise ClusterDataIOError(f"can't find {data_i} in {_subdir} with appname {_appname}")
            else:
                appnames = (_appname, )
        else:
            if not allow_mutil_appendname and len(appnames) > 1:
                raise ElementsAmbiguousError(f'idx {data_i} has more than one appendname: {appnames}, its path is ambiguous. ')

        if not return_app:
            path_list = []
            for n in appnames:
                path = self.format_path(data_i, subdir=subdir, appname=n)
                if not os.path.exists(path):
                    raise ValueError(f"can't find {data_i} in {subdir} with appname {n}")
                path_list.append(path)
            return path_list
        else:
            return subdir, appnames

    def check_storage(self):
        paths = glob.glob(os.path.join(self.directory, "**/*" + self.suffix), recursive=True)
        files = [int(os.path.split(path)[1][:self.filllen]) for path in paths]
        files_set = set(files)
        cur_keys_set = set(self.keys())        
        if len(files_set) != len(files):
            raise ValueError(f"there are duplicate files in {self.directory}")
        if len(files_set) != len(cur_keys_set):
            return False
        elif files_set == cur_keys_set:
            return True
        else:
            return False
    
    def clear_files_with_cache(self):
        for data_i, info in self.data_info_map.items():
            if info.has_cache:
                for appname in info.appnames:
                    if not info.has_file:
                        self.write(data_i, info.get_cache(appname), info.dir, appname)

    def __parse_cache_kw(self, **kwargs):
        assert all([isinstance(v, list) for v in kwargs.values()])
        assert all([len(v) == len(self) for v in kwargs.values()])
        kw_keys = list(kwargs.keys())
        kw_values = list(kwargs.values())
        return kw_keys, kw_values

    def cache_to_data_info_map(self, save = True, **kwargs):
        if self.is_closed():
            warnings.warn("can't cache to data_info_map, because it is close or readonly", ClusterNotRecommendWarning)
            return
        if not len(self.data_info_map) > 0:
            warnings.warn("len(self._data_info) == 0, nothing to save", ClusterNotRecommendWarning)
            return
        self.cache_priority = False
        kw_keys, kw_values = self.__parse_cache_kw(**kwargs)
        for data_i, data_info in tqdm(self.data_info_map.items(), desc="save cache for {}".format(self.directory)):
            for appname in data_info.appnames:
                if data_info.get_file_exist(appname):
                    read_kw = {k:v[data_i] for k, v in zip(kw_keys, kw_values)}
                    elem = self.read(data_i, **read_kw)
                    self._data_info_map[data_i].set_cache(elem)
        self.cache_priority = True    

        if save:
            self.save_data_info_map()

    def unzip_cache(self, force = False, **kwargs):
        '''
        unzip the cache to a dict
        '''
        assert os.path.exists(os.path.join(self.directory, self.DIR_MAP_FILE))
        kw_keys, kw_values = self.__parse_cache_kw(**kwargs)
        if len(self._data_info_map) == 0:
            return
        progress = tqdm(self.data_info_map.items(), 
                        desc="unzip cache for {}".format(self.directory),
                        total=len(self.data_info_map))
        for data_i, info in progress:
            write_kw = {k:v[data_i] for k, v in zip(kw_keys, kw_values)}
            ## TODO
            # for appname, cache in info.cache_items():
            #     if cache is None:
            #         continue
            #     self.write(data_i, cache, info.dir, appname, force=force, **write_kw)
  
    def save_data_info_map(self, **kwargs):
        self._data_info_map.save()

class SingleFile(Generic[VDCT]):
    def __init__(self, sub_path:str, read_func:Callable, write_func:Callable) -> None:
        self.sub_path = sub_path
        self.read_func:Callable = read_func
        self.write_func:Callable = write_func
        self.cluster:FileCluster = None

    @property
    def path(self):
        return os.path.join(self.cluster.directory, self.sub_path)

    @property
    def exist(self):
        return os.path.exists(self.path)

    def set_cluster(self, cluster:"FileCluster"):
        self.cluster = cluster

    def read(self) -> VDCT:
        return self.read_func(self.path)
    
    def write(self, data):
        self.write_func(self.path, data)

    @staticmethod
    def new_json_singlefile(sub_path:str):
        return SingleFile(sub_path, JsonIO.load_json, JsonIO.dump_json)
    
    @staticmethod
    def new_pickle_singlefile(sub_path:str):
        return SingleFile(sub_path, deserialize_object, serialize_object)
    
    @staticmethod
    def new_npy_singlefile(sub_path:str):
        return SingleFile(sub_path, partial(np.load, allow_pickle=True), partial(np.save, allow_pickle=True))
    
    @staticmethod
    def new_txt_singlefile(sub_path:str):
        def read_txt(path):
            with open(path, 'r') as f:
                return f.readlines()
            
        def write_txt(path, data):
            with open(path, 'w') as f:
                f.writelines(data)
        return SingleFile(sub_path, read_txt, write_txt)

class FileCluster(_DataCluster[DSNT, VDCT, "FileCluster"]):
    '''
    a cluster of multiple files, they may have different suffixes and i/o operations
    but they must be read/write together
    '''
    SingleFile = SingleFile


    class _read(_DataCluster._read["FileCluster", VDCT], Generic[VDCT]):
        def _call(self, data_i, **kwargs: Any) -> VDCT:
            file_path = self.cluster.filter_data_i(data_i)
            return self.cluster.fileobjs_dict[file_path].read()

    class _write(_DataCluster._write["FileCluster", VDCT], Generic[VDCT]):
        def is_overwriting(self, data_i, value, subdir = "", **kwargs):
            return False

        def _call(self, data_i, value:VDCT):
            file_path = self.cluster.filter_data_i(data_i)
            self.cluster.fileobjs_dict[file_path].write(value)

    class _remove(_DataCluster._remove["FileCluster", VDCT], Generic[VDCT]):
        def _call(self, data_i, **kwargs: Any):
            path = self.cluster.filter_data_i(data_i)
            self.cluster.remove_file(data_i)
            os.remove(path)

    class _move(_DataCluster._move["FileCluster", VDCT], Generic[VDCT]):
        def _call(self, key, value, **kwargs: Any):
            raise NotImplementedError("FileCluster doesn't support move operation")

        def __call__(self, src:int, dst:int, *args, force=False, **kwargs):
            raise NotImplementedError("FileCluster doesn't support move operation")

    class _load_in(_DataCluster._load_in["FileCluster", VDCT], Generic[VDCT]):
        def check_key(self, src:str, **kwargs) -> bool:
            rlt = isinstance(src, str)
            rlt = rlt and super().check_key(src, **kwargs)

        def check_value(self, dst:int, **kwargs) -> bool:
            rlt = isinstance(dst, int) and dst >= 0
            rlt = rlt and super().check_value(dst, **kwargs)
            return rlt
        
        def _call(self, src_path:str, dst:int, **kwargs: Any):
            dst_path = self.cluster.fileobjs_dict[self.cluster.filter_data_i(dst)].path
            shutil.copy(src_path, dst_path)
            
    def __init__(self, dataset_node: DSNT, sub_dir, register = True, name = "", singlefile_list:list[SingleFile] = []) -> None:
        super().__init__(dataset_node, sub_dir, register, name, singlefile_list)
        os.makedirs(self.directory, exist_ok=True)

    @property
    def all_exist(self):
        for f in self.fileobjs_dict.values():
            if not f.exist:
                return False
        return True

    def __len__(self):
        super().__len__()
        return len(self.fileobjs_dict)

    def keys(self):
        super().keys()
        return list(self.fileobjs_dict.keys())
    
    def values(self) -> list[SingleFile[VDCT]]:
        super().values()
        return list(self.fileobjs_dict.values())
    
    def items(self):
        super().items()
        return self.keys(), self.values()

    def get_file_path(self, data_i, *args, **kwargs):
        return self.fileobjs_dict[data_i].path

    def _make_continuous(self, *args, **kwargs):
        pass

    def copyto(self, dst: str, cover = False, *args, **kwargs):
        os.makedirs(dst, exist_ok=True)
        for f in self.fileobjs_dict.values():
            dst_path = os.path.join(dst, f.sub_path)
            if os.path.exists(dst_path):
                if cover:
                    os.remove(dst_path)
                else:
                    raise FileExistsError(f"{dst_path} already exists, please set cover=True")
            shutil.copy(f.path, dst_path)

    # def _merge_one(self, src: "FileCluster", key:str, merge_funcs=[], *args, **kwargs):
    #     ### TODO
    #     if key in self.fileobjs_dict:
    #         this_i = list(self.fileobjs_dict.keys()).index(key)
    #         func = merge_funcs[this_i]
    #         src_data = src.read(key)
    #         this_data = self.read(this_i)
    #         new_data = func(this_data, src_data)
    #         self.write(this_i, new_data)
    #     else:
    #         new_dir = os.path.join(self.directory, self.sub_dir)
    #         new_path = os.path.join(new_dir, src.fileobjs_dict[key].sub_path)
    #         os.makedirs(new_dir, exist_ok=True)
    #         shutil.copy(src.fileobjs_dict[key].path, new_path)
    #         self.update_file(
    #             SingleFile(
    #                 src.fileobjs_dict[key].sub_path, 
    #                 src.fileobjs_dict[key].read_func, 
    #                 src.fileobjs_dict[key].write_func))
            
    #         self.log_to_mark_file(key, self.LOG_APPEND)

    def merge_from(self, src: "FileCluster", merge_funcs:Union[list, Callable]=[], *args, **kwargs):
        raise NotImplementedError("FileCluster doesn't support merge operation")

    def _init_attr(self, singlefile:list[SingleFile], *args, **kwargs):
        super()._init_attr(singlefile, **kwargs)
        self.fileobjs_dict:dict[str, SingleFile] = {}

        for f in singlefile:
            self.update_file(f)

    def filter_data_i(self, data_i, return_index = False):
        def process_func(key, string):
            return key == get_mainname(string)
        target_int = int_str_cocvt(self.fileobjs_dict, data_i, return_index = True, process_func = process_func)
        if not return_index:
            return list(self.fileobjs_dict.keys())[target_int]
        else:
            return target_int

    def read_all(self):
        return [f.read() for f in self.values()]

    def write_all(self, values):
        assert len(values) == len(self), f"the length of value must be {len(self)}"
        for f, d in zip(self.values(), values):
            f.write(d)

    def remove_file(self, idx:Union[int, str]):
        path = self.filter_data_i(idx)
        self.fileobjs_dict.pop(path)

    def update_file(self, singlefile:SingleFile):
        singlefile.set_cluster(self)
        self.fileobjs_dict[singlefile.path] = singlefile

    def get_file(self, idx:Union[int, str]):
        return self.fileobjs_dict[self.filter_data_i(idx)]

    def paths(self):
        return list(self.keys())

class IntArrayDictElement(Elements[DSNT, dict[int, np.ndarray]]):
    def __init__(self, dataset_node: DSNT, sub_dir:str, array_shape:tuple[int], array_fmt:str = "", register=True, name = "", filllen=6, fillchar='0') -> None:
        super().__init__(dataset_node, sub_dir, register, name, self._read_func, self._write_func, ".txt", filllen, fillchar)
        self.array_shape:tuple[int] = array_shape
        self.array_fmt = array_fmt if array_fmt else "%.4f"
    
    def _to_dict(self, array:np.ndarray)->dict[int, np.ndarray]:
        '''
        array: np.ndarray [N, 5]
        '''
        dict_ = {}
        for i in range(array.shape[0]):
            dict_[int(array[i, 0])] = array[i, 1:].reshape(self.array_shape)
        return dict_

    def _from_dict(self, dict_:dict[int, np.ndarray]):
        '''
        dict_: dict[int, np.ndarray]
        '''
        array = []
        for i, (k, v) in enumerate(dict_.items()):
            array.append(
                np.concatenate([np.array([k]).astype(v.dtype), v.reshape(-1)])
                )
        array = np.stack(array)
        return array

    def _read_format(self, array:np.ndarray, **kwargs):
        return array

    def _write_format(self, array:np.ndarray, **kwargs):
        return array

    def _read_func(self, path, **kwargs):
        raw_array = np.loadtxt(path, dtype=np.float32)
        if len(raw_array.shape) == 1:
            raw_array = np.expand_dims(raw_array, 0)
        raw_array = self._read_format(raw_array, **kwargs)
        intarraydict = self._to_dict(raw_array)
        return intarraydict

    def _write_func(self, path, intarraydict:dict[int, np.ndarray], **kwargs):
        raw_array = self._from_dict(intarraydict)
        raw_array = self._write_format(raw_array, **kwargs)
        np.savetxt(path, raw_array, fmt=self.array_fmt, delimiter='\t')

class _ClusterMap(dict[str, DCT]):
    def __init__(self, dataset_node:"DatasetNode", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_node = dataset_node

    def __set_update(self):
        self.dataset_node.updated = True
        if self.dataset_node.inited:
            self.dataset_node.update_dataset()

    # def values(self) -> dict_values[str, DCT]:
    #     return super().values()
    
    # def items(self) -> dict_items[str, DCT]:
    #     return super().items()

    def __setitem__(self, __key: Any, __value: Any) -> None:
        self.__set_update()
        return super().__setitem__(__key, __value)
    
    def update(self, __m, **kwargs: Any) -> None:
        self.__set_update()
        return super().update(__m, **kwargs)

    def setdefault(self, __key: Any, __default: Any = ...) -> Any:
        self.__set_update()
        return super().setdefault(__key, __default)
    
    # def __getitem__(self, __key: Any) -> Any:
    #     return search_in_dict(self, __key, process_func=self._search_func)
    #     # return super().__getitem__(__key)

    def search(self, __key: Any, return_index = False):
        return search_in_dict(self, __key, process_func=self._search_func) ### TODO
    
    def add_cluster(self, cluster:DCT):
        cluster.dataset_node = self.dataset_node
        self[cluster.identity_string()] = cluster
        self.dataset_node.data_overview_table.add_column(cluster.key_identity_string(), exist_ok=True)

    def remove_cluster(self, cluster:DCT):
        if cluster.dataset_node == self.dataset_node:
            del self[cluster.identity_string()]
            self.dataset_node.data_overview_table.remove_column(cluster.key_identity_string(), not_exist_ok=True)

    def get_keywords(self):
        keywords = []
        for cluster in self.values():
            keywords.append(cluster.key_identity_string())
        return keywords

    @staticmethod
    def _search_func(indetity_string:str):
        _,_,_,sub_dir, name = _DataCluster.parse_identity_string(indetity_string)
        if name != "":
            return name
        else:
            return sub_dir

class DatasetNode(InstanceRegistry, WriteController, ABC, Generic[DCT, VDST]):
    '''
    DatasetNode, only gather the clusters. 
    have no i/o operations
    '''
    WRITING_MARK = ".dfsw" # the mark file of writing process, dataset format streamly writing
    
    def __init__(self, directory, parent:"DatasetNode" = None) -> None:

        InstanceRegistry.__init__(self)
        WriteController.__init__(self)
        ABC.__init__(self)

        self.__inited = False # if the dataset has been inited   
        self.__init_node(parent)
        self.init_dataset_attr_hook()
        self.__init_clusters()
        self.__inited = True # if the dataset has been inited

    ### init ###
    def __init_node(self, parent):
        self.parent:DatasetNode = parent
        self.children:list[DatasetNode] = []
        self.move_node(parent)

    def init_dataset_attr_hook(self):
        def is_directory_inside(base_dir, target_dir):
            base_dir:str = os.path.abspath(base_dir)
            target_dir:str = os.path.abspath(target_dir)
            return target_dir.startswith(base_dir)
        
        os.makedirs(self.directory, exist_ok=True)
        self._data_num = 0
        self._data_i_upper = 0    
        self._updated = False
        self._unfinished_operation = 0
        if self.parent is not None:
            assert is_directory_inside(self.parent.directory, self.directory), f"{self.directory} is not inside {self.parent.directory}"
            self.sub_dir:str = os.path.relpath(self.directory, self.parent.directory)
        else:
            self.sub_dir:str = self.directory
        self.data_overview_table = Table[int, str, bool](default_value_type=bool, row_name_type=int, col_name_type=str)
        self.cluster_map = _ClusterMap[DCT](self)
        

    def __init_clusters(self):
        unfinished = self.mark_exist()
        if unfinished:
            y:int = _DataCluster.choose_unfinished_operation(self)
            self._unfinished_operation = y        

        self._init_clusters()
        self.update_dataset()        

        self.load_overview()        
        if unfinished:
            self.process_unfinished()
            os.remove(self.get_writing_mark_file())

    ############

    ### node ###
    def add_child(self, child_node:"DatasetNode"):
        assert isinstance(child_node, DatasetNode), f"child_node must be Node, not {type(child_node)}"
        child_node.parent = self
        self.children.append(child_node)

    def remove_child(self, child_node:"DatasetNode"):
        assert isinstance(child_node, DatasetNode), f"child_node must be Node, not {type(child_node)}"
        if child_node in self.children:
            child_node.parent = None
            self.children.remove(child_node)

    def move_node(self, new_parent:"DatasetNode"):
        assert isinstance(new_parent, DatasetNode) or new_parent is None, f"new_parent must be DatasetNode, not {type(new_parent)}"
        if self.parent is not None:
            self.parent.remove_child(self)
        if new_parent is not None:
            new_parent.add_child(self)
    ############

    ### implement InstanceRegistry
    def _init_identity_paramenter(self, directory, *args, **kwargs):
        self.directory:str = os.path.normpath(directory)

    def identity_string(self):
        return self.gen_identity_string(self.directory)

    @staticmethod
    def parse_identity_string(identity_string:str):
        return identity_string.split(':')

    @classmethod
    def gen_identity_string(cls, directory, *args, **kwargs):
        return f"{cls.__name__}:{directory}"
    ###############################

    ### cluster ###
    def _init_clusters(self):
        pass    

    @property
    def parent_directory(self):
        if self.parent is None:
            return ""
        else:
            return self.parent.directory

    @property
    def inited(self):
        return self.__inited

    @property
    def updated(self):
        return self._updated
    
    @updated.setter
    def updated(self, value:bool):
        self._updated = bool(value)

    @property
    def clusters(self) -> list[DCT]:
        clusters = list(self.cluster_map.values())
        return clusters

    @property
    def data_clusters(self) -> list[DCT]:
        clusters = [x for x in self.clusters if not isinstance(x, FileCluster)]
        return clusters

    @property
    def opened_clusters(self):
        clusters = [x for x in self.clusters if not x.is_closed()]
        return clusters

    @property
    def opened_data_clusters(self):
        clusters = [x for x in self.data_clusters if not x.is_closed()]
        return clusters

    @property
    def jsondict_map(self):
        '''
        select the key-value pair whose value is JsonDict
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, JsonDict)}

    @property
    def elements_map(self):
        '''
        select the key-value pair whose value is Elements
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, Elements)}
    
    @property
    def filecluster_map(self):
        '''
        select the key-value pair whose value is FileCluster
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, FileCluster)}

    def save_elements_data_info_map(self):
        for elem in self.elements_map.values():
            elem.save_data_info_map()

    def set_elements_cache_priority(self, mode:bool):
        for elem in self.elements_map.values():
            elem.cache_priority = bool(mode)
        
    def close_all(self, value = True):
        for obj in self.clusters:
            obj.close() if value else obj.open()

    def open_all(self, value = True):
        self.close_all(not value)

    def set_all_readonly(self, value = True):
        for obj in self.clusters:
            obj.set_readonly(value)

    def set_all_writable(self, value = True):
        self.set_all_readonly(not value)

    def set_all_overwrite_allowed(self, value = True):
        for obj in self.clusters:
            obj.set_overwrite_allowed(value)

    def get_element_paths_of_one(self, data_i:int):
        '''
        brief
        -----
        get all paths of a data
        '''
        paths = {}
        for elem in self.elements_map.values():
            paths[elem.sub_dir] = elem.auto_path(data_i, allow_mutil_appendname=False)[0]
        return paths
    
    def get_all_clusters(self, _type:Union[type, tuple[type]] = None, only_opened = False) -> _ClusterMap[DCT]:
        cluster_map = _ClusterMap[_DataCluster](self)
        cluster_map.update(self.cluster_map)

        for k, v in list(cluster_map.items()):
            if _type is not None:
                if not isinstance(v, _type):
                    cluster_map.pop(k)
            if only_opened:
                if v.is_closed():
                    cluster_map.pop(k)

        for child in self.children:
            cluster_map.update(child.get_all_clusters(_type, only_opened))

        return cluster_map

    def copyto(self, dst:str, asroot = True, cover = False):
        if not asroot:
            progress = tqdm(self.opened_clusters)
        else:
            progress = tqdm(self.get_all_clusters(only_opened=True).values())

        for c in progress:
            progress.set_postfix({'copying': "{}:{}".format(c.__class__.__name__, c.directory)})            
            c.copyto(os.path.join(dst, c.sub_dir), cover=cover)
    ##########

    ### IO ###
    @property
    def data_num(self):
        return self._data_num
    
    @property
    def data_i_upper(self):
        return self._data_i_upper
        # return max([x.cluster_data_i_upper for x in self.opened_clusters])

    @abstractmethod
    def read_one(self, data_i, **kwargs) -> VDST:
        pass

    def _write_jsondict(self, data_i, data):
        pass

    def _write_elements(self, data_i, data, subdir="", appname=""):
        pass

    def _write_files(self, data_i, data):
        pass

    def _update_dataset(self, data_i = None):
        nums = [len(x) for x in self.opened_data_clusters]
        num = np.unique(nums)
        if len(num) > 1:
            raise ValueError("Unknown error, the numbers of different datas are not equal")
        elif len(num) == 1:
            self._data_num = int(num)
        else:
            self._data_num = 0
        try:
            self._data_i_upper = max([x.cluster_data_i_upper for x in self.opened_data_clusters])
        except ValueError:
            self._data_i_upper = 0
        if self._data_i_upper != self.data_num:
            warnings.warn(f"the data_i_upper of dataset:{self.directory} is not equal to the data_num, \
                          it means the the data_i is not continuous, this may cause some errors", ClusterParaWarning)
        if data_i is not None:
            self.calc_overview(data_i)

    def write_one(self, data_i, data:VDST, **kwargs):
        assert self.is_writing or all([not x.write_streamly for x in self.jsondict_map.values()]), \
            "write_one cannot be used when any jsondict's stream_dumping_json is True. \
                considering use write_to_disk instead"
        self._write_jsondict(data_i, data)
        self._write_elements(data_i, data)
        self._write_files(data_i, data)

        self.update_dataset(data_i)

        # self.log_to_mark_file(self.LOG_APPEND, data_i) ### TODO

    def remove_one(self, data_i, *args, **kwargs):
        for cluster in self.opened_clusters:
            cluster.remove(data_i, not_exist_ok=True, *args, **kwargs)
        self.update_dataset(data_i)

    def update_dataset(self, data_i = None, f = False):
        if self.updated or f:
            self._update_dataset(data_i)
            self.updated = False

    def process_unfinished(self):
        pass

    def read_from_disk(self, with_data_i = False):
        '''
        brief
        ----
        *generator
        Since the amount of data may be large, return one by one
        '''
        if not with_data_i:
            for i in self.data_overview_table.row_names:
                yield self.read_one(i)
        else:
            for i in self.data_overview_table.row_names:
                yield i, self.read_one(i)

    def write_to_disk(self, data:VDST, data_i = -1):
        '''
        brief
        -----
        write elements immediately, write basejsoninfo to cache, 
        they will be dumped when exiting the context of self.writer
        
        NOTE
        -----
        For DatasetFormat, the write mode has only 'append'. 
        If you need to modify, please call 'DatasetFormat.clear' to clear all data, and then write again.
        '''
        if not self.is_writing:
            print("please call 'self.start_writing' first, here is the usage example:")
            print(extract_doc(DatasetFormat.__doc__, "example"))
            raise ValueError("please call 'self.start_writing' first")
        if data_i == -1:
            # self._updata_data_num()        
            data_i = self.data_i_upper
        
        self.write_one(data_i, data)

    def start_writing(self, overwrite_allowed = False):
        if super().start_writing(overwrite_allowed):
            print(f"start to write to {self.directory}")
            self.close_all(False)
            self.set_all_readonly(False)
            self.set_all_overwrite_allowed(overwrite_allowed)
        else:
            return False

    def stop_writing(self):
        if not self.is_writing:
            return False
        else:
            for jd in self.jsondict_map.values():
                jd.stop_writing()              
            self.set_all_overwrite_allowed(False)
            # self.close_all()
            self.save_overview()
            super().stop_writing()
            return True

    def clear(self, ignore_warning = False, force = False):
        '''
        brief
        -----
        clear all data, defalut to ask before executing
        '''
        if not ignore_warning:
            y = input("All files in {} will be removed, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            if force:
                self.close_all(False)
                self.set_all_readonly(False)
                cluster_to_clear = self.clusters
            else:
                cluster_to_clear = self.opened_clusters

            for cluster in cluster_to_clear:
                cluster.clear(ignore_warning=True)

        self.set_all_readonly(True)

    def values(self):
        return self.read_from_disk()
    
    def keys(self):
        for i in self.data_overview_table.row_names:
            yield i

    def items(self):
        return self.read_from_disk(True)

    def __setattr__(self, name, value):
        ### 同名变量赋值时，自动将原有对象解除注册
        if name in self.__dict__:
            obj = self.__getattribute__(name)
            if isinstance(obj, DatasetNode):
                assert isinstance(value, DatasetNode), f"the type of {name} must be DatasetNode, not {type(value)}"
                if obj.parent == self:
                    obj.move_node(None)
            elif isinstance(obj, _DataCluster):
                assert isinstance(value, _DataCluster), f"the type of {name} must be _DataCluster, not {type(value)}"
                if obj.dataset_node == self:
                    obj.unregister_from_dataset()
        super().__setattr__(name, value)

    def __getitem__(self, data_i:Union[int, slice]) -> Union[Generator[VDST, Any, None], VDST]:
        if isinstance(data_i, slice):
            # 处理切片操作
            start, stop, step = data_i.start, data_i.stop, data_i.step
            if start is None:
                start = 0
            if step is None:
                step = 1
            stop = min(stop, self.data_i_upper)
            def g():
                for i in range(start, stop, step):
                    yield self.read_one(i)
            return g()
        elif isinstance(data_i, int):
            # 处理单个索引操作
            return self.read_one(data_i)
        else:
            raise TypeError("Unsupported data_i type")

    def __setitem__(self, data_i:int, value):
        self.write_one(data_i, value)

    def __len__(self):
        return self.data_num

    def __iter__(self):
        return self.read_from_disk()
    
    def load_overview(self):
        overview_path = os.path.join(self.directory, self.__class__.__name__ + "-" + DatasetFormat.OVERVIEW)
        if os.path.exists(overview_path):
            self.data_overview_table = Table[int, str, bool].from_json(overview_path)
        else:
            self.init_overview()
            self.save_overview()
        return self.data_overview_table

    def init_overview(self):
        if len(self.data_clusters) > 0:
            rows = [x for x in range(self.data_i_upper)]
            cols = [x.key_identity_string() for x in self.data_clusters]
            self.data_overview_table = Table(rows, cols, bool, row_name_type=int, col_name_type=str)
            for data_i in tqdm(self.data_overview_table.data, desc="initializing data frame"):
                self.calc_overview(data_i)

    def save_overview(self):
        if not self.data_overview_table.empty:
            overview_path = os.path.join(self.directory, self.__class__.__name__ + "-" + DatasetFormat.OVERVIEW)
            self.data_overview_table.save(overview_path)

    def calc_overview(self, data_i):
        self.data_overview_table.add_row(data_i, exist_ok=True)        
        for cluster in self.data_clusters:
            self.data_overview_table[data_i, cluster.key_identity_string()] = data_i in cluster.keys()

    def clear_invalid_data_i(self):
        raise NotImplementedError

    def get_writing_mark_file(self):
        return os.path.join(self.directory, self.WRITING_MARK)
    ##########

class DatasetFormat(DatasetNode[DCT, VDST]):
    """
    # Dataset Format
    -----
    A dataset manager, support mutiple data types.
    It is useful if you have a series of different data. 
    For example, a sample contains an image and a set of bounding boxes. 
    There is a one-to-one correspondence between them, and there are multiple sets of such data.

    properties
    ----
    * inited : bool, if the dataset has been inited
    * updated : bool, if the dataset has been updated, it can bes set.
    * directory : str, the root directory of the dataset
    * unfinished : bool, the last writing process of the dataset has not been completed
    * clusters : list[_DataCluster], all clusters of the dataset
    * opened_clusters : list[_DataCluster], all opened clusters of the dataset
    * jsondict_map : dict[str, JsonDict], the map of jsondict
    * elements_map : dict[str, Elements], the map of elements
    * files_map : dict[str, FileCluster], the map of fileclusters
    * data_num : int, the number of data in the dataset
    * data_i_upper : int, the max index of the iterator

    virtual function
    ----
    * read_one: Read one piece of data

    recommended to rewrite
    ----
    * _init_clusters: init the clusters
    * _write_jsondict: write one piece of data to jsondict
    * _write_elementss: write one piece of data to elements
    * _write_files: write one piece of data to files
    * _update_dataset: update the dataset, it should be called when the dataset is updated

    not necessary to rewrite
    ----
    * update_dataset: update the dataset, it should be called when the dataset is updated
    * read_from_disk: read all data from disk as a generator
    * write_to_disk: write one piece of data to disk
    * start_writing: start writing
    * stop_writing: stop writing    
    * clear: clear all data of the dataset
    * close_all: close all clusters
    * open_all : open all clusters
    * set_all_readonly: set all file streams to read only
    * set_all_writable: set all file streams to writable
    * get_element_paths_of_one: get the paths of one piece of data
    * __getitem__: get one piece of data
    * __setitem__: set one piece of data
    * __iter__: return the iterator of the dataset

    example
    -----
    * read

    df1 = DatasetFormat(directory1) 

    df2 = DatasetFormat(directory2) 

    for data in self.read_from_disk(): 
        ...

    * write_to_disk : 1

    df2.write_to_disk(data) × this is wrong

    with df2.writer:  # use context manager
    
        df2.write_to_disk(data)

    df2.clear()

    * write_to_disk : 2

    df2.start_writing()

    df2.write_to_disk(data)

    df2.stop_writing()
    
    * write_one

    df2.write_one(data_i, data)
    '''

    """
    OVERVIEW = "overview.json"
    DEFAULT_SPLIT_TYPE = ["default"]
    SPLIT_DIR = "ImageSets"

    KW_TRAIN = "train"
    KW_VAL = "val" 

    def __init__(self, directory, split_rate = 0.75, parent = None) -> None:
        super().__init__(directory, parent)

        self.split_default_rate = split_rate

    def init_dataset_attr_hook(self):
        super().init_dataset_attr_hook()
        self.spliter_group = SpliterGroup(os.path.join(self.directory, self.SPLIT_DIR), 
                                    self.DEFAULT_SPLIT_TYPE,
                                    self)

    @property
    def train_idx_array(self):
        return self.spliter_group.cur_training_spliter.get_idx_list(self.KW_TRAIN)
    
    @property
    def val_idx_array(self):
        return self.spliter_group.cur_training_spliter.get_idx_list(self.KW_VAL)

    @property
    def default_train_idx_array(self):
        return self.spliter_group[0].get_idx_list(self.KW_TRAIN, 0)
    
    @property
    def default_val_idx_array(self):
        return self.spliter_group[0].get_idx_list(self.KW_VAL, 0)

    def process_unfinished(self):
        logs = self.load_from_mark_file()
        if self._unfinished_operation == 2:
            ### process spliter
            for log, idx, value in logs:
                if log == self.LOG_APPEND:
                    self.spliter_group.remove_one(idx)
                else:
                    raise ValueError("cannot rollback")
            self.spliter_group.save()

    def stop_writing(self):
        if not self.is_writing:
            return False
        else:
            self.spliter_group.save()
            return super().stop_writing()

#### Posture ####
class PostureDatasetFormat(DatasetFormat[_DataCluster, ViewMeta]):
    def _init_clusters(self):
        self.labels_elements     = IntArrayDictElement(self, "labels", (4,), array_fmt="%8.8f")
        self.bbox_3ds_elements   = IntArrayDictElement(self, "bbox_3ds", (-1, 2), array_fmt="%8.8f") 
        self.landmarks_elements  = IntArrayDictElement(self, "landmarks", (-1, 2), array_fmt="%8.8f")
        self.extr_vecs_elements  = IntArrayDictElement(self, "trans_vecs", (2, 3), array_fmt="%8.8f")

    def read_one(self, data_i, **kwargs):
        labels_dict:dict[int, np.ndarray] = self.labels_elements.read(data_i)
        extr_vecs_dict:dict[int, np.ndarray] = self.extr_vecs_elements.read(data_i)
        bbox_3d_dict:dict[int, np.ndarray] = self.bbox_3ds_elements.read(data_i)
        landmarks_dict:dict[int, np.ndarray] = self.landmarks_elements.read(data_i)
        return ViewMeta(color=None,
                        depth=None,
                        masks=None,
                        extr_vecs = extr_vecs_dict,
                        intr=None,
                        depth_scale=None,
                        bbox_3d = bbox_3d_dict, 
                        landmarks = landmarks_dict,
                        visib_fracts=None,
                        labels=labels_dict)

    def _write_elements(self, data_i: int, viewmeta: ViewMeta, subdir="", appname=""):
        self.labels_elements.write(data_i, viewmeta.labels, subdir=subdir, appname=appname)
        self.bbox_3ds_elements.write(data_i, viewmeta.bbox_3d, subdir=subdir, appname=appname)
        self.landmarks_elements.write(data_i, viewmeta.landmarks, subdir=subdir, appname=appname)
        self.extr_vecs_elements.write(data_i, viewmeta.extr_vecs, subdir=subdir, appname=appname)

    def calc_by_base(self, mesh_dict:dict[int, MeshMeta], overwitre = False):
        '''
        brief
        -----
        calculate data by base data, see ViewMeta.calc_by_base
        '''
        with self.writer:
            self.set_overwrite_allowed(True)
            for i in range(self.data_num):
                viewmeta = self.read_one(i)
                viewmeta.calc_by_base(mesh_dict, overwitre=overwitre)
                self.write_to_disk(viewmeta, i)
            self.set_overwrite_allowed(False)

class LinemodFormat(PostureDatasetFormat):
    KW_CAM_K = "cam_K"
    KW_CAM_DS = "depth_scale"
    KW_CAM_VL = "view_level"
    KW_GT_R = "cam_R_m2c"
    KW_GT_t = "cam_t_m2c"
    KW_GT_ID = "obj_id"
    KW_GT_INFO_BBOX_OBJ = "bbox_obj"
    KW_GT_INFO_BBOX_VIS = "bbox_visib"
    KW_GT_INFO_PX_COUNT_ALL = "px_count_all"
    KW_GT_INFO_PX_COUNT_VLD = "px_count_valid"
    KW_GT_INFO_PX_COUNT_VIS = "px_count_visib" 
    KW_GT_INFO_VISIB_FRACT = "visib_fract"

    RGB_DIR = "rgb"
    DEPTH_DIR = "depth"
    MASK_DIR = "mask"
    GT_FILE = "scene_gt.json"
    GT_CAM_FILE = "scene_camera.json"
    GT_INFO_FILE = "scene_gt_info.json"
    
    class _MasksElements(Elements["LinemodFormat", dict[int, np.ndarray]]):
        def id_format(self, class_id):
            id_format = str(class_id).rjust(6, "0")
            return id_format

        def _read(self, data_i, **kwargs) -> dict[int, np.ndarray]:
            masks = {}
            subdir = ""
            for n, scene_gt in enumerate(self.dataset_node.scene_gt_dict[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask:np.ndarray = super()._read(data_i)
                if mask is None:
                    continue
                masks[id_] = mask
            return masks

        def _write(self, data_i, value: dict[int, ndarray], subdir="", appname="", **kwargs):
            for n, scene_gt in enumerate(self.dataset_node.scene_gt_dict[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask = value[id_]
                super().write(data_i, mask, appname=self.id_format(n))
            super()._write(data_i, value, subdir, appname, **kwargs)
            return self.LOG_APPEND

    def _init_clusters(self):
        super()._init_clusters()
        self.rgb_elements   = Elements(self,      self.RGB_DIR,
                                       read_func=cv2.imread,                                    
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.depth_elements = Elements(self,      self.DEPTH_DIR,    
                                       read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH),   
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.masks_elements = self._MasksElements(self, self.MASK_DIR,     
                                                  read_func=lambda x:cv2.imread(x, cv2.IMREAD_GRAYSCALE),  
                                                  write_func=cv2.imwrite, 
                                                  suffix='.png')      

        self.scene_gt_dict              = JsonDict(self, self.GT_FILE)        
        self.scene_camera_dict          = JsonDict(self, self.GT_CAM_FILE)
        self.scene_gt_info_dict         = JsonDict(self, self.GT_INFO_FILE)

    def _write_elements(self, data_i:int, viewmeta:ViewMeta, subdir="", appname=""):
        super()._write_elements(data_i, viewmeta)
        self.rgb_elements.  write(data_i, viewmeta.color, subdir=subdir, appname=appname)
        self.depth_elements.write(data_i, viewmeta.depth, subdir=subdir, appname=appname)
        self.masks_elements.write(data_i, viewmeta.masks, subdir=subdir, appname=appname)

    def _write_jsondict(self, data_i:int, viewmeta:ViewMeta):
        super()._write_jsondict(data_i, viewmeta)
        gt_one_info = []
        for obj_id, trans_vecs in viewmeta.extr_vecs.items():
            posture = Posture(rvec=trans_vecs[0], tvec=trans_vecs[1])
            gt_one_info .append(
                {   LinemodFormat.KW_GT_R: posture.rmat.reshape(-1),
                    LinemodFormat.KW_GT_t: posture.tvec.reshape(-1),
                    LinemodFormat.KW_GT_ID: int(obj_id)})
        self.scene_gt_dict.write(self.data_num, gt_one_info)

        ###
        gt_cam_one_info = {self.KW_CAM_K: viewmeta.intr.reshape(-1), self.KW_CAM_DS: viewmeta.depth_scale, self.KW_CAM_VL: 1}
        self.scene_camera_dict.write(self.data_num, gt_cam_one_info)

        ### eg:
        # "0": 
        # [{"bbox_obj": [274, 188, 99, 106], 
        # "bbox_visib": [274, 188, 99, 106], 
        # "px_count_all": 7067, 
        # "px_count_valid": 7067, 
        # "px_count_visib": 7067, 
        # "visib_fract": 1.0}],
        gt_info_one_info = []
        bbox_2d = viewmeta.bbox_2d
        for obj_id in viewmeta.masks.keys():
            mask = viewmeta.masks[obj_id]
            bb = bbox_2d[obj_id]
            vf = viewmeta.visib_fracts[obj_id]
            mask_count = int(np.sum(mask))
            mask_visib_count = int(mask_count * vf)
            gt_info_one_info.append({
                self.KW_GT_INFO_BBOX_OBJ: bb,
                self.KW_GT_INFO_BBOX_VIS: bb,
                self.KW_GT_INFO_PX_COUNT_ALL: mask_count, 
                self.KW_GT_INFO_PX_COUNT_VLD: mask_count, 
                self.KW_GT_INFO_PX_COUNT_VIS: mask_visib_count,
                self.KW_GT_INFO_VISIB_FRACT: vf
            })
        self.scene_gt_info_dict.write(self.data_num, gt_info_one_info)         

    def read_one(self, data_i, **kwargs):
        super().read_one(data_i, **kwargs)
        color     = self.rgb_elements.read(data_i)
        depth   = self.depth_elements.read(data_i)
        masks   = self.masks_elements.read(data_i)
        bbox_3d = self.bbox_3ds_elements.read(data_i)
        landmarks = self.landmarks_elements.read(data_i)
        intr           = self.scene_camera_dict[data_i][LinemodFormat.KW_CAM_K].reshape(3, 3)
        depth_scale    = self.scene_camera_dict[data_i][LinemodFormat.KW_CAM_DS]

        ids = [x[LinemodFormat.KW_GT_ID] for x in self.scene_gt_dict[data_i]]
        postures = [Posture(rmat =x[LinemodFormat.KW_GT_R], tvec=x[LinemodFormat.KW_GT_t]) for x in self.scene_gt_dict[data_i]]
        extr_vecs = [np.array([x.rvec, x.tvec]) for x in postures]
        extr_vecs_dict = as_dict(ids, extr_vecs)
        # visib_fracts    = [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in self.scene_gt_info_dict[data_i]]
        visib_fracts_dict = zip_dict(ids, self.scene_gt_info_dict[data_i], 
                                         lambda obj: [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in obj])
        return ViewMeta(color, depth, masks, 
                        extr_vecs_dict,
                        intr,
                        depth_scale,
                        bbox_3d,
                        landmarks,
                        visib_fracts_dict)

class VocFormat(PostureDatasetFormat):
    IMGAE_DIR = "images"

    class cxcywhLabelElement(IntArrayDictElement):
        def __init__(self, dataset_node: Any, sub_dir: str, array_fmt: str = "", register=True, name="", filllen=6, fillchar='0') -> None:
            self.skip = False
            self.__trigger = False

            self.default_image_size = None
            super().__init__(dataset_node, sub_dir, (4,), array_fmt, register, name, filllen, fillchar)

        def skip_once(self):
            self.skip = True
            self.__trigger = True

        def _reset_trigger(self):
            if self.__trigger:
                self.__trigger = False
                self.skip = False

        def _read_format(self, labels: np.ndarray, image_size = None):
            if image_size is not None:
                bbox_2d = labels[:,1:].astype(np.float32) #[cx, cy, w, h]
                bbox_2d = VocFormat._normedcxcywh_2_x1y1x2y2(bbox_2d, image_size)
                labels[:,1:] = bbox_2d   
            return labels         
        
        def _write_format(self, labels: np.ndarray, image_size = None):
            labels = labels.astype(np.float32)
            if image_size is not None:
                bbox_2d = labels[:,1:].astype(np.float32) #[cx, cy, w, h]
                bbox_2d = VocFormat._x1y1x2y2_2_normedcxcywh(bbox_2d, image_size)
                labels[:,1:] = bbox_2d
            return labels

        class _read(IntArrayDictElement._read):
            def __init__(self, cluster) -> None:
                super().__init__(cluster)
                self.cluster: VocFormat.cxcywhLabelElement = cluster

            def _call(self, data_i, *, image_size = None, **kwargs: Any):
                if self.cluster.skip:
                    self.cluster._reset_trigger()
                    raise ClusterDataIOError("skip")
                if image_size is None:
                    image_size = self.cluster.default_image_size
                if image_size is None:
                    warnings.warn("image_size is None, bbox_2d will not be converted from normed cxcywh to x1x2y1y2",
                                    ClusterParaWarning)
                return super()._call(data_i, image_size = image_size, **kwargs)
            
            def __call__(self, data_i, *args, force=False, image_size = None, **kwargs: Any):
                return super().__call__(data_i, *args, force=force, image_size = image_size, **kwargs)

        class _write(IntArrayDictElement._write):
            def __init__(self, cluster) -> None:
                super().__init__(cluster)
                self.cluster: VocFormat.cxcywhLabelElement = cluster

            def _call(self, data_i, labels_dict:dict[int, np.ndarray], subdir="", *,
                      image_size = None, **kwargs):
                if self.cluster.skip:
                    self.cluster._reset_trigger()
                    raise ClusterDataIOError("skip")
                if image_size is None:
                    image_size = self.cluster.default_image_size
                if image_size is None:
                    warnings.warn("image_size is None, bbox_2d will not be converted from x1x2y1y2 to normed cxcywh",
                                    ClusterParaWarning)
                return super()._call(data_i, labels_dict, subdir, image_size = image_size, **kwargs)
            
            def __call__(self, data_i, labels_dict:dict[int, np.ndarray], subdir="", *args, 
                         force = False, image_size = None, **kwargs):
                return super().__call__(data_i, labels_dict, subdir, *args, 
                                        force=force, image_size = image_size, **kwargs)

    def __init__(self, directory, split_rate = 0.25, parent = None) -> None:
        super().__init__(directory, split_rate, parent)

    def _init_clusters(self):
        super()._init_clusters()
        self.images_elements:Elements[VocFormat, np.ndarray]     = Elements(self, self.IMGAE_DIR,    
                                            read_func=cv2.imread, 
                                            write_func=cv2.imwrite,
                                            suffix = ".jpg")
        self.depth_elements      = Elements(self, "depths",       
                                            read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH), 
                                            write_func=cv2.imwrite, 
                                            suffix = '.png')
        self.masks_elements      = Elements(self, "masks",        
                                            read_func=lambda x: deserialize_image_container(deserialize_object(x), cv2.IMREAD_GRAYSCALE),
                                            write_func=lambda path, x: serialize_object(path, serialize_image_container(x)),  
                                            suffix = ".pkl")
        self.intr_elements       = Elements[VocFormat, np.ndarray](self, "intr",
                                            read_func=loadtxt_func((3,3)), 
                                            write_func=savetxt_func("%8.8f"), 
                                            suffix = ".txt")
        self.depth_scale_elements         = Elements(self, "depth_scale",
                                            read_func=lambda path: float(loadtxt_func((1,))(path)), 
                                            write_func=savetxt_func("%8.8f"), 
                                            suffix = ".txt")
        self.visib_fracts_elements= IntArrayDictElement(self, "visib_fracts", ())
        self.labels_elements     = self.cxcywhLabelElement(self, "labels", )

        self.labels_elements.default_image_size = (640, 480)

    def get_default_set(self, data_i):
        if data_i in self.default_train_idx_array:
            sub_set = VocFormat.KW_TRAIN
        elif data_i in self.default_val_idx_array:
            sub_set = VocFormat.KW_VAL
        else:
            raise ValueError("can't find datas of index: {}".format(data_i))
        return sub_set
    
    def decide_default_set(self, data_i):
        try:
            return self.get_default_set(data_i)
        except ValueError:
            spliter = self.spliter_group[self.DEFAULT_SPLIT_TYPE[0]]
            return spliter.set_one_by_rate(data_i, self.split_default_rate)

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

        # lt = bbox_2d[..., :2]
        # rb = bbox_2d[..., 2:]

        # cx, cy = (lt + rb) / 2
        # w, h = rb - lt
        # # 归一化
        # cy, h = np.array([cy, h]) / img_size[0]
        # cx, w = np.array([cx, w]) / img_size[1]
        # return np.array([cx, cy, w, h])

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

    def _write_elements(self, data_i: int, viewmeta: ViewMeta, subdir="", appname=""):
        sub_set = self.decide_default_set(data_i)
        self.labels_elements.skip_once()
        super()._write_elements(data_i, viewmeta, subdir=sub_set)
        #
        self.images_elements.write(data_i, viewmeta.color, subdir=sub_set)
        #
        self.depth_elements.write(data_i, viewmeta.depth, subdir=sub_set)
        #
        self.masks_elements.write(data_i, viewmeta.masks, subdir=sub_set)
        
        ###
        self.labels_elements.write(data_i, viewmeta.bbox_2d, subdir=sub_set, image_size = viewmeta.color.shape[:2][::-1]) # necessary to set image_size
        # labels = []
        # for id_, mask in viewmeta.masks.items():
        #     img_size = mask.shape
        #     point = np.array(np.where(mask))
        #     if point.size == 0:
        #         continue
        #     bbox_2d = viewmeta_bbox2d[id_]
        #     cx, cy, w, h = self._x1y1x2y2_2_normedcxcywh(bbox_2d, img_size)
        #     labels.append([id_, cx, cy, w, h])

        # self.labels_elements.write(data_i, labels, subdir=sub_set)

        self.intr_elements.write(data_i, viewmeta.intr, subdir=sub_set)
        self.depth_scale_elements.write(data_i, np.array([viewmeta.depth_scale]), subdir=sub_set)
        self.visib_fracts_elements.write(data_i, viewmeta.visib_fracts, subdir=sub_set)
    
    def read_one(self, data_i, **kwargs) -> ViewMeta:
        # 判断data_i属于train或者val
        self.labels_elements.skip_once()
        viewmeta = super().read_one(data_i, **kwargs)
        # 读取
        color:np.ndarray = self.images_elements.read(data_i)
        viewmeta.set_element(ViewMeta.COLOR, color)
        #
        depth = self.depth_elements.read(data_i)
        viewmeta.set_element(ViewMeta.DEPTH, depth)
        #
        labels_dict = self.labels_elements.read(data_i, image_size = color.shape[:2][::-1]) # {id: [cx, cy, w, h]}
        viewmeta.set_element(ViewMeta.LABELS, labels_dict)
        #
        masks_dict = self.masks_elements.read(data_i)
        viewmeta.set_element(ViewMeta.MASKS, masks_dict)
        #
        intr = self.intr_elements.read(data_i)
        viewmeta.set_element(ViewMeta.INTR, intr)
        #
        ds    = self.depth_scale_elements.read(data_i)
        viewmeta.set_element(ViewMeta.DEPTH_SCALE, ds)
        #
        visib_fracts_dict = self.visib_fracts_elements.read(data_i)
        viewmeta.set_element(ViewMeta.VISIB_FRACTS, visib_fracts_dict)

        return viewmeta

    def save_elements_data_info_map(self):
        if self.labels_elements.default_image_size is not None:
            kw = {}
        else:
            kw = {"image_size" : [img.shape[:2][::-1] for img in self.images_elements]}
        self.labels_elements.cache_to_data_info_map(True, **kw)
        for c in [self.bbox_3ds_elements, 
                  self.depth_scale_elements,
                  self.intr_elements, 
                  self.landmarks_elements, 
                  self.extr_vecs_elements,
                  self.visib_fracts_elements]:
            c:Elements = c
            c.cache_to_data_info_map(True)

    def synchronize_default_split(self):
        self.spliter_group.cur_spliter_name = self.DEFAULT_SPLIT_TYPE[0]
        for data_i, info in self.labels_elements.data_info_map.items():
            subdir = info.dir
            self.spliter_group.cur_training_spliter.set_one(data_i, subdir)


class _LinemodFormat_sub1(LinemodFormat):
    class _MasksElements(LinemodFormat._MasksElements):

        def _read(self, data_i, **kwargs) -> dict[int, ndarray]:
            masks = {}
            for n in range(100):
                mask = super().read(data_i, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[n] = mask
            return masks
        
        def _write(self, data_i, value: dict[int, ndarray], subdir="", appname="", **kwargs):
            for id_, mask in value.items():
                super().write(data_i, mask, appname=self.id_format(id_))
            super()._write(data_i, value, subdir, appname, **kwargs)
            return self.LOG_APPEND

    def __init__(self, directory, clear = False) -> None:
        super().__init__(directory, clear)
        self.rgb_elements   = Elements(self, "rgb", 
                                       read_func=cv2.imread,  
                                       write_func=cv2.imwrite, 
                                       suffix='.jpg')

    def read_one(self, data_i, **kwargs):
        viewmeta = super().read_one(data_i, **kwargs)

        for k in viewmeta.bbox_3d:
            viewmeta.bbox_3d[k] = viewmeta.bbox_3d[k][:, ::-1]
        for k in viewmeta.landmarks:
            viewmeta.landmarks[k] = viewmeta.landmarks[k][:, ::-1]
        viewmeta.depth_scale *= 1000

        return viewmeta
  
class Mix_VocFormat(VocFormat):
    DEFAULT_SPLIT_TYPE = ["default", "posture", "reality", "basis"]

    REALITY_SUBSETS = ["real", "sim"]
    BASIS_SUBSETS = ["basic", "augment"]

    MODE_DETECTION = 0
    MODE_POSTURE = 1

    IMGAE_DIR = "images"

    def __init__(self, directory, split_rate=0.75, parent = None) -> None:
        super().__init__(directory, split_rate, parent)

        self.spliter_name = self.DEFAULT_SPLIT_TYPE[1]
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[0]].split_for = Spliter.SPLIT_FOR_TRAINING
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]].split_for = Spliter.SPLIT_FOR_TRAINING
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[2]].split_for = Spliter.SPLIT_FOR_DATATYPE
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[3]].split_for = Spliter.SPLIT_FOR_DATATYPE

        self.spliter_group[self.DEFAULT_SPLIT_TYPE[2]].set_default_subsets(self.REALITY_SUBSETS)
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[3]].set_default_subsets(self.BASIS_SUBSETS)

    @property
    def posture_train_idx_list(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]].get_idx_list(self.KW_TRAIN)

    @property
    def posture_val_idx_list(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]].get_idx_list(self.KW_VAL)

    @property
    def default_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[0]]
    
    @property
    def posture_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]]
    
    @property
    def reality_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[2]]
    
    @property
    def basis_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[3]]

    def _init_clusters(self):
        super()._init_clusters()

    def gen_posture_log(self, ratio = 0.15, source:list[int] = None):
        """
        Only take ratio of the real data as the verification set
        """
        # assert self.reality_spliter.total_num == self.data_num, "reality_spliter.total_num != data_num"
        # assert self.basis_spliter.total_num == self.data_num, "basis_spliter.total_num != data_num"
        
        real_idx = self.reality_spliter.get_idx_list(0).copy()
        real_base_idx = list(set(real_idx).intersection(self.basis_spliter.get_idx_list(0)))
        sim_idx  = self.reality_spliter.get_idx_list(1).copy()

        if source is None:
            pass
        else:
            real_idx = list(set(real_idx).intersection(source))
            real_base_idx = list(set(real_base_idx).intersection(source))
            sim_idx  = list(set(sim_idx).intersection(source))

        train_real, val_real = Spliter.gen_split(real_base_idx, ratio, 2)

        self.posture_val_idx_list.clear()
        self.posture_val_idx_list.extend(val_real)

        posture_train_idx_list = np.setdiff1d(
            np.union1d(real_idx, sim_idx),
            self.posture_val_idx_list
            ).astype(np.int32).tolist()
        self.posture_train_idx_list.clear()
        self.posture_train_idx_list.extend(posture_train_idx_list)

        self.posture_val_idx_list.sort()
        self.posture_train_idx_list.sort()

        with self.posture_spliter.writer:
            self.posture_spliter.save()
            
    def record_data_type(self, data_i, is_real, is_basic):
        reality = "real" if is_real else "sim"
        basis = "basic" if is_basic else "augment"
        self.reality_spliter.set_one(data_i, reality)
        self.basis_spliter.set_one(data_i, basis)

    # def get_data_type_as_bool(self, data_i):
    #     types = self.get_data_type(data_i)
    #     for k, v in types.items():
    #         types[k] = (v == self.spliter_group[k].subsets[0])
    #     return types

class Spliter(FileCluster["SpliterGroup", list[int]]):
    SPLIT_FOR_TRAINING = 0
    SPLIT_FOR_DATATYPE = 1

    SPLIT_MODE_BASE = "base"

    SPLIT_FOR_FILE = "__split_for.txt"
    SPLIT_FILE = "split.json"

    class SpliterTable(Table[str, str, list[int]]):
        def __init__(self, spliter:"Spliter"):
            super().__init__()
            self.add_column(Spliter.SPLIT_MODE_BASE)
            for row in spliter.default_subsets:
                self.add_row(row)
            self.spliter = spliter
        
        def gen_default_value(self) -> list[int]:
            return []

        def get_base_filter(self):
            return self.get_column(Spliter.SPLIT_MODE_BASE)
        
        def qurey(self, data_i:int, return_key:bool = False):
            if return_key:
                result_keys:list[tuple[str]] = []
                for row_name, col_name, value in self.tranverse(with_key=True):
                    if data_i in value:
                        result_keys.append((row_name, col_name))
                return result_keys
            else:
                result_table = Table[str, str, bool](self.row_names, self.col_names, bool)
                for row_name, col_name, value in self.tranverse(with_key=True):
                    result_table[row_name, col_name] = data_i in value
                return result_table
            
        def __str__(self) -> str:
            tile = "\n" + self.spliter.spliter_name + ":\n"
            string = super().__str__()
            # match self.spliter.split_mode and add '>' '<' to it
            string = string.replace(' ' + self.spliter.split_mode + ' ', f">{self.spliter.split_mode}<")
            return tile + string

    def __init__(self, dataset_node:"SpliterGroup", sub_dir:str, split_for=None, default_subsets:list = [], register = True, name = "") -> None:
        self.split_for_file = SingleFile[tuple[int, list[str]]](
            self.SPLIT_FOR_FILE, 
            self.load_split_for_info_func, 
            self.save_split_for_info_func)
        self.split_file = SingleFile[dict[str, dict[str, list[int]]]](
            self.SPLIT_FILE,
            partial(JsonIO.load_json, cvt_list_to_array=False),
            partial(JsonIO.dump_json, regard_list_as_array = True))
        super().__init__(dataset_node, sub_dir, register, name, singlefile_list=[self.split_for_file, self.split_file])
        ### init split_for
        if split_for is None:
            if self.split_for_file.exist:
                self.split_for = self.split_for_file.read()
            else:
                self.split_for = self.SPLIT_FOR_TRAINING
                self.split_for_file.write(self.split_for)

        ### init default_subsets
        if self.split_for == self.SPLIT_FOR_TRAINING:
            default_subsets = [VocFormat.KW_TRAIN, VocFormat.KW_VAL]
        if self.split_file.exist:
            default_subsets += list(self.split_file.read().keys())
        
        self.__default_subsets:list[str] = remove_duplicates(default_subsets)

        self.split_table = self.SpliterTable(self)
        
        self.__split_mode = self.SPLIT_MODE_BASE
        self.__exclusive = True
        self.load()

    @property
    def data(self):
        return self.split_table.data

    @property
    def active_split(self):
        return self.get_split(self.split_mode)

    @property 
    def spliter_name(self):
        return os.path.split(self.sub_dir)[-1]

    @property
    def split_mode(self):
        return self.__split_mode
    
    @split_mode.setter
    def split_mode(self, mode:Union[int,str]):
        self.set_split_mode(mode)

    @property
    def exclusive(self):
        return self.__exclusive
    
    @exclusive.setter
    def exclusive(self, value):
        self.set_exclusive(value)

    @property
    def default_subsets(self):
        return self.__default_subsets

    @property
    def subsets(self):
        return tuple(self.split_table.row_names)
    
    @property
    def split_mode_list(self):
        return tuple(self.split_table.col_names)
    
    @property
    def subset_fileobjs_dict(self) -> dict[str, SingleFile[list[int]]]:
        objs_dict = {}
        for k in self.fileobjs_dict.keys():
            if self.SPLIT_FOR_FILE in k:
                continue
            k = get_mainname(k)
            objs_dict.update({k: self.fileobjs_dict[self.filter_data_i(k)]})
        return objs_dict

    @property
    def total_num(self):
        return sum([len(v) for v in self.split_table.get_base_filter().values()])

    def set_split_mode(self, split_mode:int):
        split_mode = self.split_table._col_name_filter(split_mode)
        assert split_mode in self.split_table.col_names, "split_mode must be in {}".format(self.split_table.col_names)
        self.__split_mode = str(split_mode)

    def set_exclusive(self, value:bool):
        if self.split_for == self.SPLIT_FOR_TRAINING:
            self.__exclusive = True
        else:
            self.__exclusive = bool(value)

    def set_default_subsets(self, value:Union[str, Iterable[str]]):
        if isinstance(value, str):
            value = [value]
        if isinstance(value, Iterable):
            self.__default_subsets = tuple(remove_duplicates(value))
        for t in value:
            self.split_table.add_row(t, exist_ok=True)
        for t in self.split_table.row_names:
            if t not in value:
                self.split_table.remove_row(t)
        self.split_table.resort_row(self.__default_subsets)
        self.load() 

    def add_split_mode(self, mode:str, exist_ok:bool = False):
        self.split_table.add_column(mode, exist_ok=exist_ok)

    def remove_split_mode(self, mode:str, not_exist_ok:bool = False):
        self.split_table.remove_column(mode, not_exist_ok=not_exist_ok)

    def get_split(self, split_mode):
        return self.split_table.get_column(split_mode)

    def get_idx_list(self, subset:Union[str, int], split_mode = None) -> list[int]:
        if split_mode is None:
            split_mode = self.split_mode
        return self.split_table[subset, split_mode]
    
    def read_idx_list(self, subset:Union[str, int], split_mode = None) -> tuple[int]:
        return tuple(self.get_idx_list(subset, split_mode))

    def clear_idx(self):
        for item in self.split_table.tranverse():
            item.clear()

    def load_split_for_info_func(self, path:str):
        with open(path, 'r') as file:
            lines = file.readlines()

        # parse split_for
        try:
            split_for = int(lines[0].strip())
        except ValueError:
            return None
        return split_for

    def save_split_for_info_func(self, path:str, value:int):
        with open(path, 'w') as file:
            file.write(str(value))

    def loadsplittxt_func(self, path:str):
        if os.path.exists(path):
            with warnings.catch_warnings():
                rlt = np.loadtxt(path).astype(np.int32).reshape(-1).tolist()
            return rlt
        else:
            return []
    
    def load(self):
        # load split_for_
        self.split_for = self.split_for_file.read()
        # load split
        if self.split_file.exist:
            split = self.split_file.read()
        else:
            split = {}
        self.split_table.update(split)

    def save(self):
        with self.writer.allow_overwriting():
            os.makedirs(self.directory, exist_ok=True)
            split_data = dict(self.data)
            self.split_file.write(self.data)
            self.split_for_file.write(self.split_for)

    @staticmethod
    def process_split_rate(split_rate:Union[float, Iterable[float], dict[str, float]], 
                            split_num:Union[int, list[str]]):
        if isinstance(split_num, int):
            subsets = None
        elif isinstance(split_num, Iterable):
            subsets = split_num
            split_num = len(subsets)
        else:
            raise ValueError("split_num must be int or Iterable")
        
        assert split_num > 1, "len(subsets) must > 1"
        
        if split_num == 2 and isinstance(split_rate, float):
            split_rate = (split_rate, 1 - split_rate)
        if subsets is not None and isinstance(split_rate, dict):
            split_rate = tuple([split_rate[subset] for subset in subsets])
        elif isinstance(split_rate, Iterable):
            split_rate = tuple(split_rate)
        else:
            raise ValueError("split_rate must be Iterable or dict[str, float], (or float if len(subsets) == 2)")
        assert len(split_rate) == split_num, "splite_rate must have {} elements".format(split_num)
        
        return split_rate

    @staticmethod
    def gen_split(data_num:Union[int, Iterable], 
                  split_rate:Union[float, Iterable[float], dict[str, float]],
                  split_num:int):
        assert isinstance(data_num, (int, Iterable)), "data_num must be int"
        assert isinstance(split_num, int), "split_num must be int"

        split_rate = Spliter.process_split_rate(split_rate, split_num)

        if isinstance(data_num, int):
            _s = np.arange(data_num, dtype=np.int32)
        else:
            _s = np.array(data_num, dtype=np.int32)
            data_num = len(_s)
        np.random.shuffle(_s)

        _s:list[int] = _s.tolist()
        rlt:list[list[int]] = []
        for i, rate in enumerate(split_rate):
            if i != split_num - 1:
                num = int(data_num * rate)
                seq = _s[:num]
                _s = _s[num:]
            else:
                seq = _s
            seq.sort()
            rlt.append(seq)
        return tuple(rlt)
            
    def set_one(self, data_i, subset, split_mode=None, sort = False):
        if split_mode is None:
            split_mode = self.split_mode
        if data_i not in self.split_table[subset, split_mode]:
            self.split_table[subset, split_mode].append(data_i)
        if self.exclusive:
            # remove from other subsets
            for _subset in self.subsets:
                if _subset != subset and data_i in self.split_table[_subset, split_mode]:
                    self.split_table[_subset, split_mode].remove(data_i)
        if sort:
            self.split_table[subset, split_mode].sort()

    def set_one_by_rate(self, data_i, split_rate, split_mode=None):
        if split_mode is None:
            split_mode = self.split_mode
        split_rate = self.process_split_rate(split_rate, self.subsets)
        total_nums = [len(st) for st in self.split_table.get_column(0).values()]
        if sum(total_nums) == 0:
            # all empty, choose the first
            subset_idx = 0
        else:
            rates = np.array(total_nums) / sum(total_nums)
            subset_idx = 0
            for idx, r in enumerate(rates):
                if r <= split_rate[idx]:
                    subset_idx = idx
                    break
        self.set_one(data_i, self.subsets[subset_idx], split_mode)
        return self.subsets[subset_idx]

    def remove_one(self, idx):
        for item in self.split_table:
            try:
                item.remove(idx)
            except ValueError:
                pass

    def sort_all(self):
        for value in self.split_table.tranverse():
            value.sort()

    def get_one_subset(self, data_i):
        return self.split_table.qurey(data_i)
    
    def print(self):
        print(self.split_table)

    def __getitem__(self, key):
        return self.split_table[key]

class SpliterGroup(DatasetNode[Spliter, VDST]):
    DEFAULT_SPLIT_TYPE = ["default"]

    def __init__(self, directory, spliter_name_list:list, parent = None) -> None:
        self.__spliter_name_list = spliter_name_list
        super().__init__(directory, parent)
        self.__spliter_name:str = self.DEFAULT_SPLIT_TYPE[0]
        self.cluster_map:_ClusterMap[Spliter] = self.cluster_map

    def _init_clusters(self):
        self.__spliter_name_list = remove_duplicates(self.__spliter_name_list + os.listdir(self.split_subdir))
        _spliter_list:list = [Spliter(self, m, register=True) for m in self.__spliter_name_list]
        return super()._init_clusters()

    ###
    def read_one(self, data_i, **kwargs) -> VDST:
        raise NotImplementedError("read_one should not be called in SpliterGroup")
    ###

    @property
    def split_subdir(self):
        return self.sub_dir
    
    @property
    def spliter_name_list(self):
        # mode_list = []
        # for key, spliter in self.cluster_map.items():
        #     _, _, _, sub_dir, name = spliter.parse_identity_string(key)
        #     mode_list.append(get_mainname(path))
        # return tuple(mode_list)
        return tuple(self.cluster_map.get_keywords())

    @property
    def cur_spliter_name(self):
        return self.__spliter_name
    
    @cur_spliter_name.setter
    def cur_spliter_name(self, spliter_name:str):
        self.set_cur_spliter_name(spliter_name)
    
    @property
    def cur_training_spliter(self):
        spliter = self.cluster_map.search(self.cur_spliter_name)
        assert spliter.split_for == Spliter.SPLIT_FOR_TRAINING, "cur_training_spliter must be Spliter.SPLIT_FOR_TRAINING"
        return spliter

    def set_cur_spliter_name(self, spliter_name:str):
        assert spliter_name in self.spliter_name_list, "spliter_name must be one of {}".format(self.spliter_name_list)
        self.__spliter_name = spliter_name
        
    def query_set(self, data_i:int, split_for:int = None) -> dict[str, Table[str, str, bool]]:
        set_ = {}
        for s in self.cluster_map.values():
            if split_for is None or s.split_for == split_for:
                set_.update({s.spliter_name: s.get_one_subset(data_i)})
        return set_

    def query_datatype_set(self, data_i:int) -> dict[str, Table[str, str, bool]]:
        return self._query(data_i, Spliter.SPLIT_FOR_DATATYPE)

    def query_training_set(self, data_i) -> dict[str, Table[str, str, bool]]:
        return self._query(data_i, Spliter.SPLIT_FOR_TRAINING)

    def record_set(self, data_i:int, set_:dict[str, Table[str, str, bool]]):
        for spliter_name, table in set_.items():
            for row, col, v in table.tranverse(with_key=True):
                if v:
                    spliter = self.cluster_map.search(spliter_name)
                    spliter.add_split_mode(col, exist_ok=True)
                    spliter.set_one(data_i, row, col)

    def save(self):
        for spliter in self.cluster_map.values():
            spliter.save()
    
    def remove_one(self, idx):
        for spliter in self.clusters:
            spliter.remove_one(idx)

    def __getitem__(self, key):
        return self.cluster_map.search(key)



def serialize_object(file_path, obj:dict):
    # if os.path.splitext(file_path)[1] == '.pkl':
    #     file_path = os.path.splitext(file_path)[0] + ".npz"
    # np.savez(file_path, **obj)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

# 从文件反序列化对象
def deserialize_object(serialized_file_path):
    with open(serialized_file_path, 'rb') as file:
        elements = pickle.load(file)
        return elements

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def zip_dict(ids:list[int], item:Union[Iterable, None, Any], func = lambda x: x):
    if item:
        processed = func(item)
        return as_dict(ids, processed)
    else:
        return None
    
def get_mainname(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def dataset_test(ds:DatasetNode, dst_ds:DatasetNode):
    # read
    v = ds[0]
    # write
    dst_ds.write_one(0, v)


def _cvt_old_element_map(dataset_node:DatasetNode, sub_dir:str, suffix:str):
    path = os.path.join(dataset_node.directory, sub_dir, Elements.DATA_INFO_MAP_FILE)
    if not os.path.exists(path):
        return
    _map:dict[int, dict] = deserialize_object(path)
    for k, v in _map.items():
        if not isinstance(v["cache"], list):
            continue
        _map[k]["cache"] = _map[k]["cache"][0]
    serialize_object(path, _map)