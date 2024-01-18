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
import sys

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Generator, Optional, List
from functools import partial
import functools
import copy
import importlib
import inspect
import time
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor

from . import Posture, JsonIO, JSONDecodeError, Table, BinDict, extract_doc, search_in_dict, int_str_cocvt,\
      serialize_object, deserialize_object, test_pickleable, read_file_as_str, write_str_to_file
from .mesh_manager import MeshMeta


DEBUG = False

NODE = TypeVar('NODE', bound="Node")

IOSM = TypeVar('IOSM', bound="IOStatusManager") # type of the IO status manager
RGSITEM = TypeVar('RGSITEM')
DMT  = TypeVar('DMT',  bound="DataMapping") # type of the cluster
DSNT = TypeVar('DSNT', bound='DatasetNode') # dataset node type
FCT = TypeVar('FCT', bound="FilesCluster") # type of the files cluster
FHT = TypeVar('FHT', bound="FilesHandle") # type of the files handle
VDMT = TypeVar('VDMT') # type of the value of data cluster
VDST = TypeVar('VDST') # type of the value of dataset
from numpy import ndarray

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
T = TypeVar('T')
T_MUITLSTR = TypeVar('T_MUITLSTR', str, List[str])

# region Errors and Warnings ###
class AmbiguousError(ValueError):
    pass

class IOMetaParameterError(ValueError):
    pass

class IOMetaPriorityError(ValueError):
    pass

class KeyNotFoundError(KeyError):
    pass

class ClusterDataIOError(RuntimeError):
    pass

class ClusterWarning(Warning):
    pass

class DataMapExistError(OSError):
    pass

class IOStatusWarning(Warning):
    pass

class ClusterIONotExecutedWarning(ClusterWarning):
    pass

class ClusterNotRecommendWarning(ClusterWarning):
    pass
# endregion Errors and Warnings ###

# region functions ###
def method_exit_hook_decorator(cls, func:Callable, exit_hook_func, enter_condition_func = None):
    """
    Decorator function that adds an exit hook to a method based on an enter condition.

    Parameters
    ----------
    cls : class
        The class to which the method belongs.
    func : callable
        The method to be decorated.
    exit_hook_func : callable
        The exit hook function to be called after the method execution.
    enter_condition_func : callable, optional
        The enter condition function to determine if the method should be executed. Defaults to None.

    Returns
    -------
    callable
        The decorated method.
    """
    enter_condition_func = (lambda x : True) if enter_condition_func is None else enter_condition_func
    def wrapper(self, *args, **kw):
        if enter_condition_func(self):
            func(self, *args, **kw)
            if cls == self.__class__:
                exit_hook_func(self)
                if DEBUG:
                    print(f"{self} runs {func.__name__}")
        else:
            if DEBUG:
                print(f"{self} skips {func.__name__}")
    return wrapper

def parse_kw(**kwargs) -> list[dict[str, Any]]:
    """
    Parses keyword arguments into a list of dictionaries.

    Parameters
    ----------
    **kwargs : dict[str, Any]
        The keyword arguments to be parsed.

    Returns
    -------
    list[dict[str, Any]]
        The parsed keyword arguments.
    """
    if len(kwargs) == 0:
        return []
    assert all([isinstance(v, list) for v in kwargs.values()])
    length = len(list(kwargs.values())[0])
    assert all([len(v) == length for v in kwargs.values()])
    kw_keys = list(kwargs.keys())
    kw_values = list(kwargs.values())
    
    kws = []
    for data_i in range(length):
        kws.append({k:v[data_i] for k, v in zip(kw_keys, kw_values)})

    return kws

def is_subpath(child_path, parent_path):
    """
    Check if a path is a subpath of another path.

    Parameters
    ----------
    child_path : str
        The path to check if it is a subpath.
    parent_path : str
        The parent path to compare against.

    Returns
    -------
    bool
        True if the child_path is a subpath of the parent_path, False otherwise.
    """
    relative_path:str = os.path.relpath(child_path, parent_path)
    return not relative_path.startswith('..')

def get_with_priority(*args:Optional[T]) -> T:
    """
    Returns the first non-None argument from the given arguments.

    Parameters
    ----------
    *args : Optional[T]
        The arguments to check for non-None values.

    Returns
    -------
    T
        The first non-None argument, or None if all arguments are None.
    """
    assert len(args) > 0, "at least one parameter is needed"
    for arg in args:
        if arg is not None:
            return arg
    return None # type: ignore
        
def get_func_name(obj, func):
    """
    Get the name of a function within an object.

    Parameters
    ----------
    obj : object
        The object containing the function.
    func : function
        The function to get the name of.

    Returns
    -------
    str or None
        The name of the function if found, None otherwise.
    """
    func_name = func.__name__
    if hasattr(obj, func_name):
        return func_name
    else:
        for name in dir(obj):
            if hasattr(obj, name) and getattr(obj, name) == func:
                return name
        return None
            
def get_function_args(function, *exclude:str):
    """
    Get the arguments of a function, excluding specified arguments.

    Parameters
    ----------
    function : function
        The function to get the arguments of.
    *exclude : str
        The arguments to exclude.

    Returns
    -------
    rlt
        The signature of the function, excluding specified arguments.
    """
    signature = inspect.signature(function)
    rlt:list[str] = []
    ok_kind = [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_ONLY]
    for param in signature.parameters.values():
        if param.kind in ok_kind and param.name not in exclude:
            rlt.append(param.name)
    return rlt
# endregion functions ###

class IOStatusManager():
    """
    Manages the status and operations related to input/output (IO) operations.

    Attributes
    -----
    WRITING_MARK: str
        The file extension used to mark a writing operation.
    LOG_READ: int
        Log type for read operation.
    LOG_ADD: int
        Log type for add operation.
    LOG_REMOVE: int
        Log type for remove operation.
    LOG_CHANGE: int
        Log type for change operation.
    LOG_MOVE: int
        Log type for move operation.
    LOG_OPERATION: int
        Log type for generic operation.
    LOG_KN: list
        List of valid log types.
    _DEBUG: bool
        Flag to enable debug mode.
    __closed: bool
        Flag indicating if the IOStatusManager object is closed.
    __readonly: bool
        Flag indicating if the IOStatusManager object is read-only.
    __wait_writing: bool
        Flag indicating if the IOStatusManager object is waiting for writing or a writing process is executing.
    __overwrite_allowed: bool
        Flag indicating if overwriting is allowed.
    __writer: _Writer
        The writer object for performing IO operations.
    """

    WRITING_MARK = '.writing'

    LOG_READ = 0
    LOG_ADD = 1
    LOG_REMOVE = 2
    LOG_CHANGE = 3
    LOG_MOVE   = 4
    LOG_OPERATION = 5
    LOG_KN = [LOG_READ, LOG_ADD, LOG_REMOVE, LOG_CHANGE, LOG_MOVE, LOG_OPERATION]

    _DEBUG = False

    def __init__(self) -> None:
        """
        Initializes the IOStatusManager object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.__closed = True
        self.__readonly = True
        self.__wait_writing = True
        self.__overwrite_allowed = False

        self.__writer = self._Writer(self)

    # region context and writer ###
    class _IOContext():
        """
        Context manager for managing the IO context.

        Attributes
        ----------
        DEFAULT_INPUT_OPEN : bool
            Default value for input open flag.
        DEFAULT_INPUT_WRITABLE : bool
            Default value for input writable flag.
        DEFAULT_INPUT_OVERWRITE_ALLOWED : bool
            Default value for input overwrite allowed flag.

        Methods
        -------
        __init__ : 
            Initializes the _IOContext object.
        reset_input : 
            Resets the input flags to their default values.
        set_input : 
            Sets the input flags to the specified values.
        enter_hook : 
            Hook method called when entering the context.
        exit_hook : 
            Hook method called when exiting the context.
        """

        DEFAULT_INPUT_OPEN                  = False
        DEFAULT_INPUT_WRITABLE              = False
        DEFAULT_INPUT_OVERWRITE_ALLOWED     = False

        def __init__(self, obj:"IOStatusManager") -> None:
            """
            Initializes the _IOContext object.

            Parameters
            ----------
            obj : IOStatusManager
                The IOStatusManager object associated with this context.
            """
            self.obj:IOStatusManager = obj
            
            self.orig_closed:bool               = True
            self.orig_readonly:bool             = True
            self.orig_overwrite_allowed:bool    = False

            self.reset_input()

            self.working = False

        def reset_input(self) -> None:
            """
            Resets the input flags to their default values.
            """ 
            self.input_open                 = self.DEFAULT_INPUT_OPEN                  
            self.input_writable             = self.DEFAULT_INPUT_WRITABLE              
            self.input_overwrite_allowed    = self.DEFAULT_INPUT_OVERWRITE_ALLOWED     

        def set_input(self, open = False, writable = False, overwrite_allowed = False):
            """
            Sets the input flags to the specified values.

            Parameters
            ----------
            open : bool
                Flag indicating if the input should be opened.
            writable : bool
                Flag indicating if the input should be writable.
            overwrite_allowed : bool
                Flag indicating if overwriting is allowed.

            Returns
            -------
            _IOContext
                The _IOContext object.
            """
            self.input_open                 = open
            self.input_writable             = writable
            self.input_overwrite_allowed    = overwrite_allowed
            return self

        def enter_hook(self) -> None:
            """
            Hook method called when entering the context.
            """
            self.orig_closed                = self.obj.closed
            self.orig_readonly              = self.obj.readonly
            self.orig_overwrite_allowed     = self.obj.overwrite_allowed

            self.obj.open(self.input_open)
            self.obj.set_writable(self.input_writable)
            if self.input_writable:
                self.obj.start_writing()
            self.obj.set_overwrite_allowed(self.input_overwrite_allowed)

        def exit_hook(self):
            """
            Hook method called when exiting the context.

            Returns
            -------
            bool
                Indicates successful exit from the context.
            """
            self.obj.set_overwrite_allowed(self.orig_overwrite_allowed)
            if self.obj.is_writing:
                self.obj.stop_writing()
            self.obj.set_readonly(self.orig_readonly)
            self.obj.close(self.orig_closed)
            return True   

        def __enter__(self):
            """
            Called when entering the context.

            Returns
            -------
            _IOContext
                The _IOContext object.
            """
            if self.working:
                raise RuntimeError(f"the IOContext of {self.obj.identity_string()} is already working")
            
            self.working = True
            if IOStatusManager._DEBUG:
                print(f"enter:\t{self.obj.identity_string()}")
            
            self.enter_hook()

            self.reset_input() # reset

            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            """
            Called when exiting the context.

            Parameters
            ----------
            exc_type : type
                The type of exception that caused the context to be exited, if any.
            exc_value : Exception
                The instance of the exception that caused the context to be exited, if any.
            traceback : traceback
                A traceback object encapsulating the call stack at the point where the exception was raised, if any.

            Returns
            -------
            bool
                Indicates successful exit from the context.
            """
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                rlt = False
                if IOStatusManager._DEBUG:
                    print(f"exit:\t{self.obj.identity_string()}")
                rlt = self.exit_hook()
                self.working = False
                return rlt

    class _Writer(_IOContext):
        """
        Context manager for managing the writer context.

        Attributes
        ----------
        DEFAULT_INPUT_OPEN : bool
            Default value for input open flag.
        DEFAULT_INPUT_WRITABLE : bool
            Default value for input writable flag.

        Methods
        -------
        allow_overwriting(self, overwrite_allowed:bool=True) -> "_Writer"
            Sets the input overwrite allowed flag.
        """

        DEFAULT_INPUT_OPEN                  = True
        DEFAULT_INPUT_WRITABLE              = True

        def allow_overwriting(self, overwrite_allowed = True):
            """
            Sets the input overwrite allowed flag.

            Parameters
            ----------
            overwrite_allowed : bool, optional
                Flag indicating if overwriting is allowed. Default is True.

            Returns
            -------
            _Writer
                The _Writer object.
            """
            self.input_overwrite_allowed = overwrite_allowed
            return self

    class _Empty_Writer(_Writer):
        """
        A subclass of _Writer that represents an empty writer.

        Attributes
        ----------
        None

        Methods
        -------
        None
        """

        def __enter__(self):
            """
            A subclass of _Writer that represents an empty writer.

            Attributes
            ----------
            None

            Methods
            -------
            None
            """
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            """
            Called when exiting the empty writer context.

            Parameters
            ----------
            exc_type : type
                The type of the exception raised, if any.
            exc_value : Exception
                The exception raised, if any.
            traceback : traceback
                The traceback object associated with the exception.

            Returns
            -------
            None

            Raises
            ------
            Exception
                If an exception was raised.
            """
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)

    def get_writer(self, valid = True):
        """
        Returns a writer object for performing IO operations.

        Parameters
        ----------
        valid : bool, optional
            Flag indicating if the writer is valid. If valid is True, return self.__writer, 
            else return self._Empty_Writer(self). An _Empty_Writer will do nothing when entering 
            and exiting the context. Default is True.

        Returns
        -------
        _Writer or _Empty_Writer
            The writer object.

        Examples
        --------
        >>> with self.get_writer() as writer:
        >>>     for ...:
        >>>         ...
        >>> with self.get_writer().allow_overwriting() as writer:
        >>>     ...
        """
        ok = False
        if valid:
            ok = not self.is_writing
        
        if ok:
            return self.__writer
        else:
            return self._Empty_Writer(self) ###继续实现，修改之前的代码
    # endregion context and writer ###

    # region mark file ###
    @abstractmethod
    def get_writing_mark_file(self) -> str:
        """
        Returns the file path for the writing mark file.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The file path for the writing mark file.
        """
        pass

    @abstractmethod
    def identity_string(self, regen = False) -> str:
        """
        Returns a string representation of the IOStatusManager object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The string representation of the IOStatusManager object.
        """
        pass

    def mark_exist(self):
        """
        Checks if the writing mark file exists.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the writing mark file exists, False otherwise.
        """
        return os.path.exists(self.get_writing_mark_file())

    def remove_mark(self):
        """
        Removes the writing mark file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.mark_exist():
            os.remove(self.get_writing_mark_file())

    def load_from_mark_file(self):
        """
        Loads data from the writing mark file.

        Parameters
        ----------
        None

        Returns
        -------
        result (list): 
            The loaded data from the writing mark file.

        Raises
        ------
        AssertionError: If the format of the mark file is incorrect.
        """
        file_path = self.get_writing_mark_file()
        if os.path.exists(file_path):
            result = []
            with open(file_path, 'r') as file:
                for line in file:
                    # split string
                    columns = line.strip().split(', ')
                    assert len(columns) == 4, f"the format of {file_path} is wrong"
                    log_type, src, dst, value_str = columns
                    log_type = int(log_type)
                    # log_type must be in LOG_KN, and key must be int
                    assert log_type in self.LOG_KN, f"the format of {file_path} is wrong"
                    src = int(src) if src.isdigit() else src
                    dst = int(dst) if dst.isdigit() else dst
                    if value_str == 'None':
                        value = None
                    else:
                        try: value = int(value_str)
                        except: value = value_str
                    result.append([log_type, src, dst, value])
            return result
        else:
            return []

    def log_to_mark_file(self, log_type, src=None, dst=None, value=None):
        """
        Log the specified information to a mark file.

        Parameters
        ----------
        log_type (str): 
            The type of log, should be one of [LOG_READ, LOG_ADD, LOG_REMOVE, LOG_CHANGE, LOG_MOVE, LOG_OPERATION]
        src (str, optional): 
            The source of the log. Defaults to None.
        dst (str, optional): 
            The destination of the log. Defaults to None.
        value (Any, optional): 
            The value associated with the log. Defaults to None.
        """
        if src is None and dst is None and value is None:
            return 
        assert log_type in self.LOG_KN, f"log_type must be in {self.LOG_KN}"
        file_path = self.get_writing_mark_file()
        with open(file_path, 'a') as file:
            line = f"{log_type}, {src}, {dst}, {type(value)}\n"
            file.write(line)
    # endregion mark file ###

    # region IOStatus ###
    @property
    def closed(self):
        """
        Flag indicating if the IOStatusManager object is closed.
        Any IO operation will not be executed if the IOStatusManager object is closed.
        """
        return self.__closed
    
    @property
    def opened(self):
        """
        Flag indicating if the IOStatusManager object is opened.
        """
        return not self.__closed

    @property
    def readonly(self):
        """
        Flag indicating if the IOStatusManager object is read-only.
        Any write operation will not be executed if the IOStatusManager object is read-only.
        """
        return not self.writable # self.__readonly and not self.closed
    
    @property
    def writable(self):
        """
        Flag indicating if the IOStatusManager object is writable.
        """
        return not self.__readonly and not self.closed

    @property
    def wait_writing(self):
        """
        Flag indicating if the IOStatusManager object is wait_writing.

        see :link:`is_writing`, :link:`stop_writing`, :link:`start_writing`

        Examples
        -----
        >>> object = FilesCluster(...)
        >>> object.wait_writing # True
        True
        >>> with object.get_writer() as writer:
        >>>    object.wait_writing # False
        >>>    ...
        >>> object.wait_writing # True
        True
        """
        return not self.is_writing  #self.__wait_writing and not self.readonly and not self.closed

    @property
    def is_writing(self):
        """
        Flag indicating if the IOStatusManager object is writing.

        See :link:`wait_writing`, :link:`stop_writing`, :link:`start_writing`
        """
        return not self.__wait_writing and not self.readonly and not self.closed

    @property
    def overwrite_allowed(self):
        """
        Flag indicating if the IOStatusManager object allows overwriting.
        
        See :link:`set_overwrite_allowed`, :link:`set_overwrite_forbidden`, :link:`overwrite_forbidden`
        """
        return not self.overwrite_forbidden # not self.__overwrite_allowed and not self.readonly and not self.closed
    
    @property
    def overwrite_forbidden(self):
        """
        Flag indicating if the IOStatusManager object doesn't allow overwriting.
        
        See :link:`set_overwrite_allowed`, :link:`set_overwrite_forbidden`, :link:`overwrite_allowed`
        """
        return not self.__overwrite_allowed and not self.readonly and not self.closed
    
    def __close_core(self, closed:bool = True):
        if not self.closed and closed:
            self.stop_writing()
            self.set_readonly()
            self.close_hook()
        elif self.closed and not closed:
            self.open_hook()
        self.__closed = closed

    def __set_readonly_core(self, readonly:bool = True):
        if (self.readonly ^ readonly) and self.closed:
            warnings.warn(f"the Status is closed, please call '{self.set_readonly.__name__}' when it's opened", IOStatusWarning)
        if not self.readonly and readonly:
            self.stop_writing()
            self.readonly_hook()
        elif self.readonly and not readonly:
            self.writable_hook()
        self.__readonly = readonly

    def __stop_writing_core(self, stop_writing:bool = True):
        if (self.wait_writing ^ stop_writing) and (self.closed or self.readonly):
            warnings.warn(f"the Status is closed or readonly, please call '{self.stop_writing.__name__}' when it's writable", IOStatusWarning)
        if not self.wait_writing and stop_writing:
            # self.set_overwrite_forbidden()
            self.stop_writing_hook()
            if os.path.exists(self.get_writing_mark_file()):
                os.remove(self.get_writing_mark_file())
            self.__wait_writing = True
        elif self.wait_writing and not stop_writing:
            self.__wait_writing = False
            with open(self.get_writing_mark_file(), 'w'):
                pass
            self.start_writing_hook()

    def __set_overwrite_allowed_core(self, overwrite_allowed:bool = True):
        if (self.overwrite_allowed ^ overwrite_allowed) and (self.closed or self.readonly):
            warnings.warn(f"the Status is closed or readonly, please call '{self.set_overwrite_allowed.__name__}' when it's writable", IOStatusWarning)
        if not self.overwrite_allowed and overwrite_allowed:
            self.set_overwrite_allowed_hook()
        elif self.overwrite_allowed and not overwrite_allowed:
            self.set_overwrite_forbidden_hook()
        self.__overwrite_allowed = overwrite_allowed

    def close(self, closed: bool = True):
        """
        Close the IOAbstract object. all the io operation will not be executed.

        Parameters
        ----
        closed (bool, optional): 
            Flag indicating whether the object should be closed. 
            Defaults to True.
        """
        self.__close_core(closed)

    def open(self, opened: bool = True):
        """
        Open the IOAbstract object.

        Parameters
        ----
        opened (bool): 
            If True, opens the object. If False, closes the object. Defaults to True.

        Returns
        -----
            None
        """
        self.__close_core(not opened)

    def reopen(self):
        """
        Reopens the file or resource for reading or writing.
        This method first closes the file or resource using the `close()` method,
        and then opens it again using the `open()` method.
        """
        self.close()
        self.open()

    def set_readonly(self, readonly:bool = True):
        """
        Set the readonly flag for the object.
        
        Parameters
        -----
        readonly (bool): 
            Whether the object should be set as readonly or not. Default is True.
        
        Examples
        -----
        >>> object = FilesCluster(...)
        >>> object.set_readonly() # set readonly
        >>> object.read(0) # OK
        >>> object.write(0, ...) # raise ClusterIONotExecutedWarning
        """
        self.__set_readonly_core(readonly)
        
    def set_writable(self, writable:bool = True):
        """
        Set the writable flag for the object.

        Parameters
        -----
        writable (bool):
            Whether the object should be set as writable or not. Default is True.

        Warns
        ----
        NOT recommend to use this method. use `get_writer` instead.

        Examples
        ----
        NOT recommend:
        >>> object = FilesCluster(...)
        >>> object.set_writable() # set writable
        >>> object.read(0) # OK
        >>> object.write(0, ...) # OK
        >>> object.set_readonly() # set readonly

        recommend:
        >>> object = FilesCluster(...)
        >>> with object.get_writer() as writer:
        >>>     object.read(0) # OK
        >>>     object.write(0, ...) # OK
        """
        self.__set_readonly_core(not writable)    

    def stop_writing(self, stop_writing:bool = True):
        """
        Set the stop_writing flag for the object.
        `stop_writing_hook` wiil be called

        see :link:`start_writing`, :link:`wait_writing`, :link:`is_writing`

        Examples
        ----
        NOT recommend:
        >>> object = FilesCluster(...)
        >>> object.set_writable() # set writable
        >>> object.start_writing() # set start_writing
        >>> object.read(0) # OK
        >>> object.write(0, ...) # OK
        >>> object.stop_writing() # set stop_writing
        >>> object.set_readonly() # set readonly

        recommend:
        >>> object = FilesCluster(...)
        >>> with object.get_writer() as writer:
        >>>     object.read(0) # OK
        >>>     object.write(0, ...) # OK
        """
        self.__stop_writing_core(stop_writing)

    def start_writing(self, start_writing:bool = True, overwrite_allowed:bool = False):
        """
        `start_writing_hook` wiil be called

        see :link:`stop_writing`, :link:`wait_writing`, :link:`is_writing`

        """
        self.set_overwrite_allowed(overwrite_allowed)
        self.__stop_writing_core(not start_writing)

    def set_overwrite_allowed(self, overwrite_allowed:bool = True):
        """
        set the overwrite_allowed flag for the object.

        See :link:`set_overwrite_forbidden`, :link:`overwrite_allowed`, :link:`overwrite_forbidden`

        Examples
        -----
        >>> object = FilesCluster(...)
        >>> object.set_overwrite_forbidden() # set overwrite_forbidden
        >>> with object.get_writer() as writer:
        >>>     object.write(0, ...) # OK
        >>>     object.write(0, ...) # raise ClusterIONotExecutedWarning

        >>> object.set_overwrite_allowed() # set overwrite_allowed
        >>> with object.get_writer() as writer:
        >>>     object.write(0, ...) # OK
        >>>     object.write(0, ...) # OK
        """
        self.__set_overwrite_allowed_core(overwrite_allowed)

    def set_overwrite_forbidden(self, overwrite_forbidden:bool = True):
        """
        set the overwrite_forbidden flag for the object.

        See :link:`set_overwrite_allowed`, :link:`overwrite_allowed`, :link:`overwrite_forbidden`
        """
        self.__set_overwrite_allowed_core(not overwrite_forbidden)
    
    def is_closed(self, with_warning = False):
        '''Method to check if the cluster is closed.'''
        if with_warning and self.closed:
            warnings.warn(f"{self.identity_string()} is closed, any I/O operation will not be executed.",
                            ClusterIONotExecutedWarning)
        return self.closed

    def is_readonly(self, with_warning = False):
        '''Method to check if the cluster is read-only.'''
        if with_warning and self.readonly:
            warnings.warn(f"{self.identity_string()} is read-only, any write operation will not be executed.",
                ClusterIONotExecutedWarning)
        return self.readonly
    
    def close_hook(self):
        """
        called when `close` is called
        * recommend to override this method
        """
        pass

    def open_hook(self):
        """
        called when `open` is called
        * recommend to override this method
        """
        pass

    def readonly_hook(self):
        """
        called when `set_readonly` is called
        * recommend to override this method
        """
        pass

    def writable_hook(self):
        """
        called when `set_writable` is called
        * recommend to override this method
        """
        pass

    def stop_writing_hook(self):
        """
        called when `stop_writing` is called
        * recommend to override this method
        """
        pass

    def start_writing_hook(self):
        """
        called when `start_writing` is called
        * recommend to override this method
        """
        pass

    def set_overwrite_allowed_hook(self):
        """
        called when `set_overwrite_allowed` is called
        * recommend to override this method
        """
        pass

    def set_overwrite_forbidden_hook(self):
        """
        called when `set_overwrite_forbidden` is called
        * recommend to override this method
        """
        pass
    # endregion IOStatus ###

class _RegisterInstance(ABC, Generic[RGSITEM]):
    """
    A base class for registering instances with unique identities.

    * Every instance of a subclass of _RegisterInstance will be registered in 
    `_RegisterInstance._registry` with its identity_string.

    Attributes
    -----
    _registry (dict[str, RGSITEM])
        A dictionary to store registered instances.
    INDENTITY_PARA_NAMES (list)
        A list of parameter names that define the identity of an instance.
    """

    _registry:dict[str, RGSITEM] = {}
    INDENTITY_PARA_NAMES = []

    # region _RegisterInstance ###
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.init_identity(*args, **kwargs)  # initialize identity_paras
        obj.__lock_identity_paras()          # lock it
        obj._identity_string = obj.identity_string()
        obj._identity_name   = obj.identity_name()
        id_str = obj.identity_string()
        if cls.has_instance(obj):
            obj = cls.get_instance(id_str, obj)
        else:
            cls.register(id_str, obj)
        return obj

    @classmethod
    @abstractmethod
    def register(cls, identity_string, obj):
        """
        register an instance with its identity_string.
        
        * the strategy of registering is defined by the subclass
        """
        pass

    @classmethod
    @abstractmethod
    def get_instance(cls, identity_string, obj):
        """
        get an instance by identity_string.
        
        * the strategy of getting is defined by the subclass
        """
        pass

    def identity_string(self, regen = False):
        """
        Note
        ----
        `identity_string` returns a string that can uniquely identify an instance
        """
        if regen:
            self._identity_string = self.gen_identity_string()
        try:
            return self._identity_string
        except:
            self._identity_string = self.gen_identity_string()
            return self._identity_string

    def identity_name(self):
        """
        Note
        ----
        `indentity_string()` might be too long, so sometime we need a short name to represent it.
        If you are sure there will be no ambiguity, you can use `identity_name()` to distinguish different instances.
        """
        try:
            return self._identity_name
        except:
            self._identity_name = self.gen_identity_name()
            return self._identity_name

    @abstractmethod
    def gen_identity_string(self):
        """
        generate the identity_string of an instance
        """
        pass

    @abstractmethod
    def gen_identity_name(self):
        """
        generate the identity_name of an instance
        """
        pass

    @abstractmethod
    def init_identity(self, *args, **kwargs):
        '''
        principle: the parameter of identity can not be changed once it has been inited
        '''
        pass

    @classmethod
    def has_instance(cls, obj:"_RegisterInstance"):
        """
        if an instance has been registered

        Parameters
        -----
        obj (_RegisterInstance):
            the instance to be checked
        """
        identity_string = obj.identity_string()
        return identity_string in cls._registry

    def __lock_identity_paras(self, lock = True):
        self.__identity_paras_locked = lock  
    # endregion _RegisterInstance ###

    # region __setattr__###
    def __setattr__(self, name: str, value) -> Any:
        __identity_paras_locked = self.__identity_paras_locked if hasattr(self, "__identity_paras_locked") else False 
        if __identity_paras_locked and name in self.INDENTITY_PARA_NAMES:
            raise AttributeError(f"cannot set {name}")
        super().__setattr__(name, value)
    # endregion __setattr__###

class _Prefix(dict[str, str]):
    """
    A class representing a prefix configuration with options for prefix and joiner.

    Parameters
    ----------
    prefix : str, optional
        The prefix to be used, by default an empty string.
    joiner : Any, optional
        The joiner to be used, by default an empty string.

    Attributes
    ----------
    KW_PREFIX : str
        Keyword for the prefix attribute.
    KW_JOINER : str
        Keyword for the joiner attribute.

    Methods
    -------
    prefix : str
        Getter and setter for the prefix attribute.
    joiner : Any
        Getter and setter for the joiner attribute.
    get_with_joiner() -> str
        Returns the concatenation of prefix and joiner.
    __repr__() -> str
        Returns a string representation of the Prefix object.
    as_dict() -> dict
        Returns the Prefix object as a dictionary.
    from_dict(dict_) -> _Prefix
        Creates a Prefix object from a dictionary.

    Examples
    --------
    >>> prefix_instance = _Prefix(prefix="abc", joiner="-")
    >>> print(prefix_instance.get_with_joiner())
    "abc-"

    >>> prefix_dict = {'prefix': 'xyz', 'joiner': '_'}
    >>> new_prefix_instance = _Prefix.from_dict(prefix_dict)
    >>> print(new_prefix_instance)
    "Prefix(xyz, _)"
    """

    KW_PREFIX = "prefix"
    KW_JOINER = "joiner"

    def __init__(self, prefix: str = "", joiner: str = "") -> None:
        """
        Initialize the Prefix object.

        Parameters
        ----------
        prefix : str, optional
            The prefix to be used, by default an empty string.
        joiner : Any, optional
            The joiner to be used, by default an empty string.
        """
        super().__init__()
        self[self.KW_PREFIX] = prefix
        self[self.KW_JOINER] = joiner

    @property
    def prefix(self) -> str:
        """
        Getter for the prefix attribute.

        Returns
        -------
        str
            The current prefix value.
        """
        return self[self.KW_PREFIX]

    @prefix.setter
    def prefix(self, value: str) -> None:
        """
        Setter for the prefix attribute.

        Parameters
        ----------
        value : str
            The new value for the prefix.
        """
        assert isinstance(value, str), f"prefix must be a str"
        self[self.KW_PREFIX] = value

    @property
    def joiner(self) -> str:
        """
        Getter for the joiner attribute.

        Returns
        -------
        str
            The current joiner value.
        """
        return self[self.KW_JOINER]

    @joiner.setter
    def joiner(self, value: str) -> None:
        """
        Setter for the joiner attribute.

        Parameters
        ----------
        value : str
            The new value for the joiner.
        """
        self[self.KW_JOINER] = value

    def get_with_joiner(self) -> str:
        """
        Returns the concatenation of prefix and joiner.

        Returns
        -------
        str
            The result of concatenating prefix and joiner.
        """
        return self.prefix + self.joiner

    def __repr__(self) -> str:
        """
        Returns a string representation of the Prefix object.

        Returns
        -------
        str
            A string representation of the Prefix object.
        """
        return f"Prefix({self.prefix}, {self.joiner})"

    def as_dict(self) -> dict:
        """
        Returns the Prefix object as a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the Prefix object.
        """
        return dict(self)

    @classmethod
    def from_dict(cls, dict_: dict) -> "_Prefix":
        """
        Creates a Prefix object from a dictionary.

        Parameters
        ----------
        dict_ : dict
            The dictionary containing the prefix and joiner values.

        Returns
        -------
        _Prefix
            A new Prefix object created from the provided dictionary.
        """
        prefix = dict_[cls.KW_PREFIX]
        joiner = dict_[cls.KW_JOINER]
        return cls(prefix, joiner)


from typing import Union, List, Iterable

class _AppendNames(dict[str, Union[List[str], str]]):
    """
    A class representing a configuration for appending names with options for a list of appendable names and a joiner.

    Parameters
    ----------
    appendnames : list of str, optional
        The list of names to be appended, by default an empty list.
    joiner : str, optional
        The joiner to be used, by default "_".

    Attributes
    ----------
    KW_APPENDNAMES : str
        Keyword for the appendnames attribute.
    KW_JOINER : str
        Keyword for the joiner attribute.

    Methods
    -------
    joiner : str
        Getter for the joiner attribute.
    appendnames : list of str
        Getter for the appendnames attribute.
    get_with_joiner() -> list of str
        Returns a list of names with the joiner added.
    extend(names: Union[str, Iterable[str]]) -> None
        Extends the appendnames attribute with new names.
    add_appendname(appendname: str) -> bool
        Adds an appendname to the list if it does not already exist.
    remove_appendname(appendname: str) -> bool
        Removes an appendname from the list if it exists.
    conditional_return(mutil_file: bool, list_like: list of str) -> Union[str, List[str]]
        Conditionally returns either the list of names or the first name based on the value of mutil_file.
    __repr__() -> str
        Returns a string representation of the AppendNames object.
    from_dict(dict_: dict) -> _AppendNames
        Creates an AppendNames object from a dictionary.
    as_dict() -> dict
        Returns the AppendNames object as a dictionary.

    Examples
    --------
    >>> append_names_instance = _AppendNames(appendnames=["name1", "name2"], joiner="-")
    >>> print(append_names_instance.get_with_joiner())
    ["-name1", "-name2"]

    >>> append_names_dict = {'appendnames': ['item1', 'item2'], 'joiner': '+'}
    >>> new_append_names_instance = _AppendNames.from_dict(append_names_dict)
    >>> print(new_append_names_instance)
    "AppendNames({'appendnames': ['item1', 'item2'], 'joiner': '+'})"

    """

    KW_APPENDNAMES = "appendnames"
    KW_JOINER = "joiner"

    def __init__(self, appendnames: List[str] = None, joiner: str = "_") -> None:  # type: ignore
        """
        Initialize the AppendNames object.

        Parameters
        ----------
        appendnames : list of str, optional
            The list of names to be appended, by default an empty list.
        joiner : str, optional
            The joiner to be used, by default "_".
        """
        super().__init__()
        assert isinstance(appendnames, list) or appendnames is None, f"appendnames must be a list or None"
        assert isinstance(joiner, str), f"joiner must be a str"
        appendnames = appendnames if appendnames is not None else []
        self[self.KW_APPENDNAMES] = appendnames
        self[self.KW_JOINER] = joiner

    @property
    def joiner(self) -> str:
        """
        Getter for the joiner attribute.

        Returns
        -------
        str
            The current joiner value.
        """
        return self[self.KW_JOINER]  # type: ignore

    @property
    def appendnames(self) -> List[str]:
        """
        Getter for the appendnames attribute.

        Returns
        -------
        list of str
            The current list of appendable names.
        """
        return self[self.KW_APPENDNAMES]  # type: ignore

    def get_with_joiner(self) -> List[str]:
        """
        Returns a list of names with the joiner added.

        Returns
        -------
        list of str
            The result of concatenating the joiner with each name in the appendnames list.
        """
        rlt_list: List[str] = []
        for x in self.appendnames:
            x = self.joiner + x
            rlt_list.append(x)
        return rlt_list

    def extend(self, names: Union[str, Iterable[str]]) -> None:
        """
        Extends the appendnames attribute with new names.

        Parameters
        ----------
        names : str or iterable of str
            The names to be added to the appendnames list.
        """
        if isinstance(names, str):
            names = [names]
        assert isinstance(names, Iterable), f"names must be an iterable"
        assert all([isinstance(x, str) for x in names]), f"names must be a list of str"
        self.appendnames.clear()
        self.appendnames.extend(names)

    def add_appendname(self, appendname: str) -> bool:
        """
        Adds an appendname to the list if it does not already exist.

        Parameters
        ----------
        appendname : str
            The name to be added to the appendnames list.

        Returns
        -------
        bool
            True if the name was added, False if it already existed.
        """
        if appendname not in self.appendnames:
            self.appendnames.append(appendname)
            return True
        return False

    def remove_appendname(self, appendname: str) -> bool:
        """
        Removes an appendname from the list if it exists.

        Parameters
        ----------
        appendname : str
            The name to be removed from the appendnames list.

        Returns
        -------
        bool
            True if the name was removed, False if it did not exist.
        """
        if appendname in self.appendnames:
            self.appendnames.remove(appendname)
            return True
        return False

    @staticmethod
    def conditional_return(mutil_file: bool, list_like: List[str]) -> Union[str, List[str]]:
        """
        Conditionally returns either the list of names or the first name based on the value of mutil_file.

        Parameters
        ----------
        mutil_file : bool
            A boolean flag indicating whether to return the full list or just the first item.
        list_like : list of str
            The list of names.

        Returns
        -------
        Union[str, List[str]]
            Either the full list or the first item based on the value of mutil_file.
        """
        if mutil_file:
            return list_like
        return list_like[0]

    def __repr__(self) -> str:
        """
        Returns a string representation of the AppendNames object.

        Returns
        -------
        str
            A string representation of the AppendNames object.
        """
        return f"AppendNames({self})"

    @classmethod
    def from_dict(cls, dict_: dict) -> "_AppendNames":
        """
        Creates an AppendNames object from a dictionary.

        Parameters
        ----------
        dict_ : dict
            The dictionary containing the appendnames and joiner values.

        Returns
        -------
        _AppendNames
            A new AppendNames object created from the provided dictionary.
        """
        appendnames = dict_[cls.KW_APPENDNAMES]
        joiner = dict_[cls.KW_JOINER]
        return cls(appendnames, joiner)

    def as_dict(self) -> dict:
        """
        Returns the AppendNames object as a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the AppendNames object.
        """
        return dict(self)
class CacheProxy(Generic[VDMT]):
    """
    A generic class representing a cache proxy with options for the cache, value type, and initialization function.

    Parameters
    ----------
    cache : VDMT or any
        The cache to be used.
    value_type : type[VDMT], optional
        The type of values stored in the cache, by default None.
    value_init_func : Callable, optional
        The initialization function for the cache, by default None.

    Attributes
    ----------
    KW_cache : str
        Keyword for the cache attribute.
    KW_value_type : str
        Keyword for the value_type attribute.

    Methods
    -------
    value_type : type[VDMT]
        Getter for the value_type attribute.
    cache : VDMT or None
        Getter and setter for the cache attribute.
    as_dict() -> dict
        Returns the CacheProxy object as a dictionary.
    from_dict(dict_: dict) -> CacheProxy
        Creates a CacheProxy object from a dictionary.
    init_cache() -> bool
        Initializes the cache using the specified initialization function.
    
    Examples
    --------
    >>> cache_instance = CacheProxy(cache=SomeCache(), value_type=int)
    >>> print(cache_instance.value_type)
    <class 'int'>

    >>> cache_dict = {'cache': SomeCache(), 'value_type': int}
    >>> new_cache_instance = CacheProxy.from_dict(cache_dict)
    >>> print(new_cache_instance)
    "CacheProxy({'cache': SomeCache(), 'value_type': int})"

    """

    KW_cache = "cache"
    KW_value_type = "value_type"

    _unpickleable_type = []
    _pickleable_type = []

    def __init__(self, cache, value_type: Optional[type[VDMT]] = None, value_init_func: Optional[Callable] = None) -> None:
        """
        Initialize the `CacheProxy` object.

        Parameters
        ----------
        cache : `VDMT` or any
            The cache to be used.
        value_type : type[`VDMT`], optional
            The type of values stored in the cache, by default None.
        value_init_func : Callable, optional
            The initialization function for the cache, by default None.
        """
        self.__cache: Optional[VDMT] = None
        self.synced = False
        self.__value_type: Optional[type[VDMT]] = value_type
        self.__value_init_func: Optional[Callable[..., VDMT]] = value_init_func if value_init_func is not None else value_type

        self.cache = cache
        self.init_cache()

    @property
    def value_type(self) -> type[VDMT]:
        """
        Getter for the value_type attribute.

        Returns
        -------
        type[`VDMT`]
            The current value_type.
        """
        return self.__value_type  # type: ignore

    @property
    def cache(self) -> Union[VDMT, None]:
        """
        Getter for the cache attribute.

        Returns
        -------
        `VDMT` or None
            The current cache value.
        """
        return self.__cache

    @cache.setter
    def cache(self, cache: Union[VDMT, Any]):
        """
        Setter for the cache attribute.

        Parameters
        ----------
        cache : `VDMT` or any
            The new cache value.
        """
        if self.value_type is not None and cache is not None:
            if not isinstance(cache, self.value_type) and self.__value_init_func is not None:
                cache = self.__value_init_func(cache)
        self.__cache = copy.deepcopy(cache)

    def as_dict(self) -> dict:
        """
        Returns the `CacheProxy` object as a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the `CacheProxy` object.
        """
        dict_ = {}

        cache = self.__cache
        if type(cache) in self._unpickleable_type:
            unpickleable = True
        elif type(cache) in self._pickleable_type:
            unpickleable = False
        else:
            unpickleable = not test_pickleable(cache)
            if unpickleable:
                self._unpickleable_type.append(type(cache))
            else:
                self._pickleable_type.append(type(cache))

        if unpickleable:
            cache = None

        dict_[self.KW_cache] = cache
        dict_[self.KW_value_type] = self.__value_type
        return dict_

    @classmethod
    def from_dict(cls, dict_: dict) -> "CacheProxy":
        """
        Creates a `CacheProxy` object from a dictionary.

        Parameters
        ----------
        dict_ : dict
            The dictionary containing the cache and value_type values.

        Returns
        -------
        `CacheProxy`
            A new `CacheProxy` object created from the provided dictionary.
        """
        cache = dict_[cls.KW_cache]
        value_type = dict_[cls.KW_value_type]
        return cls(cache, value_type)

    def init_cache(self) -> bool:
        """
        Initializes the cache using the specified initialization function.

        Returns
        -------
        bool
            True if initialization was successful, False otherwise.
        """
        if self.value_type is not None and self.cache is None and self.__value_init_func is not None:
            try:
                init_cache = self.__value_init_func()
            except Exception as e:
                if DEBUG:
                    raise Exception(e).with_traceback(sys.exc_info()[2])
                else:
                    pass
            else:
                self.cache = init_cache
                return True
        return False

class FilesHandle(_RegisterInstance["FilesHandle"], Generic[FCT, VDMT]):
    '''
    Note
    -----
    A class representing a file handle for input/output operations. 

    * Its instance manages some attributes of one file or a group of files, depending on the value of the `multi_files` attribute.

    * Its instance is immutable object, once created, it can't be changed.

    Parameters
    -----
    cluster (FilesCluster):
        The cluster to which the file handle belongs.
    data_path (str): 
        The top path to the file or directory.
    sub_dir (str):
        The subdirectory of the file or directory. Defaults to "".
    corename (str):
        The corename of the file or directory. Defaults to "".
    suffix (str):
        The suffix of the file or directory. Defaults to "".
    _appendnames_obj (`_AppendNames`):
        The `_AppendNames` object. Defaults to None.
    _prefix_obj (`_Prefix`):
        The `_Prefix` object. Defaults to None.
    read_func (Callable):
        The function to be used for reading the file. Defaults to None.
    write_func (Callable):
        The function to be used for writing the file. Defaults to None.
    cache_proxy (CacheProxy):
        The cache proxy to be used. Defaults to None.

    Examples
    -----
    >>> cluster = FilesCluster("./top")
    >>> fh = FilesHandle(cluster, sub_dir="sub", corename="core", suffix=".txt")
    >>> fh.get_path()
    "./top/sub/core.txt"
    >>> fh2 = FilesHandle(cluster, sub_dir="sub", corename="core", suffix=".txt", prefix="prefix", appendnames=["append1", "append2"], prefix_joiner="-", appendnames_joiner="_")
    >>> fh2.get_path()
    ['./top/sub/prefix-core_append1.txt', './top/sub/prefix-core_append2.txt']
    '''

    # region class variables ###
    KW_data_path         = "data_path"
    KW_sub_dir           = "sub_dir"
    KW_corename          = "corename"
    KW_suffix            = "suffix"
    KW_appendnames       = "appendnames"   
    KW_prefix            = "prefix" 
    KW_read_func         = "read_func"
    KW_write_func        = "write_func"
    KW_cache             = "cache"  

    DEFAULT_FILE_TYPE = {
        ".json": [JsonIO.load_json, JsonIO.dump_json, dict],
        ".npy":  [partial(np.load, allow_pickle=True), partial(np.save, allow_pickle=True), None],
        ".npz":  [partial(np.load, allow_pickle=True), partial(np.savez, allow_pickle=True), None],
        ".pkl":  [deserialize_object, serialize_object, None],
        ".txt":  [read_file_as_str, write_str_to_file, None],
        ".png":  [cv2.imread, cv2.imwrite, None],
        ".jpg":  [cv2.imread, cv2.imwrite, None],
        ".jpeg": [cv2.imread, cv2.imwrite, None],
        ".bmp":  [cv2.imread, cv2.imwrite, None],
        ".tif":  [cv2.imread, cv2.imwrite, None],
    }
    """
    some default file type and their read/write functions
    """

    DEFAULT_SUFFIX = None
    """can be overrided by subclass"""
    DEFAULT_PREFIX = None
    """can be overrided by subclass"""
    DEFAULT_PREFIX_JOINER = None
    """can be overrided by subclass"""
    DEFAULT_APPENDNAMES = None
    """can be overrided by subclass"""
    DEFAULT_APPENDNAMES_JOINER = None
    """can be overrided by subclass"""
    DEFAULT_READ_FUNC = None
    """can be overrided by subclass"""
    DEFAULT_WRITE_FUNC = None
    """can be overrided by subclass"""
    DEFAULT_VALUE_TYPE = None
    """can be overrided by subclass"""
    DEFAULT_VALUE_INIT_FUNC = None
    """can be overrided by subclass"""

    LOAD_CACHE_ON_INIT = False
    """
    * can be overrided by subclass
    * if True, load cache from disk when init
    """

    KW_INIT_WITHOUT_CACHE = "INIT_WITHOUT_CACHE"

    GET_INSTANCE_STRATEGY = 0 
    """
    control the strategy of `get_instance`

    * 0: return the inited instance, 
    * 1: return the original instance and cover the inited instance
    * 2: return the original instance and not register it
    """

    __unpickleable_func = []
    __pickleable_func   = []
    KW_FUNC_NAME        = "func_name"
    KW_FUNC_MODULE      = "func_module"
    KW_FUNC_BIND_ARGS   = "func_bind_args"
    KW_FUNC_BIND_KWARGS = "func_bind_kwargs"
    # endregion class variables ###

    # region override _RegisterInstance###
    INDENTITY_PARA_NAMES = ["sub_dir", "corename", "suffix", "prefix", "appendnames", 
                            "prefix_joiner", "appendnames_joiner", "data_path"]

    @classmethod
    def register(cls, identity_string, obj:"FilesHandle"):
        cls._registry[identity_string] = obj

    @classmethod
    def get_instance(cls, identity_string, orig_obj:"FilesHandle"):
        """
        Get an instance of the class based on the specified identity string and original object.

        Note
        -----
        the strategy is controled by `GET_INSTANCE_STRATEGY`

        * 0: return the inited instance, 
        * 1: return the original instance and cover the inited instance
        * 2: return the original instance and not register it

        Parameters
        -----
        identity_string (str): 
            The identity string.
        orig_obj (FilesHandle): 
            The original object.

        Returns
        -----
        FilesHandle: 
            The instance of the class.

        Raises
        -----
        ValueError: 
            If `GET_INSTANCE_STRATEGY` is not 0, 1, or 2.
        """
        if cls.GET_INSTANCE_STRATEGY == 0:
            obj:FilesHandle = cls._registry[identity_string]
            if obj.multi_files:
                for appname in orig_obj._appendnames_obj.appendnames:
                    obj.add_appendname(appname)
            return obj
        elif cls.GET_INSTANCE_STRATEGY == 1:
            if orig_obj.multi_files:
                obj:FilesHandle = cls._registry[identity_string]
                for appname in obj._appendnames_obj.appendnames:
                    orig_obj.add_appendname(appname)
            cls.register(identity_string, orig_obj)
            cls.GET_INSTANCE_STRATEGY = 0
            return orig_obj
        elif cls.GET_INSTANCE_STRATEGY == 2:
            cls.GET_INSTANCE_STRATEGY = 0
            return orig_obj
        else:
            raise ValueError(f"GET_INSTANCE_STRATEGY must be 0, 1 or 2, not {cls.GET_INSTANCE_STRATEGY}")

    def gen_identity_string(self):
        """
        """
        if not self.multi_files:
            return self.get_path()
        else:
            dir_ = self.get_dir()
            name = self.prefix_with_joiner + self.corename + self._appendnames_obj.joiner + '[...]' + self.suffix 
            return os.path.join(dir_, name)

    def gen_identity_name(self):
        """
        """
        return self.corename

    def init_identity(self, cluster:FCT, sub_dir:str, corename:str, suffix:str, * ,
                 prefix:Optional[str] = None, appendnames:Optional[Union[str, list[str]]] = None,  # type: ignore
                 prefix_joiner:Optional[str] = None, appendnames_joiner:Optional[str] = None, # type: ignore
                 data_path = "",
                 read_func:Optional[Callable] = None, write_func:Optional[Callable] = None, 
                 cache = None, value_type:Optional[type] = None):
        '''
        principle: the parameter of identity can not be changed once it has been inited
        '''
        self.cluster:FCT = cluster

        # sub_dir             = get_with_priority(sub_dir)
        # corename            = get_with_priority(corename)
        suffix              = get_with_priority(suffix, self.DEFAULT_SUFFIX) # type: ignore

        prefix:str              = get_with_priority(prefix,             cluster.DEFAULT_PREFIX,             self.DEFAULT_PREFIX,             '') # type: ignore
        appendnames:list[str]         = get_with_priority(appendnames,        cluster.DEFAULT_APPENDNAMES,        self.DEFAULT_APPENDNAMES,        ['']) # type: ignore
        if isinstance(appendnames, str):
            appendnames = [appendnames]
        prefix_joiner:str       = get_with_priority(prefix_joiner,      cluster.DEFAULT_PREFIX_JOINER,      self.DEFAULT_PREFIX_JOINER,      '') # type: ignore
        appendnames_joiner:str  = get_with_priority(appendnames_joiner, cluster.DEFAULT_APPENDNAMES_JOINER, self.DEFAULT_APPENDNAMES_JOINER, '') # type: ignore

        if len(suffix) == 0 or suffix[0] != '.':
            suffix = '.' + suffix
    
        data_path = self.cluster.data_path if data_path == "" else data_path
        # if ".." in data_path:
        data_path = os.path.normpath(data_path)
        assert is_subpath(data_path, self.cluster.data_path), f"data_path must be in {self.cluster.data_path}"

        self.data_path = data_path
        self.sub_dir = sub_dir
        self.corename = corename
        self.suffix = suffix

        self._prefix_obj         = _Prefix(prefix, prefix_joiner)
        
        self._appendnames_obj    = _AppendNames(appendnames, appendnames_joiner)

    # endregion override _RegisterInstance###

    # region hook
    def __init_subclass__(cls, **kwargs):
        ### __init__ ###
        cls.__init__ = method_exit_hook_decorator(cls, cls.__init__, cls.set_inited, cls.has_not_inited)
        super().__init_subclass__(**kwargs)

    def set_inited(self):
        self._inited = True

    def has_not_inited(self):
        try:
            self._inited
        except AttributeError:
            return True
        else:
            return False
    # endregion hook               

    # region init and create instance

    # region - init
    def __init__(self, cluster: FCT, sub_dir: str, corename: str, suffix: str, *,
                 prefix: Optional[str] = None, appendnames: Union[str, list[str]] = None,
                 prefix_joiner: Optional[str] = None, appendnames_joiner: Optional[str] = None,
                 data_path: str = "",
                 read_func: Optional[Callable] = None, write_func: Optional[Callable] = None,
                 cache=None, value_type: Optional[type[VDMT]] = None) -> None:
        """
        Initialize the IOAbstract class.

        Parameters
        ----
        cluster (FCT): 
            The cluster object.
        sub_dir (str): 
            The sub-directory.
        corename (str): 
            The core name.
        suffix (str): 
            The suffix.
        prefix (Optional[str], optional): 
            The prefix. Defaults to None.
        appendnames (Union[str, list[str]], optional): 
            The append names. Defaults to None.
        prefix_joiner (Optional[str], optional): 
            The prefix joiner. Defaults to None.
        appendnames_joiner (Optional[str], optional): 
            The append names joiner. Defaults to None.
        data_path (str, optional): 
            The data path. Defaults to "".
        read_func (Optional[Callable], optional): 
            The read function. Defaults to None.
        write_func (Optional[Callable], optional): 
            The write function. Defaults to None.
        cache (Any, optional): 
            The cache. Defaults to None.
        value_type (Optional[type[VDMT]], optional): 
            The value type. Defaults to None.

        Returns
        -----
        None
        """
        super().__init__()

        # Call init_input_hook and assign the returned values to variables
        (sub_dir, corename, suffix,
         prefix, appendnames, prefix_joiner, appendnames_joiner,
         data_path,
         read_func, write_func,
         cache, value_type) = \
            self.init_input_hook(sub_dir=sub_dir, corename=corename, suffix=suffix,
                                 prefix=prefix, appendnames=appendnames,
                                 prefix_joiner=prefix_joiner, appendnames_joiner=appendnames_joiner,
                                 data_path=data_path,
                                 read_func=read_func, write_func=write_func,
                                 cache=cache, value_type=value_type)

        # Check if read_func and write_func are lambda functions
        for func in [read_func, write_func]:
            if hasattr(func, "__code__"):
                assert not func.__code__.co_name == '<lambda>', "function should not be lambda"

        self.read_func = read_func
        self.write_func = write_func

        # Set cache based on the LOAD_CACHE_ON_INIT flag and all_file_exist condition
        if cache == self.KW_INIT_WITHOUT_CACHE:
            cache = None
        else:
            if self.LOAD_CACHE_ON_INIT and self.all_file_exist:
                cache = self.read()
            else:
                cache = cache

        # Get the default value type based on priority
        default_value_type = get_with_priority(self.cluster.DEFAULT_VALUE_INIT_FUNC,
                                               self.DEFAULT_VALUE_INIT_FUNC,
                                               self.DEFAULT_VALUE_TYPE)

        # Create a CacheProxy object
        self.cache_proxy: CacheProxy[VDMT] = CacheProxy[VDMT](cache, value_type, default_value_type)

        # Call init_additional_hook
        self.init_additional_hook()

    def init_input_hook(self, *, sub_dir, corename, suffix, 
                             prefix, appendnames, prefix_joiner, appendnames_joiner, 
                             data_path, 
                             read_func, write_func, 
                             cache, value_type):
        """
        Initializes the input hook with the specified parameters.

        * This method can be overridden by subclasses to customize the input hook.

        Args
        -----
        sub_dir (str): 
            The sub-directory where the data is located.
        corename (str): 
            The core name of the data file.
        suffix (str): 
            The suffix of the data file.
        prefix (str): 
            The prefix of the data file.
        appendnames (List[str]): 
            The list of names to append to the data file.
        prefix_joiner (str): 
            The joiner used to concatenate the prefix and appendnames.
        appendnames_joiner (str): 
            The joiner used to concatenate the appendnames.
        data_path (str): 
            The top dir to the data file.
        read_func (callable): 
            The function used to read the data file.
        write_func (callable): 
            The function used to write the data file.
        cache (bool): 
            Flag indicating whether to cache the data.
        value_type (type): 
            The type of the data values.

        Returns
        -----
        Tuple[str, str, str, str, List[str], str, str, str, callable, callable, bool, type]: The initialized input hook parameters.
        """
        return (sub_dir, corename, suffix, 
                prefix, appendnames, prefix_joiner, appendnames_joiner, 
                data_path, 
                read_func, write_func, 
                cache, value_type)

    def init_additional_hook(self):
        pass
    
    def add_appendname(self, appendname):
        return self._appendnames_obj.add_appendname(appendname)
    
    def remove_appendname(self, appendname):
        return self._appendnames_obj.remove_appendname(appendname)
    # endregion - init

    # region - export and import func
    @classmethod
    def export_func(cls, func: Union[Callable, None]):
        """
        Export a function for serialization. If the function is unpickleable, it returns a dictionary containing

        Args
        -----
        func (Callable or None): 
            The function to be exported.

        Returns
        -----
        Union[dict, Callable, None]: 
            If the function is unpickleable, it returns a dictionary containing
            information about the function. Otherwise, it returns the function itself.
        """
        if func is None:
            return None
        if func in cls.__unpickleable_func:
            unpickleable = True
        elif func in cls.__pickleable_func:
            unpickleable = False
        else:
            unpickleable = not test_pickleable(func)
            if unpickleable:
                cls.__unpickleable_func.append(func)
            else:
                cls.__pickleable_func.append(func)
        if unpickleable:
            if isinstance(func, partial):
                raw_func = func.func
                bind_args = func.args
                bind_kwargs = func.keywords
                func = raw_func
            else:
                bind_args = tuple()
                bind_kwargs = {}
            func_info = {
                cls.KW_FUNC_NAME: func.__name__,
                cls.KW_FUNC_MODULE: func.__module__,
                cls.KW_FUNC_BIND_ARGS: bind_args,
                cls.KW_FUNC_BIND_KWARGS: bind_kwargs
            }
            return func_info
        else:
            return func

    @classmethod
    def import_func(cls, import_obj):
        """
        Import and return a function based on the given import object.

        Parameters
        -----
        import_obj: 
            The import object, which can be a dictionary or a callable.

        Returns
        -----
        The imported function.

        Raises
        -----
        ValueError: 
            If the import object is not a dictionary or a callable.
            It suggests installing the module if the module name is provided in the import object.
        """
        if isinstance(import_obj, dict):
            module = importlib.import_module(import_obj[cls.KW_FUNC_MODULE])
            func = getattr(module, import_obj[cls.KW_FUNC_NAME])
            bind_args = import_obj[cls.KW_FUNC_BIND_ARGS]
            bind_kwargs = import_obj[cls.KW_FUNC_BIND_KWARGS]
            if len(bind_args) > 0 or len(bind_kwargs) > 0:
                func = partial(func, *bind_args, **bind_kwargs)
        elif isinstance(import_obj, Callable):
            func = import_obj
        else:
            raise ValueError(f"unknown function, try to install module:{import_obj[cls.KW_FUNC_MODULE]}")

        return func
    # endregion - export and import func

    # region - export and import dict
    def as_dict(self):
        """
        convert the FilesHandle to a dict

        Returns
        -----
        dict
        """
        dict_ = {}
        dict_[self.KW_data_path]     = self.data_path
        dict_[self.KW_sub_dir]       = self.sub_dir
        dict_[self.KW_corename]      = self.corename
        dict_[self.KW_suffix]        = self.suffix
        dict_[self.KW_prefix]        = self._prefix_obj.as_dict()
        dict_[self.KW_appendnames]   = self._appendnames_obj.as_dict()
        dict_[self.KW_read_func]     = self.export_func(self.read_func)
        dict_[self.KW_write_func]    = self.export_func(self.write_func)
        dict_[self.KW_cache]         = self.cache_proxy.as_dict()
        return dict_

    @classmethod
    def from_dict(cls, cluster:FCT, dict_:dict):
        """
        create a FilesHandle from a dict

        Returns
        -----
        FilesHandle
        """
        data_path:str       = get_with_priority(cluster.data_path, dict_[cls.KW_data_path]) # type: ignore
        sub_dir         = dict_[cls.KW_sub_dir]
        corename        = dict_[cls.KW_corename]
        suffix          = dict_[cls.KW_suffix]      

        prefix                  = dict_[cls.KW_prefix][_Prefix.KW_PREFIX]
        prefix_joiner           = dict_[cls.KW_prefix][_Prefix.KW_JOINER]
        appendnames             = dict_[cls.KW_appendnames][_AppendNames.KW_APPENDNAMES]
        appendnames_joiner      = dict_[cls.KW_appendnames][_AppendNames.KW_JOINER]

        read_func       = cls.import_func(dict_[cls.KW_read_func])
        write_func      = cls.import_func(dict_[cls.KW_write_func])

        cache           = dict_[cls.KW_cache][CacheProxy.KW_cache]
        value_type      = dict_[cls.KW_cache][CacheProxy.KW_value_type]

        obj = cls(cluster, sub_dir, corename, suffix,
            prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
            data_path = data_path,
            read_func = read_func, write_func = write_func,
            cache = cache, value_type = value_type)
        return obj
    # endregion - export and import dict

    # region - create instance
    @classmethod
    def get_default_file_type(cls, file:str):
        """
        Get the default file type for a given file. If the file type is not found in the `DEFAULT_FILE_TYPE` dictionary, None is returned.

        Parameters
        -----
        file (str): 
            The file name or path.

        Returns
        -----
        tuple: 
            A tuple containing the read function, write function, and value type associated with the file type.
            If the file type is not found in the DEFAULT_FILE_TYPE dictionary, None is returned for all three values.
        """
        if file.startswith('.'):
            suffix = file
        else:
            suffix = os.path.splitext(file)[1]
        if suffix in cls.DEFAULT_FILE_TYPE:
            read_func, write_func, value_type = cls.DEFAULT_FILE_TYPE[suffix]
        else:
            read_func, write_func, value_type = None, None, None

        return read_func, write_func, value_type
    
    @classmethod
    def create_temp(cls):
        """
        Creates a temporary instance of the class. This method sets the `GET_INSTANCE_STRATEGY` attribute of the class to `2` and returns the modified class.
        `GET_INSTANCE_STRATEGY` will automatically be reset to `0` after a new instance is created.

        * the new instance will not be registered.

        Returns
        ----
        cls: 
            The modified class with the `GET_INSTANCE_STRATEGY` attribute set to `2`.

        Examples
        ----
        >>> path = "./sub/core.txt"
        >>> cluster = FilesCluster("./top")
        >>> fh = FilesHandle.from_path(cluster, path) # OK
        >>> fh2 = FilesHandle.from_path(cluster, path) # OK, fh and fh2 are the same instance
        >>> fh3 = FilesHandle.create_temp().from_path(cluster, path) # OK, fh3 is a new instance, and it is not registered.
        """
        cls.GET_INSTANCE_STRATEGY = 2
        return cls

    @classmethod
    def create_new_and_cover(cls):
        """
        Creates a new instance of the class by setting the `GET_INSTANCE_STRATEGY` to `1`.
        `GET_INSTANCE_STRATEGY` will automatically be reset to `0` after a new instance is created.

        * the new instance will cover the old instance with the same identity_string.

        Returns
        ----
        cls: 
            The modified class with the `GET_INSTANCE_STRATEGY` attribute set to `1`.

        Examples
        -----
        >>> path = "./sub/core.txt"
        >>> cluster = FilesCluster("./top")
        >>> fh = FilesHandle.from_path(cluster, path) # OK
        >>> fh2 = FilesHandle.from_path(cluster, path) # OK, fh and fh2 are the same instance
        >>> fh3 = FilesHandle.create_new_and_cover().from_path(cluster, path) # OK, fh3 is a new instance, and it is registered and covers the old instance.
        >>> FilesHandle.has_instance(fh3)
        True
        >>> FilesHandle.has_instance(fh)
        False
        """
        cls.GET_INSTANCE_STRATEGY = 1
        return cls

    @staticmethod
    def _parse_path_to_name(cluster: "FilesCluster", path: T_MUITLSTR) -> T_MUITLSTR:
        """
        Parses the given path or list of paths to extract the filename(s) relative to the data_path of the cluster.

        see :link:`_parse_one_path_to_paras`
        see :link:`_parse_one_path_to_paras` and :link:`_default_parse_file_name`, 

        Parameters
        -----
        cluster (FilesCluster): 
            The FilesCluster object.
        path (str or list[str]): 
            The path(s) to be parsed.

        Returns
        -----
        T_MUITLSTR: 
            The filename(s) relative to the data_path of the cluster.

        Raises
        -----
        AssertionError: 
            If the path is not a subpath of the data_path.
        AssertionError: 
            If the path is not a string or a list of strings.
        """
        datapath = cluster.data_path
        if isinstance(path, str):
            assert is_subpath(path, datapath), "cannot create a fileshandle object which is not in the data_path"
            filename = os.path.relpath(path, datapath)
            return filename
        else:
            assert len(path) > 0, f"path must be a str or a list of str"
            assert all([datapath in p for p in path]), "cannot create a fileshandle object which is not in the data_path"
            filename = [os.path.relpath(p, datapath) for p in path]
            return filename

    @staticmethod
    def _default_parse_file_name(file_name:str, prefix_joiner:str, appendnames_joiner:str, _extract_corename_func:Optional[Callable] = None):
        """
        Parses the given file name and extracts corename

        see :link:`_parse_one_path_to_paras`, :link:`_parse_path_to_name`
        
        Parameters
        -----
        file_name (str): 
            The file name to be parsed.
        prefix_joiner: 
            The character or string used to join the prefix with the rest of the file name.
        appendnames_joiner: 
            The character or string used to join the core name with the append name.
        _extract_corename_func: 
            Optional function to extract the core name from the file name. Used for custom parsing.

        Returns
        ------
        tuple: A tuple containing the following components:
            - sub_dir (str): The sub-directory of the file.
            - corename (str): The core name of the file.
            - suffix (str): The file suffix.
            - prefix (str): The prefix of the file name.
            - appendname (str): The append name of the file.
            - _prefix_joiner: The prefix joiner used.
            - _appendnames_joiner: The append names joiner used.
        
        Examples
        -----
        >>> FilesHandle._default_parse_file_name("sub_dir/pre_corename-append.txt", "_", "-")
        >>> ('sub_dir', 'corename', '.txt', 'pre', 'append', '_', '-')
        """
        splitlist = file_name.split(os.sep, 2)
        if len(splitlist) == 1:
            sub_dir, file_name = "", splitlist[0]
        else:
            sub_dir, file_name = splitlist[0], splitlist[1]
        basename, suffix = os.path.splitext(file_name)

        if _extract_corename_func is not None:
            corename, prefix, appendname, _prefix_joiner, _appendnames_joiner = _extract_corename_func(basename)
        else:
            _prefix_joiner = prefix_joiner
            _appendnames_joiner = appendnames_joiner
            if prefix_joiner == "":
                prefix, rest = "", basename
            else:
                prefix, rest = basename.split(prefix_joiner, 1)

            if appendnames_joiner == "":
                corename, appendname = rest, ""
            else:
                corename, appendname = rest.split(appendnames_joiner, 1)

        return sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner
    
    @staticmethod
    def _parse_one_path_to_paras(cluster:"FilesCluster", path:str, prefix_joiner:str, appendnames_joiner:str, _extract_corename_func:Optional[Callable] = None):
        """
        Parses a single path to extract various parameters.
        
        see :link:`_parse_path_to_name` and :link:`_default_parse_file_name` for more details.

        Parameters
        -----
        cluster (FilesCluster): 
            The FilesCluster object.
        path (str): 
            The path to be parsed.
        prefix_joiner: 
            The prefix joiner.
        appendnames_joiner: 
            The append names joiner.
        _extract_corename_func:
            Optional function to extract the core name from the file name. Used for custom parsing.

        Returns
        -----
        tuple: 
            A tuple containing the following parameters:
            - sub_dir (str): The sub directory.
            - corename (str): The core name.
            - suffix (str): The suffix.
            - prefix (str): The prefix.
            - appendname (str): The append name.
            - prefix_joiner: The prefix joiner.
            - appendnames_joiner: The append names joiner.
        """
        name:str = FilesHandle._parse_path_to_name(cluster, path)
        prefix_joiner = "" if prefix_joiner is None else prefix_joiner
        appendnames_joiner = "" if appendnames_joiner is None else appendnames_joiner
        sub_dir, corename, suffix, prefix, appendname, prefix_joiner, appendnames_joiner =\
            FilesHandle._default_parse_file_name(name, prefix_joiner, appendnames_joiner, _extract_corename_func)
        return sub_dir, corename, suffix, prefix, appendname, prefix_joiner, appendnames_joiner

    @staticmethod
    def _compose_paras_to_name(corename:str, suffix:str = "", prefix:str = "", appendname:str = "", prefix_joiner:str = "", appendnames_joiner:str = ""):
        """
        Composes the parameters into a file name.

        Parameters
        -----
        sub_dir (str): 
            The sub directory.
        corename (str): 
            The core name.
        suffix (str): 
            The suffix.
        prefix (str): 
            The prefix.
        appendname (str): 
            The append name.
        prefix_joiner: 
            The prefix joiner.
        appendnames_joiner: 
            The append names joiner.

        Returns
        -----
        str: 
            The composed file name.
        """
        if len(prefix) > 0:
            prefix = prefix + prefix_joiner
        if len(appendname) > 0:
            appendname = appendnames_joiner + appendname
        if len(suffix) > 0 and suffix[0] != '.': 
            suffix = '.' + suffix
        return prefix + corename + appendname + suffix

    @classmethod
    def _default_compose_paras_to_name(cls, corename:str, suffix:str = ""):
        return cls._compose_paras_to_name(corename, suffix, cls.DEFAULT_PREFIX, cls.DEFAULT_APPENDNAMES, cls.DEFAULT_PREFIX_JOINER, cls.DEFAULT_APPENDNAMES_JOINER)

    @classmethod
    def from_path(cls, cluster:FCT, path:Union[str, list[str]], *,
                      prefix_joiner:Optional[str] = None, appendnames_joiner:Optional[str] = None, 
                      read_func:Optional[Callable] = None, write_func:Optional[Callable] = None, 
                      cache:Optional = None, value_type:Optional[type] = None,  #type: ignore
                      _extract_corename_func:Optional[Callable[[str], tuple[str, str, str, str, str]]] = None) -> Any:
        """
        Create a fileshandle object from a file path or a list of file paths.

        Parameters
        -----
        cluster (FCT): 
            The cluster object.
        path (Union[str, list[str]]): 
            The file path or a list of file paths.
        prefix_joiner (str, optional): 
            The prefix joiner. Defaults to None.
        appendnames_joiner (str, optional): 
            The append names joiner. Defaults to None.
        read_func (Callable, optional): 
            The read function. Defaults to None.
        write_func (Callable, optional): 
            The write function. Defaults to None.
        cache (Any, optional): 
            The cache object. Defaults to None.
        value_type (Callable, optional): 
            The value type. Defaults to None.
        _extract_corename_func (Callable[[str], tuple[str, str, str, str, str]], optional): 
            The extract core name function. Defaults to None.

        Returns
        ------
        Any: The fileshandle object.
        """
        datapath = cluster.data_path
        if isinstance(path, str):
            assert is_subpath(path, datapath), "cannot create a fileshandle object which is not in the data_path"
            filename = os.path.relpath(path, datapath)
        else:
            assert len(path) > 0, f"path must be a str or a list of str"
            assert all([datapath in p for p in path]), "cannot create a fileshandle object which is not in the data_path"
            filename = [os.path.relpath(p, datapath) for p in path]
        return cls.from_name(cluster, filename,
                                prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
                                read_func = read_func, write_func = write_func, 
                                cache = cache, value_type = value_type,  #type: ignore
                                _extract_corename_func = _extract_corename_func)

    @classmethod
    def from_name(cls, cluster:FCT, filename:Union[str, list[str]], *,
                  prefix_joiner:Optional[str] = None, appendnames_joiner:Optional[str] = None, 
                  read_func:Optional[Callable] = None, write_func:Optional[Callable] = None, 
                  cache = None, value_type:Optional[type] = None,  #type: ignore
                  _extract_corename_func:Optional[Callable[[str], tuple[str, str, str, str, str]]] = None):
        """
        Create an instance of the class using the given file name.

        Parameters
        ----------
        cluster : FCT
            The FilesCluster object.
        filename : Union[str, List[str]]
            The name of the file, can be a string or a list of strings.
        prefix_joiner : Optional[str], default: None
            The joiner between prefix and corename.
        appendnames_joiner : Optional[str], default: None
            The joiner between appendnames.
        read_func : Optional[Callable], default: None
            The function to read the file.
        write_func : Optional[Callable], default: None
            The function to write the file.
        cache : any, default: None
            The cache object.
        value_type : Optional[type], default: None
            The type of the cache object.
        _extract_corename_func : Optional[Callable[[str], tuple[str, str, str, str, str]]], default: None
            The function to extract the corename,
            the function should return a tuple:
            (sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner)

        Returns
        -------
        FilesHandle
            An instance of the class.

        Raises
        ------
        AssertionError
            If the filename is not a string or a list of strings.
            If the sub_dir, corename, suffix, prefix, or appendname are not the same for all filenames in the list.

        Examples
        --------
        >>> instance = FilesHandle.from_name(cluster, "filename.txt")
        >>> instance = FilesHandle.from_name(cluster, ["file_1.txt", "file_2.txt"], appendnames_joiner="_")
        """
        prefix_joiner       = get_with_priority(prefix_joiner,      cluster.DEFAULT_PREFIX_JOINER,      cls.DEFAULT_PREFIX_JOINER,      '')
        appendnames_joiner  = get_with_priority(appendnames_joiner, cluster.DEFAULT_APPENDNAMES_JOINER, cls.DEFAULT_APPENDNAMES_JOINER, '')

        if isinstance(filename, str):
            sub_dir, corename, suffix, prefix, appendname, prefix_joiner, appendnames_joiner =\
                cls._default_parse_file_name(filename, prefix_joiner, appendnames_joiner, _extract_corename_func)
            appendnames = list[str]([appendname])
        else:
            assert len(filename) > 0, f"path must be a str or a list of str"
            sub_dirs, corenames, suffixes, prefixes, appendnames = [], [], [], [], []
            for n in filename:
                sub_dir, corename, suffix, prefix, appendname, prefix_joiner, appendnames_joiner =\
                    cls._default_parse_file_name(n, prefix_joiner, appendnames_joiner, _extract_corename_func)
                sub_dirs.append(sub_dir)
                corenames.append(corename)
                suffixes.append(suffix)       
                prefixes.append(prefix)         
                appendnames.append(appendname)

            assert len(set(sub_dirs)) == 1, f"sub_dir must be the same"
            assert len(set(corenames)) == 1, f"corename must be the same"
            assert len(set(suffixes)) == 1, f"suffix must be the same"
            assert len(set(prefixes)) == 1, f"prefix must be the same"
            sub_dir = sub_dirs[0]
            corename = corenames[0]
            suffix = suffixes[0]
            prefix = prefixes[0]
        
        suffix              = get_with_priority(suffix, cluster.DEFAULT_SUFFIX, cls.DEFAULT_SUFFIX)
        prefix              = get_with_priority(prefix, cluster.DEFAULT_PREFIX, cls.DEFAULT_PREFIX)

        _read_func, _write_func, _value_type = cls.get_default_file_type(suffix)
        read_func           = get_with_priority(read_func,  cluster.DEFAULT_READ_FUNC,  cls.DEFAULT_READ_FUNC,  _read_func)
        write_func          = get_with_priority(write_func, cluster.DEFAULT_WRITE_FUNC, cls.DEFAULT_WRITE_FUNC, _write_func)
        value_type          = get_with_priority(value_type, cluster.DEFAULT_VALUE_TYPE, cls.DEFAULT_VALUE_TYPE, _value_type)

        data_path = cluster.data_path
        return cls(cluster, sub_dir, corename, suffix,
                   prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
                   data_path = data_path,
                   read_func = read_func, write_func = write_func, 
                   cache = cache, value_type = value_type)

    @classmethod
    def from_fileshandle(cls, cluster:FCT, file_handle:"FilesHandle", *,
                        sub_dir:Optional[str] = None, corename:Optional[str] = None, suffix:Optional[str] = None, #type: ignore
                        prefix:Optional[str] = None, appendnames:Union[str, list[str]] = None, prefix_joiner:Optional[str] = None, appendnames_joiner:Optional[str] = None, #type: ignore
                        read_func:Optional[Callable] = None, write_func:Optional[Callable] = None, 
                        cache = None, value_type:Optional[type] = None): #type: ignore
        """
        Create an instance of the class from another fileshandle object.

        Parameters
        ----------
        cluster : FCT
            The cluster object.
        file_handle : FilesHandle
            The file handle object.
        sub_dir : str, optional
            The sub-directory path. Defaults to None.
        corename : str, optional
            The core name. Defaults to None.
        suffix : str, optional
            The file suffix. Defaults to None.
        prefix : str, optional
            The file prefix. Defaults to None.
        appendnames : str or list[str], optional
            The append names. Defaults to None.
        prefix_joiner : str, optional
            The joiner for prefix and append names. Defaults to None.
        appendnames_joiner : str, optional
            The joiner for append names. Defaults to None.
        read_func : Callable, optional
            The read function. Defaults to None.
        write_func : Callable, optional
            The write function. Defaults to None.
        cache : optional
            The cache object. Defaults to None.
        value_type : type, optional
            The value type. Defaults to None.

        Returns
        -------
        FilesHandle
            An instance of the class.
        """
        sub_dir             = get_with_priority(sub_dir, file_handle.sub_dir)
        corename            = get_with_priority(corename, file_handle.corename)
        suffix              = get_with_priority(suffix, file_handle.suffix, cluster.DEFAULT_SUFFIX, cls.DEFAULT_SUFFIX)

        prefix              = get_with_priority(prefix, file_handle.prefix, cluster.DEFAULT_PREFIX, cls.DEFAULT_PREFIX)
        appendnames         = get_with_priority(file_handle._appendnames_obj.appendnames, appendnames)
        prefix_joiner       = get_with_priority(prefix_joiner,      file_handle._prefix_obj.joiner,      cluster.DEFAULT_PREFIX_JOINER,      cls.DEFAULT_PREFIX_JOINER,      '')
        appendnames_joiner  = get_with_priority(appendnames_joiner, file_handle._appendnames_obj.joiner, cluster.DEFAULT_APPENDNAMES_JOINER, cls.DEFAULT_APPENDNAMES_JOINER, '')

        _read_func, _write_func, _value_type = cls.get_default_file_type(suffix)
        read_func           = get_with_priority(read_func,  file_handle.read_func,  cluster.DEFAULT_READ_FUNC,  cls.DEFAULT_READ_FUNC,  _read_func)
        write_func          = get_with_priority(write_func, file_handle.write_func, cluster.DEFAULT_WRITE_FUNC, cls.DEFAULT_WRITE_FUNC, _write_func)
        value_type          = get_with_priority(value_type, file_handle.value_type, cluster.DEFAULT_VALUE_TYPE, cls.DEFAULT_VALUE_TYPE, _value_type)

        cache               = get_with_priority(cache, file_handle.cache)
        return cls(cluster, sub_dir, corename, suffix,
                    prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
                    data_path = "",
                    read_func = read_func, write_func = write_func, 
                    cache = cache, value_type = value_type)

    @classmethod
    def create_not_exist_fileshandle(cls, cluster: FCT):
        """
        Create a file handle for a non-existent file.

        Parameters
        ----------
        cluster (FCT): 
            The cluster object.

        Returns
        -------
        FilesHandle
            An instance of the class for the non-existent file.
        """
        return cls(cluster, "", "notexist", ".fhnotexist")
    # endregion - create instance

    # endregion init and create instance   

    # region make immutable
    def __hash__(self):
        '''
        hash(self.identity_string())
        '''
        return hash(self.identity_string()) # + id(self)
    
    def __eq__(self, o: object) -> bool:
        """
        compare the identity_string of two FilesHandle objects
        """
        if isinstance(o, FilesHandle):
            return hash(self) == hash(o)
        else:
            return False

    def immutable_attr_same_as(self, object: "FilesHandle") -> bool:
        """
        Check if the immutable attributes of the current object are the same as another FilesHandle object.

        Immutable attributes include
        ----
        * data_path
        * sub_dir
        * corename
        * suffix
        * prefix
        * appendnames
        * read_func
        * write_func
        * value_type

        Parameters
        ----------
        object : FilesHandle
            The FilesHandle object to compare with.

        Returns
        -----
        bool: 
            True if the immutable attributes are the same, False otherwise.
        """
        rlt = self.data_path == object.data_path and \
                self.sub_dir == object.sub_dir and \
                self.corename == object.corename and \
                self.suffix == object.suffix and \
                self.prefix == object.prefix and \
                self.appendnames == object.appendnames and \
                self.read_func == object.read_func and \
                self.write_func == object.write_func and \
                self.value_type == object.value_type
        return rlt

    def __setattr__(self, name, value):
        if hasattr(self, "_inited"):
            raise AttributeError(f"FilesHandle is immutable, you can't change its attribute")
        return super().__setattr__(name, value)
    # endregion make immutable

    # region property from cluster
    @property
    def valid(self):
        """
        `self.cluster` is not None
        """
        return self.cluster is not None

    @property
    def multi_files(self):
        """
        Code
        ----
        `self.cluster.MULTI_FILES`

        Note
        ----
        `multi_files` is decided by `self.cluster`. It indicates how to deal with the appendnames.

        * see :link:`FilesCluster.MULTI_FILES`, :link:`FilesHandle.get_name`, :link:`FilesHandle.get_path`
        """
        return self.cluster.MULTI_FILES

    @property
    def is_closed(self):
        """
        `self.cluster.closed`
        """
        return self.cluster.closed

    @property
    def is_readonly(self):
        """
        `self.cluster.readonly`
        """
        return self.cluster.readonly

    @property
    def overwrite_forbidden(self):
        """
        `self.cluster.readonly`
        """
        return self.cluster.overwrite_forbidden
    # endregion IO status    

    # region get immutable
    @property
    def prefix(self) -> str:
        return self._prefix_obj.prefix

    @property
    def prefix_with_joiner(self) -> str:
        return self._prefix_obj.get_with_joiner()

    @property
    def appendnames(self) -> Union[list[str], str]:
        appendnames = self._appendnames_obj.appendnames
        return _AppendNames.conditional_return(self.multi_files, appendnames)

    @property
    def appendnames_with_joiner(self) -> Union[list[str], str]:
        apwj = self._appendnames_obj.get_with_joiner()
        return _AppendNames.conditional_return(self.multi_files, apwj)

    @property
    def synced(self):
        """
        Whether the cache is synced with the file(s).

        Note
        -----

        Each FilesHandle instance corresponds to one (or more) files. 
        When performing IO operations, you can directly perform IO operations on the files, 
        or you can modify the `cache` of the FilesHandle instance in memory. 

        * See :link:`IO_CTRL_STRATEGY` for details.

        When only files are written, or only cache is modified, the data in file(s) and cache are asynchronous,
        and `synced` is False. To make `synced` be `True`, there are the following 3 methods:

        1. 
            Call `cache_to_file` or `file_to_cache` of the FilesCluster instance (hereinafter referred to as `cluster`)
            to which the FilesHandle instance belongs to manually synchronize data.

        2. 
            Inherit `FilesCluster` and `FilesCluster._write` (assumed the subclasses to be named `NewFC`, `NewFC._write`), 
            and rewrite the `W_SYNC` parameter of `NewFC._write` to True. This will modify the IO operation logic. 
            When NewFC.write is called, `file(s)` and `cache` are written/modified synchronously. 
            But please pay special attention: Frequent IO operations will consume a lot of resources, 
            especially when the file is large. Therefore, when a large amount of data needs to be written, 
            it is still recommended to modify `cache` first and then call `cache_to_file`.

        3. 
            (Not recommended) When writing data, call set_io_ctrl_strategy of `cluster`.
        """
        return self.cache_proxy.synced
    
    @property
    def value_type(self) -> type[VDMT]:
        return self.cache_proxy.value_type

    @property
    def full_directory(self):
        """
        the full directory of the file(s)
        """
        return self.get_dir()

    @property
    def all_file_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return False
        return all([os.path.exists(x) for x in paths])
    
    @property
    def all_file_not_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return True
        return all([not os.path.exists(x) for x in paths])
    
    @property
    def any_file_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return False
        return any([os.path.exists(x) for x in paths])
    
    @property
    def any_file_not_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return True
        return any([not os.path.exists(x) for x in paths])
    
    @property
    def has_cache(self):
        return (self.cache is not None)

    @property
    def empty(self):
        """
        Check if the object does not have cache and all files do not exist.

        Returns
        -----
        bool: True if the object is empty, False otherwise.
        """
        return not (self.any_file_exist or self.has_cache)

    @property
    def file_exist_status(self) -> list[bool]:
        """
        Returns
        -----
        list[bool]: 
            A list of boolean values indicating whether each file exists.
        """
        paths = self.get_path(get_list=True)
        return [os.path.exists(x) for x in paths]

    def get_name(self, get_list:bool=False) -> Union[list[str], str]:
        """
        Returns the name of the file(s).

        Parameters
        -----
        get_list (bool, optional): 
            If True, returns a list of names. Defaults to False.
            If `multi_files` is True, returns a list of names regardless of the value of `get_list`.

        Returns:
        -----
        Union[list[str], str]: 
            The name(s) of the file(s).
        """
        if self.multi_files or get_list:
            awu_list: list[str] = self._appendnames_obj.get_with_joiner() 
            return [self.prefix_with_joiner + self.corename + x + self.suffix for x in awu_list]
        else:
            awu: str = self.appendnames_with_joiner # type: ignore
            return self.prefix_with_joiner + self.corename + awu + self.suffix

    def get_dir(self):
        """
        the full directory of the file(s)
        """
        try:
            return self.__dir
        except AttributeError:
            self.__dir = os.path.join(self.data_path, self.sub_dir)
            return self.__dir

    def get_path(self, get_list=False) -> Union[list[str], str]:
        """
        Returns the path of the file(s) associated with the object.

        Parameters
        -----
        get_list (bool): 
            If True, returns a list of paths. Defaults to False.
            If `multi_files` is True, returns a list of paths regardless of the value of `get_list`.

        Returns
        -----
        Union[list[str], str]: 
            The path(s) of the file(s).
        """
        if self.multi_files or get_list:
            file_name: list[str] = self.get_name(get_list=True)  # type: ignore 
            dir_ = self.get_dir()
            return [os.path.join(dir_, x) for x in file_name]
        else:
            if not hasattr(self, "_path") or self._path is None:
                file_name: str = self.get_name(get_list=False)  # type: ignore 
                dir_ = self.get_dir()
                self._path = os.path.join(dir_, file_name)
            return self._path
            
    def get_key(self):
        """
        Get its key in the `self.cluster.MemoryData`
        """
        return self.cluster.MemoryData.query_value(self) # type: ignore

    # endregion path, dir, name

    # region cache
    @property
    def cache(self) -> Union[VDMT, None]:
        """
        Returns the cache of the instance.

        * If `is_readonly == True`, returns a shallow copy of the cache object. Else, returns the cache object.
        
        Returns
        -----
        Union[VDMT, None]: The cache object or a shallow copy of the cache object.
        """
        if not self.is_readonly:
            return self.cache_proxy.cache
        else:
            return copy.copy(self.cache_proxy.cache)

    def set_cache(self, cache):
        """
        Set the cache for the IOAbstract object.

        Parameters
        -----
        cache (object): The cache object to be set.

        Note
        -----
        If `is_readonly == True` or `is_closed == True`, the cache will not be set.

        Returns
        -----
        None
        """
        if not self.is_closed and not self.is_readonly:
            if cache is None:
                self.erase_cache()
            else:
                self.cache_proxy.cache = cache

    def erase_cache(self):
        """
        Erase the cache.

        Note
        -----
        If `self.is_readonly == True` or `self.is_closed == True`, the cache will not be erased.
        """
        if not self.is_closed and not self.is_readonly:
            self.cache_proxy.cache = None    

    def _unsafe_get_cache(self) -> Union[VDMT, None]:
        '''
        Get cache without any check

        * NOT recommended, use with caution
        '''
        return self.cache_proxy.cache

    def read(self) -> Union[VDMT, list[VDMT], None]:
        """
        Reads data from the specified path using the read function.
        If `self.read_func` is `None`, returns `None`.

        Returns
        -----
        Union[VDMT, list[VDMT], None]: 
            The read data, or None if the path does not exist.
        """
        if self.read_func is not None:
            path = self.get_path()
            if isinstance(path, str):
                if os.path.exists(path):
                    return self.read_func(path)
            else:
                rlts = []
                for p in path:
                    if os.path.exists(p):
                        rlts.append(self.read_func(p))
                    else:
                        rlts.append(None)
                return rlts
        return None

    def _set_synced(self, synced:bool = True):
        """
        Set the synchronization status of the cache.

        Parameters
        -----
        synced (bool): The synchronization status. Default is True.

        Note
        -----
        DO NOT call this function directly, use `cache_to_file` or `file_to_cache` instead.

        Returns:
        None
        """
        if self.has_cache and self.all_file_exist:
            pass
        elif not self.has_cache and self.all_file_not_exist:
            synced = True
        else:
            synced = False
        self.cache_proxy.synced = synced
    # endregion cache

    # region other
    def clear_notfound(self):
        """
        clear those files which are not found
        """
        new_appendnames = []
        for i, e in enumerate(self.file_exist_status):
            if e:
                new_appendnames.append(self._appendnames_obj.appendnames[i])
        return self.from_fileshandle(self.cluster, self, appendnames = new_appendnames)

    def __repr__(self) -> str:
        if len(self._appendnames_obj.appendnames) == 1:
            string = f"FilesHandle({self.get_path()})"
        else:
            paths = self.get_path(get_list=True)
            string = f"FilesHandle({paths[0]}) + {len(paths) - 1} files,"
        string += f" file exist: {self.file_exist_status}, has cache: {self.has_cache}, synced: {self.synced}"
        return string
    # endregion other

class FhBinDict(BinDict[int, FHT], Generic[FHT]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_data:Optional[dict[int, dict]] = None
        self.fileshandle_init_func:Callable[[dict], FHT] = None

    def __activate(self, key):
        self[key] = self.fileshandle_init_func(self._lazy_data[key])
        self._lazy_data.__delitem__(key) # can only use once

    def __getitem__(self, key:int) -> FHT:
        return self.get(key)

    def activate(self, keys:Union[int, list[int]] = None):
        if keys is None:
            keys = list(set(self._lazy_data.keys()).difference(self.keys()))
        else:
            if isinstance(keys, int):
                keys = [keys]
            assert all([isinstance(k, int) for k in keys]), f"keys must be int or list of int, not {keys}"
            keys = list(set(keys).union(self._lazy_data.keys()).difference(self.keys()))
        for key in keys:
            self.__activate(key)

    def get(self, key, default = None):
        fh = dict.get(self, key, default)
        if fh is None:
            if key in self._lazy_data:
                self.__activate(key)
                return self[key]
            else:
                raise KeyError(f"key {key} not found")
        else:
            return fh
        
    def update(self, *args: dict[int, FHT], **kwargs):
        super().update(*args, **kwargs)
        for arg in args:
            if isinstance(arg, FhBinDict):
                # update _lazy_data
                if arg._lazy_data is not None:
                    if self._lazy_data is None:
                        self._lazy_data = arg._lazy_data.copy()
                    else:
                        self._lazy_data.update(arg._lazy_data)
                # update fileshandle_init_func
                if self.fileshandle_init_func is None:
                    self.fileshandle_init_func = arg.fileshandle_init_func
                else:
                    assert self.fileshandle_init_func == arg.fileshandle_init_func, \
                        f"fileshandle_init_func must be the same, not {self.fileshandle_init_func} and {arg.fileshandle_init_func}"

    def has(self, key):
        return key in self or (self._lazy_data is not None and key in self._lazy_data)

class Node(Generic[NODE]):
    """
    A class representing a node in a tree structure.
    """

    PARENT_TYPE: Optional[type["Node"]] = None
    CHILD_TYPE: Optional[type["Node"]] = None

    def __init__(self) -> None:
        """
        Initializes a Node object.
        """
        self.__parent: list[Optional[NODE]] = [None]
        self.__children: list[NODE] = []

        self.linked_with_children = True
        self.follow_parent = True

    @staticmethod
    def __backward(obj: "Node", funcname, *args, **kwargs):
        """
        Helper method to propagate a function call to the parent node and its ancestors in a backward direction.
        """
        if obj.parent is not None and obj.linked_with_children and hasattr(obj.parent, funcname):
            getattr(obj.parent, funcname)(*args, **kwargs)

    @staticmethod
    def __forward(obj: "Node", funcname, *args, **kwargs):
        """
        Helper method to propagate a function call to the child nodes in a forward direction.
        """
        for child in obj._children:
            child:Node
            if child.follow_parent and hasattr(child, funcname):
                getattr(child, funcname)(*args, **kwargs)

    @staticmethod
    def downward_preorder_propagate(func):
        '''
        Decorator to propagate a function call to the leaves of the node in a downward preorder traversal.
        '''
        @functools.wraps(func)
        def forward_propagate_wrapper(obj: Node[NODE], *args, only_this = False, **kwargs):
            funcname = get_func_name(obj, func)
            func(obj, *args, **kwargs)
            if funcname is not None and not only_this:
                Node.__forward(obj, funcname, *args, **kwargs)
            else:
                warnings.warn(f"can't find function {func} in {obj}, propagate ends")
            
        return forward_propagate_wrapper

    @staticmethod
    def downward_postorder_propagate(func):
        '''
        Decorator to propagate a function call to the leaves of the node in a downward postorder traversal.
        '''
        @functools.wraps(func)
        def forward_propagate_wrapper(obj: Node[NODE], *args, only_this = False, **kwargs):
            funcname = get_func_name(obj, func)
            if funcname is not None and not only_this:
                Node.__forward(obj, funcname, *args, **kwargs)
            else:
                warnings.warn(f"can't find function {func} in {obj}, propagate ends")
            func(obj, *args, **kwargs)
            
        return forward_propagate_wrapper

    @staticmethod
    def upward_preorder_propagate(func):
        '''
        Decorator to propagate a function call to the ancestors of the node in a backtracking preorder traversal.
        '''
        @functools.wraps(func)
        def upward_preorder_propagate_wrapper(obj: Node[NODE], *args, only_this = False, **kwargs):
            funcname = get_func_name(obj, func)
            if funcname is not None and not only_this:
                Node.__backward(obj, funcname, *args, **kwargs)
            else:
                warnings.warn(f"can't find function {func} in {obj}, propagate ends")
            func(obj, *args, **kwargs)
        return upward_preorder_propagate_wrapper
    
    @staticmethod
    def upward_postorder_propagate(func):
        '''
        Decorator to propagate a function call to the ancestors of the node in a backtracking postorder traversal.
        '''
        def upward_postorder_propagate_wrapper(obj: Node[NODE], *args, only_this = False, **kwargs):
            funcname = get_func_name(obj, func)
            func(obj, *args, **kwargs)
            if funcname is not None and not only_this:
                Node.__backward(obj, funcname, *args, **kwargs)
            else:
                warnings.warn(f"can't find function {func} in {obj}, propagate ends")
        return upward_postorder_propagate_wrapper

    @property
    def parent(self) -> Union[NODE, None]:
        """
        Get the parent node of the current node.
        """
        return self.__parent[0]
    
    @property
    def _children(self) -> tuple[NODE]:
        """
        Get the children nodes of the current node.
        """
        return tuple(self.__children)

    def _set_parent(self, parent):
        """
        Set the parent node of the current node.
        """
        self.__parent[0] = parent

    def _add_child(self, child_node: "Node"):
        """
        Add a child node to the current node.
        """
        child_type = self.CHILD_TYPE if self.CHILD_TYPE is not None else Node
        assert isinstance(child_node, child_type), f"child_node must be Node, not {type(child_node)}"
        if child_node in self._children:
            return
        child_node._set_parent(self)
        self.__children.append(child_node) # type: ignore

    def _remove_child(self, child_node: "Node"):
        """
        Parameters
        -----
        child_node: Node

        Remove a child node from the current node.
        """
        child_type = self.CHILD_TYPE if self.CHILD_TYPE is not None else Node
        assert isinstance(child_node, child_type), f"child_node must be Node, not {type(child_node)}"
        if child_node in self._children:
            child_node._set_parent(None)
            self.__children.remove(child_node) # type: ignore

    def _move_node(self, new_parent: NODE):
        """
        Move the current node to a new parent node.
        """
        parent_type = self.PARENT_TYPE if self.PARENT_TYPE is not None else Node
        assert isinstance(new_parent, parent_type) or new_parent is None, f"new_parent must be Node, not {type(new_parent)}"
        if self.parent is not None:
            self.parent._remove_child(self)
        if new_parent is not None:
            new_parent._add_child(self)

class IO_CTRL_STRATEGY():
    """
    Enum representing different IO control strategies.

    Note
    -----
    there are 3 control flags for `DataMapping`: `cache_priority`, `strict_priority_mode`, `write_synchronous`

    * `cache_priority`

        IO operations have 2 destinations: cache and file(s). Cache is in memory, and file(s) is on disk.
        You can read from(write to) cache or file(s). They each have their own advantages under different conditions.
        
        When the file is large, such as an image or a video, it is recommended to read from(write to) the file(s) directly. 
        Because reading from(writing to) the file(s) directly will not consume memory.
        `cache_priority` is recommended to be `False` in this case.
            
        When the file is small, such as a configuration file, it is recommended to read from(write to) the cache.
        Because reading from(writing to) the cache is faster than reading from(writing to) the file(s). 
        Or if the file contains a large amounts of pieces of data, like a json/xml file, 
        If the file is written to the disk when each piece of data is written, a lot of resources will be consumed.
        In this case, it is recommended to write to the cache first, and then write to the file(s) in batches.
        `cache_priority` is recommended to be `True` in this case.

    * `strict_priority_mode`

        When `strict_priority_mode` is `False`. If the one set as priority fails to perform a read (or write) operation, 
        it will try to perform a read (or write) operation on the other one. If failed again, it will raise an exception.

        When `strict_priority_mode` is `True`. When the priority fails, it will raise an exception without trying the other one.

    * `write_synchronous`

        When `write_synchronous` is `False`. If the cache is modified, it will not be written to the file immediately. 
        The same goes for files.

        When `write_synchronous` is `True`. If the cache is modified, it will be written to the file immediately. The same goes for files.

    Attributes
    ------
    CACHE_PRIORITY: 
        Set Cache as Priority
    FILE_PRIORITY: 
        Set File as Priority
    STRICT_PRIORITY_MODE: 
        Set `strict_priority_mode`
    NON_STRICT_PRIORITY_MODE: 
        Clear `strict_priority_mode`
    WRITE_SYNCHRONOUS: 
        Set `write_synchronous`
    WRITE_ASYNCHRONOUS: 
        Clear `write_synchronous`

    Methods
    ------
    get_ctrl_flag(strategy: int) -> Tuple[bool, bool, bool]: 
        Get control flags based on the strategy.
    get_ctrl_strategy(cache_priority: bool, strict_priority_mode: bool, write_synchronous: bool) -> int: 
        Get strategy based on control flags.
    """

    FILE_IDPNDT             = 0
    """FILE_PROIRITY, NON_STRICT_PRIORITY_MODE, WRITE_ASYNCHRONOUS"""
    FILE_SYNC               = 1
    """FILE_PROIRITY, NON_STRICT_PRIORITY_MODE, WRITE_SYNCHRONOUS"""
    FILE_STRICK_IDPNDT      = 2
    """FILE_PROIRITY, STRICT_PRIORITY_MODE, WRITE_ASYNCHRONOUS"""
    FILE_STRICK_SYNC        = 3
    """FILE_PROIRITY, STRICT_PRIORITY_MODE, WRITE_SYNCHRONOUS"""
    CACHE_IDPNDT            = 4
    """CACHE_PRIORITY, NON_STRICT_PRIORITY_MODE, WRITE_ASYNCHRONOUS"""
    CACHE_SYNC              = 5
    """CACHE_PRIORITY, NON_STRICT_PRIORITY_MODE, WRITE_SYNCHRONOUS"""
    CACHE_STRICK_IDPNDT     = 6
    """CACHE_PRIORITY, STRICT_PRIORITY_MODE, WRITE_ASYNCHRONOUS"""
    CACHE_STRICK_SYNC       = 7
    """CACHE_PRIORITY, STRICT_PRIORITY_MODE, WRITE_SYNCHRONOUS"""

    CACHE_PRIORITY          = 4
    FILE_PROIRITY           = 0

    STRICT_PRIORITY_MODE    = 2
    NON_STRICT_PRIORITY_MODE= 0

    WRITE_SYNCHRONOUS       = 1
    WRITE_ASYNCHRONOUS      = 0

    @classmethod
    def get_ctrl_flag(cls, strategy:int):
        '''
        return cache_priority, strict_priority_mode, write_synchronous

        Examples
        -----
        >>> IO_CTRL_STRATEGY.get_ctrl_flag(IO_CTRL_STRATEGY.CACHE_IDPNDT)
        (True, False, False)
        >>> IO_CTRL_STRATEGY.get_ctrl_flag(IO_CTRL_STRATEGY.CACHE_PRIORITY | IO_CTRL_STRATEGY.STRICT_PRIORITY_MODE | IO_CTRL_STRATEGY.WRITE_SYNCHRONOUS)
        (True, True, True)
        '''
        if strategy == cls.CACHE_IDPNDT:
            return True, False, False
        elif strategy == cls.CACHE_STRICK_IDPNDT:
            return True, True, False
        elif strategy == cls.FILE_IDPNDT:
            return False, False, False
        elif strategy == cls.FILE_STRICK_IDPNDT:
            return False, True, False
        elif strategy == cls.CACHE_SYNC:
            return True, False, True
        elif strategy == cls.FILE_SYNC:
            return False, False, True
        elif strategy == cls.CACHE_STRICK_SYNC:
            return True, True, True
        elif strategy == cls.FILE_STRICK_SYNC:
            return False, True, True
        else:
            raise ValueError(f"strategy must be IO_CTRL_STRATEGY, not {strategy}")
        
    @classmethod
    def get_ctrl_strategy(cls, cache_priority:bool, strict_priority_mode:bool, write_synchronous:bool):
        """
        Get the control strategy based on the given parameters.

        Parameters
        -----
        cache_priority :bool
            The priority of cache.
        strict_priority_mode :bool
            The mode of strict priority.
        write_synchronous :bool
            The mode of write synchronization.

        Returns
        -----
        int: 
            The control strategy calculated based on the given parameters.
        """
        return int(cache_priority) * 4 + int(strict_priority_mode) * 2 + int(write_synchronous)

class DataMapping(IOStatusManager, _RegisterInstance["DataMapping"], Node["DataMapping"], ABC, Generic[_VT, DMT, VDMT]):
    """
    An abstract class representing bidirectional mapping between data and files.

    Attributes
    ------
    MEMORY_DATA_FILE (str): 
        
        The default filename for storing memory data.
    MEMORY_DATA_TYPE (type): 
        The default data type for memory data storage (FhBinDict).
    KEY_TYPE (type): 
        The type of keys used in the mapping (int).
    FILESHANDLE_TYPE (type): 
        The type of files handle used (FilesHandle).

    Methods
    -----
    try_open(): 
        Tries to open the data mapping for operation.
    has_not_inited(): 
        Checks if the data mapping has not been initialized.
    _clear_empty_dir(): 
        Clears empty directories in the data mapping.
    _rebuild_done(): 
        Handles actions after the rebuild operation is completed.
    init_identity(parent_like, mapping_name, *args, flag_name="", **kwargs): 
        Initializes the identity of the data mapping.
    get_child(name): 
        Gets the child node with the specified name.
    save_memory_func(path, data): 
        Serializes and saves the memory data to a file.
    load_memory_func(path): 
        Loads and deserializes memory data from a file.
    register(identity_string, obj): 
        Registers a data mapping instance with a unique identity string.
    get_instance(identity_string, _obj): 
        Retrieves a registered data mapping instance based on its identity string.
    identity_string(): 
        Returns the identity string of the data mapping.
    identity_name(): 
        Returns the identity name of the data mapping.
    gen_identity_string(): 
        Generates a new identity string for the data mapping.
    gen_identity_name(): 
        Generates a new identity name for the data mapping.
    make_path(): 
        Creates the necessary directory structure for the data mapping.
    set_parent(parent): 
        Sets the parent node of the data mapping.
    open_hook(): 
        Hook method called during the open operation.
    stop_writing_hook(): 
        Hook method called when writing is stopped.
    get_writing_mark_file(): 
        Returns the path to the writing mark file.
    close(closed=True): 
        Closes the data mapping.
    open(opened=True): 
        Opens the data mapping.
    set_readonly(readonly=True): 
        Sets the data mapping to readonly mode.
    set_writable(writable=True): 
        Sets the data mapping to writable mode.
    stop_writing(stop_writing=True): 
        Stops writing to the data mapping.
    start_writing(start_writing=True): 
        Starts writing to the data mapping.
    set_overwrite_forbidden(overwrite_forbidden=True): 
        Forbids overwriting in the data mapping.
    set_overwrite_allowed(overwrite_allowed=True): 
        Allows overwriting in the data mapping.
    remove_mark(): 
        Removes the mark file associated with the data mapping.
    _set_MemoryData_modified(): 
        Sets the flag indicating modification in memory data.
    _reset_MemoryData_modified(): 
        Resets the flag indicating modification in memory data.
    MemoryData_modified: 
        Property indicating whether memory data has been modified.
    MemoryData: 
        Property representing the memory data of the mapping.
    MemoryData_path: 
        Property representing the path to the memory data file.
    pickle_MemoryData_path: 
        Property representing the path to the pickled memory data file.
    data_path: 
        Property representing the data path of the mapping.
    save_preprecess(): 
        Hook method for preprocessing data before saving.
    load_postprocess(data): 
        Hook method for post-processing data after loading.
    remove_memory_data_file(): 
        Removes the memory data file.
    rebuild(force=False): 
        Abstract method for rebuilding the data mapping.
    merge_MemoryData(MemoryData: dict): 
        Abstract method for merging memory data.
    sort(): 
        Sorts the memory data.
    num: 
        Property representing the number of items in the mapping.
    i_upper: 
        Property representing the upper limit of indices in the mapping.
    next_valid_i: 
        Property representing the next valid index in the mapping.
    continuous: 
        Property indicating whether the indices in the mapping are continuous.
    set_io_ctrl_strategy(strategy): 
        Sets the I/O control strategy based on the provided strategy.
    get_io_ctrl_strategy(): 
        Gets the I/O control strategy.
    read(src): 
        Abstract method for reading data from the mapping.
    write(dst, value, force=False, **other_paras): 
        Abstract method for writing data to the mapping.
    modify_key(src, dst, force=False, **other_paras): 
        Abstract method for modifying keys in the mapping.
    remove(dst, force=False, **other_paras): 
        Abstract method for removing data from the mapping.
    merge_from(src: DMT, force=False): 
        Abstract method for merging data from another mapping.
    copy_from(src: DMT, cover=False, force=False): 
        Abstract method for copying data from another mapping.
    append(value, force=False, **other_paras): 
        Appends a value to the mapping.
    clear(force=False): 
        Clears the data mapping.
    make_continuous(force=False): 
        Makes the indices in the mapping continuous.
    cache_to_file(data_i=None, force=False, **other_paras): 
        Caches data to files.
    file_to_cache(data_i=None, save=True, force=False, **other_paras): 
        Reads data from files and updates the cache.
    clear_files(force=False): 
        Clears the files associated with the data mapping.
    clear_cache(force=False): 
        Clears the cache associated with the data mapping.
    _set_unfinished_operation(unfinished_operation): 
        Sets the unfinished operation flag.
    _get_unfinished_operation(): 
        Gets the unfinished operation flag.
    _set_last_write_unfinished(): 
        Sets the flag indicating the last write operation was unfinished.
    _reset_last_write_unfinished(): 
        Resets the flag indicating the last write operation was unfinished.
    _get_last_write_unfinished(): 
        Gets the flag indicating the last write operation was unfinished.
    choose_unfinished_operation(): 
        Asks the user to choose an operation to continue when an operation is unfinished.
    process_unfinished(): 
        Processes unfinished operations in the data mapping.

    """

    _DMT = TypeVar('_DMT', bound="DataMapping")
    _VDMT = TypeVar('_VDMT')

    MEMORY_DATA_FILE = ".datamap"
    MEMORY_DATA_TYPE = FhBinDict

    KEY_TYPE = int
    FILESHANDLE_TYPE:type[FilesHandle] = FilesHandle

    load_memory_func:Callable[[str], dict]         = deserialize_object
    save_memory_func:Callable[[str, dict], None]   = serialize_object

    # region DataMapping override for _RegisterInstance ##

    INDENTITY_PARA_NAMES = ["_top_directory", "mapping_name", "flag_name"]
    
    def __new__(cls, dataset_node:Union[str, "DatasetNode"], mapping_name: str = "", *args, flag_name = "", **kwargs):
        return super().__new__(cls, dataset_node, mapping_name, *args, flag_name = flag_name, **kwargs)
    
    @classmethod
    def register(cls, identity_string, obj:"DataMapping"):
        """
        see :link:`RegisterInstance.register`
        """
        for other_obj in cls._registry.values():
            if isinstance(other_obj, DataMapping) and obj.MemoryData_path == other_obj.MemoryData_path:
                raise ValueError(f"{obj} and {other_obj} has the same MemoryData_path, try set 'flag_name' to distinguish them")
        cls._registry[identity_string] = obj

    @classmethod
    def get_instance(cls, identity_string, _obj):
        """
        see :link:`RegisterInstance.get_instance`
        """
        obj = cls._registry[identity_string]
        return obj

    def identity_string(self, regen = False):
        """
        see :link:`RegisterInstance.identity_string`
        """
        return _RegisterInstance.identity_string(self, regen)

    def identity_name(self):
        """
        see :link:`RegisterInstance.identity_name`
        """
        return _RegisterInstance.identity_name(self)

    def gen_identity_string(self):
        """
        see :link:`RegisterInstance.gen_identity_string`
        """
        return f"{self.__class__.__name__}:{self.data_path}|{self.flag_name}"

    def gen_identity_name(self):
        """
        see :link:`RegisterInstance.gen_identity_name`

        Note
        ----
        if `self.flag_name` is not empty, return `self.flag_name`, otherwise return `self.mapping_name`
        """
        if self.flag_name != "":
            return self.flag_name
        else:
            return self.mapping_name

    def init_identity(self, parent_like:Union[str, "DatasetNode"], mapping_name: str, *args, flag_name = "", **kwargs):
        '''
        Initializes the identity of the data mapping.

        Parameters
        -----
        parent_like: 
            A string or DatasetNode representing the parent of the data mapping.
        mapping_name: 
            A string representing the name of the data mapping.
        flag_name: 
            A string representing the flag name.

        Raises
        -----
        TypeError: 
            If `parent_like` is not a string or DatasetNode.

        * see :link:`DataMapping.__init__` for more information.
        '''
        if isinstance(parent_like, str):
            self._top_directory = os.path.join(parent_like, mapping_name)
        elif isinstance(parent_like, DatasetNode):
            self._top_directory = os.path.join(parent_like.data_path, mapping_name)
            # assert is_subpath(self._top_directory, parent_like.top_directory), f"{self.top_directory} is not inside {parent_like.top_directory}"
        else:
            raise TypeError(f"parent_like must be str or DatasetNode, not {type(parent_like)}")

        self.mapping_name = mapping_name       
        self.flag_name = flag_name 
    # endregion DataMapping override ##

    # region DataMapping override for Node ####

    @property
    def children_names(self):
        """
        get the identity names of the children nodes
        """
        return [x.identity_name() for x in self._children]

    def get_child(self, name:str):
        """
        get the child node with the specified name
        """
        for child in self._children:
            if child.identity_name() == name:
                return child
        raise KeyError(f"{name} not in {self}")
    
    # endregion DataMapping override for Node ####

    # region DataMapping ####
    def __init_subclass__(cls, **kwargs):
        '''
        Adds hooks to some methods.
        '''
        ### __init__ ###
        cls._orig_init = cls.__init__ # type: ignore
        cls.__init__ = method_exit_hook_decorator(cls, cls.__init__, cls.try_open, cls.has_not_inited)
        ### clear ###
        cls.clear = method_exit_hook_decorator(cls, cls.clear, cls._clear_empty_dir)
        ### rebuild ###
        cls.rebuild = method_exit_hook_decorator(cls, cls.rebuild, cls._rebuild_done)
        super().__init_subclass__(**kwargs)

    def __init__(self, top_directory:Union[str, "DatasetNode"], mapping_name: str = "", *args, flag_name:str = "", **kwargs) -> None:
        '''
        Initializes the data mapping with the provided top_directory, mapping_name, and registration flag.

        Parameters
        ----
        top_directory: 
            A string or DatasetNode representing the top directory of the data mapping.
        mapping_name: 
            A string representing the name of the data mapping.
            It's used as sub directory name under `top_directory`.for `FilesCluster`(the subclass of `DataMapping`)
        flag_name: 
            A string representing the registration flag. 
            It's used to distinguish different instances of the same data mapping.
        '''
        IOStatusManager.__init__(self)
        Node.__init__(self)
        Generic.__init__(self)

        self.set_parent(top_directory)

        self._MemoryData:dict[int, _VT] = self.load_postprocess({})

        self.cache_priority     = True
        self.strict_priority_mode    = False
        self.write_synchronous  = False
        self._MemoryData_modified = True

        self.__unfinished_operation = 0
        self.__last_write_unfinished = False

        if self.mark_exist():
            self._set_last_write_unfinished()

    def try_open(self):
        '''
        Tries to open the data mapping for operation.
        If the data path exists, it opens the data mapping.
        Otherwise, it closes the data mapping.
        '''
        if os.path.exists(self.data_path):
            self.open()  # Opens the cluster for operation.
        else:
            self.close()
        if DEBUG:
            print(f"{self.identity_string()} inited")
        if self.parent is None and self._get_last_write_unfinished():
            if isinstance(self, DatasetNode):
                if len(self.elem_clusters) == len(self.MemoryData.col_names):
                    self.process_unfinished()
            else:
                self.process_unfinished()

        # if self.mark_exist():
        #     self.process_unfinished()
        self._inited = True

    def has_not_inited(self):
        '''
        Checks if the data mapping has not been initialized.

        Returns
        ----
        bool: 
            True if the data mapping has not been initialized, False otherwise.
        '''
        try:
            self._inited
        except AttributeError:
            return True
        else:
            return False

    def _clear_empty_dir(self):
        '''
        remove empty directories in the data mapping.
        '''
        for r, d, f in os.walk(self.data_path):
            for dir_ in d:
                if len(os.listdir(os.path.join(r, dir_))) == 0:
                    os.rmdir(os.path.join(r, dir_))
    
    def _rebuild_done(self):
        self._reset_MemoryData_modified()
        self.relugar_MemoryData()
        self.save(True)

    @property
    def top_directory(self):
        return self._top_directory

    @top_directory.setter
    def top_directory(self, value):
        return None

    def make_path(self):
        """
        Create the directory path specified by `self.data_path` if it does not exist.
        If `self.data_path` ends with a file extension, create the parent directory and an empty file.
        If `self.data_path` does not end with a file extension, create the directory recursively.
        """
        if not os.path.exists(self.data_path):
            if '.' in os.path.basename(self.data_path):
                dir_ = os.path.dirname(self.data_path)
                os.makedirs(dir_, exist_ok=True)
                with open(self.data_path, 'w'):
                    pass
            else:
                os.makedirs(self.data_path, exist_ok=True)
    
    def set_parent(self, parent:"DataMapping"):
        if isinstance(parent, DataMapping) or parent is None:
            self._move_node(parent)
    # endregion DataMapping new methods ####

    # region DataMapping override for IOStatusManager####
    def open_hook(self):
        self.make_path()
        self.load()
        self._reset_MemoryData_modified()
        
    def stop_writing_hook(self):
        self.sort()
        self.save()
        if self.parent is not None:
            self.parent.save()

    def get_writing_mark_file(self):
        # return os.path.join(self.top_directory, self.mapping_name, self.WRITING_MARK)
        return os.path.join(self.top_directory, self.WRITING_MARK)

    @Node.downward_preorder_propagate
    def close(self, closed = True):
        """
        see :link:`IOStatusManager.close`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().close(closed)    

    @Node.downward_preorder_propagate
    def open(self, opened = True):
        """
        see :link:`IOStatusManager.open`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().open(opened)

    @Node.downward_preorder_propagate
    def set_readonly(self, readonly = True):
        """
        see :link:`IOStatusManager.set_readonly`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().set_readonly(readonly)

    @Node.downward_preorder_propagate
    def set_writable(self, writable = True):
        """
        see :link:`IOStatusManager.set_writable`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().set_writable(writable)

    @Node.downward_preorder_propagate
    def stop_writing(self, stop_writing = True):
        """
        see :link:`IOStatusManager.stop_writing`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().stop_writing(stop_writing)

    @Node.downward_preorder_propagate
    def start_writing(self, start_writing = True):
        """
        see :link:`IOStatusManager.start_writing`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().start_writing(start_writing)

    @Node.downward_preorder_propagate
    def set_overwrite_forbidden(self, overwrite_forbidden = True):
        """
        see :link:`IOStatusManager.set_overwrite_forbidden`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().set_overwrite_forbidden(overwrite_forbidden)

    @Node.downward_preorder_propagate
    def set_overwrite_allowed(self, overwrite_allowed = True):
        """
        see :link:`IOStatusManager.set_overwrite_allowed`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().set_overwrite_allowed(overwrite_allowed)

    @Node.downward_preorder_propagate
    def remove_mark(self):
        """
        see :link:`IOStatusManager.remove_mark`

        this operation will be propagated to all children nodes in preorder traversal
        """
        super().remove_mark()
    # endregion DataMapping override for IOStatusManager####
    
    # region Memorydata operation ####
    @Node.upward_preorder_propagate
    def _set_MemoryData_modified(self):
        """
        this operation will be propagated to all parent nodes in preorder traversal.

        Examples
        ------
        - dm
          - dm_child1
          - dm_child2
            - dm_child3

        >>> dm_child3._set_MemoryData_modified()
        >>> dm_child2.MemoryData_modified
        True
        >>> dm.MemoryData_modified
        True
        >>> dm_child1.MemoryData_modified
        False        
        """
        self._MemoryData_modified = True

    def _reset_MemoryData_modified(self):
        self._MemoryData_modified = False
    
    @property
    def MemoryData_modified(self):
        """
        DO NOT use `_MemoryData_modified` directly, use this property instead.
        """
        return self._MemoryData_modified

    @property
    def MemoryData(self):
        """
        DO NOT use `_MemoryData` directly, use this property instead.
        """
        return self._MemoryData
    
    @property
    def MemoryData_path(self):
        # return os.path.join(self.top_directory, self.mapping_name, self.flag_name + self.MEMORY_DATA_FILE)
        return os.path.join(self.top_directory, self.flag_name + self.MEMORY_DATA_FILE)
    
    @property
    def pickle_MemoryData_path(self):
        return os.path.join(self.top_directory, self.flag_name + ".datamapping_pickle")

    @property
    def data_path(self):
        # return os.path.join(self.top_directory, self.mapping_name).replace('\\', '/')
        """
        top directory of the data mapping
        """
        return os.path.join(self.top_directory).replace('\\', '/')

    def save_preprecess(self):
        """
        the opertions to do before saving memory data

        * recommend to override this method in subclass
        """
        return self.MemoryData

    def load_postprocess(self, data):
        """
        the opertions to do after loading memory data

        * recommend to override this method in subclass
        """
        return data

    def remove_memory_data_file(self):
        if os.path.exists(self.MemoryData_path):
            os.remove(self.MemoryData_path)

    @abstractmethod
    def rebuild(self, force=False):
        """
        Rebuilds `self.MemoryData` by the files.
        if `MemoryData_path` does not exist or `load_postprocess` fails, this method will be called.

        * must override this method in subclass
        """
        pass

    def save(self, force = False):
        """
        save `self.MemoryData` to file
        """
        if self.MemoryData_modified or force:
            self.make_path()
            print("\r", f"saving MemoryData: {self.MemoryData_path}", end="")
            self.__class__.save_memory_func(self.MemoryData_path, self.save_preprecess())
            self._reset_MemoryData_modified()

    @Node.downward_preorder_propagate
    def save_as_pickle(self, force = False):
        if len(self.MemoryData) > 1000:
            with open(self.pickle_MemoryData_path, 'wb') as f:
                pickle.dump(self.MemoryData, f)

    @Node.downward_preorder_propagate
    def remove_pickle_datamaping(self):
        if os.path.exists(self.pickle_MemoryData_path):
            os.remove(self.pickle_MemoryData_path)

    def load(self):
        """
        load `self.MemoryData` from file
        """
        if os.path.exists(self.pickle_MemoryData_path):
            with open(self.pickle_MemoryData_path, 'rb') as f:
                MemoryData = pickle.load(f)
            self.merge_MemoryData(MemoryData)
        if os.path.exists(self.MemoryData_path):
            try:
                MemoryData = self.load_postprocess(self.__class__.load_memory_func(self.MemoryData_path))
            except:
                rebuild = True
            else:
                self.merge_MemoryData(MemoryData) ## TODO
                rebuild = False
        else:
            rebuild = True
        if not os.path.exists(self.MemoryData_path) or rebuild:
            if len(self.MemoryData) > 0:
                if DEBUG:
                    warnings.warn(f"will not rebuild")
                self.save()
            else:
                self.rebuild()
                # self.save(True)
        # if DEBUG:
        #     print(self)
        #     print(self.MemoryData)

    @abstractmethod
    def merge_MemoryData(self, MemoryData:dict):
        pass

    def relugar_MemoryData(self):
        pass

    def sort(self):
        new_dict = dict(sorted(self.MemoryData.items(), key=lambda x:x[0])) # type: ignore
        self.MemoryData.clear()
        self.MemoryData.update(new_dict)    
    # endregion Memorydata operation ####

    # region io #####

    # region - property
    @property
    def num(self):
        return len(self.keys())
    
    @property
    def i_upper(self):
        return max(self.keys(), default=-1) + 1

    @property
    def next_valid_i(self):
        return self.i_upper

    @property
    def continuous(self):
        return self.num == self.i_upper
    # endregion - property

    # region - io metas #
    def set_io_ctrl_strategy(self, strategy):
        self.cache_priority, self.strict_priority_mode, self.write_synchronous = IO_CTRL_STRATEGY.get_ctrl_flag(strategy)

    def get_io_ctrl_strategy(self):
        return IO_CTRL_STRATEGY.get_ctrl_strategy(self.cache_priority, self.strict_priority_mode, self.write_synchronous)

    @abstractmethod
    def read(self, src:int) -> VDMT:
        pass

    @abstractmethod
    def write(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        pass
        
    @abstractmethod
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        pass

    @abstractmethod
    def remove(self, dst:int, *, force = False, **other_paras) -> None:
        pass

    @abstractmethod
    def merge_from(self, src:DMT, *, force = False) -> None:
        pass

    @abstractmethod
    def copy_from(self, src:DMT, *, cover = False, force = False) -> None:
        pass
    # endregion io metas #####

    # region - general #    
    def keys(self):
        return self.MemoryData.keys()

    def values(self) -> Generator[VDMT, Any, None]:
        def value_generator():
            keys = sorted(list(self.keys()))
            for i in keys:
                yield self.read(i)
        return value_generator()
    
    def items(self):
        def items_generator():
            keys = sorted(list(self.keys()))
            for i in keys:
                yield i, self.read(i)
        return items_generator()

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
    
    def __setitem__(self, data_i, value:VDMT):
        return self.write(data_i, value)

    def __iter__(self) -> Iterable[VDMT]:
        return self.values()
    
    def __len__(self):
        return len(self.MemoryData)
    # endregion - general ###

    # region - complex io ####
    def append(self, value:VDMT, *, force = False, **other_paras):
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        dst = self.next_valid_i
        self.write(dst, value, force=force, **other_paras)

    def clear(self, *, force = False):
        with self.get_writer(force).allow_overwriting():
            ### TODO
            for key in tqdm(list(self.keys()), desc=f"clear {self}"):
                self.remove(key)

    def make_continuous(self, *, force = False):
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        if self.continuous:
            return
        with self.get_writer(force).allow_overwriting():
            for i, key in tqdm(enumerate(list(self.keys()))):
                self.modify_key(key, i)

    @abstractmethod
    def cache_to_file(self, data_i:Optional[int] = None, *, force = False, **other_paras):
        pass

    @abstractmethod 
    def file_to_cache(self, data_i:Optional[int] = None, *, save = True, force = False, **other_paras):
        pass

    @abstractmethod 
    def clear_files(self, *, force = False):
        pass

    @abstractmethod 
    def clear_cache(self, *, force = False):
        pass
    # endregion - complex io ####

    # endregion####

    # region process_unfinished ####
    @Node.downward_preorder_propagate
    def _set_unfinished_operation(self, unfinished_operation):
        self.__unfinished_operation = unfinished_operation

    def _get_unfinished_operation(self):
        return self.__unfinished_operation

    @Node.upward_preorder_propagate
    def _set_last_write_unfinished(self):
        self.__last_write_unfinished = True

    @Node.downward_preorder_propagate
    def _reset_last_write_unfinished(self):
        self.__last_write_unfinished = False

    def _get_last_write_unfinished(self):
        return self.__last_write_unfinished

    def choose_unfinished_operation(self):
        '''
            0. skip
            1. clear the unfinished data
            2. try to rollback the unfinished data
            3. exit"))
        '''
        choice = int(input(f"please choose an operation to continue:\n\
                    0. \n\
                    1. clear the unfinished data\n\
                    2. try to rollback the unfinished data\n\
                    3. ignore\n"))
        if choice not in [0, 1, 2, 3]:
            raise ValueError(f"invalid choice {choice}")
        self._set_unfinished_operation(choice)
        return choice

    def process_unfinished(self):    
        if self._get_unfinished_operation() == 0:
            self.choose_unfinished_operation()
        if self._get_unfinished_operation() != 0:
            self.rebuild()            
            if self._get_unfinished_operation() == 1:
                self.clear(force=True)
            elif self._get_unfinished_operation() == 2:
                # try roll back
                log = self.load_from_mark_file()
                with self.get_writer().allow_overwriting():
                    for log_i, src, dst, value in log:
                        if log_i == self.LOG_ADD and dst in self.keys():
                            self.remove(dst)
                            print(f"try to rollback, {dst} in {self.identity_string()} is removed.")
                        else:
                            raise ValueError("can not rollback")
            elif self._get_unfinished_operation() == 3:
                # reinit
                pass
            else:
                raise ValueError(f"invalid operation {self._get_unfinished_operation()}")
        self.remove_mark()
        self.save()
        self._reset_last_write_unfinished()
    # endregion TODO ####

    def __repr__(self):
        return self.identity_string()
    
class IOMeta(ABC, Generic[FCT, VDMT, FHT]):
    '''
    This class represents the abstract base class for input/output (IO) operations.
    Pre-process/post-process input and output, format conversion, and perform IO operations. 
    Need to be used in combination with `FilesCluster.io_decorator`

    Attributes:
        READ (bool): Flag indicating if the IO operation is a read operation.
        PATH_EXISTS_REQUIRED (bool): Flag indicating if the existence of paths is required.
        LOG_TYPE: The log type for IO operations.
        WARNING_INFO (str): Information about the warning.

        W_SYNC (bool): Flag indicating if the write operation should be synchronous.

        OPER_ELEM (bool): Flag indicating if the operation is performed on individual elements.

    Methods:
        __init__(self, files_cluster: FCT): Initializes the IOMeta object.
        _set_ctrl_flag(self, ctrl_strategy=IO_CTRL_STRATEGY.CACHE_IDPNDT, io_raw=False): Sets the control flags for IO operations.
        _clear_ctrl_flag(self): Clears the control flags for IO operations.
        __enter__(self): Enters the control mode.
        __exit__(self, exc_type, exc_value, traceback): Exits the control mode.
        _set_synced_flag(self, src_handle: FilesHandle, dst_handle: FilesHandle, synced=False): Sets the synced flag for file handles.
        key_type(self): Returns the key type of the files cluster.
        multi_files(self): Returns whether the files cluster contains multiple files.
        cache_priority(self): Returns the cache priority.
        strict_priority_mode(self): Returns the strict priority mode.
        write_synchronous(self): Returns whether the write operation is synchronous.
        io_raw(self): Returns whether the IO operation is raw.
        _FCMemoryData(self): Returns the memory data of the files cluster.
        get_FilesHandle(self, src, dst, value, **other_paras): Gets the file handles for the IO operation.
        _query_fileshandle(self, data_i: int): Queries the file handle for the given data index.
        get_file_core_func(self, src_file_handle: FHT, dst_file_handle: FHT, value): Gets the core function for the file handles.
        progress_FilesHandle(self, src_file_handle: FHT, dst_file_handle: FHT, postprocessed_value, **other_paras): Progresses the file handles after the IO operation.
        io_cache(self, src_file_handle: FHT, dst_file_handle: FHT, value=None): Performs the IO operation with caching.
        cvt_to_core_paras(self, src_file_handle: FHT, dst_file_handle: FHT, value): Converts the parameters for the core function.
        preprogress_value(self, value, **other_paras): Preprocesses the value before the IO operation.
        postprogress_value(self, value, **other_paras): Postprocesses the value after the IO operation.
        format_value(self, value: Optional[VDMT]): Formats the value before the IO operation.
        inv_format_value(self, formatted_value): Inversely formats the value after the IO operation.
        core_func_hook(self, *core_args): Hook function for the core function.
        operate_elem(self, src, dst, value, **other_paras): Performs the operation on individual elements.
        gather_mutil_results(self, results: list): Gathers multiple results.
        split_value_as_mutil(self, *core_values): Splits the value into multiple core values.
        assert_path_exists(self, path): Asserts that the path exists.
        io_file(self, src_file_handle, dst_file_handle, value: Optional[VDMT] = None): Performs the IO operation on a file.

    Methods (abstract):
        get_FilesHandle(self, src, dst, value, **other_paras): Gets the file handles for the IO operation.
        io_cache
        cvt_to_core_paras

    Methods recommended to be overridden:
        get_file_core_func: Gets the core function of the file handles.
        preprogress_value: Preprocesses the value before the I operation.
        postprogress_value: Postprocesses the value after the O operation.
        format_value: Formats the value before the I operation.
        inv_format_value: Inversely formats the value after the O operation.
        core_func_hook: Hook function for the core function.

    Methods need to be overridden if `OPER_ELEM`:
        operate_elem

    Methods need to be overridden if the files cluster contains multiple files:
        gather_mutil_results
        split_value_as_mutil

    '''
    READ = True
    PATH_EXISTS_REQUIRED = True
    LOG_TYPE = IOStatusManager.LOG_READ
    WARNING_INFO = "no description"

    W_SYNC = False

    OPER_ELEM = False

    def __init__(self, files_cluster:FCT) -> None:
        self.files_cluster:FCT = files_cluster
        self.core_func:Optional[Callable] = None #type: ignore
        self.core_func_binded_paras = {}

        self.__cache_priority = True
        self.__strict_priority_mode = False
        self.__write_synchronous = False
        self.__io_raw = False

        self.ctrl_mode = False

        self.save_memory_after_writing = False

    # region ctrl flags
    def _set_ctrl_flag(self, ctrl_strategy = IO_CTRL_STRATEGY.CACHE_IDPNDT, io_raw = False):
        self.__cache_priority, self.__strict_priority_mode, self.__write_synchronous = IO_CTRL_STRATEGY.get_ctrl_flag(ctrl_strategy)
        self.__io_raw = io_raw
        self.ctrl_mode = True
        return self
    
    def _clear_ctrl_flag(self):
        self.__cache_priority = True
        self.__strict_priority_mode = False
        self.__write_synchronous = False
        self.__io_raw = False
        self.ctrl_mode = False
    
    def __enter__(self):
        self.ctrl_mode = True
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(exc_value).with_traceback(traceback)
        self.ctrl_mode = False

    def _set_synced_flag(self, src_handle:FilesHandle, dst_handle:FilesHandle, synced = False):
        if src_handle is not None:
            src_handle._set_synced(synced)
        if dst_handle is not None:
            dst_handle._set_synced(synced)
    # endregion ctrl flags

    # region properties
    @property
    def key_type(self):
        return self.files_cluster.KEY_TYPE

    @property
    def multi_files(self):
        return self.files_cluster.MULTI_FILES

    @property
    def cache_priority(self):
        if self.ctrl_mode:
            return self.__cache_priority
        else:
            return self.files_cluster.cache_priority

    @property
    def strict_priority_mode(self):
        if self.ctrl_mode:
            return self.__strict_priority_mode
        else:
            return self.files_cluster.strict_priority_mode

    @property
    def write_synchronous(self):
        if self.ctrl_mode:
            return self.__write_synchronous
        else:
            return self.files_cluster.write_synchronous

    @property
    def io_raw(self):
        if self.ctrl_mode:
            return self.__io_raw
        else:
            return False

    @property
    def _FCMemoryData(self)  -> FhBinDict[FHT]:
        return self.files_cluster.MemoryData # type: ignore
    # endregion properties

    # region hook funcs
    @abstractmethod
    def get_FilesHandle(self, src, dst, value, **other_paras) -> tuple[FHT, FHT]:
        pass

    def _query_fileshandle(self, data_i:int) -> FHT:
        return self.files_cluster.query_fileshandle(data_i) # type: ignore

    def get_file_core_func(self, src_file_handle: FHT, dst_file_handle: FHT, value) -> Callable:
        """
        This method returns the file core function.

        Args:
            src_file_handle (FilesHandle): The source file handle.
            dst_file_handle (FilesHandle): The destination file handle.
            value: The value.

        Returns:
            Callable: The file core function.
        """
        return None  # type: ignore

    def progress_FilesHandle(self, 
                            src_file_handle:FHT, 
                            dst_file_handle:FHT, 
                            postprocessed_value, 
                            **other_paras) -> tuple[FHT]: # type: ignore
        pass

    @abstractmethod
    def io_cache(self, src_file_handle:FHT, dst_file_handle:FHT, value = None) -> Any:
        pass

    @abstractmethod
    def cvt_to_core_paras(self, src_file_handle: FHT, dst_file_handle: FHT, value) -> tuple:
        """
        Convert the given source file handle and destination file handle to core parameters.

        Args:
            src_file_handle (FilesHandle): The source file handle.
            dst_file_handle (FilesHandle): The destination file handle.
            value: The value to be converted.

        Returns:
            path: The path or paths that the core function will use.
            *core_values: The core values that the core function will use.
        """
        pass

    def preprogress_value(self, value, **other_paras) -> Any:
        """
        Preprocesses the value before the O operation.

        See Also
        -----
        * the difference between `format_value` and `preprogress_value` see :link:`format_value`
        
        Args:
            value: The value to be preprocessed.
            **other_paras: Additional parameters for preprocessing.

        Returns:
            The preprocessed value.
        """
        return value

    def postprogress_value(self, value, **other_paras) -> Any:
        """
        Postprocesses the value after the I operation.
        
        See Also
        -----
        * the difference between `inv_format_value` and `postprogress_value` see :link:`format_value`

        Args:
            value: The value to be processed.
            **other_paras: Additional parameters.
        
        Returns:
            The processed value.
        """
        return value

    def format_value(self, value:Optional[VDMT]) -> Any:
        """
        Formats the value before the O operation.
        
        * the difference between `format_value` and `preprogress_value` is that:

        `preprogress_value` has additional parameters: `**other_paras`, which allows more flexible preprocessing, such as resize an image.

        `format_value` need no additional parameters. 
        It is used to format the value to the type that the core function can handle, such as converting a numpy array to a tensor. 
        
        Args:
            value (Optional[VDMT]): The value to be formatted.
        
        Returns:
            Any: The formatted value.
        """
        return value
    
    def inv_format_value(self, formatted_value) -> VDMT:
        """
        Inversely formats the value after the I operation.

        * the difference between `inv_format_value` and `postprogress_value` see :link:`format_value`

        Args:
            formatted_value: The formatted value to be inverse formatted.

        Returns:
            The inverse formatted value.
        """
        return formatted_value

    def core_func_hook(self, *core_args):
        """
        This is a hook function that can be overridden in subclasses.
        It is called after the core function is executed.
        
        Parameters:
            *core_args: Variable number of arguments passed to the core_func.
        
        Returns:
            None
        """
        pass

    def operate_elem(self, src, dst, value, **other_paras):
        """
        # WARNING: must be overridden if `OPER_ELEM = True`
        Perform an operation on the elements.
        Args:
            src: The source element.
            dst: The destination element.
            value: The value to be used in the operation.
            **other_paras: Additional parameters for the operation.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def gather_mutil_results(self, results: list):
        """
        # WARNING: must be overridden if `self.multi_files = True`
        Gathers multiple results into a single result.

        Args:
            results (list): A list of results to be gathered.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        FilesHandle.multi_files = True
        raise NotImplementedError

    def split_value_as_mutil(self, *core_values):
        """
        # WARNING: must be overridden if `self.multi_files = True`
        Splits the given core values into multiple values.
        
        Args:
            *core_values: The core values to be split.
        
        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError
    
    def assert_path_exists(self, path: T_MUITLSTR):
        """
        Asserts that the given path or paths exist.

        Args:
            path (str or list[str]): The path or paths to check.

        Raises:
            IOMetaPriorityError: If the path or any of the paths do not exist and PATH_EXISTS_REQUIRED is True.
        """
        if isinstance(path, str):
            if not os.path.exists(path) and self.PATH_EXISTS_REQUIRED:
                raise IOMetaPriorityError("file not found")
        elif isinstance(path, list):
            if any([not os.path.exists(x) for x in path]) and self.PATH_EXISTS_REQUIRED:
                raise IOMetaPriorityError("file not found")
    # endregion hook funcs

    # region io pipeline
    def io_file(self, src_file_handle:FilesHandle, dst_file_handle:FilesHandle, value:Optional[VDMT] = None) -> Any:
        """
        Perform file input/output operation.

        Args:
            src_file_handle: The source file handle.
            dst_file_handle: The destination file handle.
            value: Optional value to be used in the operation.

        Returns:
            The result of the file operation.

        Raises:
            IOMetaPriorityError: If core_func is None.
        """
        value = self.format_value(value)
        path, *core_values = self.cvt_to_core_paras(src_file_handle = src_file_handle, 
                                            dst_file_handle = dst_file_handle, 
                                            value = value)
        self.assert_path_exists(path)
        # get core func with priority: the core_func of the file_handle has higher priority than the core func of the files cluster
        core_func = get_with_priority(self.get_file_core_func(src_file_handle, dst_file_handle, value), self.core_func)
        if core_func is not None:
            if self.multi_files and self.__class__.__name__ == "_write":
                core_values = self.split_value_as_mutil(*core_values) # split core values to multiple values, because `core_values` contains the datas for mutil files 
            rlt = self.execute_core_func(core_func, path, *core_values) # execute core func
            if self.multi_files and self.__class__.__name__ == "_read":
                return self.gather_mutil_results(rlt) # gather mutil results to a single result, because `rlt` contains the datas for mutil files
        else:
            raise IOMetaPriorityError("core_func is None")
        try:
            rlt = self.inv_format_value(rlt)
        except:
            raise IOMetaPriorityError("inv_format_value raises exception")
        return rlt

    def execute_core_func(self, core_func, *core_args, **other_paras):
        """
        Executes the core function with the given arguments and parameters.

        Args:
            core_func: The core function to be executed.
            core_args: The arguments to be passed to the core function.
            other_paras: Other parameters to be passed to the core function.

        Returns:
            The result of the core function execution.
        """
        if self.multi_files:
            rlts = []
            for i in range(len(core_args[0])):
                try:
                    rlt = core_func(*[x[i] for x in core_args], **self.core_func_binded_paras)
                except:
                    raise IOMetaPriorityError("unexpected error while executing core_func")
                self.core_func_hook(*[x[i] for x in core_args], **other_paras)
                rlts.append(rlt)
            return rlts
        else:
            try:
                rlt = core_func(*core_args, **self.core_func_binded_paras)
            except:
                raise IOMetaPriorityError("unexpected error while executing core_func")
            self.core_func_hook(*core_args)
            return rlt

    def io(self, src_handle:FilesHandle, dst_handle:FilesHandle, preprocessed_value):
        """
        Perform input/output operations on the given source and destination handles.

        IO operations have a sequential order.
        Try to perform IO on the file and cache (or cache first and then the file) in sequence.

        * the first operation is called 'priority', and the second operation is called 'secondary'.

        If 'priority' fails, continue to execute 'secondary', 
        otherwise the result will be returned directly.

        you may control the strategy of the IO operations by setting the control flags by `_set_ctrl_flag` method:
        * `self.cache_priority`: If True, the cache will be used first.
        * `self.strict_priority_mode`: If True, an exception will be raised if the priority fails.
        * `self.write_synchronous`: If True, both file and cache will be written synchronously.
        * `self.io_raw`: If True, the IO operation will be raw.

        Tip
        ----
        You should pay attention to the definition of :link:`cache_priority`,
        :link:`self.strict_priority_mode` and :link:`self.write_synchronous`, see their docs for details
        
        Args:
        ----
            src_handle (FilesHandle): The source handle for the input operation.
            dst_handle (FilesHandle): The destination handle for the output operation.
            preprocessed_value: The preprocessed value to be used in the operations.

        Returns:
        ----
            The result of the input/output operations.
        """
        
        priority_ok = False
        secondary_ok = False
        if self.cache_priority:
            func_priority = self.io_cache
            func_secondary = self.io_file
        else:
            func_priority = self.io_file
            func_secondary = self.io_cache
        try:
            rlt = func_priority(src_handle, dst_handle, preprocessed_value) # type: ignore
            priority_ok = True
            exe_secondary = not self.READ and (self.write_synchronous or self.W_SYNC)
        except IOMetaPriorityError:
            if self.strict_priority_mode:
                raise ClusterDataIOError
            exe_secondary = True
        
        if exe_secondary:
            try:
                rlt = func_secondary(src_handle, dst_handle, preprocessed_value) # type: ignore
            except IOMetaPriorityError:
                if priority_ok == False:
                    raise ClusterDataIOError
            else:
                secondary_ok = True

        if priority_ok and secondary_ok:
            self._set_synced_flag(src_handle, dst_handle, True)

        return rlt # type: ignore

    def __call__(self, *, src=None, dst=None, value=None, **other_paras) -> Any:
        """
        Make the IOMeta object callable.

        Args:
            src: The source parameter.
            dst: The destination parameter.
            value: The value parameter.
            **other_paras: Additional parameters.

        Returns:
            The result of the call.

        Raises:
            None.
        """
        if not self.__io_raw:
            value = self.preprogress_value(value, **other_paras)
        if self.OPER_ELEM:
            rlt = self.operate_elem(src, dst, value, **other_paras)
            for fh in self.files_cluster.query_all_fileshandle():
                fh._set_synced(False)
        else:
            src_handle, dst_handle = self.get_FilesHandle(src=src, dst=dst, value=value, **other_paras)
            rlt = self.io(src_handle, dst_handle, value)
            # self.progress_FilesHandle(src_handle, dst_handle, rlt, **other_paras)
        if not self.__io_raw:
            rlt = self.postprogress_value(rlt, **other_paras)
        self.files_cluster._clear_rely()
        return rlt
    # endregion io pipeline

    # region conditions func
    def check_src(self, src):
        """
        Check if the given source is valid.

        Parameters:
            src (self.key_type): The source to be checked.

        Returns:
            bool: True if the source is valid, False otherwise.
        """
        if not isinstance(src, self.key_type):
            return False
        if not self.files_cluster.has(src):
            return False
        return True
        
    def check_dst(self, dst):
        """
        Check if the given destination is of the correct type.

        Parameters:
            dst (self.key_type): The destination to be checked.

        Returns:
            bool: True if the destination is of the correct type, False otherwise.
        """
        if not isinstance(dst, self.key_type):
            return False
        return True

    def check_value(self, value: Any):
        return True

    def is_overwriting(self, dst:int):
        return not self.files_cluster.idx_unwrited(dst)
    # endregion conditions func

class FilesCluster(DataMapping[FHT, FCT, VDMT], Generic[FHT, FCT, DSNT, VDMT]):
    """
    # FilesCluster
    A class representing a cluster of files.

    ## Class Attributes:
    ----

    ### _IS_ELEM
    ----

    ### _ELEM_BY_CACHE

    需要说明的是data和elem的定义：
    `data`是指`FilesCluster`对象包含的文件，当`_ELEM_BY_CACHE`为`False`时，
    所使用的索引是默认针对在`FilesCluster.MemoryData`中对应的`FilesHandle`的。
    而在某些情况下，文件的IO操作可能是不推荐直接进行的，而应该先操作`FilesHandle`的`cache`，
    然后再调用`cache_to_file`。否则，如果每条数据的写入都需要进行IO操作，会消耗大量资源。
    例如一个包含了多条数据的json/xml文件。这时需要设置`_ELEM_BY_CACHE`为`True`，
    使得索引默认针对`elem`，即对所有`FilesHandle`的cache同名的`key`。

    #### `_ELEM_BY_CACHE`的值会影响一系列属性和方法的默认调用，你也可以调用对应的明确方法：
    #### 属性：
    * `num`:            `data_num`/`elem_num`
    * `i_upper`:        `data_i_upper`/`elem_i_upper`
    * `next_valid_i`:   `next_valid_elem_i`/`net_valid_data_i`
    #### 方法：
    * `read`:           `read_data`/`read_elem`
    * `write`:          `write_data`/`write_elem`
    * `modify_key`:     `modify_data_key`/`modity_elem_key`
    * `remove`:         `remove_data`/`remove_elem`
    * `merge_from`:     `merge_data_from`/`merge_elem_from`
    * `copy_from`:      `copy_data_from`/`copy_elem_from`
    * `clear`:          `clear_data`/`clear_elem`
    * `keys`:           `data_keys`/`elem_keys`
    * `idx_unwritten`

    #### 举个例子，如果在`self.MemoryData`中有2个FilesHandle：
    >>> fc.MemoryData 
    >>> {0: fh0, 1: fh1}
    >>> fh0.cache
    >>> {0: array([0,1,2]), 1: array([3,4,5]) 2: array([6,7,8])}
    >>> fh1.cache
    >>> {0: 1.0, 1: 5.0, 2: 10.0}


    如果`_ELEM_BY_CACHE`为`False`，调用`read`时，实际上`read_data`会被调用
    >>> fc.read(0) # 等效于fc.read_data(0)
    >>> {0: array([0,1,2]), 1: array([3,4,5]) 2: array([6,7,8])}

    如果`_ELEM_BY_CACHE`为`True`，调用`read`时，实际上`read_elem`会被调用
    >>> fc.read(0) # 等效于fc.read_elem(0)
    >>> (array([0,1,2]), 1.0)
    
    ----

    * KEY_TYPE: The type of the keys used to access the files in the cluster.
    * ALWAYS_ALLOW_WRITE: A boolean indicating whether write operations are always allowed.
    * ALWAYS_ALLOW_OVERWRITE: A boolean indicating whether overwriting files is always allowed.
    * DEFAULT_SUFFIX: The default suffix for file names.
    * DEFAULT_PREFIX: The default prefix for file names.
    * DEFAULT_PREFIX_JOINER: The default joiner for prefix and core names.
    * DEFAULT_APPENDNAMES: The default append names for file names.
    * DEFAULT_APPENDNAMES_JOINER: The default joiner for append names.
    * DEFAULT_READ_FUNC: The default function for reading files.
    * DEFAULT_WRITE_FUNC: The default function for writing files.
    * DEFAULT_VALUE_TYPE: The default type of the values stored in the files.
    * DEFAULT_VALUE_INIT_FUNC: The default function for initializing values.
    * MULTI_FILES: A boolean indicating whether the cluster contains multiple files.
    * STRATEGY_ONLY_CACHE: The strategy for file operations when only using cache.
    * STRATEGY_ONLY_FILE: The strategy for file operations when only using file.
    * STRATEGY_CACHE_AND_FILE: The strategy for file operations when using both cache and file.

    Properties commonly used:
    * dataset_node: The dataset node that the cluster belongs to.
    * continuous: A boolean indicating whether the cluster is continuous.
    * num: The number of files in the cluster.
    * i_upper: The upper bound of the indices of the files in the cluster.

    Methods commonly used:
    * query_all_fileshandle: Queries all the file handles in the cluster.
    * read: Reads a file from the cluster.
    * write: Writes a file to the cluster.
    * modify_key: Modifies the key of a file in the cluster.
    * remove: Removes a file from the cluster.
    * merge_from: Merges the cluster from another cluster.
    * copy_from: Copies the cluster from another cluster.
    * append: Appends a file to the cluster.
    * clear: Clears the cluster.
    * make_continuous: Makes the cluster continuous.

    """

    _IS_ELEM = False

    _ELEM_BY_CACHE = False

    KEY_TYPE = int
    """
    xyz
    """
    
    ALWAYS_ALLOW_WRITE = False
    ALWAYS_ALLOW_OVERWRITE = False

    # fileshandle_default_paras

    DEFAULT_SUFFIX = None
    DEFAULT_PREFIX = None
    DEFAULT_PREFIX_JOINER = None
    DEFAULT_APPENDNAMES = None
    DEFAULT_APPENDNAMES_JOINER = None
    DEFAULT_READ_FUNC = None
    DEFAULT_WRITE_FUNC = None
    DEFAULT_VALUE_TYPE = None
    DEFAULT_VALUE_INIT_FUNC = None

    MULTI_FILES = False

    STRATEGY_ONLY_CACHE = 0
    STRATEGY_ONLY_FILE = 1
    STRATEGY_CACHE_AND_FILE = 2

    # region FilesCluster override ####

    # region - override new #
    # def __new__(cls, dataset_node:Union[str, "DatasetNode"], mapping_name: str = "", *args, flag_name = "", **kwargs):
    #     return super().__new__(cls, dataset_node, mapping_name, *args, flag_name = flag_name, **kwargs)

    # def init_identity(self, dataset_node:Union[str, "DatasetNode"], mapping_name: str, *args, flag_name = "", **kwargs):
    #     if isinstance(dataset_node, str):
    #         self._top_directory = dataset_node
    #     elif isinstance(dataset_node, DatasetNode):
    #         self._top_directory = os.path.join(dataset_node.data_path, name)
    #         assert is_subpath(self._top_directory, dataset_node.top_directory), f"{self.top_directory} is not inside {dataset_node.top_directory}"
    #     elif dataset_node is None:
    #         self._top_directory = ""
    #     else:
    #         raise TypeError(f"dataset_node must be str or DatasetNode, not {type(dataset_node)}")
    #     self.mapping_name = name       
    #     self.flag_name = flag_name 
    # endregion - override new #

    # region - override methods #
    def __init__(self, dataset_node: Union[str, "DatasetNode"], mapping_name: str, *args, flag_name = "", **kwargs) -> None:
        super().__init__(dataset_node, mapping_name)

        self.init_io_metas()
        self.init_attrs()
        # self.register_to_dataset()

    @property
    def MemoryData(self) -> FhBinDict[FHT]:
        return self._MemoryData # type: ignore
    
    # endregion - override methods #

    # endregion FilesCluster override ####

    # region FilesCluster new methods ####
    @property
    def dataset_node(self) -> "DatasetNode":
        return self.parent
    
    @property
    def registerd(self):
        return self.dataset_node is not None and \
            self.identity_string() in self.dataset_node.clusters_map

    def get_rely_io_parameters(self):
        pass
    # endregion FilesCluster new methods ####
    
    # region fileshandle operation ########
    def query_all_fileshandle(self):
        """
        Returns all the file handles stored in the MemoryData dictionary.
        
        Returns:
            list: A list of file handles.
        """
        return self.MemoryData.values()
    
    def query_fileshandle_from_iterable(self, iterable:Union[Iterable, slice]) -> Generator[FHT, Any, None]:
        """
        Queries fileshandle from an iterable or a slice.

        Args:
            iterable (Union[Iterable, slice]): An iterable or a slice.

        Yields:
            Generator[FHT, Any, None]: A generator that yields fileshandle.

        Returns:
            None
        """
        if isinstance(iterable, slice):
            iterable = range(iterable.start, iterable.stop, iterable.step)
        for i in iterable:
            yield self.query_fileshandle(i)

    def query_fileshandle(self, data_i:int) -> FHT:
        """
        Retrieve the fileshandle at the specified index from the MemoryData.

        Args:
            data_i (int): The index of the fileshandle to retrieve.

        Returns:
            FHT: The fileshandle at the specified index.
        """
        return self.MemoryData[data_i]
    
    @abstractmethod
    def create_fileshandle_in_iometa(self, src, dst, value, **other_paras) -> FHT:
        pass

    def format_corename(self, data_i:int) -> str:
        """
        Formats the core name of the data.
        * recommended to be overridden in subclasses
        
        e.g.: 
        >>> 1 -> "000001"
        >>> 309 -> "000309"

        Args:
            data_i (int): The data index.

        Returns:
            str
        """
        return None
    
    def deformat_corename(self, corename:str) -> int:
        """
        Deformats the core name of the data.
        * recommended to be overridden in subclasses

        e.g.:
        >>> "000001" -> 1
        >>> "000309" -> 309

        Args:
            corename (str): The core name.

        Returns:
            int
        """
        return None
    
    def _set_fileshandle(self, data_i, fileshandle:FHT):
        '''
        Set the fileshandle for the given data index.

        Parameters:
            data_i (int): The index of the data.
            fileshandle (FHT): The fileshandle to be set.

        Returns:
            bool: True if the fileshandle is set successfully, False otherwise.
        '''
        assert isinstance(fileshandle, self.FILESHANDLE_TYPE), f"fileshandle must be {self.FILESHANDLE_TYPE}, not {type(fileshandle)}"
        if not self.MemoryData.has_value(fileshandle):
            self.MemoryData[data_i] = fileshandle
            self._set_MemoryData_modified()
            return True
        return False

    def _pop_fileshandle(self, data_i):
        """
        Pop the file handle for the given data index from the MemoryData dictionary.

        Args:
            data_i (int): The index of the data.

        Returns:
            object: The popped file handle.
        """
        if self.has_data(data_i):
            fh = self.MemoryData.pop(data_i)
            if fh is not None:
                self._set_MemoryData_modified()
    # endregion fileshandle operation ########

    # region io #####
    
    # region - io rely #

    @property
    def use_rely(self):
        """
        Check if use rely
        """
        if self.dataset_node is None:
            return False
        else:
            return self.dataset_node.cluster_use_rely

    def link_rely_on(self, cluster:"FilesCluster"):
        """
        Links the current cluster to the specified cluster as a relying cluster.

        Details:
        -----
        The IO of the cluster may depend on other clusters. 
        For example, the normalized pixel coordinates depend on the size of the image. 
        In this case, you can use `link_rely_on` to set the cluster that self depends on.

        * relying_clusters: those clusters that rely on self
        * relied_clusters: those clusters that self relies on
        
        Args:
        -----
        cluster (FilesCluster): The cluster to be linked as a relying cluster.
        """
        if cluster not in cluster.__relying_clusters:
            cluster.__relying_clusters.append(self)
            self.__relied_clusters.append(cluster)
            if cluster.dataset_node is not None:
                cluster.dataset_node.cluster_use_rely = True
            if self.dataset_node is not None:
                self.dataset_node.cluster_use_rely = True

    def unlink_rely_on(self, cluster:"FilesCluster"):
        """
        Unlinks the cluster from relying. 

        Args:
        -----
            cluster (FilesCluster): The cluster to be unlinked from relying.
        """
        if cluster in cluster.__relying_clusters:
            cluster.__relying_clusters.remove(self)
            self.__relied_clusters.remove(cluster)

    def _send_rely(self, rlt):
        '''
        send parameters to the relying clusters
        '''
        if self.use_rely:
            for c in self.__relying_clusters:
                c._set_rely(self, rlt)

    def _set_rely(self, relied:"FilesCluster", rlt):
        '''
        set relied parameters from the relied cluster, 
        relied is the cluster that send the parameters, rlt is the parameters
        '''
        pass

    def _get_rely(self, name):
        if name not in self.__relied_paras:
            raise KeyError(f"rely parameter not found, maybe you should call 'read' or 'write' of the relying cluster:{self.__relied_clusters} before that of self")
        rlt = self.__relied_paras[name]
        return rlt
    
    def _update_rely(self, name, rlt):
        self.__relied_paras[name] = rlt

    def _clear_rely(self):
        self.__relied_paras.clear()
    # endregion - io rely #

    # region - io metas #
    def init_io_metas(self):
        self.read_meta:IOMeta[FCT, VDMT, FHT]            = self._read(self)
        self.write_meta:IOMeta[FCT, VDMT, FHT]           = self._write(self)
        self.modify_key_meta:IOMeta[FCT, VDMT, FHT]      = self._modify_key(self)
        self.remove_meta:IOMeta[FCT, VDMT, FHT]          = self._remove(self)
        self.paste_file_meta:IOMeta[FCT, VDMT, FHT]      = self._paste_file(self)
        self.change_dir_meta:IOMeta[FCT, VDMT, FHT]      = self._change_dir(self)

    def init_attrs(self):
        self.__relying_clusters:list[FilesCluster]    = [] # clusters that rely on self
        self.__relied_clusters:list[FilesCluster]     = [] # clusters that self relies on
        self.__relied_paras                           = {} # parameters that IO operation relies on

        self.merge_strategy = self.STRATEGY_ONLY_CACHE

    def cvt_key(self, key):
        if self.KEY_TYPE == int:
            if isinstance(key, (np.intc, np.integer)): # type: ignore
                key = int(key)
        return key

    def io_decorator(self, io_meta:IOMeta, force = False):
        """
        Decorator function for input/output operations.

        Args:
            io_meta (IOMeta): The metadata object for the input/output operation.
            force (bool): Flag indicating whether to force the operation. Defaults to False.

        Returns:
            function: The decorated function.
        """
        func = io_meta.__call__ # io_meta
        is_read = io_meta.READ
        log_type = io_meta.LOG_TYPE
        warning_info = io_meta.WARNING_INFO

        allow_write         = (force or self.ALWAYS_ALLOW_WRITE)        and not self.writable
        allow_overwrite     = (force or self.ALWAYS_ALLOW_OVERWRITE)    and not self.overwrite_allowed
        
        def wrapper(*, src:int = None, dst:int = None, value = None, **other_paras): # type: ignore
            """
            A wrapper function that performs input/output operations for the data.

            Args:
                src (int, optional): The source key. Defaults to None.
                dst (int, optional): The destination key. Defaults to None.
                value (any, optional): The value to be written. Defaults to None.
                **other_paras: Additional parameters.

            Returns:
                any: The result of the original function call.

            Raises:
                ClusterIONotExecutedWarning: If the input/output operation is not executed.

            """
            nonlocal self, log_type, warning_info
            src = self.cvt_key(src)
            dst = self.cvt_key(dst)
            rlt = None
            # with self._IOContext(self, force, force, force): 
            with self.get_writer(allow_write).allow_overwriting(allow_overwrite):
                io_error = False
                overwrited = False

                # check if the contidions are satisfied
                if self.is_closed(with_warning=True) or (not is_read and self.is_readonly(with_warning=True)):
                    return None
                if src is not None and not io_meta.check_src(src):
                    io_error = True
                    warning_info = "src is invalid"
                elif dst is not None and not io_meta.check_dst(dst):
                    io_error = True
                    warning_info = "dst is invalid"
                elif value is not None and not io_meta.check_value(value):
                    io_error = True
                    warning_info = "value is invalid"
                if io_meta.is_overwriting(dst):
                    if not self.overwrite_allowed and not force:
                        warnings.warn(f"{self.__class__.__name__}:{self.mapping_name} " + \
                                    "is not allowed to overwitre, any write operation will not be executed.",
                                    ClusterIONotExecutedWarning)
                        io_error = True
                        warning_info = "overwriting is not allowed"
                        return None
                    overwrited = True 

                if not io_error:                
                    # ready to execute the original function
                    try:
                        if not is_read and not self.is_writing:
                            self.start_writing()
                        rlt = func(src=src, dst = dst, value = value, **other_paras)  # Calls the original function.
                    except ClusterDataIOError as e:
                        rlt = None
                        if str(e) == "skip":
                            pass
                        else:
                            io_error = True     
                            warning_info = "ClusterDataIOError was raised during running func:" + str(e)
                    else:
                        if not is_read:
                            self._set_MemoryData_modified() # Marks the cluster as updated after writing operations.
                            if overwrited and log_type == self.LOG_ADD:
                                log_type = self.LOG_CHANGE
                            self.log_to_mark_file(log_type, src, dst, value) # Logs the operation.
                            if self.dataset_node is not None and (io_meta.OPER_ELEM == self._ELEM_BY_CACHE):
                                self.dataset_node.update_overview(log_type, src, dst, value, self)
                else:
                    io_type_name = ["READ", "APPEND", "REMOVE", "CHANGE", "MOVE", "OPERATION"]
                    warning_str =   f"{self.__class__.__name__}:{self.mapping_name} " + \
                                    f"{io_type_name[log_type]}: src:{src}, dst:{dst}, value:{value} failed:" + \
                                    f"{warning_info}"
                    warnings.warn(warning_str, ClusterIONotExecutedWarning)

            return rlt
        return wrapper

    _FCT = TypeVar('_FCT', bound="FilesCluster")
    _VDMT = TypeVar('_VDMT')
    _FHT = TypeVar('_FHT', bound=FilesHandle)

    class _read(IOMeta[_FCT, _VDMT, _FHT]):
        def get_FilesHandle(self, src, dst, value, **other_paras):
            if self.files_cluster.has_data(src):
                return self._query_fileshandle(src), None
            else:
                raise ClusterDataIOError

        def get_file_core_func(self, src_file_handle:FHT, dst_file_handle:FHT, value) -> Callable:
            return src_file_handle.read_func

        def io_cache(self, src_file_handle:FilesHandle, dst_file_handle, value=None) -> Any:
            if src_file_handle.has_cache:
                return src_file_handle.cache
            else:
                raise IOMetaPriorityError

        def cvt_to_core_paras(self, 
                              src_file_handle: FilesHandle, 
                              dst_file_handle: FilesHandle, 
                              value, 
                              **other_paras) -> tuple:
            return (src_file_handle.get_path(), )

    class _operation(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        LOG_TYPE = IOStatusManager.LOG_OPERATION

    class _write(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        PATH_EXISTS_REQUIRED = False
        LOG_TYPE = IOStatusManager.LOG_ADD

        def get_FilesHandle(self, src, dst, value,  **other_paras):
            if not self.files_cluster.has_data(dst):
                fh:FilesHandle = self.files_cluster.create_fileshandle_in_iometa(src, dst, value, **other_paras)
                self.files_cluster._set_fileshandle(dst, fh)
            return None, self._query_fileshandle(dst)

        def get_file_core_func(self, src_file_handle, dst_file_handle:FilesHandle, value) -> Callable[..., Any]:
            return dst_file_handle.write_func

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value=None) -> Any:
            dst_file_handle.set_cache(value)

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value, 
                                **other_paras) -> tuple:
            dst_path = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return dst_path, value
        
    class _modify_key(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        LOG_TYPE = IOStatusManager.LOG_MOVE

        W_SYNC = True

        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = os.rename

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value=None) -> Any:
            pass

        def get_FilesHandle(self, src, dst, value, **other_paras):
            src_handle:FilesHandle = self.files_cluster._pop_fileshandle(src) # type: ignore
            if not self.files_cluster.has_data(dst):
                dst_handle = self.files_cluster.FILESHANDLE_TYPE.create_temp().\
                    from_fileshandle(self.files_cluster, src_handle, 
                        corename= self.files_cluster.format_corename(dst))
                self.files_cluster._set_fileshandle(dst, dst_handle)
            dst_handle = self._query_fileshandle(dst)
                
            return src_handle, dst_handle

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value, 
                                **other_paras) -> tuple:
            src_path = src_file_handle.get_path()
            dst_path = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return src_path, dst_path

    class _remove(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        LOG_TYPE = IOStatusManager.LOG_REMOVE

        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = os.remove

        def check_dst(self, dst):
            rlt = super().check_dst(dst)
            if dst not in self.files_cluster.keys():
                return False
            return True and rlt

        def get_FilesHandle(self, src, dst, value, **other_paras):
            if not self.files_cluster.has_data(dst):
                fh = self.files_cluster.FILESHANDLE_TYPE.create_not_exist_fileshandle(self.files_cluster)
            else:
                fh = self.files_cluster._pop_fileshandle(dst)
            return None, fh

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value=None) -> Any:
            dst_file_handle.erase_cache()

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value, 
                                **other_paras) -> tuple:
            return (dst_file_handle.get_path(), )

    class _paste_file(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        # W_SYNC = True
        PATH_EXISTS_REQUIRED = True
        LOG_TYPE = IOStatusManager.LOG_ADD

        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = shutil.copy

        def get_FilesHandle(self, src, dst, value, **other_paras):
            src_handle:FilesHandle = value
            if not self.files_cluster.has_data(dst):
                dst_handle = self.files_cluster.FILESHANDLE_TYPE.from_fileshandle(self.files_cluster, src_handle, 
                                                                                  corename= self.files_cluster.format_corename(dst),
                                                                                  cache = FilesHandle.KW_INIT_WITHOUT_CACHE)
                self.files_cluster._set_fileshandle(dst, dst_handle)
            dst_handle = self._query_fileshandle(dst)
            return None, self._query_fileshandle(dst)

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value:FilesHandle=None) -> Any: # type: ignore
            dst_file_handle.set_cache(value.cache)

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value: FilesHandle, 
                                **other_paras) -> tuple:
            src_path = value.get_path()
            dst_path = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return src_path, dst_path

    class _change_dir(_operation[_FCT, _VDMT, _FHT]):
        W_SYNC = True
        
        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = os.rename

        def get_FilesHandle(self, src, dst, value:str):
            src_handle = self.files_cluster._pop_fileshandle(src)
            dst_handle = self.files_cluster.FILESHANDLE_TYPE.from_fileshandle(self.files_cluster, src_handle, sub_dir=value)
            self.files_cluster._set_fileshandle(src, dst_handle)
            return src_handle, dst_handle

        def io_cache(self, src_file_handle, dst_file_handle, value=None) -> Any:
            pass

        def cvt_to_core_paras(self,                                 
                              src_file_handle: FilesHandle, 
                              dst_file_handle: FilesHandle, 
                              value: FilesHandle,  
                              **other_paras) -> tuple:
            src_paths:Union[str, list[str]] = src_file_handle.get_path()
            dst_paths:Union[str, list[str]] = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return src_paths, dst_paths
        
        def check_value(self, value: str):
            return isinstance(value, str)
    # endregion io metas #####

    # region - common operation #
    @property
    def continuous(self):
        return self.num == self.i_upper

    @property
    def data_continuous(self):
        return self.data_num == self.data_i_upper
    
    @property
    def elem_continuous(self):
        return self.elem_num == self.elem_i_upper

    def has(self, i):
        return i in self.keys()

    def has_data(self, elem_i):
        return elem_i in self.data_keys()

    def has_elem(self, elem_i):
        return elem_i in self.elem_keys()
    
    @property
    def data_num(self):
        return len(self.data_keys())

    @property
    def data_i_upper(self):
        if self.KEY_TYPE != int:
            raise TypeError(f"the key_type of {self.__class__.__name__} is not int")
        return max(self.data_keys(), default = -1) + 1# type: ignore
    
    @property
    def next_valid_data_i(self):
        return self.data_i_upper

    @property
    def elem_num(self):
        return len(self.elem_keys())
    
    @property
    def elem_i_upper(self):
        return max(self.elem_keys(), default=-1) + 1

    @property
    def next_valid_elem_i(self):
        return self.elem_i_upper

    def __contains__(self, i):
        return self.has(i)
    
    def paste_file(self, dst:int, file_handler:FilesHandle, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.paste_file_meta, force)(dst = dst, value = file_handler, **other_paras)

    def change_dir(self, dst:int, new_dir_name, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.change_dir_meta, force)(dst = dst, value = new_dir_name, **other_paras)

    def cache_to_file(self, data_i:int = None, *, force = False, **other_paras):
        rlt = False
        self.write_meta._set_ctrl_flag(ctrl_strategy=IO_CTRL_STRATEGY.FILE_IDPNDT, io_raw=True)
        data_i_list = [data_i] if isinstance(data_i, int) else self.data_keys()
        
        with self.get_writer(force).allow_overwriting():
            for data_i in tqdm(data_i_list, desc=f"write {self} to file", total=len(data_i_list), leave=False):
                fh = self.query_fileshandle(data_i)
                if fh.synced or not fh.has_cache:
                    continue
                value = fh.cache
                rlt = self.write_data(data_i, value, **other_paras)
                self.query_fileshandle(data_i)._set_synced(True)

        self.write_meta._clear_ctrl_flag()
        return rlt

    def file_to_cache(self, data_i:int = None, *, save = True, force = False, auto_decide = True, **other_paras):
        if auto_decide:
            sizes = [os.path.getsize(fh.get_path()) for fh in self.query_all_fileshandle()] # TODO
            mean_size = sum(sizes) / len(sizes)
            not_execute = mean_size > 1e4
            if not_execute:
                warnings.warn(f"{self}: the size of files is too large, so they won't be loaded into memory. " + \
                              "if you really want to load them, please set 'auto_decide' to False",
                                ClusterIONotExecutedWarning)
                return False
        
        rlt = False
        data_i_list = [data_i] if isinstance(data_i, int) else self.data_keys()
        
        self.read_meta._set_ctrl_flag(ctrl_strategy=IO_CTRL_STRATEGY.FILE_STRICK_IDPNDT, io_raw=True)
        self.write_meta._set_ctrl_flag(ctrl_strategy=IO_CTRL_STRATEGY.CACHE_STRICK_IDPNDT, io_raw=True)

        with self.get_writer(force).allow_overwriting():
            for data_i in tqdm(data_i_list, desc=f"load {self} to cache", total=len(data_i_list), leave=False):
                fh = self.query_fileshandle(data_i)
                if fh.synced or not fh.file_exist_status:
                    continue
                value = self.read_data(data_i, **other_paras)
                if value is None:
                    continue
                rlt = self.write_data(data_i, value, **other_paras)
                self.query_fileshandle(data_i)._set_synced(True)

        self.write_meta._clear_ctrl_flag()
        self.read_meta._clear_ctrl_flag()

        if save:
            self.save()
        return rlt

    def clear_files(self, *, force = False):
        orig_cache_priority = self.cache_priority
        self.cache_priority = False
        self.clear(force = force, clear_both=False)
        self.cache_priority = orig_cache_priority

    def clear_cache(self, *, force = False):
        orig_cache_priority = self.cache_priority
        self.cache_priority = True
        self.clear(force = force, clear_both=False)
        self.cache_priority = orig_cache_priority
    # endregion common operation ###

    # region - data operation #
    def read_data(self, src:int, *, force = False, **other_paras) -> VDMT:
        return self.io_decorator(self.read_meta, force)(src = src, **other_paras) # type: ignore

    def write_data(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.write_meta, force)(dst = dst, value = value, **other_paras)
    
    def modify_data_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.modify_key_meta, force)(src = src, dst = dst, **other_paras)
    
    def remove_data(self, dst:int, remove_both = False, *, force = False, **other_paras) -> None:
        if remove_both:
            self.remove_meta._set_ctrl_flag(ctrl_strategy=IO_CTRL_STRATEGY.CACHE_SYNC)
        rlt = self.io_decorator(self.remove_meta, force)(dst = dst, **other_paras)
        self.read_meta._clear_ctrl_flag()
        return rlt

    def merge_data_from(self, src_data_map:FCT, *, force = False):
        def merge_func(src_data_map:FCT, dst_data_map:FCT, src_data_i:int, dst_data_i:int):
            src_fh = src_data_map.query_fileshandle(src_data_i)
            dst_data_map.paste_file(
                dst_data_i,
                src_fh) # type: ignore
        
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        assert src_data_map.KEY_TYPE == int, f"the key_type of {src_data_map.__class__.__name__} is not int"
        assert type(src_data_map) == type(self), f"can't merge {type(src_data_map)} to {type(self)}"
        # assert self.opened and self.writable, f"{self.__class__.__name__} is not writable"
        assert src_data_map.opened, f"{src_data_map.__class__.__name__} is not opened"

        # if self.continuous:
        #     return

        with self.get_writer(force):
            for data_i in tqdm(src_data_map.keys(), desc=f"merge {src_data_map} to {self}", total=len(src_data_map)):
                merge_func(src_data_map, self, data_i, self.next_valid_data_i)

    def copy_data_from(self, src_data_map:FCT, *, cover = False, force = False):
        # if only_data_map:
        #     if os.path.exists(self.MemoryData_path):
        #         if cover:
        #             os.remove(self.MemoryData_path)
        #             shutil.copyfile(src_data_map.MemoryData_path, self.MemoryData_path)
        #             self.load()
        #         else:
        #             raise IOError(f"{self.MemoryData_path} already exists")
        # else:
        if os.path.exists(self.data_path) and len(self) > 0:
            if cover:
                shutil.rmtree(self.data_path)
                self.make_path()
                self.MemoryData.clear()
                self.save()
            else:
                raise IOError(f"{self.data_path} already exists")
        self.merge_from(src_data_map, force = force) 
    
    def clear_data(self, *, force = False, clear_both = True):
        if clear_both:
            with self.remove_meta._set_ctrl_flag(ctrl_strategy=IO_CTRL_STRATEGY.CACHE_SYNC):
                super().clear(force = force)
        else:
            super().clear(force = force)
            for fh in self.query_all_fileshandle():
                fh:FHT
                fh._set_synced(False)
    
    def data_keys(self):
        return self.MemoryData.keys()
    # endregion data operation ###
    
    # region - elem operation #
    def read_elem(self, src:int, *, force = False, **other_paras):
        raise NotImplementedError
        
    def write_elem(self, dst:int, value:VDMT, *, force = False, **other_paras):
        raise NotImplementedError

    def modify_elem_key(self, src:int, dst:int, *, force = False, **other_paras):
        raise NotImplementedError

    def remove_elem(self, dst:int, *, force = False, **other_paras):
        raise NotImplementedError
    
    def merge_elem_from(self, src_data_map:FCT, *, force = False):
        assert type(src_data_map) == type(self), f"src_data_map type {type(src_data_map)} != cluster type {type(self)}"
        assert len(src_data_map) == len(self), f"src_data_map length {len(src_data_map)} != cluster length {len(self)}"
        with self.get_writer(force).allow_overwriting():
            for elem_i in tqdm(src_data_map.elem_keys(), desc=f"merge {src_data_map} to {self}", total=src_data_map.elem_num):
                self.write_elem(self.next_valid_elem_i, src_data_map.read_elem(elem_i))
    
    def copy_elem_from(self, src_data_map:FCT, *, cover = False, force = False):
        if self.elem_num > 0:
            if cover:
                print("clearing elem...")
                self.clear_elem(force=True)
            else:
                raise DataMapExistError("can't copy to a non-empty cluster")
        self.merge_elem_from(src_data_map, force=force)
    
    def clear_elem(self, *, force = False, clear_both = True):
        with self.get_writer(force).allow_overwriting():
            for i in tqdm(list(self.elem_keys())):
                self.remove_elem(i)

    def elem_keys(self):
        raise NotImplementedError
    # endregion - elem operation ###

    # region - general #    
    def _switch_io_operation(self, io_data_func, io_elem_func):
        if self._ELEM_BY_CACHE:
            return io_elem_func
        else:
            return io_data_func

    @property
    def num(self):
        if self._ELEM_BY_CACHE:
            return self.elem_num
        else:
            return self.data_num
    
    @property
    def i_upper(self):
        if self._ELEM_BY_CACHE:
            return self.elem_i_upper
        else:
            return self.data_i_upper
        
    @property
    def next_valid_i(self):
        if self._ELEM_BY_CACHE:
            return self.next_valid_elem_i
        else:
            return self.next_valid_data_i

    def read(self, src:int, *, force = False, **other_paras) -> VDMT:
        rlt = self._switch_io_operation(self.read_data, self.read_elem)(src = src, force = force, **other_paras)
        self._send_rely(rlt)
        return rlt

    def write(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        self._send_rely(value)
        return self._switch_io_operation(self.write_data, self.write_elem)(dst = dst, value = value, force = force, **other_paras)
        
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        return self._switch_io_operation(self.modify_data_key, self.modify_elem_key)(src = src, dst = dst, force = force, **other_paras)
        
    def remove(self, dst:int, remove_both = False, *, force = False, **other_paras) -> None:
        return self._switch_io_operation(self.remove_data, self.remove_elem)(dst = dst, remove_both = remove_both, force = force, **other_paras)
    
    def merge_from(self, src_data_map:FCT, *, force = False):
        return self._switch_io_operation(self.merge_data_from, self.merge_elem_from)(src_data_map = src_data_map, force = force)
    
    def copy_from(self, src_data_map:FCT, *, cover = False, force = False):
        return self._switch_io_operation(self.copy_data_from, self.copy_elem_from)(src_data_map = src_data_map, cover = cover, force = force)
    
    def clear(self, *, force = False, clear_both = True):
        rlt = self._switch_io_operation(self.clear_data, self.clear_elem)(force = force, clear_both = clear_both)
        # if clear_both:
        #     self.remove_memory_data_file()
        return rlt
    
    def keys(self):
        return self._switch_io_operation(self.data_keys, self.elem_keys)()
    
    def idx_unwrited(self, idx):
        if self._ELEM_BY_CACHE:
            return not self.has_elem(idx)
        else:
            if self.has_data(idx):
                return self.query_fileshandle(idx).empty
            else:
                return True
    # endregion - general ###
    
    # endregion io #####

    # region Memorydata operation ####
    def matching_path(self):
        paths:list[str] = []
        paths.extend(glob.glob(os.path.join(self.data_path, "**/*"), recursive=True))
        return paths

    @classmethod
    def create_fh(cls, path):
        fh = cls.FILESHANDLE_TYPE.create_new_and_cover().from_path(cls, path)
        data_i = cls.deformat_corename(fh.corename)
        data_i = data_i if data_i is not None else cls.data_i_upper
        if fh.all_file_exist:
            cls._set_fileshandle(data_i, fh)
        else:
            cls.paste_file(data_i, fh)
        return fh

    def rebuild(self, force = False):
        if not self.MemoryData_modified and not force:
            return
        self.MemoryData.clear()
        paths:list[str] = self.matching_path()
        # # load the first file
        # fh = self.FILESHANDLE_TYPE.create_new_and_cover().from_path(self, paths[0])

        # prefix_joiner = fh._prefix_obj.joiner
        # appnames_joiner = fh._appendnames_obj.joiner

        # for path in tqdm(paths, desc=f"rebuilding {self}", leave=False):
        #     self.FILESHANDLE_TYPE.create_new_and_cover()()
    
        for path in tqdm(paths, desc=f"rebuilding {self}", leave=False):
            fh:FilesHandle = self.FILESHANDLE_TYPE.create_new_and_cover().from_path(self, path)
            data_i = self.deformat_corename(fh.corename)
            data_i = data_i if data_i is not None else self.data_i_upper
            if fh.all_file_exist:
                self._set_fileshandle(data_i, fh)
                # self.MemoryData[data_i] = fh
            else:
                self.paste_file(data_i, fh)

        for fh in list(self.MemoryData.values()):
            if fh.empty:
                self.remove(fh)

        self.sort()
    
    def merge_MemoryData(self, to_merge_MemoryData:FhBinDict[FHT]):
        assert type(to_merge_MemoryData) == self.MEMORY_DATA_TYPE, f"MemoryData type {type(to_merge_MemoryData)} != cluster type {type(self)}"
        same_fh:list[FilesHandle] = []
        for loaded_fh in to_merge_MemoryData.values(): # TODO: imporve speed
            if self.MemoryData.has_value(loaded_fh):
                this_fh = self.query_fileshandle(self.MemoryData.query_value(loaded_fh))
                assert loaded_fh.immutable_attr_same_as(this_fh), f"the fileshandle {loaded_fh} is not the same as {this_fh}"
                # cover cache
                this_fh.cache_proxy.cache = loaded_fh.cache_proxy.cache
                same_fh.append(loaded_fh)
        for fh in same_fh:
            to_merge_MemoryData.pop(fh.get_key(), None)
        self.MemoryData.update(to_merge_MemoryData)
        self._set_MemoryData_modified()

    def save_preprecess(self, MemoryData:FhBinDict[FHT] = None ):
        MemoryData = self.MemoryData if MemoryData is None else MemoryData
        to_save_dict = {item[0]: item[1].as_dict() for item in MemoryData.items()}
        return to_save_dict
    
    def load_and_process_data(self, key, value):
        processed_value = self.FILESHANDLE_TYPE.from_dict(self, value)
        return int(key), processed_value

    def load_postprocess(self, data:dict):
        if not data:
            return self.MEMORY_DATA_TYPE({})  # 返回一个空的 data_info_map

        new_dict = {int(k): None for k in data.keys()}
        # for k, v in tqdm(data.items(), desc=f"loading {self}", total=len(new_dict), leave=False):
        #     new_dict[int(k)] = self.FILESHANDLE_TYPE.from_dict(self, v)
        # keys, values = zip(*data.items())

        for k, v in tqdm(data.items(), desc=f"loading {self}", total=len(new_dict), leave=False):
            new_dict[int(k)] = self.FILESHANDLE_TYPE.from_dict(self, v)

        data_info_map = self.MEMORY_DATA_TYPE(new_dict)
        return data_info_map
    # endregion

    # region create instance ####
    @classmethod
    def from_cluster(cls:type[FCT], cluster:FCT, dataset_node:DSNT = None, mapping_name = None, *args, flag_name = "",  **kwargs) -> FCT:
        dataset_node    = cluster.dataset_node if dataset_node is None else dataset_node
        mapping_name    = cluster.mapping_name if mapping_name is None else mapping_name
        flag_name       = cluster.flag_name if flag_name == "" else flag_name

        init_paras = get_function_args(cls._orig_init, "dataset_node", "mapping_name", "flag_name", "self")
        para_dict = {}
        for para_name in init_paras:
            if hasattr(cluster, para_name):
                para_dict[para_name] = cluster.__getattribute__(para_name)

        new_cluster:FCT = cls(dataset_node, mapping_name, flag_name = flag_name, **para_dict)
        return new_cluster
    # endregion create instance ####

class DatasetNode(DataMapping[dict[str, bool], DSNT, VDST], ABC, Generic[FCT, DSNT, VDST]):
    """
    Represents a dataset node in the data mapping hierarchy.

    Attributes:
        MEMORY_DATA_TYPE (Table[int, str, bool]): The memory data type.
        MEMORY_DATA_FILE (str): The memory data file.
        load_memory_func (function): The function used to load memory data.
        save_memory_func (function): The function used to save memory data.
    """

    MEMORY_DATA_TYPE = Table[int, str, bool]
    MEMORY_DATA_FILE = ".overview"
    load_memory_func = partial(JsonIO.load_json, format = False)
    save_memory_func = JsonIO.dump_json 

    # region FilesCluster override ####

    # region - override new #
    # def init_identity(self, top_directory:str, *args, parent:"DatasetNode" = None, flag_name = "", **kwargs):

    #     self.__unfinished_operation = 0
    #     if parent is not None:
    #         self._top_directory = os.path.join(parent.data_path, top_directory)
    #         assert is_subpath(self.top_directory, parent.top_directory), f"{self.top_directory} is not inside {parent.top_directory}"
    #     else:
    #         self._top_directory = top_directory

    #     self.mapping_name:str = "" # self.top_directory
    #     self.flag_name = flag_name
    # endregion - override new #

    # region - override methods #
    def __init__(self, top_directory:Union[str, "DatasetNode"], mapping_name: str = "", *args, flag_name = "", lazy = False, **kwargs) -> None:
        """
        Initializes a DatasetNode instance.

        Args:
            top_directory (Union[str, DatasetNode]): The top directory of the dataset node.
            mapping_name (str, optional): The mapping name. Defaults to "".
            flag_name (str, optional): The flag name. Defaults to "".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(top_directory, mapping_name, *args, flag_name = flag_name,  **kwargs)
        if self.parent is None:
            print(f"initilizing {self}")
        # if parent is not None:
        #     super().__init__(os.path.join(parent.data_path, directory), "", flag_name=flag_name)
        # else:
        #     super().__init__(directory, "")

        self.init_dataset_attr_hook()
        
        self.clusters_map:dict[str, FCT] = dict()
        self.child_nodes_map:dict[str, DatasetNode] = dict()
        self.init_clusters_hook(lazy)

    @property
    def MemoryData(self) -> Table[int, str, bool]:
        """
        Gets the memory data.

        Returns:
            Table[int, str, bool]: The memory data.
        """
        return self._MemoryData

    def init_dataset_attr_hook(self):
        """
        Initializes the dataset attribute hook.
        """
        os.makedirs(self.top_directory, exist_ok=True) 
        self._cluster_use_rely = False
        self.__unfinished_operation = 0
        self.is_update_clusters = False
    # endregion - override methods #

    # endregion FilesCluster override ####

    # region maps ####

    # region - map operation #
    def __dsn_add_child(self, child:DataMapping, map:dict):
        """
        Adds a child to the data mapping.

        Args:
            child (DataMapping): The child to add.
            map (dict): The mapping to add the child to.
        
        Raises:
            KeyError: If the child already exists in the mapping with a different instance.
        """
        child.set_parent(self)
        # unique check
        if child.identity_name() in map and map[child.identity_name()] is not child:
            raise KeyError(f"{child.identity_name()} already exists, but the child not the same as the new one")
        else:
            map[child.identity_name()] = child
            self._add_child(child)
        # open
        if self.opened:
            child.open()
        # add column
        if isinstance(child, FilesCluster) and child._IS_ELEM:
            self.MemoryData.add_column(child.identity_name(), exist_ok=True)

    def __dsn_remove_child(self, child:DataMapping, map:dict):
        """
        Removes a child from the data mapping.

        Args:
            child (DataMapping): The child to remove.
            map (dict): The mapping to remove the child from.
        """
        if child.parent == self:
            self._remove_child(child)
            map.pop(child.identity_name())
            self.MemoryData.remove_column(child.identity_name(), not_exist_ok=True)  
    # endregion - map operation #
   
    # region - cluster_map ####
    def add_cluster(self, cluster:FCT):
        """
        Adds a cluster to the dataset node.

        Args:
            cluster (FCT): The cluster to add.
        """
        self.__dsn_add_child(cluster, self.clusters_map)

    def remove_cluster(self, cluster:FCT):
        """
        Removes a cluster from the dataset node.

        Args:
            cluster (FCT): The cluster to remove.
        """
        self.__dsn_remove_child(cluster, self.clusters_map)

    def get_cluster(self, identity_name:str):
        """
        Gets a cluster from the dataset node.

        Args:
            identity_name (str): The identity name of the cluster.

        Returns:
            FCT: The cluster.
        """
        return self.clusters_map[identity_name]

    def cluster_keys(self):
        """
        Gets the keys of the clusters in the dataset node.

        Returns:
            dict_keys: The keys of the clusters.
        """
        return self.clusters_map.keys()

    @property
    def clusters(self) -> list[FCT]:
        """
        Gets the clusters in the dataset node.

        Returns:
            list[FCT]: The clusters.
        """
        clusters = list(self.clusters_map.values())
        return clusters

    @property
    def elem_clusters(self) -> list[FCT]:
        """
        Gets the element clusters in the dataset node.

        Returns:
            list[FCT]: The element clusters.
        """
        clusters = [x for x in self.clusters if x._IS_ELEM]
        return clusters

    @property
    def opened_clusters(self):
        """
        Gets the opened clusters in the dataset node.

        Returns:
            list[FCT]: The opened clusters.
        """
        clusters = [x for x in self.clusters if x.opened]
        return clusters

    @property
    def opened_elem_clusters(self):
        """
        Gets the opened element clusters in the dataset node.

        Returns:
            list[FCT]: The opened element clusters.
        """
        clusters = [x for x in self.elem_clusters if x.opened]
        return clusters
    
    @property
    def clusters_num(self) -> int:
        """
        Gets the number of clusters in the dataset node.

        Returns:
            int: The number of clusters.
        """
        return len(self.clusters_map)

    def get_all_clusters(self, _type:Union[type, tuple[type]] = None, only_opened = False) -> dict[int, FCT]:
        """
        Gets all clusters in the dataset node.

        Args:
            _type (Union[type, tuple[type]], optional): The type of clusters to retrieve. Defaults to None.
            only_opened (bool, optional): Whether to retrieve only opened clusters. Defaults to False.

        Returns:
            dict[int, FCT]: A dictionary of clusters.
        """
        cluster_map:dict[str, FCT] = {}

        for k in self.cluster_keys():
            if k in cluster_map:
                raise NotImplementedError(f"the mapping_name {k} is already in cluster_map")
            cluster_map[k] = self.clusters_map[k]

        if self.linked_with_children:
            for k, v in copy.copy(cluster_map.items()):
                if _type is not None:
                    if not isinstance(v, _type):
                        cluster_map.pop(k)
                if only_opened:
                    if v.opened:
                        cluster_map.pop(k)

        for child_node in self.child_nodes:
            cluster_map.update(child_node.get_all_clusters(_type, only_opened))

        return cluster_map
    # endregion cluster_map END ####

    # region - child_nodes_map ####
    def add_child_node(self, child_node:"DatasetNode"):
        """
        Adds a child node to the dataset node.

        Args:
            child_node (DatasetNode): The child node to add.
        """
        self.__dsn_add_child(child_node, self.child_nodes_map)

    def remove_child_node(self, child_node:"DatasetNode"):
        """
        Removes a child node from the dataset node.

        Args:
            child_node (DatasetNode): The child node to remove.
        """
        self.__dsn_remove_child(child_node, self.child_nodes_map)

    def get_child_node(self, identity_name:str):
        """
        Gets a child node from the dataset node.

        Args:
            identity_name (str): The identity name of the child node.

        Returns:
            DatasetNode: The child node.
        """
        return self.child_nodes_map[identity_name]
    
    def child_node_keys(self):
        """
        Gets the keys of the child nodes in the dataset node.

        Returns:
            dict_keys: The keys of the child nodes.
        """
        return self.child_nodes_map.keys()
    
    @property
    def child_nodes(self) -> list["DatasetNode"]:
        return list(self.child_nodes_map.values())
    
    @property
    def child_node_num(self) -> int:
        return len(self.child_nodes_map)
    # endregion
    
    # endregion

    # region node and cluster operation #
    def operate_clusters(self, func:Union[Callable, str], *args, **kwargs):
        for obj in self.clusters:
            self.operate_one_cluster(obj, func, *args, **kwargs)

    def operate_children_node(self, func:Union[Callable, str], *args, **kwargs):
        for child_node in self.child_nodes:
            self.operate_one_child_node(child_node, func, *args, **kwargs)

    def operate_one_cluster(self, cluster, func:Union[Callable, str], *args, **kwargs):
        func_name = func.__name__ if isinstance(func, Callable) else func
        cluster.__getattribute__(func_name)(*args, **kwargs)

    def operate_one_child_node(self, child_node:"DatasetNode", func:Union[Callable, str], *args, **kwargs):
        func_name = func.__name__ if isinstance(func, Callable) else func
        if self.linked_with_children and child_node.follow_parent:
            child_node.__getattribute__(func_name)(*args, **kwargs)
    
    # def close_hook(self):
    #     super().close_hook()
    #     self.operate_clusters(FilesCluster.close)
    #     self.operate_children_node(DatasetNode.close)

    # def open_hook(self):
    #     super().open_hook()
    #     self.operate_clusters(FilesCluster.open)
    #     self.operate_children_node(DatasetNode.open)

    # def readonly_hook(self):
    #     super().readonly_hook()
    #     self.operate_clusters(FilesCluster.set_readonly)
    #     self.operate_children_node(DatasetNode.set_readonly)

    # def writable_hook(self):
    #     super().writable_hook()
    #     self.operate_clusters(FilesCluster.set_writable)
    #     self.operate_children_node(DatasetNode.set_writable)

    # def stop_writing_hook(self):
    #     super().stop_writing_hook()
    #     self.operate_clusters(FilesCluster.stop_writing)
    #     self.operate_children_node(DatasetNode.stop_writing)

    # def start_writing_hook(self):
    #     super().start_writing_hook()
    #     self.operate_clusters(FilesCluster.start_writing)
    #     self.operate_children_node(DatasetNode.start_writing)

    # def set_overwrite_allowed_hook(self):
    #     super().set_overwrite_allowed_hook()
    #     self.operate_clusters(FilesCluster.set_overwrite_allowed)
    #     self.operate_children_node(DatasetNode.set_overwrite_allowed)

    # def set_overwrite_forbidden_hook(self):
    #     super().set_overwrite_forbidden_hook()
    #     self.operate_clusters(FilesCluster.set_overwrite_forbidden)
    #     self.operate_children_node(DatasetNode.set_overwrite_forbidden)
    # endregion node and cluster operation #

    # region clusters #####
    @property
    def cluster_use_rely(self):
        return self._cluster_use_rely
    
    @cluster_use_rely.setter
    def cluster_use_rely(self, value):
        self._cluster_use_rely = value
        if value == True and not self.cluster_use_rely:
            for cluster in self.clusters:
                cluster._clear_rely()

    def cluster_use_rely_decorator(func):
        def cluster_use_rely_decorator_wrapper(self:"DatasetNode", *args, **kwargs):
            self.cluster_use_rely = True
            rlt = func(self, *args, **kwargs)
            self.cluster_use_rely = False
            return rlt
        return cluster_use_rely_decorator_wrapper

    def init_clusters_hook(self, lazy):
        pass

    def __setattr__(self, name, value):
        ### 赋值DataMapping对象时，自动注册
        ### 同名变量赋值时，自动将原有对象解除注册
        if isinstance(value, (FilesCluster, DatasetNode)):
            if not hasattr(self, name):
                pass
            else:
                obj = self.__getattribute__(name)
                if id(value) != id(obj):
                    if isinstance(obj, DatasetNode) and obj.parent == self:
                        self._remove_child(obj)
                    elif isinstance(obj, FilesCluster) and obj.dataset_node == self:
                        self.remove_cluster(obj)
            # auto register     
            if isinstance(value, DatasetNode):
                self.add_child_node(value)
            elif isinstance(value, FilesCluster):
                self.add_cluster(value)
        super().__setattr__(name, value)
    # endregion clusters #

    # region io ###
    @cluster_use_rely_decorator
    def raw_read(self, src, *, force = False, **other_paras) -> dict[str, Any]:
        read_rlt = {}
        with self.get_writer(force).allow_overwriting():
            for obj in self.elem_clusters:
                rlt = obj.read(src, **other_paras)
                read_rlt[obj.identity_name()] = rlt
        return read_rlt
    
    @cluster_use_rely_decorator
    def raw_write(self, dst, values:dict[str, Any], *, force = False, **other_paras) -> None:
        assert len(values) == len(self.elem_clusters), f"the length of values {len(values)} != the length of clusters {len(self.elem_clusters)}"
        if isinstance(values, (list, tuple)):
            # cvt to dict
            warnings.warn("the values is list or tuple and has been converted to dict, but a dict is recommended")
            values = {self.elem_clusters[i].identity_name(): values[i] for i in range(len(values))}
        with self.get_writer(force).allow_overwriting():
            for obj in self.elem_clusters:
                obj.write(dst, values[obj.identity_name()], **other_paras)

    def read(self, src:int) -> VDST:
        return self.raw_read(src)

    def write(self, dst:int, value:VDST, *, force = False, **other_paras) -> None:
        return self.raw_write(dst, value, force = force, **other_paras)
        
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        with self.get_writer(force).allow_overwriting():
            for obj in self.elem_clusters:
                obj.modify_key(src, dst, **other_paras)

    def remove(self, dst:int, *, force = False, **other_paras) -> None:
        with self.get_writer(force).allow_overwriting():
            for obj in self.elem_clusters:
                obj.remove(dst, **other_paras)

    def merge_from(self, src_dataset_node:DSNT, *, force = False) -> None:
        assert type(src_dataset_node) == type(self), f"can't merge {type(src_dataset_node)} to {type(self)}"
        if self.clusters_num > 0:
            assert src_dataset_node.clusters_num == self.clusters_num, f"the clusters_num of {src_dataset_node} is not equal to {self}"
       
        if self.clusters_num == 0:
            # add clusters
            for cluster_name in src_dataset_node.cluster_keys():
                src_cluster = src_dataset_node.clusters_map[cluster_name]
                self.add_cluster(src_cluster.__class__.from_cluster(src_cluster, self))
        
        with self.get_writer(force).allow_overwriting():  
            for cluster_name in self.cluster_keys():
                this_cluster = self.clusters_map[cluster_name]
                src_cluster = src_dataset_node.clusters_map[cluster_name]
                self.operate_one_cluster(this_cluster, FilesCluster.merge_from, src_cluster)
            for this_child_node in self.child_nodes:
                src_child_node = src_dataset_node.get_child(this_child_node.identity_name())
                self.operate_one_child_node(this_child_node, DatasetNode.merge_from, src_child_node)

    def copy_from(self, src_dataset_node:DSNT, *, cover = False, force = False) -> None:
        if self.num > 0:
            if cover:
                print(f"clear {self}")
                self.clear(force = force)
            else:
                raise DataMapExistError(f"{self} is not empty")
        self.merge_from(src_dataset_node, force = force)

    def clear(self, *, force = False, clear_both = True, clear_completely = False):
        if clear_completely:
            self.close()
            shutil.rmtree(self.data_path)
        else:
            with self.get_writer(force).allow_overwriting():
                self.operate_clusters(DataMapping.clear, clear_both = clear_both)
                self.operate_children_node(DataMapping.clear, clear_both = clear_both)
    # endregion io ###

    # region cache operation ######
    @Node.downward_preorder_propagate
    def cache_to_file(self, *, force = False):
        pass
        # self.operate_clusters(FilesCluster.cache_to_file, force=force)
        # self.operate_children_node(DatasetNode.all_cache_to_file, force=force)

    @Node.downward_preorder_propagate
    def file_to_cache(self, *, force = False):
        pass
        # self.operate_clusters(FilesCluster.file_to_cache, force=force)
        # self.operate_children_node(DatasetNode.all_file_to_cache, force=force)

    @Node.downward_preorder_propagate
    def clear_files(self, *, force = False):
        pass
    
    @Node.downward_preorder_propagate
    def clear_cache(self, *, force = False):
        pass
    # endregion cache operation ######

    # region Memorydata operation ####
    @Node.downward_preorder_propagate
    def update_overview(self, log_type, src, dst, value, cluster:FilesCluster):
        if not cluster._IS_ELEM:
            return
        col_name = cluster.identity_name()
        is_child = cluster in self.clusters
        if log_type == self.LOG_READ or\
        log_type == self.LOG_CHANGE or\
        log_type == self.LOG_OPERATION:
            return
        if log_type == self.LOG_ADD:
            if self.MemoryData.add_row(dst, exist_ok=True):
                self.log_to_mark_file(log_type, src, dst, None)
            if is_child:
                self.MemoryData[dst, col_name] = True

        if log_type == self.LOG_REMOVE and dst in self.MemoryData:
            if is_child:
                self.MemoryData[dst, col_name] = False
            if self._clear_empty_row(dst):
                self.operate_children_node(DatasetNode._clear_empty_row, dst, force=True)
                self.log_to_mark_file(log_type, src, dst, value)

        if log_type == self.LOG_MOVE:
            self.MemoryData.add_row(dst, exist_ok=True) 
            if is_child:
                self.MemoryData[dst, col_name] = True
                self.MemoryData[src, col_name] = False
            if self._clear_empty_row(src):
                self.operate_children_node(DatasetNode._move_row, src, dst)
                self.log_to_mark_file(log_type, src, dst, value)

        # update clusters
        if not is_child and not self.is_update_clusters:
            self.is_update_clusters = True
            self.update_clusters(log_type, src, dst, value, cluster)
            self.is_update_clusters = False

    def update_clusters(self, log_type, src, dst, value, cluster):
        raise NotImplementedError

    @Node.downward_postorder_propagate
    def rebuild(self, force = False):
        # TODO rebuild 必须基于叶的rebuild
        if len(self.elem_clusters) > 0:
            rows = [x for x in range(self.i_upper)]
            cols = [x.identity_name() for x in self.elem_clusters]
            self._MemoryData = Table(row_names = rows, col_names = cols, default_value_type = bool, row_name_type=int, col_name_type=str) # type: ignore
            
            i_upper = max([x.i_upper for x in self.elem_clusters])            
            build_data_i_list = list(range(i_upper))
            self.MemoryData.add_new_row_in_batch(build_data_i_list)
            for data_i in tqdm(build_data_i_list, desc=f"rebuilding {self}"):
                self.__calc_overview(data_i)

    @Node.downward_postorder_propagate
    def build_partly(self, build_data_i_list:Union[int, Iterable[int]]):
        if len(self.elem_clusters) > 0:
            i_upper = max([x.i_upper for x in self.elem_clusters])
            build_data_i_list = list(set(build_data_i_list).intersection(range(i_upper)))
            self.MemoryData.add_new_row_in_batch(build_data_i_list)
            for data_i in tqdm(build_data_i_list, desc=f"building {self}"):
                self.__calc_overview(data_i)

    def merge_MemoryData(self, MemoryData:Table[int, str, bool]):
        # self.MemoryData.merge(MemoryData)
        self.MemoryData.update(MemoryData)

    def relugar_MemoryData(self):
        self.MemoryData.sort()
        for row_i in list(self.MemoryData.keys()):
            if not any(self.MemoryData.get_row(row_i).values()):
                self.MemoryData.remove_row(row_i)

    def save_preprecess(self, MemoryData:Table = None ):
        MemoryData = self.MemoryData if MemoryData is None else MemoryData
        to_save_dict = dict(MemoryData.data)
        for row_i in MemoryData.data.keys():
            if not any(to_save_dict[row_i].values()):
                to_save_dict.pop(row_i)
        return to_save_dict
    
    def load_postprocess(self, data:dict):
        data_ = {}
        for k, v in data.items():
            data_[int(k)] = v
        data_info_map = self.MEMORY_DATA_TYPE(default_value_type=bool, row_name_type=int, col_name_type=str, data=data_)
        return data_info_map

    def __calc_overview(self, data_i):
        to_update:dict[str, bool] = {}
        for cluster in self.elem_clusters:
            to_update[cluster.identity_name()] = cluster.has(data_i)
        self._MemoryData._GET_DATA_UNSAFELY()[data_i] = to_update
            
    def _clear_empty_row(self, data_i:int, force = False):
        if (data_i in self.MemoryData) and (not any(self.MemoryData.get_row(data_i).values()) or force):
            self.MemoryData.remove_row(data_i, not_exist_ok=True)
            return True
        else:
            return False
        
    def _move_row(self, src, dst):
        self.MemoryData.add_row(dst, exist_ok=True)
        self.MemoryData.set_row(dst, self.MemoryData.get_row(src))
        self.MemoryData.remove_row(src, not_exist_ok=True)
            
    # def rebuild_all(self):
    #     self.rebuild()
    #     self.operate_clusters(FilesCluster.rebuild)
    #     self.operate_children_node(DatasetNode.rebuild_all)

    def save_all(self, force = False):
        self.save(force=force)
        self.operate_clusters(FilesCluster.save, force=force)
        self.operate_children_node(DatasetNode.save_all, force=force)
    # endregion Memorydata operation ####

    # # region 
    # def _set_unfinished_operation(self, unfinished_operation):
    #     super()._set_unfinished_operation(unfinished_operation)
    #     self.operate_clusters(FilesCluster._set_unfinished_operation, unfinished_operation)
    #     self.operate_children_node(DatasetNode._set_unfinished_operation, unfinished_operation)
    # # endregion

