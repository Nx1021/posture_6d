import os
import numpy as np
from _ctypes import PyObj_FromPtr
import json
from json import JSONDecodeError
import re
from typing import Any
import time
import io
import re
import pickle
import warnings
import cv2

from typing import Generic, TypeVar, Union, Callable, Iterable, Type, Mapping
import types
from collections import OrderedDict


def get_bbox_connections(bbox_3d_proj:np.ndarray):
    '''
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

def modify_class_id(dict_list:list[dict[int, Any]], modify_class_id_pairs:list[tuple[int]]):
    orig_keys = [x[0] for x in modify_class_id_pairs]
    new_keys  = [x[1] for x in modify_class_id_pairs]
    assert len(orig_keys) == len(set(orig_keys))
    assert len(new_keys)  == len(set(new_keys))
    assert all([len(x) == 2 for x in modify_class_id_pairs])
    for orig_dict in dict_list:
        new_dict = {}
        for pair in modify_class_id_pairs:
            if pair[0] in orig_dict:
                new_dict[pair[1]] = orig_dict[pair[0]]
        orig_dict.clear()
        orig_dict.update(new_dict)    

def get_meta_dict(obj):
    orig_dict_list = []
    for name, orig_dict in vars(obj).items():
        if isinstance(orig_dict, dict) and all([isinstance(x, int) for x in orig_dict.keys()]):
            orig_dict_list.append(orig_dict)
    return orig_dict_list

def extract_doc(doc:str, title:str):
    idx = doc.find(title)
    sub_doc = doc[idx:]
    idx = re.search(r'\n\s*?\n', sub_doc).start()
    sub_doc = sub_doc[:idx]
    return sub_doc

def serialize_object(file_path, obj):
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

def is_image(array):
    if not isinstance(array, np.ndarray):
        return False
    if array.ndim not in [2, 3]:
        return False
    if array.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        return False
    return True

def test_pickleable(obj):
    temp_path = "./__pickle_test__.temp"
    try:
        serialize_object(temp_path, obj)
    except:
        pickleable = False
    else:
        pickleable = True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return pickleable

def read_file_as_str(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def write_str_to_file(file_path, string):
    with open(file_path, 'w') as file:
        file.write(string)

def _ignore_warning(func, category = Warning):
    def warpper(*args, **kwargs):
        warnings.filterwarnings("ignore", category=category) # do not show warning of image size
        rlt = func(*args, **kwargs)
        warnings.filterwarnings("default", category=category) # recover warning of image size
        return rlt
    return warpper


class JsonIO():
    class _NoIndent(object):
        """ Value wrapper. """
        def __init__(self, value):
            self.value = value

    class _MyEncoder(json.JSONEncoder):
        FORMAT_SPEC = '@@{}@@'
        regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

        def __init__(self, **kwargs):
            # Save copy of any keyword argument values needed for use here.
            self.__sort_keys = kwargs.get('sort_keys', None)
            super(JsonIO._MyEncoder, self).__init__(**kwargs)

        def default(self, obj):
            return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, JsonIO._NoIndent)
                    else super(JsonIO._MyEncoder, self).default(obj))

        def encode(self, obj):
            format_spec = self.FORMAT_SPEC  # Local var to expedite access.
            json_repr = super(JsonIO._MyEncoder, self).encode(obj)  # Default JSON.

            # Replace any marked-up object ids in the JSON repr with the
            # value returned from the json.dumps() of the corresponding
            # wrapped Python object.
            for match in self.regex.finditer(json_repr):
                # see https://stackoverflow.com/a/15012814/355230
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                json_repr = json_repr.replace(
                                '"{}"'.format(format_spec.format(id)), json_obj_repr)

            return json_repr

    class Stream():
        def __init__(self, path, open = False, buffer_length = 100000) -> None:
            self.path = path
            self.buffer = ""
            self.buffer_length = buffer_length
            self._closed = True
            if open:
                self.open()

        @property
        def closed(self):
            return self._closed
        
        @closed.setter
        def closed(self, value):
            value = bool(value)
            if value == True:
                self.close()
            else:
                self.open()
            self._closed = value

        def open(self):
            # 
            if not self.closed:
                return
            print("open JsonIO stream of {}".format(self.path))
            if os.path.exists(self.path):
                try:
                    with open(self.path, 'rb+') as f:
                        f.seek(-3, 2)
                        f.truncate()
                    with open(self.path, 'a') as f:
                        f.write(",")
                except OSError:
                    pass
            else:
                with open(self.path, 'w') as f:
                    f.write("{")   
            self._closed = False         

        def close(self):
            if self.closed:
                return
            print("close JsonIO stream of {}".format(self.path))
            self.save_buffer()
            with open(self.path, 'rb+') as f:
                f.seek(-1, 2)
                f.truncate()
            with open(self.path, 'a') as f:
                f.write('\n}')
            self._closed = True

        def save_buffer(self):
            with open(self.path, 'a') as f:
                f.write(self.buffer)
            self.buffer = ""            

        def write(self, to_dump_dict):
            string = JsonIO._dumps(to_dump_dict)
            self.buffer += string
            if len(self.buffer) > self.buffer_length:
                self.save_buffer()

        def __del__(self):
            self.close()

    @staticmethod
    def create_stream(path):
        stream = JsonIO.Stream(path)
        return stream

    @staticmethod
    def __convert_formatdict_from_json(dictionary, cvt_list_to_array = True):
        def cvt_key(key):
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    pass
            return key

        def cvt_value(value):
            if isinstance(value, list):
                try:
                    if cvt_list_to_array:
                        array = np.array(value)
                        if np.issubdtype(array.dtype, np.number):
                            new_value = array
                        else:
                            raise ValueError
                    else:
                        new_value = value
                except ValueError:
                    new_value = []
                    for item in value:
                        if isinstance(item, dict):
                            item = JsonIO.__convert_formatdict_from_json(item, cvt_list_to_array)
                        else:
                            item = cvt_value(item)
                        new_value.append(item)
            elif isinstance(value, dict):
                new_value = JsonIO.__convert_formatdict_from_json(value, cvt_list_to_array)
            else:
                new_value = value
            return new_value

        new_dict = {}
        for key, value in dictionary.items():
            new_key = cvt_key(key)
            new_value = cvt_value(value)

            new_dict[new_key] = new_value
        return new_dict

    @staticmethod
    def __convert_dict_to_jsonformat(dictionary, regard_list_as_array = False):
        def cvt_key(key):
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    pass
            if isinstance(key, np.intc):
                key = int(key)
            return key

        def cvt_value(value):
            if isinstance(value, np.intc):
                new_value = int(value)
            if isinstance(value, np.ndarray):
                new_value = np.around(value, decimals=4).tolist()
                new_value = JsonIO._NoIndent(new_value)
            elif isinstance(value, list):
                if regard_list_as_array:
                    new_value = value
                    new_value = JsonIO._NoIndent(new_value)
                else:
                    new_value = []
                    for item in value:
                        if isinstance(item, dict):
                            item = JsonIO.__convert_dict_to_jsonformat(item, regard_list_as_array)
                        else:
                            item = cvt_value(item)
                        new_value.append(item)
            elif isinstance(value, dict):
                new_value = JsonIO.__convert_dict_to_jsonformat(value, regard_list_as_array)
                if not any([isinstance(x, JsonIO._NoIndent) for x in new_value.values()]) and\
                list(new_value.values()) == list(value.values()):
                    new_value = JsonIO._NoIndent(new_value)
            else:
                new_value = value
            return new_value

        new_dict = {}
        for key, value in dictionary.items():
            new_key = cvt_key(key)
            new_value = cvt_value(value)
            new_dict[new_key] = new_value
        return new_dict

    @staticmethod
    def load_json(path, format = True, cvt_list_to_array = True):
        with open(path, 'r') as jf:
            dict_ = json.load(jf)
        if format:
            dict_ = JsonIO.__convert_formatdict_from_json(dict_, cvt_list_to_array)
        return dict_

    @staticmethod
    def _dumps(to_dump_dict, regard_list_as_array = False):
        to_dump_dict = JsonIO.__convert_dict_to_jsonformat(to_dump_dict, regard_list_as_array)
        string = ""            
        for k, v in to_dump_dict.items():
            json_data = json.dumps({k: v}, cls=JsonIO._MyEncoder, ensure_ascii=False, sort_keys=True, indent=2)
            string += json_data[1:-2] + ','
        return string

    @staticmethod
    def dump_json(path, to_dump_dict, regard_list_as_array = False):
        string = JsonIO._dumps(to_dump_dict, regard_list_as_array)
        string = '{' + string[:-1] + '\n}'
        with open(path, 'w') as fw:
            fw.write(string)

KT = TypeVar("KT")
VT = TypeVar("VT")
def search_in_dict(_dict:dict[KT, VT], 
                   key:Union[int, str], 
                   process_func:Callable = None):
    key = int_str_cocvt(_dict.keys(), key, return_index=False, process_func=process_func)
    return _dict[key]

def int_str_cocvt(ref_iterable:Iterable[str], 
                   query:Union[int, str], 
                   return_index=False, 
                   process_func:Callable = None):
    def default_func(string):
        return string 

    ref_iterable = tuple(ref_iterable)

    if process_func is None:
        process_func:Callable = default_func

    if isinstance(query, int):
        key = ref_iterable[query]
        return query if return_index else key
    elif isinstance(query, str):
        if query in ref_iterable:
            return ref_iterable.index(query) if return_index else query
        else:
            matching_keys = [query == process_func(k) for k in ref_iterable]
            if sum(matching_keys) == 1:
                idx = matching_keys.index(True)
                return idx if return_index else ref_iterable[idx]
            else:
                raise ValueError("Key not found or ambiguous")
    else:
        raise TypeError("Key must be an int or str")


ITEM = TypeVar('ITEM')
ROWKEYT = TypeVar('ROWKEYT', int, str)
COLKETT = TypeVar('COLKETT', int, str)
class Table(Generic[ROWKEYT, COLKETT, ITEM]):
    '''
    # 使用示例
    table = Table()
    table.add_column("Name")
    table.add_column("Age")
    table.add_row("Row1")
    table.add_row("Row2")

    # 增加数据
    table["Row1", "Name"] = "Alice"
    table["Row1", "Age"] = 25
    table["Row2", "Name"] = "Bob"
    table["Row2", "Age"] = 30

    print(table.get_row("Row1"))  # 输出: {'Name': ['Alice'], 'Age': [25]}
    print(table.get_column("Age"))  # 输出: {'Row1': [25], 'Row2': [30]}

    # 删除数据
    del table["Row1", "Age"]
    print(table.get_row("Row1"))  # 输出: {'Name': ['Alice'], 'Age': []}

    '''

    KW_EMPTY = "empty"
    KW_row_names = "row_names"
    KW_col_names = "col_names"
    KW_default_value_type = "default_value_type"
    KW_row_name_type = "row_name_type"
    KW_col_name_type = "col_name_type"
    KW_data = "data"

    def __init__(self, data:dict = None, *, 
                 row_names:list[ROWKEYT] = None, 
                 col_names:list[COLKETT] = None, 
                 default_value_type:type[ITEM] = None,
                 row_name_type = str, 
                 col_name_type = str):
        self.__data:dict[ROWKEYT, dict[COLKETT, ITEM]] = {}
        self.__row_names:list[ROWKEYT] = []
        self.__col_names:list[COLKETT] = []
        self.__default_value_type = default_value_type
        data = {} if data is None else data
        assert isinstance(data, (Mapping, Table)), "data must be a dict"
        if isinstance(data, Table):
            data = data.to_dict()
        data = dict(data)
        if len(data) > 0:
            row_name_type, col_name_type = self.get_type(data, row_name_type, col_name_type)
            self.__row_name_type:Type[ROWKEYT] = row_name_type
            self.__col_name_type:Type[COLKETT] = col_name_type
            self.update(data)
        else:
            self.__row_name_type:Type[ROWKEYT] = row_name_type
            self.__col_name_type:Type[COLKETT] = col_name_type

            for row_name in row_names or []:
                self.add_row(row_name)
            for col_name in col_names or []:
                self.add_column(col_name)

    @staticmethod
    def __type_process(self, value:Union[str, type], orig_list:list, orig_flag):
        if isinstance(value, str):
            assert value in ['int', 'str']
            if value == 'int':
                value = int
            elif value == 'str':
                value = str
        assert isinstance(value, type)
        if orig_flag != value:
            assert all([isinstance(x, value) for x in orig_list]), "row_name_type cannot be changed"
        return value

    @staticmethod
    def __name_filter(name, _type, _names):
        if _type == int:
            name = int(name)
        elif _type == str:
            name = int_str_cocvt(_names, name)
        return name

    @staticmethod
    def __key_assert(name, target:list[ROWKEYT], is_required, ignore = False):
        if ((name in target) ^ is_required):
            if ignore:
                return False
            else:
                raise ValueError(f"Key '{name}' {'does not exist' if is_required else 'already exists'}")
        return True

    @property
    def data(self):
        return types.MappingProxyType(self.__data)
    
    @property
    def row_names(self):
        return tuple(self.__row_names)
    
    @property
    def col_names(self):
        return tuple(self.__col_names)
    
    @property
    def default_value_type(self):
        return self.__default_value_type
    
    @default_value_type.setter
    def default_value_type(self, value):
        assert isinstance(value, type)
        if self.__default_value_type is not None:
            warnings.warn(f"default_value_type has been set from {self.__default_value_type} to {value}, but it is not recommended")
        self.__default_value_type = value

    @property
    def row_name_type(self):
        return self.__row_name_type

    @property
    def col_name_type(self):
        return self.__col_name_type    

    @row_name_type.setter
    def row_name_type(self, value:Union[str, type]):
        self.__row_name_type = self.__type_process(value, self.__row_names, self.__row_name_type)
    
    @col_name_type.setter
    def col_name_type(self, value:Union[str, type]):
        self.__col_name_type = self.__type_process(value, self.__col_names, self.__col_name_type)

    @property
    def empty(self):
        return len(self.__data) == 0

    def gen_default_value(self)->ITEM:
        if self.__default_value_type is not None:
            return self.__default_value_type()
        return None

    def assert_default_value_type(self, value):
        if self.__default_value_type is not None:
            assert isinstance(value, self.__default_value_type), "value must be an instance of default_value_type"

    def _row_name_filter(self, row_name:ROWKEYT) -> ROWKEYT:
        return self.__name_filter(row_name, self.__row_name_type, self.__row_names)
    
    def _col_name_filter(self, col_name:COLKETT) -> COLKETT:
        return self.__name_filter(col_name, self.__col_name_type, self.__col_names)

    def add_row(self, row_name:ROWKEYT, exist_ok=False):
        assert isinstance(row_name, self.row_name_type), "row_name must be an instance of row_name_type"
        if self.__key_assert(row_name, self.__row_names, False, ignore=exist_ok):
            self.__row_names.append(row_name)
            self.__data[row_name] = {col_name: self.gen_default_value() for col_name in self.__col_names}
            return True
        return False

    def remove_row(self, row_name:Union[int,str], not_exist_ok=False):
        row_name = self._row_name_filter(row_name)
        if self.__key_assert(row_name, self.__row_names, True, ignore=not_exist_ok):
            del self.__data[row_name]
            self.__row_names.remove(row_name)
            return True
        return False

    def add_column(self, col_name:COLKETT, exist_ok=False):
        assert isinstance(col_name, self.col_name_type)
        if self.__key_assert(col_name, self.__col_names, False, ignore=exist_ok):
            self.__col_names.append(col_name)
            for row_name in self.__row_names:
                self.__data[row_name][col_name] = self.gen_default_value()
            return True
        return False

    def remove_column(self, col_name:Union[int,str], not_exist_ok=False):
        col_name = self._col_name_filter(col_name)
        if self.__key_assert(col_name, self.__col_names, True, ignore=not_exist_ok):
            for row_name in self.__row_names:
                del self.__data[row_name][col_name]
            self.__col_names.remove(col_name)
            return True
        return False

    def resort_row(self, new_row_names:list[Union[int,str]]):
        new_row_names = [self._row_name_filter(row_name) for row_name in new_row_names]
        assert set(new_row_names) == set(self.__row_names), "new_row_names must contain all row names"
        if new_row_names == self.__row_names:
            return
        new_data = {}
        for row_name in new_row_names:
            new_data[row_name] = self.__data[row_name]
        self.__data = new_data
        self.__row_names = new_row_names

    def resort_column(self, new_col_names:list[Union[int,str]]):
        new_col_names = [self._col_name_filter(col_name) for col_name in new_col_names]
        assert set(new_col_names) == set(self.__col_names), "new_col_names must contain all col names"
        if new_col_names == self.__col_names:
            return
        for row_name in self.__row_names:
            new_row = {}
            for col_name in new_col_names:
                new_row[col_name] = self.__data[row_name][col_name]
            self.__data[row_name] = new_row
        self.__col_names = new_col_names

    def move_row(self, row_name:Union[int,str], new_name:Union[int,str]):
        row = self.get_row(row_name)
        self.remove_row(row_name)
        self.add_row(new_name)
        self.set_row(new_name, row)

    def move_column(self, col_name:Union[int,str], new_name:Union[int,str]):
        col = self.get_column(col_name)
        self.remove_column(col_name)
        self.add_column(new_name)
        self.set_column(new_name, col)

    def get_row(self, row_name:Union[int,str]) -> dict[COLKETT, ITEM]:
        row_name = self._row_name_filter(row_name)
        if self.__key_assert(row_name, self.__row_names, True):
            return types.MappingProxyType(self.__data[row_name])

    def get_column(self, col_name:Union[int,str]) -> dict[ROWKEYT, ITEM]:
        col_name = self._col_name_filter(col_name)
        if self.__key_assert(col_name, self.__col_names, True):
            return {row_name: self.__data[row_name][col_name] for row_name in self.__row_names}

    def set_row(self, row_name:Union[int,str], row:dict[COLKETT, ITEM]):
        row_name = self._row_name_filter(row_name)
        if self.__key_assert(row_name, self.__row_names, True):
            for col_name, value in row.items():
                self.__data[row_name][col_name] = value

    def set_column(self, col_name:Union[int,str], col:dict[ROWKEYT, ITEM]):
        col_name = self._col_name_filter(col_name)
        if self.__key_assert(col_name, self.__col_names, True):
            for row_name, value in col.items():
                self.__data[row_name][col_name] = value

    def tranverse(self, with_key=False):
        for row_name in self.__row_names:
            for col_name in self.__col_names:
                if with_key:
                    yield row_name, col_name, self.__data[row_name][col_name]
                else:
                    yield self.__data[row_name][col_name]

    def update(self, other:Union[dict[ROWKEYT, dict[COLKETT, ITEM]], "Table[ROWKEYT, COLKETT, ITEM]"]):
        for row_name, row in other.items():
            self.add_row(row_name, exist_ok=True)
            for col_name, value in row.items():
                self.add_column(col_name, exist_ok=True)
                self.__data[row_name][col_name] = value
    
    def clear(self):
        self.__data.clear()
        self.__row_names.clear()

    def sort(self, key:Callable = None, reverse:bool = False):
        self.__row_names.sort(key=key, reverse=reverse)
        self.__data = {row_name: self.__data[row_name] for row_name in self.__row_names}

    def merge(self, other:dict[str, dict[str, ITEM]], merge_func:Callable = None):
        merge_func = lambda x, y: True
        assert callable(merge_func)
        for row_name, row in other.items():
            self.add_row(row_name, exist_ok=True)
            for col_name, value in row.items():
                self.add_column(col_name, exist_ok=True)
                if merge_func(self.__data[row_name][col_name], value):
                    self.__data[row_name][col_name] = value

    def clean_invalid(self, judge_invalid_func:Callable):
        assert callable(judge_invalid_func)
        for row_name in self.__row_names:
            for col_name in self.__col_names:
                if judge_invalid_func(self.__data[row_name][col_name]):
                    self.__data[row_name][col_name] = self.gen_default_value()

    def __process_item(self, keys:Union[int, str, tuple[Union[int, str]], slice]):
        def get_key_list(key, names, filter_func:Callable):
            if isinstance(key, slice):
                key_list = names[key]
            elif isinstance(key, (list, tuple, range)):
                key_list = [filter_func(x) for x in key]
            else:
                key_list = [filter_func(key)]
            return key_list
        
        if not isinstance(keys, tuple):
            # if keys is a row_name, convert it to a tuple
            assert isinstance(keys, self.__row_name_type), f"key must be a tuple or {self.__row_name_type.__name__}"
            keys = (keys, slice(None, None, None))
        else:
            assert len(keys) == 2, "Keys must be a tuple of length 2"
        row_key, col_key = keys
        if isinstance(row_key, (int, str)) and isinstance(col_key, (int, str)):
            row_key = self._row_name_filter(row_key)
            col_key = self._col_name_filter(col_key)
            return row_key, col_key, False
        elif isinstance(row_key, (list, tuple, range, slice)) or isinstance(col_key, (list, tuple, range, slice)):
            row_key_list = get_key_list(row_key, self.__row_names, self._row_name_filter)
            col_key_list = get_key_list(col_key, self.__col_names, self._col_name_filter)
            return row_key_list, col_key_list, True
        else:
            raise TypeError("invalid key type for Table")

    def __assert_keys_list(self, row_key_list, col_key_list):
        for row_name in row_key_list:
            assert row_name in self.__row_names, f"row_name '{row_name}' does not exist"
        for col_name in col_key_list:
            assert col_name in self.__col_names, f"col_name '{col_name}' does not exist"

    def __getitem__(self, keys:Union[int, str, tuple[Union[int, str]]]):
        row_key, col_key, table_mode = self.__process_item(keys)
        if table_mode:
            self.__assert_keys_list(row_key, col_key)
            table_dict:dict[ROWKEYT, dict[COLKETT, ITEM]] = {}
            for rn in row_key:
                table_dict[rn] = {}
                for cn in col_key:
                    table_dict[rn][cn] = self.__data[rn][cn]
            return table_dict
        else:                
            return self.__data[row_key][col_key]
        
    def __setitem__(self, 
                    keys:tuple[Union[int, str]], 
                    value:Union[ITEM, dict[ROWKEYT, dict[COLKETT, ITEM]], "Table[ROWKEYT, COLKETT, ITEM]"]  = None):
        row_key, col_key, table_mode = self.__process_item(keys)
        if table_mode:
            self.__assert_keys_list(row_key, col_key)
            # check value
            table_dict = {}
            for rn in row_key:
                table_dict[rn] = {}
                for cn in col_key:
                    _v = self.__data[rn][cn]
                    self.assert_default_value_type(_v)
                    table_dict[rn][cn] = self.__data[rn][cn]
            self.update(table_dict)
        else:              
            self.assert_default_value_type(value)
            self.__data[row_key][col_key] = value
        
    def __str__(self) -> str:
        if len(self.__data) == 0:
            return f"Empty Table :" + \
                f"row_name{self.row_names}, col_name{self.col_names} " + \
                f"default_value_type:{self.default_value_type.__name__} " + \
                f"row_name_type:{self.row_name_type.__name__} " + \
                f"col_name_type:{self.col_name_type.__name__} "

        max_col_widths = {col_name: len(col_name) for col_name in self.__col_names}
        str_data = {}

        _row_names = [str(x) for x in self.__row_names]
        _col_names = [str(x) for x in self.__col_names]

        for row_name in _row_names:
            str_data[row_name] = {}
            for col_name in _col_names:
                value = self[row_name, col_name]
                if isinstance(value, Iterable) and not isinstance(value, str):
                    str_value = type(value).__name__ + f"({len(value)})"
                else:
                    str_value = str(value)[:20]
                str_data[row_name][col_name] = str_value
                max_col_widths[col_name] = max(max_col_widths[col_name], len(str_value))

        # Create the table string
        max_col_widths['Row Name'] = max(len(x) for x in _row_names)
        table_str = f"{'':{max_col_widths['Row Name']}} | "
        for col_name in _col_names:
            table_str += f"{col_name.center(max_col_widths[col_name])} | "
        table_str += "\n"
        table_str += "-" * (sum(max_col_widths.values()) + len(max_col_widths) * 3) + "\n"

        for row_name in _row_names:
            table_str += f"{row_name.center(max_col_widths['Row Name'])} | "
            for col_name in _col_names:
                value:str = str_data[row_name][col_name]
                table_str += f"{value.center(max_col_widths[col_name])} | "
            table_str += "\n"

        return table_str

    def keys(self):
        return self.__data.keys()
    
    def values(self):
        return self.__data.values()
    
    def items(self):
        return self.__data.items()

    def __len__(self):
        return len(self.__data)
    
    def __iter__(self):
        return self.__data.__iter__()
    
    def __contains__(self, item):
        return self.__data.__contains__(item)

    @staticmethod
    def get_type(table_dict, default_row_type = str, default_col_type = str):
        try:
            row_type = table_dict.keys().__iter__().__next__().__class__
        except StopIteration:
            row_type = default_row_type
        try:
            col_type = table_dict.values().__iter__().__next__().keys().__iter__().__next__().__class__
        except StopIteration:
            col_type = default_col_type
        return row_type, col_type

    @classmethod
    def from_json(cls, path):
        table_dict:dict[ROWKEYT, dict[COLKETT, ITEM]] = JsonIO.load_json(path)
        # row_type, col_type = Table.get_type(table_dict)
        # table = Table(row_name_type=row_type, col_name_type=col_type)
        # table.update(table_dict)
        table = cls.from_dict(table_dict)
        return table
    
    @classmethod
    def to_json(cls, path, table:"Table"):
        table_dict = table.to_dict()
        JsonIO.dump_json(path, table_dict)

    def to_dict(self):
        dict_ = {}
        dict_[self.KW_EMPTY]                = self.empty
        dict_[self.KW_row_names]            = self.row_names
        dict_[self.KW_col_names]            = self.col_names
        dict_[self.KW_default_value_type]   = self.default_value_type.__name__
        dict_[self.KW_row_name_type]        = self.row_name_type.__name__
        dict_[self.KW_col_name_type]        = self.col_name_type.__name__
        dict_[self.KW_data] = dict(self.data)
        return dict_

    @classmethod
    def from_dict(cls:type["Table"], dict_:dict):
        table = cls(data        = dict_[cls.KW_data],
                    row_names   = dict_[cls.KW_row_names], 
                    col_names   = dict_[cls.KW_col_names], 
                    default_value_type  = eval(dict_[cls.KW_default_value_type]),
                    row_name_type       = eval(dict_[cls.KW_row_name_type]), 
                    col_name_type       = eval(dict_[cls.KW_col_name_type]))
        return table

    def save(self, path):
        JsonIO.dump_json(path, self.to_dict())

def rebind_methods(obj, method_name:Union[str, Callable], new_func:Callable):
    if isinstance(method_name, Callable):
        # search for the method name
        found = False
        for name in dir(obj):
            if getattr(obj, name) == method_name:
                method_name = name
                found = True
                break
        if not found:
            raise ValueError(f"method {method_name} not found")
    else:
        # check if the method exists
        assert hasattr(obj, method_name), f"method {method_name} not found"
    assert isinstance(method_name, str), f"method {method_name} not found"
    setattr(obj, method_name, types.MethodType(new_func, obj))