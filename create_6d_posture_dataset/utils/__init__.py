import numpy as np

def homo_pad(arr):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy ndarray.")

    if len(arr.shape) < 2:
        raise ValueError("Input array must have at least 2 dimensions.")

    # 获取输入数组的形状
    shape = arr.shape
    # 将最后一维增加一个维度，填充全1
    pad_arr = np.ones((*shape[:-1], 1), dtype=arr.dtype)

    # 使用np.concatenate将填充后的数组与原数组连接
    result = np.concatenate([arr, pad_arr], axis=-1)

    return result