import numpy as np
import numpy.typing as npt
import cv2
from scipy.spatial.transform import Rotation as _R
import matplotlib.pyplot as plt

from typing import Generic, TypeVar, Any, Union

T = TypeVar('T', 'Posture', npt.NDArray)

class Posture:
    '''
    姿态类，接收各种类型的姿态的输入，并转换为矩阵
    '''
    POSTURE_VEC = 0
    POSTURE_MAT = 1
    POSTURE_HOMOMAT = 2
    POSTURE_EULARZYX = 3
    
    def __init__(self, *, rvec:np.ndarray = None, tvec:np.ndarray = None,
                                        rmat:np.ndarray = None,
                                        homomat:np.ndarray = None,
                                        **Eular
                                        ) -> None:
        self.trans_mat:np.ndarray  = np.eye(4)
        if rvec is not None:
            rvec = np.array(rvec, np.float32).squeeze()
            self.set_rvec(rvec)
        if rmat is not None:
            self.set_rmat(np.reshape(rmat, (3,3)))
        if tvec is not None:
            tvec = np.array(tvec, np.float32).squeeze()
            self.set_tvec(tvec)
        if homomat is not None:
            self.set_homomat(homomat)
        if len(Eular) > 0:
            seq = list(Eular.keys())[0]
            eular = list(Eular.values())[0]
            self.set_rmat(_R.from_euler(seq, eular, degrees=False).as_matrix())
        
    def __mul__(self, obj:T) -> T:
        if isinstance(obj, Posture):
            posture = Posture(homomat = self.trans_mat.dot(obj.trans_mat))
            return posture
        elif isinstance(obj, np.ndarray):
            # the shape of the array must be [N, 3] or [N, 4]
            if (len(obj.shape) == 1 and obj.size == 3 or obj.size == 4):
                pass
            elif (len(obj.shape) == 2 and obj.shape[1] == 3 or obj.shape[1] == 4):
                pass
            else:
                raise ValueError
            rlt:npt.NDArray = (self.rmat.dot(obj[..., :3].T)).T + self.tvec
            return rlt
    
    def __div__(self, obj):
        if isinstance(obj, Posture):
            return self * obj.inv()

    def inv(self):
        inv_transmat = self.inv_transmat
        return Posture(homomat=inv_transmat)

    @property
    def inv_transmat(self) -> np.ndarray :
        return np.linalg.inv(self.trans_mat)
    
    @property
    def rvec(self) -> np.ndarray :
        return cv2.Rodrigues(self.trans_mat[:3,:3])[0][:,0]
    
    @property
    def tvec(self) -> np.ndarray :
        return self.trans_mat[:3,3].T
    
    @property
    def rmat(self) -> np.ndarray :
        return self.trans_mat[:3,:3]
    
    @property
    def eularZYX(self):
        pass
    
    def set_rvec(self, rvec):
        self.trans_mat[:3,:3] = cv2.Rodrigues(rvec)[0]

    def set_tvec(self, tvec):
        self.trans_mat[:3,3] = tvec

    def set_rmat(self, rmat):
        self.trans_mat[:3,:3] = rmat

    def set_homomat(self, homomat):
        self.trans_mat:np.ndarray = homomat.copy()

    def get_eular(self, seq, degrees=False):
        assert seq in ['xyz', 'zyx', 'yxz', 'xzy', 'zxy', 'yzx', 'XYZ', 'ZYX', 'YXZ', 'XZY', 'ZXY', 'YZX']
        r = _R.from_matrix(self.rmat)
        euler_angles = r.as_euler(seq, degrees=degrees)  # 'xyz'表示顺序是绕X、Y、Z轴旋转
        return euler_angles

    def get_NED(self, degrees=False):
        eular = self.get_eular('zyx', degrees=degrees)
        return eular
    
    @classmethod
    def from_vecs(cls, src_vec:np.ndarray, dst_vec:np.ndarray, tvec = None):
        ### 转换为旋转向量
        rvec = Rotation.get_rvec_from_destination(dst_vec, src_vec)
        return cls(rvec=rvec, tvec = tvec)

class Rotation:
    def __init__(self) -> None:
        self.vecs = np.array([])

    @staticmethod
    def get_rvec_from_destination(dest:np.ndarray, base:Union[np.ndarray, list] = [0,0,1]):
        if len(dest.shape) == 1:
            dest = np.expand_dims(dest, 0)
        ### 转换为旋转向量
        base = np.tile(base, [dest.shape[0],1]).astype(np.float32)
        times = np.sum( dest * base, axis=-1)
        angle = np.arccos(times) #旋转角度
        rot = np.cross(base, dest)
        return rot * np.tile(np.expand_dims(angle/ np.linalg.norm(rot, axis=-1), -1), [1,3])

    def get_rvec(self):   
        vecs = self.vecs 
        self.rvec = self.get_rvec_from_destination(vecs)
    
    def plot_destination(self, ax):
        ax = plt.axes(projection='3d')  # 设置三维轴
        ax.scatter(self.vecs[:,0], self.vecs[:,1], self.vecs[:,2], s=5, marker="o", c='r')
        plt.show()

    def __iter__(self):
        return map(lambda x: Posture(rvec=x), self.rvec)

class Icosahedron(Rotation):
    def __init__(self) -> None:
        phi = (np.sqrt(5) - 1)/2
        vertexs = []
        for frac_count in range(3):
            count = (frac_count + 1) % 3
            for i in [-1, 1]:
                for j in [-1, 1]:
                    frac_value = i * (1/phi)
                    value = j * phi
                    vertex = np.zeros(3)
                    vertex[frac_count] = frac_value
                    vertex[count] = value
                    vertexs.append(vertex)
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1,1]:
                    vertexs.append(np.array([i, j, k]))
        
        vecs = np.array(vertexs)
        # normalize
        length = np.linalg.norm(vecs, axis=1)[0]
        vecs = vecs/length
        self.vecs = vecs

        self.get_rvec()

class SphereAngle(Rotation):
    def __init__(self, nums_points = 500) -> None:
        nums_points = nums_points
        radius = 1
        loc = np.zeros((nums_points, 3))
        ii = np.arange(1, nums_points+1, 1)
        phi_array = np.arccos(-1.0 + (2.0 * ii - 1.0) / nums_points)
        theta_array = np.sqrt(nums_points * np.pi) * phi_array
        loc[:,0] = radius * np.cos(theta_array) * np.sin(phi_array)
        loc[:,1] = radius * np.sin(theta_array) * np.sin(phi_array)
        loc[:,2] = radius * np.cos(phi_array)
        self.vecs = loc
        self.get_rvec()