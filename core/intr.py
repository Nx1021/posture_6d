import numpy as np
from .utils import JsonIO
from typing import Union

class CameraIntr():
    INTR_M = "intr_M"
    CAM_FX = "cam_fx"
    CAM_FY = "cam_fy"
    CAM_CX = "cam_cx"
    CAM_CY = "cam_cy"
    CAM_WID = "cam_wid"
    CAM_HGT = "cam_hgt"
    DEPTH_SCALE = "depth_scale"
    EPS = "eps"
    MAX_DEPTH = "max_depth"
    def __init__(self, intr_M, cam_wid = 0, cam_hgt= 0, depth_scale = 0.0, eps = 1.0e-6, max_depth = 4000.0) -> None:
        if isinstance(intr_M, CameraIntr):
            self = intr_M
        elif isinstance(intr_M, np.ndarray):
            self.intr_M = intr_M  
            self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = CameraIntr.parse_intr_matrix(intr_M)
            
            assert cam_wid >= 0 and cam_hgt >= 0, "cam_wid or cam_hgt is illegal"
            self.cam_wid,   self.cam_hgt =  cam_wid, cam_hgt # 重投影到的深度图尺寸

            assert depth_scale >= 0, "depth_scale is illegal"
            self.depth_scale = depth_scale

            assert eps > 0, "eps is illegal"
            self.eps = eps
            
            assert max_depth > 0, "max_depth is illegal"
            self.max_depth = max_depth
        else:
            raise ValueError("intr_M is illegal")

    @staticmethod
    def parse_intr_matrix(intr_M):
        '''
        return CAM_FX, CAM_FY, CAM_CX, CAM_CY
        '''
        cam_fx, cam_fy, cam_cx, cam_cy = intr_M[0,0], intr_M[1,1], intr_M[0,2], intr_M[1,2]
        return cam_fx, cam_fy, cam_cx, cam_cy

    def __mul__(self, points:np.ndarray):
        '''
        points: [N, (x,y,z)]

        return
        ----
        pixels: [N, (x,y)]
        '''
        assert isinstance(points, np.ndarray), "points is not ndarray"
        assert len(points.shape) == 2, "points.shape must be [N, (x,y,z)]"
        assert points.shape[-1] == 3, "points.shape must be [N, (x,y,z)]"
        cam_fx, cam_fy, cam_cx, cam_cy = self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy
        z = points[:, 2]
        u = points[:, 0] * cam_fx / z + cam_cx
        v = points[:, 1] * cam_fy / z + cam_cy
        pixels = np.stack([u,v], axis=-1)
        return pixels
    
    def filter_in_view(self, pixels:np.ndarray):
        '''
        pixels: ndarray [N, (x,y)]
        '''
        if self.cam_wid == 0 or self.cam_hgt == 0:
            raise ValueError("CAM_WID or CAM_HGT is not set")
        valid = np.bitwise_and(np.bitwise_and((pixels[:, 0] >= 0), (pixels[:, 0] < self.cam_wid)),
                    np.bitwise_and((pixels[:, 1] >= 0), (pixels[:, 1] < self.cam_hgt)))
        return pixels[valid]
    
    @staticmethod
    def from_json(path:Union[str, dict]):
        if isinstance(path, str) and path.endswith(".json"):
            dict_ = JsonIO.load_json(path)
        elif isinstance(path, dict):
            dict_ = path
        else:
            raise ValueError("path is illegal")
        if CameraIntr.INTR_M in dict_:
            intr_M = dict_[CameraIntr.INTR_M]
        elif all([x in dict_ for x in [CameraIntr.CAM_FX, CameraIntr.CAM_FY, CameraIntr.CAM_CX, CameraIntr.CAM_CY]]):
            intr_M = np.array([[dict_[CameraIntr.CAM_FX], 0, dict_[CameraIntr.CAM_CX]],
                            [0, dict_[CameraIntr.CAM_FY], dict_[CameraIntr.CAM_CY]],
                            [0, 0, 1]])
        else:
            raise ValueError("intr_M is not in json")
        if CameraIntr.EPS in dict_:
            eps = dict_[CameraIntr.EPS]
        else:
            eps = 1e-6        
        if CameraIntr.MAX_DEPTH in dict_:
            max_depth = dict_[CameraIntr.MAX_DEPTH]
        else:
            max_depth = 4000.0
        return CameraIntr(intr_M, 
                          dict_[CameraIntr.CAM_WID], 
                          dict_[CameraIntr.CAM_HGT], 
                          dict_[CameraIntr.DEPTH_SCALE], 
                          eps, 
                          max_depth)

    def save_as_json(self, path):
        dict_ = {}
        dict_[CameraIntr.INTR_M] = self.intr_M
        dict_[CameraIntr.CAM_WID] = self.cam_wid
        dict_[CameraIntr.CAM_HGT] = self.cam_hgt
        dict_[CameraIntr.DEPTH_SCALE] = self.depth_scale
        dict_[CameraIntr.EPS] = self.eps
        dict_[CameraIntr.MAX_DEPTH] = self.max_depth
        JsonIO.dump_json(path, dict_)