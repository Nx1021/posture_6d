# -- coding: utf-8 --
"""
compute_gt_poses.py
---------------

Main Function for registering (aligning) colored point clouds with ICP/aruco marker 
matching as well as pose graph optimizating, output transforms.npy in each directory

"""
from open3d import *
import numpy as np
import cv2
import os
import glob
# from utils.camera import *
# from registration import icp, feature_registration, match_ransac, rigid_transform_3D
# from open3d import pipelines
# registration = pipelines.registration
from tqdm import trange
import time
import sys
# from config.registrationParameters import *
import json
import png
# from excude_pipeline import *
# from utils.plane import findplane, fitplane, point_to_plane, findplane_wo_outliers
import matplotlib.pyplot as plt
from sko.GA import GA

import cv2.aruco as aruco
from .. import JsonIO, Posture, CameraIntr
from typing import Union

from .camera_sys import convert_depth_frame_to_pointcloud
from .plane import findplane_wo_outliers
from . import homo_pad




class ArucoDetector():
     '''
     detect aruco marker and compute its pose
     '''

     aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
     parameters = aruco.DetectorParameters()
     _inner_aruco_detector = aruco.ArucoDetector(aruco_dict, parameters)

     def __init__(self, aruco_floor:Union[str, dict, np.ndarray], *, long_side_real_size = None) -> None:
          if isinstance(aruco_floor, str):
               if aruco_floor :    
                    suffix = aruco_floor.split('.')[-1]
                    if suffix == 'json':
                         self.C0_aruco_3d_dict = JsonIO.load_json(aruco_floor)
                    elif suffix == "png" or suffix == "bmp":
                         assert long_side_real_size is not None, "long_side_real_size must be specified"                         
                         self.C0_aruco_3d_dict = self.get_C0_aruco_3d_dict_from_image(aruco_floor, long_side_real_size=long_side_real_size)
                         raise ValueError("long_side_real_size must be specified")
               else:
                    pass
          elif isinstance(aruco_floor, dict):
               # check the dict 
               for k, v in aruco_floor.items():
                    assert isinstance(k, int), "the key of aruco_floor must be int"
                    assert isinstance(v, np.ndarray), "the value of aruco_floor must be ndarray"
                    assert v.shape[0] == 4 and v.shape[1] == 3, "the shape of aruco_floor must be [4, 3]"
                    assert np.issubdtype(v.dtype, np.floating)
               self.C0_aruco_3d_dict = aruco_floor
          elif isinstance(aruco, np.ndarray):
               assert long_side_real_size is not None, "long_side_real_size must be specified"
               self.C0_aruco_3d_dict = self.get_C0_aruco_3d_dict_from_image(aruco_floor, long_side_real_size=long_side_real_size)
          self.verify_tol = 0.01
     
     @classmethod
     def detect_aruco_2d(cls, image:np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
          '''
          return
          ----
          corners_src: np.ndarray, [N, 4, 2]
          ids: np.ndarray, [N]
          rejectedImgPoints: np.ndarray, [N, 1, 2]
          '''
          # if the image has 3-channels, convert it to gray scale
          if len(image.shape) == 3:
               gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
          corners_src, ids, rejectedImgPoints = cls._inner_aruco_detector.detectMarkers(gray)
          if len(corners_src) > 0:
               corners_src = np.squeeze(np.array(corners_src), axis=1) # [N, 4, 2]
               ids = np.array(ids).squeeze(axis=-1) # [N]
          else:
               corners_src = np.zeros((0,4,2)) # [0, 4, 2]
               ids = np.array([]) # [0]
          return corners_src, ids, rejectedImgPoints

     @classmethod
     def detect_aruco_3d(cls, color, depth, camera_intrinsics:Union[np.ndarray, CameraIntr], corners_src = None, ids = None):
          '''
          brief
          ----
          detect aruco marker in 2d image and compute its 3d pose by depth image

          params
          ----
          color: np.ndarray, [H, W, 3]
          depth: np.ndarray, [H, W] np.uint16
          camera_intrinsics: np.ndarray, [3, 3] or CameraIntr
          corners_src: Optional, np.ndarray, [N, 4, 2], to avoid repeatly detect aruco marker if detect_aruco_2d has been called in context
          ids: Optional, np.ndarray, [N], to avoid repeatly detect aruco marker if detect_aruco_2d has been called in context
          '''
          camera_intrinsics = CameraIntr(camera_intrinsics)
          if corners_src is None or ids is None:
               corners_src, ids, rejectedImgPoints = ArucoDetector.detect_aruco_2d(color)
          depth = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)
          scene_aruco = np.round(corners_src).astype(np.int32)
          scene_aruco_3d = depth[scene_aruco[:, :, 1], scene_aruco[:, :, 0]]   #[N,4,3]
          return scene_aruco_3d, ids

     @staticmethod
     def get_C0_aruco_3d_dict_from_image(image:Union[str, np.ndarray], long_side_real_size:float):
          '''
          params
          ----
          image: np.ndarray, [H, W, 3]
          long_side_real_size: float, the real size of the long side of the image, unit: mm
          '''
          assert isinstance(image, str) or isinstance(image, np.ndarray), "image must be a path or np.ndarray"
          if isinstance(image, str):
               assert os.path.exists(image), "image not exists"
               af_image = cv2.imread(image) # aruco floor image
          af_image:np.ndarray = np.pad(af_image,[(100,100),(100,100)],'constant',constant_values=255)
          # 粗测
          zoom_1 = 600 / np.max(af_image.shape)
          coarse_arcuo_floor = cv2.resize(af_image, (-1,-1), fx=zoom_1, fy=zoom_1)
          corners_src, ids, rejectedImgPoints = ArucoDetector.detect_aruco_2d(coarse_arcuo_floor)
          
          # 精测
          refine_corners_src_list = []
          for cs in corners_src:
               refine_corner_list = []
               for corner in cs: #[2]
                    crop_rect_min = np.clip(corner - 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
                    crop_rect_max = np.clip(corner + 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
                    crop_rect_min = np.round(crop_rect_min / zoom_1).astype(np.int32)
                    crop_rect_max = np.round(crop_rect_max / zoom_1).astype(np.int32)
                    local = af_image[crop_rect_min[1]: crop_rect_max[1], crop_rect_min[0]: crop_rect_max[0]]
                    refine_corner = np.squeeze(cv2.goodFeaturesToTrack(local, 1, 0.05, 0)) + crop_rect_min
                    refine_corner_list.append(refine_corner)
               refine_corners_src_list.append(refine_corner_list)
          refine_corners_src = np.array(refine_corners_src_list) # [N, 4, 2]

          refine_corners_src = np.reshape(refine_corners_src, (-1, 2)) # [N*4, 2]
          ### map refine_corners_src to 0~long_side_real_size
          refine_corners_src = refine_corners_src - np.min(refine_corners_src, axis=0)
          refine_corners_src = refine_corners_src / np.max(af_image.shape) * long_side_real_size
          # stack z axis, the value of z is 0
          refine_corners_src = np.hstack((refine_corners_src, np.zeros((refine_corners_src.shape[0], 1)))) # [N*4, 3]
          refine_corners_src = np.reshape(refine_corners_src, (-1, 4, 3)) # [N, 4, (y, x, z)]
          # swap axis
          refine_corners_src_3d = refine_corners_src[:,:,(1,0,2)] #[N, 4, (x, y, z)]
          # zip dict
          C0_aruco_3d_dict = dict(zip(ids.flatten().tolist(), refine_corners_src_3d))

          return C0_aruco_3d_dict
          # min_pos = np.min(refine_corners_src, axis=0) #最小范围
          # refine_corners_src = refine_corners_src - min_pos
          # max_pos = np.max(refine_corners_src, axis=0) #最大范围
          # refine_corners_src = refine_corners_src * long_side_real_size / np.max(max_pos)
          # # refine_corners_src = np.swapaxes(refine_corners_src, 0, 1)
          # refine_corners_src = np.hstack((refine_corners_src, np.zeros((refine_corners_src.shape[0], 1)))) # [N*4, 3]
          # refine_corners_src = np.reshape(refine_corners_src, (-1, 4, 3)) # [N, 4, 3]
          # refine_corners_src = refine_corners_src[:,:,(1,0,2)]
          # predef_arcuo_SCS = {}
          # for id, pos in zip(ids, refine_corners_src):
          #      predef_arcuo_SCS.update({str(int(id)): pos.tolist()})
          # dump_int_ndarray_as_json(predef_arcuo_SCS, os.path.join(self.directory, ARUCO_FLOOR + ".json"))

     @staticmethod
     def project(camera_intrinsics:Union[np.ndarray, CameraIntr], points_C):
          '''
          points_C: [N, 3]
          '''
          camera_intrinsics = CameraIntr(camera_intrinsics)
          points_I = camera_intrinsics * points_C
          return points_I #[N, 2]

     @staticmethod
     def restore(camera_intrinsics:Union[np.ndarray, CameraIntr], points_I):
          '''
          points_I: [N, 2]
          '''
          camera_intrinsics = CameraIntr(camera_intrinsics)
          points_I = homo_pad(points_I)
          K = camera_intrinsics.intr_M
          points_C = np.linalg.inv(K).dot(points_I.T) #[3, N]
          return points_C.T #[N, 3]          

     def collect_if_id_in_C0_aruco(self, ids:np.ndarray, *values):
          assert isinstance(ids, np.ndarray)
          assert np.issubdtype(ids.dtype, np.integer)
          assert all([len(ids) == len(v) for v in values]), "all the values must has the same size as ids"
          collectors = [[] for _ in range(len(values))]
          C0_aruco_3d = []
          for i, _id in enumerate(ids):
               if _id in self.C0_aruco_3d_dict:
                    C0_aruco_3d.append(self.C0_aruco_3d_dict[_id])
                    for c, v in zip(collectors, values):
                         c.append(v[i])
          return C0_aruco_3d, *collectors

     def get_T_3d(self, color, depth, camera_intrinsics, tol = 0.01, return_coords = True):
          scene_aruco_3d, ids, _ = ArucoDetector.detect_aruco_3d(color, depth, camera_intrinsics)
          C0_aruco_3d, common_scene_aruco_3d = self.collect_if_id_in_C0_aruco(ids, scene_aruco_3d)
          if len(common_scene_aruco_3d) == 0:
               transform = np.eye(4)
          else:
               C0_aruco_3d = np.array(C0_aruco_3d).reshape((-1, 4, 3))   # [N, 4, 3]
               scene_aruco_3d = np.array(common_scene_aruco_3d).reshape((-1, 4, 3))  # [N, 4, 3]
               ### 只选取近处点
               scene_aruco_3d_centers = np.mean(scene_aruco_3d, axis=1) #[N, 3]
               distances = np.linalg.norm(scene_aruco_3d_centers, axis = -1)
               argsort_idx = np.argsort(distances)[:3] #最多3个
               scene_aruco_3d = scene_aruco_3d[argsort_idx]
               C0_aruco_3d = C0_aruco_3d[argsort_idx]
               ### 
               scene_aruco_3d = np.reshape(scene_aruco_3d, (-1, 3))
               C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 3))
               sol = findplane_wo_outliers(scene_aruco_3d)
               if sol is not None:
                    sol = sol/np.linalg.norm(sol[:3])
                    distance = np.abs(np.sum(scene_aruco_3d * sol[:3], axis=-1) + sol[3])
                    if np.sum(distance < 0.005) < 4:
                         _scene_aruco_3d = np.reshape(scene_aruco_3d, (-1, 4, 3))
                         colors = plt.get_cmap('jet')(np.linspace(0, 1, scene_aruco_3d.shape[0]))
                         plt.figure(0)
                         ax = plt.axes(projection = '3d')
                         for C0,c  in zip(_scene_aruco_3d, colors):
                              ax.scatter(C0[:,0], C0[:,1], C0[:,2], s = 10, marker = 'o', color = c[:3])
                         plt.show()
                         sol = None
                    else:
                         # 对投影缩放，使得所有点在同一平面
                         distance = np.sum(scene_aruco_3d * sol[:3], axis=-1)
                         affine_ratio = -distance / sol[3]
                         ingroup_scene_aruco_3d = (scene_aruco_3d.T / affine_ratio).T
                         ingroup_C0_aruco_3d = C0_aruco_3d
               if sol is None:
                    ingroup_scene_aruco_3d = scene_aruco_3d
                    ingroup_C0_aruco_3d = C0_aruco_3d
               transform = match_ransac(ingroup_scene_aruco_3d, ingroup_C0_aruco_3d, tol = tol)
          if transform is None:
               transform = np.eye(4)
          else:
               transform = np.asarray(transform)
          if return_coords:
               re_trans = transform.dot(np.hstack((ingroup_scene_aruco_3d, np.ones((ingroup_scene_aruco_3d.shape[0], 1)))).T).T[:, :3]
               return transform, (ingroup_scene_aruco_3d, C0_aruco_3d, re_trans, ids)
          else:
               return transform

     def get_T_2d(self, color:np.ndarray, depth:np.ndarray, camera_intrinsics:CameraIntr, return_coords = True):
          assert self.C0_aruco_3d_dict is not None
          scene_aruco_2d, ids, _ = ArucoDetector.detect_aruco_2d(color) # [N, 4, 2], [N]
          (C0_aruco_3d, 
           common_scene_aruco_2d) = self.collect_if_id_in_C0_aruco(ids, scene_aruco_2d)
          if len(common_scene_aruco_2d) < 1:
               transform = np.eye(4)
          else:
               C0_aruco_3d = np.array(C0_aruco_3d).reshape((-1, 4, 3))   #[N, 3]
               common_scene_aruco_2d = np.array(common_scene_aruco_2d).reshape((-1, 4, 2)) #[N,2]
               ### 只选取近处点
               # cube_idx = np.where(np.array(common_scene_aruco_ids)<20)[0]
               # scene_aruco_3d_centers = np.mean(common_scene_aruco_3d, axis=1) #[N, 3]
               # distances = np.linalg.norm(scene_aruco_3d_centers, axis = -1)
               # distances[cube_idx] = 1e10
               # floor_idx = np.argsort(distances)[:3] #最多3个
               # argsort_idx = np.concatenate((cube_idx, floor_idx))
               # common_scene_aruco_2d = common_scene_aruco_2d[argsort_idx]
               # C0_aruco_3d = C0_aruco_3d[argsort_idx]
               ### 
               C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 3))
               common_scene_aruco_2d = np.reshape(common_scene_aruco_2d, (-1, 2))
               fx = camera_intrinsics.cam_fx
               fy = camera_intrinsics.cam_fy
               ppx = camera_intrinsics.cam_cx
               ppy = camera_intrinsics.cam_cy
               cameraMatrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0,0,1]])
               distCoeffs = np.array([0.0,0,0,0,0])
               _, rvec, tvec = cv2.solvePnP(C0_aruco_3d, common_scene_aruco_2d, cameraMatrix, distCoeffs)
               posture = Posture(rvec = rvec, tvec = np.squeeze(tvec))
               transform = posture.inv_transmat
          if return_coords:
               re_trans = cv2.projectPoints(C0_aruco_3d, rvec, tvec, cameraMatrix, distCoeffs)[0]
               re_trans = np.squeeze(re_trans, axis=1)
               return transform, (scene_aruco_2d, C0_aruco_3d, re_trans, ids)
          else:
               return transform

     def verify_frame(self, cad, depth, camera_intrinsics, if_2d = True):
          if not if_2d:
               transform, (scene_aruco_3d, C0_aruco_3d, re_trans, ids) = self.get_T_3d(
                    cad, depth, camera_intrinsics, tol = self.verify_tol, return_coords=True)
               errors = np.linalg.norm(np.abs(re_trans - C0_aruco_3d), axis = -1)
               errors = errors[np.argsort(errors)][:int(len(errors)*0.8)]
               if len(ids) >= 8: 
                    if np.any(errors > 0.005) and np.mean(errors) > 0.003:
                         return False, None             
               else:
                    if np.any(errors > 0.004) and np.mean(errors) > 0.002:
                         return False, None  
               return True, transform
          else:
               # cannot be verified by 2d
               transform = self.get_T_2d(
                    cad, depth, camera_intrinsics, return_coords = False)
               return True, transform
          

