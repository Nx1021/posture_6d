from .derive import PnPSolver
from .data.mesh_manager import MeshManager
from .core.posture import Posture
import numpy as np
import pandas as pd
import cv2
from sklearn.neighbors import NearestNeighbors

from typing import Union, Iterable

class MetricResult():
    def __init__(self, 
                 type:int, 
                 passed:bool,
                 error:float) -> None:
        self.type = type
        self.passed = bool(passed)
        self.error = error
        
    @property
    def type_name(self):
        if self.type == MetricCalculator.REPROJ:
            return "2d reproj"
        if self.type == MetricCalculator.ADD:
            return "ADD(s)"
        if self.type == MetricCalculator._5CM5D:
            return "5cm5°"

    def __str__(self) -> str:
        return "{:<10}  error: {:<6.2}  passed: {}".format(self.type_name, self.error, self.passed)

    def print(self):
        print(self.__str__())

class MetricCalculator():
    ALL = 0
    REPROJ = 1
    ADD = 2
    _5CM5D = 3
    def __init__(self, pnpsolver:PnPSolver, mesh_manager:MeshManager) -> None:
        '''
        class_num: 包含背景在内的类数
        '''
        super().__init__()
        self.pnpsolver:PnPSolver = pnpsolver
        self.mesh_manager:MeshManager = mesh_manager

    @property
    def mesh_manager(self):
        return self.__mesh_manager
    
    @mesh_manager.setter
    def mesh_manager(self, value:MeshManager):
        assert isinstance(value, MeshManager)
        self.__mesh_manager = value
        self.pass_record = np.zeros((4,  self.class_num)) # 4行分别是总数、重投影、ADD、5cm5°
        self.error_record = np.zeros((4, self.class_num)) # 4行分别是总数、重投影error和、ADDerror和、5cm5°error和

    @property
    def class_num(self):
        return self.mesh_manager.class_num

    def clear(self):
        self.pass_record[:] = 0
        self.error_record[:] = 0

    def select_metrics(self, type:int = 0):
        if type == MetricCalculator.REPROJ:
            return self._reprojection_metric
        elif type == MetricCalculator.ADD:
            return self._add_metric
        elif type == MetricCalculator._5CM5D:
            return self._5cm5d_metric
        else:
            raise ValueError(f"unexpected metrics type: {type}")

    def record(self, error_result:list[MetricResult], class_id):
        self.pass_record[0, class_id] += 1
        self.error_record[0, class_id] += 1
        for er in error_result:
            self.pass_record[er.type, class_id] += er.passed
            self.error_record[er.type, class_id]  += er.error

    def calc_one_error(self, class_id:int, 
                       pred_posture:Posture, gt_posture:Posture,
                       selected_metrics:Union[int, Iterable[int]] = ALL) -> tuple[MetricResult]:
        assert isinstance(pred_posture, Posture) and isinstance(gt_posture, Posture), "pred_posture and gt_posture must be Posture object"  
        assert isinstance(selected_metrics , (int, Iterable)), "selected_metrics must be int or Iterable[int]"
        if isinstance(selected_metrics, int):
            if selected_metrics == MetricCalculator.ALL:
                # all metrics
                selected_metrics = [MetricCalculator.REPROJ, MetricCalculator.ADD, MetricCalculator._5CM5D]
            else:
                selected_metrics = [selected_metrics]
        else:
            # only keep the valid metrics
            selected_metrics = [x for x in selected_metrics if (x in [MetricCalculator.REPROJ, MetricCalculator.ADD, MetricCalculator._5CM5D])]

        error_result:list[MetricResult] = []  
        for metrics_type in selected_metrics:
            metrics = self.select_metrics(metrics_type)
            rlt, error = metrics(class_id, pred_posture, gt_posture)
            error_result.append(MetricResult(metrics_type, rlt, error))

        self.record(error_result, class_id)
        return tuple(error_result)

    def print_result(self, rate=False):
        '''
        print the result of all classes
        '''
        col_names= [str(x) for x in range(self.class_num)]
        row_names = ["total number", "reproj", "ADD(s)", "5cm5°"]
        print("accuracy:")
        if rate:
            with np.errstate(divide='ignore', invalid='ignore'):
                data = self.pass_record / self.pass_record[0,:]
        else:
            data = self.pass_record.astype(np.int32)
        df = pd.DataFrame(data, index=row_names, columns=col_names)
        print(df)

        print("error:")
        with np.errstate(divide='ignore', invalid='ignore'):
            data = self.error_record / self.error_record[0,:]
        df = pd.DataFrame(data, index=row_names, columns=col_names)
        print(df)
        print()

    def _reprojection_metric(self, class_id, pred_posture:Posture, gt_posture:Posture) -> tuple[bool, float]:
        '''
        reprojection metric

        parameters
        ----------
        class_id: int, the class id of the object
        pred_posture: Posture, the predicted posture
        gt_posture: Posture, the ground truth posture

        return
        ------
        rlt: bool, whether the error is less than 5
        error: float, the mean error of all points
        '''
        import cv2
        gt_point2D = self.pnpsolver.calc_reproj(self.mesh_manager.get_model_pcd(class_id).astype(np.float32), 
                                                posture= gt_posture)
        # 重投影
        pred_point2D = self.pnpsolver.calc_reproj(self.mesh_manager.get_model_pcd(class_id).astype(np.float32), 
                                                  posture= pred_posture)
        # 重投影误差
        error = np.mean(np.linalg.norm(gt_point2D-pred_point2D, axis=-1), axis=-1)
        if error < 5:
            return 1, error
        else:
            return 0, error

    def _add_metric(self, class_id, pred_posture:Posture, gt_posture:Posture) -> tuple[bool, float]:
        '''
        add metric

        parameters
        ----
        * class_id: int, the class id of the object
        * pred_posture: Posture, the predicted posture
        * gt_posture: Posture, the ground truth posture

        return
        ----
        * rlt: bool, whether the prediction is correct
        * error: float, the error
        '''
        gt_pointcloud_OCS = self.mesh_manager.get_model_pcd(class_id).astype(np.float32) #物体坐标系下的点云

        pred_pointcloud_OCS = pred_posture.inv() * gt_posture * gt_pointcloud_OCS
        # homo_gt_pointcloud_OCS = np.concatenate((gt_pointcloud_OCS.T, np.ones((1, gt_pointcloud_OCS.T.shape[1]))))
        # np.dot(np.linalg.inv(pred_posture), np.dot(gt_posture, homo_gt_pointcloud_OCS)).T[:, :3]
        diameter = self.mesh_manager.get_model_diameter(class_id) #直径

        symmetries = self.mesh_manager.get_model_symmetries(class_id)
        if symmetries is None:
            # ADD
            delta = pred_pointcloud_OCS - gt_pointcloud_OCS #[N, 3]
            distances = np.linalg.norm(delta, axis= -1)
        else:
            # ADD-S
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(gt_pointcloud_OCS)
            distances, indices  = neigh.kneighbors(pred_pointcloud_OCS,return_distance=True) #[N], [N]
        error = np.mean(distances)
        if error < diameter * 0.1:
            return True, error
        else:
            return False, error
            
    def _5cm5d_metric(self, class_id, pred_posture:Posture, gt_posture:Posture) -> tuple[bool, float]:
        """
        5cm5° metric

        Parameters
        ----
        class_id: int, unused
        pred_posture:   Posture
        gt_posture:     Posture
        
        Returns
        ----
        rlt : bool,  if 5cm5d metric is satisfied
        rotation_angle_deg : float, rotation angle error in degree
        """
        # get rotation matrix
        pred_matrix_R = pred_posture.rmat
        gt_matrix_R = gt_posture.rmat

        # angle error
        rotation_angle = np.arccos(np.clip((np.trace(np.dot(pred_matrix_R.T, gt_matrix_R)) - 1) / 2.0, -1, 1))
        rotation_angle_deg = np.degrees(rotation_angle)

        # translation error
        translation_distance = np.linalg.norm(pred_posture.tvec - gt_posture.tvec)

        # if satisfies the 5cm5d criterion
        if translation_distance <= 50 and rotation_angle_deg <= 5.0:
            rlt = True
        else:
            rlt = False

        return rlt, rotation_angle_deg # Only angle is returned as the result is more sensitive to it
