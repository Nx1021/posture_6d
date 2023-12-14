import os
import numpy as np
from .utils.aruco_detector import ArucoDetector
from .capturing import Capturing, RsCamera
from .data_manager import DataRecorder, ModelManager, FrameMeta
from .interact_icp import InteractIcp
from .pcd_creator import PcdCreator
from . import ARUCO_FLOOR, FRAMETYPE_DATA, MeshManager, ViewMeta, Posture, MeshMeta, CameraIntr, cvt_by_intr

from typing import Union

class PipeLine():
    def __init__(self, dataset_name, sub_dir) -> None:
        self.dataset_name = dataset_name
        self.sub_dir = sub_dir
        self.directory = os.path.join(dataset_name, sub_dir)
        #
        self.model_manager = ModelManager(self.directory, "", flag_name="model_manager")
        #
        self.data_recorder = DataRecorder(self.directory, "", flag_name="data_recorder")
        #
        if self.data_recorder.aruco_floor_json.all_files_exist:
            self.aruco_detector = ArucoDetector(self.data_recorder.aruco_floor_json.read(0))
        elif self.data_recorder.aruco_floor_png.all_files_exist:
            image       = self.data_recorder.aruco_floor_png.read(0)
            long_side   = self.data_recorder.aruco_floor_png.read(1)
            self.aruco_detector = ArucoDetector(image, long_side_real_size=long_side)
        else:
            raise ValueError("aruco_floor.json or (aruco_floor.png and aruco_floor_long_side.txt) must exist")
        #
        self.capturing = Capturing(self.data_recorder, self.aruco_detector, self.model_manager)
        self.capturing.data_recorder.add_skip_seg(-1)
        #
        self.pcd_creator = PcdCreator(self.data_recorder, self.aruco_detector, self.model_manager)
        #
        self.interact_icp = InteractIcp(self.data_recorder, self.model_manager)

    @property
    def data_num(self):
        return self.data_recorder.num

    def capture_image(self):
        is_recording_model = True
        record_gate = True

        def callback_CALI(data_recorder:DataRecorder):
            if data_recorder.current_category_index != FRAMETYPE_DATA:
                return False
            else:
                return True

        def callback_DATA(data_recorder:DataRecorder):
            return False

        
        for mode, func in zip([RsCamera.MODE_CALI, RsCamera.MODE_DATA], [callback_CALI, callback_DATA]):
            imu_calibration_fh = self.data_recorder.imu_calibration.query_fileshandle(0)
            self.capturing.rs_camera = RsCamera(mode, imu_calibration=imu_calibration_fh.get_path())
            # self.capturing.rs_camera.intr.save_as_json(os.path.join(self.directory, "intrinsics_" + str(mode) + ".json"))
            self.capturing.start(func)

    def register_pcd(self, update=False):
        self.pcd_creator.register(downsample=False, update=update)

    def segment_pcd(self, update=False):
        self.pcd_creator.auto_seg(update)

    def icp(self):
        self.interact_icp.start()

    def export_data(self, mesh_manager:Union[MeshManager, dict[int, MeshMeta]], cvt_intr:CameraIntr = None):
        if isinstance(mesh_manager, MeshManager):
            mmd = mesh_manager.get_meta_dict()
        elif isinstance(mesh_manager, dict):
            mmd = mesh_manager
        for framemeta in self.data_recorder:
            framemeta:FrameMeta
            color = framemeta.color
            depth = framemeta.depth
            intr_M = framemeta.intr_M["intr_M"]
            if cvt_intr is not None:
                assert isinstance(cvt_intr, CameraIntr), "cvt_intr must be CameraIntr"
                color = cvt_by_intr(color, intr_M, cvt_intr)
                depth = cvt_by_intr(depth, intr_M, cvt_intr)
                intr_M = cvt_intr.intr_M
            ds = framemeta.intr_M["depth_scale"]
            trans_mat_Cn2W = framemeta.trans_mat_Cn2C0
            extr = {}
            for data_i, T_O_2_W in self.model_manager.icp_trans.items():
                Posture_O_2_Cn = Posture(homomat=np.linalg.inv(trans_mat_Cn2W).dot(T_O_2_W))
                extr[data_i] = np.array([Posture_O_2_Cn.rvec, Posture_O_2_Cn.tvec])
            viewmeta = ViewMeta(color, depth, None, extr, intr_M, ds, None, None, None, None)
            viewmeta.calc_by_base(mmd, True)
            yield viewmeta

    def plot_captured(self):
        pass

    def plot_dataset(self):
        pass