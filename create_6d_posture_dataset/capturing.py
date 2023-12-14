
"""
record2.py
---------------

Main Function for recording a video sequence into cad (color-aligned-to-depth)
images and depth images

Using librealsense SDK 2.0 with pyrealsense2 for SR300 and D series cameras


"""

# record for 40s after a 5s count down
# or exit the recording earlier by pressing q

RECORD_LENGTH = 80
WAIT_LENGTH = 5

BUFFER_MAX_LENGTH = 9
DELTA_THRESHOLD = 15
LPLC_THRESHOLD = 0.0012
NEW_FRAME_MUL = 2.0
FIRST_FRAME_ARUCO_NUM = 12
BASE_FRAME_ARUCO_NUM = 8
SINGLE_BASE_FRAME_ARUCO_NUM = 3

STOP_COLOR_THRESHOLD = 25
STOP_BLACK_TIME = 3

TEST = False

from . import RGB_DIR, DEPTH_DIR, CALI_INTR_FILE, DATA_INTR_FILE, FRAMETYPE_DATA
from . import JsonIO
import pyrealsense2 as rs
import json
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import os
import shutil
import sys
import matplotlib.pyplot as plt
from typing import Union


from . import CameraIntr
from .utils.aruco_detector import ArucoDetector
from .data_manager import DataRecorder, ModelManager
# from config.DataAcquisitionParameters import DEPTH_THRESH

class Motion():
    def __init__(self, sensor, imu_calibration:Union[str, dict] = None) -> None:
        # self.sync_imu_by_this_stream = rs.stream.any
        active_imu_profiles = []

        active_profiles = dict()
        self.imu_sensor = None
        for pr in sensor.get_stream_profiles():
            if pr.stream_type() == rs.stream.gyro and pr.format() == rs.format.motion_xyz32f:
                active_profiles[pr.stream_type()] = pr
                self.imu_sensor = sensor
            if pr.stream_type() == rs.stream.accel and pr.format() == rs.format.motion_xyz32f:
                active_profiles[pr.stream_type()] = pr
                self.imu_sensor = sensor
        if not self.imu_sensor:
            print('No IMU sensor found.')
            return False
        print('\n'.join(['FOUND %s with fps=%s' % (str(ap[0]).split('.')[1].upper(), ap[1].fps()) for ap in active_profiles.items()]))
        active_imu_profiles = list(active_profiles.values())
        if len(active_imu_profiles) < 2:
            print('Not all IMU streams found.')
            return False
        try:
            self.imu_sensor.stop()
            self.imu_sensor.close()
        except:
            pass
        self.imu_sensor.open(active_imu_profiles)
        self.imu_start_loop_time = time.time()

        # Make the device use the original IMU values and not already calibrated:
        if self.imu_sensor.supports(rs.option.enable_motion_correction):
            self.imu_sensor.set_option(rs.option.enable_motion_correction, 0)
        self.read_calibrate(imu_calibration)

        self.last_a_time = 0
        self.last_g_time = 0
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        # self.init_angular_position = np.zeros(3)
        self.angular_position = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        self.g = 9.798

        self.stable_time_threshold = 500 # ms
        self.stable_gyro_threshold = 0.1
        self.last_moving_time = 0
        self.this_time = 0

        self.imu_sensor.start(self.imu_callback)
        # f = self.imu_sensor.wait_for_frames()
        # print(f)

    def read_calibrate(self, path):
        '''
        读取校准矩阵
        '''
        if path is None:
            self.accel_X = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1],
                                     [0, 0, 0]], np.float32)
            self.gyro_X = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1],
                                     [0, 0, 0]], np.float32)
        else:
            if isinstance(path, dict):
                cali_dict = path["imus"][0]
            elif isinstance(path, str):
                cali_dict = JsonIO.load_json(path)["imus"][0]
            else:
                raise TypeError("path must be str or dict")
            accel_X = np.zeros(12) #加速度的补偿矩阵
            accel_X[:9]     = np.array(cali_dict["accelerometer"]["scale_and_alignment"])
            accel_X[9:12]   = np.array(cali_dict["accelerometer"]["bias"])
            self.accel_X    = np.reshape(accel_X, (4,3))
            gyro_X          = np.zeros(12) #角加速度的补偿矩阵
            gyro_X[:9]      = np.array(cali_dict["gyroscope"]["scale_and_alignment"])
            gyro_X[9:12]    = np.array(cali_dict["gyroscope"]["bias"])
            self.gyro_X     = np.reshape(gyro_X, (4,3))

    def imu_callback(self, frame):
        '''
        回调函数
        '''
        self.this_time = frame.timestamp
        if frame.profile.stream_type() == rs.stream.gyro:
            ## 读取并修正
            data = frame.as_motion_frame().get_motion_data()
            data_array = np.array([data.x, data.y, data.z, -1])
            fixed_gyro_array = data_array.dot(self.gyro_X)
            gyro_norm = np.linalg.norm(fixed_gyro_array)
            # 打印
            fixed_gyro_array_strings = ["{:>2.5f}".format(x).rjust(10) for x in fixed_gyro_array]
            gyro_string = ", ".join(fixed_gyro_array_strings)
            print("\rgyro: {}, normal:{:>2.5f}".format(gyro_string, gyro_norm).rjust(40), end="")
            if np.any(np.abs(fixed_gyro_array) > self.stable_gyro_threshold):
                self.last_moving_time = self.this_time # 每次运动之后至少需要self.stable_time_threshold的时间才能恢复稳定
            print("  稳定状态：", self.is_stable, end='')
        return

    @property
    def is_stable(self):
        '''
        是否是稳定状态
        '''
        if self.this_time - self.last_moving_time > self.stable_time_threshold:
            return True
        else:
            return False

    def cal_position(self, frame):
        this_time = frame.timestamp

        if frame.profile.stream_type() == rs.stream.accel:
            ## 读取并修正
            data = frame.as_motion_frame().get_motion_data()
            data_array = np.array([data.x, data.y, data.z, -1])
            fixed_accel_array = data_array.dot(self.accel_X)
            # fixed_accel_array = data_array[:3]
            accel_norm = np.linalg.norm(fixed_accel_array)
            # 计算时间差

            ### 计算位置和姿态
            if self.last_a_time == 0:
                # 第一个位置，必须是静止的
                self.angular_position = fixed_accel_array / accel_norm
                self.last_a_time = this_time
                return
            delta_time = (this_time - self.last_a_time) / 1000
            # 排除重力
            # fixed_accel_array = fixed_accel_array - self.g * self.angular_position
            # 积分
            self.velocity = self.velocity + delta_time * fixed_accel_array
            self.position = self.position + delta_time * self.velocity
            # 打印
            fixed_accel_array_strings = ["{:>2.5f}".format(x).ljust(10) for x in fixed_accel_array]
            accel_string = ", ".join(fixed_accel_array_strings)
            position_strings = ["{:>2.5f}".format(x).ljust(10) for x in self.position]
            position_string = ", ".join(position_strings)
            print("\raccel: {}, normal:{:>2.5f}, position: {}".format(accel_string, accel_norm, position_string).ljust(40), end="")
            # 修改时间
            self.last_a_time = this_time
        if frame.profile.stream_type() == rs.stream.gyro:
            if self.last_a_time == 0 or self.last_g_time == 0:
                self.last_g_time = this_time
                return
            ## 读取并修正
            data = frame.as_motion_frame().get_motion_data()
            data_array = np.array([data.x, data.y, data.z, -1])
            fixed_gyro_array = data_array.dot(self.gyro_X)
            # 计算时间差
            delta_time = (this_time - self.last_g_time) / 1000
            ### 计算位置和姿态
            # 积分
            self.angular_position = self.angular_position + delta_time * fixed_gyro_array
            self.angular_position = self.angular_position + delta_time * self.angular_position
            # 修改时间
            self.last_g_time = this_time
        return

class RsCamera():
    MODE_CALI = 0
    MODE_DATA = 1
    def __init__(self, mode, imu_calibration = None) -> None:
        # 创建一个pipeline实例
        self.pipeline = rs.pipeline()
        # 创建配置实例
        self.config = rs.config()

        # 获取相机设备，并获取深度传感器对象
        self.pipeline_profile = self.config.resolve(self.pipeline)
        self.device = self.pipeline_profile.get_device()
        sensors = self.device.query_sensors()
        self.depth_sensor = rs.depth_sensor(sensors[0])
        self.color_sensor = rs.color_sensor(sensors[1])
        self.motion_module = Motion(sensors[2], imu_calibration)
        self.depth_sensor.set_option(rs.option.visual_preset, 3)
        # 将深度单位设置为毫米
        self.depth_sensor.set_option(rs.option.depth_units, 0.0005)

        # 配置深度流和彩色流
        self.mode = mode
        if mode == RsCamera.MODE_DATA:
            intr_name = DATA_INTR_FILE
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        elif mode == RsCamera.MODE_CALI:
            intr_name = CALI_INTR_FILE
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # 启动pipeline，并获取彩色帧和深度帧
        self.profile = self.pipeline.start(self.config)
        frame = self.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        self.start_time = 0
        self.call_num = 0

        # 获取相机内参
        rs_intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_parameters = {CameraIntr.CAM_FX: rs_intr.fx, CameraIntr.CAM_FY: rs_intr.fy,
                            CameraIntr.CAM_CX: rs_intr.ppx, CameraIntr.CAM_CY: rs_intr.ppy,
                            CameraIntr.CAM_HGT: rs_intr.height, CameraIntr.CAM_WID: rs_intr.width,
                            CameraIntr.DEPTH_SCALE: self.profile.get_device().first_depth_sensor().get_depth_scale() * 1000 # m 2 mm
                            }
        self.intr = CameraIntr.from_json(camera_parameters)

        # 设置亮度、对比度和曝光度等参数
        sensors[1].set_option(rs.option.brightness,0)
        sensors[1].set_option(rs.option.contrast, 50)
        sensors[1].set_option(rs.option.exposure, 312)
        sensors[1].set_option(rs.option.saturation, 70)

        # 定义深度图滤波器
        self.spatial = rs.spatial_filter() # 空间滤波器
        self.spatial.set_option(rs.option.filter_magnitude,2)
        self.spatial.set_option(rs.option.filter_smooth_alpha,0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta,20)

        self.hole_filling = rs.hole_filling_filter() # 空洞滤波器
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        self.temporal = rs.temporal_filter() # 时间滤波器
        self.temporal.set_option(rs.option.filter_smooth_alpha,0.3)
        self.temporal.set_option(rs.option.filter_smooth_delta,18)
        if self.mode == self.MODE_CALI:
            self.temporal.set_option(rs.option.holes_fill, 7)

        self.decimation = rs.decimation_filter() # 抽取滤波器
        self.decimation.set_option(rs.option.filter_magnitude,2.0)

        self.fwd_disparity = rs.disparity_transform(True) # 视差变换函数
        self.inv_disparity = rs.disparity_transform(False)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.color_frame = None
        self.depth_frame = None

    # def callback(self, frame):
    #     frame_list = []
    #     frame.foreach(lambda x: frame_list.append(x))
    #     # composite_frame_get = False
    #     # if frame.profile.stream_type() == rs.stream.color:
    #     #     print(frame)
    #     #     self.color_frame = frame
    #     #     if self.depth_frame is not None:
    #     #         composite_frame = rs.composite_frame([self.color_frame, self.depth_frame])
    #     #         composite_frame_get = True
    #     #         self.color_frame = None
    #     #         self.depth_frame = None
    #     if frame.profile.stream_type() == rs.stream.depth:
    #         print(frame)
    #         self.depth_frame = frame
    #         if self.color_frame is not None:
    #             composite_frame = rs.composite_frame([self.color_frame, self.depth_frame])
    #             composite_frame_get = True
    #             self.color_frame = None
    #             self.depth_frame = None
    #     if frame.profile.stream_type() == rs.stream.gyro:
    #         # print(frame)
    #         pass
    #     if composite_frame_get:
    #         print(composite_frame)
    #     # print(frame.profile.stream_type())

    def get_frames(self):
        frame = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frame)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # print(depth_frame, color_frame)
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            return None, None
        depth_frame = self.fwd_disparity.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        # depth_frame = self.hole_filling.process(depth_frame)
        # depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.inv_disparity.process(depth_frame)

        return color_frame, depth_frame
    
        

def get_corner_dict(img: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray, bool]:
    corners, ids, _ = ArucoDetector.detect_aruco_2d(img)
    corner_dict = {id: c for id, c in zip(ids, corners)}
    return corner_dict, ids, len(ids) > 0

def multiframe_distortion_correction(color_list: list[np.ndarray], image_size) -> np.ndarray:
    '''
    correct distortion of multiframe
    color_list: list of color image
    image_size: size of image
    '''
    if len(color_list) > 1:
        all_ids = []
        all_corners:list[dict[int, np.ndarray]] = []
        for num, img in enumerate(color_list):
            if num > 20:
                continue
            corner_dict, ids, ok = get_corner_dict(img)
            if ok:
                all_corners.append(corner_dict)
                all_ids.append(ids)
        ### 共有id和对应的角点的坐标
        common_ids = np.intersect1d(all_ids[0], all_ids[1])
        for i in range(2, len(all_ids)):
            common_ids = np.intersect1d(common_ids, all_ids[i])
        common_corners = []
        for cor_dict in all_corners:
            corners = [cor_dict[id_] for id_ in common_ids]
            common_corners.append(corners)

        ### 计算变换矩阵
        # 以最后一帧为参考
        ref_corners = common_corners[-1]
        ref_corners = np.reshape(ref_corners, (-1, 2))
        ref_ids = all_ids[-1]
        transform_matrices = []
        for i in range(len(common_corners) - 1):
            curr_corners = common_corners[i]
            curr_corners = np.reshape(curr_corners, (-1, 2))
            transform_matrix, _ = cv2.findHomography(curr_corners, ref_corners)
            transform_matrices.append(transform_matrix)

        ### 对图像进行变换
        new_color_list = []
        for color, transform_matrix in zip(color_list, transform_matrices):
            height, width = color.shape[:2]
            new_height, new_width = image_size
            new_color = cv2.warpPerspective(color, transform_matrix, (new_width, new_height))
            new_color_list.append(new_color)
        # 计算中位数图像
        new_color_list = np.array(new_color_list)
        median_color = np.median(new_color_list, axis=0).astype(np.uint8)
    else:
        median_color = color_list[0]
    return median_color

class Capturing():
    def __init__(self, data_recorder:DataRecorder, aruco_detector:ArucoDetector, model_manager:ModelManager, rs_camera:RsCamera = None) -> None:
        self.data_recorder = data_recorder
        self.aruco_detector = aruco_detector
        self.model_manager = model_manager
        self.trans_mat_list = []
        self.record_pos_list = []

        self.color_image_list    = [] # 色彩图缓存队列
        self.depth_frame_list    = [] # 深度图缓存队列
        self.aruco_dict_list     = [] # 交点队列
        self.this_color:np.ndarray = None
        self.this_depth_frame:np.ndarray = None
        self.this_floor_aruco_dict:dict[int, np.ndarray] = None
        self.this_other_aruco_dict:dict[int, np.ndarray] = None

        self.keep_at_last_position = True #是否已经离开上一位置，用于防止在同一位置连续采集
        self.record_pos_list_updated = True

        self.info_txt:str = "" #显示在可视化界面的文字信息
        self.record_pos_image = None
        # 物品例像字典
        self.obj_exanple_img_dict = {}
        for name, img in zip(self.model_manager.std_meshes_names, self.model_manager.std_meshes_image):
            if img is None:
                img = np.zeros((480, 640, 3), np.uint8)
            self.obj_exanple_img_dict.update({name: img})

        self.ignore_stable = False

        self.__rs_camera = rs_camera

    @property
    def rs_camera(self):
        return self.__rs_camera
    
    @rs_camera.setter
    def rs_camera(self, rs_camera:RsCamera):
        assert isinstance(rs_camera, RsCamera), "para:rs_camera is not a RsCamera"
        self.__rs_camera = rs_camera

        intr_json_dir = self.data_recorder.intr_file.query_fileshandle(rs_camera.mode).get_path()
        # if rs_camera.mode == 0:
        #     intr_json_dir = self.data_recorder.intr_file.query_fileshandle(rs_camera.mode).get_dir()
        # elif rs_camera.mode == 1:
        #     intr_json_dir = self.data_recorder.intr_file.query_fileshandle(0).get_dir()
        # else:
        #     raise Exception("rs_camera.mode is illegal")
        self.__rs_camera.intr.save_as_json(intr_json_dir)

    def read_trans_mats(self):
        # 清空变换矩阵列表和记录位置列表
        self.trans_mat_list.clear()
        self.record_pos_list.clear()
        # 遍历模型索引范围中的每个模型
        for trans_mat in self.data_recorder.trans_elements.in_current_category():
            # 将变换矩阵添加到变换矩阵列表中
            self.trans_mat_list.append(trans_mat)
            # 将变换矩阵转换为位置点并添加到记录位置列表中
            record_pos = self.__trans_mat_2_pos_point(trans_mat)
            self.pushback_to_record_pos_list(record_pos)
        self.record_pos_list_updated = True

    def pushback_to_record_pos_list(self, record_pos):
        self.record_pos_list_updated = True
        self.record_pos_list.append(record_pos)

    def update_record_pos_image(self, H, W):
        self.record_pos_list_updated = False
        img = np.zeros((int(H), int(W), 3))
        info_mat = np.array([   [220,  0, img.shape[1]/2],
                                [0,  220, img.shape[0]/2],
                                [0,    0,                   1]])
        # 绘制外圈圆
        img = cv2.circle(img, info_mat.dot(np.array([0,0,1]))[:2].astype(np.int32), 225, (255, 255, 255), 2)
        # 绘制外圈圆
        img = cv2.circle(img, info_mat.dot(np.array([0,0,1]))[:2].astype(np.int32), int(225 *0.866) , (255, 255, 255), 1)
        # 绘制圆心
        img = cv2.circle(img, info_mat.dot(np.array([0,0,1]))[:2].astype(np.int32), 4, (255, 255, 255), -1)
        # 循环遍历圆心列表，并绘制圆
        # 反序
        N = 5
        for i, center in enumerate(self.record_pos_list[::-1]):
            center = np.append(center[:2], 1)
            center = info_mat.dot(center)[:2].astype(np.int32)
            ck = min(N, i)
            color = (30 * ck, 30 * ck, 255 - 20 * ck)
            # if i == len(self.record_pos_list) - 1: #最后一位
            #     color = (0, 0, 255)
            # else:
            #     color = (255, 255, 255)
            # 绘制圆
            img = cv2.circle(img, center, 4, color, -1)
        self.record_pos_image = img

    def start(self, break_callback, record_gate = True):
        assert self.rs_camera is not None, "RsCamera is None"
        self.data_recorder.open()
        self.record_gate = record_gate

        self.T_start = time.time() # 开始时间
        last_recorded_color_image = None # 上一次被记录的色彩图

        pushto_buffer_permitted = True        # 标志位：准许进入缓存队列
        has_made_directories = False        # 标志位：已经生成文件夹
        is_recording_model = True
        skip = False
        start_stand_by_time = 0
        self.read_trans_mats()
        while True:
            info_txt = ""
            color_frame, depth_frame = self.rs_camera.get_frames()
            if color_frame is None:
                continue
            c = np.asanyarray(color_frame.get_data())
            c_int16 = c.astype(np.int16)
            pushto_buffer_permitted = self.process_one_frame(c, depth_frame)
            if not self.is_waiting:
                ### 根据输入动作切换采集的模型                
                if break_callback(self.data_recorder) or self.data_recorder.is_all_recorded or skip:
                    break #结束采集
                if np.mean(c_int16) < STOP_COLOR_THRESHOLD:
                    if time.time() - start_stand_by_time > STOP_BLACK_TIME and is_recording_model == False:
                        skip = True
                    if is_recording_model == True:
                        start_stand_by_time = time.time()
                        skip = False
                        is_recording_model = False
                else:
                    if is_recording_model == False:
                        is_recording_model = True
                        self.data_recorder.inc_idx()
                        if self.data_recorder.current_category_index in self.data_recorder.skip_segs:
                            skip = True
                        self.read_trans_mats()
                if pushto_buffer_permitted:
                    self.process_buffer()
            if not is_recording_model:
                self.info_txt = "stand by" + "." * int(time.time() - start_stand_by_time)
            if not self.is_waiting and not has_made_directories:
                # 倒计时结束后清空文件夹
                # self.data_recorder.make_directories()
                has_made_directories = True                
            self.visualise()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.data_recorder.skip_to_seg()
        # self.data_recorder.rename_all()
        # self.data_recorder.save_record()
        cv2.destroyAllWindows()  # 关闭窗口
        self.rs_camera.pipeline.stop()

        self.data_recorder.close()

    @property
    def is_waiting(self):
        if TEST:
            return False
        return time.time() - self.T_start < WAIT_LENGTH

    @property
    def this_floor_aruco_corners(self):
        return list(self.this_floor_aruco_dict.values())

    @property
    def this_other_aruco_corners(self):
        return list(self.this_other_aruco_dict.values())
        
    def visualise(self):
        if self.data_recorder.is_all_recorded:
            return
        ### 可视化
        c = self.this_color.copy()
        # 深度图色彩化
        d = np.asanyarray(self.this_depth_frame.get_data())  # 获取深度图像数据
        max_dn = self.rs_camera.intr.max_depth / self.rs_camera.intr.depth_scale
        d[d > max_dn] = max_dn
        norm_depth_image = d.astype(np.float32)  # 转换数据类型为浮点型
        # 对深度图像进行限制，超过一定深度范围的值设为最大深度值
        # norm_depth_image[norm_depth_image > int(2 / self.rs_camera.intr.depth_scale)] = int(2 / self.rs_camera.intr.depth_scale)
        norm_depth_image = (norm_depth_image)/(norm_depth_image.max()+1)  # 归一化深度图像
        norm_depth_image = (norm_depth_image*255).astype(np.uint8)  # 缩放深度图像的像素值到 0~255
        depth_colormap = cv2.applyColorMap(norm_depth_image, cv2.COLORMAP_JET)  # 将深度图像转换为伪彩色图像

        ### 图片拼接
        shape = np.array((c.shape[1], c.shape[0]))
        if self.rs_camera.mode == RsCamera.MODE_CALI:
            c = cv2.rectangle(c, (shape*0.25).astype(np.int32),
                                (shape*0.75).astype(np.int32), (255,0,0), 4)
        camera_field = np.hstack((c, depth_colormap))  # 将彩色图像和深度图像水平拼接在一起
        resize_ratio = 480 / camera_field.shape[0]  # 计算缩放比例
        camera_field = cv2.resize(camera_field, (-1,-1), fx = resize_ratio, fy = resize_ratio)  # 按比例缩放图像
        
        # obj_exanple_img 在每个物品采集过程的第一帧，显示图像进行提示
        obj_exanple_img = np.zeros((camera_field.shape[0], int(camera_field.shape[1]/2), 3))
        if self.data_recorder.AddNum == 0:
            try:
                # TODO:
                if self.data_recorder.current_category_index > 1 and self.data_recorder.current_category_index < 11:
                    _name = self.model_manager.std_meshes_names
                    obj_exanple_img = self.obj_exanple_img_dict[_name[self.data_recorder.current_category_index - 2]]
                    obj_exanple_img = cv2.resize(obj_exanple_img, (int(camera_field.shape[1]/2), camera_field.shape[0]))
            except KeyError:
                pass

        # record_pos_img 
        if self.record_pos_list_updated:
            self.update_record_pos_image(camera_field.shape[0], int(camera_field.shape[1]/2))
        record_pos_image = self.record_pos_image
        info_board = np.hstack((obj_exanple_img, record_pos_image))
        info_board = cv2.resize(info_board, (camera_field.shape[1], camera_field.shape[0])).astype(np.uint8)

        # 拼接
        shown_image = np.vstack((camera_field, info_board))  # 将信息板和信息图像垂直拼接在一起

        # 如果正在等待开始，显示倒计时
        if self.is_waiting:
            text_pos = tuple((np.array(shown_image.shape[:2]) / 2).astype(np.uint)[::-1])  # 计算文本位置坐标
            shown_image = cv2.putText(shown_image, str(WAIT_LENGTH-int(time.time() - self.T_start)),
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 6,(0,0,255),3,cv2.LINE_AA)
        # 如果已经开始记录数据
        else:
            # 显示当前已记录数据的数量和最大深度变化值
            cv2.putText(shown_image,"all:" + str(self.data_recorder.num) + \
                        "|" * len(self.color_image_list),
                        (0,480 + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),2,cv2.LINE_AA, False)
            # 显示当前正在记录的模型名称和已记录的数据范围
            cv2.putText(shown_image,"{}:{}".format(self.data_recorder.current_category_name,
                                                    self.data_recorder.current_categroy_num),
                (0,480 + 200),
                cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),2,cv2.LINE_AA, False)
            cv2.putText(shown_image,"add:{}".format(str(self.data_recorder.AddNum)),
                        (0,480 + 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),2,cv2.LINE_AA, False)
            cv2.putText(shown_image, self.info_txt,
                        (0,480 + 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),2,cv2.LINE_AA, False)
            # 显示aruco位置
            for aruco in self.this_floor_aruco_corners:
                point = np.mean(aruco, axis=0)
                point = point * resize_ratio
                shown_image = cv2.circle(shown_image, point.astype(np.int16), 12, (0,0,255), -1)
            for aruco, id in zip(self.this_other_aruco_corners, self.this_other_aruco_ids):
                point = np.mean(aruco, axis=0)
                point = point * resize_ratio
                cv2.putText(shown_image, str(id),
                        point.astype(np.int16),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1, cv2.LINE_AA, False)
                shown_image = cv2.circle(shown_image, point.astype(np.int16), 12, (0, 255, 0), -1)
        cv2.imshow('COLOR&DEPTH IMAGE',shown_image)

    def process_one_frame(self, color, depth_frame):
        '''
        return
        -----
        color_image, depth_frame, info_txt
        '''
        ### 判断是否可以采集
        # 检测aruco
        gray_src:cv2.Mat = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners_src, ids_src, rejectedImgPoints = \
            self.aruco_detector._inner_aruco_detector.detectMarkers(gray_src)
        ### 区分floor_aruco和其他aruco
        if ids_src is None:
            this_floor_aruco_num = 0
            this_other_aruco_num = 0
            this_other_aruco_ids = np.array([], np.int32)
            floor_idx = None
            other_idx = None
        else:
            ids_src = np.squeeze(ids_src, axis=-1)
            floor_idx = (ids_src >= 20) * (ids_src < 32)
            other_idx = np.logical_not(floor_idx)
            floor_idx = np.where(floor_idx)[0]
            other_idx = np.where(other_idx)[0]
            this_floor_aruco_ids = ids_src[floor_idx]
            this_other_aruco_ids = ids_src[other_idx]
            this_floor_aruco_num = this_floor_aruco_ids.size #id必须是20-31
            this_other_aruco_num = this_other_aruco_ids.size
            corners_src = np.squeeze(corners_src,1)
        corners_dict_list = []
        for num, idx in zip((this_floor_aruco_num, this_other_aruco_num),
                            (floor_idx, other_idx)):
            corners_dict = {}
            if num > 0:
                cs = corners_src[idx]
                # cs = cs[np.argsort(idx)]
                # aid = idx[np.argsort(idx)]
                for i, c in zip(idx, cs):
                    id = ids_src[i]
                    corners_dict.update({id: c})
            else:
                pass
            corners_dict_list.append(corners_dict)
        floor_aruco_dict, other_aruco_dict = corners_dict_list
        # aruco数量阈值
        if self.data_recorder.current_category_index == 0:
            aruco_num_thre = FIRST_FRAME_ARUCO_NUM
        elif self.data_recorder.current_category_index == 1:
            aruco_num_thre = BASE_FRAME_ARUCO_NUM
        else:
            aruco_num_thre = SINGLE_BASE_FRAME_ARUCO_NUM
        if this_floor_aruco_num >= aruco_num_thre and (self.rs_camera.motion_module.is_stable or self.ignore_stable):
            self.info_txt = "ok"
            enter_buffer_permitted = True
        else:
            enter_buffer_permitted = False
            if this_floor_aruco_num < aruco_num_thre:
                self.info_txt = "aruco:{}/{}".format(str(this_floor_aruco_num).rjust(2, ' '), aruco_num_thre)
            else:
                self.info_txt = "not stable!"

        if self.ignore_stable:
            self.keep_at_last_position = False
            # enter_buffer_permitted = True
        else:
            if self.keep_at_last_position and not self.rs_camera.motion_module.is_stable:
                self.keep_at_last_position = False
            if self.keep_at_last_position:
                enter_buffer_permitted = False # 是否离开了上一个

        self.this_color                 = color
        self.this_depth_frame           = depth_frame
        self.this_floor_aruco_dict      = floor_aruco_dict 
        self.this_other_aruco_dict      = other_aruco_dict
        self.this_other_aruco_ids       = this_other_aruco_ids

        if enter_buffer_permitted:
            pass
        else:
            self.color_image_list.clear()
            self.depth_frame_list.clear()
            self.aruco_dict_list.clear()

        return enter_buffer_permitted

    def __pushbask_to_buffer(self):
        '''
        color: np.ndarray [H, W]
        depth_frame: rs.depthframe
        aruco_corners: np.ndarray [N, 4, 3]
        向buffer添加元素
        '''
        if self.this_color is None:
            return False
        self.color_image_list.append(self.this_color)
        self.depth_frame_list.append(self.this_depth_frame)
        self.aruco_dict_list.append(self.this_floor_aruco_dict)
        if len(self.color_image_list) > BUFFER_MAX_LENGTH:
            self.color_image_list.pop(0)
            self.depth_frame_list.pop(0)
            self.aruco_dict_list.pop(0)
            return True
        return False

    def __clear_buffer(self):
        '''
        向buffer添加元素
        '''
        self.color_image_list.clear()
        self.depth_frame_list.clear()
        self.aruco_dict_list.clear()

    @staticmethod
    def __trans_mat_2_pos_point(transform:np.ndarray)->np.ndarray:
        # 计算采集的方位
        z_points = np.array([[0,0,0,1], [0,0,1,1]]) #[2,4]
        z_points_C = transform.dot(z_points.T).T #[2,4]
        pos = (z_points_C[0] - z_points_C[1])[:3]
        return pos

    def process_buffer(self):
        '''
        添加到缓冲区
        '''
        if not TEST:
            if not self.__pushbask_to_buffer():
                return
            ### 比较最后一帧与前面逐帧的aruco位置的差，总差求和
            try:
                ids_list = [list(d.keys()) for d in self.aruco_dict_list]
                sets = [set(lst) for lst in ids_list]# 将所有子列表转换为set类型
                common_ids = list(set.intersection(*sets))# 计算所有子列表的交集
                if len(common_ids) < 2:
                    max_delta = 1e6
                else:
                    arcuo_array = np.array([[d[id_] for id_ in common_ids] for d in self.aruco_dict_list])
                    deltas = arcuo_array - arcuo_array[-1] #[MAX_LENGTH, N, 4, 2]
                    max_delta = np.max(np.linalg.norm(deltas, axis = -1))
                    # print(max_delta)
            except ValueError:
                max_delta = 1e6
            ### 根据采集的内容调整对精度的严格程度
            if self.data_recorder.current_category_index == 0:
                restrict = 0.85
            elif self.data_recorder.current_category_index == len(self.model_manager.std_meshes_names) - 1:
                restrict = 2.0
            else:
                restrict =  1.0
            ### 采集后处理与保存
            if self.record_gate and max_delta < DELTA_THRESHOLD*restrict:
                # 可以采集
                c = multiframe_distortion_correction(self.color_image_list, 
                                                    (self.rs_camera.intr.cam_hgt, self.rs_camera.intr.cam_wid))
                for df in self.depth_frame_list:
                    depth_frame = self.rs_camera.temporal.process(df)
                d = np.asanyarray(depth_frame.get_data())
                # 对采集的帧进行误差检验
                ok, transform = self.aruco_detector.verify_frame(c, d, self.rs_camera.intr)
                # ok = True
                if ok:
                    self.data_recorder.save_frames(c, d, transform)
                    self.keep_at_last_position = True
                    # 计算采集的方位
                    pos = self.__trans_mat_2_pos_point(transform)
                    self.pushback_to_record_pos_list(pos)
                else:
                    pass
                self.__clear_buffer()
        else:
            c = np.load("cad_test.npy")
            d = np.load("depth_test.npy")
            # 对采集的帧进行误差检验
            ok, transform = self.aruco_detector.verify_frame(c, d, self.rs_camera.intr)
            # ok = True
            if ok:
                self.data_recorder.save_frames(c, d, transform)
                self.keep_at_last_position = True
                # 计算采集的方位
                pos = self.__trans_mat_2_pos_point(transform)
                self.pushback_to_record_pos_list(pos)
            else:
                pass
            self.__clear_buffer()           


    # for mode, func in zip([Camera.MODE_DATA], [callback_DATA]):
    #     BUFFER_MAX_LENGTH = 4        
    #     record_pipeline.rs_camera = Camera(folder, mode)
    #     # record_pipeline.ignore_stable = True     
    #     record_pipeline.data_recorder.skip_to_seg()   
    #     record_pipeline.start(func)


   

# if __name__ == "__main__":

#     is_recording_model = True
#     record_gate = True

#     def callback_CALI(data_recorder:Recorder):
#         if data_recorder.current_category_index != FRAMETYPE_DATA:
#             return False
#         else:
#             return True

#     def callback_DATA(data_recorder:Recorder):
#         return False

#     ### 记录器
#     std_model_path = os.path.join(os.path.abspath(os.path.join(folder, "..")), "models")
#     data_recorder = Recorder(folder, std_model_path)
#     start_black_time = 0
#     ### 相机初始化
#     rs_camera = Camera(folder, Camera.MODE_CALI)
#     gt_posture_computer = GtPostureComputer(folder)

#     record_pipeline = RecordingPipeline(folder, data_recorder, rs_camera, gt_posture_computer)
#     # record_pipeline.data_recorder.remove(list(range(421, 847)))
#     record_pipeline.data_recorder.add_skip_seg(len(record_pipeline.data_recorder.std_meshes_names) - 1)
    
#     for mode, func in zip([Camera.MODE_CALI, Camera.MODE_DATA], [callback_CALI, callback_DATA]):
#         record_pipeline.rs_camera = Camera(folder, mode)
#         record_pipeline.start(func)

#     # for mode, func in zip([Camera.MODE_DATA], [callback_DATA]):
#     #     BUFFER_MAX_LENGTH = 4        
#     #     record_pipeline.rs_camera = Camera(folder, mode)
#     #     # record_pipeline.ignore_stable = True     
#     #     record_pipeline.data_recorder.skip_to_seg()   
#     #     record_pipeline.start(func)



