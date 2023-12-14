# from ..data import dataset_format
from ..data import viewmeta
from ..data.mesh_manager import MeshManager, MeshMeta, Voxelized
from ..derive import calc_masks, cvt_by_intr
from ..core.posture import Posture, SphereAngle
from ..core.intr import CameraIntr
# from ..data.dataset_format import *
from ..data.IOAbstract import FilesCluster, FilesHandle, DatasetNode
from ..data.dataCluster import UnifiedFileCluster, UnifiedFilesHandle, \
                                DisunifiedFileCluster, DisunifiedFilesHandle,\
                                DictLikeCluster, DictLikeHandle, DictFile
from ..data.dataset import Dataset
                            
from ..data.viewmeta import ViewMeta
from ..core.utils import JsonIO

# RGB_DIR     = LinemodFormat.RGB_DIR
# MASK_DIR    = LinemodFormat.MASK_DIR
# DEPTH_DIR   = LinemodFormat.DEPTH_DIR

RGB_DIR     = "rgb"
MASK_DIR    = "masks"
DEPTH_DIR   = "depth"
OUTPUT_DIR  = "output"
AUG_OUTPUT_DIR = "aug_output"
TRANS_DIR   = "trans"
VORONOI_SEGPCD_DIR = "voronoi_segpcd"
SEGMESH_DIR = "segmesh"
REGIS_DIR   = "registeredScene"
ICP_DIR     = "icp"
REFINER_DIR = "refiner"
REFINER_FRAME_DATA = "frame_data"
ARUCO_FLOOR = "aruco_floor"

CALI_INTR_FILE = "intrinsics_0.json"
DATA_INTR_FILE = "intrinsics_1.json"

FRAMETYPE_GLOBAL = "global_base_frames"
FRAMETYPE_LOCAL = "local_base_frames"
FRAMETYPE_DATA = "dataset_frames"