import numpy as np


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics: dict):
    """
    Convert the depthmap to a 3D point cloud
    Parameters:
    -----------
    depth_frame : (m,n) uint16
            The depth_frame containing the depth map

    camera_intrinsics : dict 
            The intrinsic values of the depth imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    pointcloud : (m,n,3) float
            The corresponding pointcloud in meters

    """

    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    fx, ppx, fy, ppy  = camera_intrinsics['intr_M'].flatten()[(0, 2, 4, 5),]
    x = (u.flatten() -
         float(ppx))/float(fx)
    y = (v.flatten() -
         float(ppy))/float(fy)
    depth_image = depth_image*float(camera_intrinsics['depth_scale'])
    z = depth_image.flatten()
    z[z == 0] = camera_intrinsics['max_depth'] #不允许深度为0
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    pointcloud = np.dstack((x, y, z)).reshape(
        (depth_image.shape[0], depth_image.shape[1], 3))

    return pointcloud