import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


class pc_voxelize_reture(enumerate):
    BOX = 0
    SURF = 1
    O3DPCD = 2

def pc_voxelize(pcd, voxel_size = 1, reture_type:pc_voxelize_reture = pc_voxelize_reture.BOX, pcd_color = np.array([0,0,0])):
    '''
    点云体素化
    pcd: [N,4]
    '''
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(pcd[:,:3])
    if pcd_color.size == 3:
        pcd_color = np.tile(np.expand_dims(pcd_color, 0), [pcd.shape[0], 1])
    elif pcd_color.shape[0] == pcd.shape[0]:
        pass
    else:
        raise ValueError("shape of pcd_color is illegal")
    if np.issubdtype(pcd_color.dtype, np.floating):
        pcd_color = (pcd_color * 255).astype(np.uint8)
    o3dpcd.colors = o3d.utility.Vector3dVector(pcd_color[:,:3])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3dpcd, voxel_size=voxel_size)
    voxels:list = voxel_grid.get_voxels()  # returns list of voxels
    
    if len(voxels) == 0:
        surf_indices = np.array([], np.float)
        surf_color = np.array([], np.float)
        offset = np.array([0, 0, 0], np.float)
    else:
        surf_indices = np.stack(list(vx.grid_index for vx in voxels))
        surf_color = np.stack(list(vx.color for vx in voxels))
        offset = np.min(pcd[:,:3], 0)
    restore_mat = np.eye(4)*voxel_size
    restore_mat[3, 3] = 1
    restore_mat[:3, 3] = offset
    if reture_type == pc_voxelize_reture.BOX:
        box = np.zeros(np.max(surf_indices, axis=0)+1, np.uint8)
        box[surf_indices[:,0], surf_indices[:,1], surf_indices[:,2]] = 1
        color_box = np.zeros([*box.shape, 3], np.uint8)
        color_box[surf_indices[:,0], surf_indices[:,1], surf_indices[:,2]] = surf_color
        return box, color_box, restore_mat
    elif reture_type == pc_voxelize_reture.SURF:
        return surf_indices, surf_color, restore_mat
    elif reture_type == pc_voxelize_reture.O3DPCD:
        return voxel_grid, restore_mat
