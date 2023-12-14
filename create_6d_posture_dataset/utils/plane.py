"""
plane.py
---------------

Functions related to plane segmentation.

"""
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.optimize import leastsq
from sklearn.ensemble import IsolationForest

def f_min(X,p):
    plane_xyz = p[0:3]
    distance = (plane_xyz*X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, signal, X):
    return f_min(X, params)

import matplotlib.pyplot as plt
def findplane_wo_outliers(points_3d, aruco_size = 0.056, return_out_index = False):
    '''
    points_3d [N, 3] 
    '''
    N = points_3d.shape[0]
    ifshow = False
    sol = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]
    if ifshow:
        plt.figure(0)
        ax = plt.axes(projection='3d')
        ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], s = 10, marker = 'o')
        plt.show()
    out_side = []
    out_index = []
    ### 检查每一组的形状，形状不合格的边将被记录，
    # 端点出现1次以上的将被直接排除
    # 其他嫌疑点暂时排除，用非嫌疑点匹配平面，再计算嫌疑点到平面的距离，超过方差*3的被排除，剩余点再次计算平面
    # 首先排除离群组内，最近
    for i, p in enumerate(points_3d):
        cpindx = i % 4
        if cpindx == 3:
            p1 = i
            p2 = i - 3
        else:
            p1 = i
            p2 = i + 1 
        dis = np.linalg.norm(points_3d[p1,:] - points_3d[p2,:])
        if np.abs(dis - aruco_size) > 0.010:
            out_side.append((p1, p2))
    out_side = np.reshape(np.array(out_side), -1)
    suspect_index = np.unique(out_side)
    unsuspect_index = np.setdiff1d(np.arange(N), suspect_index)
    for i in suspect_index:
        if np.sum(out_side == i) > 1:
            out_index.append(i)
    suspect_index = np.setdiff1d(suspect_index, out_index)
    if unsuspect_index.size < 4:
        return None
    else:
        unsuspect_point = points_3d[unsuspect_index]
        sol = leastsq(residuals, sol, args=(None, unsuspect_point.T))[0]
        sol = sol/np.linalg.norm(sol[:3])

    if suspect_index.size > 0:
        distance = np.abs(np.sum(points_3d * sol[:3], axis=-1) + sol[3]) # 所有点到平面的距离
        unsuspect_distance = distance[unsuspect_index] #非嫌疑点到平面的距离
        suspect_distance = distance[suspect_index]     #嫌疑点到平面的距离
        unsuspect_std = np.std(unsuspect_distance)
        out_index += suspect_index[suspect_distance > 3.5 * unsuspect_std].tolist()
        # 再次计算
        remain = np.setdiff1d(np.arange(N), np.array(out_index))
        points_3d = points_3d[remain]
        if ifshow:
            plt.figure(0)
            ax = plt.axes(projection='3d')
            ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], s = 10, marker = 'o')
            plt.show()
        sol = leastsq(residuals, sol, args=(None, points_3d.T))[0]
        sol = sol/np.linalg.norm(sol[:3])        
    else:
        out_index = np.array([], np.int32)
    if return_out_index:
        return sol, np.array(out_index, np.int32)
    else:
        return sol
    # depth = np.linalg.norm(points_3d, axis=-1)
    # std = np.std(depth)
    # mean = np.mean(depth)
    # distance = np.abs(depth - mean)
    # points_3d = points_3d[distance < 3*std]

    # max_distance = 100
    # max_iter = int(np.round(points_3d.shape[0] * 0.2))
    # iter = 0
    # plot = False
    # while iter < max_iter:
    #     plot = False
    #     sol = leastsq(residuals, sol, args=(None, points_3d.T))[0]
    #     sol = sol/np.linalg.norm(sol[:3])
    #     distance = np.abs(np.sum(points_3d * sol[:3], axis=-1) + sol[3])
    #     if plot:
    #         plt.figure(0)
    #         ax = plt.axes(projection='3d')
    #         ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], s = 10, marker = 'o')
    #         plt.show()
    #     max_distance = np.max(distance)
    #     if (max_distance > 0.007):
    #         remain = np.setdiff1d(np.arange(points_3d.shape[0]), np.argmax(distance))
    #         points_3d = points_3d[remain]
    #         iter += 1
    #     else:
    #         break
    # if iter == max_iter:
    #     return None
    # else:
    #     return sol

def findplane(cad,d):
    p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]
    sol = None
    gray = cv2.cvtColor(cad, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    XYZ = [[],[],[]]
    if np.all(ids != None):
        for index,cornerset in enumerate(corners):
            cornerset = cornerset[0]
            for corner in cornerset:
                if d[int(corner[1])][int(corner[0])][2]!= 0:
                    XYZ[0].append(d[int(corner[1])][int(corner[0])][0])
                    XYZ[1].append(d[int(corner[1])][int(corner[0])][1])
                    XYZ[2].append(d[int(corner[1])][int(corner[0])][2])


        XYZ = np.asarray(XYZ)
        sol = leastsq(residuals, p0, args=(None, XYZ))[0]

    return sol

def fitplane(p0,points):
  
    XYZ = np.asarray(points.T)
    sol = leastsq(residuals, p0, args=(None, XYZ))[0]

    return sol

def point_to_plane(X,p):
    height,width,dim = X.shape
    X = np.reshape(X,(height*width,dim))
    plane_xyz = p[0:3]
    distance = (plane_xyz*X).sum(axis=1) + p[3]
    distance = distance / np.linalg.norm(plane_xyz)
    distance = np.reshape(distance,(height,width))
    return distance
