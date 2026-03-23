
import numpy as np
import cv2

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from .point import Point
from .frame import Frame
from .pose import pose_matrix_to_vector, pose_vector_to_matrix
## 修改自
## https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
## 速度太慢，效果太差，須確認原理後修改

def rotate(points, rot_vecs):
    """rotate 
    以指定的旋轉向量旋轉點。
    使用 Rodrigues 的旋轉公式。
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """透過投影至影像，將 3-D 點轉換為 2-D。"""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
   

    return points_proj


# def project(points, frame_pose):
#     """_summary_

#     Args:
#         points (np.ndarray): 形狀為 (N, 4) 的齊次座標3D點 n*(X,Y,Z,1)
#         frame_pose (np.ndarray): 形狀為 (4, 4) 的相機位姿投影矩陣[R|t]

#     Returns:
#         points(np.ndarray): 形狀為 (N, 2) 的非齊次座標2D點 n*(X,Y)
#     """
    
#     points_f = (frame_pose @ points.T).T
#     points_projected = points_f[:,:2] / points_f[:,2][:, np.newaxis]
#     return points_projected

def compute_res(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """計算殘差。
    
    `params` 包含攝影機參數和 3D 座標。
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


# def compute_res(params,n_cameras, n_points, camera_indices, point_indices, points_2d):
#     """計算殘差。"""
#     print(params.shape,n_cameras, n_points, camera_indices.shape, point_indices.shape, points_2d.shape)
    
#     frames_pose = params[:n_cameras * 6].reshape((n_cameras, 6))
#     points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
#     print(frames_pose.shape)
#     frame_pose_v = frames_pose[camera_indices]
#     print(frame_pose_v.shape)
#     frame_pose_m = pose_vector_to_matrix(frame_pose_v[:3],frame_pose_v[3:])
#     points_projected = project(points_3d[point_indices],frame_pose_m)
    
#     return (points_projected - points_2d).ravel()


def jacobian_sparsity_structure(n_cameras, n_points, camera_indexes, point_indexes):
    #雅可比稀疏結構（即已知非零的標記元素）
    m = camera_indexes.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indexes.size)
    for s in range(6):
        A[2 * i, camera_indexes * 6 + s] = 1
        A[2 * i + 1, camera_indexes * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indexes * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indexes * 3 + s] = 1

    return A



def bundle_adjustment_core(frames_pose,camera_indices,points_3d,point_indices,points_2d):
    
    # print('frames_pose.shape',frames_pose.shape)
    # print('points_3d.shape',points_3d.shape)
    
    n_cameras = frames_pose.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((frames_pose.ravel(), points_3d.ravel()))
    A = jacobian_sparsity_structure(n_cameras, n_points, camera_indices, point_indices)
    
    # 由於least_squares的特性，不能將旋轉矩陣限制成 SO(3),所以將選轉矩陣轉為向量
    # 由於least_squares的特性，3D點齊次座標(n*4)也需要轉為非齊次座標(n*3)
    result = least_squares(compute_res, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-2, method='trf', 
                           args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    
    
    
    return result

def bundle_adjustment(frames:list[Frame]):
    
    points:set[Point] = set()
    points_3d = []
    
    points_2d = []
    frame_indices = []
    point_indices = []
    
    frames_pose = []
    
    # 輸入的 frames 的 pose ，按照 frame.id 排序，並添加所有出現的 points 
    for frame in frames:
        frames_pose.append(pose_matrix_to_vector(frame.pose))  # 將4*4轉換矩陣分解成6*1旋轉向量+平移向量
        points.update(set(frame.points)) # 使用 set 群 格式，避免重複添加同一point
        
    
    # 根據 point.id 排序點的3D位置，並添加其在對應frame的2D位置
    for i, point in enumerate(points):
        points_3d.append(point.location[:3]) # 齊次座標轉為非齊次座標
        
        for frame in point.frames:
            if frame in frames:
                
                f_kp_index = frame.point_to_kp_idx[point.id]
                points_2d.append(frame.kps[f_kp_index])
                
                index = frames.index(frame)
                frame_indices.append(index)
                point_indices.append(i)
        
        

    
    
    result = bundle_adjustment_core(np.array(frames_pose),np.array(frame_indices),np.array(points_3d),np.array(point_indices),np.array(points_2d))
    
    
    
    result_frames_pose = result.x[:len(frames) * 6].reshape((len(frames), 6))


    adj_points_3d = result.x[len(frames) * 6:].reshape((len(points), 3))
    
    # 旋轉向量+平移向量 轉換回 4*4轉換矩陣
    for frame, result_frame_pose in zip(frames, result_frames_pose):
        adj_pose =  pose_vector_to_matrix(result_frame_pose[:3],result_frame_pose[3:])
        frame.pose = adj_pose

    # 轉回齊次座標
    for point,adj_point in zip(points,adj_points_3d):
        point.location = np.append(adj_point, 1)

