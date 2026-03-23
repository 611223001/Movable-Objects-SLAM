import numpy as np
import cv2
import g2o

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from pytransform3d import transformations,rotations

from .utils import add_ones

from typing import TYPE_CHECKING,Union
if TYPE_CHECKING:
    from .map import Map
    from .frame import Frame, KeyFrame
    from .point import Point
    from .slam import Slam

class Pose():
    def __init__(self,pose=np.eye(4)):
        '''
        pose  定義為4x4 齊次座標轉換矩陣 Twc = Rt，描述「相機座標系到世界座標系」的變換，或是表達相機「在世界座標系」的位置。此時 t 為pose座標中心點在世界座標系的的座標
            
        T = 
            |R t|
            |0 1|
        '''        
        self._pose:np.ndarray[tuple[int, int],np.dtype[np.float64]] = pose

    @property
    def pose(self): # Twc
        return self._pose
    @pose.setter
    def pose(self, value:np.ndarray[tuple[int, int],np.dtype[np.float64]]):
        """ Twc
        """
        self._pose = value
    
    @property
    def inv(self):
        # Tcw
        return np.linalg.inv(self._pose)
    
    @property
    def Twc(self):
        return self._pose
    @property
    def Rwc(self):
        return self._pose[:3, :3]
    @property
    def twc(self):
        return self._pose[:3,  3]
    
    @property
    def Tcw(self):
        Tcw = np.eye(4)
        Tcw[:3, :3] = self.Rcw
        Tcw[:3, 3] = self.tcw
        return Tcw

        # return self.inv()
    @property
    def Rcw(self):
        return self.Rwc.T
    @property
    def tcw(self):
        return - (self.Rwc.T @ self.twc)
    
    @property
    def Ow(self):
        return self.twc
    
    @property
    def quat(self):
        # 世界座標下相機 方向 四元數
        return rotations.quaternion_from_matrix(self.Rwc) 
    @property
    def euler(self):
        # 世界座標下相機 方向 歐拉角 xyz
        return rotations.euler_from_matrix(self.Rwc, 0, 1, 2, extrinsic=False) 
    
    @property
    def orientation(self):
        # 世界座標下相機 方向 歐拉角 
        return self.euler
    
    @property
    def position(self):
        """
        np.array(x,y,z)
        """
        return self.twc.T
    
    
    def set_pose(self,t_vec,R_mat=None,quat=None,euler=None):
        if R_mat is not None:
            pose = transformations.transform_from(R_mat, t_vec)
        elif quat is not None:
            pose = transformations.transform_from_pq(np.hstack((t_vec, quat)))
        elif euler is not None:
            R_euler = rotations.matrix_from_euler(euler, 0, 1, 2, False)
            pose = transformations.transform_from(R_euler, t_vec)
        
        self._pose = pose


def triangulate(pose1, pose2, pts1, pts2):
    """
    根據兩個相機位姿和對應的圖像點對，進行三角測量以恢復三維點。

    Parameters
    ----------
    pose1 : np.ndarray
        第一個相機的 4x4 位姿矩陣（世界到相機的變換）。Twc
    pose2 : np.ndarray
        第二個相機的 4x4 位姿矩陣（世界到相機的變換）。Twc
    pts1 : np.ndarray
        第一個圖像上的點座標，形狀為 (N, 2)，每一列是一個點的 (x, y)。
    pts2 : np.ndarray
        第二個圖像上的點座標，形狀為 (N, 2)，每一列是一個點的 (x, y)。

    Returns
    -------
    ret : np.ndarray
        三維齊次座標點，形狀為 (N, 4)，每一行是一個點的齊次三維座標 (X, Y, Z, W)。
    """
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret





def computeF12(kf1:'KeyFrame', kf2:'KeyFrame'):
        """
        根據兩個 KeyFrame 的姿態與內參計算基礎矩陣 F12
        """
        R1w = kf1.Rcw
        t1w = kf1.tcw
        R2w = kf2.Rcw
        t2w = kf2.tcw

        R12 = R1w @ R2w.T
        t12 = -R1w @ R2w.T @ t2w + t1w

        # Skew-symmetric matrix
        def skew(t):
            return np.array([
                [0, -t[2], t[1]],
                [t[2], 0, -t[0]],
                [-t[1], t[0], 0]
            ])

        t12x = skew(t12.flatten())

        K1 = kf1.camera.K
        K2 = kf2.camera.K

        F12 = np.linalg.inv(K1.T) @ t12x @ R12 @ np.linalg.inv(K2)
        return F12

def check_dist_epipolar_line(un_kp1, un_kp2, oct2:int, F12: np.ndarray, kf2:'KeyFrame') -> bool:
    """
    檢查點 pt2 到由 pt1 與 F12 定義的極線距離是否小於閾值。
    輸入:
      pt1: (x,y) 參考影像點座標
      pt2: (x,y) 目標影像點座標
      oct2: pt2 的 octave（用於查詢 sigma^2）
      F12: 3x3 基礎矩陣
      kf2: KeyFrame 物件，需提供 get_sigma2(octave)（或類似）方法
    回傳:
      bool: 若距離平方 < 3.84 * sigma2 回傳 True
    """
    x1, y1 = float(un_kp1[0]), float(un_kp1[1])
    x2, y2 = float(un_kp2[0]), float(un_kp2[1])

    # 計算極線 l = x1' * F12 = [a b c]
    a = x1 * float(F12[0, 0]) + y1 * float(F12[1, 0]) + float(F12[2, 0])
    b = x1 * float(F12[0, 1]) + y1 * float(F12[1, 1]) + float(F12[2, 1])
    c = x1 * float(F12[0, 2]) + y1 * float(F12[1, 2]) + float(F12[2, 2])

    num = a * x2 + b * y2 + c
    den = a * a + b * b

    if den == 0.0:
        return False

    dsqr = (num * num) / den

    sigma2 = kf2.get_sigma2(oct2)

    return dsqr < 3.84 * sigma2







# def estimate_pose(F):
#     W = np.asmatrix([[0,-1,0],
#                      [1, 0,0],
#                      [0, 0,1]])
    
#     U,d,Vt = np.linalg.svd(F)
#     if np.linalg.det(U) < 0:
#         # return None, True 
#         U *= -1.0
    
#     if np.linalg.det(Vt) < 0:
#         Vt *= -1
#     R = np.dot(np.dot(U, W), Vt)
#     if np.sum(R.diagonal()) < 0:
#         R = np.dot(np.dot(U, W.T), Vt)
#     t = U[:, 2]
#     Rt = np.eye(4)
#     Rt[:3, :3] = R
#     Rt[:3, 3] = t

#     return Rt, False # Transform Matrix 

# def predict_pose(idx1,idx2,ret):
    
#     assert len(ret) > 8

#     # # Fit matrix
#     # model, inliers = ransac((ret[:, 0], 
#     #                         ret[:, 1]), FundamentalMatrixTransform, 
#     #                         min_samples=8, residual_threshold=0.02, 
#     #                         max_trials=400)
    
#     # # Ignore outliers
#     # ret = ret[inliers]
#     # Rt, bad = estimate_pose(model.params)
    
    
#     # E, mask = cv2.findEssentialMat(ret[:, 0], ret[:, 1], focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)  # 計算本質矩陣
#     # _, R, t, mask = cv2.recoverPose(E, ret[:, 0], ret[:, 1], focal=self.focal, pp=self.pp)  # 恢復位姿
    
#     F, mask = cv2.findFundamentalMat(ret[:, 0], ret[:, 1], method=cv2.RANSAC, ransacReprojThreshold=0.02, confidence=0.999)

#     _, R, t, mask = cv2.recoverPose(F, ret[:, 0], ret[:, 1],mask=mask)  # 恢復位姿

#     Rt = np.eye(4)
#     Rt[:3, :3] = R
#     Rt[:3, 3] = t.flatten()
    
#     return idx1, idx2, Rt

#     #     if bad:
#     #         return np.array([]),np.array([]),np.array([]),True
        
#     #     return idx1[inliers], idx2[inliers], Rt, False
#     # else:

#     #     return np.array([]),np.array([]),np.array([]),True
    
    

# # Extract matched keypoints
# pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
# pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
 
# # Camera intrinsic parameters (example values, replace with your camera's calibration data)
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])
 
# # Compute the Fundamental matrix using RANSAC
# F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
 
# # Compute the Essential matrix using the camera's intrinsic parameters 
# E = K.T @ F @ K
 
# # Decompose the Essential matrix to get R and t
# _, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

# np.array.T

# 棄用
'''
def pose_matrix_to_vector(transform_matrix):
    """
    將 4x4 轉換矩陣分解為旋轉向量和平移項。

    Args:
        transform_matrix (np.ndarray): 形狀為 (4, 4) 的轉換矩陣。

    Returns:
        np.ndarray: (旋轉向量, 平移項)
    """
    # 提取旋轉矩陣 (3x3) 和平移向量 (3x1)
    rotation_matrix = transform_matrix[:3, :3]
    translation_vector = transform_matrix[:3, 3]

    # 將旋轉矩陣轉換為旋轉向量
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)

    return rotation_vector.flatten(),translation_vector

def pose_vector_to_matrix(rotation_vector, translation_vector):
    """
    將旋轉向量和平移項組合為 4x4 的轉換矩陣。

    Args:
        rotation_vector (np.ndarray): 形狀為 (3,) 的旋轉向量。
        translation_vector (np.ndarray): 形狀為 (3,) 的平移項。

    Returns:
        np.ndarray: 形狀為 (4, 4) 的轉換矩陣。
    """
    # 將旋轉向量轉換為旋轉矩陣
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 組合旋轉矩陣和平移向量
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector

    return transform_matrix

def quaternion_to_rotation_vector(quaternion:g2o.Quaternion):
    """
    將四元數轉換為旋轉向量。

    Args:
        quaternion (list or np.ndarray): 四元數 [w, x, y, z]

    Returns:
        np.ndarray: 旋轉向量 (3,)
    """
    w, x, y, z = quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()

    # 計算旋轉角度 theta
    theta = 2 * np.arccos(w)

    # 計算旋轉軸 u
    sin_theta_half = np.sqrt(1 - w**2)
    if sin_theta_half < 1e-6:  # 避免除以零
        u = np.array([0, 0, 0])
    else:
        u = np.array([x, y, z]) / sin_theta_half

    # 計算旋轉向量
    rotation_vector = theta * u
    return rotation_vector
'''
'''
def predict_points(f1:Frame,f2:Frame,kp_idxs1,kp_idxs2):
    # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
    #predict pose

    pts4d = triangulate(f1.pose, f2.pose, f1.kps[kp_idxs1], f2.kps[kp_idxs2])
    
    # This line normalizes the 3D points by dividing each row by its fourth coordinate W
    # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates
    pts4d /= pts4d[:, 3:]


    # Reject points without enough "Parallax" and points behind the camera
    # checks if the absolute value of the fourth coordinate W is greater than 0.005.
    # checks if the z-coordinate of the points is positive.
    # returns, A boolean array indicating which points satisfy both criteria.
    # 拒絕沒有足夠「視差」的點和攝影機後面的點
    # 檢查第四座標 W 的絕對值是否大於 0.005。 TODO: 確認為什麼
    # 檢查點的 z 坐標是否為正值。
    # 返回，一個布林陣列，表示哪些點符合這兩個條件。
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    
    # good_pts4d &= np.array([f1.points[i] is None for i in kp_idxs1])

    for i, (p_loc, kp_id1, kp_id2) in enumerate(zip(pts4d,kp_idxs1,kp_idxs2)):

        #  If the point is not good (i.e., good_pts4d[i] is False), the loop skips the current iteration and moves to the next point.
        if  good_pts4d[i]:
            
            if f2.points[kp_id2]is None:
                pt = Point(f1.slam, p_loc, frame=f2, idx=kp_id2)
            else:
                pt = f2.points[kp_id2]
            
            f1.map.add_point_frame_relation(pt,f1,kp_id1)
            # pt.add_frame(f2, kp_id2)
            
            pt.class_id = f1.kp_index_get_class_id(kp_id1)
            
            x, y = f1.raw_kps[kp_id1]
            pt.color = f1.img[y][x]

    return 

'''