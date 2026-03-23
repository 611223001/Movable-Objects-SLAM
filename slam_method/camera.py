import numpy as np
from collections import deque
import cv2

from .utils import add_ones,is_blurry



class Camera:
    
    def __init__(self,config:dict):
        
        self.width:int = int(config['Camera.width'])
        self.height:int = int(config['Camera.height'])
        self.fx:float = float(config['Camera.fx'])
        self.fy:float = float(config['Camera.fy'])
        self.cx:float = float(config['Camera.cx'])
        self.cy:float = float(config['Camera.cy'])
        D = config['Camera.D']
        
        self.baseline = 0.0 #相機間的間距(m)，單目相機為0 ， g2o SBA需要
        
        
        # 保存最近的影像拉普拉斯變異數，用於判斷影像是否模糊
        self.blurry_var_buffer = deque(maxlen=30)
        self.blurry_var_buffer.append(0)
        
        
        self.fps:int = int(config.get('Camera.fps',10))
        # if self.fps is None:
        #     self.fps = 10
        
        
        self.K = np.array([[self.fx ,0       ,self.cx],
                           [0       ,self.fy ,self.cy],
                           [0       ,0       ,1         ]])
        
        self.Kinv = np.linalg.inv(self.K)
        
        # 畸變係數
        self.D = np.array(D, dtype=np.float32)
        
        self.viewing_angle = (self.normalize(np.array([[0,0],[self.width,self.height]]))) # [[x0,y0],[x1,y1]]
        
        self.radius = np.linalg.norm(self.viewing_angle[0]-self.viewing_angle[1])/2 # 投影到1m平面(標準平面)上的對角線半徑
        
        self.pixel_to_meter:float = 1/(self.fx * self.fx + self.fy * self.fy) # 用於像素閾值到標準化閾值的轉換
        
    def dynamic_determine_blurry(self,img):
        # 根據閾值判斷是否模糊並修改閼值
        threshold = np.percentile(self.blurry_var_buffer, 75) # 取前25%
        be_blurry, variance = is_blurry(img,threshold)
        
        self.blurry_var_buffer.append(variance)
        
        return be_blurry,variance
        
    def normalize(self, pts)->np.ndarray[int,float]:    #-> (x,y)
        # The inverse camera intrinsic matrix 𝐾 − 1 transforms 2D homogeneous points 
        # from pixel coordinates to normalized image coordinates. This transformation centers 
        # the points based on the principal point (𝑐𝑥 , 𝑐𝑦) and scales them 
        # according to the focal lengths 𝑓𝑥 and 𝑓𝑦, effectively mapping the points 
        # to a normalized coordinate system where the principal point becomes the origin and 
        # the distances are scaled by the focal lengths.
        # 逆相機本徵矩陣(intrinsic matrix) 𝐾 − 1 將二維齊次點從像素座標轉換為標準化影像座標。
        # 此變換以主點 (𝑐𝑥 , 𝑐𝑦) 為中心，並根據焦距 𝑓𝑥 和 𝑓𝑦 對它們進行縮放，
        # 有效地將點映射到標準化坐標系，其中主點成為原點，距離按焦距縮放。
        
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]
        # `[:, 0:2]` 選擇結果數組的前兩列，即標準化的 x 和 y 座標。
        # `.T` 將結果轉置回 N x 3.
        
    # def denormalize(self, pt):
    #     """

    #     輸入:
    #         pt: [x,y,1]的齊次2D座標

    #     輸出:
    #         tuple[int]: (x,y)像素座標
    #     """
    #     # [x,y,1]
    #     ret = np.dot(self.K, pt)
    #     ret /= ret[2]
    #     return int(round(ret[0])), int(round(ret[1]))
    
    # def denormalize(self, pt):
    #     """

    #     輸入:
    #         pt: [x,y,1]的齊次2D座標

    #     輸出:
    #         tuple[int]: (x,y)像素座標
    #     """
    #     # [x,y,1]
    #     ret = np.dot(self.K, pt)
    #     ret /= ret[2]
    #     return int(round(ret[0])), int(round(ret[1]))
    
    def denormalize(self, pts:np.ndarray[(-1,3),float])->np.ndarray[(-1,2),int]:
        """
        輸入:
            pts: shape (n,3) [x,y,1] 的齊次2D座標 numpy array

        輸出:
            np.ndarray: shape (n,2) [x,y]的像素座標
        """
        # pts: (n,3)
        pts = np.atleast_2d(pts)  # 保證至少是2維
        ret = np.dot(self.K, pts.T).T  # (n,3)
        ret /= ret[:, [2]]  # 每行除以自身的第三個元素
        return np.round(ret[:, :2]).astype(int)
    
    def denormalize_pt(self, pt):
        """
        pt: shape (3,) 的齊次2D座標 (X, Y, 1)
        回傳: (u, v) 像素座標
        """
        pt = np.asarray(pt, dtype=np.float32)
        pixel = self.K @ pt
        pixel /= pixel[2]
        return int(round(pixel[0])), int(round(pixel[1]))
        
    
    def undistorte(self,raw_kps):
        raw_kps = raw_kps.copy()
        # raw_kps = np.asarray(raw_kps, dtype=np.float32)
        
        raw_kps = raw_kps.reshape(-1, 1, 2) # undistortPoints 要求 shape 為 (N, 1, 2)
        un_kps = cv2.undistortPoints(raw_kps,self.K,self.D,P=self.K)
        return un_kps.reshape(-1, 2) # shape  (N, 1, 2) -> (N, 2)
        
    
    
    def in_view_angle(self, points2D):
        # 輸入 np.array
        #  points2D:必須是 shape (N,3) 的 numpy array，且是標準化座標。 N*(X,Y,1)
        in_range = (self.viewing_angle[0,0] <= points2D[:, 0]) & (points2D[:, 0] <= self.viewing_angle[1,0]) & \
                   (self.viewing_angle[0,1] <= points2D[:, 1]) & (points2D[:, 1] <= self.viewing_angle[1,1])
        
        
        return in_range