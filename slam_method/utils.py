import numpy as np
import cv2
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .frame import KeyFrame,Frame

class SlamState(Enum):
    
    SYSTEM_NOT_READY=-1
    NO_IMAGE = 0
    INITIALIZING = 1
    # NOT_INITIALIZED = 2
    WORKING = 2
    LOST = 3
    
    
class PointKeypointMap:
    def __init__(self):
        self.pt_to_kp:dict[int:int] = {}
        self.kp_to_pt:dict[int:int] = {}

    def bind(self, pt, kp):
        # 綁定pt和kp
        self.pt_to_kp[pt] = kp
        self.kp_to_pt[kp] = pt

    def getpt(self, kp)->int:
        # 根據kp取得pt
        return self.kp_to_pt.get(kp, None)

    def getkp(self, pt):
        # 根據pt取得kp
        return self.pt_to_kp.get(pt, None)

    def delete(self, key):
        # 刪除一組數據，無論是pt還是kp
        if key in self.pt_to_kp:
            # 如果 key 是pt
            key_int = self.pt_to_kp.pop(key)
            self.kp_to_pt.pop(key_int, None)
        elif key in self.kp_to_pt:
            # 如果 key 是kp
            key_str = self.kp_to_pt.pop(key)
            self.pt_to_kp.pop(key_str, None)



def add_ones(x):
    # concatenates the original array x with the column of ones along the second axis (columns). 
    # This converts the N×2 array to an N×3 array where each point is represented 
    # in homogeneous coordinates as [x,y,1].
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(v):
    """
    正規化給定的向量。
    參數:
        v (numpy.ndarray): 要正規化的輸入向量。
    回傳:
        tuple:
            - numpy.ndarray: 正規化後的向量（與輸入形狀相同）。
    如果向量的範數小於 1.e-10，回傳向量會較小避免錯誤。
    """
    norm = np.linalg.norm(v)
    if norm < 1.e-10: 
        return v / (norm + 1e-12)
    return v/norm

def normalize_vector(v):
    """
    正規化給定的向量。
    參數:
        v (numpy.ndarray): 要正規化的輸入向量。
    回傳:
        tuple:
            - numpy.ndarray: 正規化後的向量（與輸入形狀相同）。
            - float: 輸入向量的範數（長度）。
    如果向量的範數小於 1.e-10，則回傳原始向量及其範數。
    """
    norm = np.linalg.norm(v)
    if norm < 1.e-10: 
       return v / (norm + 1e-12), norm
    return v/norm, norm


def mask_block(mask,x1,x2,y1,y2):
    if mask is None:
        return None 
    else: 
        return mask[y1:y2, x1:x2]
# create a generator over an image to extract 'row_divs' x 'col_divs' sub-blocks 
def img_mask_blocks(img, mask, row_divs, col_divs):
    rows, cols = img.shape[:2]
    #print('img.shape: ', img.shape)
    xs = np.uint32(np.rint(np.linspace(0, cols, num=col_divs+1)))   # num = Number of samples to generate
    ys = np.uint32(np.rint(np.linspace(0, rows, num=row_divs+1)))
    #print('img_blocks xs: ', xs)
    #print('img_blocks ys: ', ys)
    ystarts, yends = ys[:-1], ys[1:]
    xstarts, xends = xs[:-1], xs[1:]
    for y1, y2 in zip(ystarts, yends):
        for x1, x2 in zip(xstarts, xends):
            yield img[y1:y2, x1:x2], mask_block(mask,x1,x2,y1,y2), y1, x1    # return block, row, col
            
def is_blurry(img,threshold=None):

    # 檢測圖片是否模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # variance = laplacian.var()
    
    # 計算影像的拉普拉斯取得邊緣，取得影像斜率變異數，變異數越小越模糊。
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if threshold == None:
        return variance
    else:
        is_blurry = variance<threshold   # 影像是否模糊
        
        return is_blurry,variance
    
    
def computeE21_pose(pose1:np.ndarray, pose2:np.ndarray=np.eye(4)) -> np.ndarray:
    """
    pose1 = Tw1 or T21
    pose2 = Tw2 or eye(4)
    
    def:
        E21 = t21^T × R21
        x2^T * E21 * x1 = 0 
    說明:
        × 是外積 T 是轉置。
        x1 x2 是 X(3D point) 在兩個投影平面的投影。
        t21 是從 相機1的座標系 到 相機2的座標系 的平移向量。
        R21 是從 相機1的座標系 到 相機2的座標系 的旋轉矩陣。
        E21 * x1 = t21^T × R21 * x1 = L21[a b c] 代表x1 在相機2上的線 對應直線方程 ax + by + c = 0
        x2^T * L21=0 代表 x2 在 L21 上
    """
    
    
    # T21 = T2w @ Tw1
    T21 = np.linalg.inv(pose2) @ pose1
    
    
    R21 = T21[:3,:3]
    t21 = T21[:3, 3]
    
    # 計算本質矩陣
    # def skew(t):
    #     return np.array([
    #         [0, -t[2], t[1]],
    #         [t[2], 0, -t[0]],
    #         [-t[1], t[0], 0]
    #     ])
    # t12x = skew(t12.flatten()) # 反對稱矩陣 用於表示外積操作
    # E12 = t12x @ R12
    E21 = np.cross(t21.reshape(-1, 1), R21, axis=0)
    
    return E21
    
def computeE21(kf1:'KeyFrame', kf2:'KeyFrame') -> np.ndarray:
    """
    def:
        E21 = t21^T × R21
        x2^T * E21 * x1 = 0 
    說明:
        × 是外積 T 是轉置。
        x1 x2 是 X(3D point) 在兩個投影平面的投影。
        t21 是從 相機1的座標系 到 相機2的座標系 的平移向量。
        R21 是從 相機1的座標系 到 相機2的座標系 的旋轉矩陣。
        E21 * x1 = t21^T × R21 * x1 = L21[a b c] 代表x1 在相機2上的線 對應直線方程 ax + by + c = 0
        x2^T * L21=0 代表 x2 在 L21 上
    """
    
    Tw1 = kf1.Twc
    T2w = kf2.Tcw
    T21 = T2w @ Tw1
    R21 = T21[:3,:3]
    t21 = T21[:3, 3]
    
    # 計算本質矩陣
    # def skew(t):
    #     return np.array([
    #         [0, -t[2], t[1]],
    #         [t[2], 0, -t[0]],
    #         [-t[1], t[0], 0]
    #     ])
    # t12x = skew(t12.flatten()) # 反對稱矩陣 用於表示外積操作
    # E12 = t12x @ R12
    E21 = np.cross(t21.reshape(-1, 1), R21, axis=0)
    
    return E21

def compute_epipolar_error(kp1, kp2, E21):
    """
    計算單一對特徵點的極約束誤差。
    Args:
        kp1: 第1個影像中的特徵點 (x1, y1)。
        kp2: 第2個影像中的特徵點 (x2, y2)。
        E21: img1 到 img2 的本質矩陣。
    Returns:
        error: float, 該特徵點對的極約束誤差，點到極線距離平方。
    """
    # 將特徵點轉換為齊次座標
    x1_h = np.array([kp1[0], kp1[1], 1.0])  # shape: (3,)
    x2_h = np.array([kp2[0], kp2[1], 1.0])  # shape: (3,)

    # 計算極線參數 L21 = E21 * x1
    L21 = E21 @ x1_h  # shape: (3,)

    # 提取極線參數 (a, b, c)
    a, b, c = L21

    # 計算特徵點到極線的代數距離
    num = a * x2_h[0] + b * x2_h[1] + c
    den = a * a + b * b

    # 計算距離平方
    error = (num * num) / den
    return error

def compute_epipolar_errors(kps1, kps2, E21):
    """
    計算極約束誤差。
    Args:
        kps1: shape(N,2)第1個影像中的特徵點列表 [(x1, y1), ...]。
        kps2: shape(N,2)第2個影像中的特徵點列表 [(x2, y2), ...]。
        E21: img1 到 img2 的本質矩陣。
    Returns:
        errors: np.ndarray shape(N,), 每個特徵點對的極約束誤差，點到極線距離平方。
    """
    # 將特徵點轉換為齊次座標
    kps1_h = add_ones(np.array(kps1))  # shape: (N, 3)
    kps2_h = add_ones(np.array(kps2))  # shape: (N, 3)

    # 計算極線參數 L21 = E21 * x1
    lines = (E21 @ kps1_h.T).T  # shape: (N, 3)
    
    # 計算每個特徵點對的極約束誤差
    # 提取極線參數 (a, b, c)
    a = lines[:, 0] # shape: (N,)
    b = lines[:, 1]
    c = lines[:, 2]
    
    # 計算距離平方
    # (ax+by+c)^2
    # ———————————
    #   a^2+B^2
    
    num = a * kps2_h[:, 0] + b * kps2_h[:, 1] + c # 計算每個特徵點到極線的代數距離
    den = a * a + b * b
    errors = (num * num) / den
    
    return errors


def check_epipolar(f:'Frame',kp_idx:int,epipolar_err:float, th:float=3.84)->bool:
    sigma2 = f.get_sigma2(f.octaves[kp_idx])
    # print(sigma2 , kf2.camera.pixel_to_meter)
    epipolar_check = epipolar_err < (th * sigma2 * f.camera.pixel_to_meter)
    # print('epipolar_check',epipolar_check,epipolar_err , (3.84 * sigma2 * kf2.camera.pixel_to_meter))
    return epipolar_check



def show_epilines(img1, img2, kps1, kps2, E, K):
    """
    使用 OpenCV 在兩幅圖像中顯示對極線和對應的特徵點。
    Args:
        img1: 第一幅圖像。
        img2: 第二幅圖像。
        kps1: 第一幅圖像中的特徵點。
        kps2: 第二幅圖像中的特徵點。
        E: 本質矩陣。
    """
    K1_inv = np.linalg.inv(K)
    K2_inv_T = np.linalg.inv(K).T
    F = K2_inv_T @ E @ K1_inv
    
    # 計算對極線
    lines1 = cv2.computeCorrespondEpilines(kps2.reshape(-1, 1, 2), 2, F)  # 對應 img1 的對極線
    lines2 = cv2.computeCorrespondEpilines(kps1.reshape(-1, 1, 2), 1, F)  # 對應 img2 的對極線
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)

    # 繪製對極線和特徵點
    img1_with_lines = draw_epilines(img1, lines1, kps1)
    img2_with_lines = draw_epilines(img2, lines2, kps2)

    # 顯示結果
    cv2.imshow("Epilines and Points in Image 1", img1_with_lines)
    cv2.imshow("Epilines and Points in Image 2", img2_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_epilines(img, lines, points):
    """
    在圖像中繪製對極線和特徵點。
    Args:
        img: 圖像。
        lines: 對極線參數 (ax + by + c = 0)。
        points: 特徵點。
    Returns:
        帶有對極線和特徵點的圖像。
    """
    img_with_lines = img.copy()
    h, w = img.shape[:2]
    for r, pt in zip(lines, points):
        color = tuple(np.random.randint(0, 255, 3).tolist())  # 隨機顏色
        x0, y0 = map(int, [0, -r[2] / r[1]])  # 對極線的起點
        x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])  # 對極線的終點
        cv2.line(img_with_lines, (x0, y0), (x1, y1), color, 1)  # 繪製對極線
        cv2.circle(img_with_lines, tuple(pt.astype(int)), 5, color, 2)  # 繪製特徵點
    return img_with_lines


