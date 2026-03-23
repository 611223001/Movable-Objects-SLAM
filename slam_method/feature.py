import cv2
import numpy as np
from collections import defaultdict


from .utils import img_mask_blocks
from .pose import computeF12


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .frame import Frame,KeyFrame
    from .point import Point
    from .objects import Thing,Stuff







class FeatureTool():
    def __init__(self,config=None):
        
        self.detector = None # keypoints detector
        self.computer = None # descriptors computer
        self.matcher = None
        
        

        
        self.init_orb_params = dict(
            nfeatures=1000,          # 最大特徵點數量，控制檢測到的特徵點數目，數值越大，檢測的特徵點越多。
            scaleFactor=1.2,         # 金字塔縮放比例，每層金字塔的影像大小縮放比例，1.2 表示每層影像縮小 20%。
            nlevels=8,               # 金字塔層數，控制金字塔的層數，層數越多，對多尺度特徵的檢測能力越強。
            patchSize=31,            # 特徵點鄰域的大小，用於計算描述子的區域大小，單位為像素。
            edgeThreshold=10,        # 邊界閾值，特徵點距離影像邊界的最小距離，避免檢測到邊界上的特徵點。
            firstLevel=0,            # 金字塔的第一層索引，通常為 0，表示從原始影像開始。
            WTA_K=2,                 # 每次比較的像素數量，2 表示使用對偶比較，4 表示使用四元比較。
            scoreType=cv2.ORB_FAST_SCORE,
                                     # 特徵點排序的方式，`cv2.ORB_FAST_SCORE` 表示使用 FAST 分數排序，
                                     # `cv2.ORB_HARRIS_SCORE` 表示使用 Harris 分數排序。
            fastThreshold=20,        # FAST 特徵檢測的閾值，數值越高，檢測到的特徵點越少，但更穩定。
                                    )
        
        self.orb_params = self.init_orb_params
        
        self.orb:cv2.ORB = cv2.ORB_create(**self.init_orb_params)
        
        self.detector = self.orb
        self.computer = self.orb
        
        # self.detect:function = self.block_adaptor_detect
        self.detect:function = self.detector.detect
        
        self.compute:function = self.orb_compute
        
        self.lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    def set_orb_params(self,**kwargs):
        # 設置orb參數，沒設置的參數會使用預設值，不會保存上次過去參數
        self.orb_params =dict(self.init_orb_params)
        self.orb_params.update(kwargs)
        print(self.orb_params)
        self.orb = cv2.ORB_create(**self.orb_params)
        self.detector = self.orb
        self.computer = self.orb
        
        self.detect:function = self.detector.detect
        self.compute:function = self.orb_compute
    
    def shi_tomasi_detect(self,img)->cv2.KeyPoint:
        #Shi-Tomasi 角點偵測器
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(gray_img, 8000, qualityLevel=0.01, minDistance=10)
        
        if pts is not None: 
            kps = [cv2.KeyPoint(p[0][0], p[0][1], 20) for p in pts ]
        else:
            kps = []
        return kps
    
    def orb_compute(self,img,kps)->tuple[tuple[cv2.KeyPoint],np.ndarray]:
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kps, des = self.orb.compute(gray_img, kps)
        
        return kps, des 
    
    def block_adaptor_detect(self, img, mask=None)->np.ndarray[cv2.KeyPoint]:
        block_generator = img_mask_blocks(img, mask, row_divs=5, col_divs=5)
        kps_all = []
        
        def detect_block(b, m, i, j):                         
            kps = self.detector.detect(b, mask=m)
            #print('adaptor: detected #features: ', len(kps), ' in block (',i,',',j,')')  
            for kp in kps:
                #print('kp.pt before: ', kp.pt)
                kp.pt = (kp.pt[0] + j, kp.pt[1] + i)        
                #print('kp.pt after: ', kp.pt)                                                                     
            kps_all.extend(kps)
        
        for b, m, i, j in block_generator:
            detect_block(b,m,i,j)
        
        return np.array(kps_all)
    
    def mask_adaptor_detect(self, img ,masks):
        # 根據分割遮罩挑選kps
        # 效果不佳
        
        kps_all = []
        for mask in masks:
            kps = self.detector.detect(img, mask=mask.astype(np.uint8))
            kps_all.extend(kps)
        return np.array(kps_all)
            
            
    
    def OF_feature_tracking(self,image_ref, image_cur, px_ref):
        # Optical Flow feature tracking
        # 使用金字塔光流法追蹤特徵點
        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]

        # 過濾掉追蹤失敗的特徵點
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]

        return kp1, kp2
    
    def descriptor_distance(self, a, b):
        if self.computer == self.orb or True:
            # norm_type == cv2.NORM_HAMMING
            return np.count_nonzero(a!=b)
        else: print('descriptor_distance error')
        
    def descriptor_distances(self, a, b):
        if self.computer == self.orb or True:
            # norm_type == cv2.NORM_HAMMING
            return np.count_nonzero(a!=b,axis=1)
        else: print('descriptor_distance error')

TH_HIGH = 100
TH_LOW = 50
HISTO_LENGTH = 30



'''
def window_search(f_ref:'Frame', f_cur:'Frame', window_size=30, min_scale_level=0, max_scale_level=np.inf, nn_ratio=0.8, distance_threshold=50, histo_length=30, check_orientation=True):
    """
    WindowSearch，於 f_ref 的 MapPoint 在 f_cur window 內搜尋最佳匹配。
    Args:
        f_ref: 參考幀 (Frame)
        f_cur: 目標幀 (Frame)
        window_size: 搜尋窗口大小 (像素)
        min_scale_level: 最小金字塔層級
        max_scale_level: 最大金字塔層級
        nn_ratio: 最近鄰比值
        distance_threshold: Hamming 距離閾值
        histo_length: 旋轉直方圖 bin 數
        check_orientation: 是否進行旋轉一致性檢查
    Returns:
        matches: list, (ref_idx, cur_idx)
    """
    nmatches = 0
    N_cur = len(f_cur.kps)
    vp_map_point_matches2 = [None] * N_cur
    vn_matches21 = [-1] * N_cur
    rot_hist = [[] for _ in range(histo_length)]
    factor = 1.0 / histo_length

    for i1, point in enumerate(f_ref.points):
        if point is None:
            continue
        
        point:'Point'
        # 若有 isBad 屬性可加上判斷
        if point.is_bad:
            continue

        # 取得參考幀的特徵資訊
        pt1 = f_ref.raw_kps[i1]  # (x, y) 已消除扭曲
        level1 = f_ref.octaves[i1]
        angle1 = f_ref.angles[i1]

        if min_scale_level > 0 and level1 < min_scale_level:
            continue
        if max_scale_level < np.inf and level1 > max_scale_level:
            continue

        # 只搜尋同層級且在 window 內的 keypoint
        indices2 = [i2 for i2 in range(N_cur)
                    if np.linalg.norm(f_cur.raw_kps[i2] - pt1) < window_size
                    and f_cur.octaves[i2] == level1
                    and vp_map_point_matches2[i2] is None]

        if not indices2:
            continue

        d1 = f_ref.des[i1]
        best_dist = float('inf')
        best_dist2 = float('inf')
        best_idx2 = -1

        for i2 in indices2:
            d2 = f_cur.des[i2]
            dist = np.count_nonzero(d1 != d2)
            if dist < best_dist:
                best_dist2 = best_dist
                best_dist = dist
                best_idx2 = i2
            elif dist < best_dist2:
                best_dist2 = dist

        if best_dist <= best_dist2 * nn_ratio and best_dist <= distance_threshold:
            vp_map_point_matches2[best_idx2] = point
            vn_matches21[best_idx2] = i1
            nmatches += 1

            rot = angle1 - f_cur.angles[best_idx2]
            if rot < 0.0:
                rot += 360.0
            bin_idx = int(round(rot * factor)) % histo_length
            rot_hist[bin_idx].append(best_idx2)

    # 旋轉一致性檢查
    if check_orientation:
        hist_sizes = [len(h) for h in rot_hist]
        inds = np.argsort(hist_sizes)[-3:]  # 取最大三個 bin
        for i in range(histo_length):
            if i not in inds:
                for idx in rot_hist[i]:
                    vp_map_point_matches2[idx] = None
                    vn_matches21[idx] = -1
                    nmatches -= 1

    # 回傳 (ref_idx, cur_idx) 配對
    matches = [(vn_matches21[i2], i2) for i2, mp in enumerate(vp_map_point_matches2) if mp is not None]
    return matches
'''
'''
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

def search_for_triangulation(kf1:'KeyFrame', kf2:'KeyFrame', distance_threshold=50, histo_length=30, check_orientation=True):
    """
    配對兩個 KeyFrame 的特徵點，需滿足描述子與極線約束。
    TODO:加入BoW
    回傳: kp_idxs1, kp_idxs2, match_pairs
    """
    
    F12 = computeF12(kf1,kf2)
    
    N1 = len(kf1.kps)
    N2 = len(kf2.kps)
    matches12 = [-1] * N1
    matched2 = [False] * N2
    rot_hist = [[] for _ in range(histo_length)]
    factor = 1.0 / histo_length

    nmatches = 0
    
    for idx1 in range(N1):
        # 只考慮沒有地圖點的特徵點
        if kf1.points[idx1] is not None:
            continue
        d1 = kf1.des[idx1]
        un_kp1 = kf1.un_kps[idx1]
        angle1 = kf1.angles[idx1]

        best_dist = float('inf')
        best_idx2 = -1
        
        for idx2 in range(N2):
            if matched2[idx2]:
                continue
            if kf2.points[idx2] is not None:
                continue
            d2 = kf2.des[idx2]
            dist = np.count_nonzero(d1 != d2)
            if dist > distance_threshold:
                continue
            oct2 = kf2.octaves[idx2]
            un_kp2 = kf2.un_kps[idx2]
            # 極線約束
            if not check_dist_epipolar_line(un_kp1,un_kp2,F12,oct2,kf2):
                continue
            


            if dist < best_dist:
                best_dist = dist
                best_idx2 = idx2

        if best_idx2 >= 0:
            matches12[idx1] = best_idx2
            matched2[best_idx2] = True
            nmatches += 1

            if check_orientation:
                angle2 = kf2.angles[best_idx2]
                rot = angle1 - angle2
                if rot < 0.0:
                    rot += 360.0
                bin_idx = int(round(rot * factor)) % histo_length
                rot_hist[bin_idx].append(idx1)

    # 旋轉一致性過濾
    if check_orientation:
        hist_max = sorted(range(histo_length), key=lambda i: len(rot_hist[i]), reverse=True)[:3]
        for i in range(histo_length):
            if i not in hist_max:
                for idx1 in rot_hist[i]:
                    matches12[idx1] = -1
                    nmatches -= 1

    # 整理配對結果
    kp_idxs1 = []
    kp_idxs2 = []
    match_pairs = []
    for idx1, idx2 in enumerate(matches12):
        if idx2 >= 0:
            kp_idxs1.append(idx1)
            kp_idxs2.append(idx2)
            match_pairs.append((idx1, idx2))

    return np.array(kp_idxs1), np.array(kp_idxs2), match_pairs
'''
'''
def flannMatches(f1:'Frame', f2:'Frame',ratio_threshold=0.7):
    
    flann = cv2.FlannBasedMatcher(indexParams=dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), 
                                  searchParams=dict(checks=50))
    try:
        matches = flann.knnMatch(f1.des, f2.des, k=2)
    except:
        matches = []
    
    good_matches = []
    ret = []
    idxs1, idxs2 = [], []

    distance_threshold=50.0
    try:
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance and m.distance < distance_threshold:
                good_matches.append(m)
                idxs1.append(m.queryIdx)
                idxs2.append(m.trainIdx)
                ret.append((f1.kps[m.queryIdx], f2.kps[m.trainIdx]))
                
    except ValueError:
        pass
    
    # 如果匹配點數量足夠(>=4)，則使用 RANSAC 計算單應性矩陣（Homography Matrix）。
    # 使用 RANSAC 的遮罩（mask）過濾掉外點，保留內點作為最終的匹配結果。
    if  len(good_matches) < 4:
        return np.array(idxs1), np.array(idxs2), np.array(ret)
    
    # if type(f1.kps[0]) == cv2.KeyPoint:
    #     src_pts = np.float32([f1.kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([f2.kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # else:
    
    
    # src_pts = np.float32([f1.kps[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32([f2.kps[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)


    ## Homography 只適合所有點都在一個平面上的情況
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # # Use the mask to select inlier matches
    # good_matches = [m for m, msk in zip(good_matches, mask) if msk[0] == 1]
    
    # idxs1 = [idx1 for idx1, msk in zip(idxs1, mask) if msk[0] == 1]
    # idxs2 = [idx2 for idx2, msk in zip(idxs2, mask) if msk[0] == 1]
    # ret = [p1_p2 for p1_p2, msk in zip(ret, mask) if msk[0] == 1]
    return np.array(idxs1), np.array(idxs2), np.array(ret)

'''


###### 以下函式未被使用 ######




    # print("matched points",len(ret))





# def extract(img)->list[cv2.KeyPoint]:
#     orb = cv2.ORB_create()
    
#     # Convert to grayscale
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detection
#     kps = cv2.goodFeaturesToTrack(gray_img, 8000, qualityLevel=0.01, minDistance=10)

#     if kps is None:
#         return np.array([]), None

#     # Extraction
#     kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in kps]
#     kps, des = orb.compute(gray_img, kps)

#     return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des



'''class Matcher():
    def __init__(self,matcher='BF_knn'):
        

        match matcher:
            case 'BF_knn':
                self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
                self.match = self.bf_knn_matcher
            case _:
                self.match = None
    
    def bf_knn_matcher(self,f1:'Frame', f2:'Frame',k=2):
        matches = self.matcher.knnMatch(f1.des, f2.des,k=2)
        # Lowe's ratio test
        ret = []
        idx1, idx2 = [], []
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                p1 = f1.pts[m.queryIdx]
                p2 = f2.pts[m.trainIdx]
                
                # Distance test
                # dditional distance test, ensuring that the 
                # Euclidean distance between p1 and p2 is less than 0.1
                if np.linalg.norm((p1-p2)) < 0.1:
                    # Keep idxs
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    ret.append((p1, p2))
                    pass
                
        return idx1, idx2, ret'''




