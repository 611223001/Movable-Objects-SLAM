import cv2
import numpy as np
from collections import defaultdict

from .pose import computeF12,check_dist_epipolar_line


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .frame import Frame, KeyFrame
    from .point import Point
    from .objects import Thing, Stuff



## Keypoint ↔ Keypoint （純特徵點匹配）
class KeypointMatcher:
    """
    特徵點與特徵點之間的批配
    
    包含：FLANN、視窗搜尋、初始化、光流、三角化等純描述子匹配
    """
    
    @staticmethod
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
        
        return np.array(idxs1), np.array(idxs2), np.array(ret)

    @staticmethod
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
        
    @staticmethod
    def search_kps_by_flow(frame:'Frame', f_last:'Frame', flow, radius_factor:float=1.0, nn_ratio:float=0.8, distance_threshold:int=100)->int:
        """
        以光流方式在 frame 上搜尋 kps 的最佳匹配。
        Args:
            frame: 目標 Frame
            f_last: 上個 frame
            flow: frame 間光流
            radius_factor: 搜尋半徑縮放因子
            nn_ratio: 最近鄰比值
            distance_threshold: Hamming 距離閾值
        Returns:
            nmatches: 匹配數量
            matches: list
        """
        nmatches = 0
        matches = []
        for idx2, point in enumerate(f_last.points):
            # if point is None or point.is_bad:
            #     continue
            
            # 取得光流對應位置
            u, v = f_last.raw_kps[idx2]
            uu,vv = flow[v,u]
            uv = (u+uu, v+vv)
            
            
            # 根據點的尺度調整搜尋半徑
            scale_level = int(f_last.octaves[idx2])
            base_radius = f_last.sizes[idx2]
            radius = radius_factor * base_radius
            
            # KDTree搜尋半徑內的keypoint
            near_indices = frame.raw_kd.query_ball_point(uv, radius)
            
            
            # 只考慮octave在合理範圍的keypoint
            near_indices = [idx1 for idx1 in near_indices
                            if (scale_level >= frame.octaves[idx1] and frame.octaves[idx1] >= scale_level-1)]
            
            if not near_indices:
                continue
            
            # 找最佳描述子匹配
            best_dist = float('inf')
            best_dist2 = float('inf')
            best_idx = -1
            
            des2 = f_last.des[idx2]
            for idx1 in near_indices:
                des1 = frame.des[idx1]
                dist = descriptor_distance(des2, des1)
                if dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = dist
                    best_idx = idx1
                elif dist < best_dist2:
                    best_dist2 = dist
            
            # NN ratio 與距離檢查
            if best_dist <= distance_threshold:
                if best_dist > nn_ratio * best_dist2:
                    continue
                
                matches.append((best_idx,idx2))
                nmatches += 1
                

        return nmatches, matches

    @ staticmethod
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

    @ staticmethod
    def goodMatchesOneToOne(f1:'Frame', f2:'Frame', ratio_threshold=0.7):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches = bf.knnMatch(f1.des, f2.des, k=2)
        
        ret = []
        idxs1, idxs2 = [], []
        if matches is not None:
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)   
            index_match = dict()  
            for m, n in matches:
                if m.distance > ratio_threshold * n.distance:
                    continue
                dist = dist_match[m.trainIdx]
                if dist == float_inf: 
                    # trainIdx has not been matched yet
                    dist_match[m.trainIdx] = m.distance
                    idxs1.append(m.queryIdx)
                    idxs2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idxs2)-1
                else:
                    if m.distance < dist: 
                        # we have already a match for trainIdx: if stored match is worse => replace it
                        #print("double match on trainIdx: ", m.trainIdx)
                        index = index_match[m.trainIdx]
                        assert(idxs2[index] == m.trainIdx) 
                        idxs1[index]=m.queryIdx
                        idxs2[index]=m.trainIdx
            
            
            for idx1,idx2 in zip(idxs1,idxs2):
                ret.append((f1.kps[idx1],f2.kps[idx2]))
            
        
        return np.array(idxs1), np.array(idxs2), np.array(ret)


    @ staticmethod
    def match_frames(f1:'Frame', f2:'Frame'):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches = bf.knnMatch(f1.des, f2.des, k=2)

        # Lowe's ratio test
        ret = []
        idx1, idx2 = [], []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                p1 = f1.kps[m.queryIdx]
                p2 = f2.kps[m.trainIdx]
                
                # Distance test
                # dditional distance test, ensuring that the 
                # Euclidean distance between p1 and p2 is less than 0.1
                if np.linalg.norm((p1-p2)) < 0.1:
                    # Keep idxs
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    ret.append((p1, p2))
                    
        ret = np.array(ret) # [[x1,y1], [x2,y2]]
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)

        return idx1, idx2, ret



## MapPoint ↔ Keypoint （地圖點與特徵點匹配）
class MapPointMatcher:
    """
    地圖點投影與特徵點的批配
    """
    @staticmethod
    def search_by_projection(frame:'Frame', points:list['Point|None'], radius_factor:float=1.0, nn_ratio:float=0.8, distance_threshold:int=100,add_point=True, check_dynamic=False)->int:
        """
        以投影方式在 frame 上搜尋 points 的最佳匹配。
        Args:
            frame: 目標 Frame
            points: 要投影搜尋的地圖點列表
            radius_factor: 搜尋半徑縮放因子
            nn_ratio: 最近鄰比值
            distance_threshold: Hamming 距離閾值
        Returns:
            nmatches: 匹配數量
        """
        nmatches = 0
        matches = []
        scale_factors = frame.scale_factors
        n_max_level = len(scale_factors)-1
        
        if check_dynamic:
            # 紀錄動態的物件
            dynamic_objs = set()
            # frame 判斷為動態的物件
            for obj, dynamic in zip(frame.objects,frame.objects_dynamic):
                if dynamic is not False:
                    dynamic_objs.add(obj)
            # 地圖紀錄的動態物件
            for obj,dynamic in frame.map.dynamic_objcts.items():
                if dynamic is not False:
                    dynamic_objs.add(obj)
                
        
        for point in points:
            if point is None or point.is_bad:
                continue
            
            if check_dynamic:
                if point.object in dynamic_objs:
                    continue
                

            # 投影到影像
            # visible, proj = frame.is_visible(point)
            # proj_xy = proj[:2]
            uv, visible = frame.project_point_to_img(point.location)
            if not visible:
                continue

            # # 根據點的尺度調整搜尋半徑（可用平均size或固定值）
            # scale_level = np.median(frame.octaves) # if len(frame.octaves) > 0 else 0
            # base_radius = np.median(frame.sizes) # if hasattr(frame, "sizes") and len(frame.sizes) > 0 else 10.0
            # radius = base_radius * radius_factor
            
            # 深度與尺度一致性檢查
            min_dist,max_dist = point.update_depth()
            dist3D = np.linalg.norm(point.location_3D - frame.position)
            if not (min_dist < dist3D < max_dist):
                continue
            # 預測金字塔層級
            ratio = dist3D / (min_dist + 1e-6)# if min_dist > 0 else 1.0
            pred_octave = min(np.searchsorted(scale_factors, ratio), n_max_level)
            # 搜尋半徑根據尺度調整
            radius = radius_factor * scale_factors[pred_octave]
            
            # print('radius',radius,n_pred_level,ratio,dist3D,min_dist)
            
            
            # KDTree搜尋半徑內的keypoint
            near_indices = frame.raw_kd.query_ball_point(uv, radius)
            
            # # 只考慮未配對的keypoint
            # near_indices = [idx for idx in near_indices if frame.points[idx] is None]
            
            # 只考慮未配對且octave在合理範圍的keypoint
            near_indices = [idx for idx in near_indices
                            if frame.points[idx] is None and
                            (frame.octaves[idx] >= pred_octave-1 and frame.octaves[idx] <= pred_octave)]

            if not near_indices:
                continue

            # 找最佳描述子匹配
            best_dist = float('inf')
            best_dist2 = float('inf')
            best_idx = -1
            for idx in near_indices:
                d = frame.des[idx]
                dist = descriptor_distance(point.des, d)
                if dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = dist
                    best_idx = idx
                elif dist < best_dist2:
                    best_dist2 = dist

            # NN ratio 與距離檢查
            if best_dist <= distance_threshold:
                if best_dist > nn_ratio * best_dist2:
                    continue
                if add_point:
                    frame.add_point(point, best_idx)
                    frame.set_track_method(best_idx,2)
                    
                matches.append((best_idx,point))
                nmatches += 1
                
        
        return nmatches
    
    @staticmethod
    def search_by_projection_frame(f_cur:'Frame', f_last:'Frame', th: float, nn_ratio: float = 0.8, check_ori: bool = True) -> int:
        """
        - 使用 f_cur.is_visible / KDTree / descriptors 做匹配
        """
        HISTO_LENGTH = 30
        rotHist = [ [] for _ in range(HISTO_LENGTH) ]
        factor = HISTO_LENGTH / 360.0
        TH_HIGH = 50  # Hamming distance threshold

        nmatches = 0

        cam = f_cur.camera
        fx = cam.fx
        # fx = cam.fx if hasattr(cam, "fx") else 1.0
        # fy = cam.fy if hasattr(cam, "fy") else fx
        # cx = cam.cx if hasattr(cam, "cx") else 0.0
        # cy = cam.cy if hasattr(cam, "cy") else 0.0

        img_h, img_w = f_cur.img.shape[:2]

        # 支援 LastFrame 為 Frame 或 points list



        for i, point in enumerate(f_last.points):
            if point is None:
                continue
            point: Point

            # 若來源是 Frame 且有 outliers 標記，跳過
            if f_last.outliers[i]:
                continue

            # 使用 Frame 的可見性檢查（包含深度 / 視域 / 法向等）
            visible, proj = f_cur.is_visible(point)
            if not visible:
                continue
            
            pixel = cam.denormalize(proj)
            
            u=pixel[0][0]
            v=pixel[0][1]
            xn = float(proj[0]); yn = float(proj[1])

            # 轉為像素座標並檢查影像範圍
            # u = fx * xn + cx
            # v = fy * yn + cy
            if u < 0 or u >= img_w or v < 0 or v >= img_h:
                continue
            
            

            # 預測 octave（優先使用 LastFrame 的資訊，否則嘗試 point 屬性或 0）
            nPredictedOctave = int(f_last.octaves[i])

            # 搜尋半徑（像素），以 CurrentFrame.sizes 為基準
            
            base_radius = float(f_cur.sizes[nPredictedOctave])
            
            radius_pixels = th * base_radius

            # 將像素半徑轉為 normalized（KDTree 使用 normalized coords）
            radius_norm = radius_pixels / max(fx, 1e-6)

            # 在 KDTree 查詢鄰近 keypoints（normalized proj）
            candidate_idxs = f_cur.kd.query_ball_point([xn, yn], radius_norm)

            if not candidate_idxs:
                continue

            # 篩選候選：未被配對，且 octave 差異 <= 1（若有 octave 資料）
            cand_filtered = []
            for idx in candidate_idxs:
                if idx >= len(f_cur.points):
                    continue
                if f_cur.points[idx] is not None:
                    continue
                if len(f_cur.octaves) > idx:
                    if abs(int(f_cur.octaves[idx]) - nPredictedOctave) > 1:
                        continue
                cand_filtered.append(idx)

            if not cand_filtered:
                continue

            # MapPoint 描述子
            dMP = point.des
            if dMP is None:
                continue

            # 找最佳與次佳
            bestDist = float("inf")
            bestDist2 = float("inf")
            bestIdx = -1
            for idx in cand_filtered:
                d = f_cur.des[idx]
                dist = descriptor_distance(dMP, d)
                if dist < bestDist:
                    bestDist2 = bestDist
                    bestDist = dist
                    bestIdx = idx
                elif dist < bestDist2:
                    bestDist2 = dist

            if bestIdx < 0 or bestDist > TH_HIGH:
                continue

            # NN-ratio 檢查
            if bestDist2 < float("inf") and bestDist > nn_ratio * bestDist2:
                continue

            # 建立關聯
            f_cur.add_point(point, bestIdx)
            nmatches += 1

            # 記錄角度直方圖（last frame 的角度）
            if check_ori:
                rot = f_last.angles[i] - f_cur.angles[bestIdx]
                if rot < 0.0:
                    rot += 360.0
                bin_idx = int(round(rot * factor))
                if bin_idx == HISTO_LENGTH:
                    bin_idx = 0
                if 0 <= bin_idx < HISTO_LENGTH:
                    rotHist[bin_idx].append(bestIdx)

        # 應用旋轉一致性（保留前三大 bins）
        if check_ori:
            counts = [len(b) for b in rotHist]
            if sum(counts) > 0:
                inds = np.argsort(counts)[-3:]
                keep = set()
                for ii in inds:
                    keep.update(rotHist[ii])
                for kp_idx, pt in enumerate(list(f_cur.points)):
                    if pt is None:
                        continue
                    if kp_idx not in keep:
                        f_cur.remove_point(f_cur.points[kp_idx])
                        nmatches -= 1

        return nmatches
    
    @staticmethod
    def search_by_flow(frame:'Frame', f_last:'Frame', flow, radius_factor:float=1.0, nn_ratio:float=0.8, distance_threshold:int=100,add_point=True)->int:
        """
        以光流方式在 frame 上搜尋 points 的最佳匹配。
        Args:
            frame: 目標 Frame
            f_last: 必須包含地圖點
            radius_factor: 搜尋半徑縮放因子
            nn_ratio: 最近鄰比值
            distance_threshold: Hamming 距離閾值
        Returns:
            nmatches: 匹配數量
            matches: list
        """
        nmatches = 0
        matches = []
        for idx2, point in enumerate(f_last.points):
            point:'Point'
            if point is None or point.is_bad:
                continue
            
            # 取得光流對應位置
            u, v = f_last.raw_kps[idx2]
            uu,vv = flow[v,u]
            uv = (u+uu, v+vv)
            
            
            # 根據點的尺度調整搜尋半徑
            scale_level = int(f_last.octaves[idx2])
            base_radius = f_last.sizes[idx2]
            radius = radius_factor * base_radius
            
            # KDTree搜尋半徑內的keypoint
            near_indices = frame.raw_kd.query_ball_point(uv, radius)
            
            # # 只考慮未配對的keypoint
            # near_indices = [idx for idx in near_indices if frame.points[idx] is None]
            
            # 只考慮未配對且octave在合理範圍的keypoint
            near_indices = [idx1 for idx1 in near_indices
                            if frame.points[idx1] is None and
                            (scale_level >= frame.octaves[idx1] and frame.octaves[idx1] >= scale_level-1)]
            
            if not near_indices:
                continue
            
            # 找最佳描述子匹配
            best_dist = float('inf')
            best_dist2 = float('inf')
            best_idx = -1
            for idx1 in near_indices:
                des = frame.des[idx1]
                dist = descriptor_distance(point.des, des)
                if dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = dist
                    best_idx = idx1
                elif dist < best_dist2:
                    best_dist2 = dist
            
            # NN ratio 與距離檢查
            if best_dist <= distance_threshold:
                if best_dist > nn_ratio * best_dist2:
                    continue
                if add_point:
                    frame.add_point(point, best_idx)
                    frame.set_track_method(best_idx,3)
                
                matches.append((best_idx,idx2))
                nmatches += 1
                

        return nmatches, matches
    
    
    
    


## Object ↔ Keypoint
class ObjectPointMatcher:
    """
    物體內地圖點與特徵點的批配
    
    包含：物體區域內點的一對一匹配
    """
    
    @staticmethod
    def match_points_in_object(obj:'Thing|Stuff', 
                            kf:'KeyFrame',
                            nn_ratio:float = 0.8,
                            descriptor_dist_thr:float = 50.0) -> dict['Point', int]:
        """
        將物體的地圖點與關鍵幀遮罩內的特徵點進行一對一匹配
        
        使用貪心策略：優先保留描述子距離最小的匹配，保證一對一對應
        
        Returns:
            dict[Point, int]: 匹配成功的 {地圖點: 關鍵幀特徵點索引}
        """
        
        matches = {}
        
        # 1. 獲取遮罩內未匹配的特徵點
        if kf not in obj.keyframes:
            return matches
        
        obj_idxs = obj.keyframes[kf]
        mask_kp_idxs = set()
        for obj_idx in obj_idxs:
            kp_idxs = kf.obj_to_pts[obj_idx]
            mask_kp_idxs.update(kp_idxs)
        
        unmatched_kp_idxs = [idx for idx in mask_kp_idxs if kf.points[idx] is None]
        
        if len(unmatched_kp_idxs) < 4:
            return matches
        
        # 2. 準備描述子
        valid_points = [pt for pt in obj.points 
                        if not pt.is_bad 
                        and pt in obj.static_points_relative
                        and pt.des is not None]
        
        if len(valid_points) == 0:
            return matches
        
        # 驗證描述子形狀一致性
        expected_shape = valid_points[0].des.shape
        valid_points = [pt for pt in valid_points if pt.des.shape == expected_shape]
        
        if len(valid_points) == 0:
            return matches
        
        points_des = np.array([pt.des for pt in valid_points], dtype=np.uint8)
        kf_des = kf.des[unmatched_kp_idxs]
        
        # 3. BFMatcher 找 k=2 最近鄰
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(points_des, kf_des, k=2)
        
        
        # 4. 一對一匹配（goodMatchesOneToOne 邏輯）
        kp_best_dist = defaultdict(lambda: float('inf'))  # {kp_idx: 最佳距離}
        kp_best_pt = {}  # {kp_idx: 對應的 Point}
        
        
        for matches_pair in raw_matches:
            if len(matches_pair) == 0:
                continue
            
            m = matches_pair[0]
            
            # Lowe's ratio test（如果有次近鄰）
            if len(matches_pair) >= 2:
                n = matches_pair[1]
                if m.distance >= nn_ratio * n.distance:
                    continue
            
            # 距離閾值
            if m.distance > descriptor_dist_thr:
                continue
            
            pt = valid_points[m.queryIdx]
            kp_idx = unmatched_kp_idxs[m.trainIdx]
            
            # 直接比較：如果新距離更小，就更新
            if m.distance < kp_best_dist[kp_idx]:
                kp_best_dist[kp_idx] = m.distance
                kp_best_pt[kp_idx] = pt
        
        
        # 5. 建立最終匹配字典
        for kp_idx, pt in kp_best_pt.items():
            matches[pt] = kp_idx
        
        return matches


## Object ↔ Object
class ObjectMatcher:
    """
    物體遮罩與物體遮罩的批配（基於 IoU）    
    """
    
    @staticmethod
    def greedy_match(iou_mat:np.ndarray[any, np.float32],iou_thr:float)->list[tuple[int,int,float]]:
        """
        根據 IoU 矩陣進行物件遮罩的貪婪配對。
        
        參數：
            iou_mat: np.ndarray，IoU 矩陣，形狀為 (N_l, N_c)，每個元素代表上一幀物件與當前幀物件的遮罩重疊程度
            iou_thr: float，IoU 配對門檻值，只有 IoU 大於此值才會配對
        回傳：
            matches: list[tuple[int,int,float]]，配對結果，每個元素為 (上一幀物件索引, 當前幀物件索引, IoU值)
        """
        # 貪婪配對：每次挑最大 IoU
        matches = []
        while True:
            # 找目前最大的 IoU
            i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            best = float(iou_mat[i, j])
            if best < iou_thr:
                break
            matches.append((i, j, best))
            # 將該列/行設為 -1，避免重覆配
            iou_mat[i, :] = -1.0
            iou_mat[:, j] = -1.0
            
        return matches
    
    
    
def descriptor_distance(a, b):
    return np.count_nonzero(a!=b)
    
def descriptor_distances(a, b):
    return np.count_nonzero(a!=b,axis=1)
