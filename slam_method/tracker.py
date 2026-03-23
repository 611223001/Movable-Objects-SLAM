
import cv2
import numpy as np
from enum import Enum
from typing import List
from collections import defaultdict


from .pose import triangulate
from .utils import SlamState, add_ones,computeE21,compute_epipolar_error,compute_epipolar_errors,check_epipolar,computeE21_pose
from .frame import Frame, KeyFrame
from .point import Point
from .feature import FeatureTool
from .matcher import MapPointMatcher, KeypointMatcher




from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .slam import Slam

class Tracker():
    """
    追蹤 2D影像 kp 及 3D地圖 point 之間的關聯及變化
    """
    def __init__(self,slam:'Slam'):
        
        self.slam =slam
        self.map = slam.map
        self.camera = slam.camera
        self.feature_tool = slam.feature_tool
        self.vo =  slam.vo
        self.mm = slam.vo.mm
        self.map_display =  slam.map_display
        self.state_display =  slam.state_display
        self.control =  slam.control
        
        # 光流
        self.dis = cv2.DISOpticalFlow.create(2) # input 0:很快, 1:快, 2:中等 （品質↑速度↓）
        
        # 屬性: frame 
        self.f_cur:Frame
        self.f_last:Frame
        
        self.kf_cur:KeyFrame
        self.kf_last:KeyFrame
        self.f_ref:Frame|KeyFrame# f_ref 可能是f_last或kf_cur，用於與f_cur比較kps
        
        # 屬性: other
        self.last_reloc_frame_id:int = 0
        self.need_kf:bool = False
        
    def optical_flow(self,f_cur:Frame,f_last:Frame):
        img_cur = cv2.cvtColor(f_cur.img,cv2.COLOR_BGR2GRAY)
        img_last = cv2.cvtColor(f_last.img,cv2.COLOR_BGR2GRAY)
        flow = self.dis.calc(img_last,img_cur, None,)
        return flow
    
    def track_previous_frame(self, f_last:'Frame', f_cur:'Frame'):
        """
        追蹤上一幀的地圖點到當前幀。
        Args:
            f_last: 前一幀 Frame
            f_cur: 當前幀 Frame
        Returns:
            success: 是否追蹤成功（bool）
            nmatches: 匹配數量
        """
        # 設定金字塔層級範圍
        max_octave = np.max(f_cur.octaves)
        min_octave = 0
        if len(self.map.keyframes) > 5:
            min_octave = max_octave // 2 + 1

        # 第一次 window search（粗略配對）
        matches = KeypointMatcher.window_search(f_last, f_cur, window_size=200, min_scale_level=min_octave, max_scale_level=max_octave)
        nmatches = len(matches)
        vp_map_point_matches = [None] * len(f_cur.kps)
        for ref_idx, cur_idx in matches:
            vp_map_point_matches[cur_idx] = f_last.points[ref_idx]
        
        # 若配對不足，縮小窗口並移除層級限制再搜尋
        if nmatches < 10:
            matches = KeypointMatcher.window_search(f_last, f_cur, window_size=100, min_scale_level=0, max_scale_level=max_octave)
            nmatches = len(matches)
            vp_map_point_matches = [None] * len(f_cur.kps)
            for ref_idx, cur_idx in matches:
                vp_map_point_matches[cur_idx] = f_last.points[ref_idx]
            if nmatches < 10:
                vp_map_point_matches = [None] * len(f_cur.kps)
                nmatches = 0

        # 複製上一幀 pose 到當前幀
        f_cur.pose = f_last.pose.copy()

        # 設定當前幀的地圖點
        # f_cur.points = np.array(vp_map_point_matches, dtype=object)

        for idx, pt in enumerate(vp_map_point_matches):
            if pt is None:continue
            f_cur.add_point(pt,idx)

        # 若配對足夠，優化 pose 並用投影再搜尋
        if nmatches >= 10:
            # 優化 pose
            self.vo.pose_optimize(f_cur)

            # 移除外點
            num_clear = f_cur.clear_outliers()
            nmatches -= num_clear
            
            # 用投影再搜尋
            nmatches += MapPointMatcher.search_by_projection(f_cur, f_last.points, 15)
        else:
            # 最後機會，用較大窗口投影搜尋
            nmatches = MapPointMatcher.search_by_projection(f_cur, f_last.points, 50)
            pass


        # 若配對仍不足，回傳失敗
        if nmatches < 10:
            return False, nmatches

        # 最後再優化 pose
        self.vo.pose_optimize(f_cur)

        # 再次移除外點
        num_clear = f_cur.clear_outliers()
        nmatches -= num_clear


        return nmatches >= 10, nmatches
    
    def track_by_motion_model(self):
        Tlc = self.mm.predict_pose()
        self.f_cur.pose = self.f_last.pose @ Tlc # Twl * Tlc = Twc
        
        self.f_cur.clear_points()
        
        
        nmatches = MapPointMatcher.search_by_projection(self.f_cur, self.f_last.points, 15, nn_ratio = 0.9)
        # nmatches =MapPointMatcher.search_by_projection_frame(self.f_cur, self.f_last,15,nn_ratio = 0.9,check_ori=True)
        
        
        if nmatches <20:
            return False
        
        err = self.vo.pose_optimize(self.f_cur)
        num_clear = self.f_cur.clear_outliers()
        nmatches -= num_clear
        
        return nmatches>=10
    
    
    def track_by_semantic_optical_flow(self,f_cur:Frame,f_last:Frame,flow):
        f_cur.pose = f_last.pose.copy()
        
        nmatches, matches = MapPointMatcher.search_by_flow(f_cur, f_last, flow, 7, nn_ratio = 0.9)
        
        if nmatches <20:
            return False
        
        # like SOF-SLAM
        # 初步標記物體是否動態
        
        # objs_pt_num = [0]*len(f_cur.objects)
        
        obj_matches:defaultdict[int,list[tuple[int,int]]] = defaultdict(list)
        for idx_cur,idx_last in matches:
            obj_idx = f_cur.get_obj_idx(idx_cur)
            obj_matches[obj_idx].append((idx_cur,idx_last))
        
        
        for obj_idx,obj in enumerate(f_cur.objects):
            
            obj_info = f_cur.obj_infos[obj_idx]
            # print(np.array(f_cur.obj_to_pts[obj_idx]),'np.array(f_cur.obj_to_pts[obj_idx])')
            
            # kp_idxs = list(f_cur.obj_to_pts[obj_idx])
            # print(kp_idxs,'kp_idxs')
            # objs_pt_num[obj_idx] = np.sum(f_cur.points[kp_idxs] != None) # 紀錄每個物體上觀察到的地圖點數量
            
            if obj_info['isthing']:
                if obj_info['category_id']==0: # person
                    f_cur.objects_dynamic[obj_idx] = True
                elif obj is not None and (obj.dynamic is True): # 已知動態的地圖物件
                    f_cur.objects_dynamic[obj_idx] = True
                else:
                    f_cur.objects_dynamic[obj_idx] = None
                            
            else:
                f_cur.objects_dynamic[obj_idx]= False
                
        
        
        def _get_matches_by_dynamic_statuses(obj_matches: dict, objects_dynamic:list, *statuses:bool|None) -> np.ndarray:
            """
            快速獲取指定動態狀態的 matches
            Args:
                statuses: True (動態), False (靜態), None (可能動態)
            Returns:
                shape (N, 2) 的 matches 陣列
            """
            matches_list = []
            for obj_idx, obj_matches_list in obj_matches.items():
                if obj_idx >= len(objects_dynamic):
                    continue
                if objects_dynamic[obj_idx] in statuses:
                    matches_list.extend(obj_matches_list)
            
            if len(matches_list) == 0:
                return np.empty((0, 2), dtype=int)
            return np.array(matches_list, dtype=int)
        
        
        if self.mm.working:
            Tlc = self.mm.predict_pose()
            E21 = computeE21_pose(Tlc)
            
        else:
            first_matches = _get_matches_by_dynamic_statuses(obj_matches, f_cur.objects_dynamic, False,None)
            E21, mask = cv2.findEssentialMat(
            f_cur.kps[first_matches[:,0]], f_last.kps[first_matches[:,1]],
            method=cv2.RANSAC, prob=0.999, threshold=0.02
            )
        
        # # 靜態特徵點
        # static_matches = _get_matches_by_dynamic_statuses(obj_matches, f_cur.objects_dynamic, False)
        # kp_idxs1, kp_idxs2 = static_matches[:,0], static_matches[:,1]
        # errors_E21 = compute_epipolar_errors(f_cur.kps[kp_idxs1], f_last.kps[kp_idxs2], E21)
        # for error_E21, kp_idx1, kp_idx2 in zip(errors_E21,kp_idxs1,kp_idxs2):
        #     epipolar_check = check_epipolar(f_last,kp_idx2,error_E21,9.21)
        #     if not epipolar_check:
        #         point:Point = f_cur.points[kp_idx1]
        #         f_cur.remove_point(point)
        #         nmatches -= 1
        #         point.set_dynamic(f_cur.id)
        
        
        # 潛在動態特徵點
        potentially_dynamic_matches = _get_matches_by_dynamic_statuses(obj_matches, f_cur.objects_dynamic, None)
        kp_idxs1, kp_idxs2 = potentially_dynamic_matches[:,0], potentially_dynamic_matches[:,1]
        errors_E21 = compute_epipolar_errors(f_cur.kps[kp_idxs1], f_last.kps[kp_idxs2], E21)
        for error_E21, kp_idx1, kp_idx2 in zip(errors_E21,kp_idxs1,kp_idxs2):
            epipolar_check = check_epipolar(f_last,kp_idx2,error_E21,9.21)
            if not epipolar_check:
                point:Point = f_cur.points[kp_idx1]
                f_cur.remove_point(point)
                nmatches -= 1
                # point.set_dynamic(f_cur.id) # TODO: 這裡不應該設置動態
        
        # 動態特徵點
        dynamic_matches = _get_matches_by_dynamic_statuses(obj_matches, f_cur.objects_dynamic, True)
        kp_idxs1, kp_idxs2 = dynamic_matches[:,0], dynamic_matches[:,1]
        errors_E21 = compute_epipolar_errors(f_cur.kps[kp_idxs1], f_last.kps[kp_idxs2], E21)
        for error_E21, kp_idx1, kp_idx2 in zip(errors_E21,kp_idxs1,kp_idxs2):
            point:Point = f_cur.points[kp_idx1]
            f_cur.remove_point(point)
            nmatches -= 1
            # point.set_dynamic(f_cur.id) # TODO: 這裡不應該設置動態
                
        
        
        err = self.vo.pose_optimize(f_cur)
        num_clear = self.f_cur.clear_outliers()
        nmatches -= num_clear
        
        # # 判斷動態分數並再次標記物體是否動態
        # # dynamic_scores = {}
        # for obj_idx,obj in enumerate(f_cur.objects):
        #     if f_cur.objects_dynamic[obj_idx] is not None:continue
            
        #     obj_info = f_cur.obj_infos[obj_idx]
        #     obj_kp_idxs = list(f_cur.obj_to_pts[obj_idx])
        #     obj_static_pt_num = np.sum(f_cur.points[obj_kp_idxs] != None)
            
        #     # obj_pt_num = objs_pt_num[obj_idx]
        #     len_obj_match = len(obj_matches[obj_idx])
            
            
        #     # if f_cur.obj_infos[obj_idx]['isthing']:
        #     #     if len(obj_matches[obj_idx])<5:
        #     #         dynamic_scores[obj_idx] = None
        #     #     else:
        #     #         dynamic_scores[obj_idx] = float( (len_obj_match-obj_static_pt_num) / len_obj_match)
            
        #     # if obj_static_pt_num*2 > len_obj_match:
        #     #     f_cur.objects_dynamic[obj_idx] = False
            
        #     # if obj_static_pt_num*5 < len_obj_match:
        #     #     f_cur.objects_dynamic[obj_idx] = True
        #     #     for pt in f_cur.points[obj_kp_idxs]:
        #     #         f_cur.remove_point(pt)
            
        #     if len(obj_matches[obj_idx])<5:
        #         dynamic_score = None
        #     else:
        #         dynamic_score = float( (len_obj_match-obj_static_pt_num) / len_obj_match)
            
        #     f_cur.objects_dynamic_score[obj_idx] = dynamic_score
            
        #     if dynamic_score is None:
        #         if obj is None:
        #             dynamic = False
        #         else:
        #             dynamic = obj.dynamic
            
        #     else:
        #         dynamic = dynamic_score > 0.7
            
        #     if dynamic:
        #         for pt in f_cur.points[obj_kp_idxs]:
        #             f_cur.remove_point(pt)
            
        #     f_cur.objects_dynamic[obj_idx] = dynamic

        # print({k: f"{v:.3f}" if v is not None else None for k, v in enumerate(f_cur.objects_dynamic_score)})
                
        # print({k: f"{v:.3f}" if v is not None else None for k, v in dynamic_scores.items()})
        
        return nmatches>=10
    
    def track_by_semantic_optical_flow2(self,f_cur:Frame,f_last:Frame,flow):
        f_cur.pose = f_last.pose.copy()
        
        nmatches, matches = KeypointMatcher.search_kps_by_flow(f_cur, f_last, flow, 5, nn_ratio = 0.9)
        
        if nmatches <20:
            return False
        
        # like SOF-SLAM
        # 初步標記物體是否動態
        objs_pt_num = [0]*len(f_cur.objects)
        # for obj_idx in range(len(f_cur.objects)):
            # obj = f_cur.objects[obj_idx]
        
        if self.mm.working:
            Tlc = self.mm.predict_pose()
            E21 = computeE21_pose(Tlc)
            scores = {}
            
            obj_matches:defaultdict[int,list[tuple[int,int]]] = defaultdict(list)
            for idx_cur,idx_last in matches:
                obj_idx = f_cur.get_obj_idx(idx_cur)
                # dynamic = f_cur.objects_dynamic[obj_idx]
                obj_matches[obj_idx].append((idx_cur,idx_last))
                
                
            
            
            for obj_idx,obj in enumerate(f_cur.objects):
                if not f_cur.obj_infos[obj_idx]['isthing']:continue
                matches = obj_matches[obj_idx]
                if f_cur.obj_infos[obj_idx]['category_id']==0: # person
                    score = 1.0
                else:
                    
                    score = self.compute_dynamic_score(f_cur,f_last,matches,E21)
                scores[obj_idx] = score
            
            print('dynamic_score',scores)
        
        
        return matches
    
    # 移至 matcher
    # def search_by_projection_frame(self, f_cur:Frame, f_last:Frame, th: float, nn_ratio: float = 0.8, check_ori: bool = True) -> int:
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
                dist = self.feature_tool.descriptor_distance(dMP, d)
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
    
    # def search_by_projection(self, frame:Frame, points:list[Point|None], radius_factor:float=1.0, nn_ratio:float=0.8, distance_threshold:int=100,add_point=True, check_dynamic=False)->int:
    #     """
    #     以投影方式在 frame 上搜尋 points 的最佳匹配。
    #     Args:
    #         frame: 目標 Frame
    #         points: 要投影搜尋的地圖點列表
    #         radius_factor: 搜尋半徑縮放因子
    #         nn_ratio: 最近鄰比值
    #         distance_threshold: Hamming 距離閾值
    #     Returns:
    #         nmatches: 匹配數量
    #     """
    #     nmatches = 0
    #     matches = []
    #     scale_factors = frame.scale_factors
    #     n_max_level = len(scale_factors)-1
        
    #     if check_dynamic:
    #         dynamic_objs = set()
    #         for obj, dynamic in zip(frame.objects,frame.objects_dynamic):
    #             if dynamic is not False:
    #                 dynamic_objs.add(obj)
                    
    #         for obj,dynamic in self.map.dynamic_objcts.items():
    #             if dynamic is not False:
    #                 dynamic_objs.add(obj)
                
        
    #     for point in points:
    #         if point is None or point.is_bad:
    #             continue
            
    #         if check_dynamic:
    #             if point.object in dynamic_objs:
    #                 continue
                

    #         # 投影到影像
    #         # visible, proj = frame.is_visible(point)
    #         # proj_xy = proj[:2]
    #         uv, visible = frame.project_point_to_img(point.location)
    #         if not visible:
    #             continue

    #         # # 根據點的尺度調整搜尋半徑（可用平均size或固定值）
    #         # scale_level = np.median(frame.octaves) # if len(frame.octaves) > 0 else 0
    #         # base_radius = np.median(frame.sizes) # if hasattr(frame, "sizes") and len(frame.sizes) > 0 else 10.0
    #         # radius = base_radius * radius_factor
            
    #         # 深度與尺度一致性檢查
    #         min_dist,max_dist = point.update_depth()
    #         dist3D = np.linalg.norm(point.location_3D - frame.position)
    #         if not (min_dist < dist3D < max_dist):
    #             continue
    #         # 預測金字塔層級
    #         ratio = dist3D / (min_dist + 1e-6)# if min_dist > 0 else 1.0
    #         pred_octave = min(np.searchsorted(scale_factors, ratio), n_max_level)
    #         # 搜尋半徑根據尺度調整
    #         radius = radius_factor * scale_factors[pred_octave]
            
    #         # print('radius',radius,n_pred_level,ratio,dist3D,min_dist)
            
            
    #         # KDTree搜尋半徑內的keypoint
    #         near_indices = frame.raw_kd.query_ball_point(uv, radius)
            
    #         # # 只考慮未配對的keypoint
    #         # near_indices = [idx for idx in near_indices if frame.points[idx] is None]
            
    #         # 只考慮未配對且octave在合理範圍的keypoint
    #         near_indices = [idx for idx in near_indices
    #                         if frame.points[idx] is None and
    #                         (frame.octaves[idx] >= pred_octave-1 and frame.octaves[idx] <= pred_octave)]

    #         if not near_indices:
    #             continue

    #         # 找最佳描述子匹配
    #         best_dist = float('inf')
    #         best_dist2 = float('inf')
    #         best_idx = -1
    #         for idx in near_indices:
    #             d = frame.des[idx]
    #             dist = self.feature_tool.descriptor_distance(point.des, d)
    #             if dist < best_dist:
    #                 best_dist2 = best_dist
    #                 best_dist = dist
    #                 best_idx = idx
    #             elif dist < best_dist2:
    #                 best_dist2 = dist

    #         # NN ratio 與距離檢查
    #         if best_dist <= distance_threshold:
    #             if best_dist > nn_ratio * best_dist2:
    #                 continue
    #             if add_point:
    #                 frame.add_point(point, best_idx)
    #                 frame.set_track_method(best_idx,2)
                    
    #             matches.append((best_idx,point))
    #             nmatches += 1
                
        
    #     return nmatches
    
    # def search_by_flow(self, frame:Frame, f_last:Frame, flow, radius_factor:float=1.0, nn_ratio:float=0.8, distance_threshold:int=100,add_point=True)->int:
    #     """
    #     以光流方式在 frame 上搜尋 points 的最佳匹配。
    #     Args:
    #         frame: 目標 Frame
    #         f_last: 必須包含地圖點
    #         radius_factor: 搜尋半徑縮放因子
    #         nn_ratio: 最近鄰比值
    #         distance_threshold: Hamming 距離閾值
    #     Returns:
    #         nmatches: 匹配數量
    #         matches: list
    #     """
    #     nmatches = 0
    #     matches = []
    #     for idx2, point in enumerate(f_last.points):
    #         if point is None or point.is_bad:
    #             continue
            
    #         # 取得光流對應位置
    #         u, v = f_last.raw_kps[idx2]
    #         uu,vv = flow[v,u]
    #         uv = (u+uu, v+vv)
            
            
    #         # 根據點的尺度調整搜尋半徑
    #         scale_level = int(f_last.octaves[idx2])
    #         base_radius = f_last.sizes[idx2]
    #         radius = radius_factor * base_radius
            
    #         # KDTree搜尋半徑內的keypoint
    #         near_indices = frame.raw_kd.query_ball_point(uv, radius)
            
    #         # # 只考慮未配對的keypoint
    #         # near_indices = [idx for idx in near_indices if frame.points[idx] is None]
            
    #         # 只考慮未配對且octave在合理範圍的keypoint
    #         near_indices = [idx1 for idx1 in near_indices
    #                         if frame.points[idx1] is None and
    #                         (scale_level >= frame.octaves[idx1] and frame.octaves[idx1] >= scale_level-1)]
            
    #         if not near_indices:
    #             continue
            
    #         # 找最佳描述子匹配
    #         best_dist = float('inf')
    #         best_dist2 = float('inf')
    #         best_idx = -1
    #         for idx1 in near_indices:
    #             des = frame.des[idx1]
    #             dist = self.feature_tool.descriptor_distance(point.des, des)
    #             if dist < best_dist:
    #                 best_dist2 = best_dist
    #                 best_dist = dist
    #                 best_idx = idx1
    #             elif dist < best_dist2:
    #                 best_dist2 = dist
            
    #         # NN ratio 與距離檢查
    #         if best_dist <= distance_threshold:
    #             if best_dist > nn_ratio * best_dist2:
    #                 continue
    #             if add_point:
    #                 frame.add_point(point, best_idx)
    #                 frame.set_track_method(best_idx,3)
                
    #             matches.append((best_idx,idx2))
    #             nmatches += 1
                

    #     return nmatches, matches
    
    # def search_kps_by_flow(self, frame:Frame, f_last:Frame, flow, radius_factor:float=1.0, nn_ratio:float=0.8, distance_threshold:int=100)->int:
    #     """
    #     以光流方式在 frame 上搜尋 kps 的最佳匹配。
    #     Args:
    #         frame: 目標 Frame
    #         f_last: 上個 frame
    #         flow: frame 間光流
    #         radius_factor: 搜尋半徑縮放因子
    #         nn_ratio: 最近鄰比值
    #         distance_threshold: Hamming 距離閾值
    #     Returns:
    #         nmatches: 匹配數量
    #         matches: list
    #     """
    #     nmatches = 0
    #     matches = []
    #     for idx2, point in enumerate(f_last.points):
    #         # if point is None or point.is_bad:
    #         #     continue
            
    #         # 取得光流對應位置
    #         u, v = f_last.raw_kps[idx2]
    #         uu,vv = flow[v,u]
    #         uv = (u+uu, v+vv)
            
            
    #         # 根據點的尺度調整搜尋半徑
    #         scale_level = int(f_last.octaves[idx2])
    #         base_radius = f_last.sizes[idx2]
    #         radius = radius_factor * base_radius
            
    #         # KDTree搜尋半徑內的keypoint
    #         near_indices = frame.raw_kd.query_ball_point(uv, radius)
            
            
    #         # 只考慮octave在合理範圍的keypoint
    #         near_indices = [idx1 for idx1 in near_indices
    #                         if (scale_level >= frame.octaves[idx1] and frame.octaves[idx1] >= scale_level-1)]
            
    #         if not near_indices:
    #             continue
            
    #         # 找最佳描述子匹配
    #         best_dist = float('inf')
    #         best_dist2 = float('inf')
    #         best_idx = -1
            
    #         des2 = f_last.des[idx2]
    #         for idx1 in near_indices:
    #             des1 = frame.des[idx1]
    #             dist = self.feature_tool.descriptor_distance(des2, des1)
    #             if dist < best_dist:
    #                 best_dist2 = best_dist
    #                 best_dist = dist
    #                 best_idx = idx1
    #             elif dist < best_dist2:
    #                 best_dist2 = dist
            
    #         # NN ratio 與距離檢查
    #         if best_dist <= distance_threshold:
    #             if best_dist > nn_ratio * best_dist2:
    #                 continue
                
    #             matches.append((best_idx,idx2))
    #             nmatches += 1
                

    #     return nmatches, matches
    
    
    def compute_dynamic_score(self, f_cur:Frame, f_last:Frame, matches, E):
        """計算物體的動態分數 (0-1)"""
        
        if len(matches) < 5:  # 少於 5 個點不可靠
            return None
        
        epipolar_violations = 0
        total_points = 0
        for kp_idx_cur ,kp_idx_last in matches:
            # if f_cur.points[kp_idx] is None: continue
            
            # print(f_cur.kps[kp_idx], kp_ret)
            # 檢查對極約束
            error = compute_epipolar_error(
                f_cur.kps[kp_idx_cur], 
                f_last.kps[kp_idx_last], 
                E
            )
            # if error > 9.21:  # chi-square threshold
            #     epipolar_violations += 1
            
            if not check_epipolar(f_cur, kp_idx_cur, error,9.21):
                epipolar_violations += 1
                
            total_points += 1
        
        # if total_points == 0:
        #     return 0.0
        
        return epipolar_violations / total_points
    
    def track_local_map(self,frame:Frame,local_points:list[Point|None],last_reloc_frame_id:int=None):
        
        if last_reloc_frame_id is None:
            last_reloc_frame_id = self.last_reloc_frame_id
                    
        # points in frame increase_visable
        for p in frame.points:
            if p is None:continue
            p:Point
            p.increase_visable()
        
        # visable points not in frame increase_visable
        n_to_matches:int = 0
        for i, pt in enumerate(local_points):
            if pt.is_bad: continue
            if frame in pt.frames: continue
            
            visible, point_project = frame.is_visible(pt,0.5)
            if visible:
                pt.increase_visable()
                n_to_matches += 1
            else:
                local_points[i] = None
        
        
        if n_to_matches>0:
            
            if frame.id< last_reloc_frame_id+2:
                th = 5
            else: th = 2
            
            MapPointMatcher.search_by_projection(frame,local_points,radius_factor=th,nn_ratio=0.8, check_dynamic=True)
            
            err = self.vo.pose_optimize(frame)
            # 不要清理 outliers
            self.state_display.set_info('pose_opt',err)
            
            # points in frame increase_found
            for p in frame.points:
                if p is None:continue
                p:Point
                p.increase_found()
        
        # 判斷追蹤是否成功
        # 如果最近進行了重新定位，則限制更強
        if frame.id < last_reloc_frame_id+self.slam.max_frames:
            if frame.points_num < 40:
                return False
        else:
            if frame.points_num < 20:
                return False
        
        return True
    
    
    
    # 不太準確
    # 重新定位
    def relocalization(self,frame:'Frame')->tuple[bool,'KeyFrame']:
        best_inliers = np.array([])
        best_kp_idxs1 = np.array([])
        best_kp_idxs2 = np.array([])
        best_kf:'KeyFrame' = None
        best_pose = None
        
        
        kfs = list(self.map.keyframes)
        for kf in kfs:
            
            # kp_idxs1, kp_idxs2, ret = flannMatches(frame, kf)
            # if len(ret) < 20:
            #     continue
            
            flow = self.optical_flow(frame,kf)
            nmatches, matches = MapPointMatcher.search_by_flow(frame, kf, flow, 15, nn_ratio = 0.75,distance_threshold=50,add_point=False)
            if nmatches < 20:
                continue
            
            pts_3d = []
            pts_3d_idxs = []
            pts_2d = []
            pts_2d_idxs = []
            
            # for kp_idx1,kp_idx2 in zip(kp_idxs1, kp_idxs2):
            for kp_idx1,kp_idx2 in matches:
                
                pt_3d = kf.points[kp_idx2]
                if pt_3d is None: continue
                pt_3d:'Point'
                
                pts_3d.append(pt_3d.location[:3]) #去除齊次化
                pts_2d.append(frame.kps[kp_idx1])
                pts_3d_idxs.append(kp_idx2)
                pts_2d_idxs.append(kp_idx1)
                
            pts_3d = np.array(pts_3d, dtype=np.float32)
            pts_2d = np.array(pts_2d, dtype=np.float32)
            pts_2d_idxs = np.array(pts_2d_idxs)
            pts_3d_idxs = np.array(pts_3d_idxs)
            
            if len(pts_3d) < 20 or len(pts_2d) < 20:continue
            # 需要至少 4 個 3D 和 2D 點才能進行 PnP 計算，取大於20的保證穩定性
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, cameraMatrix=np.eye(3),distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=self.camera.radius/100, iterationsCount=100
            )
            # success: 布林值，表示是否成功找到解。
            # rvec: 旋轉向量，表示相機的旋轉。shape(3, 1)
            # tvec: 平移向量，表示相機的平移。shape(3, 1)
            # inliers: 內點的索引，表示哪些點被認為是內點。
            inliers = np.squeeze(inliers) # 刪除陣列中形狀為 1 的維度
            # 內點至少10個，估計pose比較可信
            if not success or inliers is None or len(inliers) < 10:
                continue
            
            if len(inliers) > len(best_inliers):
                best_pose = (rvec, tvec)
                best_inliers = inliers
                best_kf = kf
                best_kp_idxs1 = pts_2d_idxs[inliers]
                best_kp_idxs2 = pts_3d_idxs[inliers]
        
        print('relocal',len(kfs),len(best_inliers),best_pose)
        
        inliers_num = 0
        # 若成功求得位姿，則進行位姿優化（PoseOptimization）。
        if best_pose is not None and len(best_inliers) >= 10:
            
            kf_ref = best_kf
            kp_idxs2 = best_kp_idxs2
            kp_idxs1 = best_kp_idxs1
            
            for pt,kp_idx1 in zip(kf_ref.points[kp_idxs2],kp_idxs1):
                assert pt is not None,pt
                frame.add_point(pt,kp_idx1)
                # self.map.add_point_frame_relation(pt,frame,kp_idx1)
                
            # 設定當前幀的位姿
            pose = np.eye(4)
            R, _ = cv2.Rodrigues(best_pose[0])
            pose[:3, :3] = R
            pose[:3, 3] = best_pose[1].flatten()
            
            
            frame.pose = pose
            
            
            
            self.vo.pose_optimize(frame)
            frame.clear_outliers()
            inliers_num = frame.points_num
            print('A',inliers_num)
            
            # 若內點（inlier）數太少，則進行投影匹配補充（SearchByProjection），再優化。
            if (inliers_num > 10) and (inliers_num < 50):
                
                local_points = self.map.get_covisible_points(kf_ref,20)
                
                nmatches = MapPointMatcher.search_by_projection(frame, local_points, 10, nn_ratio = 0.9)
                print('B',nmatches, inliers_num,nmatches + inliers_num)
                if nmatches + inliers_num >50:
                    
                    self.vo.pose_optimize(frame)
                    # frame.clear_outliers()
                    # inliers_num = frame.points_num
                    inliers_num = frame.points_num - np.sum(frame.outliers)
                    
                    print('C',frame.points_num , np.sum(frame.outliers))
                    
                    # 若內點數仍不足，進一步縮小投影窗口再補充匹配。
                    if inliers_num>30 and inliers_num<50:
                        nmatches = MapPointMatcher.search_by_projection(frame, local_points, 3, nn_ratio = 0.9)
                        print('D',nmatches,inliers_num)
                        if inliers_num+ inliers_num > 50 :
                            self.vo.pose_optimize(frame)
                            frame.clear_outliers()
                            inliers_num = frame.points_num
                            print('E',inliers_num)
                            
        print('out',inliers_num,frame.points_num)
        
        if  inliers_num>=50:
            return True ,kf_ref
        else:
            return False ,None
        
        
    def track_2D_2D(self,f1:'Frame',f2:'Frame',kp_idxs1:int,kp_idxs2:int):
        """
        
        在兩個影像幀之間追蹤2D關鍵點，並建立點與幀的關聯。
        對於每一對對應的關鍵點索引，如果第二個幀中的點存在，
        則將該點與第一個幀的指定關鍵點索引建立關聯。
        Args:
            f1 (Frame): 第一個包含要追蹤關鍵點的影像幀。
            f2 (Frame): 第二個包含關鍵點及其對應地圖點的影像幀。
            kp_idxs1 (int): 第一個幀中關鍵點的索引。
            kp_idxs2 (int): 第二個幀中對應關鍵點的索引。
        Returns:
            int: 增加的關聯數量
        """ 
        track_pts_count = 0
        for idx1,idx2 in zip(kp_idxs1,kp_idxs2):
            if f2.points[idx2] is None:continue
                
            self.map.add_point_frame_relation(f2.points[idx2],f1,idx1)
            track_pts_count += 1
        
        return track_pts_count
        
        
    def track_3D_2D(self,frame:Frame,points:list[Point]|np.ndarray[Point|None]):
        """
        通過將地圖點投影到frame上，建立地圖點與影像關鍵點的關聯，以追蹤點的變化

        Args:
            frame (Frame):  包含要追蹤關鍵點的影像幀。
            points (list[Point] | np.ndarray[Point | None]): 投影到影像幀的地圖點。

        Returns:
            int: 增加的關聯數量
        """
        track_pts_count = 0
        points = [point for point in points if point is not None] # 過濾None
        if len(points)==0: return track_pts_count
        
        
        # 檢查點是否可見 及投影點
        points_position = np.array([point.location for point in points ])
        points_visible,  points_project =frame.are_visible(points_position)        
        
        for i, point in enumerate(points) :

            if (frame in point.frames) or (not points_visible[i]) or (point.kf_num < 3):continue # 跳過以觀察到的點、觀察不到的點、還沒被校準的點
            
            point.increase_visable()

            for m_idx in frame.kd.query_ball_point(points_project[i][:2], frame.camera.radius / 200):
                # if point unmatched
                if frame.points[m_idx] is None:
                    b_dist = self.feature_tool.descriptor_distance(point.des,frame.des[m_idx])
                    # if any descriptors within 64
                    if b_dist < 50.0:
                        
                        fp = frame.points[m_idx]
                        if fp is None:
                            # self.add_point_frame_relation(point,frame,best_kp_idx)
                            frame.add_point(point, m_idx)
                        else:
                            point.fuse(fp)
                            fused_pts_count += 1
                        
                        track_pts_count += 1
                        break
        
        return track_pts_count
    
    def track_2D_3D(self):
        # 尚未製作
        ...
    def track_3D_3D(self):
        # 尚未製作
        ...
    
        # 初始化
    

    def need_new_keyframe_old(self)->bool:
        
        # # 上個kf模糊且這個f清楚
        # if (self.f_cur.is_clear) and (not self.kf_cur.is_clear):
        #     return True
            # 當前frame追蹤的點數量太少
        if np.sum(self.f_cur.points != None) <100:
            return True
        
        # 離上個keyframe太近
        if self.f_cur.id < self.kf_cur.id +int(self.camera.fps/6):
            return False
        
        # 離上個keyframe超過x幀
        if self.f_cur.id > self.kf_cur.id +int(self.camera.fps*3):
            return True



        if self.f_cur.points_num< self.kf_cur.points_num*0.75: 
            return True

        
        return False
    
    def need_new_keyframe(self)->bool:
        if False:   # 如果局部映射因循環閉合而凍結，則不要插入關鍵幀
            return False
        
        # 如果距離上次重定位的幀數不夠，則不插入
        if self.f_cur.id < self.last_reloc_frame_id + 2 :
            return False
        # A = True # 如果距離上次重新定位的幀數不夠，則不插入關鍵幀
        
        
        tracking_num = self.f_cur.points_num
        ref_tracking_num = self.kf_cur.points_num# - int(self.kf_cur.new_pt_num/2)
        
        # Local Mapping 是否閒置
        local_mapping_idle:bool = True # TODO
        
        frame_num = self.f_cur.id - self.kf_cur.id
        # 條件1a：距離上次插入關鍵幀已超過最大幀數
        c1a = frame_num >= self.slam.max_frames
        # 條件1b：距離上次插入關鍵幀已超過最小幀數，且 Local Mapping 閒置
        c1b = frame_num >= self.slam.min_frames and local_mapping_idle
        
        # 條件2：目前追蹤到的 inlier 數量少於參考關鍵幀的 90%，且 inlier 數量大於 15
        c2 = tracking_num < ref_tracking_num*0.90 and tracking_num >15
        
        
        c3 = tracking_num < ref_tracking_num*0.70 and tracking_num >15 
        
        
        
        if ((c1a or c1b) and c2) or c3:
            print(f"need_new_keyframe\n c1a {c1a},c1b {c1b}, c2 {c2}, c3 {c3}")
            print(f"：目前追蹤到的 inlier {tracking_num} 數量少於參考關鍵幀的{ref_tracking_num} 75%，且 inlier 數量大於 15")
            if local_mapping_idle:
                return True
            else:
                # self.slam.local_mapper.interrupt_ba() # TODO:中斷local_mapper的localBA
                return False
        else:
            return False
        
        
        # B = self.f_cur.id > self.kf_cur.id +int(self.camera.fps/3) #局部映射處於空閒狀態，或自上次插入關鍵影格以來已過去了 20 多幀。
        # C = np.sum(self.f_cur.points != None) >50 #當前幀追蹤至少 50 個點。
        
        # D = tracking_num < ref_tracking_num*0.90 #目前影格追蹤的點數比 Kref 少 90%。
        
        # return B and C and D

    
    def show_step(self,node:str):
        
        # 結束一個階段時等待並顯示當前畫面
        # if args[2] == 0: 
        #     img = paint_feature(self.f_cur.img.copy(),self.f_cur)
        #     cv2.imshow('current frame',img)
        #     cv2.waitKey(1)
        #     while self.control.stop and not self.control.next_step:
        #         ...
        
        # 顯示現在階段
        self.state_display.set_process_state(1,node)
    
    def track(self,img):
        use_motion_model = True
        
        
        
        self.show_step("add_f")
        frame = Frame(self.slam, img) # img之後會添加標記，所以使用img的副本
        
        self.f_last = self.f_cur
        self.f_cur = frame
        self.f_cur.kf_ref = self.kf_cur
        self.f_ref = self.f_last
        
        f_cur = self.f_cur
        flow = self.optical_flow(self.f_cur,self.f_last)
        
        # matches = self.track_by_semantic_optical_flow2(f_cur,self.f_last, flow)
        self.slam.obj_tool.obj_tracker.add_miss_counter()
        self.slam.obj_tool.track_objects_by_OpticalFlow(f_cur,self.f_last, flow)
        
        if self.slam.state == SlamState.WORKING:
            
            # 預測位姿
            self.show_step("track")
            
            
            
            # if (not use_motion_model) or (len(self.map.keyframes)<4) or (not self.mm.working) or (self.f_cur.id < self.last_reloc_frame_id+2)and True :
            #     # 將前一幀的點添加至當前幀
            #     track_success = self.track_previous_frame(self.f_last, self.f_cur)
            #     self.state_display.set_info('predict_pose_model',f"{self.f_cur.id}: window search") # track & opt
            # else:
            #     track_success = self.track_by_motion_model()
            #     if track_success:
            #         self.state_display.set_info('predict_pose_model',f"{self.f_cur.id}: motion model")
                    
            #     else :
            #         track_success = self.track_previous_frame(self.f_last, self.f_cur)
            #         self.state_display.set_info('predict_pose_model',f"{self.f_cur.id}: window search")
            
            

            track_success = self.track_by_semantic_optical_flow(f_cur,self.f_last, flow)
            self.state_display.set_info('predict_pose_model',f"{f_cur.id}: optical flow")
            
        
        else :
            ... # relocation
            self.show_step("relocal")
            
            track_success,kf_ref = self.relocalization(f_cur)
            self.state_display.set_info('relocal',(track_success,frame.points_num))
            
            self.last_reloc_frame_id = f_cur.id

        
        if  track_success:
            self.show_step("retrack")
            # track_local_map
            # 從局部地圖追蹤
            
            
            ptnum0 = f_cur.points_num
            local_pts = self.map.get_covisible_points(self.kf_cur)
            
            track_success = self.track_local_map(f_cur,local_pts)
            
            ptnum1 = f_cur.points_num
            
            self.state_display.set_info('track_num',(ptnum0,ptnum1-ptnum0))
            
        
        
        self.state_display.set_info('tracking_point',f_cur.points_num) # tracking points
        # 如果追蹤良好，檢查是否插入關鍵幀
        if track_success:
            self.slam.set_state(SlamState.WORKING)
            

            
            
            # 判斷動態分數並再次標記物體是否動態
            
            # for obj_idx,obj in enumerate(f_cur.objects):
            #     if obj is None:continue
            #     if not obj.is_thing:continue
                
            #     obj_info = f_cur.obj_infos[obj_idx]
            #     obj_kp_idxs = list(f_cur.obj_to_pts[obj_idx])
            #     obj_static_pt_num = np.sum(f_cur.points[obj_kp_idxs] != None)
                
            #     # len_obj_match = len(obj_matches[obj_idx])
                
            #     n_visible = 0
            #     for pt in obj.points:
            #         # if not pt in self.kf_cur.points:continue
            #         if not pt in local_pts:continue
                    
                    
            #         visible, point_project = f_cur.is_visible(pt,0.5)
            #         if visible:
            #             n_visible += 1
                    
            #     if n_visible <5:
            #         dynamic_score = None
            #     else:
            #         dynamic_score = float( (n_visible-obj_static_pt_num) / n_visible)
                
            #     f_cur.objects_dynamic_score[obj_idx] = dynamic_score
                
            #     if dynamic_score is None:
            #         dynamic = None
                
            #     else:
            #         dynamic = dynamic_score > 0.8
                
            #     if dynamic:
            #         for pt in f_cur.points[obj_kp_idxs]:
            #             f_cur.remove_point(pt)
                
            #     f_cur.objects_dynamic[obj_idx] = dynamic
            #     if obj.dynamic:
            #         print(f"oid:{obj.oid}, {(obj_static_pt_num,n_visible)}")
            
            
            # 判斷動態分數並再次標記物體是否動態
            nmatches, matches = KeypointMatcher.search_kps_by_flow(f_cur, self.f_last, flow, 5, nn_ratio = 0.9)
            Tlc = self.f_last.Tcw @ f_cur.Twc
            E21 = computeE21_pose(Tlc)
            obj_matches:defaultdict[int,list[tuple[int,int]]] = defaultdict(list)
            for idx_cur,idx_last in matches:
                obj_idx = f_cur.get_obj_idx(idx_cur)
                # dynamic = f_cur.objects_dynamic[obj_idx]
                obj_matches[obj_idx].append((idx_cur,idx_last))
            
            dynamic_scores2 = {}
            for obj_idx,obj in enumerate(f_cur.objects):
                if obj is None:continue
                if not obj.is_thing:continue
                matches = obj_matches[obj_idx]
                score = self.compute_dynamic_score(f_cur, self.f_last,matches,E21)
                
                if f_cur.obj_infos[obj_idx]['category_id']==0:  # person
                    dynamic = True
                    score = 1.0
                else:
                    if score is None:
                        dynamic = None
                    elif score > 0.7:  # 50% 違反 → 動態
                        dynamic = True
                    elif score < 0.3:
                        dynamic = False
                    else:
                        dynamic = None

                f_cur.objects_dynamic[obj_idx] = dynamic
                f_cur.objects_dynamic_score[obj_idx] = score
                
                # 測試用記得刪除
                f_cur.objects_dynamic[obj_idx] = False
                f_cur.objects_dynamic_score[obj_idx] = 0.0
                # 測試用記得刪除
                
                dynamic_scores2[obj_idx] = score
                
                
            # print({k: f"{v:.3f}" if v is not None else None for k, v in enumerate(f_cur.objects_dynamic_score)})
            # print({k: f"{v}" if v is not None else None for k, v in enumerate(f_cur.objects_dynamic)})
            print({k: f"{v:.3f}" if v is not None else None for k, v in dynamic_scores2.items()})
            
            
            self.show_step(0)# 輸入無效值清空顯示
            # print('debug',0)
            self.need_kf = self.need_new_keyframe()
            if self.need_kf and self.slam.state == SlamState.WORKING:
                # print('debug',1)
                self.state_display.set_info('f num btw kf',self.f_cur.id-self.kf_cur.id)
                kf_new = self.slam.local_mapper.local_map(self.f_cur)
                
                self.kf_last = self.kf_cur
                self.f_cur = kf_new
                self.kf_cur = kf_new
                self.state_display.set_info('kf_cur_point',self.kf_cur.points_num)
                
            
            # TODO
                # 我們允許創新點（Huber 函數認為是異常值）
                # 傳遞到新的關鍵幀，以便最終由BA決定它們是否是異常值。
                # 我們不希望下一幀使用這些點來估計其位置，因此我們將這些點從該幀中丟棄。
                # self.f_cur.clear_outliers()
            
            # kf_clear_outliers = self.f_cur.clear_outliers()
            # print('kf_clear_outliers:',kf_clear_outliers)
        else:
            self.slam.set_state(SlamState.LOST)
        
        # # 如果初始化後不久就lost，reset
        # if self.slam.state == SlamState.LOST:
        #     if len(self.map.keyframes) <=5:
        #         # self.reset() # TODO
        
        
        # 更新 motion model
        if use_motion_model:
            if track_success:
                self.mm.update_pose(self.f_cur.pose)
                
            else:
                self.mm.clear_pose()
        
        
        
        
        
        self.state_display.set_info('f_num',len(self.map.frames))
        return self.f_cur
        
        
        
    def compute_object_flow_consistency(self, f_cur:Frame, f_last:Frame, obj_idx:int, flow:np.ndarray) -> float:
        """
        計算物體遮罩內光流的一致性
        
        Args:
            f_cur: 當前幀
            f_last: 上一幀
            obj_idx: 物體索引
            flow: 光流場 (H, W, 2)，從 f_last 到 f_cur 的光流
                
        Returns:
            consistency_score: 0-1，越高越一致（靜態）
            None 如果無法計算
        """
        obj_kp_idxs = list(f_cur.obj_to_pts[obj_idx])
    
        if len(obj_kp_idxs) < 10:
            return None
        
        # 1. 計算相機的相對運動（從 cur 到 last）
        T_cl = f_cur.Tcw @ f_last.Twc
        R_cl = T_cl[:3, :3]
        t_cl = T_cl[:3, 3]
        
        
        # 根據相機運動調整閾值
        camera_motion = np.linalg.norm(t_cl)
        if camera_motion > 0.5:
            threshold = 8.0
        elif camera_motion > 0.2:
            threshold = 6.0
        else:
            threshold = 5.0
        
        # 2. 對於每個特徵點，比較實際光流與預測光流
        flow_errors = []
        
        for kp_idx in obj_kp_idxs:
            # 確保座標為整數
            u_cur = int(round(f_cur.raw_kps[kp_idx][0]))
            v_cur = int(round(f_cur.raw_kps[kp_idx][1]))
            
            # 檢查 cur 邊界
            if v_cur < 0 or v_cur >= flow.shape[0] or u_cur < 0 or u_cur >= flow.shape[1]:
                continue
            
            # 預測光流（基於相機運動）
            kp_norm_cur = f_cur.kps[kp_idx]  # 歸一化座標 (x_n, y_n)
            
            # 將 cur 的點投影到 last（假設點在無窮遠，只考慮旋轉）
            kp_3d_cur = np.array([kp_norm_cur[0], kp_norm_cur[1], 1.0])
            kp_3d_last = R_cl @ kp_3d_cur
            
            # 處理除以零
            if abs(kp_3d_last[2]) < 1e-6:
                continue
            
            kp_norm_last = kp_3d_last[:2] / kp_3d_last[2]
            kp_norm_last = np.array([kp_norm_last[0], kp_norm_last[1], 1.0])  # 添加齊次座標
            
            # 轉回像素座標
            pixel_last = f_last.camera.denormalize(kp_norm_last)[0]
            
            
            pixel_cur = np.array([u_cur, v_cur], dtype=np.float64)
            
            # 預測光流：從 last 到 cur 的位移
            predicted_flow = pixel_cur - pixel_last
            
            # 實際光流（在 last 的位置取值）
            u_last = int(round(pixel_last[0]))
            v_last = int(round(pixel_last[1]))
            
            # 檢查 last 的邊界
            if v_last < 0 or v_last >= flow.shape[0] or u_last < 0 or u_last >= flow.shape[1]:
                continue
            
            actual_flow = flow[v_last, u_last]
            
            # 計算誤差
            flow_error = np.linalg.norm(actual_flow - predicted_flow)
            flow_errors.append(flow_error)
        
        if len(flow_errors) < 5:
            return None
        
        # 3. 統計一致性
        # consistent_count = np.sum(np.array(flow_errors) < threshold)
        # consistency_score = consistent_count / len(flow_errors)
        consistency_score =  np.sum(np.array(flow_errors)) / len(flow_errors)
        
        return consistency_score
            