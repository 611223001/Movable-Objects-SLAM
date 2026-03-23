import cv2
import numpy as np
from enum import Enum
import math
from typing import List
import scipy.spatial.transform
import g2o


from .matcher import MapPointMatcher, KeypointMatcher

from .g2o_bundle_adjustment import BundleAdjustment
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .map import Map
    from .slam import Slam
    from .frame import Frame, KeyFrame
    from .point import Point
    from .objects import Thing, Stuff

class MotionModel:
    def __init__(self):
        self.poses:list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = [] # 紀錄最近的五個frame.pose(包含當前)
        self.tracking = [False]*5 # 紀錄最近的5個frame是否成功追蹤(不含當前)
        
        self.working:bool = False
        
        self.Rt:np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.eye(4)
        
        self.vel = None     # 速度(velocity)
        self.acc = None     # 加速度(acceleration)
        self.ang_vel = None    # 角速度(angular_velocity)
        self.ang_acc = None    # 角加速度(angular_acceleration)
    
    def clear_pose(self):
        self.poses = []
        self.working = False
        
    def update_pose(self,pose):
        """_summary_
        
        Args:
            pose (np.ndarray):shape(4,4) Twc 
        """
        # 注意不要輸入還不確定位置的幀位姿
        
        self.poses.append(pose)
        
        if len(self.poses) >=2:
            self.working = True
        
        
        if len(self.poses) > 3:
            self.poses.pop(0)
        
        
        if len(self.poses) > 2:
        # 計算速度
            p2 = self.poses[-2][:3, 3]
            p1 = self.poses[-1][:3, 3]
            self.vel = p1 - p2
        # 計算角速度
            R2 = self.poses[-2][:3, :3]
            R1 = self.poses[-1][:3, :3]
            r2 = scipy.spatial.transform.Rotation.from_matrix(R2)
            r1 = scipy.spatial.transform.Rotation.from_matrix(R1)
            delta_rot = r1 * r2.inv()
            self.ang_vel = delta_rot.as_rotvec()
            
        if len(self.poses) > 3:
        # 計算加速度
            v2 = self.poses[-2][:3, 3] - self.poses[-3][:3, 3]
            v1 = self.poses[-1][:3, 3] - self.poses[-2][:3, 3]
            self.acc = v1 - v2
        # 計算角加速度
            R3 = self.poses[-3][:3, :3]
            R2 = self.poses[-2][:3, :3]
            R1 = self.poses[-1][:3, :3]
            rot3 = scipy.spatial.transform.Rotation.from_matrix(R3)
            rot2 = scipy.spatial.transform.Rotation.from_matrix(R2)
            rot1 = scipy.spatial.transform.Rotation.from_matrix(R1)
            ang_vel3 = (rot2 * rot3.inv()).as_rotvec()
            ang_vel2 = (rot1 * rot2.inv()).as_rotvec()
            self.ang_acc = ang_vel2 - ang_vel3
    
    def predict_pose(self):

        # slam連續正常追蹤，可以使用運動模型
        if self.working:
            
            Tw2 = self.poses[-2]
            Tw1 = self.poses[-1]
            
            T21 = np.linalg.inv(Tw2)@Tw1
            
            Rt = T21
            
            self.Rt = Rt # T21 = T10(Tlc)
        else:
            self.Rt = np.eye(4)
        
        return self.Rt # Tlc

    
class VisualOdometry:
    
    def __init__(self,slam:'Slam'):
        self.slam = slam
        self.map = slam.map
        self.camera = slam.camera
        self.mm = MotionModel()
        
    def match_points(self,f_cur:'Frame',f_ret:'Frame'):
        
        # kps 批配
        kp_idxs1, kp_idxs2, ret = KeypointMatcher.flannMatches(f_cur, f_ret)
        # ret 的內容是座標對
        
        if len(ret)>8: # 雖然EssentialMat 只需要5對點，但是8對點比較穩定
            # 使用成像座標計算本質矩陣 E
            E, mask = cv2.findEssentialMat(
                    f_cur.kps[kp_idxs1], f_ret.kps[kp_idxs2],
                    method=cv2.RANSAC, prob=0.999, threshold=0.02
                )
            # TODO: 加入二次批配、位置最佳化、投影補充批配點
            # mask 表示哪些點是內點。
            # 表示這些點符合 對極幾何約束（epipolar constraint）
            kp_idxs1 = kp_idxs1[mask.ravel() == 1]
            kp_idxs2 = kp_idxs2[mask.ravel() == 1]
            ret = ret[mask.ravel() == 1]
            
            return kp_idxs1, kp_idxs2, ret, E ,True
        else:
            
            return np.array([]), np.array([]), np.array([]),None ,False
        
    '''
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
        matches = window_search(f_last, f_cur, window_size=200, min_scale_level=min_octave, max_scale_level=max_octave)
        nmatches = len(matches)
        vp_map_point_matches = [None] * len(f_cur.kps)
        for ref_idx, cur_idx in matches:
            vp_map_point_matches[cur_idx] = f_last.points[ref_idx]

        # 若配對不足，縮小窗口並移除層級限制再搜尋
        if nmatches < 10:
            matches = window_search(f_last, f_cur, window_size=100, min_scale_level=0, max_scale_level=max_octave)
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
        f_cur.points = np.array(vp_map_point_matches, dtype=object)

        # 若配對足夠，優化 pose 並用投影再搜尋
        if nmatches >= 10:
            # 優化 pose
            self.pose_optimize(f_cur)

            # 移除外點
            for i, outlier in enumerate(f_cur.outliers):
                if outlier:
                    f_cur.points[i] = None
                    f_cur.outliers[i] = False
                    nmatches -= 1

            # 用投影再搜尋（stub，日後可補充）
            # nmatches += self.search_by_projection(last_frame, current_frame, 15, current_frame.points)
        else:
            # 最後機會，用較大窗口投影搜尋（stub，日後可補充）
            # nmatches = self.search_by_projection(last_frame, current_frame, 50, current_frame.points)
            pass

        # 再次設定地圖點
        f_cur.points = np.array(f_cur.points, dtype=object)

        # 若配對仍不足，回傳失敗
        if nmatches < 10:
            return False, nmatches

        # 最後再優化 pose
        self.pose_optimize(f_cur)

        # 再次移除外點

        for i, outlier in enumerate(f_cur.outliers):
            if outlier:
                f_cur.points[i] = None
                f_cur.outliers[i] = False
                nmatches -= 1

        return nmatches >= 10, nmatches
    '''
    
    
    def predict_pose(self,f_cur:'Frame',f_last:'Frame',use_motion_model=False,was_tracking=False):
        
        
        kp_idxs1, kp_idxs2, ret, E ,match_success = self.match_points(f_cur,f_last)
        
        if match_success: # 可以使用位姿恢復
            _, R, t, mask = cv2.recoverPose(E, f_cur.kps[kp_idxs1], f_last.kps[kp_idxs2]) # 恢復位姿，不能準確預測距離
            # t 只代表方向，是一個表示平移方向的單位向量。
            # mask 用於指示哪些點對計算旋轉和平移有貢獻
        
        
        frames = list(self.map.frames)[-6:-1] # -1 位置未知，不應該輸入
        self.mm.update_pose(frames) 
        self.mm.predict_pose(was_tracking) # 運動模型預測，角度沒recoverPose準
        
        if self.mm.working and use_motion_model:# and np.linalg.norm(self.motion_modal.vel) > 1e-3: # use motion model，旋轉角度不準確
            Rt = self.mm.Rt
            if len(ret)>20: # 條件隱含 match_success == True
                Rt[:3, :3] = R # 如果批配點夠多，使用motion model的位置及recoverPose的旋轉
            self.slam.state_display.set_info('predict_pose_model',"motion model")
            # print('using motion model')
        
        
        elif match_success: # use recoverPose 
        
            scale = 1 # 設定距離尺度
            if (self.map.groundtruth is not None) and False:   # 讀取groundtruth 的位移位移量作為 t 的長度，只用於測試，加入BA後要修改成False
                id = f_cur.id
                f_gcur = self.map.groundtruth[id]
                f_glast = self.map.groundtruth[id-1]
                scale = np.linalg.norm(f_gcur[:3,3] - f_glast[:3,3])
                # print('f_gcur',scale)
            
            Rt = np.eye(4) 
            Rt[:3, :3] = R
            Rt[:3, 3] = scale*t.flatten()
            
            self.slam.state_display.set_info('predict_pose_model',"recover pose")
            # print('using recoverPose')
        
        else:
            Rt = np.eye(4)
            self.slam.state_display.set_info('predict_pose_model',"last pose")
        
        
        return kp_idxs1, kp_idxs2, Rt # Tlc
        
    '''    
    def poses_optimize_old(self,local_frames:list['Frame'],max_iterations=10):
        # 只動輸入 frames 的 pose 相關 points 與 frames 固定
        ba = BundleAdjustment()
        
        local_points:set['Point'] = set()
        
        for frame in local_frames:
            
           
            for kp_id,point in enumerate(frame.points):
            
                if point is not None:
                    point:'Point'
                    assert point in self.map.points, point
                    if len(point.frames)>3 and not point.is_bad and (point.kf_num >2): # 只BA有三個以上觀察者的point
                        local_points.add(point)
                
            
            ba.add_pose(frame.id,frame.pose,fixed=False) 
            
        for point in local_points:
            
            ba.add_point(point.id,point.location[:3],fixed=True)
            
            for frame in point.frames:
                if frame in local_frames:
                    kp_id = frame.find_point(point)
                    ba.add_edge(point.id,frame.id,frame.kps[kp_id])
            
        ba.optimize(max_iterations=max_iterations)
        
        # TODO:
        # 多階段最佳化與內點判斷
        # 進行 4 次優化，每次縮小 inlier 門檻（chi2）。
        # 每次最佳化後，根據 chi2 判斷每個邊是否為 outlier，並調整其 level。
        # 只保留 inlier 參與後續最佳化。

        # for point in local_points:
            
        #     ba.get_edge(point.id)

        
        for frame in local_frames:
            frame.pose = ba.get_pose(frame.id)
        
        
        
        
        err = ba.active_chi2()
        return err 
  
    
    def pose_optimize_old(self,frame:'Frame',max_iterations=10):
        
        # 只動輸入 frames 的 pose 相關 points 與 frames 固定
        ba = BundleAdjustment()
        
        local_points:set['Point'] = set()
        
            
           
        for point in frame.points:
        
            if point is not None:
                point:'Point'
                assert point in self.map.points, point
                if len(point.frames)>3 and not point.is_bad and (point.kf_num >2): # 只BA有三個以上觀察者的point
                    local_points.add(point)
            
        
        ba.add_pose(frame.id,frame.pose,fixed=False) 
        
        for point in local_points:
            
            ba.add_point(point.id,point.location[:3],fixed=True)
            

            kp_id = frame.find_point(point)
            ba.add_edge(point.id,frame.id,frame.kps[kp_id])
            
        ba.optimize(max_iterations=max_iterations)
        
        # TODO:
        # 多階段最佳化與內點判斷
        # 進行 4 次優化，每次縮小 inlier 門檻（chi2）。
        # 每次最佳化後，根據 chi2 判斷每個邊是否為 outlier，並調整其 level。
        # 只保留 inlier 參與後續最佳化。
        # for point in local_points:
            
        #     edge = ba.get_edge(point.id,frame.id)
        #     if edge is not None:
        #         print(edge.measurement())
        #         print(edge.chi2())
        
        
        frame.pose = ba.get_pose(frame.id)
        
        
        err = ba.active_chi2()
        return err

    def local_bundle_adjustment_old(self,local_keyframes:list['KeyFrame'],max_iterations=30):
        # local bundle adjustment
        ba = BundleAdjustment()
        local_points:set['Point'] = set()
        neighbor_keyframes:set['KeyFrame'] = set()
        edges = []
        
        # 收集局部地圖點
        # for frame in local_frames.copy():
        for frame in local_keyframes.copy():
            # if frame.points_num <10:
            #     local_keyframes.remove(frame) # 避免添加無/少point 的 frame
            #     continue
            if frame.is_bad:
                local_keyframes.remove(frame)
                continue
            
            for kp_idx,point in enumerate(frame.points):
            
                if point is not None:
                    point:'Point'
                    assert point in self.map.points
                    
                    if len(point.frames)>2 and not point.is_bad and (point.kf_num >2): # 只BA有三個以上觀察者的point
                        local_points.add(point)
                
        # 收集固定幀 將觀測到局部地圖點、但不屬於局部幀的其他幀，作為固定幀加入列表。
        for point in local_points:
            for frame in point.frames:
                if frame not in local_keyframes:
                    neighbor_keyframes.add(frame)
        
        # #debug
        # for frame in local_frames:
        #     assert frame in frames
        
        # 添加幀頂點(vertex)
        for frame in neighbor_keyframes:

            ba.add_pose(frame.id,frame.pose,fixed= True)
        
        for frame in local_keyframes: 
            ba.add_pose(frame.id,frame.pose,fixed= (frame.kid==0)) # 固定起始幀
        
        
        
        # 添加地圖點頂點(vertex)、地圖點與幀的邊(edge)
        for point in local_points:
            ba.add_point(point.id,point.location[:3],fixed=False)

            for frame in point.frames:
                kp_idx = frame.find_point(point)
                assert kp_idx is not None and kp_idx >= 0, f"find_point failed for point {point.id} in frame {frame.id}"
                edge = ba.add_edge(point.id,frame.id,frame.kps[kp_idx])
                
                edges.append((edge,frame,kp_idx,point))
                
        
            
        ba.optimize(max_iterations=max_iterations)
        for frame in neighbor_keyframes:
            if frame in local_keyframes:
                frame.pose = ba.get_pose(frame.id)
            
        for point in local_points:
            point.location = ba.get_point(point.id)
        print(f"local BA pose num: {len(local_keyframes)}, point num: {len(local_points)}")
        err = ba.active_chi2()
        return err
    

    def global_bundle_adjustment_old(self,max_iterations=20):
        
        ba = BundleAdjustment()
        points:set['Point'] = set()
        keyframes:set['KeyFrame'] = set()
        
        # 收集局部地圖點
        for keyframe in keyframes:
            
            for kp_id,point in enumerate(keyframe.points):
            
                if point is not None:
                    point:'Point'
                    assert point in self.map.points
                    if len(point.frames)>2 and not point.is_bad and (point.kf_num >2): # 只BA有三個以上觀察者的point
                        points.add(point)
                
        # 收集固定幀 將觀測到局部地圖點、但不屬於局部幀的其他幀，作為固定幀加入列表。
        for point in points:
            for keyframe in point.keyframes:
                keyframes.add(keyframe)
        
        # # debug
        # for keyframe in local_keyframes:
        #     assert keyframe in keyframes, np.sum(keyframe.points != None)
        
        # 添加幀頂點(vertex)
        for keyframe in keyframes:
            if keyframe in keyframes: 
                ba.add_pose(keyframe.id,keyframe.pose,fixed= keyframe.kid==0) # 固定起始幀
            else : 
                ba.add_pose(keyframe.id,keyframe.pose,fixed= True)
        
        # 添加地圖點頂點(vertex)、地圖點與幀的邊(edge)
        for point in points:
            
            ba.add_point(point.id,point.location[:3],fixed=False)

            for keyframe in point.keyframes:
                kp_id = keyframe.find_point(point)
                assert kp_id is not None and kp_id >= 0, f"find_point failed for point {point.id} in frame {keyframe.id}"
                ba.add_edge(point.id,keyframe.id,keyframe.kps[kp_id])
            
            
        ba.optimize(max_iterations=max_iterations)
        
        for keyframe in keyframes:
            if keyframe in keyframes:
                keyframe.pose = ba.get_pose(keyframe.id)
            
        for point in points:
            point.location = ba.get_point(point.id)
            
            
        print(f"global BA pose num: {len(keyframes)}, point num: {len(points)}")
        err = ba.active_chi2()
        return err
    
    '''
    
    
    
    def pose_optimize(self,frame:'Frame'):
        
        # 只動輸入 frames 的 pose 相關 points 與 frames 固定
        ba = BundleAdjustment()
        
        local_points:set['Point'] = set()
        
        edges = []
        K = frame.camera.K
           
        for point in frame.points:
        
            if point is not None:
                point:'Point'
                assert point in self.map.points, point
                if len(point.frames)>3 and not point.is_bad and (point.kf_num >2): # 只BA有三個以上觀察者的point
                    local_points.add(point)
            
        # print('pose opt local_points',len(local_points))
        ba.add_pose(frame.id,frame.pose,fixed=False) 
        
        for point in local_points:
            
            ba.add_point(point.id,point.location[:3],fixed=True,marginalized=False)
            kp_idx = frame.find_point(point)
            
            
            # 並根據特徵點金字塔層級設 information matrix
            octave = frame.octaves[kp_idx]
            sigma2 = 1.2 ** octave # 根據 orb scaleFactor 設定
            information = np.identity(2) * (1.0 / sigma2)
            measurement = frame.raw_kps[kp_idx]
            
            # invSigma2_pixel = (1.0 / sigma2)
            # information = np.identity(2)
            # information[0,0] = invSigma2_pixel*camera.fx*camera.fx
            # information[1,1] = invSigma2_pixel*camera.fy*camera.fy
            
            # thPixel = np.sqrt(5.991)
            # thNorm = thPixel / ((camera.fx + camera.fy)*0.5)
            # robust_kernel = g2o.RobustKernelHuber(thNorm)
            
            edge = ba.add_edge(point.id,frame.id,measurement,K,information)
            edges.append((edge,kp_idx))
        
        
        # 多階段最佳化與內點判斷
        # 進行 4 次優化，每次縮小 inlier 門檻（chi2）。
        # 每次最佳化後，根據 chi2 判斷每個邊是否為 outlier，並調整其 level。
        # 只保留 inlier 參與後續最佳化。
            
        
        # 多次最佳化，每次縮小 inlier 門檻
        chi2_thresholds = [9.21, 5.99, 3.84, 2.71]  # 依照自由度與置信區間可調整
        its = [10,10,7,5]
        
        for it, chi2_thr in zip(its, chi2_thresholds):
            
            ba.optimize(max_iterations= it)
            # 標記 inlier/outlier
            for (edge,kp_idx) in edges:
                if frame.outliers[kp_idx]:
                    edge.compute_error() # 重新計算該點的重投影誤差
                
                chi2 = edge.chi2()
                
                if chi2 > chi2_thr:
                    frame.outliers[kp_idx] = True
                    edge.set_level(1)
                                    
                elif chi2 <= chi2_thr:
                    frame.outliers[kp_idx] = False
                    edge.set_level(0)

            if len(ba.edges()) <10 :
                break
                
                
        
        frame.pose = ba.get_pose(frame.id)
        
        err = ba.active_chi2()
        
        return err 

    def local_bundle_adjustment(self,local_keyframes:list['KeyFrame']):
        # local bundle adjustment
        ba = BundleAdjustment()
        local_points:set['Point'] = set()
        neighbor_keyframes:set['KeyFrame'] = set()
        
        
        # 收集局部地圖點
        # for frame in local_frames.copy():
        for frame in local_keyframes.copy():
            # if frame.points_num <10:
            #     local_keyframes.remove(frame) # 避免添加無/少point 的 frame
            #     continue
            if frame.is_bad:
                local_keyframes.remove(frame)
                continue
            
            for kp_idx,point in enumerate(frame.points):
            
                if point is not None and not point.is_bad: # 只BA有三個以上觀察者的point
                    point:'Point'
                    assert point in self.map.points
                    if point.dynamic:continue
                    
                    # if not point.is_bad: 
                    local_points.add(point)
                
        # 收集固定幀 將觀測到局部地圖點、但不屬於局部幀的其他幀，作為固定幀加入列表。
        for point in local_points:
            neighbor_keyframes.update(point.keyframes.keys())
        
        for frame in local_keyframes:
            neighbor_keyframes.discard(frame) 
        
        # #debug
        # for frame in local_frames:
        #     assert frame in frames
        
        # 添加幀頂點(vertex)
        for frame in neighbor_keyframes:
            ba.add_pose(frame.id,frame.pose,fixed= True)
        
        for frame in local_keyframes: 
            ba.add_pose(frame.id,frame.pose,fixed= (frame.kid==0)) # 固定起始幀
        
        
        edges:list[tuple[any,'KeyFrame',int,'Point',bool]] = []
        # 添加地圖點頂點(vertex)、地圖點與幀的邊(edge)
        for point in local_points:
            ba.add_point(point.id,point.location_3D,fixed=False)
            
            for frame,kp_idx in point.keyframes.items():
                

                
                K = frame.camera.K
                # kp_idx = frame.find_point(point)
                # assert kp_idx is not None and kp_idx >= 0, f"find_point failed for point {point.id} in frame {frame.id}"
                
                # 並根據特徵點金字塔層級設 information matrix
                octave = frame.octaves[kp_idx]
                sigma2 = 1.2 ** octave # 根據 orb scaleFactor 設定
                information = np.identity(2) * (1.0 / sigma2)
                
                edge = ba.add_edge(point.id,frame.id,frame.raw_kps[kp_idx],K,information)
                outlier=False
                edges.append((edge,frame,kp_idx,point,outlier))
                
        
        # 第一次opt
        ba.optimize(max_iterations=5)
        
        # clean outlier
        for i, (edge,frame,kp_idx,point,outlier) in enumerate(edges):
            
            
            if (edge.chi2() > 5.991) or (not ba.is_depth_positive(edge)):
                
                frame.remove_point(point)                
                ba.remove_edge(edge)
                
                edges[i] = (edge,frame,kp_idx,point,True)
        
        
        # 使地圖與姿態資料反映最新的最佳化結果。
        
        for frame in local_keyframes:
            frame.pose = ba.get_pose(frame.id)
            
        for point in local_points:
            point.location = ba.get_point(point.id)
        
        
        
        ba.optimize(max_iterations=10)
        
        # clean outlier
        for i, (edge,frame,kp_idx,point,outlier) in enumerate(edges):
            
            if outlier: continue
            if point.is_bad: continue
            
            if (edge.chi2() > 5.991) or (not ba.is_depth_positive(edge)):
                frame.remove_point(point)
            
        
        for frame in local_keyframes:
            frame.pose = ba.get_pose(frame.id)
            
        for point in local_points:
            point.location = ba.get_point(point.id)
            point.update_all()
        
        print(f"local BA pose num: {len(local_keyframes)}, point num: {len(local_points)}")
        err = ba.active_chi2()
        return err


    def global_bundle_adjustment(self,max_iterations=20,kf_fixed = False):
        
        ba = BundleAdjustment()
        
        points:set['Point'] = set(self.map.points)
        keyframes:set['KeyFrame'] = set(self.map.keyframes)
        
        max_kid:int = 0
        
        # set keyframe
        for keyframe in keyframes:
            if keyframe.is_bad:continue
            fixed = kf_fixed or keyframe.kid==0
            
            ba.add_pose(keyframe.id,keyframe.pose,fixed= fixed)
            
            if keyframe.kid > max_kid:
                max_kid = keyframe.kid
            
        edges:list[tuple[any,'KeyFrame',int,'Point',bool]] = []
        for point in points:
            if point.is_bad:continue
            if point.dynamic:continue
            if kf_fixed:marginalized=False
            else:marginalized = True
            
            ba.add_point(point.id,point.location_3D,fixed=False,marginalized=marginalized)
            
            for kf,kp_idx in point.keyframes.items():
                if kf.is_bad: continue
                
                K = kf.camera.K
                # kp_idx = kf.find_point(point)
                
                
                # 並根據特徵點金字塔層級設 information matrix
                octave = kf.octaves[kp_idx]
                sigma2 = 1.2 ** octave # 根據 orb scaleFactor 設定
                information = np.identity(2) * (1.0 / sigma2)
                
                edge = ba.add_edge(point.id,kf.id,kf.raw_kps[kp_idx],K,information= information)
                outlier=False
                edges.append((edge,kf,kp_idx,point,outlier))
        
        
        
        ba.optimize(max_iterations=max_iterations)
        
        # # clean outlier
        # for i, (edge,frame,kp_idx,point,outlier) in enumerate(edges):
            
        #     if outlier: continue
        #     if point.is_bad: continue
            
        #     if (edge.chi2() > 5.991) or (not ba.is_depth_positive(edge)):
        #         frame.remove_point(point)
        
                
        for keyframe in keyframes:
            if keyframe.is_bad:continue
            keyframe.pose = ba.get_pose(keyframe.id)
            
        for point in points:
            if point.is_bad:continue
            if point.dynamic:continue
            point.location = ba.get_point(point.id)
            point.update_all()
        
        
        pose_count = sum(1 for v in ba.vertices().values() if v.dimension() == 6)
        point_count = sum(1 for v in ba.vertices().values() if v.dimension() == 3)
        print(f"global BA pose num: {pose_count}, point num: {point_count}")
        
        err = ba.active_chi2()
        return err
        


    