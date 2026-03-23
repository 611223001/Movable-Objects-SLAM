
import cv2
import numpy as np
from enum import Enum
from typing import List

from .pose import triangulate
from .utils import SlamState, add_ones, show_epilines
from .frame import Frame, KeyFrame
from .point import Point
from .feature import FeatureTool
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .slam import Slam

class Initializer():

    def __init__(self,slam:'Slam'):
        
        self.slam =slam
        self.map = slam.map
        self.camera = slam.camera
        self.feature_tool = slam.feature_tool
        self.vo = slam.vo
        self.tracker = slam.tracker
        self.map_display = slam.map_display
        self.state_display = slam.state_display
        self.control = slam.control
        
        
        # 屬性: frame 
        self.f_cur:Frame
        self.f_last:Frame
        
        self.kf_cur:KeyFrame
        self.kf_last:KeyFrame
        self.f_ref:Frame|KeyFrame# f_ref 可能是f_last或kf_cur，用於與f_cur比較kps
        
    
    def initialize1(self, img):
        """
        
        """
        self.feature_tool.set_orb_params(nfeatures=2000)
        # 檢查特徵點數量
        frame = Frame(self.slam, img)
        if len(frame.kps) > 100:
            self.slam.set_state(SlamState.INITIALIZING)
            
            self.f_cur = frame
            self.f_last = frame
            # self.f_last = Frame(self.slam, img)
            # self.mvb_prev_matched = [tuple(pt) for pt in frame.kps]  # 儲存上一幀的 keypoint 座標
            
            return True
        else:
            return False
    
    def initialize2(self, img):
        """
        
        """
        frame = Frame(self.slam, img)
        self.f_cur = frame
        
        # ##### 記得刪除 加速用
        # if frame.id <20:
        #     return False
        # #####

        # 檢查特徵點數量
        if len(self.f_cur.kps) <= 100:
            # self.mv_ini_matches = [-1] * len(self.f_cur.kps)
            # self.slam.set_state(SlamState.NOT_INITIALIZED)
            return False
        
        # 找初始匹配
        # nmatches, kp_idxs1, kp_idxs2 = search_for_initialization(self.f_last, self.f_cur, 100)
        
        # 使用光流找初始匹配
        dis = cv2.DISOpticalFlow.create(2)
        img_cur = cv2.cvtColor(self.f_cur.img,cv2.COLOR_BGR2GRAY)
        img_last = cv2.cvtColor(self.f_last.img,cv2.COLOR_BGR2GRAY)
        flow = dis.calc(img_cur,img_last, None,)
        # flow = dis.calc(img_last,img_cur, None,)
        
        nmatches, kp_idxs1, kp_idxs2 = search_kps_by_flow(self.f_last, self.f_cur, flow, radius_factor=7,nn_ratio=0.9,distance_threshold=30,only_frist_octave=True)

        
        
        # 判斷像素移動量（視差）是否足夠
        pts1 = self.f_last.raw_kps[kp_idxs1]
        pts2 = self.f_cur.raw_kps[kp_idxs2]
        pixel_shifts = np.linalg.norm(pts1 - pts2, axis=1)
        mean_shift = np.mean(pixel_shifts)
        median_shift = np.median(pixel_shifts)
        print(f"初始化匹配平均像素移動量: {mean_shift:.2f}, 中位數: {median_shift:.2f}, 批配特徵點:{nmatches}/{len(self.f_cur.kps)}")
        
        canvas = show_matches(self.f_last.img, self.f_cur.img, self.f_last.raw_kps, self.f_cur.raw_kps, kp_idxs1, kp_idxs2)
        self.map.slam.map_display.display_info('img',('Init Matches', canvas),self.f_cur.id)
        # cv2.imshow('Init Matches', canvas)
        # cv2.waitKey(0)
        
        # 7 10
        if median_shift < 7 or nmatches < 100:  # 閾值依場景調整
            return False
        
        print('start')
        
        self.map.slam.map_display.display_info('clean',('Init Matches'),self.f_cur.id)
        # if cv2.getWindowProperty('Init Matches', cv2.WND_PROP_VISIBLE) >= 1:
        #     cv2.destroyWindow('Init Matches')

        
        
        # 初始化（計算 Rcw, tcw, 3D點等）
        success, Tlc, pts4d, good_pts = self.init_pose_triangulate(self.f_cur, kp_idxs1, kp_idxs2)
        
        
        if success:
            
            self.create_initial_map(Tlc, pts4d, kp_idxs1, kp_idxs2,good_pts)
            clear_pt_num = self.map.clear_bad_points(self.kf_cur)
            print('initial  clear_pt_num:',clear_pt_num)
            
            self.tracker.f_cur = self.f_cur
            self.tracker.f_last = self.f_last
            
            self.tracker.kf_cur = self.kf_cur
            self.tracker.kf_last = self.kf_last
            # self.tracker.last_reloc_frame_id = self.f_cur.id
            self.map.update_covisibility_graph()
            self.feature_tool.set_orb_params(nfeatures=1000)
            self.state_display.set_info('pt_num',len(self.map.points))
            # 更新狀態
            self.slam.set_state(SlamState.WORKING)
            return True
        else:
            # self.slam.set_state(SlamState.NOT_INITIALIZED)
            return False
    
    
    def init_pose_triangulate(self, f2:'Frame', kp_idxs1, kp_idxs2, max_iterations=200):
        """
        根據初始匹配，估算 R, t 並三角化 3D 點。
        Args:
            f1: 參考 Frame
            f2: 當前 Frame current_frame
            kp_idxs1: 參考幀的 keypoint 索引
            kp_idxs2: 當前幀的 keypoint 索引
            K: 相機內參
            max_iterations: RANSAC 次數
        Returns:
            success: 是否初始化成功
            R: 旋轉矩陣
            t: 平移向量
            pts3d: 三角化後的 3D 點
            good_pts: 每個匹配是否成功三角化
        """
        f1:'Frame' = self.f_last
        
        # 準備匹配點
        pts1 = f1.kps[kp_idxs1]
        pts2 = f2.kps[kp_idxs2]
        
         
        
        # 使用成像座標計算本質矩陣 E
        E, mask = cv2.findEssentialMat(
                    f1.kps[kp_idxs1], f2.kps[kp_idxs2], 
                    method=cv2.RANSAC, prob=0.999, threshold=0.02
                )
        
        # mask 表示哪些點是內點。
        # 表示這些點符合 對極幾何約束（epipolar constraint）
        kp_idxs1 = kp_idxs1[mask.ravel() == 1]
        kp_idxs2 = kp_idxs2[mask.ravel() == 1]
        
        print(len(mask),'Essential mask',np.sum(mask.ravel() == 1))
        # show_epilines(f1.img, f2.img,f1.raw_kps[kp_idxs1],f2.raw_kps[kp_idxs2],E  ,f1.camera.K)
        
        
        _, R, t, mask = cv2.recoverPose(E, f1.kps[kp_idxs1], f2.kps[kp_idxs2]) # 恢復位姿，不能準確預測距離
        # X2 = R21 @ X1 + t21
        # Rcl,tcl
        
        
        Tcl = np.eye(4)
        Tcl[:3, :3] = R
        Tcl[:3, 3] = t.reshape(3)
        
        Tlc = np.linalg.inv(Tcl)
        pts4d = triangulate(np.eye(4), Tlc, pts1, pts2)
        
        # Tlc = np.eye(4)
        # Tlc[:3, :3] = R
        # Tlc[:3, 3] = t.reshape(3)
        
        # pts4d = triangulate(np.eye(4), Tlc, pts1, pts2)
        
        # P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        # P2 = np.hstack((R, t.reshape(3, 1)))
        # pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T
        
        good_pts = (np.abs(pts4d[:, 3]) > 0.0005) # 0005
        print('good_pts',np.sum(good_pts))
        
        pts4d /= pts4d[:, 3:]
        vb_triangulated = pts4d[:, 2] > 0
        print(f'pts4d:{len(pts4d)} w>0:{np.sum(good_pts)} visable:{np.sum(vb_triangulated)}')
        
        good_pts = vb_triangulated & good_pts
        print(f'pts4d good:{np.sum(good_pts)}')
        
        success = np.sum(good_pts) > 30
        

        # # RANSAC 計算 Homography 和 Fundamental
        # H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        # F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)

        # # 計算 inlier 分數
        # SH = np.sum(mask_H) if mask_H is not None else 0
        # SF = np.sum(mask_F) if mask_F is not None else 0
        # RH = SH / (SH + SF + 1e-6)

        # # 根據分數選擇重建方式
        # if RH > 0.40:
        #     success, R, t, pts3d, vb_triangulated = self.reconstruct_from_homography(pts1, pts2, H, K)
        # else:
        #     success, R, t, pts3d, vb_triangulated = self.reconstruct_from_fundamental(pts1, pts2, F, K)

        # print(Tlc)
        return success, Tlc, pts4d, good_pts

    

    
    def create_initial_map(self,Tlc, pts4d, kp_idxs1, kp_idxs2,good_pts):
        """
        建立初始地圖
        Args:
            Rcw: 旋轉矩陣
            tcw: 平移向量
            ini_p3d: 三角化得到的 3D 點 (N, 3)
            kp_idxs1: 參考幀的 keypoint 索引
            kp_idxs2: 當前幀的 keypoint 索引
        """
        # 設定初始 KeyFrame pose
        self.f_last.pose = self.map.pose0 # Twl
        

        Twc = self.map.pose0 @ Tlc
        self.f_cur.pose = Twc
        # Rt 代表從 f_last 坐標系到 f_cur 坐標系的轉換。
        # 將 Rt 與 f_last.pose 相乘，就可以得到一個新的變形，直接將世界座標系映射到 f_cur 的座標系。
        # Tw1 = Tw2 * T21
        

        # 建立 KeyFrame
        kf_last = KeyFrame(self.f_last)
        kf_cur = KeyFrame(self.f_cur)
        self.f_last = kf_last
        self.f_cur = kf_cur
        self.kf_last = kf_last
        self.kf_cur = kf_cur

        # 建立 MapPoint 並關聯到 KeyFrame
        assert len(good_pts)==len(kp_idxs1)and len(good_pts)==len(kp_idxs2),(len(good_pts),len(kp_idxs1),len(kp_idxs2))
        for i, (kp_idx1, kp_idx2) in enumerate(zip(kp_idxs1, kp_idxs2)):
            # if kp_idx1<0 or kp_idx2 < 0:
            #     continue
            if not good_pts[i]:continue
            
            x, y = kf_last.raw_kps[kp_idx1]
            color= kf_last.img[y][x][::-1] # [::-1] 會將陣列反轉順序，BGR 格式轉成 RGB
            # world_pos = ini_p3d[i]
            loc = pts4d[i]
            # pt = Point(self.slam, loc, color=color, frame=kf_cur, idx=kp_idx2)
            # pt.add_frame(kf_last,kp_idx1)
            # 關聯到兩個 KeyFrame
            pt = Point(self.slam, loc, color=color, frame=kf_last, idx=kp_idx1)
            pt.add_frame(kf_cur,kp_idx2)
        
        
        # Bundle Adjustment
        err = self.vo.global_bundle_adjustment(40,False)
        print(f"New Map created with {len(self.map.points)} points, GBA err:{err}")

        # 設定中位深度為 1
        median_depth = kf_last.compute_points_median_depth()
        
        
        
        # if median_depth <= 0 or kf_cur.points_num < 100:
        #     print("Wrong initialization, reseting...")
        #     self.slam.reset()
        #     return

        inv_median_depth = (1.0 / median_depth)*100

        # 縮放 KeyFrame pose
        kf_cur.pose[:3, 3] *= inv_median_depth

        # 縮放所有地圖點
        for pt in self.map.points:
            pt.location *= inv_median_depth

        # 設定參考 KeyFrame
        self.map.kf_cur = kf_cur

    def initialize_old(self, current_frame:'Frame', kp_idxs1, kp_idxs2, max_iterations=200):
        """
        根據初始匹配，估算 R, t 並三角化 3D 點。
        Args:
            current_frame: 當前 Frame
            kp_idxs1: 參考幀的 keypoint 索引
            kp_idxs2: 當前幀的 keypoint 索引
            K: 相機內參
            max_iterations: RANSAC 次數
        Returns:
            success: 是否初始化成功
            R: 旋轉矩陣
            t: 平移向量
            pts3d: 三角化後的 3D 點
            vb_triangulated: 每個匹配是否成功三角化
        """
        # 準備匹配點
        pts1 = self.f_last.raw_kps[kp_idxs1].astype(float)
        pts2 = current_frame.raw_kps[kp_idxs2].astype(float)
        # pts1 = pts1.astype(float)
        # pts2 = pts2.astype(float)
        
        
        K = self.f_last.camera.K

        # RANSAC 計算 Homography 和 Fundamental
        H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)

        # 計算 inlier 分數
        SH = np.sum(mask_H) if mask_H is not None else 0
        SF = np.sum(mask_F) if mask_F is not None else 0
        RH = SH / (SH + SF + 1e-6)

        print(H,F)
        # 根據分數選擇重建方式
        if RH > 0.40:
            success, R, t, pts4d, vb_triangulated = self.reconstruct_from_homography(pts1, pts2, H, K)
        else:
            success, R, t, pts4d, vb_triangulated = self.reconstruct_from_fundamental(pts1, pts2, F, K)

        print((H,F))
        Tcl = np.eye(4)
        Tcl[:3, :3] = R
        Tcl[:3, 3] = t.reshape(3)
        Tlc = np.linalg.inv(Tcl)
        
        return success, Tlc, pts4d, vb_triangulated

    def reconstruct_from_homography(self, pts1, pts2, H, K):
        """
        根據 Homography 重建 R, t, 3D 點
        """
        # 這裡需根據 H 分解出 R, t，並三角化
        # 可用 cv2.decomposeHomographyMat 或自訂分解
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
        # 選擇最合理的解（可根據 cheirality 或其他條件）
        R = rotations[0]
        t = translations[0]
        # 三角化
        print(R,t)
        pts4d, vb_triangulated = self.triangulate_points(pts1, pts2, R, t, K)
        success = np.sum(vb_triangulated) > 10
        return success, R, t, pts4d, vb_triangulated

    def reconstruct_from_fundamental(self, pts1, pts2, F, K):
        """
        根據 Fundamental 重建 R, t, 3D 點
        """
        # 計算本質矩陣
        E = K.T @ F @ K
        # 分解本質矩陣
        R1, R2, t = cv2.decomposeEssentialMat(E)
        # 選擇最合理的解（可根據 cheirality 或其他條件）
        R = R1
        # 三角化
        print(R,t)
        pts4d, vb_triangulated = self.triangulate_points(pts1, pts2, R, t, K)
        success = np.sum(vb_triangulated) > 10
        return success, R, t, pts4d, vb_triangulated
    

    def triangulate_points(self, pts1, pts2, R, t, K):
        """
        三角化匹配點，回傳 3D 點與成功標記
        """
        # 構造投影矩陣
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        pts4d_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts4d = (pts4d_h / pts4d_h[3]).T
        # 檢查 cheirality（點是否在相機前方）
        vb_triangulated = pts4d[:, 2] > 0
        return pts4d, vb_triangulated
    
    
    
    
    
    
    def predict_points_in_init(self,f1:'Frame',f2:'Frame',kp_idxs1,kp_idxs2):

        point_counter = 0
        # # 拒絕沒有足夠「視差」的點
        # medianDepth = f2.compute_points_median_depth()
        # baseline = np.linalg.norm(f1.Ow-f2.Ow)
        # ratioBaselineDepth = baseline/medianDepth
        
        # if ratioBaselineDepth < 0.05:
        #     return point_counter
        
        # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
        #predict pose
        pts4d = triangulate(f1.pose, f2.pose, f1.kps[kp_idxs1], f2.kps[kp_idxs2])
        
        # This line normalizes the 3D points by dividing each row by its fourth coordinate W
        # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates


        # Reject points without enough "Parallax" and points behind the camera
        # checks if the absolute value of the fourth coordinate W is greater than 0.005.
        # checks if the z-coordinate of the points is positive.
        # returns, A boolean array indicating which points satisfy both criteria.
        # 檢查第四座標 W 的絕對值是否大於 0.005。 避免太接近零（除W會出現極大值）
        # 返回，一個布林陣列，表示哪些點符合這兩個條件。
        good_pts4d = (np.abs(pts4d[:, 3]) > 0.005)
        pts4d /= pts4d[:, 3:]
        
        visible1,  pts_proj1 = f1.are_visible(pts4d)
        visible2,  pts_proj2 = f2.are_visible(pts4d)

        # print('good_pts4d 1',np.sum(good_pts4d))
        good_pts4d = good_pts4d & visible1 & visible2
        # print('good_pts4d 2',np.sum(good_pts4d))
        

        for i, (p_position, kp_idx1, kp_idx2) in enumerate(zip(pts4d,kp_idxs1,kp_idxs2)):

            #  If the point is not good (i.e., good_pts4d[i] is False), the loop skips the current iteration and moves to the next point.
            if not good_pts4d[i]:continue
            
            point_counter +=1
            if f1.points[kp_idx1] is not None: continue
            
            # # check reprojection error
            # err1 = pts_proj1[i][:2] -f1.kps[kp_idx1]
            # err2 = pts_proj2[i][:2] -f2.kps[kp_idx2]
            # err1 =np.sum(err1**2)
            # err2 =np.sum(err2**2)
            # # print('err',err1,err2)
            # if (err1 >1) | (err2>1): continue # 閾值隨便設定， TODO:修改成一個有意義的值
            
            if f2.points[kp_idx2]is None:
                # 建立新 point
                x, y = f1.raw_kps[kp_idx1]
                color= f1.img[y][x][::-1] # [::-1] 會將陣列反轉順序，BGR 格式轉成 RGB
                pt = Point(f2.slam, p_position,color, frame=f2, idx=kp_idx2)
            
            else:
                # 使用追蹤中的 point
                pt = f2.points[kp_idx2]
            
            self.map.add_point_frame_relation(pt,f1,kp_idx1)
            
        return point_counter
    """
    # def initialize(self,img):
        
    #     self.slam.set_state(SlamState.INITIALIZING)
        
    #     self.feature_tool.set_orb_params(nfeatures=900) #使用3倍特徵點
    #     frame = Frame(self.slam, img)
        
    #     frame.pose = self.map.pose0 # 根據設定，起始位姿可能是groundtruth的起始位姿
    #     frame = KeyFrame(frame)
    #     self.kf_cur = frame
    #     self.f_cur = frame
    #     self.slam.obj_tool.predict_object(self.f_cur)
        
    #     return frame
    
    # def initialize2(self,img):
    #     self.f_last = self.f_cur
    #     frame = Frame(self.slam, img,self.f_last.pose)
    #     self.f_cur = frame
        
    #     for kf in self.map.keyframes:
    #         kp_idxs1, kp_idxs2, ret, E ,match_success = self.vo.match_points(self.f_cur,kf)
    #         if match_success:
    #             self.track_2D_2D(self.f_cur,kf,kp_idxs1,kp_idxs2)
            
    #     kp_idxs1, kp_idxs2, Rt = self.vo.predict_pose(self.f_cur, self.f_last)
    #     self.track_2D_2D(self.f_cur,self.f_last,kp_idxs1,kp_idxs2)
    #     self.f_cur.pose = np.dot(self.f_last.pose,Rt)
        
        
        
    #     # TODO: 加入判斷與初始幀變化夠大
    #     # img_is_blurry,_ = is_blurry(img,300) # 可以另外設定閾值
    #     # if not img_is_blurry:
    #     # if frame.is_clear:
    #     frame = KeyFrame(frame)
            
    #     self.f_cur = frame
    #     self.kf_cur = frame
    #     self.vo.pose_optimize(frame)
    #     frame.clear_outliers()
        
    #     pt_num = self.predict_points_in_init(self.f_cur,self.f_last,kp_idxs1,kp_idxs2)
        
    #     print(f'creat {pt_num} points')
        
        
    #     err = self.vo.local_bundle_adjustment([self.kf_cur])
            
    #     # self.track_3D_2D(self.f_cur,self.map.points)
        
    #     self.slam.obj_tool.track_object_by_OpticalFlow(self.f_cur,self.f_last)
    #     self.slam.obj_tool.predict_object(self.f_cur)
        
    #     medianDepth = self.kf_cur.compute_points_median_depth()
    #     baseline = np.linalg.norm(self.kf_cur.Ow-self.map.keyframes[0].Ow)
    #     ratioBaselineDepth = baseline/medianDepth # 最小視差根據f與場景的距離改變
    #     if ratioBaselineDepth > 0.1 or True:
            
            
    #         self.feature_tool.set_orb_params()  # 恢復使用正常數量特徵點
    #         self.slam.set_state(SlamState.WORKING)
            
    #         err = self.vo.global_bundle_adjustment(self.map.keyframes)
            
    #         self.track_3D_2D(self.f_cur,self.map.points)
            
            
    #     self.tracker.f_cur = self.f_cur
    #     self.tracker.f_last = self.f_last
        
    #     self.tracker.kf_cur = self.kf_cur
    #     self.tracker.kf_last = self.kf_last
    #     self.tracker.f_ref = self.f_ref
            

    #     return frame
    """
    
    
        
def show_matches(img1, img2, raw_kps1, raw_kps2, kp_idxs1, kp_idxs2, max_show=400):
    # img1, img2: 原始影像
    # kps1, kps2: keypoints (N,2)
    # kp_idxs1, kp_idxs2: 匹配索引
    # max_show: 最多顯示多少條線
    # import cv2
    # import numpy as np
    # print(kps1[:10])
    # print(kps2[:10])
    # print(kp_idxs1[:10])
    # print(kp_idxs2[:10])


    # 建立一張左右拼接的影像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # 隨機選 max_show 條線
    idxs = np.arange(len(kp_idxs1))
    if len(idxs) > max_show:
        idxs = np.random.choice(idxs, max_show, replace=False)

    for i in idxs:
        pt1 = tuple(np.round(raw_kps1[kp_idxs1[i]]).astype(int))
        pt2 = tuple(np.round(raw_kps2[kp_idxs2[i]]).astype(int) + np.array([w1, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)
        # print(f"pt1: {pt1}, pt2: {pt2}, color: {color}")
        
    return canvas


def compute_reprojection_errors(pts3d, kps1, kps2, K, R, t):
    """
    計算每個三角化點的投影誤差
    Args:
        pts3d: (N, 3) 三角化後的3D點
        kps1: (N, 2) 第一幀的像素座標
        kps2: (N, 2) 第二幀的像素座標
        K: (3, 3) 相機內參
        R, t: 第二幀相對第一幀的旋轉和平移
    Returns:
        errors1: (N,) 投影到第一幀的誤差
        errors2: (N,) 投影到第二幀的誤差
    """
    N = pts3d.shape[0]
    pts3d_h = np.hstack([pts3d, np.ones((N, 1))])  # (N, 4)
    # 第一幀投影矩陣
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # 第二幀投影矩陣
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    # 投影到第一幀
    proj1 = (P1 @ pts3d_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:]
    # 投影到第二幀
    proj2 = (P2 @ pts3d_h.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:]
    # 計算誤差
    errors1 = np.linalg.norm(proj1 - kps1, axis=1)
    errors2 = np.linalg.norm(proj2 - kps2, axis=1)
    return errors1, errors2


from .matcher import descriptor_distance

def search_kps_by_flow(f1:'Frame', f2:'Frame', flow, radius_factor:float=1.0, nn_ratio:float=0.8, distance_threshold:int=100,only_frist_octave=False)->int:
        """
        
        Args:
            f1: current Frame
            f2: last Frame
            flow: dis.calc(f2,f1, None,)
            radius_factor: 搜尋半徑縮放因子
            nn_ratio: 最近鄰比值
            distance_threshold: Hamming 距離閾值
        Outs:
            nmatches: 匹配數量
            kp_idxs1: ...
            kp_idxs2: ...
        """
        nmatches = 0
        matches = []
        kp_idxs1 = []
        kp_idxs2 = []
        
        for idx2, _ in enumerate(f2.points):
            
            if only_frist_octave:
                if f2.octaves[idx2] > 0:
                    continue
            
            
            # 取得光流對應位置
            u, v = f2.raw_kps[idx2]
            uu,vv = flow[v,u]
            uv = (u+uu, v+vv)
            
            
            # 根據點的尺度調整搜尋半徑
            scale_level = int(f2.octaves[idx2])
            base_radius = f2.sizes[idx2]
            radius = base_radius * radius_factor
            
            # KDTree搜尋半徑內的keypoint
            near_indices = f1.raw_kd.query_ball_point(uv, radius)
            
            
            
            # octave在合理範圍的keypoint
            if only_frist_octave:
                near_indices = [idx1 for idx1 in near_indices
                    if (0 == f1.octaves[idx1])]
            else:
                near_indices = [idx1 for idx1 in near_indices
                                if (scale_level >= f1.octaves[idx1] and f1.octaves[idx1] >= scale_level-1)]
                
            
            if not near_indices:
                continue

            # 找最佳描述子匹配
            best_dist = float('inf')
            best_dist2 = float('inf')
            best_idx = -1
            
            des_last = f2.des[idx2]
            for idx1 in near_indices:
                des_cur = f1.des[idx1]
                dist = descriptor_distance(des_last, des_cur)
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
                # last_idx = f_last.find_point(point)
                
                matches.append((best_idx,idx2))
                kp_idxs1.append(best_idx)
                kp_idxs2.append(idx2)
                
                nmatches += 1
                
        kp_idxs1 = np.array(kp_idxs1)
        kp_idxs2 = np.array(kp_idxs2)
        
        return nmatches, kp_idxs1, kp_idxs2


def search_for_initialization(f_ref:'Frame', f_cur:'Frame', window_size:int=100, nn_ratio:float=0.9, distance_threshold:int=50, histo_length:int=30, check_orientation:bool=True):
    """
    仿照 ORB-SLAM SearchForInitialization，於 f_ref 的每個 keypoint 在 f_cur window 內搜尋最佳匹配。
    Args:
        f_ref: 參考幀 (Frame)
        f_cur: 目標幀 (Frame)
        window_size: 搜尋窗口大小 (像素)
        nn_ratio: 最近鄰比值
        distance_threshold: Hamming 距離閾值
        histo_length: 旋轉直方圖 bin 數
        check_orientation: 是否進行旋轉一致性檢查
    Returns:
        nmatches: 匹配數量
        kp_idxs1: 參考幀 keypoint 索引
        kp_idxs2: 當前幀 keypoint 索引
    """
    N_ref = len(f_ref.raw_kps)
    N_cur = len(f_cur.raw_kps)
    vn_matches12 = [-1] * N_ref
    v_matched_distance = [float('inf')] * N_cur
    vn_matches21 = [-1] * N_cur
    rot_hist = [[] for _ in range(histo_length)]
    factor = 1.0 / histo_length
    nmatches = 0

    for i1 in range(N_ref):
        # 只用金字塔第0層
        if f_ref.octaves[i1] > 0:
            continue

        # pt1 = prev_matched[i1] if prev_matched is not None else f_ref.kps[i1]
        pt1 = f_ref.raw_kps[i1]
        # 只搜尋同層級且在 window 內的 keypoint
        indices2 = [i2 for i2 in range(N_cur)
                    if np.linalg.norm(f_cur.raw_kps[i2] - pt1) < window_size
                    and f_cur.octaves[i2] == 0]

        if not indices2:
            continue

        d1 = f_ref.des[i1]
        best_dist = float('inf')
        best_dist2 = float('inf')
        best_idx2 = -1

        for i2 in indices2:
            d2 = f_cur.des[i2]
            dist = np.count_nonzero(d1 != d2)
            if v_matched_distance[i2] <= dist:
                continue
            if dist < best_dist:
                best_dist2 = best_dist
                best_dist = dist
                best_idx2 = i2
            elif dist < best_dist2:
                best_dist2 = dist

        if best_dist <= distance_threshold:
            if best_dist < nn_ratio * best_dist2:
                # 若已被其他點配對，則取消舊配對
                # if vn_matches21[best_idx2] >= 0:
                #     vn_matches12[vn_matches21[best_idx2]] = -1
                #     nmatches -= 1
                # vn_matches12[i1] = best_idx2
                # vn_matches21[best_idx2] = i1
                # v_matched_distance[best_idx2] = best_dist
                # nmatches += 1
                
                if vn_matches21[best_idx2] >= 0:
                    old_i1 = vn_matches21[best_idx2]
                    vn_matches12[old_i1] = -1
                    vn_matches21[best_idx2] = -1
                    nmatches -= 1
                vn_matches12[i1] = best_idx2
                vn_matches21[best_idx2] = i1
                v_matched_distance[best_idx2] = best_dist
                nmatches += 1

                if check_orientation:
                    angle1 = f_ref.angles[i1]
                    angle2 = f_cur.angles[best_idx2]
                    rot = angle1 - angle2
                    if rot < 0.0:
                        rot += 360.0
                    bin_idx = int(round(rot * factor)) % histo_length
                    rot_hist[bin_idx].append(i1)

    # 可選：旋轉一致性檢查（如需可補充）

    # 回傳匹配索引
    kp_idxs1 = np.array([i1 for i1, i2 in enumerate(vn_matches12) if i2 >= 0])
    kp_idxs2 = np.array([vn_matches12[i1] for i1 in kp_idxs1])
    return nmatches, kp_idxs1, kp_idxs2
