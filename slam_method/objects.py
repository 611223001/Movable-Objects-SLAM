
import numpy as np
import cv2
from collections import Counter,defaultdict
import open3d as o3d
from skimage.measure import marching_cubes
import sys

from .object_utils import *

from .feature import FeatureTool
from .matcher import ObjectPointMatcher
from .utils import normalize
from .mesh import Mesh, VoxelGrid
from .pose import Pose
from .matcher import ObjectMatcher

import time
from typing import TYPE_CHECKING,Union
if TYPE_CHECKING:
    from .map import Map
    from .frame import Frame, KeyFrame
    from .point import Point
    from .slam import Slam

class ObjectBase(Pose):
    _id_counter = 0
    def __init__(self):
        """包含 thing|stuff 的共用函式、屬性
        """
        self.map:'Map'
        
        self.id:int = None # thing stuff 分開計
        self.oid:int = None # thing stuff 統一計

        self.is_thing:bool =None
        
        self.category_id:int = None
        self.category_ids:list[int] = []
        self.category_ids_scores:list[float]
        
        
        self.points:set['Point'] = set() # 避免重複添加point
        self.frames:defaultdict['Frame',set[int]] = defaultdict(set)    # frame:set[idx]  一個frame的多個遮罩可能會判斷成同一物體
        self.keyframes:defaultdict['KeyFrame',set[int]] = defaultdict(set)
        
        # hull
        self.volume:o3d.pipelines.integration.ScalableTSDFVolume = None
        self.vertices:np.ndarray = np.zeros((0,3), dtype=np.float64)    # vertix 面的頂點 (n,3)[x,y,z]
        self.faces:np.ndarray = np.zeros((0,3), dtype=np.int32)         # 3個點組成的三角形面 (n,3)
        
        self.planes:list[tuple[np.ndarray,o3d.geometry.PointCloud]] # list[(plane,inlier_cloud)]
                                                                    # plane: (4,) array，格式為 [a, b, c, d]，表示 ax+by+cz+d=0 的平面
                                                                    # inlier_cloud: o3d.geometry.PointCloud, 包含屬於此平面的點的座標
        # self.plane:None # 無限、無界的平面
        # self.edges:None # 由點組成的線集合
        
        self.dynamic:bool = False # 現在物體是否在移動
        self.dynamic_time:int = None # 如果移動，紀錄移動時的f.id
        self.static_time:int = 0 # 如果移動結束，紀錄固定時的f.id

        self.is_bad:bool = False
        # self.fused_by:'Thing'|'Stuff'|None
        self.fused_by: Union['Thing','Stuff','None'] = None
    
    @staticmethod
    def _temp_obj(frame:'Frame',idx_obj:int):
        """類似Thing|Stuff，但是並未儲存到map與frame，如果沒有與 Thing|Stuff fuse會被遺忘

        Args:
            frame (Frame): _description_
            idx_obj (int): _description_
        """
        
        temp_obj = ObjectBase()
        
        info = frame.obj_infos[idx_obj]
        if info['isthing']:
            temp_obj.is_thing = True
            temp_obj.category_ids_scores = []
        else:
            temp_obj.is_thing = False
        
        temp_obj.map = frame.map
        temp_obj.map.add_obj_frame_relation(temp_obj,frame,idx_obj,True)
        
        # temp_obj.add_frame(frame,idx_obj)
        return temp_obj
        
        
    @property
    def center_location(self):
        # [𝑋, 𝑌, 𝑍, 𝑊=1] 齊次化的座標
        if len(self.points) == 0:return np.array([0,0,0,0])
        
        locations = [p.location for p in self.points]
        mean = np.mean(locations, axis=0)
        return mean
            
    @property
    def location_3D(self):
        # [𝑋, 𝑌, 𝑍] 取得非齊次的座標
        if len(self.points) == 0:return np.array([0,0,0])
        
        locations = [p.location_3D for p in self.points]
        mean = np.mean(locations, axis=0)
        return mean
    
    @property # 暫時用於顯示物體大概大小
    def radius(self):
        if len(self.points) == 0:
            return 0.0
        center = self.center_location
        locations = [p.location for p in self.points]
        dists = [np.linalg.norm(loc - center) for loc in locations]
        return np.mean(dists)
    
    
    def mask(self,frame:'Frame'):
        
        mask = np.zeros((frame.camera.height,frame.camera.width),np.bool_)
        
        if not frame in self.frames:
            return mask
        
        idxs = self.frames[frame]
        
        
        for idx in idxs:
            mask_ = frame.obj_infos[idx]['mask']
            mask = np.logical_or(mask,mask_)
        
        return mask

    def add_frame(self,frame:'Frame',idx_obj:int):
        self.map.add_obj_frame_relation(self,frame,idx_obj)
    
    def remove_frame(self,frame:'Frame'):
        idxs = self.frames[frame].copy()
        for idx in idxs:
            self.map.remove_obj_frame_relation(self,frame,idx)
        return idxs
    
    def add_point(self,point:'Point'):
        self.map.add_point_object_relation(point,self)
    
    def remove_point(self,point:'Point'):
        self.map.remove_point_object_relation(point,self)
        
    def add_points(self,points:list['Point']):
        self.map.add_points_object_relation(points,self)
    
    def remove_points(self,points:list['Point']):
        self.map.remove_points_object_relation(points,self)
        
    
    def cull_bad_points(self):
        # 找到不符合此物件的點，並移除物件和點的關聯
        # TODO
        ...
        bad_points = []
        for bad_point in bad_points:
            self.remove_point(bad_point,self)
    
    def update_points_by_frames(self,only_keyframe=True):
        """
        通過擁有的frames 或 keyframes 更新擁有的points。
        只會使用靜態後的幀。如果處於動態，只會使用動態前的幀。

        Args:
            only_keyframe (bool, optional): _description_. Defaults to True.
        """
        
        
        if only_keyframe:
            frames = self.keyframes
        else:
            frames = self.frames
        
        self.remove_points(self.points) # 清空points資料
        
        
        for frame,idxs_obj in frames.items():
            if frame.id < self.static_time: continue # 只使用靜態後的幀
            
            if self.dynamic and frame.id > self.dynamic_time:continue # 如果動態，使用動態前的幀
                
            
            for idx_obj in idxs_obj:
                idxs_pt = list(frame.obj_to_pts[idx_obj])
                points = frame.points[idxs_pt]
                self.add_points(points)
                
                # for idx_pt in frame.obj_to_pts[idx_obj]:
                #     point = frame.points[idx_pt]
                #     if point is None:continue
                    
                #     self.add_point_object_relation(point,obj)
    
    def update_points_by_frame(self,frame:'Frame'):
        """
        通過frame 更新擁有的points，frame 必須在obj.frames
        
        Args:
            frame
        """
        
        if not frame in self.frames:
            return
            
        idxs_obj = self.frames[frame]
        
        for idx_obj in idxs_obj:
            idxs_pt = list(frame.obj_to_pts[idx_obj])
            points = frame.points[idxs_pt]
            self.add_points(points)
            
            # for idx_pt in frame.obj_to_pts[idx_obj]:
            #     point = frame.points[idx_pt]
            #     if point is None:continue
                
            #     self.add_point_object_relation(point,obj)
    
    # def make_mesh(self,
    #           kf_cur: 'KeyFrame',
    #           ):
        
    #     voxel_size = self.map.voxel_size
    #     sdf_trunc = voxel_size * 5
        
    #     print(self.oid, "make mesh start")
        
    #     # 1. 篩選可用 keyframes（先過濾 static/dynamic 時間）
    #     keyframes = [kf for kf in self.keyframes if kf.id >= self.static_time]
    #     if self.dynamic:
    #         keyframes = [kf for kf in keyframes if kf.id < self.dynamic_time]

    #     if len(keyframes) < 3:
    #         # keyframe 不足 → 回傳現有 mesh（如果有），否則空
    #         V = getattr(self, "vertices", np.zeros((0, 3), dtype=np.float32))
    #         F = getattr(self, "faces",    np.zeros((0, 3), dtype=np.int32))
    #         return V, F

    #     if kf_cur not in keyframes:
    #         # 當前幀不在可用 keyframes 裡 → 不更新，回舊 mesh
    #         V = getattr(self, "vertices", np.zeros((0, 3), dtype=np.float32))
    #         F = getattr(self, "faces",    np.zeros((0, 3), dtype=np.int32))
    #         return V, F

    #     # 2. 把當前幀 kf_cur 移到 keyframes[0]（之後所有 list 的 index 0 就是「當前幀」）
    #     idx = keyframes.index(kf_cur)
    #     keyframes = keyframes[idx:] + keyframes[:idx]

    #     # 3. 準備影像與位姿
    #     rgb_list  = [kf.img.copy() for kf in keyframes]
    #     gray_list = [cv2.cvtColor(kf.img, cv2.COLOR_BGR2GRAY) for kf in keyframes]

    #     # 假設 kf.Tcw 是 Tcw = world -> camera
    #     poses_cw = [kf.Tcw.copy() for kf in keyframes]

    #     # 當前幀資料（現在就是 index 0）
    #     rgb_cur  = rgb_list[0]
    #     gray_cur = gray_list[0]
    #     K        = keyframes[0].camera.K

    #     # 物體遮罩：只重建 mask 內的點
    #     mask = self.mask(keyframes[0]).astype(np.uint8)
    #     mask[mask > 0] = 1  # 轉成 0/1

    #     # 4. 如果還沒有 TSDF volume，就在世界座標系下建立一個
    #     if self.volume is None:
            
    #         center = self.location_3D.astype(np.float64)   # 世界座標物體中心

    #         # world --> TSDF (把 TSDF 原點放在物體世界位置)
    #         self.Tws = np.eye(4)
    #         self.Tws[:3, 3] = -center

    #         # TSDF --> world
    #         self.Tsw = np.eye(4)
    #         self.Tsw[:3, 3] = center

    #         self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
    #             voxel_length=voxel_size,
    #             sdf_trunc=sdf_trunc,
    #             color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    #         )
                
        
    #     # 5. 多幀光流追蹤（以當前幀 gray_list[0] 為起點）
    #     tracks = track_points_multi_frame(gray_list)
    #     print("tracks:", len(tracks))
    #     if len(tracks) == 0:
    #         V = getattr(self, "vertices", np.zeros((0, 3), dtype=np.float32))
    #         F = getattr(self, "faces",    np.zeros((0, 3), dtype=np.int32))
    #         return V, F

    #     # 6. 多幀三角化 → 得到「當前幀相機座標系」中的深度圖
    #     depth = build_depth_from_tracks(
    #         tracks,         # list of (num_frames, 2)
    #         poses_cw,       # list of Tcw（world -> camera），順序對應 gray_list
    #         K,
    #         gray_cur.shape, # (H, W)
    #         mask,
    #     )
    #     print("depth nonzero:", np.count_nonzero(depth),
    #         "min/max:", float(depth.min()), float(depth.max()))
        

        
        
    #     if np.count_nonzero(depth) == 0:
    #         # 沒有任何有效深度 → 不更新 mesh
    #         V = getattr(self, "vertices", np.zeros((0, 3), dtype=np.float32))
    #         F = getattr(self, "faces",    np.zeros((0, 3), dtype=np.int32))
    #         return V, F

    #     # 7. 準備 RGBD：mask 外全部清零，避免把背景寫進 TSDF
    #     depth[mask == 0] = 0
    #     rgb_cur[mask == 0] = 0

    #     depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    #     rgb_o3d   = o3d.geometry.Image(rgb_cur.astype(np.uint8))

        
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         rgb_o3d,
    #         depth_o3d,
    #         depth_scale=1.0,     # 已經是 meter，不需要再縮放
    #         depth_trunc=depth.max(),     # 同 max_depth，可適度放大一點
    #         convert_rgb_to_intensity=False
    #     )
    #     # 把 rgbd.depth 轉回 numpy 看看有多少非零：
    #     depth_after = np.asarray(rgbd.depth)
    #     print("depth_after nonzero:", np.count_nonzero(depth_after),
    #         "min/max:", depth_after.min(), depth_after.max())

    #     h, w = depth.shape
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         w, h,
    #         K[0, 0], K[1, 1],
    #         K[0, 2], K[1, 2]
    #     )

    #     # 8. 把這一幀的 RGBD integrate 進「世界座標系」的 TSDF volume
    #     #    Open3D TSDF 的 extrinsic 預期是 world->camera（也就是 Tcw）
    #     Tcw = kf_cur.Tcw               # world → camera
    #     extrinsic = Tcw @ self.Tsw     # TSDF → camera
    #     self.volume.integrate(rgbd, intrinsic, extrinsic)
        
    #     print("extrinsic (camera→TSDF):\n", extrinsic)
    #     print("det =", np.linalg.det(extrinsic[:3,:3]))
        
    #     Twc = kf_cur.Twc
    #     print("Tcw_cur =\n", poses_cw[0])
    #     print("Twc_cur =\n", Twc)
    #     print("Camera world position =", Twc[:3, 3])


    #     # 9. 從 TSDF volume 抽 mesh（頂點 V 是世界座標）
    #     mesh = self.volume.extract_triangle_mesh()
    #     mesh.transform(self.Tsw)   # TSDF → world

    #     V = np.asarray(mesh.vertices, dtype=np.float32)
    #     F = np.asarray(mesh.triangles, dtype=np.int32)

    #     # 存回物件屬性：用 self.vertices / self.faces
    #     self.vertices = V
    #     self.faces    = F

    #     print("mesh updated: V", V.shape, "F", F.shape)
    #     return V, F
    
    # def make_mesh(self, kf_cur:'KeyFrame'):
    #     print("\n==================== make_mesh START ====================")
    #     print('oid',self.oid)
    #     voxel_size = self.map.voxel_size      # 你現在用 2.5
    #     sdf_trunc  = voxel_size * 5

    #     # ---------- 1. 選 keyframes ----------
    #     keyframes = [kf for kf in self.keyframes if kf.id >= self.static_time]
    #     if self.dynamic:
    #         keyframes = [kf for kf in keyframes if kf.id < self.dynamic_time]

    #     print("[INFO] usable keyframes =", len(keyframes))

    #     if len(keyframes) < 3:
    #         print("[WARN] not enough keyframes")
    #         return self.vertices, self.faces

    #     if kf_cur not in keyframes:
    #         print("[WARN] kf_cur not in usable keyframes")
    #         return self.vertices, self.faces

    #     idx = keyframes.index(kf_cur)
    #     keyframes = keyframes[idx:] + keyframes[:idx]

    #     # ---------- 2. 影像 + pose ----------
    #     rgb_list  = [kf.img.copy() for kf in keyframes]
    #     gray_list = [cv2.cvtColor(kf.img, cv2.COLOR_BGR2GRAY) for kf in keyframes]
    #     poses_cw  = [kf.Tcw.copy() for kf in keyframes]   # world -> camera

    #     rgb_cur  = rgb_list[0]
    #     gray_cur = gray_list[0]
    #     Tcw_cur  = poses_cw[0]
    #     Twc_cur  = keyframes[0].Twc
    #     K        = keyframes[0].camera.K

    #     print("[INFO] Tcw_cur =\n", Tcw_cur)
    #     print("[INFO] Twc_cur =\n", Twc_cur)
    #     print("[INFO] Camera world position =", Twc_cur[:3,3])

    #     # ---------- 3. mask ----------
    #     mask = self.mask(keyframes[0]).astype(np.uint8)
        
    #     mask[mask > 0] = 1
    #     print("[INFO] mask nonzero =", np.count_nonzero(mask))

    #     H, W = gray_cur.shape

    #     # ---------- 4. TSDF volume 初始化（只做一次） ----------
    #     if self.volume is None:
    #         self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
    #             voxel_length=voxel_size,
    #             sdf_trunc=sdf_trunc,
    #             color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    #         )
    #         print("[INFO] TSDF volume created: voxel_size =", voxel_size,
    #             "sdf_trunc =", sdf_trunc)

    #     # ---------- 5. 光流 tracks ----------
    #     tracks = track_points_multi_frame(gray_list)
    #     print("[INFO] tracks =", len(tracks))
    #     if len(tracks) == 0:
    #         print("[WARN] no tracks")
    #         return self.vertices, self.faces

    #     # ---------- 6. triangulation → depth（當前幀相機座標） ----------
    #     depth = build_depth_from_tracks(
    #         tracks,
    #         poses_cw,
    #         K,
    #         (H, W),
    #         mask,
    #         max_depth=np.inf
    #     )
    #     print("[INFO] depth nonzero:", np.count_nonzero(depth),
    #         "min/max:", depth.min(), depth.max())

    #     if np.count_nonzero(depth) == 0:
    #         print("[WARN] all depth = 0")
    #         return self.vertices, self.faces

    #     # ---------- 7. 準備 RGBD ----------
    #     depth_masked = depth.copy()
    #     depth_masked[mask == 0] = 0
    #     rgb_masked = rgb_cur.copy()
    #     rgb_masked[mask == 0] = 0

    #     dmax = float(depth_masked.max())
    #     depth_o3d = o3d.geometry.Image(depth_masked.astype(np.float32))
    #     rgb_o3d   = o3d.geometry.Image(rgb_masked.astype(np.uint8))

    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         rgb_o3d,
    #         depth_o3d,
    #         depth_scale=1.0,
    #         depth_trunc=dmax * 1.1,
    #         convert_rgb_to_intensity=False
    #     )

    #     depth_after = np.asarray(rgbd.depth)
    #     print("[INFO] depth_after nonzero:", np.count_nonzero(depth_after),
    #         "min/max:", depth_after.min(), depth_after.max())

    #     # ---------- 8. intrinsic ----------
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         W, H,
    #         K[0,0], K[1,1],
    #         K[0,2], K[1,2]
    #     )

    #     # ---------- 9. extrinsic = world -> camera （照 Open3D 定義） ----------
    #     extrinsic = Tcw_cur
    #     print("[INFO] extrinsic (world->camera) =\n", extrinsic)
    #     print("[INFO] det(R) =", np.linalg.det(extrinsic[:3,:3]))

    #     # ---------- 10. integrate ----------
    #     self.volume.integrate(rgbd, intrinsic, extrinsic)
    #     print("[INFO] integrated into TSDF")

    #     # ---------- 11. extract mesh ----------
    #     mesh = self.volume.extract_triangle_mesh()
    #     V = np.asarray(mesh.vertices, dtype=np.float64)
    #     F = np.asarray(mesh.triangles, dtype=np.int32)

    #     print("[INFO] mesh V/F shapes:", V.shape, F.shape)
    #     print("==================== make_mesh END ====================\n")

    #     self.vertices = V
    #     self.faces    = F
    #     print()
    #     print(V)
    #     return V, F

    def _init_volume(self):
        voxel_size = self.map.voxel_size # 2.5
        sdf_trunc  = voxel_size * 5 # 表面附近 ±sdf_trunc 或 相機到表面前方 的體素 才會更新
        
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8 # 同時融合顏色（mesh 有顏色）
        # color_type = o3d.pipelines.integration.TSDFVolumeColorType.NoColor # 只做幾何 速度較快
        # color_type = o3d.pipelines.integration.TSDFVolumeColorType.Gray32
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type= color_type
        )
        center = self.location_3D.copy()
        # 建立物體到世界座標系的變換矩陣 T_wo
        # 物體初始朝向與世界座標一致只有平移,沒有旋轉(假設物體與世界座標系對齊)
        T_wo = np.eye(4, dtype=np.float64)
        T_wo[:3, 3] = center  # 設定平移部分
        
        self.pose = T_wo
        
    
    def make_mesh(self, kf_cur:'KeyFrame',local_kfs:list['KeyFrame']=None,flows=None):

        print(f"\n==================== make_mesh START: {self.oid} ====================")

        
        # ---------- 1. 選 keyframes ----------        
        if local_kfs is None:
            local_kfs = self.map.get_covisible_keyframes(kf_cur, include_input=False)
        
        
        keyframes = [kf for kf in self.keyframes if kf.id >= self.static_time]
        
        if self.dynamic is None: # 疑似動態/動態時不更新
            return self.vertices, self.faces
            keyframes = [kf for kf in keyframes if kf.id < self.dynamic_time]
        
        if self.dynamic:
            return self.vertices, self.faces
            keyframes = [kf for kf in keyframes if kf.id < self.dynamic_time]
            
        
        keyframes = [kf for kf in keyframes if kf in local_kfs]

        if kf_cur in keyframes:
            keyframes = [kf for kf in keyframes if kf != kf_cur]

        print("[INFO] usable keyframes =", len(keyframes))
        if len(keyframes) < 3:
            print("[WARN] not enough keyframes")
            return self.vertices, self.faces
        
        
        # idx = keyframes.index(kf_cur)
        # keyframes = keyframes[idx:] + keyframes[:idx]
        
        # ---------- 2. 準備影像 + pose ----------
        # rgb_list  = [kf.img.copy() for kf in keyframes]
        # gray_list = [cv2.cvtColor(kf.img, cv2.COLOR_BGR2GRAY) for kf in keyframes]
        
        # gray_cur = gray_list[0]
        
        
        # ---------- 3. mask ----------
        mask = self.mask(kf_cur).astype(np.uint8)
        mask[mask > 0] = 1
        print("[INFO] mask nonzero =", np.count_nonzero(mask))
        
        # H, W = gray_cur.shape
        
        
        # ---------- 4. 初始化 TSDF volume（一次） ----------
        if self.volume is None:
            self._init_volume()
        
            
            
        # # ---------- 5. 光流 tracks ----------
        
        # depth, _ = multi_frame_dense_depth(gray_list, Tcw_list, K, mask)
        depth = multi_frame_dense_depth(kf_cur,keyframes , mask, flows)
        
        # self.map.slam.map_display.display_info('obj_depth',('raw',depth),kf_cur.id)
        depth = clean_depth(depth,mask)
        # self.map.slam.map_display.display_info('obj_depth',('clean',depth),kf_cur.id)
        depth = fill_depth(depth,mask)
        # self.map.slam.map_display.display_info('obj_depth',('fill',depth),kf_cur.id)
        
        if np.count_nonzero(depth) == 0:
            print("[WARN] all depth = 0")
            return self.vertices, self.faces
        
        # ---------- 7. RGBD ----------
        depth_masked = depth.copy()
        depth_masked[mask == 0] = 0
        rgb_masked = kf_cur.img.copy()
        rgb_masked[mask == 0] = 0
        
        dmax = float(depth_masked.max())
        depth_o3d = o3d.geometry.Image(depth_masked.astype(np.float32))
        rgb_o3d   = o3d.geometry.Image(rgb_masked.astype(np.uint8))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=dmax * 1.1,
            convert_rgb_to_intensity=False
        )
        
        # depth_after = np.asarray(rgbd.depth)
        # print("[INFO] depth_after nonzero:", np.count_nonzero(depth_after),
        #     "min/max:", float(depth_after.min()), float(depth_after.max()))

        # ---------- 8. intrinsic ----------
        K        = kf_cur.camera.K
        H, W = kf_cur.camera.height , kf_cur.camera.width
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H,
            K[0,0], K[1,1],
            K[0,2], K[1,2]
        )

        # ---------- 9. extrinsic = world -> camera（Open3D 定義） ----------
        # 為了讓物體可移動，改成以物體座標為基底 
        # extrinsic =  T_co = Tcw @ Two   # object -> camera
        extrinsic = kf_cur.Tcw @ self.Twc # Tco
                
        # print("[INFO] extrinsic (world->camera) =\n", extrinsic)
        
        
        # ---------- 10. integrate ----------
        self.volume.integrate(rgbd, intrinsic, extrinsic)
        
        # ---------- 11. extract mesh ----------
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh = filter_mesh_by_components(mesh) # 物體座標系的mesh
        
        Vo = np.asarray(mesh.vertices, dtype=np.float64) # n*3
        F = np.asarray(mesh.triangles, dtype=np.int32)
        # 轉成世界座標系
        # V_w = T_wo @ V_o
        R = self.Twc[:3,:3]
        t = self.Twc[:3, 3]
        V = Vo @ R.T + t # Xw.T​=Xo.T @ ​R.T + t
        
        print("[INFO] mesh V/F shapes:", V.shape, F.shape)
        print("==================== make_mesh END ====================\n")

        self.vertices = V
        self.faces    = F
        return V, F
    
    
    def recover_mesh(self,kf_cur:'KeyFrame',
        local_kfs:list['KeyFrame']=None,
        max_points_model=30000,
        max_points_obs=50000,
        ):
        
        
        # ---------- 1. 選 keyframes ----------
        if local_kfs is None:
            local_kfs = self.map.get_covisible_keyframes(kf_cur)
        
        if kf_cur not in local_kfs:
            local_kfs.append(kf_cur) # 確保 local kfs 包含 kf_cur
        
        keyframes = [kf for kf in self.keyframes if kf.id >= self.static_time]
        if self.dynamic:
            return None
            keyframes = [kf for kf in keyframes if kf.id < self.dynamic_time]
            
        keyframes = [kf for kf in keyframes if kf in local_kfs]
        
        print("[INFO] usable keyframes =", len(keyframes))
        
        if len(keyframes) < 5:
            print("[WARN] not enough keyframes")
            return None
        
        # ------------------------------------------------------------
        # 1. 從 TSDF 取出「模型點雲」（object frame）
        # ------------------------------------------------------------
        voxel_size = self.map.voxel_size
        
        mesh = self.volume.extract_triangle_mesh()
        if len(mesh.vertices) == 0:
            print("[ERR] TSDF mesh empty")
            return None

        mesh = filter_mesh_by_components(mesh,0.1) 
        mesh.compute_vertex_normals()
        pc_model = mesh.sample_points_uniformly(max_points_model)
        # pc_model: object frame O
        Two_init = self.Twc.copy()
        
        # ------------------------------------------------------------
        # 2. 從 post-static 多幀建立「觀測點雲」（world frame）
        # ------------------------------------------------------------
        pts_world = []
        view_world = []
        for kf in keyframes:
            K = kf.camera.K
            Tcw = kf.Tcw
            Twc = kf.Twc
            
            mask = self.mask(kf).astype(np.uint8)
            mask[mask > 0] = 1
            
            # 計算深度
            depth = multi_frame_dense_depth(kf,keyframes,mask)
            # self.map.slam.map_display.display_info('obj_depth',('raw',depth),kf_cur.id)
            depth = clean_depth(depth,mask)
            # self.map.slam.map_display.display_info('obj_depth',('clean',depth),kf_cur.id)
            depth = fill_depth(depth,mask)
            # self.map.slam.map_display.display_info('obj_depth',('fill',depth),kf_cur.id)
            
            ys, xs = np.where((mask > 0) & (depth > 0))
            if len(xs) == 0:
                continue
            
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            for u, v in zip(xs, ys):
                z = depth[v, u]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                Xc = np.array([x, y, z, 1.0])
                Xw = (Twc @ Xc)[:3]
                
                # 計算視線方向
                Cw = Twc[:3, 3]                # 這個 keyframe 的相機中心（世界座標）
                v = Cw - Xw                    # point -> camera
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    continue
                v = v / nv
                
                pts_world.append(Xw)
                view_world.append(v)

        if len(pts_world) < 100:
            print("[ERR] not enough observation points")
            return None
        
        pts_world = np.asarray(pts_world, dtype=np.float64)
        view_world = np.asarray(view_world, dtype=np.float64)
        
        
        
        pc_full = o3d.geometry.PointCloud()
        pc_full.points = o3d.utility.Vector3dVector(pts_world)
        
        # 保存方便測試
        # save_pointcloud(pc_model, "cache/pc_model.ply")
        # save_pointcloud(pc_full, "cache/pc_full.ply")
        # save_pose(Two_init, "cache/Two_init.npy")
        # save_view_dirs(view_world, "cache/view_ds.npy")
        
        
        
        # 預設：使用完整點雲
        pc_obs = o3d.geometry.PointCloud(pc_full)   # 深拷貝，避免 side-effect
        view_ds = view_world.copy()

        # ------------------------------------------------------------
        # （可選）隨機下採樣，只影響速度，不影響幾何
        # ------------------------------------------------------------
        # n_pts = len(pc_obs.points)
        # if n_pts > max_points_obs:
        #     idx = np.random.choice(n_pts, max_points_obs, replace=False)
        #     pts = np.asarray(pc_obs.points)[idx]
        #     pc_obs.points = o3d.utility.Vector3dVector(pts)
        #     view_ds = view_world[idx]
        
        # 剃除游離的錯誤點雲
        pc_obs, ind = pc_obs.remove_statistical_outlier(
            nb_neighbors=20,        # 每個點看 20 個鄰居
            std_ratio=2.0           # 超過 2 個標準差就丟
            )
        view_ds = view_ds[ind]
        
        # 先用 PCA 估計 surface normals（此時仍可能是 ±n）
        pc_obs.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius= voxel_size/10, # 大概可以用的值
                max_nn=30
            )
        )
        pc_obs.normalize_normals()
        # 用觀測方向解決 ±n：讓 normal 朝向相機（dot(n, view) >= 0）
        N = np.asarray(pc_obs.normals, dtype=np.float64)
        dot = np.einsum('ij,ij->i', N, view_ds)
        flip = dot < 0
        N[flip] *= -1.0
        pc_obs.normals = o3d.utility.Vector3dVector(N)

        # ------------------------------------------------------------
        # 3. ICP：object frame → world frame
        # ------------------------------------------------------------
        
        
        print("[INFO] pc_obs points:", np.asarray(pc_obs.points).shape[0],
        "has_normals:", pc_obs.has_normals())
        
        
        
        
        icp_distance = voxel_size*10 # 大概可以用的值
        
        # 測試用
        draw_registration_result(pc_model, pc_obs, Two_init)
        point_to_point_icp(pc_model, pc_obs, icp_distance, Two_init)
        point_to_plane_icp(pc_model, pc_obs, icp_distance, Two_init)
        
        
        
        result = o3d.pipelines.registration.registration_icp(
            pc_model, # p_o_init
            pc_obs, # p_w
            max_correspondence_distance=icp_distance,
            init=Two_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        print("[INFO] ICP fitness =", result.fitness) # 對齊點比例
        print("[INFO] ICP rmse    =", result.inlier_rmse)

        if result.fitness < 0.3:
            print("[WARN] ICP failed / unreliable")
            return None

        Two_new = result.transformation

        # ------------------------------------------------------------
        # 4. Sanity check（可選但強烈建議）
        # ------------------------------------------------------------
        detR = np.linalg.det(Two_new[:3, :3])
        print("[INFO] det(Rwo) =", detR)

        if abs(detR - 1.0) > 1e-2:
            print("[WARN] invalid rotation")
            return None
        
        
        return Two_new
        
    
    
    
    def delete(self):
        for pt in list(self.points):
            self.remove_point(pt)
        
        for f in list(self.frames.keys()):
            self.remove_frame(f)
        
        if self.is_thing:
            self.map.remove_thing(self)
        else:
            self.map.remove_stuff(self)
            
        obj_tracker = self.map.slam.obj_tool.obj_tracker
        if self in obj_tracker.objects:
            obj_tracker.objects.pop(self)
        
        
        self.vertices:np.ndarray = None#np.zeros((0,3), dtype=np.float64)    # vertix 面的頂點 (n,3)[x,y,z]
        self.faces:np.ndarray = None#np.zeros((0,3), dtype=np.int32)         # 3個點組成的三角形面 (n,3)
        # print('obj 被參考次數',sys.getrefcount(self),'如果只剩下1，代表沒有其他參考。') # 如果只剩下 1（注意：getrefcount 會多算一次自己），代表沒有其他參考。
        
# 小（可數）物體，通常可以移動，前景，例如：書、椅子
class Thing(ObjectBase):
    _id_counter = 0
    def __init__(self, mapp:'Map'):
        
        self.map = mapp
        
        self.oid = ObjectBase._id_counter
        ObjectBase._id_counter += 1  # 每次建立物件時遞增
        self.id = Thing._id_counter
        Thing._id_counter += 1  # 每次建立物件時遞增
        self.is_thing = True
        
        # 類型
        self.category_id = None
        self.category_ids = []
        self.category_ids_scores = []
        
        ###rerelation###
        self.points:set['Point'] = set() # 避免重複添加point
        self.frames:defaultdict['Frame',set[int]] = defaultdict(set) # frame:list[idx_mask]  一個frame的多個遮罩可能會判斷成同一物體 TODO:根據目前的方法，只會有一個遮罩考慮去除set層
        self.keyframes:defaultdict['KeyFrame',set[int]] = defaultdict(set)
        
        self.static_points_relative: dict['Point', np.ndarray] = {}
        # key: Point, value: 點相對於物體中心的向量 (point.location_3D - self.static_center)
        
        ###hull###
        
        # 點形成面
        self.planes:list[tuple[np.ndarray,o3d.geometry.PointCloud]] = []    # list[(plane,inlier_cloud)]
                                                                            # plane: (4,) array，格式為 [a, b, c, d]，表示 ax+by+cz+d=0 的平面
                                                                            # inlier_cloud: o3d.geometry.PointCloud, 包含屬於此平面的點的座標
        
        self.make_hull_time:int = 0 # f.id
        self.carve_hull_time:int = 0 # f.id
        
        
        self.volume = None
        # mesh 網格
        self.vertices:np.ndarray = np.zeros((0,3), dtype=np.float64)  # vertix 面的頂點
        self.faces = np.zeros((0,3), dtype=np.int32) # 3個點組成的三角形 (n,3)
        # voxelgrid
        self.voxel_grid:VoxelGrid|None = None
        # self.occ:np.ndarray = None # 3D 佔據網格 (occupancy grid)
        
        
        
        
        ###描述子###
        
        # TODO: 剛體(rigid object)判斷，只有剛體能建立網格mesh
        
        
        ###其他###
        self.tracking_points_num:int = 0 # 用於判斷物體是否動態
        self.dynamic:bool|None = False # 現在物體是否在移動
        self.dynamic_time:int = None # 如果移動，紀錄移動時的f.id
        self.static_time:int = self.map.frames[-1].id # 如果移動結束，紀錄固定時的f.id
        
        self.fixed = False # 當物體移動時固定與points之間的關聯，並且禁止更新mesh,voxel_grid unused
        self.need_to_static= False
        
        self.fused_by = None
        self.is_bad:bool = False
        self.map.add_thing(self)
    
    def set_dynamic(self,fid:int,dynamic:bool|None)->bool:
        """
        更新物體的動態狀態
        Args:
            fid: 當前幀 ID
            dynamic: 新的動態狀態 (True=動態, False=靜態, None=疑似動態)
        Returns:
            bool: 狀態變成動態或 從動態變成靜態(需要操作地圖物件)
        """
        
        # 沒有狀態變化
        if self.dynamic == dynamic:
            return False
        
        match (self.dynamic, dynamic):
            
            case (False, True):
                # 靜態 → 動態：物體從靜止開始移動
                # 記錄動態時間作為移動開始時間點
                self.dynamic = True
                self.dynamic_time = fid
                return True
            
            case (False, None):
                # 靜態 → 不確定：物體可能開始移動（疑似動態）
                # 記錄此時間點作為「首次懷疑移動」的時間
                self.dynamic = None
                self.dynamic_time = fid
                return False
            
            case (None, True):
                # 不確定 → 動態：確認物體正在移動
                # 不更新 dynamic_time，保留之前 False→None 時記錄的時間
                # 這樣 dynamic_time 代表「首次懷疑移動」而非「確認動態」的時間
                self.dynamic = True
                return True
            
            case (None, False):
                # 不確定 → 靜態：確認物體是靜態的（誤判為可能移動）
                # 不設定 static_time，因為物體從未真正移動
                self.dynamic = False
                return False
            
            case (True, False):
                # 動態 → 靜態：物體從移動中停止
                # 記錄停止移動的時間點
                self.dynamic = False
                self.static_time = fid
                return True
            
            case (True, None):
                # 動態 → 不確定：可能是暫時遮擋或追蹤失敗
                # 保守策略：維持動態狀態，不改變狀態
                return False
            
            case _:
                # 其他未定義的狀態轉換（理論上不會發生）
                return False
    
    
    def dynamic_update_old(self,):
        
        self.update_points_by_frames(only_keyframe=True)
        
        self.static_center = self.location_3D.copy()
        print('進入動態pts',len(self.points))
        # 紀錄pt obj 相對關係
        for pt in self.points.copy():
            if pt.is_bad:continue
            
            # 複製pt，並刪除原始pt與obj關係
            pt.remove_obj(self)
            
            for kf in list(pt.keyframes):
                if kf.id <=self.dynamic_time: # 剃除動態後的frame
                    kf.remove_point(pt)
            
            pt = pt.copy()
            
            
            pt.add_obj(self)
            pt.set_dynamic(self.dynamic_time)
            
            
            relative_pos = pt.location_3D - self.static_center
            self.static_points_relative[pt] = relative_pos
        
        
        # 使用確定靜態的時間make hull
        self.make_hull()
        print('進入動態pts',len(self.points))
        
        
    def static_update_old(self,kf_cur:'KeyFrame'):
        
        static_center = self.static_center 
        
        
        
        # 將物體上的pt校準到當前kf
        matches = ObjectPointMatcher.match_points_in_object(self,kf_cur)
        
        for pt, kp_idx in matches.items():
            pt.add_frame(kf_cur,kp_idx)
            pt.first_kid = kf_cur.kid
            pt.kf_ref = kf_cur
        
        # 使用pt確定物體pose變化
        T_wo, mean_error = recover_object(self,kf_cur)
        
        print(f"批配特徵點:{len(matches)}")
        print(f"移動obj: {self.oid}, 原始位置{static_center} newpose {T_wo}")
        if T_wo is None:
            self.need_to_static = True
            return False
        
        self.need_to_static = False
        print(f" 移動位置{T_wo[:3,3].flatten()-static_center}")
        for pt in self.points:
            if pt not in self.static_points_relative:
                continue  # 跳過新添加的點
            
            p_o = np.append(self.static_points_relative[pt], 1.0) # 轉為齊次座標
            pt.location = T_wo @ p_o
            
            # 旋轉法向量（法向量只需要旋轉，不需要平移）
            # 從變換矩陣中提取旋轉部分 (3x3)
            R_wo = T_wo[:3, :3]
            pt.normal = R_wo @ pt.normal
            # 確保法向量仍然是單位向量（旋轉矩陣理論上保持長度，但為了數值穩定性）
            norm = np.linalg.norm(pt.normal)
            if norm > 1e-8:
                pt.normal = pt.normal / norm
        
        
        # 旋轉移動mesh並轉換體素網格
        if len(self.vertices) > 0:
            # 1. 計算頂點相對於 static_center 的相對位置
            vertices_relative = self.vertices - static_center  # (N, 3)
            # 將頂點轉為齊次座標
            vertices_homogeneous = np.c_[vertices_relative, np.ones(len(vertices_relative))]
            # 應用變換
            vertices_transformed = (T_wo @ vertices_homogeneous.T).T
            # 轉回非齊次座標
            self.vertices = vertices_transformed[:, :3]
            
            mesh = Mesh.from_numpy(self.vertices, self.faces)
            self.voxel_grid = VoxelGrid.from_mesh(mesh.legacy, voxel_size=self.map.voxel_size)

        else:
            # mesh 網格
            self.vertices:np.ndarray = np.zeros((0,3), dtype=np.float64)  # vertix 面的頂點
            self.faces = np.zeros((0,3), dtype=np.int32) # 3個點組成的三角形 (n,3)
            # voxelgrid
            self.voxel_grid:VoxelGrid|None = None
        
        del self.static_center

        return True
    
    
    
    def dynamic_update(self,):
        
        self.update_points_by_frames(only_keyframe=True)
        
        # self.static_center = self.location_3D.copy()
        print('進入動態pts',len(self.points))
        # 紀錄pt obj 相對關係
        for pt in self.points.copy():
            if pt.is_bad:continue
            
            # 複製pt，並刪除原始pt與obj關係
            pt.remove_obj(self)
            
            for kf in list(pt.keyframes):
                if kf.id <=self.dynamic_time: # 剃除動態後的frame
                    kf.remove_point(pt)
            
            pt = pt.copy()
            
            
            pt.add_obj(self)
            pt.set_dynamic(self.dynamic_time)
            
            
            # relative_pos = pt.location_3D - self.static_center
            # Po = Pw @ Two
            relative_pos = pt.location @ self.Twc
            self.static_points_relative[pt] = relative_pos
        
        
        # # 使用確定靜態的時間make hull
        # self.make_hull()
        print('進入動態pts',len(self.points))
        
    
    def static_update(self,kf_cur:'KeyFrame'):
        
        # static_center = self.static_center
        
        T_wo = self.recover_mesh(kf_cur)
        
        
        # # TODO:V將物體上的pt校準到當前kf
        # matches = match_points_in_object(self,kf_cur)
        # for pt, kp_idx in matches.items():
        #     pt.add_frame(kf_cur,kp_idx)
        #     pt.first_kid = kf_cur.kid
        #     pt.kf_ref = kf_cur
        
        # # TODO:V使用pt確定物體pose變化
        # T_wo, mean_error = recover_object(self,kf_cur)
        # print(f"批配特徵點:{len(matches)}")
        # print(f"移動obj: {self.oid}, 原始位置{static_center} newpose {T_wo}")
        
        
        if T_wo is None:
            self.need_to_static = True
            return False
        
        self.need_to_static = False
        print(f" 移動位置{T_wo[:3,3].flatten()-self.Twc[:3,3].flatten()}")
        T21 = np.linalg.inv(T_wo) @ self.Twc
        print(T21)
        
        self.pose = T_wo # 更新物體位置
        
        # 將屬於物體的地圖點移動至新位置
        for pt in self.points:
            if pt not in self.static_points_relative:
                continue  # 跳過新添加的點
            
            pt.set_static(fid= kf_cur.id)
            
            p_o = self.static_points_relative[pt]
            pt.location = T_wo @ p_o
            
            # 旋轉法向量（法向量只需要旋轉，不需要平移）
            # 從變換矩陣中提取旋轉部分 (3x3)
            R_wo = T_wo[:3, :3]
            pt.normal = R_wo @ pt.normal
            # 確保法向量仍然是單位向量（旋轉矩陣理論上保持長度，但為了數值穩定性）
            norm = np.linalg.norm(pt.normal)
            if norm > 1e-8:
                pt.normal = pt.normal / norm
        
        
        # 旋轉移動mesh並轉換體素網格
        if len(self.vertices) > 0:
            
            mesh = self.volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            mesh = filter_mesh_by_components(mesh) # 物體座標系的mesh
            
            Vo = np.asarray(mesh.vertices, dtype=np.float64) # n*3
            F = np.asarray(mesh.triangles, dtype=np.int32)
            # 轉成世界座標系
            # V_w = T_wo @ V_o
            R = self.Twc[:3,:3]
            t = self.Twc[:3, 3]
            V = Vo @ R.T + t # Xw.T​=Xo.T @ ​R.T + t
            
            print("[INFO] mesh V/F shapes:", V.shape, F.shape)

            self.vertices = V
            self.faces    = F

        else:
            # mesh 網格
            self.vertices:np.ndarray = np.zeros((0,3), dtype=np.float64)  # vertix 面的頂點
            self.faces = np.zeros((0,3), dtype=np.int32) # 3個點組成的三角形 (n,3)
            # voxelgrid
            self.voxel_grid:VoxelGrid|None = None
        
        # del self.static_center

        return True
    
    def update_pose(self,kf_cur:'KeyFrame',local_kfs:list['KeyFrame']=None):
        T_wo = self.recover_mesh(kf_cur,local_kfs)
        if T_wo is None:
            return
        
        self.pose = T_wo # 更新物體位置
        
        # 旋轉移動mesh並轉換體素網格
        if len(self.vertices) > 0:
            
            mesh = self.volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            mesh = filter_mesh_by_components(mesh) # 物體座標系的mesh
            
            Vo = np.asarray(mesh.vertices, dtype=np.float64) # n*3
            F = np.asarray(mesh.triangles, dtype=np.int32)
            # 轉成世界座標系
            # V_w = T_wo @ V_o
            R = self.Twc[:3,:3]
            t = self.Twc[:3, 3]
            V = Vo @ R.T + t # Xw.T​=Xo.T @ ​R.T + t
            
            print("[INFO] mesh V/F shapes:", V.shape, F.shape)

            self.vertices = V
            self.faces    = F

        else:
            # mesh 網格
            self.vertices:np.ndarray = np.zeros((0,3), dtype=np.float64)  # vertix 面的頂點
            self.faces = np.zeros((0,3), dtype=np.int32) # 3個點組成的三角形 (n,3)
            # voxelgrid
            self.voxel_grid:VoxelGrid|None = None
    
    def get_frames_info(self):
        ...

    def carve_hull(self, kf:'KeyFrame', keep_holes:bool= True):
        """
        切割視覺外殼
        使用體素切割
        """
        
        if self.dynamic is True or self.dynamic is None:
            return self.vertices, self.faces
        
        if self.voxel_grid is None:
            V = np.zeros((0,3), dtype=np.float64)
            F = np.zeros((0,3), dtype=np.int32)
            return V, F
        
        # 剛建立hull 不需要切割
        if kf.id == self.make_hull_time:
            V = self.vertices
            F = self.faces
            return V,F
        
        self.carve_hull_time = kf.id
        print('carve',self.oid)
        
        voxel_size = self.map.voxel_size
        points = list(self.points)
        mask = self.mask(kf)
        
        exts, holes = mask_to_polygons( mask,
                                        approx_epsilon_ratio=0.01,
                                        keep_holes=keep_holes)
        
        if not exts:
            return self.vertices, self.faces
        
        # get zn, zf
        min_zn = np.inf
        max_zf = 0.0
        # for pt in points:
        #     z = kf.point_to_camera_plane_distance(pt.location_3D)
        #     if z < min_zn:
        #         min_zn = z
        #     if z > max_zf:
        #         max_zf = z
        for v in self.vertices:
            z = kf.point_to_camera_plane_distance(v)
            if z < min_zn:
                min_zn = z
            if z > max_zf:
                max_zf = z
        zn = min_zn
        zf = max_zf
        
        
        # 將其他物體投影產生被遮擋遮罩->被遮擋多邊形
        # obscured_exts, obscured_holes = [],[]
        height, width = kf.camera.height, kf.camera.width
        obscured_mask = np.zeros((height, width),dtype=np.uint8)
        for obj in kf.objects:
            if obj is None:continue
            if (not obj.is_thing) or obj is self or len(obj.faces) == 0 :
                continue
            
            if obj.dynamic is True or obj.dynamic is None:
                continue  # 不將動態物體視為遮擋源 #TODO: 這很不合理，不過計畫將遮擋判斷重寫，所以忽視
            
            
            mask, depth = kf.project_mesh_to_img(obj.vertices,obj.faces)
            
            min_depth = np.min(depth)
            if min_depth < zn:
                obscured_mask = obscured_mask | mask
        
        obscured_exts, obscured_holes = mask_to_polygons(obscured_mask,
                                        approx_epsilon_ratio=0.01,
                                        keep_holes=True)
                
                # obscured_exts.append(obscured_ext)
                # obscured_holes.append(obscured_hole)
        
        
        
        hull = self.voxel_grid
        best_voxel_grid = None
        
        
        # # 檢查與 hull 重疊 TODO: 當前方法可能zf會不夠長，導致切除zf平面後的hull
        # voxel_grid = view_voxel_grid_union(kf,exts, holes,voxel_size,zf,zn)
        # if not voxel_grid.is_overlapping(hull):
        #     zf *= grow
        #     continue
        mesh_proj,_ = kf.project_mesh_to_img(self.vertices, self.faces)
        self.map.slam.map_display.display_info('carve_hull',(self.oid, mesh_proj, exts, holes,obscured_exts, None))
        
        hull = carve_voxel_grid(hull, kf, exts, holes, voxel_size, zf,zn, obscured_exts, obscured_holes)
        
        best_voxel_grid = hull  # 先記下目前結果，以免耗盡迭代時還有備案
        
        # TODO:(3) 雕刻後檢查

            
        
        if best_voxel_grid is None or best_voxel_grid.is_empty:
            self.vertices = np.zeros((0,3), dtype=np.float64)
            self.faces = np.zeros((0,3), dtype=np.int32)
            self.voxel_grid = None
            return np.zeros((0,3), dtype=np.float64), np.zeros((0,3), dtype=np.int32)

        
        self.voxel_grid = best_voxel_grid
        
        best_mesh = Mesh(best_voxel_grid.to_mesh())
        best_mesh.clean()
        
        V,F = best_mesh.to_numpy()
        self.vertices = V
        self.faces = F
        
        
        
        new_mesh_proj,_ = kf.project_mesh_to_img(self.vertices, self.faces)
        self.map.slam.map_display.display_info('carve_hull',(self.oid, mesh_proj, exts, holes,obscured_exts , new_mesh_proj))
        
        
        
        return V,F
    
    def make_hull(self,keep_holes:bool= True):
        """
        建立新的視覺外殼
        使用體素切割
        """
        touch_thresh=0.02
        simplify_vertices=100
        
        voxel_size = self.map.voxel_size
        
        # if self.dynamic is True or self.dynamic is None:
        #     return self.vertices, self.faces  # 保留動態前的結果
        
        
        if self.is_bad:
            return self.vertices, self.faces
        
        # 先過濾出靜態時間後的關鍵幀
        keyframes = [kf for kf in self.keyframes if kf.id >= self.static_time]
        
        # 如果物體是動態的,進一步過濾出動態時間前的關鍵幀
        if self.dynamic:
            keyframes = [kf for kf in keyframes if kf.id < self.dynamic_time]
        
        
        if len(keyframes)<3 or len(self.points)<5: # 限制要有足夠的keyframe 
            V = np.zeros((0,3), dtype=np.float64)
            F = np.zeros((0,3), dtype=np.int32)
            return V, F
        
        last_kf_id = max(keyframes, key=lambda kf: kf.id).id
        last_time = self.make_hull_time
        
        # 上次 make hull 後有新增kf 才 make hull
        if last_time > self.carve_hull_time or last_kf_id == last_time:
            V = self.vertices
            F = self.faces
            return V,F
        
        print('make hull',self.oid)
        self.make_hull_time = last_kf_id
        print(f'make hull time {self.make_hull_time},carve hull time {self.carve_hull_time},last make time {last_time}')
        
        
        
        points = list(self.points)
        
        # print('frame',len(self.frames.keys()),'keyframes',len(keyframes))

        
        masks = []
        exteriors_all = []
        holes_all = []
        znfs:list[tuple[float,float]] = []
        obscured_ext_all = []
        obscured_hole_all = []
        for kf in keyframes:
            mask = self.mask(kf)
            # 轉化成多邊形
            exts, holes = mask_to_polygons(mask, min_area=200,
                                            approx_epsilon_ratio=0.01,
                                            keep_holes=keep_holes)
            
            masks.append(mask)
            exteriors_all.append(exts)
            holes_all.append(holes)
            
            
            min_zn = np.inf
            max_zf = 0.0
            for pt in points:
                z = kf.point_to_camera_plane_distance(pt.location_3D)
                if z < min_zn:
                    min_zn = z
                if z > max_zf:
                    max_zf = z
            
            
            znfs.append((min_zn,max_zf))
            
            
            # 將其他物體投影產生被遮擋遮罩->被遮擋多邊形
            # obscured_exts, obscured_holes = [],[]
            height, width = kf.camera.height, kf.camera.width
            obscured_mask = np.zeros((height, width),dtype=np.uint8)
            for obj in kf.objects:
                if obj is None:continue
                if (not obj.is_thing) or obj is self or len(obj.faces) == 0 :
                    # mask = np.zeros((height, width),dtype=np.uint8)
                    # masks_l.append(mask)
                    continue
                
                mask, depth = kf.project_mesh_to_img(obj.vertices,obj.faces)
                
                min_depth = np.min(depth)
                if min_depth < min_zn:
                    obscured_mask = obscured_mask | mask
            
            obscured_exts, obscured_holes = mask_to_polygons(obscured_mask,
                                            approx_epsilon_ratio=0.01,
                                            keep_holes=True)
                    
                    # obscured_exts.append(obscured_ext)
                    # obscured_holes.append(obscured_hole)
                    
            obscured_ext_all.append(obscured_exts)
            obscured_hole_all.append(obscured_holes)
        
        
        ## 使用points產生平面
        # pcd = o3d.geometry.PointCloud()
        # thing_points = [point.location_3D for point in self.points]
        # print('thing_points:',len(thing_points))
        # planes=[]
        # if len(thing_points)>30:
        #     pcd.points = o3d.utility.Vector3dVector(np.vstack(thing_points))
        #     planes = detect_multi_plane(pcd,keyframes[0].position,min_plane_points=30)
        #     print('產生平面:',len(planes),'個')
        
        
        
        # 使用min_zn,max_zf 產生錐台
        
        
        # 產生最大交集
        best_voxel_grid = None
        # (1)  zn, zf 產生「所有視角錐台」並聯集
        union_voxel_grid = VoxelGrid.from_voxel_size(voxel_size)
        parts:list[VoxelGrid] = []
        
        for f, exts, holes, (zn,zf) in zip(keyframes, exteriors_all, holes_all, znfs):
            
            
            
            
            holes = [[] for _ in exts] # 產生最大交集 時忽視洞holes
            # mesh = view_frustum_union(f, exts, holes, z_far=zf)
            voxel_grid = view_voxel_grid_union(f, exts, holes,voxel_size=voxel_size, z_far=zf,z_near=zn) # 建視角錐台
            
            # if len(mesh.vertices)==0 :print('產生錐台不應該沒有頂點，檢查是否有誤')
            # union_mesh = union_mesh.boolean_union(mesh)
            # _clean_mesh_inplace(union_mesh)
            if  voxel_grid.is_empty:
                print(f'產生錐台不應該沒有體積，檢查是否有誤 zf:{zf},zn:{zn},exts{np.ndarray(exts).shape}')
                continue
                            
            # union_mesh = union_mesh.union(mesh)
            # union_mesh.clean()
            parts.append(voxel_grid)
        

        # TODO: 檢查體素有無重疊，沒有則提高zf
        # TODO: 檢查體素是否被包含在平面反面，沒有則提高zf
        
        # union_voxel_grid = parts[0]
        for voxel_grid in parts:
            union_voxel_grid = union_voxel_grid.union(voxel_grid)
        
        
        # (2) 逐視角雕刻（依模式與遮擋回呼）
        hull = union_voxel_grid
        
        
        for f, exts, holes, (zn,zf), obscured_exts, obscured_holes in zip(keyframes, exteriors_all, holes_all, znfs, obscured_ext_all, obscured_hole_all):
            
            # TODO: 修改成使用mesh 投影距離的最大最小值作為zn,zf
            # min_zn = np.inf
            # max_zf = 0.0
            # for pt in points:
            #     z = kf.point_to_camera_plane_distance(pt.location_3D)
            #     if z < min_zn:
            #         min_zn = z
            #     if z > max_zf:
            #         max_zf = z
            
            hull = carve_voxel_grid(hull, f, exts, holes, voxel_size, zf, z_near=zn, obscured_exts=obscured_exts, obscured_holes=obscured_holes)
            
        
        
        # if hull.is_empty:
        #     # 全被雕掉，多半是 zf 太小造成幾何相交不穩或負向過強，先放大再試
        #     zf *= grow
        #     continue
        
        best_voxel_grid = hull  # 先記下目前結果，以免耗盡迭代時還有備案
        
        # TODO:(3) 雕刻後檢查
        
            
        
        if best_voxel_grid is None or best_voxel_grid.is_empty:
            self.vertices = np.zeros((0,3), dtype=np.float64)
            self.faces = np.zeros((0,3), dtype=np.int32)
            self.voxel_grid = None
            return np.zeros((0,3), dtype=np.float64), np.zeros((0,3), dtype=np.int32)
        
        for plane,pcd in self.planes:
            best_voxel_grid = best_voxel_grid.cut_by_plane(plane)
            
        self.voxel_grid = best_voxel_grid
        
        best_mesh = Mesh(best_voxel_grid.to_mesh())
        best_mesh.clean()
        # best_mesh.simplify(simplify_vertices)
        
        
        V,F = best_mesh.to_numpy()
        self.vertices = V
        self.faces = F
        
        return V,F
    
    '''
    
    def carve_hull_iters(self, kf:'KeyFrame', keep_holes:bool= True, z0:float= 250.0, grow:float= 2.0, max_iters:int= 6):
        """
        切割視覺外殼
        使用體素切割
        """
        if self.voxel_grid is None:
            V = np.zeros((0,3), dtype=np.float64)
            F = np.zeros((0,3), dtype=np.int32)
            return V, F
        
        
        voxel_size = self.map.voxel_size
        
        mask = self.mask(kf)
        
        exts, holes = mask_to_polygons( mask, min_area=200,
                                        approx_epsilon_ratio=0.01,
                                        keep_holes=keep_holes)
        
        if not exts:
            return self.vertices, self.faces
        
        zf = float(z0)
        hull = self.voxel_grid
        best_voxel_grid = None
        for _ in range(max_iters):
            
            voxel_grid = view_voxel_grid_union(kf,exts, holes,zf,voxel_size)
            # 檢查與 hull 重疊 TODO: 當前方法可能zf會不夠長，導致切除zf平面後的hull
            if not voxel_grid.is_overlapping(hull):
                zf *= grow
                continue
            
            hull = carve_voxel_grid(hull, kf, exts, holes, zf, voxel_size)
            
            best_voxel_grid = hull  # 先記下目前結果，以免耗盡迭代時還有備案
            
            # TODO:(3) 雕刻後檢查

            
        
        if best_voxel_grid is None or best_voxel_grid.is_empty:
            self.vertices = np.zeros((0,3), dtype=np.float64)
            self.faces = np.zeros((0,3), dtype=np.int32)
            self.voxel_grid = None
            return np.zeros((0,3), dtype=np.float64), np.zeros((0,3), dtype=np.int32)

        
        self.voxel_grid = best_voxel_grid
        
        best_mesh = Mesh(best_voxel_grid.to_mesh())
        best_mesh.clean()
        
        V,F = best_mesh.to_numpy()
        self.vertices = V
        self.faces = F
        return V,F
    
    
    def make_hull_iters(self,keep_holes:bool= True, z0:float= 25.0, grow:float= 2.0, max_iters:int= 6 ):
        """
        建立新的視覺外殼
        使用體素切割
        """
        touch_thresh=0.02
        simplify_vertices=100
        
        voxel_size = self.map.voxel_size
        
        
        
        keyframes = [f for f in self.frames.keys() if f.is_keyframe]
        masks  = [self.mask(kf) for kf in keyframes]
        
        V = np.zeros((0,3), dtype=np.float64)
        F = np.zeros((0,3), dtype=np.int32)
        
        # print('frame',len(self.frames.keys()),'keyframes',len(keyframes))
        if len(keyframes)<1: # 限制要有足夠的keyframe 
            return V, F
        
        
        # 轉化成多邊形
        exteriors_all = []
        holes_all = []
        for mask in masks:
            exts, holes = mask_to_polygons(mask, min_area=200,
                                            approx_epsilon_ratio=0.01,
                                            keep_holes=keep_holes)
            exteriors_all.append(exts)
            holes_all.append(holes)
        
        
        
        # pcd = o3d.geometry.PointCloud()
        # thing_points = [point.location_3D for point in self.points]
        # print('thing_points:',len(thing_points))
        # planes=[]
        # if len(thing_points)>30:
        #     pcd.points = o3d.utility.Vector3dVector(np.vstack(thing_points))
        #     planes = detect_multi_plane(pcd,keyframes[0].position,min_plane_points=30)
        #     print('產生平面:',len(planes),'個')
        
        
        
        # 使用預設距離產生半徑
        # 如果如果物體結果使用半徑頂點（可能有半徑外物體沒產生），半徑翻倍並重新產生物體    
        # 產生最大交集
        zf = float(z0)
        best_voxel_grid = None
        for _ in range(max_iters):
            
            # (1) 以當前 zf 產生「所有視角錐台」並聯集
            union_voxel_grid = VoxelGrid.from_voxel_size(voxel_size)
            parts:list[VoxelGrid] = []
            
            for f, exts, holes in zip(keyframes, exteriors_all, holes_all):
                

                
                
                
                
                
                holes = [[] for _ in exts] # 產生最大交集 時忽視洞holes
                # mesh = view_frustum_union(f, exts, holes, z_far=zf)
                voxel_grid = view_voxel_grid_union(f, exts, holes, z_far=zf,voxel_size=voxel_size)# 建視角錐台
                
                # if len(mesh.vertices)==0 :print('產生錐台不應該沒有頂點，檢查是否有誤')
                # union_mesh = union_mesh.boolean_union(mesh)
                # _clean_mesh_inplace(union_mesh)
                if  voxel_grid.is_empty:
                    print('產生錐台不應該沒有體積，檢查是否有誤')
                    zf *= grow
                    continue
                                
                # union_mesh = union_mesh.union(mesh)
                # union_mesh.clean()
                parts.append(voxel_grid)
            
            if not parts:
                zf *= grow
                continue
            # TODO: 檢查體素有無重疊，沒有則提高zf
            # TODO: 檢查體素是否被包含在平面反面，沒有則提高zf
            
            # union_voxel_grid = parts[0]
            for voxel_grid in parts:
                union_voxel_grid = union_voxel_grid.union(voxel_grid)
            
            
            # (2) 逐視角雕刻（依模式與遮擋回呼）
            hull = union_voxel_grid            
            for f, exts, holes in zip(keyframes, exteriors_all, holes_all):

                hull = carve_voxel_grid(hull, f, exts, holes, zf, voxel_size)
                
            
            
            # if hull.is_empty:
            #     # 全被雕掉，多半是 zf 太小造成幾何相交不穩或負向過強，先放大再試
            #     zf *= grow
            #     continue
            
            best_voxel_grid = hull  # 先記下目前結果，以免耗盡迭代時還有備案
            
            # TODO:(3) 雕刻後檢查

            
        
        if best_voxel_grid is None or best_voxel_grid.is_empty:
            self.vertices = np.zeros((0,3), dtype=np.float64)
            self.faces = np.zeros((0,3), dtype=np.int32)
            self.voxel_grid = None
            return np.zeros((0,3), dtype=np.float64), np.zeros((0,3), dtype=np.int32)

        for plane,pcd in self.planes:
            best_voxel_grid = best_voxel_grid.cut_by_plane(plane)
            
        self.voxel_grid = best_voxel_grid
        
        best_mesh = Mesh(best_voxel_grid.to_mesh())
        best_mesh.clean()
        # best_mesh.simplify(simplify_vertices)
        
        
        V,F = best_mesh.to_numpy()
        self.vertices = V
        self.faces = F
        print(V,F)
        return V,F
    
    '''
    
    def detect_planes(self,kf:'KeyFrame')->list[tuple[np.ndarray,o3d.geometry.PointCloud]]:
        ...
        return

    
    def updata_category_id(self):
        
        assert len(self.category_ids) == len(self.category_ids_scores)
        
        sorted_pairs = sorted(zip(self.category_ids, self.category_ids_scores), key=lambda x: -x[1])
        self.category_ids, self.category_ids_scores = map(list, zip(*sorted_pairs)) if sorted_pairs else ([], [])
        
        
        self.category_id = self.category_ids[0]
        return self.category_id
    
    def fuse(self,obj:Union['Thing',ObjectBase]):
        
        if obj is self:return
        # 已經被融合檢查
        # if self.fused_by is not None:
        #     # self.fused_by.fuse(obj)
        #     return
        
        # if obj.fused_by is not None:
        #     obj = obj.fused_by
        
        # if self.dynamic is True or obj.dynamic is True:return  # 動態物體不參與融合
        
        if self.is_bad or obj.is_bad:return
        
        if self.is_thing != obj.is_thing:return
        
        
        points = list(obj.points)
        self.add_points(points)
        obj.remove_points(points)
        
        
        self.category_ids.extend(obj.category_ids)
        self.category_ids_scores.extend(obj.category_ids_scores)
        
        self.updata_category_id()
        
        for frame,idxs in obj.frames.copy().items():
            for idx in idxs.copy():
                obj.remove_frame(frame)
                self.add_frame(frame,idx)
        
                
        
        obj.fused_by = self
        obj.is_bad = True
        
    def split(self):# 將 1 thing 分成多個 thing
        ...


    
    '''
    def make_hull_mesh_CSG(self,keep_holes = True, z0:float = 5.0, grow:float = 2.0, max_iters:int = 6):
        """
        建立新的視覺外殼
        使用mesh切割
        mesh 的bool運算不能產生流形mesh，有重大缺陷所以不能使用
        """
        touch_thresh=0.02
        simplify_vertices=100
        
        
        frames = list(self.frames.keys())
        masks  = [self.mask(f) for f in frames]
        # 轉化成多邊形
        exteriors_all = []
        holes_all = []
        for mask in masks:
            exts, holes = mask_to_polygons(mask, min_area=200,
                                            approx_epsilon_ratio=0.01,
                                            keep_holes=keep_holes)
            exteriors_all.append(exts)
            holes_all.append(holes)
        
        # 使用預設距離產生半徑
        # 如果如果物體結果使用半徑頂點（可能有半徑外物體沒產生），半徑翻倍並重新產生物體    
        # 產生最大交集
        zf = float(z0)
        best_mesh = None
        for _ in range(max_iters):
            
            
            # frusta = []
            # (1) 以當前 zf 產生「所有視角錐台」並聯集
            # union_mesh = o3d.geometry.TriangleMesh()
            union_mesh = Mesh()
            for f, exts, holes in zip(frames, exteriors_all, holes_all):
                # holes = [] # 產生最大交集 時忽視洞holes
                holes = [[]]*len(exts)
                mesh = view_frustum_union(f, exts, holes, z_far=zf) # 建視角錐台
                
                # if len(mesh.vertices)==0 :print('產生錐台不應該沒有頂點，檢查是否有誤')
                # union_mesh = union_mesh.boolean_union(mesh)
                # _clean_mesh_inplace(union_mesh)
                if  mesh.is_empty:
                    print('產生錐台不應該沒有頂點，檢查是否有誤')
                    zf *= grow
                    continue
                                
                union_mesh = union_mesh.union(mesh)
                union_mesh.clean()
                
            
            # 沒有頂點，zf增加
            # if len(union_mesh.vertices) == 0:
            if union_mesh.is_empty:
                zf *= grow
                continue
            
            
            # (2) 逐視角雕刻（依模式與遮擋回呼）
            hull = union_mesh
            
            for f, exts, holes in zip(frames, exteriors_all, holes_all):
                hull = carve_mesh(hull, f, exts, holes, zf)
            
            # if len(hull.vertices) == 0:
            if union_mesh.is_empty:
                # 全被雕掉，多半是 zf 太小造成幾何相交不穩或負向過強，先放大再試
                zf *= grow
                continue
            
            # (3) **在雕刻後**檢查是否仍「貼近 zf」
            C0 = frames[0].twc.reshape(3)
            vertices,face = hull.to_numpy()
            # d  = np.linalg.norm(np.asarray(hull.vertices) - C0, axis=1)
            d  = np.linalg.norm(np.asarray(vertices) - C0, axis=1)
            ratio_close_to_far = np.mean(d > (zf * (1.0 - touch_thresh)))

            if ratio_close_to_far < 0.01:
                best_mesh = hull
                break
            else:
                # 雕刻後仍有大量點接近 zf，代表 zf 可能太近（或需更大餘裕），放大重建
                zf *= grow
                best_mesh = hull  # 先記下目前結果，以免耗盡迭代時還有備案
            
        
        if best_mesh is None or best_mesh.is_empty:
            return np.zeros((0,3), dtype=np.float64), np.zeros((0,3), dtype=np.int32)
        
        
        # （與原本相同）最後清理 + QEM 簡化 限制最大頂點數量
        # _clean_mesh_inplace(best_mesh)
        # target_tris = max(int(simplify_vertices * 2), 4)
        # m = best_mesh
        # for _ in range(8):
        #     m2 = m.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        #     m2.remove_degenerate_triangles()
        #     m2.remove_unreferenced_vertices()
        #     if len(m2.vertices) <= simplify_vertices and len(m2.vertices) >= 4:
        #         m = m2
        #         break
        #     if len(m2.vertices) < len(m.vertices):
        #         m = m2
        #     target_tris = max(int(target_tris * 0.7), 4)

        # V = np.asarray(m.vertices, dtype=np.float64)
        # F = np.asarray(m.triangles, dtype=np.int32)
        
        best_mesh.clean()
        best_mesh.simplify(simplify_vertices)
        
        
        V,F = best_mesh.to_numpy()
        self.vertices = V
        self.faces = F
        return V,F
    
    

    def make_hull_old(self):
        """
        產生可視外殼sual hull 由 vertices、faces組成
        
        使用半平面交集，只能產生凸體，且容易沒有交集輸出空mesh
        """
        # hull init
        box_center = self.location_3D
        box_half = self.radius
        V, F = make_initial_box(box_center, box_half)
        
        
        for frame in self.frames:
            mask = self.mask(frame)
            
            # 簡化mask為多邊形 TODO:處理holes
            # TODO: 避免使用影像邊緣切割物體
            # TODO: 使用face取代plane 避免多餘切割
            polys, holes = mask_to_polygons(mask, min_area=200,approx_epsilon_ratio=0.01,keep_holes=False)# polygon 多邊形
            
            for poly in polys:
                
                planes = polygon_to_halfspaces(poly, frame)
                
                for (n,d) in planes:
                    V, F = clip_poly_with_plane(V, F, n, d)
                    if len(F) == 0:
                        break
                
                if len(F) == 0:
                        break
                    
            if len(F) == 0:
                        break
        # 簡化外殼hull
        V,F = simplify_mesh_in_memory(V,F)
        
        self.vertices = V
        self.faces = F
        
        return V,F
    
    def make_hull_spherical(self,
                        ico_subdiv: int = 2,
                        k_required: int = 2,
                        use_sdf: bool = True,
                        tau_px: float = 2.0,
                        r0: float = 0.01,
                        r_max_cap: float = 10.0,
                        simplify_vertices: int = 100):
        """
        使用球面視覺外殼替代半空間交法：
        - 輸入：此 Thing 的所有 frames 與其遮罩（自取 self.mask(frame)）
        - 產出：self.vertices/self.faces
        """
        frames = list(self.frames.keys())
        masks = [self.mask(f) for f in frames]  # 取得各視角遮罩（np.bool_）
        O = self.location_3D if np.linalg.norm(self.location_3D) > 0 else np.zeros(3)
        V, F = build_spherical_visual_hull(
            frames=frames,
            masks=masks,
            center_O=O,
            ico_subdiv=ico_subdiv,
            k_required=k_required,
            use_sdf=use_sdf,
            tau_px=tau_px,
            r0=r0,
            r_max_cap=r_max_cap,
            grow=2.0,
            it_bisect=18,
            simplify_vertices=simplify_vertices
        )
        self.vertices, self.faces = V, F
        return V, F
    
    def make_hull2(self):
        # 速度太慢，並且有bug
        # 建立一個區域3D網格地圖，迭代每一格確定物體體素，最後轉化成網格外殼
        masks = [self.mask(frame) for frame in self.frames]
        voxel_size = 0.5
        occ = probabilistic_visual_hull(frames=self.frames,
                                        masks=masks,
                                        center=self.location_3D,
                                        half=self.radius,
                                        voxel_size=voxel_size
                                        )
        
        origin = self.location_3D - self.radius
        V,F = mesh_from_occupancy_and_simplify(occ,origin=origin,
                                               voxel_size=voxel_size)
        self.occ = occ
        
        self.vertices = V
        self.faces = F
        
    '''
        
        

# 大（不可數）物體，一般不能移動，背景，例如：牆、地板
class Stuff(ObjectBase):
    _id_counter = 0
    def __init__(self, mapp:'Map'):
        self.map = mapp
        
        self.oid = ObjectBase._id_counter
        ObjectBase._id_counter += 1  # 每次建立物件時遞增
        self.id = Stuff._id_counter
        Stuff._id_counter += 1  # 每次建立物件時遞增
        self.is_thing = False
        
        self.category_id = None
        self.category_ids = []
        
        self.points:set['Point']=set() # 避免重複添加point
        self.frames:defaultdict['Frame',set[int]]=defaultdict(set)
        self.keyframes:defaultdict['KeyFrame',set[int]] = defaultdict(set)
        
        
        ###其他###
        self.dynamic:bool = False # stuff 不會移動
        
        self.fused_by = None
        self.is_bad:bool = False
        self.map.add_stuff(self)
        
        
    def updata_category_id(self):
        counter = Counter(self.category_ids)
        most_common_id, _ = counter.most_common(1)[0]
        
        # sorted_pairs = sorted(zip(self.category_ids, self.category_ids_scores), key=lambda x: -x[1])
        # self.category_ids, self.category_ids_scores = map(list, zip(*sorted_pairs)) if sorted_pairs else ([], [])

        self.category_id = most_common_id
        return most_common_id
        
    def fuse(self,obj:Union['Stuff',ObjectBase]):
        if obj is self:return
        if obj.is_thing:return
        # 已經被融合檢查
        if self.fused_by is not None:
            # self.fused_by.fuse(obj)
            return
            
        if obj.fused_by is not None:
            obj = obj.fused_by
        
        points = list(obj.points)
        # self.points.update(obj.points)
        self.add_points(points)
        obj.remove_points(points)
        
        
        self.category_ids.extend(obj.category_ids)
        # self.category_ids_scores.extend(obj.category_ids_scores)
        self.updata_category_id()
        
        
        for frame,idxs in obj.frames.copy().items():
            for idx in idxs.copy():
                obj.remove_frame(frame)
                self.add_frame(frame,idx)
            
        obj.is_bad = True
    
    
    def split(self):
        ...

class ObjectTracker():
    def __init__(self):
        
        # TODO: obj在map中被刪除時同步刪除
        
        self.max_missed = 20
        self.objects:dict[Thing|Stuff,dict] = {}
        # key:thing|stuff, 
        # val: {mask, missed, source}
        
    def update_objects(self,objs_l:list[Thing|Stuff], masks_l:list, masks_c:list, matches,from_project=False):
        """
        參數：
            objs_l: list[None|Thing|Stuff]，上一幀（或來源）物件列表
            masks_l: list[np.ndarray]，上一幀（或來源）物件在當前幀對應的遮罩
            masks_l: list[np.ndarray]，物件在當前幀的遮罩
            matches: list[tuple[int,int,float]]，物件配對結果 (上一幀索引, 當前幀索引, IoU值)
        
        """
        # TODO, 沒使用投影也不會刪除
        
        # matched_prev = {i for (i, _, _) in matches}
        # matched_curr = {j for (_, j, _) in matches}
        
        # matches （[(i,j,iou), ...]  ；i,j 是原索引）
        matched_idxs = {i:j for (i, j, iou) in matches}
        
        
        for i, obj in enumerate(objs_l):
            
            info = self.objects.get(obj, {"mask": None, "missed": 0, "from_project":False})
            
            # 1) 對「有配到」的上一幀物件：用 current 遮罩暫存
            if i in matched_idxs:
                info['missed'] = 0
                j = matched_idxs[i]
                info['mask'] = masks_c[j]
            
            # 2) 對「沒配到」的上一幀物件：用 warp 遮罩暫存
            else:
                # warped mask 
                mask_warped = masks_l[i]
                if np.count_nonzero(mask_warped) == 0 and obj in self.objects:
                    self.objects.pop(obj)
                    print('pop',obj.oid,obj.is_thing,np.count_nonzero(mask_warped))
                    continue  # 完全空就不續命
                
                if from_project:
                    info['from_project']=True
                    info['missed']=0
                
                
                info["mask"] = mask_warped.copy()
            
            # 3) 儲存info
            self.objects[obj] = info
        
    def update_object(self,obj, mask, from_project=False):
        
        
        info = self.objects.get(obj, {"mask": None, "missed": 0, "from_project":False})
        info['mask'] = mask
        if from_project:
            info['missed'] = 0
            info['from_project'] = from_project
            
        
        if not np.count_nonzero(mask) == 0:
            self.objects[obj] = info
        
        
    def add_miss_counter(self):
        for info in self.objects.values():
            info["missed"] += 1
            
    
    def clean_old_lost(self,th:int=0):
        if th == 0:
            th = self.max_missed
        
        to_delete = [obj for obj, info in self.objects.items() if info["missed"] > th]
        for obj in to_delete:
            del self.objects[obj]
            
        return len(to_delete)
            
            
    def get_info(self,obj:Thing|Stuff)->tuple[np.ndarray,int]:
        
        info = self.objects.get(obj, {"mask": None, "missed": 0, "from_project":False})
        
        return info['mask'],info['missed'],info['from_project']
        

class ObjectTool():
    
    def __init__(self,slam:'Slam'):
        self.slam = slam
        self.map = slam.map
        # self.camera = slam.camera
        # self.dis = cv2.DISOpticalFlow.create(0) # input 0:很快, 1:快, 2:中等 （品質↑速度↓）
        
        
        # 補裂縫用
        radius = 2  # 通常設定1~3
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        
        
        
        self.obj_tracker = ObjectTracker()
        
        # obj tracker
        self.max_missed = 10
        # key:thing|stuff, 
        # val: {mask, missed, source}
        self.obj_track_dict:dict[Thing|Stuff,dict] = {}
        # obj: info{'mask','missed':int, 'source':proj|seg}
        
        # self.lost_tracks_stuff = {}
    
        
    def creat_object(self,frame:'Frame',idx_obj:int,min_area:int=0,min_score:float=0.5):
        '''
        min_area: int 像素
        min_score: 0~1 信心分數
        '''
        # 限制產生object的最小面積，避免產生虛假object
        info = frame.obj_infos[idx_obj]
        if info['area'] < min_area: return None
        
        # 避免產生覆蓋超過半畫面的物體，可以肯定這是影像分割模型的錯誤
        max_area = (frame.camera.height * frame.camera.width) // 2
        if info['area'] > max_area: return None
        
        
        # 建立object
        if info['isthing']:
            if info['score'] < min_score : return None # 限制產生object的最小可信度，只有thing有score
            obj = Thing(self.map)
            # obj.category_id = info['category_id']
        else:
            obj = Stuff(self.map)
            # obj.category_id = info['category_id']
            
        
        obj.add_frame(frame,idx_obj)
        obj.update_points_by_frame(frame)
        
        return obj
    
    def predict_object_by_points(self,frame:'Frame',creat_obj=True):
        '''
        根據point的object屬性判斷frame遮罩哪一個object
        '''
        obj_infos = frame.obj_infos
        mask_id = frame.img_info['mask_id']
        min_area = int(frame.camera.height*frame.camera.width /100) #最少佔據畫面1/100的空間才建立新obj
        
        
        point_groups=[list() for i in range(len(obj_infos))]
        # 先將點分群
        for idx_obj,idxs_pt in enumerate(frame.obj_to_pts):
            
            for idx_pt in idxs_pt:
                point = frame.points[idx_pt]
                if point is None: continue
                point_groups[idx_obj].append(point)

        # 統計各群屬於什麼物體
        point_groups:list[list['Point']]
        for idx,point_group in enumerate(point_groups):
            if idx == 0: continue # 0代表沒有被判斷的區域
            if len(point_group) < 1: continue # 沒有點屬於這個區域
            
            
            info = obj_infos[idx]
            main_object = None
            main_count = 0
            bad_segm = False
            
            # 統計每個 point.object 出現次數
            obj_counter = Counter([obj for point in point_group for obj in point.objects ])
            obj_counter = obj_counter.most_common() # transform to list[tuple(obj,count)]
            
            # obj_counter[0] is (None,len(point_group))
            for obj, count in obj_counter:
                if obj is None:continue
                if (info['isthing']) != (obj.is_thing):continue
                
                # 可信度測試    TODO:修改更可靠的版本
                if main_object is not None:
                    if main_count>count*2:bad_segm=False
                    else:bad_segm=True
                    break
                
                main_object = obj
                main_count = count
            # 可能有三種結果：
            # 1.obj is None
            # 2.bad_segm
            # 3.可信的 main_object
            
            # 可能會出現frame的多個mask指向同一obj
            
            
            
            if bad_segm:continue
            
            if main_object is None:
                if creat_obj:
                    obj = self.creat_object(frame,idx,min_area=min_area)
                else:
                    obj = None
                
                
            else:
                obj = main_object
                
                obj.add_frame(frame,idx)
                
            
                
    # def track_object_by_points(self,f1:'Frame',f2:'Frame'):
    #     '''
    #     通過前後幀的相同point，配對不同frame上的mask
    #     通過比較point在不同frame的分割遮罩
    #     不會創造 object
    #     f1:現在frame
    #     f2:舊frame
    #     '''
    #     f1_segment_info = f1.obj_infos
    #     f2_segment_info = f2.obj_infos
        
    #     f1_mask_id = f1.img_info['mask_id']  
    #     f2_mask_id = f2.img_info['mask_id']
            
        
    #     point_groups=[list() for i in range(len(f1_segment_info)+1)]
    #     # 先將點分群
    #     for raw_kp,point in zip(f1.raw_kps,f1.points):
    #         if point is None: continue
    #         point:'Point'
    #         x, y =raw_kp
    #         point_groups[f1_mask_id[y][x]].append(point)
        
    #     # 統計各群屬於什麼物體
    #     for idx,point_group in enumerate(point_groups):
    #         if idx == 0: continue # 0代表沒有被判斷的區域
    #         if len(point_group) < 1: continue # 沒有點屬於這個區域
    #         # 統計每個 point.object 出現次數
    #         obj_counter = Counter([point.object for point in point_group ])

    #         main_object, count = obj_counter.most_common(1)[0]
            
    #         if main_object is None:
    #             # obj = creat_object(f1,point_group,f1_segment_info[idx-1]) # 不包含沒有被判斷的區域，所以索引-1
                
    #             f1.objects[idx] = None
                
    #         else:
    #             f1.objects[idx] = main_object
    #             for point in point_group:
    #                 f1.map.add_point_object_relation(point,main_object)
    
    def track_objects_by_OpticalFlow(self,f_cur:'Frame',f_last:'Frame', flow, iou_thr:float=0.4):
        """
        根據影像光流追蹤，可以在point批配之前執行，輔助點批配
        """
        
        H,W = flow.shape[0], flow.shape[1]
        
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (xx - flow[:,:,0]).astype(np.float32)
        map_y = (yy - flow[:,:,1]).astype(np.float32)
        
        tracker = self.obj_tracker
        
        objs_l:list[Thing | Stuff] = []
        masks_remap = []
        for obj_l, info in tracker.objects.items():
            objs_l.append(obj_l)
            mask = info['mask']
            mask_remap = self.remap_mask(mask,map_x,map_y)
            masks_remap.append(mask_remap)
            
        
        oids = [obj.oid for obj in objs_l]
        # print('OF_obj,objlen',len(oids))
        if len(oids)>0:
            self.slam.map_display.display_info('OF_obj',(oids, masks_remap),f_cur.id)
         
        
        masks_cur = [info['mask'] for info in f_cur.obj_infos]
        
        Nl, Nc = len(objs_l), len(f_cur.objects)
        if Nl <= 1 or Nc <= 1:
            # 跳過IoU判斷
            matches = []
        
        else:
            
            # 建 IoU 矩陣
            objs_c = list()
            for idx, obj_c in enumerate(f_cur.objects):
                if obj_c is None:
                    obj_c = ObjectBase._temp_obj(f_cur, idx) # 建立暫時obj，之後與正式的thing|stuff合併或刪除
                objs_c.append(obj_c)
            
            iou_mat = self.compute_IoU_mat(objs_l, masks_remap, objs_c, masks_cur)
            
            # 貪婪配對：每次挑最大 IoU
            matches = ObjectMatcher.greedy_match(iou_mat,iou_thr)
            
            for i,j,_ in matches:
                obj = objs_l[i]
                obj_c = objs_c[j]
                if obj is None:continue
                obj:Stuff|Thing
                obj.fuse(obj_c)
                # obj.add_frame(f_cur,j)
        
        # 更新物件追蹤器
        tracker.update_objects(objs_l,masks_remap,masks_cur,matches)
        # self.update_obj_tracker(objects_unmatch,masks_remap,matches)
    
    def retrack_objects_by_project(self, f_cur:'Frame',objects:list[Thing|Stuff], iou_thr:float=0.6):
        """
        目前只支援Thing，只有thing有mesh
        
        輸入多個遮罩和深度圖，輸出 id 遮罩（每像素標記為最前景遮罩的 id）。
        - masks: list of (H,W) bool 或 uint8 遮罩（True/1=物體，False/0=背景）
        - depths: list of (H,W) float32 深度圖（與遮罩同 shape，背景可設為 np.inf 或 -1）
        規則：
        - 若多個遮罩重疊，取深度值最小（離相機最近）的遮罩 id
        - 若無遮罩覆蓋，id = -1
        回傳：
        - id_map: (H,W) int32，每像素為遮罩 id（0~N-1），未覆蓋為 -1
        """
        
        height, width = f_cur.camera.height, f_cur.camera.width
        id_map = np.full((height, width), -1, dtype=np.int32)
        min_depth = np.full((height, width), np.inf, dtype=np.float32)
        
        
        
        masks_l = []
        objs_l:list[Thing|Stuff] = []
        # 將物體投影產生可視遮罩
        for obj_l in objects:
            
            if obj_l.is_bad or len(obj_l.faces) == 0 or (obj_l in f_cur.objects):
                continue
            
            if obj_l.dynamic is True or obj_l.dynamic is None:
                continue  # 動態/疑似動態物體不參與投影
            
            mask, depth = f_cur.project_mesh_to_img(obj_l.vertices,obj_l.faces)
            
            # 找比目前 min_depth 更小的像素
            update = depth < min_depth
            id_map[update] = obj_l.oid
            min_depth[update] = depth[update]
            
            objs_l.append(obj_l)
            masks_l.append(mask)
        
        
        # f_cur.obj_prj_id_map = id_map
        # f_cur.obj_prj_depth_map = min_depth
        self.slam.map_display.display_info('obj_proj',(id_map, min_depth), f_cur.id)
        
        if len(objs_l)==0:
            return id_map
        
        
        # 比對mask判斷投影遮罩，判斷是否同一物體
        masks_c = [info['mask'] for info in f_cur.obj_infos]
        objs_c:list[ObjectBase] = list()
        for idx, obj_c in enumerate(f_cur.objects):
            if obj_c is None:
                obj_c = ObjectBase._temp_obj(f_cur, idx)
            
            objs_c.append(obj_c)
        
        iou_mat = self.compute_IoU_mat(objs_l,masks_l,objs_c,masks_c)
        matches = ObjectMatcher.greedy_match(iou_mat,iou_thr)

        
        for i,j,_ in matches:
            obj_l = objs_l[i]
            obj_c = objs_c[j]
            # if obj_l is None:continue
            
            if obj_c.is_thing == obj_l.is_thing:
                obj_l.fuse(obj_c)
                if obj_c.oid is None:
                    # obj_l.add_frame(f_cur,j)
                    obj_l.update_points_by_frame(f_cur)
                print(f'success retrack obj:{obj_l.oid},{obj_c.oid}')
        
        
        # self.obj_tracker.update_objects(objs_l, masks_l, masks_c,matches,from_project=True)
        
        # self.update_obj_tracker(objs_l=objects, masks_l=masks_l, matches=matches,
        #                         save_obj=False, add_miss_counter=False)
        
        
        return id_map
    
    
    def retrack_object_by_project(self, f_cur:'Frame',obj_l:Thing|Stuff, iou_thr:float=0.6):
        """
        目前只支援Thing，只有thing有mesh
        
        輸入多個遮罩和深度圖，輸出 id 遮罩（每像素標記為最前景遮罩的 id）。
        - masks: list of (H,W) bool 或 uint8 遮罩（True/1=物體，False/0=背景）
        - depths: list of (H,W) float32 深度圖（與遮罩同 shape，背景可設為 np.inf 或 -1）
        規則：
        - 若多個遮罩重疊，取深度值最小（離相機最近）的遮罩 id
        - 若無遮罩覆蓋，id = -1
        回傳：
        - id_map: (H,W) int32，每像素為遮罩 id（0~N-1），未覆蓋為 -1
        """
        
        height, width = f_cur.camera.height, f_cur.camera.width
        id_map = np.full((height, width), -1, dtype=np.int32)
        min_depth = np.full((height, width), np.inf, dtype=np.float32)
        
        
        if obj_l.is_bad or len(obj_l.faces) == 0 or (obj_l in f_cur.objects):
            
            return
        
        # 將物體投影產生可視遮罩
        mask_l, depth = f_cur.project_mesh_to_img(obj_l.vertices,obj_l.faces)
        
        
        
        # 比對mask判斷投影遮罩，判斷是否同一物體
        masks_c = [info['mask'] for info in f_cur.obj_infos]
        objs_c:list[ObjectBase] = list()
        for idx, obj_c in enumerate(f_cur.objects):
            if obj_c is None:
                obj_c = ObjectBase._temp_obj(f_cur, idx)
            
            objs_c.append(obj_c)
        
        iou_mat = self.compute_IoU_mat([obj_l],[mask_l],objs_c,masks_c)
        matches = ObjectMatcher.greedy_match(iou_mat,iou_thr)
        
        
        i,j,_ = matches[0]
        obj_l
        obj_c = objs_c[j]
        if obj_c.is_thing == obj_l.is_thing:
            obj_l.fuse(obj_c)
            if obj_c.oid is None:
                # obj_l.add_frame(f_cur,j)
                obj_l.update_points_by_frame(f_cur)
            print(f'success retrack obj:{obj_l.oid},{obj_c.oid}')
        

        
        self.obj_tracker.update_objects([obj_l],[mask_l], masks_c,matches)
        
        # self.update_obj_tracker(objs_l=objects, masks_l=masks_l, matches=matches,
        #                         save_obj=False, add_miss_counter=False)
        
        
        return id_map
    
    def check_object_dynamic_old(self,kf_cur,frame_num:int=10):
        """在切割或產生hull前檢查物體是否開始移動或停止移動
        
        """
        
        # kf_cur = self.map.keyframes[-1]
        # kf_last = self.map.keyframes[2]
        frames = list(self.map.frames)[-frame_num:]#[::-1] # 取出最近10個frame，並根據加入時間排序（越近越前面），判斷這個物體是否動態
        objects:set[Thing | Stuff | None] = set()
        for f in frames:
            objects.update(f.objects)
        
        for obj in objects:
            if obj is None:continue
            if not obj.is_thing or obj.is_bad:
                continue
            obj:Thing
            # 統計每幀的動態判斷
            dynamics = []
            last_dynamic = None
            for f in frames:
                obj_idxs = obj.frames[f]
                # obj_idx = list(obj_idxs)[0]
                
                # dynamic = f.objects_dynamic[obj_idx]
                # score = f.objects_dynamic_score[obj_idx]
                
                
                
                
                dynamic = False
                l_dynamic = [f.objects_dynamic[obj_idx]  for obj_idx in obj_idxs]
                l_score = [f.objects_dynamic_score[obj_idx]  for obj_idx in obj_idxs]
                
                
                if None in l_dynamic or not l_dynamic:
                    dynamic = last_dynamic # 如果無法判斷動態狀態(None)，視為與上一frame相同
                if True in l_dynamic:
                    dynamic = True
                
                dynamics.append(dynamic)
                last_dynamic = dynamic
            # print('dynamics',dynamics)
            
            #TODO: 當最近幾幀都是True時提早判斷成不確定
            dynamics = dynamics[::-1] # 由新到舊排序
            counts = Counter(dynamics[:2])
            need_update = False
            for dy in dynamics[3:]:
                counts[dy] += 1
                # print(f'oid:{obj.oid}, {counts}')
                
                if obj.dynamic:
                    # 檢查是否停止移動
                    if counts[False] >= counts[True] + counts[None]:
                        need_update = obj.set_dynamic(kf_cur.id,False)
                        break
                        # obj.dynamic = False
                        # obj.static_time = kf_cur.id
                    elif counts[True] >= counts[False] + counts[None]:
                        break
                        
                    
                else: #None or False
                    # 檢查是否開始移動
                    
                        
                    
                    if counts[True] >= counts[False] + counts[None]:
                        need_update = obj.set_dynamic(kf_cur.id,True)
                        break
                        # obj.dynamic = True
                        # obj.dynamic_time = kf_cur.id
                        # print(f'object: {obj.oid} is dynamic')
                    
                    
                    elif counts[None] >= counts[True] + counts[False]:
                        need_update = obj.set_dynamic(kf_cur.id,None)
                        break
                    
                    else:
                        need_update = obj.set_dynamic(kf_cur.id,False)
                        break
            
            
            if need_update:
                print(f'object: {obj.oid} is dynamic:{obj.dynamic} points:{len(obj.points)}')
                if obj.dynamic:
                    # if len(obj.points) <10: #擁有特徵點太少的obj不太可能恢復
                    if obj.volume is None:
                        print('del',obj.oid)
                        obj.delete()
                        
                    else:
                        obj.dynamic_update()
                elif not obj.dynamic:
                    obj.need_to_static = True
                    # obj.static_update(kf_cur)
            
        return objects
        
    def check_object_dynamic(self, kf_cur:'KeyFrame', frame_num:int=10):
        """
        使用滑動窗口 + 動態分數判斷物體狀態
        
        狀態轉換規則：
        1. 靜態 → 疑似：連續 2 幀 None 或 連續 2 幀分數 0.3-0.5
        2. 靜態 → 動態：連續 2 幀分數 > 0.6（罕見但允許）
        3. 疑似 → 動態：連續 2 幀分數 > 0.5 或 任意 1 幀 True
        4. 疑似 → 靜態：連續 3 幀分數 < 0.3 或 連續 3 幀 False
        5. 動態 → 靜態：連續 4 幀分數 < 0.3（直接轉換）
        """
        frames = list(self.map.frames)[-frame_num:]
        objects:set[Thing | Stuff | None] = set()
        for f in frames:
            objects.update(f.objects)
        
        for obj in objects:
            if obj is None or not obj.is_thing or obj.is_bad:
                continue
            
            obj:Thing
            
            # 收集時序動態判斷與分數
            dynamics = []
            scores = []
            for f in frames:
                obj_idxs = obj.frames.get(f, set())
                if not obj_idxs:
                    continue
                
                # 取該物體在該幀的所有遮罩中最嚴重的判斷
                frame_dynamics = [f.objects_dynamic[idx] for idx in obj_idxs]
                frame_scores = [f.objects_dynamic_score[idx] for idx in obj_idxs if f.objects_dynamic_score[idx] is not None]
                
                # 動態判斷：只要有一個 True 就是 True
                if True in frame_dynamics:
                    dynamics.append(True)
                elif None in frame_dynamics:
                    dynamics.append(None)
                else:
                    dynamics.append(False)
                
                # 動態分數：取最大值（最嚴重）
                if frame_scores:
                    scores.append(max(frame_scores))
                else:
                    scores.append(None)
            
            if len(dynamics) < 3:
                obj.set_dynamic(kf_cur.id, obj.dynamic)
                continue
            
            # 由新到舊排序
            dynamics = dynamics[::-1]
            scores = scores[::-1]
            
            # 判斷新狀態
            new_dynamic = obj.dynamic
            
            # ==================== 當前為動態 ====================
            if obj.dynamic is True:
                # 動態 → 靜態：連續 4 幀分數 < 0.3
                consecutive_low = 0
                for s in scores[:5]:
                    if s is not None and s < 0.3:
                        consecutive_low += 1
                        if consecutive_low >= 4:
                            new_dynamic = False
                            break
                    else:
                        break
            
            # ==================== 當前為疑似 ====================
            elif obj.dynamic is None:
                # 疑似 → 動態：任意 1 幀 True 或 連續 2 幀分數 > 0.5
                for i in range(min(3, len(dynamics))):
                    if dynamics[i] is True:
                        new_dynamic = True
                        break
                
                if new_dynamic is None:
                    consecutive_high = 0
                    for s in scores[:3]:
                        if s is not None and s > 0.5:
                            consecutive_high += 1
                            if consecutive_high >= 2:
                                new_dynamic = True
                                break
                        else:
                            break
                
                # 疑似 → 靜態：連續 3 幀分數 < 0.3 或 連續 3 幀 False
                if new_dynamic is None:
                    consecutive_low = 0
                    for s in scores[:4]:
                        if s is not None and s < 0.3:
                            consecutive_low += 1
                            if consecutive_low >= 3:
                                new_dynamic = False
                                break
                        else:
                            break
                    
                    if new_dynamic is None:
                        consecutive_false = 0
                        for d in dynamics[:4]:
                            if d is False:
                                consecutive_false += 1
                                if consecutive_false >= 3:
                                    new_dynamic = False
                                    break
                            else:
                                break
            
            # ==================== 當前為靜態 ====================
            else:  # obj.dynamic is False
                # 靜態 → 動態：連續 2 幀分數 > 0.6（罕見）
                consecutive_very_high = 0
                for s in scores[:3]:
                    if s is not None and s > 0.6:
                        consecutive_very_high += 1
                        if consecutive_very_high >= 2:
                            new_dynamic = True
                            break
                    else:
                        break
                
                # 靜態 → 疑似：連續 2 幀 None 或 連續 2 幀分數 0.3-0.5
                if new_dynamic is False:
                    consecutive_none = 0
                    for d in dynamics[:3]:
                        if d is None:
                            consecutive_none += 1
                            if consecutive_none >= 2:
                                new_dynamic = None
                                break
                        else:
                            break
                    
                    if new_dynamic is False:
                        consecutive_mid = 0
                        for s in scores[:3]:
                            if s is not None and 0.3 <= s <= 0.5:
                                consecutive_mid += 1
                                if consecutive_mid >= 2:
                                    new_dynamic = None
                                    break
                            else:
                                break
            
            need_update = obj.set_dynamic(kf_cur.id, new_dynamic)
        
            # 更新物體
            if need_update:
                print(f'object: {obj.oid} dynamic:{obj.dynamic} points:{len(obj.points)}')
                if obj.dynamic is True:
                    if obj.volume is None:
                        print('del', obj.oid)
                        obj.delete()
                    else:
                        obj.dynamic_update()
                elif obj.dynamic is False:
                    obj.need_to_static = True
        
        return objects
        
    def compute_IoU_mat(self, objs_l:list[None|Thing|Stuff], masks_l,
                              objs_c:list[None|Thing|Stuff], masks_c,
                              )->np.ndarray[any,np.float32]:
        
        """
        計算兩組物件遮罩之間的 IoU（Intersection over Union）矩陣。
        
        參數：
            objs_l: list[None|Thing|Stuff]，上一幀（或來源）物件列表
            masks_l: list[np.ndarray]，上一幀（或來源）物件在當前幀對應的遮罩
            objs_c: list[None|Thing|Stuff]，當前幀（或目標）物件列表
            masks_c: list[np.ndarray]，當前幀物件對應的遮罩
        回傳：
            iou_mat: np.ndarray，形狀 (len(objs_l), len(objs_c))，每個元素為對應物件遮罩的 IoU 值
        """
        
        Nl, Nc = len(objs_l), len(objs_c)
        
        iou_mat = np.zeros((Nl, Nc), dtype=np.float32)
            
        for i in range(Nl):
            if objs_l[i] is None:
                iou_mat[i, :] = 0.0
                continue
            
            for j in range(Nc):
                # if j == 0: continue # 0 為分割失敗區域
                if objs_c[j] is None:
                    iou_mat[i, j] = 0.0
                    continue
                
                # 若物件類別不同，直接設 0（不考慮配對）
                
                if objs_l[i].is_thing != objs_c[j].is_thing:
                    iou_mat[i, j] = 0.0
                    continue
                A = masks_l[i]
                B = masks_c[j]
                inter = np.count_nonzero(A & B)
                union = np.count_nonzero(A | B)
                iou_mat[i, j] = inter / (union + 1e-6) # 避免除0
        
        return iou_mat
        
    # def greedy_match(self,iou_mat:np.ndarray[any, np.float32],iou_thr:float)->list[tuple[int,int,float]]:
    #     """
    #     根據 IoU 矩陣進行物件遮罩的貪婪配對。
        
    #     參數：
    #         iou_mat: np.ndarray，IoU 矩陣，形狀為 (N_l, N_c)，每個元素代表上一幀物件與當前幀物件的遮罩重疊程度
    #         iou_thr: float，IoU 配對門檻值，只有 IoU 大於此值才會配對
    #     回傳：
    #         matches: list[tuple[int,int,float]]，配對結果，每個元素為 (上一幀物件索引, 當前幀物件索引, IoU值)
    #     """
    #     # 貪婪配對：每次挑最大 IoU
    #     matches = []
    #     while True:
    #         # 找目前最大的 IoU
    #         i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
    #         best = float(iou_mat[i, j])
    #         if best < iou_thr:
    #             break
    #         matches.append((i, j, best))
    #         # 將該列/行設為 -1，避免重覆配
    #         iou_mat[i, :] = -1.0
    #         iou_mat[:, j] = -1.0
            
    #     return matches
    
    
    def predict_object(self,frame:'Frame'):
        '''
        替frame中未追蹤的obj分割建立地圖obj
        '''
        obj_infos = frame.obj_infos
        # min_area = int(frame.camera.height*frame.camera.width /100) #最少佔據畫面1/100的空間才建立新obj
        min_area = 0
        
        for idx_obj,info in enumerate(obj_infos):
            if idx_obj == 0:continue
            obj = frame.objects[idx_obj]
            
            if obj is None:
                obj = self.creat_object(frame,idx_obj,min_area=min_area)
                if obj is None:continue
                
                print('creat obj',obj.oid)
                self.obj_tracker.update_object(obj,obj.mask(frame))
    
    
    def remap_mask(self,mask:np.ndarray[int,bool],map_x: np.ndarray, map_y: np.ndarray):
        # 映射遮罩
        mask = mask.astype(np.uint8)*255
        
        warped = cv2.remap(mask,
                    map_x, map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # 填補放大時造成的洞
        
        
        # # 洪水填充演算法 # 有bug 如果遮罩佔據（0,0） 會填充整張圖片
        # h, w = warped.shape
        # flooded = warped.copy()
        # ff_mask = np.zeros((h+2, w+2), np.uint8)
        # cv2.floodFill(flooded, ff_mask, (0,0), 255)  # 外部背景→白
        # holes = cv2.bitwise_not(flooded)             # 255 = 洞
        
        # warped =  cv2.bitwise_or(warped, holes)
        
        # 補裂縫
        warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        
        return  warped > 127
    
    
    
    def track_object_by_box(self,f1:'Frame',f2:'Frame'):
        """
        TODO:根據影像物件框追蹤，可以在point批配之前執行，輔助點批配
        """
        
        ...
        
    def object_fusion(self,obj:Thing|Stuff):
        # 找出points重疊率很高的obj
        # 一組 points 出現在多個obj的原因:
        #   1.point 被誤判歸屬
        #   2.多個obj是一個物體
        #   3.一個obj是多個物體
        
        similar_objects = set()
        for pt in obj.points:
            similar_objects.update(pt.objects)
        
        for similar_object in similar_objects:
            if similar_object is None:continue
            if obj.is_thing != similar_object.is_thing:continue
            
            similar_object:Thing|Stuff
            
            intersection = obj.points & similar_object.points # 交集
            
            distance = np.linalg.norm(obj.location_3D-similar_object.location_3D)
    
    
    def objects_fusion(self,objects:list[Thing|Stuff]):
        
        for obj in objects:
            self.object_fusion(obj)
        
        
    def track_object_by_OpticalFlow_old(self,f_cur:'Frame',f_last:'Frame', flow, iou_thr:float=0.4):
        """
        根據影像光流追蹤，可以在point批配之前執行，輔助點批配
        """
        
        # img_cur = cv2.cvtColor(f_cur.img,cv2.COLOR_BGR2GRAY)
        # img_last = cv2.cvtColor(f_last.img,cv2.COLOR_BGR2GRAY)
        # flow = self.dis.calc(img_last,img_cur, None,)
        
        # H,W = f_cur.camera.height, f_cur.camera.width
        H,W = flow.shape[0], flow.shape[1]
        # assert f_last.img.shape[:2] == (H, W) and f_cur.img.shape[:2] == (H, W),  ('flow',H, W, f_last.img.shape,f_cur.img.shape)
        
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (xx - flow[:,:,0]).astype(np.float32)
        map_y = (yy - flow[:,:,1]).astype(np.float32)
        
        objects_unmatch = f_last.objects.copy()
        # 產生遮罩映射後的遮罩
        masks_remap = []
        for info in f_last.obj_infos:
            mask = info['mask']
            mask_remap = self.remap_mask(mask,map_x,map_y)
            masks_remap.append(mask_remap)
            
        # 加入過去追蹤失敗obj
        for obj,info in self.obj_track_dict.items():
            if obj in objects_unmatch:continue
            objects_unmatch.append(obj)
            mask = info['mask']
            mask_remap = self.remap_mask(mask,map_x,map_y)
            masks_remap.append(mask_remap)
        
        masks_cur = [info['mask'] for info in f_cur.obj_infos]
        
            
        
        Nl, Nc = len(objects_unmatch), len(f_cur.objects)
        if Nl <= 1 or Nc <= 1:
            # 跳過IoU判斷
            matches = []
        
        else:
            
            # 建 IoU 矩陣
            objs_c = list()
            for idx, obj_c in enumerate(f_cur.objects):
                if obj_c is None:
                    obj_c = ObjectBase._temp_obj(f_cur, idx) # 建立暫時obj，之後與正式的thing|stuff合併或刪除
                objs_c.append(obj_c)
            
            iou_mat = self.compute_IoU_mat(objects_unmatch, masks_remap, objs_c, masks_cur)
            
            # 貪婪配對：每次挑最大 IoU
            matches = ObjectMatcher.greedy_match(iou_mat,iou_thr)
            
            # # 將f_cur 的點分群
            # point_groups=[list() for n in range(len(f_cur.obj_infos))]
            # mask_id = f_cur.img_info['mask_id']
            # for raw_kp,point in zip(f_cur.raw_kps,f_cur.points):
            #     if point is None: continue
            #     x, y =raw_kp
            #     point_groups[mask_id[y][x]].append(point)
            
            
            for i,j,_ in matches:
                obj = objects_unmatch[i]
                if obj is None:continue
                obj:Stuff|Thing
                
                obj.add_frame(f_cur,j)
        
        # 更新物件追蹤器
        self.update_obj_tracker_old(objects_unmatch,masks_remap,matches)
        

    def retrack_objects_by_project_old(self, f_cur:'Frame',objects:list[Thing|Stuff], iou_thr:float=0.6):
        """
        目前只支援Thing，只有thing有mesh
        
        輸入多個遮罩和深度圖，輸出 id 遮罩（每像素標記為最前景遮罩的 id）。
        - masks: list of (H,W) bool 或 uint8 遮罩（True/1=物體，False/0=背景）
        - depths: list of (H,W) float32 深度圖（與遮罩同 shape，背景可設為 np.inf 或 -1）
        規則：
        - 若多個遮罩重疊，取深度值最小（離相機最近）的遮罩 id
        - 若無遮罩覆蓋，id = -1
        回傳：
        - id_map: (H,W) int32，每像素為遮罩 id（0~N-1），未覆蓋為 -1
        """
        
        height, width = f_cur.camera.height, f_cur.camera.width
        id_map = np.full((height, width), -1, dtype=np.int32)
        min_depth = np.full((height, width), np.inf, dtype=np.float32)
        
        
        if len(objects)==0:
            return id_map
        
        masks_l = []
        # 將物體投影產生可視遮罩
        for obj_l in objects:
            
            if len(obj_l.faces) == 0 or (obj_l in f_cur.objects):
                mask = np.zeros((height, width),dtype=np.uint8)
                masks_l.append(mask)
                continue
            
            mask, depth = f_cur.project_mesh_to_img(obj_l.vertices,obj_l.faces)
            
            # 找比目前 min_depth 更小的像素
            update = depth < min_depth
            id_map[update] = obj_l.oid
            min_depth[update] = depth[update]
            
            masks_l.append(mask)
        
        
        f_cur.obj_prj_id_map = id_map
        f_cur.obj_prj_depth_map = min_depth
        
        
        # 比對mask判斷投影遮罩，判斷是否同一物體
        masks_c = [info['mask'] for info in f_cur.obj_infos]
        objs_c = list()
        for idx, obj_c in enumerate(f_cur.objects):
            if obj_c is None:
                obj_c = ObjectBase._temp_obj(f_cur, idx)
            
            objs_c.append(obj_c)
        
        iou_mat = self.compute_IoU_mat(objects,masks_l,objs_c,masks_c)
        matches = ObjectMatcher.greedy_match(iou_mat,iou_thr)
        
        
        for i,j,_ in matches:
            obj_l = objects[i]
            obj_c = objs_c[j]
            # if obj_l is None:continue
            
            if obj_c.is_thing == obj_l.is_thing:
                obj_l.fuse(obj_c)
                if obj_l.oid is None:
                    obj_l.add_frame(f_cur,j)
                    obj_l.update_points_by_frame(f_cur)
                print(f'success retrack obj:{obj_l.oid},{obj_c.oid}')
            
        
        
        
        self.update_obj_tracker_old(objs_l=objects, masks_l=masks_l, matches=matches,
                                save_obj=False, add_miss_counter=False)
        
        
        return id_map
    
    
    
    def update_obj_tracker_old(self,objs_l,masks_l,matches,save_obj:bool=True,add_miss_counter:bool=True):

        """
        參數：
            objs_l: list[None|Thing|Stuff]，上一幀（或來源）物件列表
            masks_l: list[np.ndarray]，上一幀（或來源）物件在當前幀對應的遮罩
            matches: list[tuple[int,int,float]]，物件配對結果 (上一幀索引, 當前幀索引, IoU值)
            save_obj: bool，是否儲存追蹤失敗物件
            add_miss_counter: bool，是否累加失敗次數，一幀應該只在第一次追蹤時添加一次
        
        """
        
        # 既有：matches 產生完畢（[(i,j,iou), ...]；i,j 是原索引）
        matched_prev = {i for (i, _, _) in matches}
        # matched_curr = {j for (_, j, _) in matches}
        
        for i, obj in enumerate(objs_l):
            info = self.obj_track_dict[obj]
            
            if i in matched_prev:
                info['missed'] = 0
            
            else:
        
                # warped mask 
                mask_warped = masks_l[i]
                if np.count_nonzero(mask_warped) == 0:
                    continue  # 完全空就不續命
                
                info = self.obj_track_dict.get(obj, {"mask": None, "missed": 0})
                info["mask"] = mask_warped.copy()
        
        
        
        
        # 1) 更新「有配到」的：清掉 lost 狀態
        for i in matched_prev:
            obj = objs_l[i]
            if obj in self.obj_track_dict:
                del self.obj_track_dict[obj]
                
            
        # 2) 對「沒配到」的上一幀物件：用 warp 遮罩暫存並遞增 missed
        # for i in range(Nl):
        for i, obj in enumerate(objs_l):
            if i in matched_prev:
                continue
            if obj is None:
                continue
            # warped mask 
            mask_warped = masks_l[i]
            if np.count_nonzero(mask_warped) == 0:
                continue  # 完全空就不續命
            
            info = self.obj_track_dict.get(obj, {"mask": None, "missed": 0})
            info["mask"] = mask_warped.copy()
            
            if add_miss_counter:
                info["missed"] += 1
            if save_obj:
                self.obj_track_dict[obj] = info
            
        # 3) 清理過期的 lost
        if add_miss_counter:
            # to_delete = [obj for obj, info in self.obj_tracker.items() if info["missed"] > self.max_missed]
            # for obj in to_delete:
            #     del self.obj_tracker[obj]
            self.obj_tracker_clean_old_lost()
    
    
    # def update_obj_tracker_one_obj(self,obj_l,mask_l,save_obj:bool=True,add_miss_counter:bool=True):

    #     """
    #     參數：
    #         objs_l: list[None|Thing|Stuff]，上一幀（或來源）物件列表
    #         masks_l: list[np.ndarray]，上一幀（或來源）物件在當前幀對應的遮罩
    #         matches: list[tuple[int,int,float]]，物件配對結果 (上一幀索引, 當前幀索引, IoU值)
    #         save_obj: bool，是否儲存追蹤失敗物件
    #         add_miss_counter: bool，是否累加失敗次數，一幀應該只在第一次追蹤時添加一次
        
    #     """
        
    #     # 既有：matches 產生完畢（[(i,j,iou), ...]；i,j 是原索引）
    #     matched_prev = {i for (i, _, _) in matches}
    #     matched_curr = {j for (_, j, _) in matches}
        
    #     # 1) 更新「有配到」的：清掉 lost 狀態
    #     for i in matched_prev:
    #         obj = objs_l[i]
    #         if obj in self.obj_tracker:
    #             del self.obj_tracker[obj]
            
    #     # 2) 對「沒配到」的上一幀物件：用 warp 遮罩暫存並遞增 missed
    #     # for i in range(Nl):
    #     for i, obj in enumerate(objs_l):
    #         if i in matched_prev:
    #             continue
    #         if obj is None:
    #             continue
    #         # warped mask 
    #         mask_warped = masks_l[i]
    #         if np.count_nonzero(mask_warped) == 0:
    #             continue  # 完全空就不續命
            
    #         info = self.obj_tracker.get(obj, {"mask": None, "missed": 0})
    #         info["mask"] = mask_warped.copy()
            
    #         if add_miss_counter:
    #             info["missed"] += 1
    #         if save_obj:
    #             self.obj_tracker[obj] = info
            
    #     # 3) 清理過期的 lost
    #     if add_miss_counter:
    #         # to_delete = [obj for obj, info in self.obj_tracker.items() if info["missed"] > self.max_missed]
    #         # for obj in to_delete:
    #         #     del self.obj_tracker[obj]
    #         self.obj_tracker_clean_old_lost()
    
    def obj_tracker_clean_old_lost(self,th:int=0):
        if th == 0:
            th = self.max_missed
        
        to_delete = [obj for obj, info in self.obj_track_dict.items() if info["missed"] > th]
        
        for obj in to_delete:
            del self.obj_track_dict[obj]
        
    
    def obj_tracker_get_info(self,obj:Thing|Stuff)->tuple[np.ndarray,int]:
        
        info = self.obj_track_dict.get(obj, {"mask": None, "missed": 0})
        
        return info['mask'],info['missed']
        
        
    

def is_ccw(poly:np.ndarray)->bool:
    # 用多邊形面積判斷是否逆時針方向
    x, y = poly[:,0], poly[:,1]
    area = np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
    return area > 0

def ensure_ccw(poly:np.ndarray)->np.ndarray:
    if not is_ccw(poly):
        poly = poly[::-1].copy()
    return poly
def ensure_cw(poly:np.ndarray)->np.ndarray:
    if is_ccw(poly):
        poly = poly[::-1].copy()
    return poly



def mask_to_polygons(mask: np.ndarray,
                        min_area: int = None,
                        approx_epsilon_ratio: float = 0.01,
                        keep_holes: bool = True):
    """
    將二值遮罩轉為多邊形集合（外環 + 可選洞）
    - mask: 0/255 或 0/1
    - min_area: 面積過小的連通域/洞會被忽略；None 不檢查
    - approx_epsilon_ratio: 多邊形近似強度（周長比例）
    - keep_holes: 是否保留洞
    回傳:
        polys: list[np.ndarray(N,2)] 外環（均為 CCW）
        holes: list[list[np.ndarray(M,2)]] 對應外環的洞（均為 CW）
    """
    # 1) 轉 uint8 & 簡單去噪
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2) 找輪廓（RETR_CCOMP 可拿外環與其 children 洞）
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [],[]
    hierarchy = hierarchy[0]  # (N,4): [next, prev, child, parent]

    polys, holes = [], []

    # 3) 逐外環（parent == -1）
    for idx, h in enumerate(hierarchy):
        if h[3] != -1:
            continue  # 只處理外環

        cnt = contours[idx]
        if isinstance(min_area, int):
            area = abs(cv2.contourArea(cnt))
            if area < min_area:
                continue

        # 外環多邊形近似
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, approx_epsilon_ratio * peri)
        ext = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

        # 4) 收集其所有洞（child）
        children_polys = []
        if keep_holes:
            visited = set()
            child = h[2]
            while child != -1 and child not in visited:
                visited.add(child)
                next_child = hierarchy[child][0]

                hole_cnt = contours[child]
                hole_area = abs(cv2.contourArea(hole_cnt))

                # 只在通過 min_area 門檻時才建立 / 追加
                if (not isinstance(min_area, int)) or (hole_area >= min_area):
                    peri_h = cv2.arcLength(hole_cnt, True)
                    eps_h = max(1.0, approx_epsilon_ratio * peri_h)
                    hole_poly = cv2.approxPolyDP(hole_cnt, eps_h, True).reshape(-1, 2)
                    children_polys.append(hole_poly)

                child = next_child

        # 5) 統一方向：外環 CCW、洞 CW
        if not is_ccw(ext):
            ext = ext[::-1].copy()
        fixed_holes = []
        for hole in children_polys:
            if is_ccw(hole):
                hole = hole[::-1].copy()
            fixed_holes.append(hole.astype(np.int32))

        polys.append(ext.astype(np.int32))
        holes.append(fixed_holes)
        # print(polys,holes)

    return polys, holes




# def _clean_mesh_inplace(mesh:o3d.geometry.TriangleMesh):
#     """
#     清理網格：移除重覆/退化/未引用元素與非流形邊，並重算法向。
#     """
#     mesh.remove_duplicated_vertices()
#     mesh.remove_duplicated_triangles()
#     mesh.remove_degenerate_triangles()
#     mesh.remove_unreferenced_vertices()
#     mesh.remove_non_manifold_edges()
#     mesh.compute_vertex_normals()
    

def project_poly_to_z(frame:'Frame', poly:np.ndarray, z:float = 1) -> np.ndarray:
    """
    將像素多邊形投影到相機座標平面 z，再轉換到世界座標。
    參數：
      poly: (N,2) 像素座標頂點序列
      z:    相機座標系的深度（例如 z_far）
    回傳：
      (N,3) 世界座標上的點
    """
    # poly = np.array(poly, dtype=np.float64)  # 確保 poly 是 NumPy 陣列
    
    Cw  = frame.twc.astype(np.float64).reshape(3)
    Rwc = frame.Rwc.astype(np.float64)
    uvn = frame.camera.normalize(poly.astype(np.float64))  # 標準化影像座標 (x,y) at z=1
    Pcam = np.c_[uvn * z, np.full(len(uvn), z, dtype=np.float64)]
    Pw   = (Rwc @ Pcam.T).T + Cw
    return Pw


def view_frustum_union(frame:'Frame',
                       exteriors: list[np.ndarray],
                       holes_per_ext: list[list[np.ndarray]],
                       z_far: float) -> Mesh:
    """
    將同一視角中的多個外環（每個外環可有多個洞）建立錐台後做聯集，回傳單一網格。
    """
    parts = []
    for ext_xy, holes_xy in zip(exteriors, holes_per_ext):
        # 產生單一外環的錐台
        Cw = frame.twc.astype(np.float64).reshape(3)
        cap_ext  = project_poly_to_z(frame, ext_xy, z_far)
        mesh = Mesh.from_cone_side(Cw, cap_ext)
        # 根據holes切割錐台
        for h in holes_xy:
            h3d = project_poly_to_z(frame, h, z_far)
            hole_side = Mesh.from_cone_side(Cw, h3d)
            try:
                # _clean_mesh_inplace(mesh)
                mesh = mesh.difference(hole_side)
                mesh.clean()
            except Exception:
                # 少數情況布林不穩，跳過此洞差集
                pass
        # _clean_mesh_inplace(mesh)
        mesh.clean()
        
        if not mesh.is_empty:
            parts.append(mesh)
    
    if not parts:
        return Mesh()
    
    # 多個外環mesh聯集成單一mesh
    union_mesh:Mesh = parts[0]
    for m in parts[1:]: 
        # if len(m.vertices) == 0: 
        if mesh.is_empty: 
            continue
        union_mesh = union_mesh.union(m)
        
    return union_mesh

# ------------------------------------------------------------
# 建立「影像視野(FOV)錐台」：使用整張影像矩形作為外環
# ------------------------------------------------------------
def build_fov_frustum(frame:'Frame', z_far: float) -> Mesh:
    """
    以整張影像的外框矩形（像素座標）建立「FOV 錐台」側面網格。
    這代表「相機可見的整個視域」。
    """
    H, W = frame.camera.height, frame.camera.width
    # 影像外框（順/逆時針皆可；布林會處理）
    rect = np.array([[0, 0],
                     [W-1, 0],
                     [W-1, H-1],
                     [0, H-1]], dtype=np.float64)
    Cw = frame.twc.astype(np.float64).reshape(3)
    cap_rect = project_poly_to_z(frame, rect, z_far)
    # fov_mesh = _cone_side_mesh(Cw, cap_rect)
    # _clean_mesh_inplace(fov_mesh)
    fov_mesh = Mesh.from_cone_side(Cw, cap_rect)
    return fov_mesh

# # ------------------------------------------------------------
# # 將「負向錐台聯集」切成多個小 mesh（以三角面連通群分割）
# # ------------------------------------------------------------
# def split_connected_submeshes(mesh:o3d.geometry.TriangleMesh) -> list[o3d.geometry.TriangleMesh]:
#     """
#     將輸入 TriangleMesh 依「三角面連通」分割成多個子網格。
#     回傳：每個子網格皆為獨立的 TriangleMesh。
#     """
#     if len(mesh.triangles) == 0:
#         return []
#     # 取得每個三角面所屬群集標籤
#     labels, counts, _ = mesh.cluster_connected_triangles()
#     labels = np.asarray(labels)
#     submeshes = []
#     for lab in np.unique(labels):
#         tri_idx = np.where(labels == lab)[0]
#         if len(tri_idx) == 0:
#             continue
#         # 取出對應三角形，建立子網格（需重新壓縮頂點索引）
#         tris = np.asarray(mesh.triangles)[tri_idx]
#         verts = np.asarray(mesh.vertices)
#         # 找到會用到的頂點
#         used_vid = np.unique(tris.flatten())
#         vid_map = {int(v): i for i, v in enumerate(used_vid)}
#         # 重新映射三角形索引
#         tris_remap = np.vectorize(lambda x: vid_map[int(x)])(tris)
#         sub = o3d.geometry.TriangleMesh(
#             vertices=o3d.utility.Vector3dVector(verts[used_vid]),
#             triangles=o3d.utility.Vector3iVector(tris_remap.astype(np.int32))
#         )
#         _clean_mesh_inplace(sub)
#         if len(sub.triangles) > 0:
#             submeshes.append(sub)
#     return submeshes


# ------------------------------------------------------------
# 逐視角雕刻（可控負向 + 遮擋回呼介面）
# ------------------------------------------------------------
def carve_mesh( union_mesh:Mesh,
                frame:'Frame',
                exts: list[np.ndarray], holes: list[list[np.ndarray]],
                z_far: float,
                obscured_exts: list[np.ndarray]=[], obscured_holes: list[list[np.ndarray]]=[],
                ) -> Mesh:
    """
    從聯集網格中，依本視角的「負向」區域進行雕刻（布林差集）。
    - 若提供 occlusion_callback，會先以 (當前網格, 當前視角, 負向錐台網格) 呼叫，
      僅當回傳 True 才執行差集；False 代表可能被遮擋或不可信，則跳過。
      
      以本視角的正向多邊形 exts/holes 建立「負向錐台」候選集合（多個小 mesh）：
      1) 正向錐台聯集：pos = view_frustum_union(frame, exts, holes_per_ext, z_far)
      2) FOV 錐台：fov = build_fov_frustum(frame, z_far)
      3) 負向聯集：neg = fov.boolean_difference(pos)
      4) 連通分割：將 neg 切成多個小 mesh，回傳列表
      
      
      
    """
    
    # 正向（前景）錐台聯集
    pos = view_frustum_union(frame, exts, holes, z_far)
    # FOV 錐台（整張影像外框）
    fov = build_fov_frustum(frame, z_far)

    # 確定被遮擋的錐台聯集，視為正向，不會被切除
    obscured = view_frustum_union(frame, obscured_exts, obscured_holes, z_far)
    
    # if len(obscured.vertices) > 0:
    #     pos = pos.boolean_union(obscured)
    #     _clean_mesh_inplace(pos)
    if not obscured.is_empty:
        pos = pos.union(obscured)
        pos.clean()
    
    try:
        # neg_union = fov.boolean_difference(pos)
        neg_union = fov.difference(pos)
    except Exception:
        # 布林失敗時，保守做法：視為沒有可靠負向候選
        # neg_union = o3d.geometry.TriangleMesh()
        neg_union = Mesh()
    
    # _clean_mesh_inplace(neg_union)
    # parts = split_connected_submeshes(neg_union)
    neg_union.clean()


    # 遮擋回呼：允許才進行布林差集
    def occlusion_callback(
        union_mesh:o3d.geometry.TriangleMesh,
        frame:'Frame',
        neg_frustum:o3d.geometry.TriangleMesh)->bool:
        ...
        

    parts = neg_union.split_connected()
    for mesh in parts:
        # TODO(未來遮擋/合理性判斷在此加入)：
        # 例如： 加入occlusion_callback判斷
        # if not your_visibility_or_validity_test(out, frame, neg_part):
        #     continue
        try:
            # union_mesh = union_mesh.boolean_difference(mesh)
            # _clean_mesh_inplace(union_mesh)
            union_mesh = union_mesh.difference(mesh)
            union_mesh.clean
        except Exception:
            # 個別小塊布林失敗就跳過，避免中斷整體
            ...
    
    return union_mesh




def view_voxel_grid_union(frame:'Frame',
                        exteriors:list[np.ndarray],
                        holes_per_ext:list[list[np.ndarray]],
                        voxel_size:float,
                        z_far:float,
                        z_near:float = 0
                        ) -> VoxelGrid:
    """
    將同一視角中的多個外環（每個外環可有多個洞）建立錐台後做聯集，回傳單一體素網格(voxel grid)。
    """
    parts:list[VoxelGrid] = []
    for ext_xy, holes_xy in zip(exteriors, holes_per_ext):
        # 產生單一外環的錐台
        Cw = frame.twc.astype(np.float64).reshape(3)
        # cap_ext  = project_poly_to_z(frame, ext_xy, z_far)
        # # mesh = Mesh.from_cone_side()
        # voxel_grid = VoxelGrid.from_cone_side(Cw, cap_ext,voxel_size)
        near_ring  = project_poly_to_z(frame, ext_xy, z_near)
        far_ring  = project_poly_to_z(frame, ext_xy, z_far)
        voxel_grid = VoxelGrid.create_frustum(Cw,voxel_size, near_ring, far_ring)
        
        # 根據holes切割錐台
        for h in holes_xy:
            # h3d = project_poly_to_z(frame, h, z_far)
            # # hole_side = Mesh.from_cone_side(Cw, h3d)
            # hole_side = VoxelGrid.from_cone_side(Cw, h3d,voxel_size)
            h3d_near = project_poly_to_z(frame, h, z_near)
            h3d_far = project_poly_to_z(frame, h, z_far)
            hole_side = VoxelGrid.create_frustum(Cw,voxel_size, h3d_near, h3d_far)
            
            # try:
            #     # _clean_mesh_inplace(mesh)
            #     mesh = mesh.difference(hole_side)
            #     mesh.clean()
            # except Exception:
            #     # 少數情況布林不穩，跳過此洞差集
            #     pass
            voxel_grid = voxel_grid.diff(hole_side)
            
        # _clean_mesh_inplace(mesh)
        # mesh.clean()
        
        # if not mesh.is_empty:
        #     parts.append(mesh)
        if not voxel_grid.is_empty:
            parts.append(voxel_grid)
    
    if not parts:
        return VoxelGrid.from_voxel_size(voxel_size)
    
    # 多個外環產生的voxel_grid聯集成單一voxel_grid
    union_grid:VoxelGrid = parts[0]
    
    for g in parts[1:]: 
        # if len(m.vertices) == 0: 
        if g.is_empty: 
            continue
        union_grid = union_grid.union(g)
        
    return union_grid


def carve_voxel_grid(union_voxel_grid:VoxelGrid,
                frame:'Frame',
                exts: list[np.ndarray], holes: list[list[np.ndarray]],
                voxel_size:float,
                z_far:float,
                z_near:float=0.0,
                obscured_exts: list[np.ndarray]=[], obscured_holes: list[list[np.ndarray]]=[],
                ) -> VoxelGrid:
    """
    從聯集網格中，依本視角的「負向」區域進行雕刻（布林差集）。
    - 若提供 occlusion_callback，會先以 (當前網格, 當前視角, 負向錐台網格) 呼叫，
      僅當回傳 True 才執行差集；False 代表可能被遮擋或不可信，則跳過。
      
      以本視角的正向多邊形 exts/holes 建立「負向錐台」候選集合（多個小 mesh）：
      1) 正向錐台聯集：pos = view_frustum_union(frame, exts, holes_per_ext, z_far)
      2) FOV 錐台：fov = build_fov_frustum(frame, z_far)
      3) 負向聯集：neg = fov.boolean_difference(pos)
      4) 連通分割：將 neg 切成多個小 mesh，回傳列表
      
      TODO: 保證 z_far 覆蓋整個物體
      
    """
    
    # 正向（前景）錐台聯集
    pos = view_voxel_grid_union(frame, exts, holes, voxel_size, z_far,z_near=z_near)
    
    # FOV 錐台（整張影像外框）
    fov = build_fov_frustum(frame, z_far)
    fov = VoxelGrid.from_mesh(fov.legacy,voxel_size) # TODO: 統一形式直接建立 voxel
    
    # 確定被遮擋的錐台聯集，視為正向，不會被切除
    obscured = view_voxel_grid_union(frame, obscured_exts, obscured_holes, voxel_size, z_far, z_near=z_near)

    # if len(obscured.vertices) > 0:
    #     pos = pos.boolean_union(obscured)
    #     _clean_mesh_inplace(pos)
    if not obscured.is_empty:
        # pos = pos.union(obscured)
        # pos.clean()
        pos = pos.union(obscured)
    

    neg_union = fov.diff(pos)
    

    # parts = neg_union.split_connected()
    # for mesh in parts:
    #     # TODO(未來遮擋/合理性判斷在此加入)：
    #     # 例如： 加入occlusion_callback判斷
    #     # if not your_visibility_or_validity_test(out, frame, neg_part):
    #     #     continue
    #     try:
    #         # union_mesh = union_mesh.boolean_difference(mesh)
    #         # _clean_mesh_inplace(union_mesh)
    #         union_mesh = union_mesh.difference(mesh)
    #         union_mesh.clean
    #     except Exception:
    #         # 個別小塊布林失敗就跳過，避免中斷整體
    #         ...
    
    
    # for mesh in parts:
    #     TODO(未來遮擋/合理性判斷在此加入)：
    #     voxel_grid= VoxelGrid.from_mesh(mesh.legacy,voxel_size)

    # neg_union = neg_union.dilate(1) # 膨脹1體素，清理更乾淨
    union_voxel_grid = union_voxel_grid.diff(neg_union)
    
    return union_voxel_grid


def detect_multi_plane(pcd:o3d.geometry.PointCloud, 
                       camera_location:np.ndarray, # shape (3,)
                       distance_threshold=0.02, 
                       ransac_n=3, 
                       num_iterations=1000, 
                       min_plane_points=50,
                       )->list[tuple[np.ndarray,o3d.geometry.PointCloud]]:
    """
    迭代偵測點雲中的多個平面。
        :param pcd: 輸入的 Open3D PointCloud 物件。
        :param camera_location: 觀察者位置，用於定向法向量。
        :param distance_threshold: RANSAC 的內群點距離閾值。
        :param ransac_n: RANSAC 抽樣點數。
        :param num_iterations: RANSAC 迭代次數。
        :param min_plane_points: 形成一個平面所需的最少點數。
        
    輸出
        :return (plane, inlier_cloud):  一個包含 (平面模型, 內群點雲) 的元組列表。
        :return plane: (4,) array，格式為 [a, b, c, d]，表示 ax+by+cz+d=0
        :return inlier_cloud: o3d.geometry.PointCloud, 包含屬於此平面的點的座標
    """
    planes = []
    remaining_pcd = pcd

    while len(remaining_pcd.points) > min_plane_points:
        # 1. 偵測: 使用 RANSAC 尋找最顯著的平面
        plane_model, inlier_indices = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # 檢查是否有足夠的點形成一個平面
        if len(inlier_indices) < min_plane_points:
            print("找不到足夠的點來形成新的平面，偵測終止。")
            break

        # 2. 分割: 將點雲分為內群點和離群點
        inlier_cloud = remaining_pcd.select_by_index(inlier_indices)
        outlier_cloud = remaining_pcd.select_by_index(inlier_indices, invert=True)

        print(f"找到一個平面，包含 {len(inlier_indices)} 個點。")

        # 3. 定向: 根據觀察者位置校正法向量方向
        a, b, c, d = plane_model
        normal = np.array([a, b, c])

        # 取得平面上的一點 (使用內群點的質心)
        plane_center = np.mean(np.asarray(inlier_cloud.points), axis=0)

        # 建立從平面指向觀察者的向量
        vec_to_camera = camera_location - plane_center

        # 透過內積檢查法向量方向
        # 如果內積為負，表示法向量背對觀察者，需要翻轉
        if np.dot(normal, vec_to_camera) < 0:
            plane_model = -np.array(plane_model)

        # 4. 儲存: 保存定向後的平面模型和其內群點雲
        planes.append((plane_model, inlier_cloud))

        # 5. 削減: 更新剩餘點雲以進行下一次迭代
        remaining_pcd = outlier_cloud

    return planes




















# 以下廢棄
'''
# ---------- 2D 有向距離圖（mask 內負、外正） ----------
def signed_distance_2d(mask: np.ndarray[np.bool_]):
    """
    計算 2D 有向距離圖 (Signed Distance Function)
    mask: np.bool_ array (True=物體，False=背景)
    回傳: float32 SDF, mask內為負距, 外為正距, 邊界約為0
    """
    # OpenCV 要求 uint8，True->1, False->0
    m = mask.astype(np.uint8)

    # 內部距離（到最近背景）
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32)

    # 外部距離（到最近物體）
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)

    # mask 內 = 負距；mask 外 = 正距
    sdf = dist_out.copy()
    sdf[m == 1] = -dist_in[m == 1]

    return sdf

# ---------- 機率式雕刻 ----------
def probabilistic_visual_hull(frames:list['Frame'],masks, center:np.ndarray, half:float, bbox_min=None, bbox_max=None,
                              voxel_size=0.01,
                              tau_px=2.0,
                              theta_view=0.5,
                              theta_global=0.6,
                              min_views=2):
    """
    輸入:
        views: list of dict { "mask":(H,W) uint8, "K":3x3, "R":3x3, "t":3, }
        center: 初始正方體中心點
        side_length: 初始正方體邊長
        bbox_min, bbox_max: 3D 邊界（世界座標），numpy(3,)
        voxel_size: 體素邊長（公尺） -> 解析度調整
        tau_px:   2D 簽距轉置信度的尺度（像素）。越大=邊界容忍越高
        theta_view: 視角內 w_i 的單視角閾值
        theta_global: 全視角平均/中位數的閾值
        min_views: 至少幾個視角支持
    回傳：
        occ_grid: (Nx,Ny,Nz) float32 佔據分數（或 bool）
    """
    # # 預算：解析度
    # mins = np.array(bbox_min, dtype=np.float32)
    # maxs = np.array(bbox_max, dtype=np.float32)
    
    # dims = np.maximum(((maxs - mins) / voxel_size).astype(int), 1)
    # Nx, Ny, Nz = dims.tolist()

    # 預計算 2D SDF (Signed Distance Function / 有向距離函數)
    # sdf_list = []
    # for v in views:
    #     sdf = signed_distance_2d(v["mask"])
    #     sdf_list.append(sdf)
    
    sdf_list = []
    
    for mask in masks:
        sdf = signed_distance_2d(mask)
        sdf_list.append(sdf)
        
    # # 建 3D 網格點（中心取樣）
    # xs = np.linspace(mins[0] + voxel_size/2, maxs[0] - voxel_size/2, Nx)
    # ys = np.linspace(mins[1] + voxel_size/2, maxs[1] - voxel_size/2, Ny)
    # zs = np.linspace(mins[2] + voxel_size/2, maxs[2] - voxel_size/2, Nz)

    # occ = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # dims = np.maximum((side_length / voxel_size).astype(int), 1)
    # Nx, Ny, Nz = dims.tolist()
    n = max(int(np.ceil(half*2 / voxel_size)), 1)
    Nx=Ny=Nz=n
    mins = center - half
    maxs = center + half
    
    # 建 3D 網格點（中心取樣）
    xs = np.linspace(mins[0] + voxel_size/2, maxs[0] - voxel_size/2, Nx) # 讓「體素格點的位置對應到體素中心」，而不是對應到格子的邊界。
    ys = np.linspace(mins[1] + voxel_size/2, maxs[1] - voxel_size/2, Ny)
    zs = np.linspace(mins[2] + voxel_size/2, maxs[2] - voxel_size/2, Nz)
    
    # 建 體素矩陣
    occ = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    # 逐體素聚合（可視情況向量化/多執行緒）
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                Xw = np.array([x, y, z, 1], dtype=np.float32) # 齊次化座標
                ws = []
                votes = 0
                for idx, f in enumerate(frames):
                    # K, R, t = v["K"], v["R"], v["t"]
                    # u, vv, zc = image_point(None, None, K, R, t, Xw)
                    # if u is None:
                    #     continue
                    # H, W = hw_list[idx]
                    # ui = int(np.round(u))
                    # vi = int(np.round(vv))
                    
                    
                    X_p, depth = f.project_point(Xw)
                    ui,vi= f.camera.denormalize(X_p)
                    if depth <= 1e-9:
                        continue
                    W,H = f.camera.width, f.camera.height
                    if ui < 0 or vi < 0 or ui >= W or vi >= H:
                        continue
                    s = sdf_list[idx][vi, ui]
                    # 連續置信度（sigmoid），邊界處不那麼敏感
                    w = 1.0 / (1.0 + np.exp(s / max(tau_px, 1e-6)))
                    ws.append(w)
                    if w >= theta_view:
                        votes += 1
                if len(ws) == 0:
                    continue
                # 全視角分數（平均或中位數都可以；中位數抗 outlier 更強）
                w_all = float(np.median(ws))
                if votes >= min_views and w_all >= theta_global:
                    occ[ix, iy, iz] = w_all  # or set 1.0

    return occ # 3D 佔用網格 (occupancy grid)
# ---------- Marching Cubes + Open3D 簡化 ----------
def mesh_from_occupancy_and_simplify(occ, origin, voxel_size,
                                     max_vertices=100,
                                     iso=0.5):
    """
    occ: (Nx,Ny,Nz) 連續分數或二值
    iso: 等值面門檻（0.5 常見）
    回傳：V_out [N',3], F_out [T',3]
    """
    # skimage 的 marching_cubes 期望軸順序 (z,y,x)，所以轉置一下
    vol = np.transpose(occ, (2,1,0)).astype(np.float32)
    if vol.max() <= 0:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.int32)

    verts, faces, _, _ = marching_cubes(volume=vol, level=iso)  # verts in voxel coords
    # 轉回世界座標
    V = np.zeros_like(verts, dtype=np.float64)
    V[:, 0] = origin[0] + verts[:, 2] * voxel_size
    V[:, 1] = origin[1] + verts[:, 1] * voxel_size
    V[:, 2] = origin[2] + verts[:, 0] * voxel_size
    F = faces.astype(np.int32)

    # Open3D in-memory 簡化
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    # 逐步降低三角形數直到頂點 ≤ max_vertices
    target_tris = max(int(max_vertices * 2), 4)
    best = mesh
    for _ in range(8):
        m2 = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        m2.remove_degenerate_triangles()
        m2.remove_unreferenced_vertices()
        if len(m2.vertices) <= max_vertices and len(m2.vertices) >= 4:
            best = m2
            break
        if len(m2.vertices) < len(best.vertices):
            best = m2
        target_tris = max(int(target_tris * 0.7), 4)

    V_out = np.asarray(best.vertices, dtype=np.float64)
    F_out = np.asarray(best.triangles, dtype=np.int32)
    return V_out, F_out



# 從「遮罩多邊形」產生「裁切平面」
def polygon_to_halfspaces(poly, frame:'Frame', sample_offset:float=0.01):
    """
    從影像多邊形的每條邊，建立一個「穿過相機中心的切平面」：
      - 取邊的兩端點 p0, p1（標準化影像座標）
      - 反投影成兩條相機座標系射線 d0_cam, d1_cam
      - 兩射線張成的平面，其相機座標法向 n_cam = normalize(cross(d0_cam, d1_cam))
      - 使用「邊的像素法向內移 sample_offset」取得 pin → 射線 d_in，
        由 sign = dot(n_cam, d_in) 決定保留側（n·x + d <= 0）
      - 轉到世界座標：n_w = Rwc @ n_cam,  d_w = n_cam @ tcw
    注意：
      - sample_offset 會乘以 frame.camera.radius（提高尺度穩定性）
      - 平面「一定」經過相機中心，因為由兩條過原點的射線張成
    回傳: list[(n_w, d_w)]，其中保留側為 n·x + d <= 0
    """
    sample_offset = sample_offset * frame.camera.radius

    # 先把像素多邊形轉到標準化影像平面（去畸變/內參）
    ext = frame.camera.normalize(poly)  # (N,2)

    planes = []
    N = len(ext)
    for i, j in zip(range(N), np.roll(range(N), -1)):
        p0, p1 = ext[i], ext[j]
        e = p1 - p0
        elen = np.linalg.norm(e)
        if elen < 1e-9:
            continue

        # 2D 切線方向 / 左法向（CCW 外環 → 左邊是內側）
        tdir = e / elen
        n2 = np.array([-tdir[1], tdir[0]], dtype=np.float64)

        # 反投影成相機座標系射線（方向）
        d0_cam = frame.deproject_point(p0, world_coor=False)
        d1_cam = frame.deproject_point(p1, world_coor=False)
        n_cam = normalize(np.cross(d0_cam, d1_cam))  # 平面法向

        # 用「內側」的小偏移點 pin 的射線，決定平面保留側
        mid = 0.5 * (p0 + p1)
        pin = mid + sample_offset * n2
        d_in = frame.deproject_point(pin, world_coor=False)

        # sign>0 代表 d_in 與 n_cam 同側，須反轉法向以滿足 n·x + d <= 0 為保留側
        if np.dot(n_cam, d_in) > 1e-12:
            n_cam = -n_cam

        # 相機→世界：方向只需旋轉；d = n·tcw（因平面經過相機原點）
        n_w = frame.Rwc @ n_cam
        d_w = float(n_cam @ frame.tcw)

        planes.append((n_w, d_w))

    return planes


# 初始化一正方體外殼
def make_initial_box(center=np.zeros(3), half=10.0):
    """
    建立一個以 center 為中心、邊長 = 2*half 的立方體凸多面體
    回傳: (V, F)
    V: (Nv,3) 頂點
    F: list of face indices (each face: list[int], CCW 外向)
    """
    cx, cy, cz = center
    hs = half
    V = np.array([
        [cx-hs, cy-hs, cz-hs],
        [cx+hs, cy-hs, cz-hs],
        [cx+hs, cy+hs, cz-hs],
        [cx-hs, cy+hs, cz-hs],
        [cx-hs, cy-hs, cz+hs],
        [cx+hs, cy-hs, cz+hs],
        [cx+hs, cy+hs, cz+hs],
        [cx-hs, cy+hs, cz+hs],
    ], dtype=np.float64)
    # 六個面（每面CCW，外向）
    F = [
        [0,1,2,3],  # -Z
        [4,5,6,7],  # +Z
        [0,4,5,1],  # -Y
        [1,5,6,2],  # +X
        [2,6,7,3],  # +Y
        [3,7,4,0],  # -X
    ]
    return V, F
# 「裁切平面」切割「視覺外殼」
def clip_poly_with_plane(V, F, n, d, eps=1e-9):
    """
    以半空間 n·x + d <= 0 裁切「凸」多面體 (V,F)
    策略（逐面 2D 裁切 + 補蓋帽）：
      - 對多面體每個面（多邊形）做「線性裁切」：
          設 a,b 為相鄰頂點、da=n·a+d、db=n·b+d
          * in→in: 保留 b
          * in→out: 加入交點 p = a + t*(b-a), t=da/(da-db)
          * out→in: 先 p 再 b
          * out→out: 丟棄
      - 收集所有交點，投影到平面 (u,v) 建立角度排序 → 形成「蓋帽面」
      - 檢查蓋帽面法向是否與 n 同向；若否則反轉點序
      - 最後做頂點去重
    """
    V = V.copy()
    new_faces = []
    intersection_points = []

    def signed_dist(p): return np.dot(n, p) + d

    for face in F:
        poly = V[face]
        m = len(poly)
        if m == 0:
            continue
        kept = []
        for i in range(m):
            a = poly[i]
            b = poly[(i + 1) % m]
            da = signed_dist(a)
            db = signed_dist(b)

            a_in = (da <= eps)
            b_in = (db <= eps)

            if a_in and b_in:
                kept.append(b)
            elif a_in and (not b_in):
                # 線段 a→b 與平面交於比例 t = da / (da - db) (推導自 da + t*(db-da)=0)
                t = da / (da - db + 1e-12)
                p = a + t * (b - a)
                kept.append(p)
                intersection_points.append(p)
            elif (not a_in) and b_in:
                t = da / (da - db + 1e-12)
                p = a + t * (b - a)
                kept.append(p)
                kept.append(b)
                intersection_points.append(p)
            else:
                pass  # out→out

        if len(kept) >= 3:
            idxs = []
            for p in kept:
                V = np.vstack([V, p.reshape(1, 3)])
                idxs.append(len(V) - 1)
            new_faces.append(idxs)

    if len(new_faces) == 0:
        return np.ndarray((0, 3)), []

    # 補蓋帽面：將所有交點在平面上排序（以 (u,v) 的 atan2）
    if len(intersection_points) >= 3:
        P = np.array(intersection_points, dtype=np.float64)
        # 建立平面座標基底 u,v：取不平行於 n 的軸 a
        a = np.array([1, 0, 0], dtype=np.float64)
        if abs(np.dot(a, n)) > 0.9:  # 避免與 n 幾乎平行
            a = np.array([0, 1, 0], dtype=np.float64)
        u = normalize(np.cross(n, a))
        v = normalize(np.cross(n, u))

        Pu = P @ u
        Pv = P @ v

        cen_u, cen_v = Pu.mean(), Pv.mean()
        angles = np.arctan2(Pv - cen_v, Pu - cen_u)
        order = np.argsort(angles)
        P_ord = P[order]

        idxs = []
        for p in P_ord:
            V = np.vstack([V, p.reshape(1, 3)])
            idxs.append(len(V) - 1)

        # 面外向應與 n 同向；否則反序
        if len(idxs) >= 3:
            p0, p1, p2 = V[idxs[0]], V[idxs[1]], V[idxs[2]]
            face_normal = normalize(np.cross(p1 - p0, p2 - p0))
            if np.dot(face_normal, n) < 0:
                idxs = idxs[::-1]

        new_faces.append(idxs)

    V2, F2 = merge_duplicate_vertices(V, new_faces)
    return V2, F2

# ---------- 在單位球面產生一批方向（以及其三角拓樸） ----------
def sphere_directions_compat(subdiv: int = 2):
    """
    產生球面方向與其三角拓樸（Open3D 相容版）
    優先使用 create_icosphere ；若該 API 在當前 Open3D 版本不存在，
    則退回 create_sphere (UV sphere)，並將頂點單位化。
    
    參數
    ----
    subdiv : int
        細分層級。對 icosphere 代表均勻細分層級（0/1/2/3...）；
        對 UV sphere 則轉成 approximate 的解析度。

    回傳
    ----
    U : (N,3) float64
        單位球面方向向量（每個方向一個頂點）。
    F : (T,3) int32
        三角形面索引（以 U 的索引組成），可直接拿來當網格拓樸。
    """
    # 嘗試使用 icosphere（均勻性較好）
    try:
        sph = o3d.geometry.TriangleMesh.create_icosphere(radius=1.0, subdivisions=subdiv)
        U = np.asarray(sph.vertices, dtype=np.float64)
        F = np.asarray(sph.triangles, dtype=np.int32)
        # 理論上已在單位球上，但保險做一次單位化
        U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        return U, F
    except AttributeError:
        pass  # 沒有 create_icosphere，改用 UV sphere

    # 退回 UV sphere：解析度大約跟 subdiv 對齊
    # subdivisions=0/1/2/3 對應 resolution 約 10/20/40/80
    resolution = int(10 * (2 ** max(0, subdiv)))
    if resolution < 10:
        resolution = 10

    sph = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    # Open3D 的 UV sphere 會有法線/UV，可忽略；我們只取頂點與三角面
    U = np.asarray(sph.vertices, dtype=np.float64)
    F = np.asarray(sph.triangles, dtype=np.int32)
    # 單位化，確保在單位球上（UV sphere 理論上半徑=1，但仍做保險）
    norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    U = U / norms
    return U, F

# ---------- 檢查一個世界點是否被「至少 K 個視角」視為在遮罩內 ----------
def point_passes_views(Xw: np.ndarray,
                       frames: list['Frame'],
                       sdf_list: list | None,
                       tau_px: float,
                       k_required: int,
                       view_thresh: float = 0.5) -> bool:
    """
    檢查世界點 Xw 是否被「至少 k_required 個視角」視為在物體內：
      - 若有 sdf_list：用 sigmoid(-sdf/tau) 轉為軟分數 w ∈ (0,1)，w>=view_thresh 算「通過」
      - 若 sdf_list=None：直接檢查二值遮罩 True/False
    注意：
      - 透過 frame.project_point(Xw) 取得相機平面座標與深度
      - 再用 frame.camera.denormalize(...) 轉像素 (u,v)，落在畫面內才計票
    """
    votes = 0
    x,y,z = Xw[:3]
    Xw = np.array([x, y, z, 1.0], dtype=np.float64)
    for i, f in enumerate(frames):
        Xp, depth = f.project_point(Xw.astype(np.float32))
        if depth is None or depth <= 1e-9:
            continue  # 在相機背後

        u, v = f.camera.denormalize(Xp)
        W, H = f.camera.width, f.camera.height
        ui, vi = int(np.round(u)), int(np.round(v))
        if (ui < 0) or (vi < 0) or (ui >= W) or (vi >= H):
            continue  # 投影落在影像外

        if sdf_list is not None:
            s = sdf_list[i][vi, ui]
            # s<0(物體內) → sigmoid(-s/tau) 接近 1； s>0(外) → 接近 0
            w = 1.0 / (1.0 + np.exp(s / max(tau_px, 1e-6)))
            if w >= view_thresh:
                votes += 1
        else:
            # 二值遮罩（非SDF）
            # 這裡假設你有與 frames 同序的 masks: list[np.bool_]
            # 若你想直接從 Thing 取：請自行包裝在外層先建好 masks
            raise RuntimeError("二值模式請改用 SDF 或傳入 masks 對應邏輯")

        if votes >= k_required:
            return True

    return False

# ---------- 沿一個方向做最遠半徑搜尋（指數擴張 + 二分） ----------
def max_radius_along_dir(O: np.ndarray,
                         u: np.ndarray,
                         frames: list,
                         sdf_list: list | None,
                         tau_px: float,
                         k_required: int,
                         r0: float = 0.01,
                         r_max_cap: float = 10.0,
                         grow: float = 2.0,
                         it_bisect: int = 18) -> float:
    """
    沿方向 u 從中心 O 向外，找滿足「至少 k_required 視角支持」的最大距離 r*：
      1) 先測 O + r0*u 是否通過；若不通過直接回 0（此方向為洞/陰影）
      2) 指數擴張 r ← r*grow，直到第一次失敗 or r 達上限 r_max_cap
      3) 在最後成功半徑與失敗半徑之間二分搜尋 it_bisect 次，取得臨界 r*
    參數可依場景尺度調整：
      - r_max_cap: 物體可能的最大半徑上限（公尺）
      - grow: 指數擴張倍率（2~3 慣用）
      - it_bisect: 二分次數（18~22 足矣）
    """
    # 初始測試
    if not point_passes_views(O + r0 * u, frames, sdf_list, tau_px, k_required):
        return 0.0

    lo, hi = r0, r0
    # 向外指數擴張直到「首次失敗」或達上限
    while True:
        hi = min(hi * grow, r_max_cap)
        if hi <= lo + 1e-9:
            break
        ok = point_passes_views(O + hi * u, frames, sdf_list, tau_px, k_required)
        if (not ok) or (hi >= r_max_cap - 1e-9):
            break
        lo = hi
        if hi >= r_max_cap:
            break

    # 二分逼近臨界半徑
    for _ in range(it_bisect):
        mid = 0.5 * (lo + hi)
        if point_passes_views(O + mid * u, frames, sdf_list, tau_px, k_required):
            lo = mid
        else:
            hi = mid

    return lo

# ---------- 主流程：球面視覺外殼 → 網格 ----------
def build_spherical_visual_hull(frames: list,
                                masks: list[np.ndarray] | None,
                                center_O: np.ndarray,
                                ico_subdiv: int = 2,
                                k_required: int = 2,
                                use_sdf: bool = True,
                                tau_px: float = 2.0,
                                r0: float = 0.01,
                                r_max_cap: float = 10.0,
                                grow: float = 2.0,
                                it_bisect: int = 18,
                                simplify_vertices: int = 100):
    """
    以「球面方向搜尋半徑」的方式建立非凸外殼（星狀）：
      - 不需要 AABB，也不做半空間交（避免越切越小/切空）
      - 對每個球面方向 u，找 O + r*u 在多視角遮罩內的最大 r*
      - 保留球面拓樸，把球面頂點半徑換成 r*，得到外殼 V,F
      - 最後用 Open3D in-memory decimation 壓到 <= simplify_vertices

    參數：
      frames: list[Frame]（提供 project_point/denormalize/相機尺寸）
      masks:  與 frames 同長度的遮罩 list（np.bool_）。若 use_sdf=True 會先轉 SDF
              （也可傳 None，表示你會在 point_passes_views 自訂取得遮罩邏輯）
      center_O: (3,) 外殼中心（建議落在物體內或附近）
      ico_subdiv: 2 (~320 個方向) / 3 (~1280 個方向)
      k_required: 至少多少視角同意（抗噪）
      use_sdf, tau_px: SDF 軟化邊界與尺度（像素）
      r0, r_max_cap, grow, it_bisect: 半徑搜尋超參
      simplify_vertices: 最終頂點上限（例如 100）

    回傳：
      V_out: (N',3) 外殼頂點
      F_out: (T',3) 三角面索引（來源為球面的三角拓樸）
    """
    O = center_O.astype(np.float64).reshape(3)

    # 1) 準備 SDF（或二值）：這裡選擇 SDF（建議），能有效消除邊界抖動
    sdf_list = None
    if use_sdf:
        assert masks is not None and len(masks) == len(frames), "use_sdf=True 時需提供與 frames 對齊的 masks"
        sdf_list = [signed_distance_2d(m.astype(np.bool_)) for m in masks]

    # 2) 取球面方向與球面拓樸
    U, F_sphere = sphere_directions_compat(subdiv=ico_subdiv)

    # 3) 沿每個方向找最大半徑
    radii = np.zeros(len(U), dtype=np.float64)
    for i, u in enumerate(U):
        radii[i] = max_radius_along_dir(
            O, u, frames, sdf_list, tau_px, k_required,
            r0=r0, r_max_cap=r_max_cap, grow=grow, it_bisect=it_bisect
        )

    # 4) 建立外殼頂點：X = O + r*u；面索引用球面的三角拓樸
    V = O[None, :] + radii[:, None] * U
    F = F_sphere.copy()

    # 5) 進 Open3D 清理與壓簡化（保留在記憶體，不落地檔）
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(F.astype(np.int32))
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    target_tris = max(int(simplify_vertices * 2), 4)
    best = mesh
    for _ in range(6):
        m2 = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        m2.remove_degenerate_triangles()
        m2.remove_unreferenced_vertices()
        if len(m2.vertices) <= simplify_vertices and len(m2.vertices) >= 4:
            best = m2
            break
        if len(m2.vertices) < len(best.vertices):
            best = m2
        target_tris = max(int(target_tris * 0.7), 4)

    V_out = np.asarray(best.vertices, dtype=np.float64)
    F_out = np.asarray(best.triangles, dtype=np.int32)
    return V_out, F_out


def merge_duplicate_vertices(V, F, tol=1e-6):
    """
    以四捨五入座標做簡單去重。
    """
    key = np.round(V / tol) * tol
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    F2 = [[int(inv[i]) for i in face] for face in F]
    return uniq, F2

# 多邊形面三角化
def polygon_faces_to_triangles(F):
    """
    將多邊形面 F（list[list[int]]，每面 >=3 個頂點）做扇形三角化。
    假設每個面是凸的（由半空間交集得到的面通常為凸）。
    回傳: np.ndarray [T,3] 的三角面索引
    """
    tris = []
    for face in F:
        if len(face) < 3:
            continue
        # 扇形三角化： (v0, v1, v2), (v0, v2, v3), ...
        v0 = face[0]
        for i in range(1, len(face) - 1):
            tris.append([v0, face[i], face[i+1]])
    return np.asarray(tris, dtype=np.int32)
# 簡化視覺外殼
def simplify_mesh_in_memory(V, F, max_vertices=100, tri_buffer_ratio=2.0,
                            remove_degenerate=True, remove_duplicated=True,
                            remove_non_manifold=True):
    """
    參數：
      - V: np.ndarray [N,3] 浮點
      - F: list[list[int]] 多邊形面（每面 ≥3 點），可凹？（建議凸）
      - max_vertices: 目標頂點上限（例如 100）
      - tri_buffer_ratio: 初始目標三角形數 = max_vertices * tri_buffer_ratio
      - remove_*: 簡單清理選項
    回傳：
      - V_out: np.ndarray [N',3]
      - F_out: np.ndarray [T',3]（三角形索引）
    """
    # 1) 先把多邊形面三角化
    triangles = polygon_faces_to_triangles(F)
    if triangles.size == 0:
        return V.copy(), np.zeros((0, 3), dtype=np.int32)

    # 2) 建立 Open3D Mesh（全在記憶體）
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(triangles.astype(np.int32))
    )

    # 3) 清理（避免簡化時出錯）
    if remove_degenerate:
        mesh.remove_degenerate_triangles()
    if remove_duplicated:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
    if remove_non_manifold:
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    # 4) 以三角形數作為目標做 QEM 簡化
    #    Open3D 的 decimation 接受「目標三角形數」，不是頂點數。
    #    這裡做一個小回圈：一步步減少三角形數，直到頂點 ≤ max_vertices。
    target_tris = max(int(max_vertices * tri_buffer_ratio), 4)
    best = mesh
    for _ in range(6):  # 最多嘗試幾次（通常 2~3 次就夠）
        m2 = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        m2.remove_degenerate_triangles()
        m2.remove_unreferenced_vertices()
        n_vert = np.asarray(m2.vertices).shape[0]
        # 保留目前最好（頂點最少但仍 ≥ 4）
        if n_vert <= max_vertices and n_vert >= 4:
            best = m2
            break
        else:
            best = m2 if np.asarray(m2.vertices).shape[0] < np.asarray(best.vertices).shape[0] else best
            # 逐步下降目標三角形數
            target_tris = max(int(target_tris * 0.7), 4)

    best.compute_vertex_normals()

    # 5) 輸出 numpy
    V_out = np.asarray(best.vertices, dtype=np.float64)
    F_out = np.asarray(best.triangles, dtype=np.int32)  # 注意：這裡是三角面

    return V_out, F_out

# def save_as_obj(path, V, F):
#     with open(path, "w") as f:
#         for v in V:
#             f.write(f"v {v[0]} {v[1]} {v[2]}\n")
#         for face in F:
#             # OBJ 面索引從 1 開始
#             idxs = [str(i+1) for i in face]
#             f.write("f " + " ".join(idxs) + "\n")

'''

