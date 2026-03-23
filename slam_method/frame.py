import cv2
import numpy as np
import math
import open3d as o3d
from typing import Optional
import weakref
import sys


from scipy.spatial import cKDTree
from pytransform3d import transformations,rotations
# from skimage.measure import ransac
# from skimage.transform import FundamentalMatrixTransform
# from .utils import normalize
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .map import Map
    from .point import Point
    from .slam import Slam
    from .objects import Thing, Stuff
from .camera import Camera
from .utils import PointKeypointMap,is_blurry,normalize_vector,normalize
from .pose import Pose
# class KeyPoint(object):
#     def __init__(self,raw_loc,loc,des,class_id):
        

#         self.raw_loc:list[int,int] = raw_loc # (X,Y)    (0,0) <= (X,Y) < (w,h)   keypoint在輸入影像的座標，單位像素
#         self.loc:list[float,float] = loc# (X,Y)  locations of points in this frame  keypoint在輸入影像的座標，消除相機扭曲並正規化(3D點投影到相機前1m平面上的座標，座標原點為焦點在平面的投影)，單位公尺
#         self.des:any = des        # keypoint 的描述子
#         self.class_id:list[int] = class_id # keypoint 屬於什麼物件類型，可能有多個也可能為空
#         self.point:Point


# 根據特徵提取器修改
scale_factor=1.2
nlevels=8

class Frame(Pose):
    
    scale_factors = [scale_factor ** i for i in range(nlevels)]
    sigma2_level = [(s * s) * 1 for s in scale_factors]
    inv_sigma2_level = [1.0 / v for v in sigma2_level]
    
    def get_sigma2(self,octave:int) -> float:
        """
        返回指定 octave 的 sigma^2
        """
        return Frame.sigma2_level[octave]
    
    def get_inv_sigma2(self,octave:int) -> float:
        """
        返回指定 octave 的 inv sigma^2
        """
        return Frame.inv_sigma2_level[octave]
    
    def get_scale_factor(self,octave:int) -> float:
        """
        返回指定 octave 的 scalefactor
        """
        return Frame.scale_factors[octave]
    
    _id_counter = 0
    
    def __init__(self, slam:'Slam', img, pose=np.eye(4)):
        
        
        
        self.slam=slam
        self.camera:Camera = slam.camera
        self.map:Map = slam.map

        self.img = img.copy() # 避免圖片來源遭到修改
        
        # pose
        Pose.__init__(self, pose) # Twc
        
        self.is_keyframe = False
        self.kf_ref:KeyFrame|None = None
        
        be_blurry, self.lap_var = self.camera.dynamic_determine_blurry(img)
        self.is_clear = not be_blurry   # 影像是否清楚
        
        
        self.id = Frame._id_counter
        Frame._id_counter += 1  # 每次建立物件時遞增


        # N的數量一樣且可以共用index
        self.raw_kps:np.ndarray[tuple[int,int],np.dtype[int]] # N*2(X,Y)    (0,0) <= (X,Y) < (w,h)   keypoint在輸入影像的座標
        self.kps:np.ndarray[tuple[int,int],np.dtype[float]]     # N*2(X,Y)  locations of points in this frame  keypoint在輸入影像的座標，消除相機扭曲並正規化(3D點投影到相機前1m平面上的座標，座標原點為焦點在平面的投影) TODO 檢查是否要去畸變
        self.un_kps:np.ndarray[tuple[int,int],np.dtype[float]] # N*2(X,Y)
        
        self.des:any             # N*?    keypoint 的描述子
        self.kps_class_id:list[list[int]]=[] # N*?  keypoint 屬於什麼物件類型，可能有多個也可能為空。 不再使用
        self.points:np.ndarray[int,'Point'|None]
        # self.outliers

        self.tracking_num = 0 # 準備刪除

        # self.octaves
        # self.sizes
        # self.angles
        
        
        # 影像物件判斷
        self.slam.object_segmenter.frame_segment(self.img)

        
        

        # kps,des相關
        detected_kps = self.slam.feature_tool.detect(self.img)
        computed_kps, self.des = self.slam.feature_tool.compute(self.img,detected_kps)
        computed_kps:tuple[cv2.KeyPoint]
        
        if computed_kps is not None:
            
            kps_data = np.array([ [computed_kp.pt[0], computed_kp.pt[1], computed_kp.octave, computed_kp.size, computed_kp.angle, computed_kp.response] for computed_kp in computed_kps ], dtype=np.float32)
            raw_kps     = kps_data[:,:2] if kps_data is not None else None
            
            self.octaves = np.uint32(kps_data[:,2]) #print('octaves: ', self.octaves)
            self.sizes   = kps_data[:,3] # 特徵點鄰域直徑（像素）。代表這個點的「影響範圍」或特徵尺度大小。通常與影像金字塔層級有關。
            self.angles  = kps_data[:,4] # 特徵點的主方向（ 單位：度，範圍 [0, 360) ）。若偵測器沒計算方向，會是 -1。
            # self.response = kps_data[:,5] # 特徵強度（cornerness / saliency）。數字越大表示該點越「顯著」。
            
            self.raw_kps = raw_kps.astype(int) # 原始影像關鍵點座標
            self.un_kps = self.camera.undistorte(raw_kps) # 消除畸變效應扭曲
            self.kps = self.camera.normalize(self.un_kps) # 消除扭曲並正規化的坐標(投影平面座標)
            
        
        
        
        self.points = np.array([None]*len(self.kps))
        self.inliers  = np.full(len(self.kps), False, dtype=np.bool) # 用於儲存pose opt 判斷合理的點 ！！沒有使用
        self.outliers  = np.full(len(self.kps), False, dtype=np.bool) # 用於儲存pose opt 判斷不合理的點
        
        
        
        # obj 相關
        
        # TODO:之後改為統一由 frame_segment 的 retrun輸出
        self.img_info = self.slam.object_segmenter.cur_info # 影像整體結果
        self.obj_infos:list[dict[str,]] = self.slam.object_segmenter.cur_seg_infos # 各個物件的資訊
        # obj_info include{
        #     'id': int, # 這幀影像的分割編號
        #     'category_id': int,
        #     'area': int,
        #     'isthing': bool,
        #     'instance_id': int, # only thing have, can be used to get instance mask
        #     'score': float, # only thing have
        # },
        
        # frame上的分割區域對應的地圖物件，self.objects[0]固定是None，屬於沒有被分類的影像區域。
        self.objects:list[None|'Thing'|'Stuff']=[None]*(len(self.obj_infos))
        self.objects_dynamic:list[bool|None]=[None]*(len(self.objects)) # 分割區域對應的物體是否移動，None 代表不確定
        self.objects_dynamic_score:list[None|float] = [0.0]*(len(self.objects))
        
        # 將各kp對應到object上
        self.obj_to_pts:list[set[int]] = [set() for n in range(len(self.obj_infos))]
        mask_id = self.img_info['mask_id']
        for idx_pt,raw_kp in enumerate(self.raw_kps):
            x, y =raw_kp
            idx_obj:int = mask_id[y][x]
            self.obj_to_pts[idx_obj].add(idx_pt)
        
        
        
        
        # 可視化用
        self.track_method = np.full(len(self.kps), 0, dtype=np.int8)
        
        
        
        
        
        self.is_bad:bool = False
        self.map.add_frame(self)
    
    # @property
    # def tracking_num(self):
    #     return np.sum(self.tracking_point)
    @property
    def points_num(self)->int:
        return np.sum(self.points != None)
    
    def add_point(self,point:'Point',idx):
        self.map.add_point_frame_relation(point,self,idx)
    def remove_point(self,point:'Point'):
        self.map.remove_point_frame_relation(point,self,None)
    def add_obj(self,obj:'Thing|Stuff',idx):
        self.map.add_obj_frame_relation(obj,self,idx)
    def remove_obj(self,obj:'Thing|Stuff'=None,idx:int=None):
        """當obj, idx都有值時，會驗證值是否對應，不對應時忽視操作
            
        Args:
            obj (Thing|Stuff, optional): 未指定時刪除idx位置的obj. Defaults to None.
            idx (int, optional): 未指定時刪除所有obj的idx. Defaults to None.
        """
        
        if obj is None and idx is not None:
            obj = self.objects[idx]
            self.map.remove_obj_frame_relation(obj,self,idx)
            return 
        
        if idx is None and obj is not None:
            idxs = obj.frames[self]
            for idx in idxs:
                self.map.remove_obj_frame_relation(obj,self,idx)
            return
        if idx is not None and obj is not None:
            if self.objects[idx] is obj:
                self.map.remove_obj_frame_relation(obj,self,idx)
            else:print(f"remove_obj 的 idx: {idx} 與 obj: {obj} 不對應。")
            return
    
    def clear_outliers(self)->int:
        
        num = 0
        for i ,out in enumerate(self.outliers):
            if out:
                self.remove_point(self.points[i])
                # self.points[i] = None
                self.outliers[i] = False
                num += 1
        if num>0:
            # print('clear_outliers num',num)
            ...
            
        return num # 移除多少 outlier
            
    
    def clear_points(self):
        for point in list(self.points):
            # idx = self.find_point(point)
            self.map.remove_point_frame_relation(point,self)
        
        assert np.all(self.points == None)

    def get_obj_idx(self,kp_idx:int)->int:
        x,y = self.raw_kps[kp_idx]
        obj_idx = self.img_info['mask_id'][y,x]
        return obj_idx

    def pt_is_dynamic(self,kp_idx:int):
        obj_idx = self.get_obj_idx(kp_idx)
        
        return self.obj_is_dynamic(obj_idx)
        
    def obj_is_dynamic(self,obj_idx:int):
        dynamic = self.objects_dynamic[obj_idx]
        
        # obj = self.objects[obj_idx]
        # if dynamic is None and obj is not None:
        #     dynamic = obj.dynamic
        
        return dynamic
        
        
    def get_obj_info(self,obj_idx:int)->int:
        
        {'id': 1,
        'isthing': True,
        'instance_id': 0,
        'area': 80407,
        'score': 0.9987379908561707,
        'category_id': 0,
        },
    
    def find_point(self, point:'Point')->int:
        
        if point is None: return None
        
        # idx = np.where(self.points == point)
        # # np.where 反回 (array([idx0, idx1, idx2, idx3, idx4, ...]),)，idx 是一個int
        # # 理論上只會有0或1個idx
        
        # if len(idx[0]) > 0:  # 確保有匹配的索引
        #     assert len(idx[0]) == 1
        #     return idx[0][0]  # 返回第一個匹配的索引
        # else:
        #     return None  # 如果沒有匹配，返回 None
        
        idx = point.frames.get(self,None)
        
        # assert idx is not None
        # if (idx is None): # debug
        #     print('find_point(kp_idx is None):')
            
        return idx
        
    def set_track_method(self,idx:int,method:int):
        """
        用於display

        Args:
            idx (int): _description_
            method (int): 
                0:None
                1:new
                2:project
                3:flow
                4:fuse
                5:project in fuse
                
        """
        
        self.track_method[idx]=method
    
    def transform_point(self, point4d):
        """
        Args:
            point (np.ndarray): shape (4,) 的齊次座標3D點 (X,Y,Z,1)，世界座標
        Returns:
            point (np.ndarray): shape (4,) 的齊次座標3D點 (X,Y,Z,1)，相機座標
        """
        # point_t = np.linalg.inv(self.pose) @ point4d
        point_t = self.Tcw @ point4d
        return point_t
    
    def project_point(self, point4d):
        """
        Args:
            point (np.ndarray): shape (4,) 的齊次座標3D點 (X,Y,Z,1)，世界座標
        Returns:
            point_p (np.ndarray): shape (3,) 的齊次座標2D點 (X,Y,1)，投影到相機平面
            深度為負時座標為(X,Y,-1)
            
            depth (float): 深度距離
            
        """
        
        point_t = self.transform_point(point4d)
        point_p = point_t[:3] / np.abs(point_t[2])
        depth = point_t[2]
        return point_p ,depth
    
    def project_point_to_img(self, point4d):
        """
        Args:
            point (np.ndarray): shape (4,) 的齊次座標3D點 (X,Y,Z,1)，世界座標
        Returns:
            point_p (np.ndarray): shape (2,) 的2D點 (U,V)，投影到像素
            
            succse (bool): 
            
        """
        point_p ,depth = self.project_point(point4d)
        camera = self.camera
        (u,v) = camera.denormalize_pt(point_p)
        

        
        succse = (depth > 0) and (0 <= u < camera.width) and (0 <= v < camera.height)
        
        return (u,v) ,succse
    
    def deproject_point(self, pt_2D,world_coor=True):
        """
        將標準化影像點座標 (x, y) 反投影成世界座標系下的射線。
        
        參數:
            pt_2D: (x, y) | (x, y, 1)，(齊次)標準化影像座標 (對應相機座標系 z=1 平面)
            world_coor: (bool) 是否使用世界座標系(world coordinate system)，否則使用相機座標系
        輸出:
            ray = 相機座標系下的射線方向（單位向量）(3,)
        """
        x,y = pt_2D[:2]
        ray_c = np.array([x, y, 1.0], dtype=np.float64)
        
        if world_coor:
            ray = self.Rwc @ ray_c
        else:
            ray = ray_c
        ray, _ = normalize_vector(ray) # 單位向量
        return ray
    
    def is_visible(self, point: 'Point',normal_check:float=None):
        """
        檢查單一 3D 點是否在相機的可見範圍內。檢查視線 v 與點平面法線方向 n 之間的角度
        Args:
            point (np.ndarray): shape (4,) 的齊次座標 3D 點 (X, Y, Z, 1)，世界座標。
            normal_check (float): 0~1 frame 到 point的方向 與 point法線 間的內積要大於多少
        Returns:
            tuple:
                - visible (bool): 該點是否可見。
                - point_project (np.ndarray): shape (3,) 的 2D 投影點 (X, Y, 1)，位於相機投影平面。
        """
        point_project,depth = self.project_point(point.location)
        # 角度合理
        in_view_angle = self.camera.in_view_angle(point_project[np.newaxis, :])[0]
        # 距離合理
        in_positive_distance = point_project[2] > 0# 距離為正
        
        visible = in_view_angle and in_positive_distance# and in_view_ray_angle  #TODO:point.normal更新還未加入，所以不使用
        
        # 檢查視線 v 與點平面法線方向 n 之間的角度
        if normal_check is not None:
            viewing_verctor,dist = normalize_vector(point.location_3D-self.position)
            in_view_ray_angle = np.dot(viewing_verctor,point.normal) > normal_check # 角度小於 cos(60 deg) normal_check=0.5
            visible = visible and in_view_ray_angle
        
        
        return visible, point_project
    
    def transform_points(self, points4d):
        '''
        Args:
            points (np.ndarray): 形狀為 (N, 4) 的齊次座標3D點 n*(X,Y,Z,1)，使用世界座標
            
        Returns:
            points (np.ndarray): 形狀為 (N, 4) 的齊次座標3D點 n*(X,Y,Z,1)，使用相機座標
        '''
        points_t = (np.linalg.inv(self.pose) @ points4d.T).T
        return points_t
    
    def project_points(self, points4d)->tuple[np.ndarray,np.ndarray]:
        """
        Args:
            points (np.ndarray): 形狀為 (N, 4) 的齊次座標3D點 n*(X,Y,Z,1)，使用世界座標

        Returns:
            points(np.ndarray): 形狀為 (N, 3) 的齊次座標2D點 n*(X,Y,1)，位於相機投影平面。
            深度為負時座標為(X,Y,-1)
            
            depths(np.ndarray): 形狀為 (N,)
        """
        points_t = self.transform_points(points4d)
        points_p = points_t[:,:3] / np.abs(points_t[:,2:3])
        # depths = points_t[:,2:3]
        depths = points_t[:,2]
        return points_p, depths
    
    def are_visible(self, points4d:np.ndarray):
        """
        檢查給定的 3D 點是否在相機的可見範圍內。不會檢查視線 v 與點平面法線方向 n 之間的角度
        Args:
            points (np.ndarray): 形狀為 (N, 4) 的齊次座標 3D 點陣列，格式為 n*(X, Y, Z, 1)，使用世界座標。
        Returns:
            tuple:
                - out_flags (np.ndarray[bool]): 布林陣列，表示每個點是否可見。
                - points_project (np.ndarray): 形狀為 (N, 3) 的 2D 投影點，格式為 n*(X, Y, 1)，位於相機投影平面。
        """
        
        # 角度合理
        points_project, depths = self.project_points(points4d)
        in_view_angle = self.camera.in_view_angle(points_project)
        
        # 距離合理
        in_positive_distance = points_project[:,2]>0# 距離為正
        
        visible = in_positive_distance & in_view_angle
        
        
        return visible,  points_project
    
    
    def project_mesh_to_img(self, vertices:np.ndarray, faces:np.ndarray):
        """
        使用 TriangleMesh 投影到影像平面產生遮罩與粗略深度圖
        Args:
            vertices(np.ndarray): 3D網格的頂點
            faces(np.ndarray): 3D網格的面
        Returns:
            mask (np.ndarray): 2D遮罩 (uint8)，位於相機影像平面
            depth_map (np.ndarray): 2D深度圖 (float32) 
        """
        
        height, width = self.camera.height, self.camera.width
        mask = np.zeros((height, width), dtype=np.uint8) # cv2.fillPoly 的第一個參數(img)必須是 np.uint8 或 np.float32
        depth_map = np.zeros((height, width), dtype=np.float32)
        depth_map.fill(np.inf)
        
        # vertices = np.asarray(mesh.vertices)
        # triangles = np.asarray(mesh.triangles)
        
        vertices_h = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        
        # cam_coords_h = (extrinsics @ vertices_h.T).T
        # cam_coords = cam_coords_h[:, :3]
        # fx, fy = intrinsics.get_focal_length()
        # cx, cy = intrinsics.get_principal_point()
        # cam_coords[:, 2][cam_coords[:, 2] == 0] = 1e-6
        # u = (cam_coords[:, 0] * fx / cam_coords[:, 2] + cx).astype(int)
        # v = (cam_coords[:, 1] * fy / cam_coords[:, 2] + cy).astype(int)
        # projected_points = np.vstack((u, v)).T
        
        points_p, depths = self.project_points(vertices_h)
        # 使用相機內參扭曲投影
        points_img = self.camera.denormalize(points_p)

        
        for face in faces:
            points = points_img[face].astype(np.int32)
            avg_depth = np.mean(depths[face])

            # 只處理深度為正且有限的面
            if depths[face].min() > 0: # and np.isfinite(avg_depth):
                # 建立臨時遮罩（型態需為uint8）
                temp_mask = np.zeros_like(depth_map, dtype=np.uint8)
                cv2.fillPoly(temp_mask, [points], 1) # cv2.fillPoly 的第一個參數(img)必須是 np.uint8 或 np.float32

                # 找出 temp_mask 區域內，比 avg_depth 大（更遠）的像素
                update_mask = (temp_mask.astype(bool)) & (depth_map > avg_depth)

                # 更新遮罩與深度圖
                mask[update_mask] = 255
                depth_map[update_mask] = avg_depth
        
        mask = mask > 0
        
        return mask, depth_map
        
    def compute_points_median_depth(self):
        points4d = np.array([p.location for p in self.points if p is not None])
        
        if len(points4d)>0:
            z = self.transform_points(points4d)[:,2]
            median_depth = np.median(z)
            return median_depth
        else:
            return -1
    
    
    def point_to_camera_plane_distance(self,point3D)->float:
        """
        計算點到相機平面的距離。
        
        Args:
            point (np.ndarray): 點的三維座標 (3,)。
            Twc (np.ndarray): 相機的世界位姿 (4, 4)，包含旋轉和平移。
        
        Returns:
            float: 點到相機平面的距離。
        """
        Twc = self.Twc
        # 提取相機位置（世界座標系）
        camera_position = Twc[:3, 3]
        
        # 提取相機的 z 軸方向（平面法向量）
        camera_z_axis = Twc[:3, 2]
        
        # 計算點到平面的距離
        numerator = np.abs(np.dot(camera_z_axis, point3D - camera_position))
        denominator = np.linalg.norm(camera_z_axis)
        distance = numerator / denominator
        
        return distance
    
    def delete(self):
        # delete relevance of self in points then delete self
        for point in list(self.points):
            idx = self.find_point(point)
            self.map.remove_point_frame_relation(point,self,idx)
            
            
        for obj in self.objects:
            if obj is None: continue
            obj.remove_frame(self)
            
        self.map.remove_frame(self)
        if self.is_keyframe:
            print('del kf',self.kid)
            self.map.remove_keyframe(self)
            
            # print(f'delete kf:{self.kid}')
            # print('frame 被參考次數',sys.getrefcount(self),'如果只剩下1，代表沒有其他參考。')
            # if sys.getrefcount(self)>1:
            #     # ref = weakref.ref(self)
            #     # print('ref',ref)
            #     import gc
            #     # print(gc.get_referrers(self)) # 這會列出所有目前持有 self 的物件（如 list、dict、其他 class 實例等）。
            #     refs = gc.get_referrers(self)
            #     external_refs = []
            #     for ref in refs:
            #         # 排除所有屬性字典（__dict__），不管是自己的還是其他物件的
            #         if isinstance(ref, dict) and hasattr(ref, '__module__'):
            #             continue
            #         # 排除自己的屬性字典
            #         if ref is self.__dict__:
            #             continue
            #         # 排除本地變數
            #         if isinstance(ref, dict) and ('self' in ref or '__module__' in ref):
            #             continue
            #         external_refs.append(ref)

            #     print("外部引用:", external_refs,":外部引用") # 這樣只會列出真正持有 frame 的外部結構
        
        
        # del self
        
  # KD tree of normalized keypoints
    @property
    def kd(self):
        if not hasattr(self, '_kd'):
            self._kd = cKDTree(self.kps)
        return self._kd
    
    @property
    def raw_kd(self):
        if not hasattr(self, '_raw_kd'):
            self._raw_kd = cKDTree(self.raw_kps)
        return self._raw_kd



class KeyFrame(Frame):
    _id_counter = 0

    def __init__(self, frame:Frame):
        
        
        # super().__init__(frame.slam, frame.img)
        self.slam = frame.slam
        self.camera:Camera = frame.camera
        self.map:Map = frame.map
        
        self.img = frame.img
        
        # pose # Twc
        Pose.__init__(self,frame.pose)
        
        self.is_keyframe = True
        self.kf_ref:KeyFrame|None = frame.kf_ref
        
        self.id =frame.id
        self.kid = KeyFrame._id_counter
        KeyFrame._id_counter += 1  # 每次建立物件時遞增
        
        
        self.raw_kps:np.ndarray[tuple[int,int],np.dtype[int]] = frame.raw_kps # N*2(X,Y)    (0,0) <= (X,Y) < (w,h)   keypoint在輸入影像的座標
        self.kps:np.ndarray[tuple[int,int],np.dtype[float]] = frame.kps     # N*2(X,Y)  locations of points in this frame  keypoint在輸入影像的座標，消除相機扭曲並正規化(3D點投影到相機前1m平面上的座標，座標原點為焦點在平面的投影)
        self.un_kps:np.ndarray[tuple[int,int],np.dtype[int]] = frame.un_kps   # N*2(X,Y)  消除相機扭曲
        
        self.des:list[any] = frame.des              # N*?    keypoint 的描述子
        self.kps_class_id:list[list[int]] = frame.kps_class_id # N*?  keypoint 屬於什麼物件類型，可能有多個也可能為空，不再使用
        
        self.points:np.ndarray[int,'Point'|None] = np.array([None]*len(self.kps), dtype=object) # point 由下面加入
        self.inliers = frame.inliers
        self.outliers = frame.outliers
        self.new_points  = np.full(len(self.kps), False, dtype=bool) # 用於儲存新建立的地圖點
        
        
        self.octaves = frame.octaves
        self.sizes = frame.sizes
        self.angles = frame.angles
        
        
        # 取得frame的關聯並刪除frame的關聯\\
        # 地圖加入f1並刪除frame
        # self.points=frame.points
        for idx,p in enumerate(frame.points.copy()):
            if p is not None:
                p:'Point'
                self.map.remove_point_frame_relation(p,frame,idx)
                self.map.add_point_frame_relation(p,self,idx)
                assert self in p.keyframes
                assert self in p.frames
        
        
        # object 相關
        self.obj_infos = frame.obj_infos
        self.img_info = frame.img_info
        self.obj_to_pts = frame.obj_to_pts
        # self.objects = frame.objects
        self.objects:list[None|'Thing'|'Stuff']=[None]*(len(self.obj_infos))
        self.objects_dynamic = frame.objects_dynamic
        self.objects_dynamic_score = frame.objects_dynamic_score
        for idx, obj in enumerate(frame.objects.copy()):
            if obj is not None:
                self.add_obj(obj,idx)
                frame.remove_obj(obj,idx)
                
                
                
        
        
        
        self.lap_var = frame.lap_var
        self.is_clear = frame.is_clear
        
        
        
        # 可視化用
        # self.obj_prj_id_map = frame.obj_prj_id_map
        # self.obj_prj_depth_map = frame.obj_prj_depth_map
        self.track_method = frame.track_method
        
        
        # keyframe 獨有
        self.is_old:bool = False
        self.mnFuseTargetForKF = None
        
        self.is_bad = frame.is_bad
        frame.delete()
        
        self.map.add_frame(self)
        self.map.add_keyframe(self)
        
    @property
    def new_pt_num(self):
        return np.sum(self.new_points == True)
    
    def clear_new_points_mark(self):
        new_pt_num = self.new_pt_num
        self.new_points.fill(False)
        return new_pt_num
    
    def update_bad(self):
        if  self.kid == 0: 
            self.is_bad = False
            return False
        if self.is_bad:return True
        
        
        
        
        return self.is_bad
        
        
        
    def update_old(self):
        if not self.is_old:
            
            if self.kid + 5 < KeyFrame._id_counter -1: # self後建立超過5個kf，self.is_old = True
                self.is_old = True
            
        return self.is_old