import numpy as np

from .feature import FeatureTool
from .utils import normalize_vector
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .map import Map
    from .frame import Frame, KeyFrame
    from .slam import Slam
    from .objects import Thing, Stuff



# def predict_points(f1:'Frame',f2:'Frame',kp_idxs1,kp_idxs2):
#     # # 拒絕沒有足夠「視差」的點
#     # baseline = np.linalg.norm(f1.pose[:3, 3] - f2.pose[:3, 3])
#     # if baseline < 0.05:  # 依據場景調整
#     #     print("基線太小，暫不建立新點")
#     #     return
    
#     # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
#     #predict pose
#     pts4d = triangulate(f1.pose, f2.pose, f1.kps[kp_idxs1], f2.kps[kp_idxs2])
    
#     # This line normalizes the 3D points by dividing each row by its fourth coordinate W
#     # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates


#     # Reject points without enough "Parallax" and points behind the camera
#     # checks if the absolute value of the fourth coordinate W is greater than 0.005.
#     # checks if the z-coordinate of the points is positive.
#     # returns, A boolean array indicating which points satisfy both criteria.
#     # 檢查第四座標 W 的絕對值是否大於 0.005。 避免太接近零（除W會出現極大值）
#     # 返回，一個布林陣列，表示哪些點符合這兩個條件。
#     good_pts4d = (np.abs(pts4d[:, 3]) > 0.005)
#     pts4d /= pts4d[:, 3:]
    
#     visible1,  pts_proj1 = f1.are_visible(pts4d)
#     visible2,  pts_proj2 = f2.are_visible(pts4d)

#     # print('good_pts4d 1',np.sum(good_pts4d))
#     good_pts4d = good_pts4d & visible1 & visible2
#     # print('good_pts4d 2',np.sum(good_pts4d))
#     point_counter = 0

#     for i, (p_position, kp_idx1, kp_idx2) in enumerate(zip(pts4d,kp_idxs1,kp_idxs2)):

#         #  If the point is not good (i.e., good_pts4d[i] is False), the loop skips the current iteration and moves to the next point.
#         if not good_pts4d[i]:continue
        
#         point_counter +=1
#         if f1.points[kp_idx1] is not None: continue
        
#         # # check reprojection error
#         # err1 = pts_proj1[i][:2] -f1.kps[kp_idx1]
#         # err2 = pts_proj2[i][:2] -f2.kps[kp_idx2]
#         # err1 =np.sum(err1**2)
#         # err2 =np.sum(err2**2)
#         # # print('err',err1,err2)
#         # if (err1 >1) | (err2>1): continue # 閾值隨便設定， TODO:修改成一個有意義的值
        
#         if f2.points[kp_idx2]is None:
#             # 建立新 point
#             x, y = f1.raw_kps[kp_idx1]
#             color= f1.img[y][x][::-1] # [::-1] 會將陣列反轉順序，BGR 格式轉成 RGB
#             pt = Point(f2.slam, p_position,color, frame=f2, idx=kp_idx2)
        
#         else:
#             # 使用追蹤中的 point
#             pt = f2.points[kp_idx2]
        
#         f1.map.add_point_frame_relation(pt,f1,kp_idx1)
        
#     return point_counter



class Point(object):
    # A Point is a 3-D point in the world
    # Each point is observed in multiple frames
    _id_counter = 0
    def __init__(self, slam:'Slam', loc:np.ndarray[int,float], color=None,frame:'KeyFrame'=None,idx=None):
        
        self.slam = slam
        self.map = slam.map

        # assigns a unique ID to the point based on the current number of points in the map.
        self.id = Point._id_counter
        Point._id_counter += 1  # 每次建立物件時遞增
        
        # adds the point instance to the map’s list of points.
        self.map.add_point(self)
        # self.frames:list['Frame'] = [] # TODO: 修改成 dict{Frame:kp_idx}
        # self.keyframes:list['KeyFrame'] = [] # TODO: 修改成 dict{KeyFrame:kp_idx}  記得修改 Frame.find_point, Map.remove_point_frame_relation
        
        self.frames:dict['Frame',int] = {}
        self.keyframes:dict['KeyFrame',int] = {}
        
        self.location:np.ndarray[int,float] = loc # [𝑋, 𝑌, 𝑍, 𝑊=1] 齊次化的座標
        # self.idxs = []
        
        self.color:np.ndarray[int,np.uint8] = (None,None,None) # 0-255 RGB
        self.des = None # 點的描述子
        self.normal = np.array([0,0,0]) # 3D vector 點在平面的法線方向，可以用於判斷點是否可見(位於物體正面)
        
        
        self.is_stable = False # 標記這個點的位置是否穩定
        
        self.object:'Thing'|'Stuff' = None # 最可能屬於的物體(目前是第一個物體)
        self.objects:set[None|'Thing'|'Stuff'] = set([None]) # 所有可能屬於的物體
        # TODO: 紀錄點的可靠性，可能用在遮罩中心/邊緣 或是全景切割的像素類型準確度

        if color is not None:
            self.color = color

        
        
        if frame is not None and idx is not None :
            # self.map.add_point_frame_relation(self,frame,idx)
            frame.add_point(self,idx)
            
            
            frame.new_points[idx] = True
            self.des = frame.des[idx]
            self.normal, dist=normalize_vector(self.location_3D-frame.position)
            
            # self.add_frame(keyframe,idx)
            self.first_kid = frame.kid
            self.kf_ref = frame
        
        self.visible_num:int = 1 # 在所有關鍵幀中被認為「應該可以看到」的次數
        self.found_num:int = 1 # 地圖點實際被正確匹配上的次數
        
        self.min_distance:float = 0.0
        self.max_distance:float = np.inf
        
        self.dynamic:bool = False
        self.dynamic_time:int = None # 如果移動，紀錄移動時的f.id
        self.static_time:int = 0
        
        self.is_bad:bool = False # 標記這個點是否要被刪除，並在特定階段統一刪除，避免迭代途中刪除導致程式更複雜
        self.fused_by:Point|None = None # 標記這個點被另一個點融合避免第三點融入這個點
        
    
    
    def copy(self):
        """
        只複製跟時間無關的屬性
        只複製跟觀測者(f,kf)無關的屬性
        """
        
        
        # === 需要深度複製的屬性 ===
        loc = self.location.copy()
        color = self.color.copy()
        
        new_pt = Point(self.slam, loc, color)
        
        # === 需要深度複製的屬性 ===
        new_pt.des = self.des.copy() if self.des is not None else None  # 描述子
        new_pt.normal = self.normal.copy()  # 法線向量
        
        
        
        return new_pt
    
    
    @property
    def location_3D(self)->np.ndarray[3,]:
        """
        取得非齊次座標 [X, Y, Z]
        
        return:
            xyz: shape (3,) 的非齊次座標陣列
        """
        # 取得非齊次的座標
        return self.location[:3]
    
    @location_3D.setter
    def location_3D(self, xyz: np.ndarray):
        """
        使用非齊次座標 [X, Y, Z] 設定齊次座標 [X, Y, Z, 1]
        
        Args:
            xyz: shape (3,) 的非齊次座標陣列
        """
        if len(xyz) != 3:
            raise ValueError(f"Expected 3D coordinates, got shape {xyz.shape}")
        self.location[:3] = xyz
        self.location[3] = 1.0
    
    
    @property
    def kf_num(self)->int: # > 2 視為穩定
        return len(self.keyframes) 
        
    ####與其他地圖實體關聯####
    def add_frame(self,frame:'Frame',idx):
        self.map.add_point_frame_relation(self,frame,idx)
    def remove_frame(self,frame:'Frame'=None):
        if frame is not None:
            self.map.remove_point_frame_relation(self,frame)
        else:
            for frame in self.frames.copy():
                self.map.remove_point_frame_relation(self,frame)
    
    def add_obj(self,obj:'Thing|Stuff'):
        self.map.add_point_object_relation(self,obj)
    def remove_obj(self,obj:'Thing|Stuff'):
        self.map.remove_point_object_relation(self,obj)
    
    
    def increase_visable(self):
        self.visible_num += 1 
    
    def increase_found(self):
        self.found_num += 1
    
    def get_found_ratio(self)->float:
        # 實際成功被匹配次數/應該可以看到次數
        return self.found_num/self.visible_num
    
    
    def set_dynamic(self,fid:int):
        self.dynamic = True
        self.dynamic_time = fid
        
    def set_static(self,fid:int):
        self.dynamic = False
        self.static_time = fid
    
    def update_all(self):
        # self.update_bad()
        self.update_normal()
        self.update_descriptor()
        self.update_depth()
        
    # 暫時不使用
    def update_stable(self):
        # 至少被3個keyframe觀測
        observations = (self.kf_num > 2)

        
        # 計算所有keyframe間的最大基線距離
        max_baseline = 0
        for i in range(len(self.keyframes)):
            for j in range(i+1, len(self.keyframes)):
                baseline = np.linalg.norm(self.keyframes[i].Ow - self.keyframes[j].Ow)
                if baseline > max_baseline:
                    max_baseline = baseline
        
        # 計算平均重投影誤差
        errs = []
        for kf in self.keyframes:
            idx = kf.find_point(self)
            if idx is not None:
                uv = kf.kps[idx]
                visible, proj = kf.is_visible(self)
                errs.append(np.linalg.norm(proj[:2] - uv))
        mean_err = np.mean(errs) if errs else 0
        
        # 設定穩定條件
        self.is_stable = (max_baseline > 0.1) and (mean_err < 2) and observations
        return self.is_stable
    
    
    def update_depth(self):
        """
        更新 min_dist / max_dist。
        若 reference KF 找不到對應的 kp_idx，會退回到 level=0。
        """
        if self.is_bad:
            return
        if not self.keyframes:
            return
        
        # 取得 reference KeyFrame
        kf_ret = self.kf_ref
        
        Pos = self.location_3D
        # 計算 reference 距離
        PC = Pos - kf_ret.position
        dist = float(np.linalg.norm(PC))
        
        # 取得在 reference KF 的 keypoint index（若找不到則降為 level 0）
        kp_idx = kf_ret.find_point(self)
        if kp_idx is None:
            octave = 0
        else:
            octave = kf_ret.octaves[kp_idx]
        
        # 取得 scale_factors
        scale_factor = kf_ret.get_scale_factor(1)
        scale_factor_ref = kf_ret.get_scale_factor(octave)
        
        self.min_distance = (1.0 / scale_factor) / scale_factor_ref * dist
        nLevels = len(kf_ret.scale_factors)
        scale_max = kf_ret.get_scale_factor(nLevels- 1 - octave) # = nLevels - 1 - octave
        self.max_distance = scale_factor * scale_max * dist
        
        return self.min_distance, self.max_distance
    
    # def update_normal(self):
    #     # 蒐集所有kf的視線，並將平均值視為法線
    #     if not self.keyframes:
    #         return self.normal
    #     normals = []
        
    #     for kf in self.keyframes:
    #         # kf.Ow
    #         normal,dist =normalize_vector(self.location_3D-kf.position)
            
    #         normals.append(normal)
        
    #     self.normal,dist = normalize_vector(np.mean(normals,axis=0))
        
    #     return self.normal
    
    def update_normal(self):
        """
        更新 self.normal：對所有關聯 KeyFrame，取從相機中心到點的單位向量平均並正規化。視為法線
        若沒有任何 keyframe，維持原值並直接返回。
        """
        if not self.keyframes:
            return self.normal

        Pos = self.location_3D
        acc = np.zeros(3, dtype=float)
        cnt = 0
        for kf in self.keyframes:
            vec = Pos - kf.position
            nrm = np.linalg.norm(vec)
            if nrm > 1e-8:
                acc += vec / nrm
                cnt += 1

        if cnt > 0:
            self.normal, dist = normalize_vector(acc / cnt)

        return self.normal
    
    def update_bad(self):
        if self.is_bad:return True
        if not self.keyframes: # list keyframes is empty
            self.is_bad = True
            return self.is_bad
        
        
        # # 如果kf很久沒更新，且kf數量不夠，視為old
        # old_point = (self.kf_num <= 2) and (self.map.keyframes[-1].kid - self.first_kid >=2)
            
        # 計算平均投影誤差
        # point_visible = True
        if self.kf_num>=4 :
            errs = []
            for f in self.keyframes:
                idx = f.find_point(self)
                uv = f.kps[idx]
                # proj = f.project_point(p.location)
                visible, proj = f.is_visible(self)
                errs.append(np.linalg.norm(proj[:2]-uv))
                # point_visible = point_visible and visible
            mean_errs = np.mean(errs)
        else:
            mean_errs = 0.0
        
        visible_ratio = self.get_found_ratio()
        if visible_ratio<0.25 or (mean_errs > 0.02): # not point_visible
            self.is_bad = True
            
            # print('update_bad',old_point , visible_ratio<0.25 , mean_errs > 0.02)
        
        
        return self.is_bad
        
    def update_descriptor(self):
        # 對於每個描述子，計算其與其他描述子的距離的中位數
        # 找到中位數距離最小的描述子，將其設為點的描述子
        if len(self.keyframes)<2:
            return
        descriptors=[]
        for kf in self.keyframes:
            idx = kf.find_point(self)
            if idx is not None:
                descriptors.append(kf.des[idx])
        
        N = len(descriptors)
        median_distances = []
        for n in range(N):
            des_distances=[]
            for d in descriptors:
                distance = self.slam.feature_tool.descriptor_distance(descriptors[n],d)
                des_distances.append(distance)
            median_distances.append(np.median(des_distances))
        
        
        self.des = descriptors[np.argmin(median_distances)].copy()
        
    def update_color(self):
        """
        從所有關聯的 frame 取得該點的顏色並平均
        """
        colors = []
        for frame in self.frames:
            idx = frame.find_point(self)
            if idx is not None:
                x, y = frame.raw_kps[idx]
                color = frame.img[y, x][::-1]  # BGR to RGB
                colors.append(color)
        if colors:
            self.color = np.mean(colors, axis=0).astype(np.uint8)
        return self.color    
            
    def delete(self):
        # delete relevance of self in frames then delete self
        # for frame in self.frames:
        #     self.remove_frame(frame)
        
        # if self.id in self.map.points:
        #     self.map.del_point(self.id)
        for frame in list(self.frames.keys()):
            # idx = frame.find_point(self)
            # self.map.remove_point_frame_relation(self, frame, idx)
            self.remove_frame(frame)
        
        for obj in self.objects.copy():
            if obj is not None:
                self.remove_obj(obj)
            
        self.map.remove_point(self)
        # del self # 這行可省略



    def fuse(self,p:'Point'):
        # 把p的觀測frames加入self
        if p is self:return
        
        
        # 已經被融合檢查
        if self.fused_by is not None:
            # self.fusion_by.fuse(p)
            return
        
        while p.fused_by is not None:
            p = p.fused_by
        
        
        # 融合位置（簡單平均）
        sf,pf = len(self.keyframes),len(p.keyframes)
        self.location = ((self.location*sf)+(p.location*pf))/(sf+pf)
        
        # for frame in p.frames.copy(): # p.frames 是 list 如果不複製會導致 索引切換因此原本的下一個元素沒被檢查
        #     idx = frame.find_point(p)
        for frame,idx in p.frames.copy().items():
            
            
            assert idx is not None
            
            p.remove_frame(frame)
            if self in frame.points :continue
            self.add_frame(frame,idx)
            
        # for frame in p.frames.copy():
        #     p.remove_frame(frame)
        
        for obj in p.objects.copy():
            if obj is None: continue
            self.add_obj(obj)
            p.remove_obj(obj)
        
        
        p.fused_by = self
        p.is_bad = True
        # updata descriptor
        self.update_descriptor()
        self.update_normal()
        self.update_color()
        
        

        
        
        
        