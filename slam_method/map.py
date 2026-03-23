import numpy as np
import math
from .feature import FeatureTool
from ordered_set import OrderedSet
from collections import Counter, deque
from itertools import islice

from .utils import normalize_vector
from .frame import Frame, KeyFrame
from .point import Point
from .objects import Thing, Stuff

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .slam import Slam

class Map(object):
    def __init__(self, groundtruth = None, voxel_size=2.5):
        self.points:set[Point] = set() # 3D points of map
        self.frames_maxlen= None # 60
        self.frames:deque[Frame] = deque(maxlen=self.frames_maxlen) # deque 可以限制長度(還不能用)，而且多線程穩定  暫時與list沒有差異
        self.keyframes:OrderedSet[KeyFrame] = OrderedSet() # frame的順序很重要，所以用OrderedSet
        
        self.things:set[Thing] = set()
        self.stuffs:set[Stuff] = set()
        
        self.dynamic_objcts:dict[Thing|Stuff,bool|None] = {}
        
        self.slam:'Slam' = None
        
        
        ###groundtruth###
        if groundtruth is None:
            self.pose0 = np.eye(4)
        else:
            if True:
            # 將起始點定修改為單位矩陣  
            # Rtc_0 * Rt1_w = Rtc_w     # groundtruth[0]為地圖真值起始點、地圖pose為單位矩陣，將 groundtruth[0] 改為地圖中心，所以Rt會是groundtruth到groundtruth[0]的變化
            # Rtc_0 = Rtc_w * Rt0_w^-1
            # Rtw_c = Rtw_0 * Rt0_c
                first_pose_inv = np.linalg.inv(groundtruth[0])
                groundtruth = first_pose_inv @ groundtruth # Rtw_c = Rt0_w^-1 Rt0_c
                self.pose0 = np.eye(4)
            else:
            # 或是將起始位置修改為pose的起始位置
            # 須注意座標系
                self.pose0 = groundtruth[0]
        
        self.groundtruth = groundtruth
        
        
        
        
        # covisibility_graph
        self.covisibility_graph = np.zeros((0, 0), dtype=bool)
        self.kf_idx_map:dict[KeyFrame,int] = {}
        
        
        ### object config
        self.voxel_size:float = voxel_size
        
        
        self.kf_cur = None # TODO del
        
        
    def scale_groundtruth(self):
        """
        將 groundtruth 的尺度與預測地圖對齊。
        不要同使是用groundtruth調整pose預測的位移距離
        """
        if self.groundtruth is None or len(self.frames) == 0:
            print("Groundtruth 或 frames 不存在，無法對齊尺度。")
            return

        # 提取 groundtruth 和預測地圖的相機位置
        gt_positions = np.array([pose[:3, 3] for pose in self.groundtruth])  # Groundtruth 相機位置
        pred_positions = np.array([frame.pose[:3, 3] for frame in self.frames])    # 預測地圖相機位置

        # 確保兩者的數量一致
        min_len = min(len(gt_positions), len(pred_positions))

        gt_positions = gt_positions[:min_len]
        pred_positions = pred_positions[:min_len]
        print('positions',gt_positions.shape,pred_positions.shape)
        # 計算 groundtruth 和預測地圖到原點的距離
        gt_distances = np.linalg.norm(gt_positions, axis=1)
        pred_distances = np.linalg.norm(pred_positions, axis=1)

        # 計算尺度比例因子
        # scale_factor = np.mean(pred_distances / gt_distances)
        scale_factor = np.sum(pred_distances )/ np.sum(gt_distances)

        # scale_factor = pred_distances[-1] / gt_distances[-1]
        print(f"計算出的尺度比例因子為: {scale_factor}")

        # 將比例因子應用到 groundtruth 的所有相機位置
        self.groundtruth[:,:3, 3] *= scale_factor  # 縮放相機位置        
            
    
    def save_map(self):
        '''
        TODO
        '''
        ...
    def load_map(self):
        '''
        TODO
        '''
        ...
        
        
    ####在map添加/刪除元素####
    def add_point(self,point):
        self.points.add(point)
    def remove_point(self,point):
        self.points.remove(point)
    
    def add_frame(self,frame:Frame):
        if len(self.frames) == self.frames_maxlen:
            self.frames[0].delete()
        self.slam.state_display.set_info('fid',int(frame.id))
        self.frames.append(frame)
    def remove_frame(self,frame):
        # try:
        #     self.frames.remove(frame)
        # except:pass
        self.frames.remove(frame)
    def add_keyframe(self,keyframe:KeyFrame):
        self.slam.state_display.set_info('kfid',keyframe.kid)
        self.keyframes.add(keyframe)
    def remove_keyframe(self,keyframe):
        self.keyframes.remove(keyframe)
        
    def add_thing(self,thing):
        self.things.add(thing)
    def remove_thing(self,thing):
        self.things.remove(thing)
    def add_stuff(self,stuff):
        self.stuffs.add(stuff)
    def remove_stuff(self,stuff):
        self.stuffs.remove(stuff)
    

    ####管理map中元素間關聯####
    def add_point_frame_relation(self,point:Point,frame:Frame,idx)->bool:
        # assert point not in frame.points and frame not in point.frames and frame not in point.keyframes
        # return 成功添加關聯
        
        if frame in point.frames:return False #避免重複添加
        if point in frame.points:return False
        if not (0 <= idx < frame.points.shape[0]): return False # 檢查 idx 合法性
        
        if frame.points[idx] is None:
            frame.points[idx] = point
            # point.frames.append(frame)
            point.frames[frame] = idx
            if frame.is_keyframe:
                # point.keyframes.append(frame)
                point.keyframes[frame] = idx
            
        # elif fusion:
        #     point.fusion(frame.points[idx])
        else:
            return False
           
        return True
    
    def remove_point_frame_relation(self,point:Point|None,frame:Frame|KeyFrame,idx:int=None):
        # print(point,idx)
        if (point is None) and (idx is None):return
        
        if point is None:
            point = frame.points[idx]
            if not isinstance(point, Point):return
        
        if idx is None:
            idx = frame.find_point(point)
            if idx is None:return
        
        
        point.frames.pop(frame)
        frame.points[idx] = None
        frame.outliers[idx] = False
        
        if frame.is_keyframe:
            frame:KeyFrame
            # point.keyframes = [f for f in point.keyframes if f != frame]
            point.keyframes.pop(frame)
            frame.new_points[idx] = False
            
            if frame is point.kf_ref:
                if point.kf_num > 0:
                    point.kf_ret = next(iter(point.keyframes.keys()))
                else:
                    point.kf_ret = None
            
            if point.kf_num<=2:
                point.is_bad = True
            
            
        
        
        
        # if idx is not None and 0 <= idx < frame.points.shape[0]:
        #     if frame.points[idx] is point:
                
                
        # else:
        #     # 速度比較慢
        #     # frame.points = np.array([None if p is point else p for p in frame.points], dtype=object)
        #     for i, p in enumerate(frame.points):
        #         if p is point:
        #             frame.points[i] = None
                    
            # # 速度比較快，有 bug 會少刪除 point ， numpy 沒有直接支援 is
            # frame.points = np.where(frame.points == point,None,frame.points)
            

    def add_point_object_relation(self,point:Point,obj:Thing|Stuff):
        
        obj.points.add(point)
        point.objects.add(obj)
        point.object = obj
        
    def remove_point_object_relation(self,point:Point,obj:Thing|Stuff):
        
        obj.points.discard(point)
        if point.object == obj:
            point.object = None
        point.objects.discard(obj)
        
    def add_points_object_relation(self, points: list[Point|None], obj: Thing | Stuff):
        """
        批次將多個 Point 與指定物件建立關聯。
        參數：
            points: list[Point]，要建立關聯的點列表
            obj: Thing | Stuff，要關聯的物件
        """
        # 使用集合運算一次性加入所有點
        obj.points.update(points)
        obj.points.discard(None)
        
        for point in points:
            if point is None:continue
            point.objects.add(obj)
            point.object = obj

    def remove_points_object_relation(self, points: list[Point|None], obj: Thing | Stuff):
        """
        批次移除多個 Point 與指定物件的關聯。
        參數：
            points: list[Point]，要移除關聯的點列表
            obj: Thing | Stuff，要移除關聯的物件
        """
        # 使用集合運算一次性移除所有點
        obj.points.difference_update(points)
        for point in points:
            if point is None:continue
            if point.object == obj:
                point.object = None
            point.objects.discard(obj)
    
    def add_obj_frame_relation(self,obj:Thing|Stuff,frame:Frame,idx_obj:int,temp:bool=False):
        """_summary_

        Args:
            obj (Thing | Stuff): _description_
            frame (Frame): _description_
            idx_obj (int): _description_
            temp (bool, optional): obj 是否是臨時，如果是臨時obj不會在frame加入obj的資訊. Defaults to False.
        """
        
        # # 添加點
        # for idx_pt in frame.obj_to_pts[idx_obj]:
        #     point = frame.points[idx_pt]
        #     if point is None:continue
        #     self.add_point_object_relation(point,obj)
        
        
        if frame.is_keyframe:
            obj.keyframes[frame].add(idx_obj)
        
        
        # 添加物件資訊
        info = frame.obj_infos[idx_obj]
        if obj.is_thing:
            obj.category_ids.append(info['category_id'])
            obj.category_ids_scores.append(info['score'])
        else:
            obj.category_ids.append(info['category_id'])
        
        obj.frames[frame].add(idx_obj)
        # assert frame.objects[idx] is None
        if not temp:
            frame.objects[idx_obj] = obj
        
    def remove_obj_frame_relation(self,obj:Thing|Stuff,frame:Frame,idx:int):
        
        frame.objects[idx] = None
        obj.frames[frame].discard(idx)
        if not obj.frames[frame]: # frame 不含 idx 時刪除
            obj.frames.pop(frame)
            
        if frame.is_keyframe:
            obj.keyframes[frame].discard(idx)
            if not obj.keyframes[frame]: # frame 不含 idx 時刪除
                obj.keyframes.pop(frame)
    
    def update_points_frames_relation(self):
        update_num = 0
        too_much_keyframe_ref = 0
        too_much_point_ref = 0
        
        for point in self.points:
            for keyframe in point.keyframes:
                idx = keyframe.find_point(point)
                if point not in keyframe.points:
                    self.remove_point_frame_relation(point,keyframe,idx)
                    
                    update_num += 1
                    too_much_point_ref += 1
        
        for keyframe in self.keyframes:
            for point in keyframe.points:
                if point is  None: continue
                point:Point
                if keyframe not in point.keyframes:
                    idx = keyframe.find_point(point)
                    self.remove_point_frame_relation(point,keyframe,idx)
                    update_num += 1
                    too_much_keyframe_ref +=1
        
        if update_num !=0:
            print('update relation num',update_num,too_much_keyframe_ref,too_much_point_ref)
    
    
    ####map管理工具####
    def cull_keyframes(self,keyframes:list[KeyFrame]=None):
    # 目前問題:有多個kf視角、位置接近時有機會刪除比較離群的kf
    # TODO: 保留離其他kfs更遠的kf，這個kf可以提供更大的穩定度
        
        culled_kf_count = 0
        if keyframes is None:
            keyframes = list(self.keyframes)
        
        for kf in keyframes:
            if kf.kid == 0 or kf.is_bad:continue
            # if not kf.update_old():continue # 避免刪除新的kf
            if kf.kid == self.keyframes[-1].kid:continue
            
            num_pt = 0
            num_pt_with_redundant_kf=0 # 有冗餘觀察者(keyframe)的point數量
            for pt_idx, pt in enumerate(kf.points): 
                pt:Point|None
                if pt is None:continue
                if pt.is_bad:continue
                num_pt += 1
                
                
                kfs_ret = [kf_ref for kf_ref in pt.keyframes if ((kf_ref is not kf) & (not kf_ref.is_bad))]
                if len(kfs_ret)<3:continue # point 至少有4個觀察者才會有冗餘觀察者
                    
                scale_level = kf.octaves[pt_idx]
                
                # pt.update_normal() # 更新法線單位向量
                # v,norm = normalize_vector(pt.location_3D-kf.position) # kf到pt 的單位向量
                # angle_close = np.dot(pt.normal,v) # 兩個單位向量內積，值越大角度越小
                # kf_closest_notmal = True
                # kf_close_ret = False
                
                # num_better_kf = 0 # 以 相同/更佳 尺度/角度 觀察的keyframe數量(冗餘觀察者)
                nObs = 0
                for kf_ref in kfs_ret: # 檢查有多少冗餘觀察者
                    
                    
                    
                    # v_ret,norm = normalize_vector(pt.location_3D-kf_ref.position)
                    # angle_close_ret = np.dot(pt.normal,v_ret)
                    
                    
                    # close_ret = np.dot(v,v_ret) > 0.866 # cos(30deg) = 0.8660254037844387 = np.cos(np.pi/6)
                    
                    # kf_close_ret = kf_close_ret and close_ret
                    # kf_closest_notmal = kf_closest_notmal and (angle_close > angle_close_ret)
                    
                    
                    pt_ret_idx = kf_ref.find_point(pt)
                    # 檢查kf_ret尺度 最多比 kf尺度 大1才會視為冗餘觀測
                    ret_scale_level = kf_ref.octaves[pt_ret_idx]
                    if ret_scale_level <= scale_level +1:
                        nObs += 1
                        if nObs>=3:
                            break
                    
                # if close_ret and not kf_closest_notmal:
                if nObs>=3:
                    num_pt_with_redundant_kf += 1
                    

                        
                    
            if num_pt_with_redundant_kf > num_pt*0.9:
                kf.is_bad = True
                culled_kf_count += 1
        
        
        return culled_kf_count
    
    def cull_points(self):
        
        
        kfs_last = self.keyframes[-4:]
        old_points:set[Point]=set()
        for kf_last in kfs_last:
            for i,pt in enumerate(kf_last.points):
                if kf_last.new_points[i]:
                    old_points.add(pt)
        
        
        culled_pt_count = 0
            
        for p in old_points:
            if p is None: continue
            
            if (p.kf_num <= 2) and (self.keyframes[-1].kid - p.first_kid >=2):
                p.is_bad = True
            p.update_bad()
            
            if p.is_bad:
                culled_pt_count += 1
                        
        return culled_pt_count
    
    def clear_bad_points(self,f_cur:Frame=None):
        # f_cur 避免刪除當前frame
        num = 0
        points = list(self.points)
        
        for p in points.copy():
            if p.is_bad:
                p.delete()
                num += 1 
                
        
        for kf in self.keyframes.copy():
            if kf.points_num == 0 and (kf is not f_cur):
                kf.is_bad = True
                
        for f in self.frames.copy():
            if f.points_num == 0 and (f is not f_cur):
                f.is_bad = True
        
        return num
                    
    def clear_bad_keyframes(self,kf_cur:KeyFrame):
        num_del_kf = 0
        for kf in self.keyframes.copy():
            if kf is kf_cur:continue # 避免刪除 kf cur
            if kf.update_bad():
                
                kf.delete()
                num_del_kf += 1
        
        return num_del_kf
    
    def clear_bad_object(self):
        
        for thing in self.things.copy():
            if thing.is_bad:
                thing.delete()
        for stuff in self.stuffs.copy():
            if stuff.is_bad:
                stuff.delete()
    

    def replace_frame(self,f1:Frame,f2:Frame):
        '''
        f1取得f2的關聯並刪除f2的關聯\\
        地圖加入f1並刪除f2
        '''
        f1.points = f2.points
        ...
    
    ####共視圖工具####
    # def update_covisibility_graph(self, min_shared_points=15):
    #     """
    #     建立/更新 keyframe 之間的共視圖（covisibility graph）。
    #     只有共同觀測點數超過 min_shared_points 才建立連線。
    #     """
    #     num_kf = len(self.keyframes)
    #     self.covisibility_graph = np.eye(num_kf, dtype=bool)
    #     kf_list = list(self.keyframes)
    #     # 建立 keyframe  到 index 的對應
    #     self.kf_idx_map = {kf: idx for idx, kf in enumerate(kf_list)}

    #     for i, kf1 in enumerate(kf_list):
    #         pts1 = set([p for p in kf1.points if p is not None])
    #         for j in range(i + 1, num_kf):
    #             kf2 = kf_list[j]
    #             pts2 = set([p for p in kf2.points if p is not None])
    #             shared = pts1 & pts2
    #             if len(shared) >= min_shared_points:
    #                 self.covisibility_graph[i, j] = True
    #                 self.covisibility_graph[j, i] = True
    
    # def get_covisible_keyframes(self, keyframe:KeyFrame,include_input=True,best_int=None)->list[KeyFrame]:
    #     """
    #     回傳與指定 keyframe 有共視關係的 keyframe 串列
    #     """
    #     if keyframe not in self.kf_idx_map:
    #         return []
    #     idx = self.kf_idx_map[keyframe]
    #     # 找出共視圖中與該 keyframe 有連線的 index
    #     covisible_indices = np.where(self.covisibility_graph[idx])[0]

    #     kf_list = list(self.keyframes)

        
    #     if include_input:
    #         local_kfs = [kf_list[i] for i in covisible_indices]
    #     else:
    #     # 排除自己
    #         local_kfs = [kf_list[i] for i in covisible_indices if i != idx]
        
    #     return local_kfs
    
    
    def update_covisibility_graph(self, min_shared_points=15):
        """
        建立/更新 keyframe 之間的共視圖（covisibility graph）。
        只有共同觀測點數超過 min_shared_points 才建立連線。
        """
        num_kf = len(self.keyframes)
        self.covisibility_graph = np.zeros((num_kf,num_kf), dtype=int)
        kf_list = list(self.keyframes)
        # 建立 keyframe  到 index 的對應
        self.kf_idx_map = {kf: idx for idx, kf in enumerate(kf_list)}
        
        for i, kf1 in enumerate(kf_list):
            pts1 = set([p for p in kf1.points if p is not None])
            for j in range(i + 1, num_kf):
                kf2 = kf_list[j]
                pts2 = set([p for p in kf2.points if p is not None])
                shared = pts1 & pts2
                shared_num = len(shared)
                if shared_num >= min_shared_points:
                    self.covisibility_graph[i, j] = shared_num
                    self.covisibility_graph[j, i] = shared_num
                    
    def get_covisible_keyframes(self, keyframe:KeyFrame,include_input=True,kfs_num:int=None,th:int=0)->list[KeyFrame]:
        """
        回傳與指定 keyframe 有共視關係的 keyframe 串列，依共視點數量排序
        
        Args:
            keyframe: 參考關鍵幀
            include_input: 是否包含輸入的 keyframe
            kfs_num: 返回的最大關鍵幀數量
            th: 共視點數量閾值
            
        Returns:
            共視關鍵幀列表，按共視點數量降序排列
        """
        if keyframe not in self.kf_idx_map:
            return []
        idx = self.kf_idx_map[keyframe]
        # 找出共視圖中與該 keyframe 有連線的 index與共視點數
        covisible_indices = np.where(self.covisibility_graph[idx] > 0)[0]
        kf_list = list(self.keyframes)
        # 取得共視點數量
        kf_with_score = [(kf_list[i], self.covisibility_graph[idx, i]) for i in covisible_indices]
        
        # if not include_input:
        #     kf_with_score = [(kf, score) for kf, score in kf_with_score if kf != keyframe]
            
        # 依共視點數量排序（由大到小）
        kf_with_score.sort(key=lambda x: x[1], reverse=True)
        
        kfs = [kf for kf, score in kf_with_score if score >= th]
        
        
        # print('keyframe in kfs',keyframe in kfs,len(kfs))
        # print(self.covisibility_graph)
        if include_input:
            kfs.insert(0,keyframe)
        # kfs.insert(0,keyframe)
            
        if kfs_num is not None:
            kfs = kfs[:kfs_num]
        
        
        
        return kfs
    
    def get_covisible_points(self, keyframe:KeyFrame, kfs_num:int=None,th:int=0)->list[Point]:
        """
        回傳與指定 keyframe 有共視關係的 keyframe 上的 point 串列
        """
        local_kfs = self.get_covisible_keyframes(keyframe, kfs_num=kfs_num, th=th)
        local_pts = set()
        for kf in local_kfs: 
            local_pts.update(kf.points)
        # del None in set
        local_pts.discard(None)# discard()。如果用remove()且None 不在 set 中，會出現錯誤
        
        return list(local_pts)
    
    def get_covisible_objects(self, keyframe:KeyFrame, kfs_num:int=None,th:int=0)->list[Thing|Stuff]:
        """
        回傳與指定 keyframe 有共視關係的 keyframe 上的 object 串列
        """
        local_kfs = self.get_covisible_keyframes(keyframe, kfs_num=kfs_num, th=th)
        local_objs:set[None|Thing|Stuff] = set()
        for kf in local_kfs: 
            local_objs.update(kf.objects)
        # del None in set
        local_objs.discard(None)# discard()。如果用remove()且None 不在 set 中，會出現錯誤
        
        return list(local_objs)
    
    def get_covisible_things(self, keyframe:KeyFrame, kfs_num:int=None,th:int=0)->list[Thing]:
        """
        回傳與指定 keyframe 有共視關係的 keyframe 上的 thing 串列
        """
        local_objs = self.get_covisible_objects(keyframe, kfs_num=kfs_num, th=th)
        
        # local_kfs = self.get_covisible_keyframes(keyframe, kfs_num=kfs_num)
        # local_objs:set[None|Thing|Stuff] = set()
        # for kf in local_kfs: 
        #     local_objs.update(kf.objects)
        # # del None in set
        # local_objs.discard(None)# discard()。如果用remove()且None 不在 set 中，會出現錯誤
        local_things = [obj for obj in local_objs if obj.is_thing]
        
        return local_things
    
    def get_covisible_stuffs(self, keyframe:KeyFrame, kfs_num:int=None,th:int=0)->list[Stuff]:
        """
        回傳與指定 keyframe 有共視關係的 keyframe 上的 stuff 串列
        """
        local_objs = self.get_covisible_objects(keyframe, kfs_num=kfs_num, th=th)
        
        # local_kfs = self.get_covisible_keyframes(keyframe, kfs_num=kfs_num)
        # local_objs:set[None|Thing|Stuff] = set()
        # for kf in local_kfs: 
        #     local_objs.update(kf.objects)
        # # del None in set
        # local_objs.discard(None)# discard()。如果用remove()且None 不在 set 中，會出現錯誤
        
        local_stuffs = [obj for obj in local_objs if not obj.is_thing]
        
        return local_stuffs    
    
    # def updata_points_frames_by_map(self):
    #     for point in self.points:
    #         for frame in point.frames:
    #     for keyframe in self.keyframes:
    
    ####檢查工具####
    def check_points_frames_consistency(self):
        # for frame in self.frames:
        #     for point in frame.points:
        #         if point is not None and point.is_bad:
        #             print(f"Warning: frame {frame.id} still holds bad point {point.id}")
        for p in self.points:
            if p.fused_by is None:continue
            for keyframe in p.keyframes:
                print(f"Warning: fused point {p.id} still holds keyframe {keyframe.kid}")
        
        for kf in self.keyframes:
            if kf.is_bad:continue
            for point in kf.points:
                point:Point
                if point is not None and point.fused_by is not None:
                    print(f"Warning: keyframe {kf.kid} still holds point {point.id} fused_by {point.fused_by.id}")
          
    # def check_points_frames_consistency(self):
    # # 檢查 frames 和 points 的一致性
    #     try:
    #         for  frame in self.frames:
    #             for point in frame.points:
    #                 if point is not None:
    #                     point:Point
    #                     assert frame in point.frames, f"Inconsistency: Frame {frame.id} not in Point {point.id}'s frames"

    #         for point in self.points:
    #             for frame in point.frames:
    #                 assert point in frame.points, f"Inconsistency: Point {point.id} not in Frame {frame.id}'s points"
    #     except:
    #         self.update_points_frames_relation()
    
    
    def get_current_ref_frames(self)->list[Frame]:
        '''
        取得上個kf及之後的f
        '''

        n = self.frames[-1].id - self.keyframes[-1].id +1
        
        frames = list(self.frames)[-n:]
        
        return frames
    
    
    ####其他工具####
    def points_fusion(self,frame:Frame,points:list[Point]):
        
        fused_pts_count = 0
        if len(points)==0:return fused_pts_count

        # 檢查點是否可見 及投影點
        points_position = np.array([point.location for point in points ])
        points_visible,  points_project =frame.are_visible(points_position)
        
        if np.sum(points_visible) == 0:
            return fused_pts_count   

        
        for i, point in enumerate(points) :
            # from collections import Counter
            # print('Counter(data)\n',Counter(keyframe.points)[None])
            point:Point
            if (frame in point.frames) | (not points_visible[i]) | point.is_bad | (point.kf_num<2): # 跳過 以觀察到的點、觀察不到的點、不佳的點
                continue
            
            # --- 法線方向判斷 ---
            point.update_normal()
            point.update_depth()

            cam_to_pt = point.location_3D - frame.position
            cam_to_pt = cam_to_pt / np.linalg.norm(cam_to_pt)
            # 內積大於閾值（如0.5），代表在前方
            if np.dot(point.normal, cam_to_pt) < 0.5:
                continue
            
            
            proj = points_project[i][:2]
            
            
            best_dist = math.inf 
            best_dist2 = math.inf
            best_kp_idx = -1
            # 找到最批配point的kp
            for kp_idx,kp in enumerate(frame.kps):
                if np.linalg.norm(kp-proj) > frame.camera.radius/200:  # 投影誤差小於相機半徑1/200
                    continue
                
                descriptor_dist = self.slam.feature_tool.descriptor_distance(point.des,frame.des[kp_idx])
                
                if descriptor_dist < best_dist:                                      
                    best_dist2 = best_dist
                    best_dist = descriptor_dist
                    best_kp_idx = kp_idx   
                elif descriptor_dist < best_dist2:  # N.O.
                    best_dist2 = descriptor_dist       
                
            if best_dist < 50 and best_kp_idx >-1: # best_dist 閾值根據特徵敘述子及距離計算方法選擇 orb:100
                
                # self.add_point_frame_relation(point,frame,best_kp_idx,fusion=True)
                
                fp = frame.points[best_kp_idx]
                if fp is None:
                    # self.add_point_frame_relation(point,frame,best_kp_idx)
                    frame.add_point(point, best_kp_idx)
                    ...
                else:
                    point.fuse(fp)
                    fused_pts_count += 1
    
        
        return fused_pts_count
    
    
    def fuse_points(self, frame: Frame, points: list[Point],th=2.5):
        fused_pts_count = 0
        if len(points) == 0:
            return fused_pts_count

        # 檢查點是否可見及投影點
        points_position = np.array([point.location for point in points])
        points_visible, points_project = frame.are_visible(points_position)
        
        points_pixel = frame.camera.denormalize(points_project)
        
        if np.sum(points_visible) == 0:
            return fused_pts_count
        
        # 預設金字塔層級與尺度因子（如有金字塔可根據距離預測層級）
        # n_max_level = int(np.max(frame.octaves))
        
        scale_factors = frame.scale_factors
        n_max_level = len(scale_factors)-1
        
        for i, point in enumerate(points):
            if point is None:
                continue
            if (point in frame.points) or (frame in point.frames) or (not points_visible[i]) or point.is_bad: # or (point.kf_num < 2):
                continue
            
            # --- 法線方向判斷 ---
            point.update_normal()
            cam_to_pt = point.location_3D - frame.position
            cam_to_pt = cam_to_pt / np.linalg.norm(cam_to_pt)
            if np.dot(point.normal, cam_to_pt) < 0.5:
                continue

            proj = points_pixel[i]
            
            # 深度與尺度一致性檢查
            min_dist,max_dist = point.update_depth()
            dist3D = np.linalg.norm(point.location_3D - frame.position)
            
            
            if not (min_dist < dist3D < max_dist):
                continue
            
            # 預測金字塔層級
            ratio = dist3D / (min_dist + 1e-6) # if min_dist > 0 else 1.0
            n_pred_level = min(np.searchsorted(scale_factors, ratio), n_max_level)
            
            # 搜尋半徑根據尺度調整
            radius = th * scale_factors[n_pred_level]
            
            
            # KDTree搜尋半徑內的keypoint
            near_idxs = frame.raw_kd.query_ball_point(proj, radius)
            # print('near_idxs 0',len(near_idxs))
            
            # # 只考慮未配對的keypoint
            # near_idxs = [idx for idx in near_idxs if frame.points[idx] is None]
            # print('near_idxs 1',len(near_idxs))
            
            
            best_dist = math.inf
            best_dist2 = math.inf
            best_kp_idx = -1
            # for kp_idx, kp in enumerate(frame.raw_kps):
            for kp_idx in near_idxs:
                kp = frame.raw_kps[kp_idx]
                
                
                # 只搜尋預測層級附近的 keypoint
                kp_level = frame.octaves[kp_idx]
                if kp_level < n_pred_level - 1 or kp_level > n_pred_level:
                    continue
                
                # if np.linalg.norm(kp - proj) > radius: # 被KDTree 取代
                #     continue

                descriptor_dist = self.slam.feature_tool.descriptor_distance(point.des, frame.des[kp_idx])

                if descriptor_dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = descriptor_dist
                    best_kp_idx = kp_idx
                elif descriptor_dist < best_dist2:
                    best_dist2 = descriptor_dist

            # NN ratio 檢查
            if best_dist < 50 and best_kp_idx > -1 and best_dist < 0.8 * best_dist2:
                
                fp = frame.points[best_kp_idx]
                fp:Point|None
                if fp is None:
                    # 雙向關聯
                    frame.add_point(point, best_kp_idx)
                    frame.set_track_method(best_kp_idx,5)
                    
                    fused_pts_count += 1
                else:
                    # Replace/合併
                    if not fp.is_bad:
                        fp.fuse(point)
                        frame.set_track_method(best_kp_idx,4)
                        
                        fused_pts_count += 1

        return fused_pts_count


