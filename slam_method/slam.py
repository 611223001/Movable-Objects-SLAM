
import cv2
import numpy as np
from enum import Enum

from .utils import SlamState
from .map import Map
from .camera import Camera
from .frame import Frame, KeyFrame
from .point import Point
from .display import MapDisplay, paint_feature,StateDisplay
from .feature import FeatureTool
from .tracker import Tracker
from .local_mapper import LocalMapper
from .initializer import Initializer
from .visual_odometry import VisualOdometry, MotionModel
from object_segmention.object_segmentation import ObjectSegmentation
from .objects import ObjectTool

from .utils import is_blurry





class Slam():
    
    def __init__(self,camera:Camera,map=Map()):
        
        self.map = map
        self.map.slam = self
        self.camera:Camera = camera
        self.feature_tool = FeatureTool()
        
        # 影像物件分割
        self.object_segmenter = ObjectSegmentation(sam2_reset_mode='no_sam2',segment_background=True,
                                                  config_file="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
                                                  checkpoint_url="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
                                                  )
        self.thing_class_names = self.object_segmenter.thing_class_names
        self.stuff_class_names = self.object_segmenter.stuff_class_names
        
        self.obj_tool = ObjectTool(self)
        
        # 視覺里程計
        self.vo = VisualOdometry(self) # 同時包含 BA
        self.motion_modal = self.vo.mm
        
        # 顯示及控制UI
        self.state_display = StateDisplay(self) # 顯示流程圖及系統資訊
        self.control = self.state_display.control # UI控制系統
        self.map_display = MapDisplay(self) # 顯示點雲圖
        
        # 子系統
        self.tracker = Tracker(self)
        self.local_mapper = LocalMapper(self)
        self.initializer = Initializer(self)
        
        # 屬性: frame 
        self.f_cur:Frame
        self.f_last:Frame
        
        self.kf_cur:KeyFrame
        self.kf_last:KeyFrame
        self.f_ref:Frame|KeyFrame# f_ref 可能是f_last或kf_cur，用於與f_cur比較kps
        
        # 屬性: state
        self.state:SlamState
        self.set_state(SlamState.NO_IMAGE)
        
        # 屬性: other
        
        # self.need_kf:bool = False
        
        # 最大/最小影格插入關鍵影格並檢查重新定位
        self.min_frames:int = int(3)
        self.max_frames:int = int(18*self.camera.fps/30)
    

    # def need_new_keyframe(self)->bool:
    #     A = True
    #     B = self.f_cur.id > self.kf_cur.id +int(self.camera.fps/3) #局部映射處於空閒狀態，或自上次插入關鍵影格以來已過去了 20 多幀。
    #     C = np.sum(self.f_cur.points != None) >50 #當前幀追蹤至少 50 個點。
        
    #     D = self.f_cur.tracking_num < np.sum(self.kf_cur.points != None)*0.90 #目前影格追蹤的點數比 Kref 少 90%。
        
    #     return A and B and C and D


        
# 暫時棄用
    """
    # def local_mapping(self):
    #     # 增加共視點

    #     # self.map.cull_points()
        
    #     fused_pts_count = self.points_fusion(self.kf_cur,list(self.map.points))
    #     print('fused_pts_count',fused_pts_count)
    #     # self.points_fusion(self.kf_cur,np.array(list(self.map.points.values())))
        
    #     self.map.update_points_frames_relation()
    #     # self.map.check_consistency()
    #     kp_idxs1, kp_idxs2, Rt = self.vo.predict_pose(self.kf_cur, self.kf_last)


    #     # for idx1,idx2 in zip(kp_idxs1,kp_idxs2):
    #     #     if self.kf_cur.points[idx2] is not None and self.f_cur.points[idx1] is None:
    #     #         self.map.add_point_frame_relation(self.kf_cur.points[idx1],self.f_cur,idx1)
        
    #     # if not bad_match:
    #     #     print('creatint points')
    #     #     predict_points(self.kf_cur,self.kf_last,kp_idxs1,kp_idxs2)


    #     # check 
    #     for pt in self.map.points.copy(): # copy() 避免迭代中改變串列
    #         for frame in pt.frames:
    #             if (frame not in self.map.frames) and frame not in self.map.keyframes : frame.delete()
    #         if len(pt.frames)==0:
    #             pt.delete()

    #     # local_frames = list(self.map.keyframes)[-4:]
    #     # frames =[]+local_frames
    #     # for local_frame in local_frames:
    #     #     for p in local_frame.points:
    #     #         if p is not None:
    #     #             p:Point
    #     #             frames.extend(p.frames)
    #     # frames = list(set(frames))# 刪除重複項
    #     # local_bundle_adjustment(frames=frames,local_frames=local_frames)
        

    #     # err = bundle_adjustment(list(self.map.frames))
    #     # print('BA error',err)
        
    #     # # clear map
    #     # for frame_id, frame in self.map.frames.copy().items():  # dict 不能在迭代途中改變大小，所以使用copy()
    #     #     if (not frame.is_keyframe) and Frame._id_counter-frame_id>20 :
    #     #         frame.delete()
        
    #     # for point_id, point in self.map.points.copy().items():  # dict 不能在迭代途中改變大小，所以使用copy()
    #     #     if len(point.frames) < 2:
    #     #         point.delete()
    """
    
    def set_state(self,state:SlamState):
        self.state = state
        self.state_display.set_process_state(0,self.state.value)
        

    
    def track(self,img):
        self.out = {}

        
        # 如果未初始化，執行初始化並退出
        if self.state == SlamState.NO_IMAGE:
            # frame = self.tracker.initialize(img)
            success = self.initializer.initialize1(img)
            frame = self.initializer.f_cur

        
        elif self.state == SlamState.INITIALIZING:
            # frame = self.tracker.initialize2(img)
            success = self.initializer.initialize2(img)
            frame = self.initializer.f_cur

            
        else:
            frame = self.tracker.track(img)
        
        
        
        # self.out['img']  = paint_feature(img,frame)
        
        
        
        return self.out