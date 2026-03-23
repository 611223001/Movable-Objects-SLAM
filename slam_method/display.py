# import sdl2
# import sdl2.ext
from multiprocessing.sharedctypes import Synchronized
import cv2
import numpy as np

from multiprocessing import Process, Queue
import multiprocessing as mp
import pangolin
import OpenGL.GL as gl
import time
import copy

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import threading
import time

from .utils import add_ones
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .slam import Slam
    from .frame import Frame
    from .point import Point
    from .camera import Camera

import hashlib
def int_to_rgb(number: int,mode256 = False) -> tuple:
    """
    將整數轉換為 RGB 顏色
    :param number: 輸入的整數
    :return: (R, G, B) 元組，每個值介於 0-255 之間
    /255 -> 0-1
    """
    # 將整數轉換為字串，並計算 SHA-256 哈希值
    hash_value = hashlib.sha256(str(number).encode()).hexdigest()
    
    # 取前 6 個十六進制字符，轉換為 R, G, B
    r = int(hash_value[0:2], 16)
    g = int(hash_value[2:4], 16)
    b = int(hash_value[4:6], 16)
    
    if not mode256:
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
        
    return (r, g, b)

def draw_ground_plane(size=10, step=1):
    gl.glColor3f(0.7, 0.7, 0.7)  # 灰色
    gl.glLineWidth(1)
    gl.glBegin(gl.GL_LINES)
    for i in range(-size, size+1, step):
        # 畫 X 軸線
        gl.glVertex3f(i, 0, -size)
        gl.glVertex3f(i, 0,  size)
        # 畫 Y 軸線
        gl.glVertex3f(-size, 0, i)
        gl.glVertex3f( size, 0, i)
    gl.glEnd()
    

def draw_camera_direction(pose):
    """
    在 Pangolin 視窗中畫出相機的觀測方向（z 軸）
    pose: 4x4 numpy array，世界座標下的相機位姿
    length: 線段長度
    """
    # 畫 z 軸（紅色）
    gl.glColor3f(1.0, 0.0, 0.0)
    gl.glLineWidth(2)
    gl.glBegin(gl.GL_LINES)
    c = pose[:3, 3]
    z_dir = pose[:3, 2]
    length = 5.0
    p2 = c + z_dir * length
    gl.glVertex3f(*c)
    gl.glVertex3f(*p2)
    gl.glEnd()

    # 畫 -y 軸（綠色）
    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glLineWidth(2)
    gl.glBegin(gl.GL_LINES)
    y_dir = pose[:3, 1]
    length = 3.0
    p3 = c + y_dir * -length
    gl.glVertex3f(*c)
    gl.glVertex3f(*p3)
    gl.glEnd()

def draw_circle_billboard(center, radius=0.2, color=(0.5,0.5,0.5), segments=32, thickness=0.05):
    '''
    畫一個空心圓盤（圓環），根據視角自動朝向相機
    thickness: 圓環寬度
    '''
    mv = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
    right = mv[:3, 0]
    up = mv[:3, 1]

    inner_radius = radius - thickness
    outer_radius = radius

    gl.glPushMatrix()
    gl.glColor3f(*color)
    gl.glBegin(gl.GL_TRIANGLE_STRIP)
    for i in range(segments+1):
        theta = 2.0 * np.pi * i / segments
        # 外圓
        pos_outer = (center +
                     outer_radius * np.cos(theta) * right +
                     outer_radius * np.sin(theta) * up)
        gl.glVertex3f(*pos_outer)
        # 內圓
        pos_inner = (center +
                     inner_radius * np.cos(theta) * right +
                     inner_radius * np.sin(theta) * up)
        gl.glVertex3f(*pos_inner)
    gl.glEnd()
    gl.glPopMatrix()
    
    
class MapDisplay(object):
    def __init__(self,slam:'Slam'):
        # self.q_map_state = None # A queue for inter-process communication. | q for visualization process
        # self.q_frame = None
        self.state = None # variable to hold current state of the map and cam pose
        self.frame_info:'Frame' = None
        
        self.slam = slam
        self.map = slam.map
        
        self.create_viewer()

    def create_viewer(self):
        # Parallel Execution: The main purpose of creating this process is to run 
        # the `viewer_thread` method in parallel with the main program. 
        # This allows the 3D viewer to update and render frames continuously 
        # without blocking the main execution flow.
        
        self.q_map_state = Queue(0) # q is initialized as a Queue
        self.q_frame = Queue(0)
        self.q_info = Queue(0)

        # initializes the Parallel process with the `viewer_thread` function 
        # the arguments that the function takes is mentioned in the args var
        # self.display_process = Process(target=self.viewer_thread, args=(self.q_map_state,self.q_frame),daemon = True)
        self.run = True
        self.display_process = threading.Thread(target=self.viewer_thread, args=(self.q_map_state, self.q_frame, self.q_info),daemon = True) 
        
        # daemon true means, exit when main program stops
        
        # self.display_process.daemon = True
        
        # starts the process
        self.display_process.start()
        
    def viewer_thread(self, q_map_state, q_frame, q_info):
        # `viewer_thread` takes the q as input
        # initializes the viz window
        self.viewer_init(1920, 1080)
        # An infinite loop that continually refreshes the viewer
        while self.run:
            self.viewer_refresh(q_map_state)
            self.img_refresh(q_frame)
            self.other_info_refresh(q_info)
            time.sleep(0.03)

            
            
    def viewer_init(self, w, h):
        # 記錄視窗大小以便完整擷取截圖
        self.window_w = w
        self.window_h = h
        pangolin.CreateWindowAndBind('Map', w, h)
        
        # This ensures that only the nearest objects are rendered, 
        # creating a realistic representation of the scene with 
        # correct occlusions.
        gl.glEnable(gl.GL_DEPTH_TEST)
        local0 = self.map.pose0[:3,3].flatten()
        camera0 = np.array([0,0,-20,1])
        camera0 = np.dot(self.map.pose0,camera0.T).T
        
        # Sets up the camera with a projection matrix and a model-view matrix
        self.scam = pangolin.OpenGlRenderState(
            # `ProjectionMatrix` The parameters specify the width and height of the viewport (w, h), the focal lengths in the x and y directions (420, 420), the principal point coordinates (w//2, h//2), and the near and far clipping planes (0.2, 10000). The focal lengths determine the field of view, 
            # the principal point indicates the center of the projection, and the clipping planes define the range of distances from the camera within which objects are rendered, with objects closer than 0.2 units or farther than 10000 units being clipped out of the scene. 
            # `ProjectionMatrix` 參數指定了視窗的寬度和高度 (w、h)、x 和 y 方向的焦距 (420、420)、主點座標 (w//2、h//2)，以及近距離和遠距離剪輯平面 (0.2、10000)。焦距決定視野、 
            # 主點表示投影的中心，剪接平面定義了渲染物件與攝影機的距離範圍，距離攝影機近於 0.2 個單位或遠於 10000 個單位的物件會被剪接出場景。
            pangolin.ProjectionMatrix(w, h,  w//2, h//2, w//2, h//2, 0.2, 10000),
            # pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0) 設定攝影機視圖矩陣，定義攝影機在 3D 场景中的位置和方向。
            # 前三個參數 (0, -10, -8) 指定攝影機在世界座標中的位置，表示攝影機位於座標 (0, -10, -8)。
            # 接下來的三個參數 (0, 0, 0) 定義攝影機所看的空間點，在此為原點。
            # 最後三個參數 (0, -1, 0) 代表向上的方向向量，表示攝影機「向上」的方向，這裡是沿著 y 軸的負方向。
            # 這個設定有效地將攝影機定位在原點向下 10 個單位、向後 8 個單位的位置，攝影機看向原點，而「向上」的方向是 y 軸向 下，這是非常規的，
            # 可能會用來在渲染的場景中達成特定的方向或透視。
            pangolin.ModelViewLookAt(camera0[0], camera0[1], camera0[2], local0[0], local0[1], local0[2], 0, -1, 0))
        # Creates a handler for 3D interaction.
        self.handler = pangolin.Handler3D(self.scam)
        
 
        # Creates a display context.
        self.dcam = pangolin.CreateDisplay()
        # Sets the bounds of the display
        self.dcam.SetBounds(0.0, 1.0, 0.1, 1.0, -w/h)
        # assigns handler for mouse clicking and stuff, interactive
        self.dcam.SetHandler(self.handler)
        # self.darr = None

        # 控制
        self.ctrl = pangolin.CreatePanel('ctrl').SetBounds(0.0, 1.0, 0.0, 0.1)
        
        self.ctrl_show_things = pangolin.VarBool('ctrl.ShowThings',  True, True)    # (name,初始值,勾選框) 勾選框 False時為按鈕
        self.ctrl_show_points = pangolin.VarBool('ctrl.ShowPoints',  True, True)
        self.ctrl_points_color = pangolin.VarBool('ctrl.PointsColor',  True, True)
        self.ctrl_points_local = pangolin.VarBool('ctrl.PointsLocal',  False, True)
        self.ctrl_points_oid = pangolin.VarBool('ctrl.PointsOid',  False, True)
        
        self.ctrl_follow = pangolin.VarBool('ctrl.Follow',  False, True) 
        self.ctrl_show_camera = pangolin.VarBool('ctrl.ShowCamera',  True, True)
        self.ctrl_show_pose = pangolin.VarBool('ctrl.ShowPose',  True, True)
        self.ctrl_show_kf = pangolin.VarBool('ctrl.ShowKF',  True, True)
        self.ctrl_show_loc_kf = pangolin.VarBool('ctrl.ShowLocKF',  True, True)
        
        self.ctrl_show_GT = pangolin.VarBool('ctrl.ShowGoundTruth',  True, True)
        self.ctrl_show_axis = pangolin.VarBool('ctrl.ShowAxisPlane',  True, True)
        
        # # image
        # width, height = 480, 270
        # self.dimg = pangolin.Display('image')
        # self.dimg.SetBounds(0, height / 768., 0.0, width / 1024., 1024 / 768.)
        # self.dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        # self.texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        # self.image = np.ones((height, width, 3), 'uint8')

    # 3D map 更新
    def viewer_refresh(self, q_map_state:Queue):

        # # Checks if the current state is None or if the queue is not empty.
        # if self.state is None or not q_map_state.empty():
        #     # Gets the latest state from the queue.
        #     self.state = q_map_state.get()

        
        while not q_map_state.empty() or self.state is None:
            self.state = q_map_state.get()
            

        # Clears the color and depth buffers.
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Sets the clear color to white.
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        
        # 控制相機位置:跟隨最新畫面
        # if self.slam.control.follow:
        if bool(self.ctrl_follow.Get()):
            # 取得最新相機 pose
            pose = self.state['localposes'][-1]
            cam_pos = pose[:3, 3]
            # 設定相機看向的點與 up 向量
            look_at = cam_pos + pose[:3, 2]  # z 軸方向
            up = -pose[:3, 1]                 # -y 軸方向
            # 設定 ModelView
            mv = pangolin.ModelViewLookAt(
                cam_pos[0], cam_pos[1], cam_pos[2],
                look_at[0], look_at[1], look_at[2],
                up[0], up[1], up[2]
            )
            self.scam.SetModelViewMatrix(mv)
            
        # Activates the display context with the current camera settings.
        self.dcam.Activate(self.scam)

        if bool(self.ctrl_show_camera.Get()):
            camera = self.state['camera']
            pose = camera['pose']
            self.draw_camera(pose,camera)
        
        
        # camera trajectory line and color setup
        # 框線粗細
        gl.glLineWidth(1)
        # draw 最近的frame poses
        if bool(self.ctrl_show_pose.Get()):
            gl.glColor3f(0.0, 0.0, 1.0)
            camera = self.state['camera']
            pose = camera['pose']
            pangolin.DrawCameras([pose],1)
            draw_camera_direction(pose)
            # pangolin.DrawCameras(self.state['localposes'],1)
            # draw_camera_direction(self.state['localposes'][-1])
        
        # draw keyframe poses
        if bool(self.ctrl_show_kf.Get()):
            gl.glColor3f(0.0, 1.0, 0.0)
            if len(self.state['keyposes'])>1:
                pangolin.DrawCameras(self.state['keyposes'],1) # DrawCameras 第二個參數是相機框線的長寬
        
        
        if bool(self.ctrl_show_loc_kf.Get()):
            loc_kf_pose = self.state['local_keyposes']
            gl.glColor3f(0.0, 1.0, 1.0)
            if len(loc_kf_pose)>1:
                pangolin.DrawCameras(loc_kf_pose,1) 
            
        
        # 地面真值 pose
        # if self.slam.control.groundtruth:
        if bool(self.ctrl_show_GT.Get()):
            groundtruth = self.state['groundtruth']
            if groundtruth is not None:
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawCameras(groundtruth,1)
                if len(groundtruth)>1:
                    draw_camera_direction(groundtruth[-1])
        
        # if groundtruth is None:...
        # elif len(groundtruth)>1:
        #     draw_camera_direction(groundtruth[-1])
        
        # 3d point cloud color setup
        # if self.state['pts'].size > 0 and bool(self.ctrl_show_points.Get()):
        #     for point,color in zip(self.state['pts'], self.state['pts_color']):
        #         # # 顯示類型顏色或是影像顏色
        #         # if False: # 類型顏色
        #         #     ...
        #         # else: # 影像顏色
        #         #     color = color
        #         self.draw_point(point,color,size=5)
        
        if self.state['pts'].size > 0 and bool(self.ctrl_show_points.Get()):
            for pt_state in self.state['pts_state']:
                point = pt_state['loc']
                
                if bool(self.ctrl_points_color.Get()):
                    color = pt_state['color']
                elif bool(self.ctrl_points_local.Get()):
                    local_level = pt_state['local_level']
                    match local_level:
                        
                        case 0: # in current frame
                            color = (0,1,0)
                        case 1: # in local kf
                            color = (0,0,1)
                        case _: # other points
                            color = (1,0,0)
                            
                    
                elif bool(self.ctrl_points_oid.Get()):
                    oid = pt_state['oid']
                    if oid is None:
                        color = (0.5,0.5,0.5)
                    else:
                        color = int_to_rgb(oid)
                else:
                    color = (0.5,0.5,0.5)
                    
                self.draw_point(point,color,size=5)


                

        
        # 繪製things
        # for thing,radius,id in self.state['things']:
        #     color = int_to_rgb(id)
        #     draw_circle_billboard(thing[0:3],radius,color)
        if bool(self.ctrl_show_things.Get()):
            for vertices,faces ,id in self.state['things']:
                color = int_to_rgb(id)
                self.draw_thing(vertices,faces,color)
        
        # 繪製平面網格
        if bool(self.ctrl_show_axis.Get()):
            draw_ground_plane(size=1000, step=1)

        
        # Finishes the current frame and swaps the buffers.
        pangolin.FinishFrame()
        
        control =  self.slam.control
        if control.shoot:
            self.save_screenshot(self.state['time'])
    
    # image 更新
    def img_refresh(self, q_frame:Queue):
        
        while not q_frame.empty() or self.frame_info is None:
            self.frame_info = q_frame.get()
            
        control =  self.slam.control
        frame_info = self.frame_info
        img = frame_info['img'].copy()
        
        width, height = img.shape[:2][::-1]
        
        # 繪製objects in map
        if control.show_objs: # control
            objs = frame_info['objs_map']
            for obj in objs:
                
                if obj['id'] is None:
                    continue
                
                if not (control.show_obj_id is None or control.show_obj_id == obj['id']): continue
            
                else:
                    if (obj['is_thing']and control.show_obj_thing) or (not obj['is_thing']and control.show_obj_stuff):
                        
                        color = int_to_rgb(obj['id'],mode256=True)
                        color = (color[2],color[1],color[0]) # rgb 2 bgr
                        img = draw_contour_with_mask(img,obj['mask'],color,1)
                        id = str(obj['id'])
                        id = ': ' + id
                        if control.show_obj_info:
                            if obj['is_thing']:
                                text = self.slam.thing_class_names[obj['category_id']]+id
                            else:
                                
                                if obj['category_id'] is None:
                                    # print(obj)
                                    text ='None'+id
                                else:
                                    text = self.slam.stuff_class_names[obj['category_id']]+id
                                
                            img = objoct_info(img,obj['mask'],text,color)
                
                if control.show_obj_dynamic:
                    dynamic:None|bool = obj['dynamic'] # None|True|False

                    match dynamic:
                        case True:
                            color = (0,0,255)
                            img = draw_mask(img,obj['mask'],color)
                        case None:
                            color = (127,127,127)
                            img = draw_mask(img,obj['mask'],color)
                        case False:
                            ...
                            
        # 繪製objects in img
        if control.show_objs_img: # control
            objs = frame_info['objs_img']
            for obj in objs:
                if not (control.show_obj_id is None or control.show_obj_id == obj['id']): continue
                
                if obj['id'] == 0:
                    
                    if control.show_obj_none:
                        color = (127,127,127)
                        img = draw_contour_with_mask(img,obj['mask'],color,1,True)
            
                else:
                    if (obj['is_thing']and control.show_obj_thing) or (not obj['is_thing']and control.show_obj_stuff):
                        
                        color = int_to_rgb(obj['id'],mode256=True)
                        color = (color[2],color[1],color[0]) # rgb 2 bgr
                        img = draw_contour_with_mask(img,obj['mask'],color,1)
                        id = str(obj['id'])
                        id = ': ' + id
                        if control.show_obj_info:
                            if obj['is_thing']:
                                text = self.slam.thing_class_names[obj['category_id']]+id
                            else:
                                text = self.slam.stuff_class_names[obj['category_id']]+id
                                
                            img = objoct_info(img,obj['mask'],text,color)
                
                if control.show_obj_dynamic:
                    dynamic:None|bool = obj['dynamic'] # None|True|False

                    match dynamic:
                        case True:
                            color = (255,0,0)
                            img = draw_mask(img,obj['mask'],color)
                        case None:
                            color = (127,127,127)
                            img = draw_mask(img,obj['mask'],color)
                        case False:
                            ...
        
        
        # 繪製kps
        if control.show_kps: # control
            kps = frame_info['kps']
            for kp in kps:
                mode = kp['mode']
                uv0 = kp['uv0']
                uvs = kp['uvs']
                
                uvs = np.array(uvs, dtype=np.int32).reshape(1,-1,2)
                
                track_method = kp['track_method']
                
                
                # color (B,G,R)
                match mode:
                    case None:      # p is None
                        if control.show_kps_detected:
                            p_color = (255,255,255)
                        else:continue
                        
                    case 'new':     # p is new
                        p_color = (127,127,127)
                        
                    case 'proj':    # p is project
                        p_color = (0,0,255)# relocaliztion/投影 取得的點 以紅色顯示
                                                
                    case 'long':    # p is long
                        p_color = (0,255,0)
                        
                    case _:
                        print('something wrong')
                        
                
                
                
                if control.mothed_color:
                    """
                    0:None
                    1:new
                    2:project
                    3:flow
                    4:fuse
                    5:project in fuse
                    """
                    # color (B,G,R)
                    line_color = (127,127,127)
                    match track_method:
                        case 0:
                            p_color = (127,127,127)
                        case 1:
                            p_color = (0,0,0)
                        case 2:
                            p_color = (0,255,255)
                            line_color = (0,255,255)
                        case 3:
                            p_color = (0,255,0)
                            line_color = (0,255,0)
                        case 4:
                            p_color = (255,0,0)
                            line_color = (255,0,0)
                        case 5:
                            p_color = (255,255,0)
                            line_color = (255,255,0)
                else:
                    line_color = getjet(uvs.shape[1])*255
                
                    
                cv2.circle(img, uv0, color=p_color , radius=2)
                
                if control.long_track_kps:
                    cv2.polylines(img,uvs,False, line_color , thickness=1, lineType=16)
                elif control.mothed_color: # mothed_color 時顯示kps變化
                    cv2.polylines(img,uvs[:,:2],False, line_color , thickness=1, lineType=16)
                    

        # 繪製其他info
        infos = frame_info['infos']
        
        # paint_info(img,infos)
        
        
        cv2.imshow('current frame',img)
        cv2.waitKey(1)
        
        # 紀錄圖片
        if control.shoot:
            fid = frame_info['fid']
            # 保存圖片
            screenshot_dir = "cache/screenshots/img"
            filename = f"{screenshot_dir}/{fid}.png"
            cv2.imwrite(filename, img)
        
        
        # # 顯示objects投影及距離
        # if frame_info['obj_prj_id_map'] is not None and frame_info['obj_prj_depth'] is not None:
        #     prjed_mask = np.zeros((height, width, 3),np.uint8)
            
        #     # 投影影像
        #     id_map = frame_info['obj_prj_id_map']
        #     ids = np.unique(id_map)
        #     for id in ids:
        #         if id == -1:
        #             color = (0,0,0)
        #         else:
        #             color = int_to_rgb(id, mode256=True)
        #         color = (color[2],color[1],color[0]) # rgb 2 bgr
        #         # mask = np.where(id_map==id)
        #         mask = (id_map == id)
        #         prjed_mask = draw_mask(prjed_mask,mask,color,1)
            
        #     # 深度影像
        #     depth_map = frame_info['obj_prj_depth']
        #     # print(f'depth_map max:{depth_map.max()} min:{depth_map.min()}')
        #     # depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        #     # depth_uint8 = depth_norm.astype(np.uint8)
        #     # # 使用 colormap 增加可視化效果（例如 COLORMAP_JET）
        #     # depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        #     # depth_color[depth_map == np.inf]=(0,0,0)
            
        #     # 建立有效深度遮罩（忽略 np.inf 與 <=0）
        #     valid_mask = (depth_map > 0) & np.isfinite(depth_map)
        #     depth_map_valid = np.where(valid_mask, depth_map, 0)
        #     # print(f'depth_map max:{depth_map_valid.max()} min:{depth_map.min()}')

        #     depth_norm = cv2.normalize(depth_map_valid, None, 0, 255, cv2.NORM_MINMAX)
        #     depth_uint8 = depth_norm.astype(np.uint8)
        #     # 使用 colormap 增加可視化效果（例如 COLORMAP_JET）
        #     depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        #     depth_color[~valid_mask] = (0, 0, 0)
        #     paint_info(depth_color,[f'depth max:{depth_map_valid.max():.3f}',f'depth min:{depth_map.min():.3f}'])
            
        #     obj_and_prjed_mask = cv2.hconcat([prjed_mask, depth_color])
        #     cv2.imshow('object and projected mask ',obj_and_prjed_mask)
        #     cv2.waitKey(1)
    
    # 其他資訊更新(可以方便的顯示一些結果，方便測試)
    def other_info_refresh(self, q_info:Queue):
        
        if q_info.empty():
            return
        
        
        key, info, timestamp = q_info.get()
        
        match key:
            case 'obj_proj':
                print('obj_proj')
                (id_map, depth_map) = info
                
                # width, height = img.shape[:2][::-1]
                height, width = depth_map.shape[:2]
                prjed_mask = np.zeros((height, width, 3),np.uint8)
            
                # 投影影像
                ids = np.unique(id_map)
                for id in ids:
                    if id == -1:
                        color = (0,0,0)
                    else:
                        color = int_to_rgb(id, mode256=True)
                    color = (color[2],color[1],color[0]) # rgb 2 bgr
                    # mask = np.where(id_map==id)
                    mask = (id_map == id)
                    prjed_mask = draw_mask(prjed_mask,mask,color,1)
                

                
                # 深度影像
                # 建立有效深度遮罩（忽略 np.inf 與 <=0）
                valid_mask = (depth_map > 0) & np.isfinite(depth_map)
                depth_map_valid = np.where(valid_mask, depth_map, 0)
                # print(f'depth_map max:{depth_map_valid.max()} min:{depth_map.min()}')

                depth_norm = cv2.normalize(depth_map_valid, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_norm.astype(np.uint8)
                # 使用 colormap 增加可視化效果（例如 COLORMAP_JET）
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                depth_color[~valid_mask] = (0, 0, 0)
                paint_info(depth_color,[f'depth max:{depth_map_valid.max():.3f}',f'depth min:{depth_map.min():.3f}'])
                
                obj_and_prjed_mask = cv2.hconcat([prjed_mask, depth_color])
                cv2.imshow('object and projected mask ',obj_and_prjed_mask)
                cv2.waitKey(1)
            
            
            
            case 'OF_obj':
                # print('OF_obj')
                (oids, masks_remap) = info
                
                # sums = [(id,np.sum(mask)) for id, mask in zip(oids, masks_remap)]
                # print(sums)
                height, width = masks_remap[0].shape[:2]
                img = np.zeros((height, width, 3),np.uint8)
                
                for id, mask in zip(oids, masks_remap):
                    
                    color = int_to_rgb(id, mode256=True)
                    color = (color[2],color[1],color[0]) # rgb 2 bgr
                    
                    # 繪製遮罩
                    img = draw_mask(img, mask, color, 0.3)
                
                cv2.setWindowTitle("OF_obj", f"OF_obj --obj num: {len(oids)}")
                cv2.imshow("OF_obj", img)
                cv2.waitKey(1)
                
                
            case 'carve_hull':
                (oid,old_mesh_projected, exts, holes, obscured_exts, new_mesh_projected) = info
                
                
                height, width = old_mesh_projected.shape[:2]
                img = np.zeros((height, width, 3),np.uint8)
                
                color = int_to_rgb(oid, mode256=True)
                color = (color[2],color[1],color[0]) # rgb 2 bgr
                
                # 繪製遮罩
                mask = old_mesh_projected
                img = draw_mask(img, mask, color, 0.5)
                
                if new_mesh_projected is not None:
                    step = 2
                    mask = new_mesh_projected
                    img = draw_mask(img, mask, color, 0.5)
                else:
                    step = 1
                
                cv2.polylines(img, exts, isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.polylines(img, obscured_exts, isClosed=True, color=(255, 0, 0), thickness=2)
                for hole in holes:
                    cv2.polylines(img, hole, isClosed=True, color=(0, 0, 255), thickness=2)

                cv2.setWindowTitle("carve_hull", f"carve_hull --step: {step} --oid: {oid}")
                cv2.imshow("carve_hull", img)
                cv2.waitKey(1)
                
            
            case 'obj_depth':
                (name, depth) = info
                
                # 深度影像
                # 建立有效深度遮罩（忽略 np.inf 與 <=0）
                valid_mask = (depth > 0) & np.isfinite(depth)
                depth_map_valid = np.where(valid_mask, depth, 0)
                # print(f'depth_map max:{depth_map_valid.max()} min:{depth_map.min()}')

                depth_norm = cv2.normalize(depth_map_valid, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_norm.astype(np.uint8)
                # 使用 colormap 增加可視化效果（例如 COLORMAP_JET）
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                depth_color[~valid_mask] = (0, 0, 0)
                
                valid_depths = depth[valid_mask]
                if valid_depths.size > 0:  # 檢查是否有有效深度
                    depth_min = valid_depths.min()
                else:
                    depth_min = 0  # 沒有有效深度時顯示 0
                
                paint_info(depth_color,[f'depth max:{depth_map_valid.max():.3f}',f'depth min:{depth_min:.3f}'])
                
                
                
                
                cv2.imshow(name, depth_color)
                cv2.waitKey(1)
                
            
            case 'img':
                (name,img) = info
                cv2.imshow(name, img)
                cv2.waitKey(1)
                
            case 'flow':
                (name,flow) = info
                
                # 將光流轉換為大小與角度
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # 建立 HSV 影像來表示光流
                h,w,_ = flow.shape
                hsv = np.zeros((h,w,3),dtype=np.uint8)
                
                hsv[..., 1] = 255  # 飽和度設為最大

                # 根據角度（流動方向）設定色相
                hsv[..., 0] = angle * 180 / np.pi / 2

                # 根據大小（流動強度）設定亮度，並正規化到 0-255
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

                # 將 HSV 轉回 BGR 以便顯示
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # 顯示光流影像

                cv2.imshow(name, flow_bgr)
                cv2.waitKey(1)
            
            case 'clean':
                (name) = info
                
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(name)
            
            case _:
                ...
        
        

            
            
            
            
            
    # ---------- 幾何處理：三角化 & 法向量 ----------
    def triangulate_face(self,face_indices):
        """
        將 N 邊形用扇形三角化： (v0, v1, v2), (v0, v2, v3), ...
        輸出為 (M, 3) 的索引陣列
        """
        if len(face_indices) < 3:
            return np.empty((0,3), dtype=np.int32)
        tris = []
        v0 = face_indices[0]
        for i in range(1, len(face_indices)-1):
            tris.append([v0, face_indices[i], face_indices[i+1]])
        return np.array(tris, dtype=np.int32)

    def compute_face_normal(self,v0, v1, v2):
        """
        以三角形三點計算法向量（右手定則）
        """
        a = v1 - v0
        b = v2 - v0
        n = np.cross(a, b)
        norm = np.linalg.norm(n) + 1e-12
        return n / norm
    
    # ---------- 繪製：面 / 邊 / 點 ----------
    def draw_faces(self,vertices, faces, color=(0.8,0.8,0.9), alpha=0.5):
        """
        vertices: 物體的頂點
        faces: 物體的面，由數個頂點的索引組成
        color = (r,g,b)
        """
        # 開啟透明混合，畫半透明面
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)   # 避免與邊線 Z-fighting
        gl.glPolygonOffset(1.0, 1.0)

        gl.glColor4f(color[0], color[1], color[2], alpha)
        gl.glBegin(gl.GL_TRIANGLES)
        for f in faces:
            tris = self.triangulate_face(f)
            for t in tris:
                v0, v1, v2 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
                n = self.compute_face_normal(v0, v1, v2)
                gl.glNormal3f(n[0], n[1], n[2])
                gl.glVertex3f(v0[0], v0[1], v0[2])
                gl.glVertex3f(v1[0], v1[1], v1[2])
                gl.glVertex3f(v2[0], v2[1], v2[2])
        gl.glEnd()

        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDisable(gl.GL_BLEND)
        gl.glColor3f(1,1,1)
    
    def draw_edges(self,vertices, faces, color=(0.1, 0.1, 0.1), line_width=1.5):
        """
        color = (r,g,b)
        """
        gl.glLineWidth(line_width)
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINES)
        for f in faces:
            # 邊：順序連成環
            for i in range(len(f)):
                a = vertices[f[i]]
                b = vertices[f[(i+1) % len(f)]]
                gl.glVertex3f(a[0], a[1], a[2])
                gl.glVertex3f(b[0], b[1], b[2])
        gl.glEnd()
        gl.glColor3f(1,1,1)
    
    def draw_stuff(self):
        ...
    def draw_thing(self,vertices, faces, color=(0.8,0.8,0.9), alpha=0.5):
        """
        color = (r,g,b)
        """
        self.draw_faces(vertices,faces,color,alpha)
        
        self.draw_edges(vertices,faces,color)
        # if vertices is not None:
        #     self.draw_points(vertices,(0.1,0.1,0.1),3)
    
    def draw_cameras(self,poses,color,linewidth=1):
        """
        color = (r,g,b)
        """
        gl.glLineWidth(linewidth)
        gl.glColor3f(*color)
        pangolin.DrawCameras(poses,1)
        draw_camera_direction(poses[-1])
    
    def draw_camera(self,pose, camera, color=(0.2, 0.7, 1.0), scale=1, linewidth=2.0):
        """
        pose: 4x4，Twc 相機在世界的位姿（從相機座標到世界座標的變換）
        scale: 相機框大小
        """
        gl.glColor3f(*color)
        img_w = camera['width'] - 1
        img_h = camera['height'] - 1
        # 影像四角（像素座標）
        pix = np.array([
            [0,     0    ],
            [img_w, 0    ],
            [img_w, img_h],
            [0,     img_h],
        ], dtype=np.float64)
        # 在相機座標系的四條光線方向（z=1 的歸一化平面）
        
        Kinv = camera['Kinv']
        rays = np.dot(Kinv, add_ones(pix).T).T# shape (4,3)
        # 正規化為單位向量
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

        # 得到平面四角（沿 +Z 方向）
        # 對於 ray=(x,y,z)，在深度 d（沿 +Z）上的 3D 點 = ray * (d / z)
        pts_c = np.array([r * (scale / r[2]) for r in rays])

        # 轉齊次座標
        pts_c = add_ones(pts_c)
        # 相機中心（原點）
        # Cc = np.array([0,0,0,1])

        # 轉到世界座標，去齊次
        pts_w = (pose @ pts_c.T).T[:,:3]

        
        # 相機中心
        Cw = pose[:3,3].T
        # print(Cw,pose,pts_w)
        
        # pangolin.DrawCameras([pose], 10)
        # self.draw_cameras([pose],color)
        # self.draw_points(pts_w,size=500)
        # self.draw_point(Cw,size=500)
        
        # 畫線
        gl.glLineWidth(linewidth)
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINES)

        # 相機中心到近/遠四角
        for i in range(4):
            gl.glVertex3f(*Cw)
            gl.glVertex3f(*pts_w[i])

        # 平面框
        idx = [0,1,2,3,0]
        for i in range(4):
            gl.glVertex3f(*pts_w[idx[i]])
            gl.glVertex3f(*pts_w[idx[i+1]])
        gl.glEnd()


        # 恢復顏色
        gl.glColor3f(1,1,1)
    
    def draw_points(self,points, color=(0.95, 0.3, 0.3), size=5.0):
        """
        color = (r,g,b)
        """
        gl.glPointSize(size)
        gl.glColor3f(color[0], color[1], color[2])
        # for p in points:
        #     gl.glVertex3f(p[0], p[1], p[2])
        pangolin.DrawPoints(points) # 高階 API，可以一次繪製多個點
        gl.glColor3f(1,1,1)
        
    def draw_point(self,point, color=(0.95, 0.3, 0.3), size=5.0):
        """
        color = (r,g,b)
        """
        gl.glPointSize(size)
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(*point[:3])
        gl.glEnd()
        gl.glColor3f(1,1,1)
    
    def save_screenshot(self,time:int):
        """
        從 OpenGL 幀緩衝區讀取像素並保存為圖片
        """
        # 讀取整個視窗影像（而非目前的子 viewport），避免左側控制面板被裁切
        x, y = 0, 0
        width = self.window_w
        height = self.window_h

        # 確保像素列不加額外對齊填充，避免右側出現黑邊
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        # 讀取前台緩衝區內容（與已顯示畫面一致）
        try:
            gl.glReadBuffer(gl.GL_FRONT)
        except Exception:
            # 某些環境不需要或不支援切換 ReadBuffer，忽略即可
            pass
        
        # 從幀緩衝區讀取像素 (RGB 格式)
        pixels = gl.glReadPixels(x, y, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        
        # 將像素數據轉換為 numpy 數組
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        
        # OpenGL 的坐標系統與圖像不同,需要垂直翻轉
        image = np.flipud(image)
        
        # BGR 轉 RGB (OpenCV 使用 BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        screenshot_dir = "cache/screenshots/map"
        # 保存圖片
        filename = f"{screenshot_dir}/{time}.png"
        cv2.imwrite(filename, image)
        
    
    
    
    def display_map(self):
        # if self.q_map_state is None:
        #     return
        kf_cur = None
        for kf in self.map.keyframes[-1:]:
            kf_cur = kf
        local_kps = self.map.get_covisible_keyframes(kf_cur,kfs_num = 20)
        local_pts = self.map.get_covisible_points(kf_cur,kfs_num = 20)
        
        f_cur = list(self.map.frames)[-1]
        
        
        keyposes,localposes, pts, pts_color,things = [], [], [], [], []
        local_keyposes = []
        pts_state:list[dict] = []
        # 提取 kf pose
        for kf in self.map.keyframes:
            # updating pose
            keyposes.append(kf.pose)
            if kf in local_kps:
                local_keyposes.append(kf.pose)
            
            
        # 提取最近幾個 f pose
        for f in list(self.map.frames)[-10:]:
            localposes.append(f.pose)
            
        # 提取 point的位置、顏色、屬於物件
        for p in self.map.points:
            
            if p.dynamic:continue
            # updating map points
            pts.append(p.location)
            pts_color.append(p.color)
            if p.object is None:
                oid = None
            else:
                oid = p.object.oid
                
            local_level = -1
            if p in f_cur.points:
                local_level = 0
            elif p in local_pts:
                local_level = 1
            
            
            state = {
                'loc':p.location,
                'color':p.color/256.0,
                'local_level':local_level,
                'oid':oid,
            }
            pts_state.append(state)
        
        # 提取 object 
        for thing in self.map.things:
            # if thing.center_location[3] == 1:
            #     things.append((thing.center_location,thing.radius,thing.id))
            if thing.is_bad:continue
            # if thing.dynamic:continue
            things.append((thing.vertices, thing.faces, thing.oid))
            
        if self.map.groundtruth is None:
            groundtruth = None
        else:
            groundtruth = self.map.groundtruth[:self.map.frames[-1].id]

        
        camera = {
            'pose':localposes[-1],
            'width':self.slam.camera.width,
            'height':self.slam.camera.height,
            'Kinv':self.slam.camera.Kinv,
        }
        # updating queue
        self.q_map_state.put({
            'time':f_cur.id,
            'localposes':np.array(localposes),
            'keyposes':np.array(keyposes),
            'local_keyposes':np.array(local_keyposes),
            'pts':np.array(pts),
            'pts_color':np.array(pts_color)/256.0,
            'pts_state':pts_state,
            'things':np.array(things, dtype=object),
            'groundtruth':groundtruth,
            'camera':camera,
        })
        
        # if self.map.groundtruth is None:
        #      self.q_map_groundtruth.put(None)
        # else:
        #     # self.q_map_groundtruth.put(self.map.groundtruth)                # 一次顯示所有groundtruth
        #     self.q_map_groundtruth.put(self.map.groundtruth[:self.map.frames[-1].id])   # 根據現在影像顯示groundtruth

    def display_img(self,frame:'Frame'=None):
        if frame is None:
            frame = self.map.frames[-1]
        
        height, width = frame.camera.height, frame.camera.width
            
        objs_map,objs_img,kps = [],[],[]
        # 物件顯示
        for idx, (obj,info,dynamic) in enumerate(zip(frame.objects, frame.obj_infos, frame.objects_dynamic)):
            
            if obj is None:
                objs_map.append({
                    'id':None,
                    'is_thing':None,
                    'mask':info['mask'],
                    'category_id':None,
                    'dynamic':None
                })
            else:
                obj.updata_category_id()
                objs_map.append({
                    'id':obj.oid,
                    'is_thing':obj.is_thing,
                    'mask':info['mask'],
                    'category_id':obj.category_id,
                    'dynamic':obj.dynamic
                })
            
            
            objs_img.append({
                'id':idx,
                'is_thing':info['isthing'],
                'mask':info['mask'],
                'category_id':info['category_id'],
                'dynamic':dynamic
            })
            
        
        
        # 特徵點追蹤
        for idx,raw_kp in enumerate(frame.raw_kps):
            uv0 = tuple(map(int, raw_kp))
            uvs = []
            p = frame.points[idx]
            if p is None:
                mode = None
            else:
                p:'Point'
                
                
                # else:
                frames_sorted = sorted(p.frames.keys(),key=lambda f: f.id,reverse=True)# 根據id排序frames 大到小
                
                f_id = frame.id
                for f in frames_sorted: # 從最新的 frame 往舊的 frame 迭代
                    if f_id != f.id: break # 避免不連續的點，保證畫出的是連續的特徵點變化
                    idx2 = f.find_point(p)
                    uvs.append(tuple(map(int, f.raw_kps[idx2])))
                    f_id -= 1
                
                if len(p.frames) < 3: # 如果是新增點
                    mode = 'new'
                elif len(uvs)>1:
                    mode = 'long'
                else:
                    mode = 'proj'
            
            
            kps.append({
                'mode':mode,
                'uv0':uv0,
                'uvs':uvs,
                'track_method':frame.track_method[idx],
                
            })
        # 影像資訊
        infos = [
            f"f.id:{frame.id}",
            f"clear:{frame.is_clear,int(frame.lap_var)}", # 越高畫面越清晰
        ]
        
        
        # 物件預測遮罩、物件投影遮罩
        # obj_mask = np.full((height, width), -1, dtype=np.int32)
        # id_map = frame.obj_prj_id_map
        # depth_map = frame.obj_prj_depth_map
        
        
        # 上傳
        frame_info={
            'fid':frame.id,
            'img':frame.img,
            'objs_map':objs_map,
            'objs_img':objs_img,
            'kps':kps,
            'infos':infos,
            # 'obj_prj_depth':depth_map,
            # 'obj_prj_id_map':id_map,
        }
        self.q_frame.put(frame_info)
    
    def display_info(self, key:str, info:any, timestamp:int=0):
        
        self.q_info.put((key, info, timestamp))
        

def draw_masks(img,masks,ids):
    width, height = img.shape[:2][::-1]

    all_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for id in ids:
        out_mask = np.zeros((height, width, 3), dtype=np.uint8)

        color = int_to_rgb(id,mode256=True)
        out_mask[masks[id]] = color


        all_mask = cv2.bitwise_or(all_mask, out_mask)

    img = cv2.addWeighted(img, 1, all_mask, 0.5, 0)
    
    return img

def draw_mask(image, mask, color, alpha=0.5):
    """
    在 image 上根據 mask 畫半透明遮罩
    image: 原始 BGR 圖片
    mask: 2D numpy array，遮罩區域為 True/1
    color: 遮罩顏色 (B, G, R)
    alpha: 透明度 (0~1)
    """
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_contour_with_mask(image, mask, color=(0, 255, 0), thickness=1, dashed=False, dash_length=5, gap_length=5):
    """
    根據 mask 在 image 上畫出遮罩區域的邊框
    image: 原始 BGR 圖片
    mask: 2D numpy array，遮罩區域為 True/1
    color: 邊框顏色 (B, G, R)
    thickness: 邊框寬度
    dashed: 是否畫虛線
    dash_length: 虛線長度
    gap_length: 虛線間隔
    """
    
    # 防呆：檢查 mask 與 image 型態與維度
    if image is None or mask is None:
        print("[Error] image or mask is None")
        return image
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        print(f"[Error] mask 型態或維度異常: {type(mask)}, ndim={getattr(mask,'ndim',None)}")
        return image
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        print(f"[Error] image 型態或維度異常: {type(image)}, ndim={getattr(image,'ndim',None)}")
        return image
    
    try:
        # 轉為 single-channel uint8 且確保連續記憶體
        if mask.dtype == np.bool_ or mask.dtype == np.bool8:
            mask_uint8 = (mask.astype(np.uint8) * 255)
        elif mask.dtype == np.uint8:
            mask_uint8 = mask.copy()
        else:
            # 如果是 float / int 類型，視非零為 mask
            mask_uint8 = (mask != 0).astype(np.uint8) * 255

        if mask_uint8.size == 0:
            return image

        mask_uint8 = np.ascontiguousarray(mask_uint8)
        kernel = np.ones((3, 3), np.uint8)
        # iterations 不能小於 1，確保 thickness>=1
        iters = max(1, int(abs(thickness)))
        mask_eroded = cv2.erode(mask_uint8, kernel, iterations=iters)  # 內縮（侵蝕）遮罩，避免邊框重疊

        # OpenCV findContours 版本差異處理
        contours_info = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        if not contours:
            return image

        if not dashed:
            cv2.drawContours(image, contours, -1, color, max(1, thickness*2))
        else:
            # 虛線繪製
            for contour in contours:
                if contour is None or contour.size == 0:
                    continue
                pts = contour.reshape(-1, 2)
                for i in range(len(pts)):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % len(pts)])
                    # 計算線段長度
                    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
                    if dist == 0:
                        continue
                    direction = (np.array(pt2) - np.array(pt1)) / dist
                    n_dashes = int(dist // (dash_length + gap_length)) + 1
                    for j in range(n_dashes):
                        start = np.array(pt1) + direction * (j * (dash_length + gap_length))
                        end = start + direction * dash_length
                        if np.linalg.norm(end - np.array(pt1)) > dist:
                            end = np.array(pt2)
                        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), color, max(1, thickness*2))
    except Exception as e:
        # 捕捉任何例外，避免 C 層崩潰造成整個 process segmentation fault
        print(f"[Warning] draw_contour_with_mask failed: {e}")
        return image

    return image
    
    # mask_uint8 = mask.astype(np.uint8) * 255
    # kernel = np.ones((3, 3), np.uint8)
    # mask_eroded = cv2.erode(mask_uint8, kernel, iterations=thickness)  # 內縮（侵蝕）遮罩，避免邊框重疊
    # contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not dashed:
    #     cv2.drawContours(image, contours, -1, color, thickness*2)
    # else:
    #     # 虛線繪製
    #     for contour in contours:
    #         pts = contour.reshape(-1, 2)
    #         for i in range(len(pts)):
    #             pt1 = tuple(pts[i])
    #             pt2 = tuple(pts[(i + 1) % len(pts)])
    #             # 計算線段長度
    #             dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    #             if dist == 0:
    #                 continue
    #             direction = (np.array(pt2) - np.array(pt1)) / dist
    #             n_dashes = int(dist // (dash_length + gap_length)) + 1
    #             for j in range(n_dashes):
    #                 start = np.array(pt1) + direction * (j * (dash_length + gap_length))
    #                 end = start + direction * dash_length
    #                 if np.linalg.norm(end - np.array(pt1)) > dist:
    #                     end = np.array(pt2)
    #                 cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness*2)
    # return image

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

def getjet(i:int):
    if i >9:
        i = 9
    return myjet[i]
    
def paint_info(img, info_text:str|list[str]=None):
    # 在左上角加入資訊
    if info_text is not None:
        # 多行資訊可用 list 傳入
        if isinstance(info_text, str):
            info_text = [info_text]
        for i, line in enumerate(info_text):
            cv2.putText(img, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def objoct_info(img,mask,text,color):
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_scale=0.5
    thickness=1
    
    # --- 取得目標區域的質心與外接框 ---
    mask_uint8 = mask.astype(np.uint8) * 255
    cnts = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    
    cx = int(round(M['m10'] / M['m00']))
    cy = int(round(M['m01'] / M['m00']))
    
    # --- 根據固定 font_scale / thickness 計算置中座標 ---
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(cx - tw / 2)
    y = int(cy + th / 2)  # putText 的 y 是基線
    cv2.rectangle(img, (x, y+baseline), (x + tw, y - th), color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    
    return img
    

def paint_feature(img,frame:'Frame'):
    slam = frame.slam
    
    # 繪製objects
    if slam.control.show_objs: # control
        masks = [info['mask'] for info in frame.obj_infos]
        
        for obj, mask in zip(frame.objects,masks):
            if obj is None:
                color = (127,127,127)
                img = draw_contour_with_mask(img,mask,color,1,True)
            else:
                color = int_to_rgb(obj.oid,mode256=True)
                color = (color[2],color[1],color[0]) # rgb 2 bgr
                img = draw_contour_with_mask(img,mask,color,1)
                
                if slam.control.show_obj_info:
                    obj.updata_category_id()
                    if obj.is_thing:
                        text = slam.thing_class_names[obj.category_id]
                    else:
                        text = slam.stuff_class_names[obj.category_id]
                        
                    img = objoct_info(img,mask,text,color)
    
    # 繪製kps
    if slam.control.show_kps: # control
        
        for idx0, raw_kp in enumerate(frame.raw_kps):
            
            uv0 = tuple(map(int, raw_kp))
            p = frame.points[idx0]
            if p is not None:
                
                p:'Point'
                
                uvs = []
                if len(p.frames) < 3: # 如果是新增點
                    cv2.circle(img, uv0, color=(127,127,127), radius=2) # color (B,G,R)
                else:
                    frames_sorted = sorted(p.frames,key=lambda f: f.id,reverse=True)# 根據id排序frames 大到小
                    
                    
                    f_id = frame.id
                    
                    for f in frames_sorted: # 從最新的 frame 往舊的 frame 迭代
                        if f_id != f.id: break # 避免不連續的點，保證畫出的是連續的特徵點變化
                        idx = f.find_point(p)
                        uvs.append(tuple(map(int, f.raw_kps[idx])))
                        f_id -= 1
                    
                    
                    # for f in reversed(p.frames[-9:]): # 從最新的 frame 往舊的 frame 迭代
                    #     if f_id != f.id: break  # 避免讀取 relocaliztion/投影 的點，保證畫出的是連續的特徵點變化
                    #     idx = f.find_point(p)
                    #     uvs.append(tuple(map(int, f.raw_kps[idx])))
                    #     f_id -= 1
                    
                    if len(uvs)>1:
                        cv2.circle(img, uv0, color=(0,255,0), radius=2)
                        if frame.slam.control.long_track_kps: # kps的全部連續變化
                            cv2.polylines(img,np.array([uvs], dtype=np.int32),False,getjet(len(uvs))*255, thickness=1, lineType=16)
                        else:  # kps的最近2frame變化
                            cv2.line(img,uvs[0],uvs[1],color=(0,255,0), thickness=1, lineType=16)
                        
                    else:
                        cv2.circle(img, uv0, color=(0,0,255), radius=2) # relocaliztion/投影 取得的點 以紅色顯示
                    
            else: # p is None
                cv2.circle(img, uv0, color=(255,255,255), radius=2)
    
    
    infos = [
        f"f.id:{frame.id}",
        f"clear:{frame.is_clear,int(frame.lap_var)}", # 越高畫面越清晰
    ]
    paint_info(img,infos)
    
    return img



### state display 

# 階段名稱與初始座標
NODES_INFO = []
EDGES = []
# process 0
NODES_INFO.append({
    0:("NO_IMAGE",   (100,  50)),
    1:("INIT", (300,  50)),
    # 2:("INITed", (300,  50)),
    2:("WORK", (500,  50)),
    3:("LOST", (700, 50)),
})
EDGES.append({
    (0,1),
    (1,2),
    (2,3),
})
# process 1
NODES_INFO.append({
    "add_f":("add f",             (100, 200)),
    "track":("track",             (300,  200)), # 追蹤位姿
    "retrack":("retrack",       (500,  200)), # 從kf追蹤
    "pose_opt":("pose\nopt",         (700,  200)), # g2o位姿調整 還包含 local map track
    "relocal":("relocal",           (900,  200)), # 重新定位
    "obj_track":("track\nobject",    (1100,  200)), # 追蹤地圖物件
})
EDGES.append([])
# process 2 local mapping part
NODES_INFO.append({
    "cull_pts":("cull_points",       (100,  350)), # 
    "fuse":("fuse",            (500, 350)), # 
    "creat_pts":("creat points",  (300,  350)), # 
    "local_BA":("local\nBA",         (700,  350)), # 
    "gobal_BA":("gobal\nBA",         (900,  350)), # 
    "obj_pre":("object\npredict",   (1100,  350)), # 
})
EDGES.append([])
INFO_BOX = {
    'img_str':('image:', (1200,  25)),
    'fid':('f ID:', (1200,  75)),
    'kfid':('kf ID:', (1200,  125)),
    'f_num':("f num:", (1200,  175)),
    'kf_num':("kf num:", (1200,  225)),
    'pt_num':('map point num:', (1200,275)), # 地圖點總數
    'tracking_point':('tracking point:', (1200,325)), # 正在追蹤的地圖點數量
    'kf_cur_point':('kf_cur point:', (1200,375)), # 正在追蹤的地圖點數量
    'f num btw kf':('f num btw kf:', (1200,425)), # 與上一個kf間隔多少f
    
    'predict_pose_model':("predict model:", (300,  275)),
    'track_num':("track retrack:", (500,  275)),
    'pose_opt':('opt err:', (700,275)), # opt pose
    'relocal':('success/track pts', (900,  275)), # relocalization 是否成功
    'cull_pt':('culled points:', (100,425)),
    'creat_pts_num':("creat num:", (300,  425)),
    'fuse':("to nebr/cur:", (500,  425)),
    'lBA':('BA err:', (700,425)), # lBA
    'gBA':('BA err:', (900,425)), # gBA
    
}

NODE_RADIUS = 50
INIT_W, INIT_H = 1300, 600
BASE_FONT_SIZE = 12
class Control():
    def __init__(self):
        # mp.Value 的形式可以在共享記憶體（shared memory）上分配變數，父子進程都能看見對方的變動。
        
        # 2D image
        self._show_objs_flag = mp.Value('b',True)
        self._show_objs_img_flag = mp.Value('b',False)
        self._show_obj_none_flag  = mp.Value('b',True)
        self._show_obj_thing_flag = mp.Value('b',True)
        self._show_obj_stuff_flag = mp.Value('b',True)
        self._show_obj_info_flag  = mp.Value('b',True)
        self._show_obj_dynamic_flag  = mp.Value('b',False)
        self._show_obj_only_flag  = mp.Value('b',False)
        self._show_obj_id_flag  = mp.Value('i',0)
        
        
        
        self._show_kps_flag = mp.Value('b',True)
        self._show_kps_detected_flag = mp.Value('b',False) # 顯示沒有對應地圖點的kp
        self._long_track_kps_flag = mp.Value('b',True) #long track 顯示kps的所有連續變化/顯示kps的最近2個frame的變化
        self._mothed_color_flag = mp.Value('b',False)
        
        # 3D map
        self._follow_flag = mp.Value('b',False)
        self._groundtruth_flag = mp.Value('b',True) # groundtruth 顯示 groundtruth
        
        # system
        
        self._shoot_flag = mp.Value('b',False)
        self._stop_flag = mp.Value('b',False)
        self.next_step_flag = mp.Value('i',0)# button 需要計算次數
        self.quit_flag = mp.Value('i',0)# button 需要計算次數
        
        
        
        self.commands = {
            'show_objs':self._show_objs_flag,
            'show_objs_img':self._show_objs_img_flag,
            'show_obj_none':self._show_obj_none_flag,
            'show_obj_thing':self._show_obj_thing_flag,
            'show_obj_stuff':self._show_obj_stuff_flag,
            'show_obj_info':self._show_obj_info_flag,
            'show_obj_dynamic':self._show_obj_dynamic_flag,
            'show_obj_only':self._show_obj_only_flag,
            'show_obj_id':self._show_obj_id_flag,
            
            
            
            'show_kps':self._show_kps_flag,
            'kps_detected':self._show_kps_detected_flag,
            'long_track_kps':self._long_track_kps_flag,
            'mothed_color':self._mothed_color_flag,
            
            
            
            'follow':self._follow_flag,
            'groundtruth':self._groundtruth_flag,
            
            
            'shoot':self._shoot_flag,
            'stop':self._stop_flag,
            'next_step':self.next_step_flag,
            'quit':self.quit_flag,
            
            }
    
    @property
    def follow(self)->bool:
        return self._follow_flag.value
    @property
    def groundtruth(self)->bool:
        return self._groundtruth_flag.value
    
    @property
    def shoot(self)->bool:
        return self._shoot_flag.value
    
    @property
    def stop(self)->bool:
        return self._stop_flag.value
    
    @property
    def show_objs(self)->bool:
        return self._show_objs_flag.value
    @property
    def show_objs_img(self)->bool:
        return self._show_objs_img_flag.value
    @property
    def show_obj_none(self)->bool:
        return self._show_obj_none_flag.value
    @property
    def show_obj_thing(self)->bool:
        return self._show_obj_thing_flag.value
    @property
    def show_obj_stuff(self)->bool:
        return self._show_obj_stuff_flag.value
    @property
    def show_obj_info(self)->bool:
        return self._show_obj_info_flag.value
    @property
    def show_obj_dynamic(self)->bool:
        return self._show_obj_dynamic_flag.value
    @property
    def show_obj_id(self)->int|None:
        if self._show_obj_only_flag.value:
            return self._show_obj_id_flag.value
        else:
            return None
    
    @property
    def show_kps(self)->bool:
        return self._show_kps_flag.value
    @property
    def show_kps_detected(self)->bool:
        return self._show_kps_detected_flag.value    
    @property
    def long_track_kps(self)->bool:
        return self._long_track_kps_flag.value
    @property
    def mothed_color(self)->bool:
        return self._mothed_color_flag.value
    
    @property
    def next_step(self)->bool:
        if self.next_step_flag.value > 0:
            self.next_step_flag.value -= 1
            return True
        else: return False
    @property
    def quit(self)->bool:
        if self.quit_flag.value > 0:
            self.quit_flag.value -= 1
            return True
        else: return False
        
    
    
    def get_state(self,*keys):
        """
        - 若未給 keys，回傳整個 dict。
        - 若只給一個 key，回傳單一值。
        - 若給多個 key，回傳 tuple。
        """
        state = {
            'stop':self.stop,
            }
        return self.stop
        if not keys:
            return state
        if len(keys) == 1:
            return state[keys[0]]
        return tuple(state[k] for k in keys)
    
            

class Node():
    def __init__(self,container,font:tkfont.Font,content:str,state) -> None:
        
        self.container = container
        self.font = font
        self.content = content
        self.state = state
    def get(self):
        return(self.container,self.font,self.content,self.state)

class StateDisplay():
    def __init__(self,slam):
        
        
        self.slam = slam
        
        self.root = tk.Tk()
        self.root.title("系統流程圖")
        self.control = Control()
        commands = self.control.commands
        
        self.q_state_change = Queue(50)
        self.q_info = Queue(50)
        
        
        self.display_process = Process(target=self._viewer_thread, args=(self.q_state_change,self.q_info,commands),daemon = False)  # daemon true means, exit when main program stops
        self.display_process.start()
        
    def _init_display(self): # 這裡建立的屬性 不會/不應 與外部接觸 
        self._init_W = INIT_W
        self._init_H = INIT_H 
        # 紀錄整體縮放比例 (相對於 INIT_W/H)
        self._scale_x = 1.0
        self._scale_y = 1.0
        
        # 按鈕字型 (同時用於所有底部按鈕)
        self.btn_font = tkfont.Font(family="Arial", size=BASE_FONT_SIZE)
        self.title_font = tkfont.Font(family="Arial", size=BASE_FONT_SIZE + 2)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # 初始化：按鈕
        self._objs = tk.BooleanVar(value=True)
        self._objs_img = tk.BooleanVar(value=False)
        self._obj_none = tk.BooleanVar(value=True)
        self._obj_thing = tk.BooleanVar(value=True)
        self._obj_stuff = tk.BooleanVar(value=True)
        self._obj_info = tk.BooleanVar(value=True)
        self._obj_dynamic = tk.BooleanVar(value=False)
        self._obj_only = tk.BooleanVar(value=False)
        self._obj_id = tk.StringVar(value="0")  # 輸入框 預設值為 0
        
        
        self._kps = tk.BooleanVar(value=True)
        self._kps_detect = tk.BooleanVar(value=False)
        self._lt_kps = tk.BooleanVar(value=True)
        self._mothed_kps = tk.BooleanVar(value=False)
        
        
        self._follow = tk.BooleanVar(value=False)
        self._groundtruth = tk.BooleanVar(value=True)
        
        
        self._shoot = tk.BooleanVar(value=False) # 初始未勾
        self._stop = tk.BooleanVar(value=False) # 初始未勾
        
        
        
        # 2D image
        # tk.Label(btn_frame,text="image viewer",     font=self.btn_font,anchor="w").pack(side=tk.TOP,)
        img_frame = tk.LabelFrame(btn_frame, text="image viewer", font=self.btn_font, padx=10, pady=10)
        img_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(img_frame,text="Objects",     font=self.btn_font,anchor="w").pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='map mode',  font=self.btn_font,command=self._update_control, variable=self._objs,       onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='img mode',  font=self.btn_font,command=self._update_control, variable=self._objs_img,   onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='none',      font=self.btn_font,command=self._update_control, variable=self._obj_none,   onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='thing',     font=self.btn_font,command=self._update_control, variable=self._obj_thing,  onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='stuff',     font=self.btn_font,command=self._update_control, variable=self._obj_stuff,  onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='info',      font=self.btn_font,command=self._update_control, variable=self._obj_info,   onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='dynamic',   font=self.btn_font,command=self._update_control, variable=self._obj_dynamic,onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='only show', font=self.btn_font,command=self._update_control, variable=self._obj_only,   onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Entry(img_frame, textvariable=self._obj_id, font=self.btn_font, width=10).pack(side=tk.TOP,)
        tk.Button(img_frame, text="flash", font=self.btn_font, command=self._update_control, anchor='e',width=5).pack(side=tk.TOP, anchor='e')
        
        tk.Label(img_frame,text="KPS",     font=self.btn_font,anchor="w").pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='kps',           font=self.btn_font,command=self._update_control, variable=self._kps,        onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='detected',      font=self.btn_font,command=self._update_control, variable=self._kps_detect, onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='LT kp',         font=self.btn_font,command=self._update_control, variable=self._lt_kps,     onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        tk.Checkbutton(img_frame, text='methed color',  font=self.btn_font,command=self._update_control, variable=self._mothed_kps, onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)

        # # 3D map
        # tk.Label(btn_frame,text="map viewer",     font=self.btn_font,anchor="w").pack(side=tk.TOP,)
        # tk.Checkbutton(btn_frame, text='follow',        font=self.btn_font,command=self._update_control, variable=self._follow,     onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        # tk.Checkbutton(btn_frame, text='groundtruth',   font=self.btn_font,command=self._update_control, variable=self._groundtruth,onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP,)
        
        # system
        sys_frame = tk.LabelFrame(btn_frame, text="system", font=self.btn_font, padx=10, pady=10)
        sys_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        # tk.Label(sys_frame,text="system",     font=self.btn_font,anchor="w").pack(side=tk.TOP,)
        
        tk.Checkbutton(sys_frame, text='shoot',          font=self.btn_font,command=self._update_control, variable=self._shoot,       onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP)

        tk.Checkbutton(sys_frame, text='Stop',          font=self.btn_font,command=self._update_control, variable=self._stop,       onvalue=True,offvalue=False,    anchor='w',width=10).pack(side=tk.TOP)
        tk.Button(sys_frame, text="next step",          font=self.btn_font, command=self.next_step).pack(side=tk.TOP)
        tk.Button(sys_frame, text="quit",          font=self.btn_font, command=self.quit).pack(side=tk.TOP)
        
        # 可縮放 Canvas
        self.canvas = tk.Canvas(self.root, width=INIT_W, height=INIT_H, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self._on_resize)
        
        # 初始化：節點、文字、連線
        self.processes:list[dict[int,Node]] = []
        for i,info in enumerate(NODES_INFO):
            self._creat_edges(EDGES[i],info)
            nodes = self._creat_process_nodes(info)
            self.processes.append(nodes)
        self._update_state()
        
        
        # 初始化：infoboxs
        self.infoboxs:dict[int,Node]={}
        for i,info in INFO_BOX.items():
            infobox = self._creat_infobox(info)
            self.infoboxs[i]=infobox
        # 設定所有下拉選單的字體
        self.root.option_add('*TCombobox*Listbox.font',self.btn_font)
    
    def change_state(self,q_state_change:Queue): # 讀取外部命令，修改狀態顯示
        
        while not q_state_change.empty():
            (method,args) = q_state_change.get()
            
            match method:
                case 0:
                    (idx_process,idx_node) = args
                    group:dict[int,Node] = self.processes[idx_process]
                    for i,node in group.items():
                        if idx_node == i:
                            node.state = 1
                        else:
                            node.state = 0
                        
                case 1:
                    (idx_process,idx_node,state) = args
                    self.processes[idx_process][idx_node].state = state
                    
            self._update_state()
            
        self.root.after(30,self.change_state,q_state_change)
    
    def show_info(self,q_info:Queue): # 顯示外部傳入的狀態
        while not q_info.empty():
            (idx,value) = q_info.get()
            infobox = self.infoboxs[idx]
            # 提取舊資訊並加入下拉選單
            vals = list(infobox.content['values'])
            vals.insert(0,value)
            infobox.content['values'] = vals
            # 顯示新資訊
            infobox.state.set(value)
            
            
        self.root.after(30,self.show_info,q_info)
    
    def _viewer_thread(self,q_state_change,q_info,commands:dict[str, Synchronized]):
        self.commands = commands
        
        # init
        self._init_display()
        
        
        # loop
        self.change_state(q_state_change)
        self.show_info(q_info)
        self.root.mainloop()
    
    def _draw_node(self,label,x,y):
        h,w =NODE_RADIUS,NODE_RADIUS
        x1 = x-h
        y1 = y-w
        x2 = x+h
        y2 = y+w
        box = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill="lightgray", outline="black", width=2)
        font = tkfont.Font(family="Arial", size=BASE_FONT_SIZE) # 字體
        text = self.canvas.create_text(x, y, text=label, font=font) # 文字
        node = Node(box,font,text,0)
        return node
    
    def _creat_process_nodes(self,nodes_info):
        nodes = {}
        
        for idx,(label, (x0, y0)) in nodes_info.items():
            nodes[idx] = self._draw_node(label,x0,y0)
            
        return nodes

    def _creat_edges(self,edges,nodes_info):
        for (idx0,idx1) in edges:
            (x0, y0) = nodes_info[idx0][1]
            (x1, y1) = nodes_info[idx1][1]
            self.canvas.create_line(x0, y0, x1, y1, arrow=tk.LAST, width=2)
    def _update_control(self):
        
        self.commands['show_objs'].value = self._objs.get()
        self.commands['show_objs_img'].value = self._objs_img.get()
        self.commands['show_obj_none'].value = self._obj_none.get()
        self.commands['show_obj_thing'].value = self._obj_thing.get()
        self.commands['show_obj_stuff'].value = self._obj_stuff.get()
        self.commands['show_obj_info'].value = self._obj_info.get()
        self.commands['show_obj_dynamic'].value = self._obj_dynamic.get()
        self.commands['show_obj_only'].value = self._obj_only.get()
        self.commands['show_obj_id'].value = int(self._obj_id.get())
        
        self.commands['show_kps'].value = self._kps.get()
        self.commands['kps_detected'].value = self._kps_detect.get()
        self.commands['long_track_kps'].value = self._lt_kps.get()
        self.commands['mothed_color'].value = self._mothed_kps.get()
        
    
        self.commands['follow'].value = self._follow.get()
        self.commands['groundtruth'].value = self._groundtruth.get()
        
        self.commands['shoot'].value = self._shoot.get()
        self.commands['stop'].value = self._stop.get()
        
    def _update_state(self):
        
        for process in self.processes:
            for node in process.values():
                (rect,font,text,state) = node.get()
                match state:
                    case 0:_color = "lightgray"
                    case 1:_color = "lightblue"
                
                self.canvas.itemconfig(rect, fill=_color) 
    
    
    def _creat_infobox(self,info):
        offset = int(NODE_RADIUS/4)
        text ,(x,y) = info
        
        selected = tk.StringVar(value='')
        font = tkfont.Font(family="Arial", size=BASE_FONT_SIZE) # 字體
        combobox = ttk.Combobox(
            self.root,
            textvariable=selected,
            state="normal", #"normal"、"readonly"或"disabled"之一。在 「readonly」狀態，該值不能直接編輯， 使用者只能從中選擇值 下拉清單。在「normal」狀態下，文字字段 可直接編輯。在「disabled」狀態下，不可以進行互動。 
            font=font,
            )
        infobox = self.canvas.create_window(x,y+offset,window=combobox,height=int(NODE_RADIUS/2),width=int(NODE_RADIUS*2))
        
        self.canvas.create_text(x, y-offset, text=text, font=font) # 說明文字
        
        return Node(infobox,font,combobox,selected)

    def _on_resize(self, event):
        # 計算新的縮放比例（以 INIT_W/INIT_H 為基準）
        new_sx = event.width  / self._init_W
        new_sy = event.height / self._init_H
        # 要對 Canvas 做的相對縮放比例
        rel_x = new_sx / self._scale_x
        rel_y = new_sy / self._scale_y

        # 縮放所有 Canvas 物件
        self.canvas.scale("all", 0, 0, rel_x, rel_y)

        # 更新文字字型大小（節點 + 按鈕）
        # 採用 (sx+sy)/2 的平均值，保持比例感
        avg_scale = (new_sx + new_sy) / 2
        new_font_size = max(int(BASE_FONT_SIZE * avg_scale), 1)
        
        # 流程節點
        for group in self.processes:
            for node in group.values():
                node.font.configure(size=new_font_size)

        # infobox
        for infobox in self.infoboxs.values():
            infobox.font.configure(size=new_font_size)
            self.canvas.itemconfigure(infobox.container,height=int(NODE_RADIUS/2*new_sy),width=int(NODE_RADIUS*2*new_sx))
            
        
        # 底部按鈕
        self.btn_font.configure(size=new_font_size)

        # 儲存最新比例
        self._scale_x = new_sx
        self._scale_y = new_sy
    
    def next_step(self):
        self.commands['next_step'].value += 1
    def quit(self):
        self.commands['quit'].value += 1
    
    def set_process_state(self,idx_process:int,idx_node:str):
        """

        Args:
            idx_process (int): ...
            idx_node (int): ...

        Effects:
            將一個process的其中一個node設定為1其他node為0
        """


        self.q_state_change.put((0,(idx_process,idx_node)))
    
    def set_node_state(self,idx_process:int,idx_node:str,state):
        '''
        Args:
            idx_process (int): ...
            idx_node (int): ...
            state (int): ...

        Effects:
            將一個process的其中一個node設定為特定值
        '''

        self.q_state_change.put((1,(idx_process,idx_node,state)))
        
    def set_info(self,idx_label:int,info):
        self.q_info.put((idx_label,info))
        