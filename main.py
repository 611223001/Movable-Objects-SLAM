import os

# 鎖定 BLAS/OMP 執行緒，避免多執行緒競爭導致 segfault
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




# import glob
import cv2
import numpy as np
import time

cv2.setNumThreads(1)

# from display import Display
# from slam_method.utils import denormalize, add_ones
import faulthandler
faulthandler.enable()

import multiprocessing as mp
# mp.set_start_method('spawn', force=True)

from slam_method.frame import Frame
from slam_method.map import Map
from slam_method.point import Point
from slam_method.camera import Camera
from slam_method.slam import Slam
from slam_method.display import MapDisplay
from slam_method.read_file import Reader,convert_groundtruth_to_pose_list, read_file_list, associate


from config import CameraConfig

if __name__== "__main__":

    reader = Reader()
    
    
    # folder_path = "../dataset/TUM/rgbd_dataset_freiburg3_long_office_household"
    # reader.TUM(folder_path)
    # 484 lost
    
    video_path = "../dataset/mydata/D435i/move_object2.mp4"
    reader.video(video_path)
    
    mapp = Map(reader.groundtruth)
    camera = Camera(config=CameraConfig.D435i)
    
    slam =Slam(camera,mapp)
    map_display = slam.map_display
    control = slam.control
    
    
    while True:
        
        img = reader.get_img()
        if img is not None:
        
            slam.state_display.set_info('img_str',reader.file_name)
            # print('image:',file_name)
            
            img = cv2.resize(img, (camera.width, camera.height))
            
            out = slam.track(img=img)
            
            # img = out.get('img',img)
            map_display.display_map()
            map_display.display_img()
            
            
        else:
            break
        
        # cv2.imshow('current frame',img)
        # stop system
        while control.stop and not control.next_step:
            time.sleep(0.005)
            # cv2.waitKey(1)
            ...
        
        # if cv2.waitKey(1) & ((0xFF == ord('q'))|control.quit):
        if control.quit:
            break
    
    print('finish')
    
    while True:
        time.sleep(0.005)
        if control.quit:
            break
    
    
    # close any OpenCV windows
    
    # 有時沒辦法一次關閉進程
    while map_display.display_process.is_alive():
        map_display.run = False
        # map_display.display_process.terminate() 
        map_display.display_process.join()
        # map_display.display_process.close()
        time.sleep(0.05)
    
    cv2.destroyAllWindows()
    
        
    
    print('end')