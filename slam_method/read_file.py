import numpy as np
from scipy.spatial.transform import Rotation
import os
import cv2




def read_file_list(filename):
    """
    從文本文件讀取軌跡。

    文件格式:
    文件格式為 "stamp d1 d2 d3 ...", 其中 stamp 表示時間戳（用於匹配），
    "d1 d2 d3.." 是與該時間戳關聯的任意數據（例如 3D 位置和 3D 姿態）。

    輸入:
    filename -- 文件名

    輸出:
    dict -- (stamp, data) 元組的字典

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset=0.0, max_difference=0.0):
    """
    將兩個 (stamp, data) 字典進行關聯。由於時間戳通常不完全匹配，
    我們嘗試為每個輸入元組找到最接近的匹配。

    輸入:
    first_list -- 第一個 (stamp, data) 元組字典
    second_list -- 第二個 (stamp, data) 元組字典
    offset -- 兩個字典之間的時間偏移（例如用於建模感測器之間的延遲）
    max_difference -- 候選生成的搜尋半徑

    輸出:
    matches -- 匹配元組列表 ((stamp1, data1), (stamp2, data2))

    """
    first_keys = list(first_list)
    second_keys = list(second_list)
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return dict(matches)

def convert_groundtruth_to_pose_list(input_file,keys=None):
    """將 groundtruth.txt 轉換為 pangolin.DrawCameras 可讀取的格式並儲存為 numpy.ndarray"""
    with open(input_file, 'r') as f:
        lines = f.readlines()

    poses = []
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        data = list(map(float, line.strip().split()))
        timestamp, tx, ty, tz, qx, qy, qz, qw = data
        if keys is not None:
            if timestamp not in keys:continue
        # 轉換四元數為旋轉矩陣
        R =Rotation.from_quat((qx, qy, qz, qw)).as_matrix() # Rwc
        # 構建 4x4 齊次變換矩陣
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz] # twc
        poses.append(T)  # 將矩陣加入 poses
    
    poses = np.array(poses).reshape(-1, 4, 4)  # 將列表轉換為 numpy.ndarray
    # Pangolin +X：向右 -Y：向上 +Z：向前（相機看向的方向）
   
   
    # 修改座標朝向！！TODO:驗證程式
    # 軸變換：將 TUM-RGBD () 轉為 Pangolin ()
    swap = np.array([
        [-1, 0, 0, 0],
        [ 0,-1, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]
    ])
    poses = swap @ poses @ np.linalg.inv(swap)

    Rz180 = np.array([
    [-1, 0, 0],
    [ 0,-1, 0],
    [ 0, 0, 1]])
    
    poses[:,:3,:3] = poses[:,:3,:3] @ Rz180
    
    poses[:,:3, 3] = poses[:,:3, 3]*30 # 調整尺度，用於將groundtruth縮放到與預測地圖接近的大小
    
    return poses #Twc



class Reader():
    def __init__(self):
        self.groundtruth = None
        self.data_type = None
        
        
        self.cap = None
        
        self.folder_path:str = None
        self.file_list:str = None
        self.file_index = 0
        self.file_name:str = None
        
        
    
    def TUM(self, folder_path:str):
        self.data_type = 'TUM'
        
        # 獲取資料夾中所有檔案的列表，過濾出所有的 PNG 檔案
        file_list = os.listdir(folder_path+'/rgb')
        file_list = [file for file in file_list if file.endswith('.png')]
        # 按檔案名稱排序
        file_list.sort()
        
        self.folder_path = folder_path
        self.file_list = file_list
        
        # 讀取地面真值(groundtruth)
        groundtruth_list = read_file_list(folder_path+'/groundtruth.txt')
        rgb_list = read_file_list(folder_path+'/rgb.txt')
        image_matches = associate(groundtruth_list,rgb_list, max_difference=0.02)
        self.groundtruth = convert_groundtruth_to_pose_list(folder_path+'/groundtruth.txt',keys=image_matches)
        
    
    def get_TUM(self):
        
        # for file_name in file_list:
        if self.file_index < len(self.file_list):
            
            self.file_name = self.file_list[self.file_index]
            file_path = os.path.join(self.folder_path+'/rgb', self.file_name)
            img = cv2.imread(file_path)
            
            self.file_index += 1
            return img
        else:
            return None
        
        
            
    def video(self, video_path:str):
        self.data_type = 'video'
        self.cap = cv2.VideoCapture(video_path)
    
    def get_video(self):
        # while self.cap.isOpened():
        ret, img = self.cap.read()
        if ret == True:
            return img
        else:
            
            self.cap.release()
            return None
            
        
    def get_img(self):
        
        match self.data_type:
            case 'TUM':
                return self.get_TUM()
            
            case 'video':
                return self.get_video()
        
        