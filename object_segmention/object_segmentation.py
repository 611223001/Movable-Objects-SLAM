# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from IPython import display


import detectron2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# from sam2.build_sam import build_sam2_camera_predictor


from timeit import time

class ObjectSegmentation():
    
    def __init__(self
                 ,sam2_reset_mode='no_sam2'
                 ,sam2_input_mode='boxes'
                 ,object_upper_limit:int=None
                 ,segment_background:bool=True
                 ,config_file:str = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
                 ,checkpoint_url:str="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
                 ,detectron2_threshold:float=None
                 ):
        '''
        sam2_reset_mode = 'keyframe' / 'object' / 'no_sam2'
        
        sam2_input_mode = 'masks' or 'boxes'
        '''
        self.sam2_input_mode = sam2_input_mode
        self.sam2_reset_mode = sam2_reset_mode
        self.object_upper_limit = object_upper_limit
        self.segment_background = segment_background
        self.current_masks:np.ndarray = None
        self.current_boxes:np.ndarray = None
        self.cur_seg_infos:list = []
        
        
        self.sam2_predictor_list=[None]
    
    
        self.detectron2_cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.detectron2_cfg.merge_from_file(model_zoo.get_config_file(config_file))
        # if detectron2_threshold is not None:
        #     self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detectron2_threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
        self.detectron2_predictor = DefaultPredictor(self.detectron2_cfg)
        
        
        
        self.sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
        self.sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        
        # self.sam2_img_predictor = build_sam2_camera_predictor(self.sam2_model_cfg, self.sam2_checkpoint)
        # 取得
        data = MetadataCatalog.get(self.detectron2_cfg.DATASETS.TRAIN[0])
        self.thing_class_names = data.get("thing_classes", None)  # thing/stuff 分類
        self.stuff_class_names = data.get("stuff_classes", None)  # thing/stuff 分類
        
        
        if (self.sam2_reset_mode != 'keyframe') & (self.sam2_reset_mode != 'object') & (self.sam2_reset_mode != 'no_sam2'):
            print("reset_mode need be 'keyframe' or 'object' or 'no_sam2' !")

        if (self.sam2_input_mode != 'masks') & (self.sam2_input_mode!= 'boxes'):
            print("sam2_input_mode need be 'masks' or 'boxes' !")
        
    def _masks_logits_to_masks(self,
        mask_logits:torch.Tensor,
        threshold:float=0.0
        ) -> np.ndarray:
        '''
        mask_logits     <class 'torch.Tensor'>  size(n,1,h,w)[float]
        threshold           float
        
        return :  
        masks               numpy.ndarray           size(n,h,w)[bool]
        '''
        
        masks= ( mask_logits > threshold ).cpu().numpy()
        h, w = masks.shape[-2:]
        masks=masks.reshape(-1,h,w)
        
        return  masks
    

    # 錯誤的程式！！！！！
    # def get_class(self,class_id)->list[str]|str:
        '''
        class_id            (list[int] or None or int) 
        
        return:
        classes             (list[str] or None or str)
        
        暫時沒有使用
        '''
        classes_str = None
        if type(class_id) == list  :
            classes_str = [self.class_names[i] for i in class_id]
        elif type(class_id) == int :
            classes_str = self.class_names[class_id]
        
        return classes_str
    
    
    # def sam2_smooth_masks(self,frame,masks):
    #     # disuseful
    #     # 沒有效果
    #     self.sam2_img_predictor.load_first_frame(frame)
        
    #     i=0
    #     for mask in masks:
    #         _, out_obj_ids, out_mask_logits = self.sam2_img_predictor.add_new_mask(
    #             frame_idx=0, obj_id=i, mask=mask
    #         )        
    #         i+=1
    #     masks = self._masks_logits_to_masks(out_mask_logits)
    #     return masks
        
    def _detectron2_get(self,frame):
        '''
        使用segment_background時從"panoptic_seg"提取masks,否則從"instances"提取masks
        Inputs:
            frame
        Returns:
            out_obj_ids
            boxes
            masks
        '''
        outputs = self.detectron2_predictor(frame)
        
        if self.segment_background:
            try:
                sem_seg = outputs['sem_seg'].to("cpu").numpy()
                instances = outputs['instances'].to("cpu")
                panoptic_segs, panoptic_infos = outputs["panoptic_seg"]
            except:
                print("使用模型沒有全景分割功能")
                print("請設置 segment_background=False")
                
            else:
                mask_id=np.asarray(panoptic_segs.to("cpu"))
                masks=[]
                class_id = []
                panoptic_segs = panoptic_segs.to("cpu").numpy()
                
                infos=[]
                # 添加無法辨識區域的info
                none_info={
                    'id': 0,
                    'isthing': False,
                    'category_id': None,
                    }
                infos.append(none_info)
                
                # 添加可辨識區域的info
                infos.extend(panoptic_infos)

                boxs = instances.pred_boxes.tensor.numpy()
                
                # 添加所有區域的mask,像素面積
                for info in infos:
                    info['area'] = int((panoptic_segs == info['id']).sum())
                    info['mask'] = (mask_id==info['id'])
                    masks.append(info['mask'])
                    
                    if info['isthing']:
                        info['box'] = boxs[info['instance_id']]
                
                    
                
                self.cur_seg_infos = infos
                self.cur_info = {'mask_id':mask_id,'masks':masks,'sem_seg':sem_seg}
                
                # class_id.append(None)
                
                # for info in panoptic_infos:
                #     class_id.append(info['category_id'])
                
                masks = np.asarray(masks)
                # class_id = np.asarray(class_id)
                
        else:
            print('不能使用')
            # masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
            # class_id = np.asarray(outputs["instances"].to("cpu").pred_classes)

        
        # boxes = np.asarray(outputs["instances"].to("cpu").pred_boxes).reshape(-1,2,2)
        
        # self.current_info['class_id']=class_id
        # self.current_info['masks']=masks
        
        out_obj_ids = list(range(len(masks)))
        
        return out_obj_ids,masks
    

        
    def keyframe_segment(self,frame):
        if self.sam2_reset_mode == 'keyframe':
            
            # outputs = self.detectron2_predictor(frame)
            

            
            # self.sam2_predictor_list[0]=build_sam2_camera_predictor(self.sam2_model_cfg, self.sam2_checkpoint)
            # self.sam2_predictor_list[0].load_first_frame(frame)
            # ann_frame_idx = 0  # the frame index we interact with
            
            # out_obj_ids,boxes,masks = self._detectron2_get(frame)
            # if self.sam2_input_mode =='boxes' :
            #     #boxes = np.asarray(outputs["instances"].to("cpu").pred_boxes).reshape(-1,2,2)
                
            #     for obj_id,box in zip(out_obj_ids,boxes):
            #         _, out_obj_ids, out_mask_logits = self.sam2_predictor_list[0].add_new_prompt(
            #             frame_idx=ann_frame_idx, obj_id=obj_id, bbox=box
            #             )
                    
            # if self.sam2_input_mode =='masks' :
            #     #masks = np.asarray(outputs["instances"].to("cpu").pred_masks)                    
            #     for obj_id,mask in zip(out_obj_ids,masks):

            #         _, out_obj_ids, out_mask_logits = self.sam2_predictor_list[0].add_new_mask(
            #             frame_idx=ann_frame_idx, obj_id=obj_id, masks=mask
            #             )    
                    
            
            
            
            # masks = self._masks_logits_to_masks(mask_logits=out_mask_logits)
            return None


        elif self.sam2_reset_mode=='object':
            outputs = self.detectron2_predictor(frame)
            
            print('todo')
            
        elif self.sam2_reset_mode=='no_sam2':
            #當不使用sam2時keyframe與一般frame執行一樣的程式
            #masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
            out_obj_ids,boxes,masks = self._detectron2_get(frame)
            

        
        self.current_masks=masks
        return out_obj_ids, masks
            
            
    def frame_segment(self,frame):
        '''
        return:
        out_obj_ids     size(n)     list[int]
        masks       numpy.ndarray size(n,h,w)[bool]
        '''
        
        if self.sam2_reset_mode == 'keyframe':
            out_obj_ids, out_mask_logits = self.sam2_predictor_list[0].track(frame)
            
            masks = self._masks_logits_to_masks(out_mask_logits,0)
            
            

        elif self.sam2_reset_mode=='object':
            print('todo')
            
        elif self.sam2_reset_mode=='no_sam2':
            #當不使用sam2時keyframe與一般frame執行一樣的程式
            out_obj_ids,masks = self._detectron2_get(frame)

        self.current_masks=masks
        return out_obj_ids, masks
            

    



import hashlib

def int_to_rgb(number: int) -> tuple:
    """
    將整數轉換為 RGB 顏色
    :param number: 輸入的整數
    :return: (R, G, B) 元組，每個值介於 0-255 之間
    """
    # 將整數轉換為字串，並計算 SHA-256 哈希值
    hash_value = hashlib.sha256(str(number).encode()).hexdigest()
    
    # 取前 6 個十六進制字符，轉換為 R, G, B
    r = int(hash_value[0:2], 16)
    g = int(hash_value[2:4], 16)
    b = int(hash_value[4:6], 16)
    
    return (r, g, b)

def draw_mask(frame,masks,object_segmenter):
    width, height = frame.shape[:2][::-1]
    if masks is not None:
        # print(all_mask.shape)
        all_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, len(masks)):
            out_mask = np.zeros((height, width, 3), dtype=np.uint8)

            color = int_to_rgb(object_segmenter.current_info['class_id'][i])
            out_mask[masks[i]] = color


            all_mask = cv2.bitwise_or(all_mask, out_mask)

        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        
    return frame


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    
    
# if __name__ == '__main__' :
    
    # cap = cv2.VideoCapture("../../dataset/other_data/video/01_dog.mp4")
    # ret, frame = cap.read()
    
    # object_segmenter=Object_Segmention(sam2_reset_mode='no_sam2',segment_background=True)
    
    # _ , _ = object_segmenter.keyframe_segment(frame)
    
    # ann_frame_idx = 0
    # fps = 0.0
    # while True:
    #     t1 = time.time()

        
    #     ret, frame = cap.read()
    #     if not ret:
    #         cv2.destroyAllWindows()
    #         break
    #     ann_frame_idx += 1
        
    #     out_obj_ids, masks = object_segmenter.frame_segment(frame)

        
    #     frame = draw_mask(frame,masks,object_segmenter)

        
    #     cv2.putText(frame,str(ann_frame_idx), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),0) #顯示當前幀
    #     cv2.putText(frame,'fps:'+str(fps), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),0) #顯示當前幀
        
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         cv2.destroyAllWindows()
    #         break
        
    #     fps = 1 / (time.time() - t1)
        

    # cap.release()

if __name__ == '__main__' :
# 定義資料夾路徑
    folder_path = "../../dataset/TUM/rgbd_dataset_freiburg2_large_with_loop/rgb"
    # 獲取資料夾中所有檔案的列表，過濾出所有的 PNG 檔案
    file_list = os.listdir(folder_path)
    file_list = [file for file in file_list if file.endswith('.png')]
    # 按檔案名稱排序
    file_list.sort()

    object_segmenter=ObjectSegmentation(sam2_reset_mode='no_sam2',segment_background=True)
    
    frame = cv2.imread(os.path.join(folder_path,file_list[0]))

    _ , _ = object_segmenter.keyframe_segment(frame)

    ann_frame_idx = 0
    fps = 0.0

    # 讀取並顯示每張圖片
    for file_name in file_list:
        t1 = time.time()

        
        file_path = os.path.join(folder_path, file_name)
        frame = cv2.imread(file_path)
        
        ann_frame_idx += 1
        
        out_obj_ids, masks = object_segmenter.frame_segment(frame)

        
        frame = draw_mask(frame,masks,object_segmenter)

        
        cv2.putText(frame,str(ann_frame_idx), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),0) #顯示當前幀
        cv2.putText(frame,'fps:'+str(fps), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),0) #顯示當前幀



        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break  # 等待按鍵以移動到下一張圖片
        fps = 1 / (time.time() - t1)

    cv2.destroyAllWindows()