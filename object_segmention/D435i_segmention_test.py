import pyrealsense2 as rs
import cv2
import numpy as np
import torch
import time

# Detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def main():
    # 使用 bfloat16 加速
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 配置 Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    # 配置 RealSense 管線
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"無法開啟 RealSense 相機: {e}")
        return
    
    print("開始即時分割... 按下 'q' 鍵停止。")
    
    frame_idx = 0
    fps = 0.0
    
    try:
        while True:
            t1 = time.time()
            
            # 獲取影像幀
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # 轉換為 NumPy 陣列
            frame = np.asanyarray(color_frame.get_data())
            original_frame = frame.copy()
            
            frame_idx += 1
            
            # 使用 Detectron2 進行物件偵測與分割
            outputs = predictor(frame)
            
            # 使用 Visualizer 繪製分割結果
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            
            # # 語意分割
            # sem_seg = outputs['sem_seg'].to("cpu")
            # sem_seg = sem_seg.argmax(0).numpy().astype("int32")  # 取得每個像素的類別 index
            # out = v.draw_sem_seg(sem_seg)
            # segmented_frame = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            
            # # 實例分割
            instances = outputs["instances"].to("cpu")
            out = v.draw_instance_predictions(instances)
            segmented_frame = out.get_image()[:, :, ::-1]
            
            # # 全景分割
            # panoptic_seg,segments_info = outputs['panoptic_seg']
            # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            # segmented_frame = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            
            # 確保記憶體連續性
            segmented_frame = np.ascontiguousarray(segmented_frame)
            
            # 顯示 FPS 和幀數
            cv2.putText(segmented_frame, f'Frame: {frame_idx}', (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(segmented_frame, f'FPS: {fps:.2f}', (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(segmented_frame, f'Objects: {len(instances)}', (20, 90), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 顯示原始影像和分割結果
            cv2.imshow('Original', original_frame)
            cv2.imshow('Segmentation', segmented_frame)
            
            # 按下 'q' 鍵停止
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("停止分割。")
                break
            
            # 計算 FPS
            fps = 1.0 / (time.time() - t1)
    
    except KeyboardInterrupt:
        print("\n程式被中斷。")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程式結束。")

if __name__ == '__main__':
    main()