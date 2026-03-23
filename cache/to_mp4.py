# cache/to_mp4.py
from pathlib import Path
from typing import Optional, List
import cv2

def clear_cache_images(dirs: List[str], dry_run: bool = False):
    """
    清除快取目錄中的影像檔
    
    Args:
        dirs: 
        dry_run: True 時只預覽不刪除
    """

    dirs = [Path(d) for d in dirs]
    
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
    
    for directory in dirs:
        if not directory.exists():
            print(f"[Skip] 目錄不存在: {directory}")
            continue
            
        found = 0
        deleted = 0
        
        for file in directory.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                found += 1
                if not dry_run:
                    try:
                        file.unlink()
                        deleted += 1
                    except Exception as e:
                        print(f"[Warn] 刪除失敗 {file}: {e}")
        
        status = "預覽" if dry_run else "已刪除"
        print(f"[{directory.name}] 找到 {found} 張，{status} {deleted} 張")


def images_to_video(
    src: str,
    out: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    fps: int = 30,
    codec: str = "mp4v"
):
    """
    將影像合成影片
    
    Args:
        src: 來源目錄
        out: 輸出檔案路徑，None 時輸出到來源目錄的 out.mp4
        start: 時間下限（檔名數字）
        end: 時間上限（檔名數字）
        fps: 影格率
        codec: 編碼器（mp4v, avc1, XVID 等）
    """
    src_path = Path(src)
    
    if not src_path.exists():
        print(f"[Error] 來源目錄不存在: {src_path}")
        return
    
    # 收集影像
    images = []
    for file in src_path.glob("*.png"):
        try:
            time = int(file.stem)
            if start is not None and time < start:
                continue
            if end is not None and time > end:
                continue
            images.append((time, file))
        except ValueError:
            continue
    
    if not images:
        print("[Error] 沒有找到符合條件的影像")
        return
    
    images.sort(key=lambda x: x[0])
    
    # 輸出路徑
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 讀取第一張確定尺寸
    first_img = cv2.imread(str(images[0][1]))
    if first_img is None:
        print("[Error] 無法讀取第一張影像")
        return
    
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    if not writer.isOpened():
        print(f"[Error] 無法建立影片寫入器")
        return
    
    count = 0
    for time, path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[Warn] 跳過無法讀取的影像: {path.name}")
            continue
        
        # 若尺寸不同則 resize
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        writer.write(img)
        count += 1
    
    writer.release()
    
    print(f"[完成] 共 {count} 張影像")
    print(f"       時間範圍: {images[0][0]} ~ {images[-1][0]}")
    print(f"       輸出: {out_path}")



# 使用範例
if __name__ == "__main__":
    
    dirs = ['screenshots/img', 'screenshots/map']
    # 清除快取（預覽模式）    
    # clear_cache_images(dirs,dry_run=True)
    
    # 實際刪除
    # clear_cache_images(dirs)
    

    
    # 合成影片（指定時間範圍）
    
    # images_to_video(src = "../show/out0/img",out="../show/out0/img_video.mp4",
    #                  fps=30)
    images_to_video(src = "../show/out0/map",out="../show/out0/map_video.mp4",
                     fps=30)
    
    # # 合成 map 資料夾的影像
    # images_to_video(src = "screenshots/map",out="video/map_video.mp4",
    #                 start=0, end=36, fps=30)

    
    pass

"""
使用
ffmpeg -i video/img_video.mp4 -c:v libx264 -c:a aac -strict experimental video/fixed_img_video.mp4
ffmpeg -i video/map_video.mp4 -c:v libx264 -c:a aac -strict experimental video/fixed_map_video.mp4
修復影片


ffmpeg -i show/out0/img_video.mp4 -c:v libx264 -c:a aac -strict experimental show/out0/img.mp4
ffmpeg -i show/out0/map_video.mp4 -c:v libx264 -c:a aac -strict experimental show/out0/map.mp4
"""