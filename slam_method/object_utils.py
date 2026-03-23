import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .frame import Frame,KeyFrame
    from .objects import Thing, Stuff
    from .point import Point



def multi_frame_dense_depth2(gray_list, poses_cw, K, mask):
    """
    gray_list  : [gray0, gray1, gray2, ...]   當前幀是 gray0
    poses_cw   : [Tcw_0, Tcw_1, Tcw_2, ...]
    mask       : HxW (0/1)
    K          : 3x3 intrinsic

    回傳：
        depth : HxW dense depth（非稀疏）
    """

    gray0 = gray_list[0]
    H, W = gray0.shape

    # -------------------------------
    # 1. 對 frame0 → frame_i 計算 dense flow
    # -------------------------------
    flows = []
    
    dis = cv2.DISOpticalFlow.create(0)
    
    for i in range(1, len(gray_list)):
        flow = dis.calc(gray0, gray_list[i], None)   # shape = (H,W,2)
        flows.append(flow)
    
    # -------------------------------
    # 2. 建立投影矩陣 P_i = K @ Tcw_i[:3]
    # -------------------------------
    P_list = []
    for Tcw in poses_cw:
        P = K @ Tcw[:3, :]    # 3x4
        P_list.append(P)
    
    # -------------------------------
    # 3. 對每個像素收集 multi-frame 對應 → triangulation
    # -------------------------------
    depth = np.zeros((H,W), dtype=np.float32)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    for v in range(H):
        for u in range(W):
            if mask[v,u] == 0:
                continue

            pts = []  # multi-frame 2D points

            # frame0（當前幀）
            pts.append(np.array([u, v], dtype=np.float32))

            # 其餘幀
            for flow in flows:
                du, dv = flow[v,u]
                u_i = u + du
                v_i = v + dv

                if 0 <= u_i < W and 0 <= v_i < H:
                    pts.append(np.array([u_i, v_i], dtype=np.float32))
                else:
                    pts.append(None)

            # 至少要兩幀有觀測
            pts_valid = [p for p in pts if p is not None]
            if len(pts_valid) < 2:
                continue

            # -------------------------------
            # triangulation
            # -------------------------------
            A = []
            for Pi, p in zip(P_list, pts):
                if p is None:
                    continue
                u_i, v_i = p
                A.append(u_i * Pi[2] - Pi[0])
                A.append(v_i * Pi[2] - Pi[1])

            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X_h = Vt[-1]
            X = X_h[:3] / X_h[3]  # world coordinate

            # -------------------------------
            # 轉成 camera0 座標 → depth = z
            # -------------------------------
            Tcw0 = poses_cw[0]
            X_c0 = Tcw0 @ np.array([X[0], X[1], X[2], 1.0])
            z = X_c0[2]

            if z > 0:
                depth[v,u] = z

    return depth, flows


def flow_fb_check(flow01, flow10, u, v, fb_thresh=1.5):
    """
    forward–backward consistency check
    flow01: frame0 -> frame1
    flow10: frame1 -> frame0
    """
    H, W, _ = flow01.shape
    du, dv = flow01[v, u]
    u1 = u + du
    v1 = v + dv

    # out of boundary → invalid flow
    if not (0 <= u1 < W and 0 <= v1 < H):
        return False

    du2, dv2 = flow10[int(v1), int(u1)]
    fb_error = abs(du + du2) + abs(dv + dv2)

    return fb_error <= fb_thresh



def multi_frame_dense_depth3(gray_list, poses_cw, K, mask, fb_th=1.5, z_limit=float('inf')):
    """
    gray_list: [gray0, gray1, gray2, ...]
    poses_cw : [Tcw_0, Tcw_1, ...]
    K        : intrinsic (3×3)
    mask     : H×W, 1/0
    fb_th: forward-backward threshold 單位為像素
    z_limit  : 深度硬限制，過遠的深度視為錯誤

    回傳:
        depth:     H×W dense depth map
        success_r: 深度預測成功率（0~1）
    """

    gray0 = gray_list[0]
    H, W = gray0.shape

    # --------------------------------------------------------
    # 1) 計算 forward/backward dense optical flow
    # --------------------------------------------------------
    dis = cv2.DISOpticalFlow.create(2)  # MEDIUM
    flows_fwd = []
    flows_bwd = []

    for i in range(1, len(gray_list)):
        f01 = dis.calc(gray0, gray_list[i], None)
        f10 = dis.calc(gray_list[i], gray0, None)
        flows_fwd.append(f01)
        flows_bwd.append(f10)

    # --------------------------------------------------------
    # 2) 建立投影矩陣
    # --------------------------------------------------------
    P_list = [K @ Tcw[:3, :] for Tcw in poses_cw]
    Tcw0 = poses_cw[0]

    depth = np.zeros((H, W), dtype=np.float32)
    pixel_valid = 0
    pixel_total = np.count_nonzero(mask)

    # --------------------------------------------------------
    # 3) mask 內每一像素：挑出"有效幀" → multi-frame DLT
    # --------------------------------------------------------
    for v in range(H):
        for u in range(W):

            if mask[v, u] == 0:
                continue

            pts = [(u, v)]  # always include frame0
            Pis = [P_list[0]]

            # ====== 檢查每一幀是否通過 FB 一致性 ======
            for idx in range(1, len(gray_list)):
                if not flow_fb_check(flows_fwd[idx-1], flows_bwd[idx-1], u, v, fb_th):
                    continue

                du, dv = flows_fwd[idx-1][v, u]
                ui = u + du
                vi = v + dv

                if 0 <= ui < W and 0 <= vi < H:
                    pts.append((ui, vi))
                    Pis.append(P_list[idx])

            # 必須至少 2 幀才能 DLT
            if len(pts) < 2:
                continue

            # --------------------------------------------------------
            # 4) multi-frame DLT（恢復原本的 A 矩陣形式）
            # --------------------------------------------------------
            A = []
            for (ui, vi), Pi in zip(pts, Pis):
                A.append(ui * Pi[2] - Pi[0])
                A.append(vi * Pi[2] - Pi[1])
            A = np.asarray(A)

            try:
                _, _, Vt = np.linalg.svd(A)
            except:
                continue

            X_h = Vt[-1]
            if abs(X_h[3]) < 1e-9:
                continue

            X_w = X_h[:3] / X_h[3]

            # world → cam0
            X_c0 = Tcw0 @ np.array([X_w[0], X_w[1], X_w[2], 1])
            z = float(X_c0[2])

            if z <= 0 or z > z_limit:
                continue

            depth[v, u] = z
            pixel_valid += 1

    # ----------------------------------------------------
    # 5) 成功率
    # ----------------------------------------------------
    success_rate = pixel_valid / (pixel_total + 1e-6)
    return depth, success_rate


# def multi_frame_dense_depth(kf_cur:'KeyFrame', local_kfs:list['KeyFrame'], mask, kf_flows:dict=None, fb_th=1.5, z_limit=float('inf')):
#     """
#     使用多幀密集光流 + forward-backward 一致性檢查進行密集深度估計
    
#     參數:
#         kf_cur: KeyFrame
#             當前關鍵幀,作為深度圖的參考幀
#         local_kfs: list[KeyFrame]
#             局部共視關鍵幀列表,用於多視角三角化
#         mask: np.ndarray, shape=(H,W), dtype=uint8
#             物體遮罩 (1=物體區域, 0=背景)
#         fb_th: float, default=1.5
#             forward-backward 一致性閾值 (單位:像素)
#             光流誤差超過此值的對應點將被過濾
#         z_limit: float, default=inf
#             深度硬限制,超過此距離的深度視為錯誤並丟棄
    
#     回傳:
#         depth: np.ndarray, shape=(H,W), dtype=float32
#             密集深度圖,值為當前幀相機座標系下的 z 距離
#             0 表示三角化失敗或被過濾的像素
    
#     演算法流程:
#         1. 計算當前幀到每個 local_kf 的 forward/backward 密集光流
#         2. 對 mask 內每個像素:
#            a. 檢查 FB 一致性,過濾不可靠的光流
#            b. 收集多幀有效對應點
#            c. 使用 DLT 進行多視角三角化
#            d. 將世界座標轉換到當前幀相機座標系得到深度 z
#         3. 過濾負深度和超過 z_limit 的異常值
    
#     注意:
#         - 每個關鍵幀可使用不同的相機內參矩陣 K
#         - 深度值是相對於當前幀 kf_cur 的相機座標系
#         - 需要至少 2 幀有效對應才能進行三角化
#     """
#     # 避免kfs 包含 kf_cur
#     kfs = [kf for kf in local_kfs if kf != kf_cur]

    
#     # gray0 = gray_list[0]
#     gray0 = cv2.cvtColor(kf_cur.img, cv2.COLOR_BGR2GRAY)
#     H, W = gray0.shape
    
#     # --------------------------------------------------------
#     # 1) 計算 forward/backward dense optical flow
#     # --------------------------------------------------------
#     dis = cv2.DISOpticalFlow.create(2)  # MEDIUM
#     flows_fwd = []
#     flows_bwd = []
    
#     for kf in kfs:
#         if kf_flows is None:
#             gray1 = cv2.cvtColor(kf.img, cv2.COLOR_BGR2GRAY)
#             f01 = dis.calc(gray0, gray1, None)
#             f10 = dis.calc(gray1, gray0, None)
#         else:
#             f01, f10 = kf_flows[kf]
            
#         flows_fwd.append(f01)
#         flows_bwd.append(f10)
    
#     idx_size = len(kfs)
        

#     # --------------------------------------------------------
#     # 2) 建立投影矩陣
#     # --------------------------------------------------------
#     # P_list = [K @ Tcw[:3, :] for Tcw in poses_cw]
#     P_list = [kf.camera.K @ kf.Tcw[:3, :] for kf in kfs] # 容許不同幀有不同K
    
#     P0 = kf_cur.camera.K @ kf_cur.Tcw[:3, :]
#     Tcw0 = kf_cur.Tcw

#     depth = np.zeros((H, W), dtype=np.float32)

#     # --------------------------------------------------------
#     # 3) mask 內每一像素：挑出"有效幀" → multi-frame DLT
#     # --------------------------------------------------------
#     for v in range(H):
#         for u in range(W):

#             if mask[v, u] == 0:
#                 continue

#             pts = [(u, v)]  # always include frame0
#             Pis = [P0]

#             # ====== 檢查每一幀是否通過 FB 一致性 ======
#             for idx in range(idx_size):
#                 if not flow_fb_check(flows_fwd[idx], flows_bwd[idx], u, v, fb_th):
#                     continue
                
#                 du, dv = flows_fwd[idx][v, u]
#                 ui = u + du
#                 vi = v + dv
                
#                 if 0 <= ui < W and 0 <= vi < H:
#                     pts.append((ui, vi))
#                     Pis.append(P_list[idx])
                
#             # 必須至少 2 幀才能 DLT
#             if len(pts) < 2:
#                 continue

#             # --------------------------------------------------------
#             # 4) multi-frame DLT（恢復原本的 A 矩陣形式）
#             # --------------------------------------------------------
#             A = []
#             for (ui, vi), Pi in zip(pts, Pis):
#                 A.append(ui * Pi[2] - Pi[0])
#                 A.append(vi * Pi[2] - Pi[1])
#             A = np.asarray(A)

#             try:
#                 _, _, Vt = np.linalg.svd(A)
#             except:
#                 continue

#             X_h = Vt[-1]
#             if abs(X_h[3]) < 1e-9:
#                 continue

#             X_w = X_h[:3] / X_h[3]

#             # world → cam0
#             X_c0 = Tcw0 @ np.array([X_w[0], X_w[1], X_w[2], 1])
#             z = float(X_c0[2])

#             if z <= 0 or z > z_limit:
#                 continue

#             depth[v, u] = z
    
#     return depth


def multi_frame_dense_depth(kf_cur:'KeyFrame', local_kfs:list['KeyFrame'], mask, kf_flows:dict=None, fb_th=1.5, z_limit=float('inf')):
    """
    使用多幀密集光流 + forward-backward 一致性檢查進行密集深度估計
    
    改進策略：
    1. 使用更穩定的光流算法（Farneback）
    2. 分批處理像素，避免記憶體累積
    3. 添加深度中值濾波提高穩定性
    4. 使用更魯棒的三角化方法
    """
    # 避免kfs 包含 kf_cur
    kfs = [kf for kf in local_kfs if kf != kf_cur]
    
    if len(kfs) == 0:
        print("[WARN] No valid keyframes for depth estimation")
        H, W = mask.shape[:2]
        return np.zeros((H, W), dtype=np.float32)
    
    gray0 = cv2.cvtColor(kf_cur.img, cv2.COLOR_BGR2GRAY)
    H, W = gray0.shape
    
    # 限制關鍵幀數量
    max_kfs = 4
    if len(kfs) > max_kfs:
        # 選擇最近的關鍵幀
        kfs = kfs[:max_kfs]
    
    # --------------------------------------------------------
    # 1) 計算 forward/backward dense optical flow - 使用穩定算法
    # --------------------------------------------------------
    print(f"[INFO] Computing optical flow for {len(kfs)} keyframes...")
    flows_fwd = []
    flows_bwd = []
    
    try:
        for i, kf in enumerate(kfs):
            if kf_flows is None:
                gray1 = cv2.cvtColor(kf.img, cv2.COLOR_BGR2GRAY)
                
                # 使用 Farneback，避免 DIS 的崩潰問題
                f01 = cv2.calcOpticalFlowFarneback(
                    gray0, gray1, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                f10 = cv2.calcOpticalFlowFarneback(
                    gray1, gray0, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
            else:
                f01, f10 = kf_flows[kf]
            
            # 檢查光流有效性
            if f01 is None or f10 is None:
                print(f"[WARN] Flow {i} is None")
                continue
            if not (np.all(np.isfinite(f01)) and np.all(np.isfinite(f10))):
                print(f"[WARN] Flow {i} contains invalid values")
                continue
                
            flows_fwd.append(f01)
            flows_bwd.append(f10)
            
    except Exception as e:
        print(f"[ERROR] Flow computation failed: {e}")
        return np.zeros((H, W), dtype=np.float32)
    
    if len(flows_fwd) == 0:
        print("[WARN] No valid flows computed")
        return np.zeros((H, W), dtype=np.float32)
    
    idx_size = len(flows_fwd)
    kfs = kfs[:idx_size]
    
    # --------------------------------------------------------
    # 2) 建立投影矩陣
    # --------------------------------------------------------
    P_list = []
    for kf in kfs:
        try:
            P = kf.camera.K @ kf.Tcw[:3, :]
            if np.all(np.isfinite(P)):
                P_list.append(P)
            else:
                print("[WARN] Invalid projection matrix, aborting")
                return np.zeros((H, W), dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to compute projection matrix: {e}")
            return np.zeros((H, W), dtype=np.float32)
    
    P0 = kf_cur.camera.K @ kf_cur.Tcw[:3, :]
    Tcw0 = kf_cur.Tcw
    
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(Tcw0))):
        return np.zeros((H, W), dtype=np.float32)

    depth = np.zeros((H, W), dtype=np.float32)

    # --------------------------------------------------------
    # 3) 分批處理 mask 內的像素，避免記憶體問題
    # --------------------------------------------------------
    ys, xs = np.where(mask > 0)
    
    if len(ys) == 0:
        return depth
    
    print(f"[INFO] Processing {len(ys)} pixels in mask")
    
    # 分批處理，每批 1000 個像素
    batch_size = 1000
    valid_count = 0
    
    for batch_start in range(0, len(ys), batch_size):
        batch_end = min(batch_start + batch_size, len(ys))
        ys_batch = ys[batch_start:batch_end]
        xs_batch = xs[batch_start:batch_end]
        
        # 處理這一批像素
        for v, u in zip(ys_batch, xs_batch):
            try:
                pts = [(u, v)]
                Pis = [P0]

                # 檢查每一幀是否通過 FB 一致性
                for idx in range(idx_size):
                    try:
                        if not flow_fb_check(flows_fwd[idx], flows_bwd[idx], u, v, fb_th):
                            continue
                        
                        du, dv = flows_fwd[idx][v, u]
                        
                        # 嚴格的數值檢查
                        if not (np.isfinite(du) and np.isfinite(dv)):
                            continue
                        if abs(du) > 50 or abs(dv) > 50:  # 過大的光流
                            continue
                        
                        ui = u + du
                        vi = v + dv
                        
                        if 0 <= ui < W and 0 <= vi < H:
                            pts.append((ui, vi))
                            Pis.append(P_list[idx])
                    except:
                        continue
                
                # 必須至少 2 幀才能 DLT
                if len(pts) < 2:
                    continue

                # --------------------------------------------------------
                # 4) multi-frame DLT - 優化數值穩定性
                # --------------------------------------------------------
                A = []
                for (ui, vi), Pi in zip(pts, Pis):
                    row1 = ui * Pi[2] - Pi[0]
                    row2 = vi * Pi[2] - Pi[1]
                    
                    if not (np.all(np.isfinite(row1)) and np.all(np.isfinite(row2))):
                        continue
                    
                    A.append(row1)
                    A.append(row2)
                
                if len(A) < 4:
                    continue
                
                # 使用 float64 提高精度
                A = np.array(A, dtype=np.float64)
                
                # 檢查矩陣是否病態
                if np.linalg.cond(A) > 1e12:
                    continue
                
                # 使用更安全的 SVD
                try:
                    U, s, Vt = np.linalg.svd(A, full_matrices=True)
                    
                    # 檢查奇異值
                    if s[-1] / s[0] < 1e-10:  # 條件數過大
                        continue
                    
                except np.linalg.LinAlgError:
                    continue
                except:
                    continue

                X_h = Vt[-1]
                
                if abs(X_h[3]) < 1e-9:
                    continue

                X_w = X_h[:3] / X_h[3]
                
                if not np.all(np.isfinite(X_w)):
                    continue

                # world → cam0
                X_c0 = Tcw0 @ np.array([X_w[0], X_w[1], X_w[2], 1], dtype=np.float64)
                z = float(X_c0[2])

                if not np.isfinite(z):
                    continue
                if z <= 0.1 or z > z_limit:  # 添加最小深度限制
                    continue

                depth[v, u] = z
                valid_count += 1
                
            except Exception as e:
                # 靜默處理
                continue
        
        # 每批處理後清理一下（幫助垃圾回收）
        if (batch_start + batch_size) % 5000 == 0:
            import gc
            gc.collect()
    
    print(f"[INFO] Successfully computed depth for {valid_count}/{len(ys)} pixels ({100*valid_count/len(ys):.1f}%)")
    
    # --------------------------------------------------------
    # 5) 後處理：中值濾波提高穩定性
    # --------------------------------------------------------
    if valid_count > 100:  # 只有足夠的點才做濾波
        depth = median_filter_depth(depth, mask, kernel_size=5)
    
    return depth


def median_filter_depth(depth, mask, kernel_size=5):
    """
    對深度圖應用中值濾波，只在 mask 內操作
    這可以顯著提高深度穩定性，去除異常值
    """
    H, W = depth.shape
    depth_filtered = depth.copy()
    k = kernel_size // 2
    
    ys, xs = np.where((mask > 0) & (depth > 0))
    
    for v, u in zip(ys, xs):
        y0, y1 = max(0, v - k), min(H, v + k + 1)
        x0, x1 = max(0, u - k), min(W, u + k + 1)
        
        # 取鄰域內的有效深度值
        neighbor_region = depth[y0:y1, x0:x1]
        neighbor_mask = mask[y0:y1, x0:x1]
        
        valid_depths = neighbor_region[(neighbor_region > 0) & (neighbor_mask > 0)]
        
        if len(valid_depths) >= 3:
            # 使用中值濾波
            median_depth = np.median(valid_depths)
            
            # 如果當前深度與中值差異太大，替換為中值
            if abs(depth[v, u] - median_depth) > 0.5 * median_depth:
                depth_filtered[v, u] = median_depth
    
    return depth_filtered

from collections import defaultdict

def calc_flows(kf_cur:'KeyFrame',local_kfs:list['KeyFrame']):
    gray0 = cv2.cvtColor(kf_cur.img, cv2.COLOR_BGR2GRAY)
    H, W = gray0.shape
    dis = cv2.DISOpticalFlow.create(2)  # MEDIUM
    kf_flows = dict()
    
    for kf in local_kfs:
        calc = False
        for obj in kf.objects:
            if obj is None:continue
            if not obj.is_thing:continue
            if obj in kf_cur.objects:
                calc = True
                break
        
        if calc:
            gray1 = cv2.cvtColor(kf.img, cv2.COLOR_BGR2GRAY)
            f01 = dis.calc(gray0, gray1, None)
            f10 = dis.calc(gray1, gray0, None)
            
            kf_flows[kf]= (f01, f10)
            
    
    return kf_flows
    
    



def clean_depth(depth, mask, k_sigma=3.0):
    """
    depth: H×W float32, triangulation 結果 (0 = 三角化失敗/洞，不能在此階段當異常處理)
    mask : H×W uint8, 物體遮罩 (1=物體, 0=背景)
    k_sigma: 幾個 robust sigma 之外視為 outlier
    
    對 depth>0 的 pixel，用全域 median + MAD(平均絕對偏差) 找異常值，設為 0
    """
    depth = depth.astype(np.float32)
    
    # -----------------------
    # (1) 去除異常值：只看 depth > 0
    # -----------------------
    valid = (mask > 0) & (depth > 0)
    vals = depth[valid]

    if len(vals) == 0:
        # 沒任何有效深度，直接回傳
        return depth
    
    med = np.median(vals) # 中位數
    abs_dev = np.abs(vals - med)
    mad = np.median(abs_dev) # 平均絕對偏差
    
    depth_clean = depth.copy()
    
    if mad > 1e-9: # 數值有足夠差異
        sigma = 1.4826 * mad  # robust std
        
        dev = np.zeros_like(depth, dtype=np.float32)
        dev[valid] = np.abs(depth[valid] - med)
        
        outlier_mask = np.zeros_like(depth, dtype=bool)
        outlier_mask[valid] = dev[valid] > (k_sigma * sigma)
        
        # outlier → 設成 0，變成洞，留給下一階段補
        depth_clean[outlier_mask] = 0.0
    else:
        # 深度幾乎都一樣，當作沒有 outlier
        pass
    
    
    return depth_clean


def filter_mesh_by_components(mesh,th:float=0.05):
    """
    只保留三角形數量 >= min_triangles 的 connected components
    min_triangles = 三角形總數*th
    用於移除零碎、不穩定 mesh
    """

    if len(mesh.triangles) == 0:
        return mesh

    # cluster_connected_triangles:
    # triangle_clusters: 每個 triangle 屬於哪個 component
    # cluster_n_triangles: 每個 component 的 triangle 數
    # cluster_area: 每個 component 的表面積
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()

    # print(cluster_n_triangles)
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    min_triangles = len(mesh.triangles)*th
    # 找出要保留的 component
    keep_clusters = set(
        np.where(cluster_n_triangles >= min_triangles)[0]
    )

    # 標記要移除的 triangle
    remove_triangles = [
        i for i, cid in enumerate(triangle_clusters)
        if cid not in keep_clusters
    ]

    mesh.remove_triangles_by_index(remove_triangles)
    mesh.remove_unreferenced_vertices()

    return mesh

# def fill_depth_nn(depth, mask, max_radius=30):
#     """
#     depth: HxW float32, 0 = hole
#     mask : HxW uint8/bool, 1=object
#     max_radius: 超過這個像素距離就不補（避免跨太遠亂補）
#     """
#     depth = depth.astype(np.float32)
#     H, W = depth.shape
#     depth_filled = depth.copy()

#     valid_y, valid_x = np.where((mask > 0) & (depth > 0))
#     hole_y, hole_x   = np.where((mask > 0) & (depth == 0))

#     if len(valid_x) == 0 or len(hole_x) == 0:
#         return depth_filled

#     valid_coords = np.column_stack([valid_y, valid_x])
#     hole_coords  = np.column_stack([hole_y, hole_x])

#     tree = cKDTree(valid_coords)
#     dists, idxs = tree.query(hole_coords, k=1)

#     # 只補距離夠近的洞（避免大洞被硬拉）
#     ok = dists <= max_radius
#     vy = valid_y[idxs[ok]]
#     vx = valid_x[idxs[ok]]

#     depth_filled[hole_y[ok], hole_x[ok]] = depth[vy, vx]
#     return depth_filled


# def fill_depth(depth, mask,neigh=10):
#     """填補深度圖中的空洞,使用鄰域中值濾波的方法。

#     Args:
#         depth (_type_): _description_
#         mask (_type_): _description_
#         neigh (int, optional): _description_. Defaults to 10.

#     Returns:
#         _type_: _description_
#     """
#     H, W = depth.shape
#     k = neigh // 2
#     depth_filled = depth.copy()
    
#     ys, xs = np.where((mask > 0) & (depth == 0))
#     for v, u in zip(ys, xs):
#         y0, y1 = max(0, v - k), min(H, v + k + 1)
#         x0, x1 = max(0, u - k), min(W, u + k + 1)

#         neigh_vals = depth[y0:y1, x0:x1]
#         neigh_vals = neigh_vals[neigh_vals > 0]

#         if len(neigh_vals) < 3:
#             continue

#         depth_filled[v, u] = float(np.median(neigh_vals))
    
#     return depth_filled

# def fill_depth(depth, mask, neigh=10):
#     """填補深度圖中的空洞,使用鄰域中值濾波的方法。
#     加入異常檢測避免crash

#     Args:
#         depth (_type_): _description_
#         mask (_type_): _description_
#         neigh (int, optional): _description_. Defaults to 10.

#     Returns:
#         _type_: _description_
#     """
#     H, W = depth.shape
#     k = neigh // 2
#     depth_filled = depth.copy()
    
#     ys, xs = np.where((mask > 0) & (depth == 0))
    
#     # 添加檢查避免空數組
#     if len(ys) == 0:
#         return depth_filled
    
#     for v, u in zip(ys, xs):
#         y0, y1 = max(0, v - k), min(H, v + k + 1)
#         x0, x1 = max(0, u - k), min(W, u + k + 1)

#         neigh_vals = depth[y0:y1, x0:x1]
#         neigh_vals = neigh_vals[neigh_vals > 0]
        
#         # 過濾異常值
#         neigh_vals = neigh_vals[np.isfinite(neigh_vals)]

#         if len(neigh_vals) < 3:
#             continue

#         try:
#             # 使用 nanmedian 更安全，並添加異常處理
#             median_val = np.nanmedian(neigh_vals)
#             if np.isfinite(median_val):
#                 depth_filled[v, u] = float(median_val)
#         except Exception as e:
#             # 如果中值計算失敗，使用平均值作為後備
#             try:
#                 mean_val = np.mean(neigh_vals)
#                 if np.isfinite(mean_val):
#                     depth_filled[v, u] = float(mean_val)
#             except:
#                 continue
    
#     return depth_filled


def fill_depth(depth, mask, neigh=10):
    """用最近鄰 + 遮罩加權盒式均值補洞，避免邊界深度被拉低且不依賴 np.median。"""
    from scipy.ndimage import distance_transform_edt

    depth_filled = depth.copy()

    # 找出需補洞區域
    holes = (mask > 0) & (depth == 0)
    if not np.any(holes):
        return depth_filled

    # 找出可用的有效深度
    valid = (depth > 0) & (mask > 0) & np.isfinite(depth)
    if not np.any(valid):
        print("[WARN] No valid depth values to interpolate from")
        return depth_filled

    # 1) 最近鄰填補
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    depth_filled[holes] = depth[indices[0, holes], indices[1, holes]]

    # 2) 遮罩加權盒式均值平滑（只作用在洞位置，避免零背景稀釋）
    ksize = max(3, neigh | 1)  # 奇數且至少 3
    weight = ((depth_filled > 0) & (mask > 0) & np.isfinite(depth_filled)).astype(np.float32)
    depth_weighted = depth_filled.astype(np.float32) * weight

    sum_depth = cv2.boxFilter(
        depth_weighted, ddepth=-1, ksize=(ksize, ksize),
        normalize=False, borderType=cv2.BORDER_REFLECT
    )
    count = cv2.boxFilter(
        weight, ddepth=-1, ksize=(ksize, ksize),
        normalize=False, borderType=cv2.BORDER_REFLECT
    )
    count = np.maximum(count, 1e-6)  # 避免除零
    avg = sum_depth / count

    depth_filled[holes] = avg[holes]

    return depth_filled


def inpaint_depth(depth, mask):
    """
    使用 OpenCV inpaint 將深度補齊。
    
    會使用遮罩外的0深度，對邊緣修復會導致錯誤
    """

    depth_norm = depth.copy()
    max_val = depth_norm.max() if depth_norm.max() > 0 else 1
    # 正規化到 0~255
    depth_8u = (depth_norm / max_val * 255).astype(np.uint8)

    # 補洞區域 = mask & 深度為0 的 pixel
    hole_mask = ((mask > 0) & (depth == 0)).astype(np.uint8) * 255

    if np.count_nonzero(hole_mask) == 0:
        return depth

    depth_inpaint = cv2.inpaint(depth_8u, hole_mask, 3, cv2.INPAINT_NS)
    depth_inpaint = depth_inpaint.astype(np.float32) / 255 * max_val

    return depth_inpaint




def compute_normals_from_depth(depth, K, mask):
    """
    depth : (H,W) float32
    K     : 3x3 intrinsic
    mask  : (H,W) {0,1}

    return:
        normals_cam : (H,W,3) camera-frame normals
        valid       : (H,W) bool
    """
    H, W = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 深度梯度（像素空間）
    dzdu = np.zeros_like(depth)
    dzdv = np.zeros_like(depth)

    dzdu[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) * 0.5
    dzdv[1:-1, :] = (depth[2:, :] - depth[:-2, :]) * 0.5

    # 只在 mask + depth>0 的地方有效
    valid = (depth > 0) & (mask > 0)

    normals = np.zeros((H, W, 3), dtype=np.float32)

    # 根據 X(u,v) 對 u,v 的偏導推導（已化簡）
    normals[..., 0] = -dzdu * fx
    normals[..., 1] = -dzdv * fy
    normals[..., 2] = 1.0

    # 正規化
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    normals /= norm

    normals[~valid] = 0.0
    return normals, valid



def build_pc_obs_from_depth(depth, K, Twc, mask):
    """
    depth : (H,W)
    K     : 3x3
    Twc   : 4x4 camera->world
    mask  : (H,W)

    return:
        pc_obs : open3d.geometry.PointCloud
    """
    H, W = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    normals_cam, valid = compute_normals_from_depth(depth, K, mask)

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return None

    Z = depth[ys, xs]
    Xc = np.zeros((len(xs), 3), dtype=np.float32)
    Xc[:, 0] = (xs - cx) * Z / fx
    Xc[:, 1] = (ys - cy) * Z / fy
    Xc[:, 2] = Z

    # camera -> world
    Rwc = Twc[:3, :3]
    twc = Twc[:3, 3]

    Xw = (Rwc @ Xc.T).T + twc
    Nw = (Rwc @ normals_cam[ys, xs].T).T

    pc_obs = o3d.geometry.PointCloud()
    pc_obs.points  = o3d.utility.Vector3dVector(Xw)
    pc_obs.normals = o3d.utility.Vector3dVector(Nw)

    return pc_obs



def track_points_multi_frame(gray_list, max_points=2000):
    """
    gray_list[0] = 當前幀 (kf_cur)
    回傳：
        tracks: list of np.ndarray，shape = (num_frames, 2)
    """
    if len(gray_list) < 2:
        return []

    h, w = gray_list[0].shape[:2]

    # 在當前幀找角點
    p0 = cv2.goodFeaturesToTrack(
        gray_list[0],
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=5
    )
    if p0 is None:
        return []

    # p0: (N,1,2) → (N,2)
    p0 = p0.reshape(-1, 2).astype(np.float32)

    # 初始化每條 track：目前只有第 0 幀的座標
    tracks = [ [p] for p in p0 ]

    prev_img = gray_list[0]
    prev_pts = p0.copy()

    # 依序往後追蹤
    for img in gray_list[1:]:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_img, img,
            prev_pts.reshape(-1, 1, 2), None
        )
        if next_pts is None:
            return []

        next_pts = next_pts.reshape(-1, 2)
        status = status.reshape(-1).astype(bool)

        new_tracks = []
        new_prev_pts = []

        for tr, ok, pt in zip(tracks, status, next_pts):
            if ok:
                tr.append(pt)
                new_tracks.append(tr)
                new_prev_pts.append(pt)

        tracks = new_tracks
        if len(tracks) == 0:
            break

        prev_pts = np.array(new_prev_pts, dtype=np.float32)
        prev_img = img

    # 只保留完整經過所有幀的 tracks
    num_frames = len(gray_list)
    full_tracks = [
        np.array(tr, dtype=np.float32)  # shape = (num_frames, 2)
        for tr in tracks
        if len(tr) == num_frames
    ]

    return full_tracks


def densify_depth_in_mask(depth_sparse, mask, do_bilateral=True):
    """
    讓深度在 mask=1 的地方變 dense（插補），避免 TSDF only sparse。
    使用 KDTree 最近鄰以確保不會 index error。
    """

    depth = depth_sparse.copy().astype(np.float32)
    H, W = depth.shape

    # 1. 找出所有有效深度的位置
    valid_y, valid_x = np.where((depth > 0) & (mask > 0))
    if len(valid_x) == 0:
        # sparse depth 全空 → 回傳原 depth
        return depth

    valid_coords = np.column_stack((valid_y, valid_x))
    valid_depth  = depth[valid_y, valid_x]

    # 建 KDTree
    tree = cKDTree(valid_coords)

    # 2. 找出 mask 中的洞（沒有深度但 mask=1）
    holes_y, holes_x = np.where((depth == 0) & (mask > 0))

    if len(holes_x) == 0:
        # 沒洞 → 已經滿了
        return depth

    hole_coords = np.column_stack((holes_y, holes_x))

    # 3. 對每個 hole 找最近的 valid depth
    dists, idxs = tree.query(hole_coords, k=1)
    
    depth_filled = depth.copy()
    depth_filled[holes_y, holes_x] = valid_depth[idxs]

    # 4. 平滑（可選）
    if do_bilateral:
        depth_smooth = cv2.bilateralFilter(
            depth_filled, 
            d=5,
            sigmaColor=float(np.std(valid_depth) + 1e-3),
            sigmaSpace=5
        )
        depth_filled[mask > 0] = depth_smooth[mask > 0]

    # 遮罩外保持 0
    depth_filled[mask == 0] = 0

    return depth_filled

# def triangulate_multi_view(P_list, pts_list):
#     """
#     P_list: list of 3x4 投影矩陣 P_i = K @ Tcw_i
#             其中 Tcw_i: world -> camera_i
#     pts_list: list / array shape (N, 2), [ (u0,v0), (u1,v1), ... ]
#               與 P_list 對應同一順序的觀測。
#     回傳:
#         X_w: 世界座標 3D 點 (3,)
#     """
#     A = []
#     for P, (u, v) in zip(P_list, pts_list):
#         # (u, v, 1) × (P X) = 0 → 標準線性三角化兩行
#         A.append(u * P[2] - P[0])
#         A.append(v * P[2] - P[1])
#     A = np.asarray(A)  # (2N, 4)

#     _, _, Vt = np.linalg.svd(A)
#     X_h = Vt[-1]            # 最後一列
#     X = X_h[:3] / X_h[3]    # homogeneous → 3D

#     return X  # 世界座標



# def build_depth_from_tracks(tracks, poses_cw, K, img_shape, mask,
#                             max_depth=np.inf):
#     """
#     tracks: list of np.ndarray, 每個 shape = (num_frames, 2)，
#             tracks[i][0] 一定是當前幀 kf_cur 的 (u,v)

#     poses_cw: list of 4x4 Tcw_i (world -> camera_i)，
#               順序與 tracks 的每個點對上的影像幀一樣：
#               poses_cw[0] 對當前幀，poses_cw[1] 對第二幀 ...

#     K: 3x3 intrinsic
#     img_shape: (H, W)，當前幀圖像大小
#     mask: HxW，物體遮罩 (0 or 1)
#     max_depth: 可選的深度上限（單位同 SLAM 世界尺度）

#     回傳:
#         depth: HxW float32，值是「在當前幀相機座標系」的 z_c0
#     """
#     H, W = img_shape
#     depth = np.zeros((H, W), dtype=np.float32)
#     cnt   = np.zeros((H, W), dtype=np.int32)

#     # 準備每一幀的投影矩陣 P_i = K [R|t]
#     P_list = []
#     for Tcw in poses_cw:
#         P = K @ Tcw[:3, :]
#         P_list.append(P)

#     Tcw0 = poses_cw[0]  # 當前幀 world -> camera0

#     for tr in tracks:
#         # tr shape (num_frames, 2)
#         if tr.shape[0] != len(P_list):
#             continue

#         X_w = triangulate_multi_view(P_list, tr)  # 世界座標

#         # 轉到「當前幀相機座標」
#         X_h = np.array([X_w[0], X_w[1], X_w[2], 1.0], dtype=np.float64)
#         X_c0 = Tcw0 @ X_h
#         z = float(X_c0[2])

#         # z 就是 depth（沿當前幀光軸）
#         if z <= 0 or z > max_depth:
#             continue

#         u0, v0 = tr[0]
#         u = int(round(u0))
#         v = int(round(v0))
#         if not (0 <= u < W and 0 <= v < H):
#             continue
#         if mask[v, u] == 0:
#             continue

#         depth[v, u] += z
#         cnt[v, u] += 1

#     valid = cnt > 0
#     if np.any(valid):
#         depth[valid] /= cnt[valid]

#     depth[mask == 0] = 0

#     return depth

def triangulate_multi_view_cam0(P_list, pts_list):
    """
    P_list: list of 3x4 投影矩陣，定義在「當前幀相機座標系」底下：
        P0 = K [ I | 0 ]                （當前幀）
        Pi = K [ R_i0 | t_i0 ]          （其它幀，相對於當前幀）

    pts_list: shape (N, 2)，對應每個 P_i 的 (u,v)

    回傳:
        X_c0: 3D 點在「當前幀相機座標系」的座標 (3,)
    """
    A = []
    for P, (u, v) in zip(P_list, pts_list):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])

    A = np.asarray(A)  # (2N, 4)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X = X_h[:3] / X_h[3]   # homogeneous → 3D

    return X  # 在 camera0 frame 座標下


def build_depth_from_tracks(tracks, Twc_list, K, img_shape, mask,
                            max_depth=np.inf, debug_limit=5):
    """
    tracks: list of np.ndarray, 每個 shape = (num_frames, 2)
            tr[0] 是當前幀的 (u,v)

    Twc_list: list of 4x4 Twc_i (camera_i -> world)
              Twc_list[0] 對應當前幀，之後依序是 keyframes[1], keyframes[2], ...

    K: 3x3 intrinsic
    img_shape: (H, W) 當前幀解析度
    mask: HxW，0/1 (這裡可以先不嚴格用，之後你再優化)

    max_depth: 過大深度的過濾上限（以「相機0座標系」的 z 為單位）

    回傳:
        depth: HxW float32，值是相機0座標系下的 z_c0
    """
    H, W = img_shape
    depth = np.zeros((H, W), dtype=np.float32)
    cnt   = np.zeros((H, W), dtype=np.int32)

    # --- 準備「相機0座標系」下的投影矩陣 ---
    Twc0 = Twc_list[0]               # camera0 -> world
    Tcw0 = np.linalg.inv(Twc0)       # world -> camera0

    P_list = []

    # 當前幀：P0 = K [I|0]
    P0 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P_list.append(P0)

    # 其他幀：Pi = K [R_i0 | t_i0]，其中 x_ci = R_i0 x_c0 + t_i0
    for Twc_i in Twc_list[1:]:
        # camera0 -> world -> camera_i
        Tcw_i = np.linalg.inv(Twc_i)
        T_ci_c0 = Tcw_i @ Twc0    # x_ci = T_ci_c0 * x_c0_h

        R_i0 = T_ci_c0[:3, :3]
        t_i0 = T_ci_c0[:3, 3:4]

        P_i = K @ np.hstack([R_i0, t_i0])
        P_list.append(P_i)

    # --- 逐條 track 做三角化 ---
    debug_count = 0
    for tr in tracks:
        if tr.shape[0] != len(P_list):
            continue

        X_c0 = triangulate_multi_view_cam0(P_list, tr)  # 在 camera0 座標系

        z = float(X_c0[2])
        if z <= 0 or z > max_depth:
            continue

        u0, v0 = tr[0]
        u = int(round(u0))
        v = int(round(v0))
        if not (0 <= u < W and 0 <= v < H):
            continue

        if mask[v, u] == 0:
            continue

        depth[v, u] += z
        cnt[v, u]   += 1

        if debug_count < debug_limit:
            print(f"[DEBUG] X_c0 = {X_c0}, z = {z}, pixel = ({u},{v})")
            debug_count += 1

    valid = cnt > 0
    if np.any(valid):
        depth[valid] /= cnt[valid]

    depth[mask == 0] = 0

    return depth

'''
def triangulate_multi_view(P_list, pts_track):
    """
    P_list: list of 3x4 投影矩陣 (K [R|t])，長度 = num_frames
    pts_track: np.ndarray, shape = (num_frames, 2)，每列是 [x, y]
    回傳:
        X_w: 世界座標 3D 點 (3,)
    """
    A = []

    for P, pt in zip(P_list, pts_track):
        x = float(pt[0])
        y = float(pt[1])
        # x * P[2] - P[0] = 0
        # y * P[2] - P[1] = 0
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.stack(A, axis=0)  # (2N, 4)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X = X_h[:3] / X_h[3]
    return X

def build_depth_from_tracks(tracks, poses_cw, K, img_shape, mask,
                            max_depth=float('inf')):
    """
    tracks: list of np.ndarray, 每條 shape = (num_frames, 2)
            tracks[i][0] 是當前幀的 pixel
    poses_cw: list of 4x4 Tcw (world -> camera)，長度 = num_frames
    K: 3x3 intrinsic
    img_shape: (H, W)
    mask: 當前幀遮罩 (uint8, 0/1 或 0/255)
    max_depth: 深度上限（過濾錯誤的超遠點）
    回傳:
        depth: HxW float32，以當前幀相機座標的 z 為值
    """
    H, W = img_shape
    depth = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.int32)

    # 投影矩陣 P_i（世界 -> 第 i 相機 -> 像平面）
    P_list = [K @ Tcw[:3, :] for Tcw in poses_cw]

    Tcw_cur = poses_cw[0]  # 當前幀的 world→camera

    for pts_track in tracks:
        # pts_track shape = (num_frames, 2)

        # 三角化得到世界座標
        X_w = triangulate_multi_view(P_list, pts_track)
        # print("X_w =", X_w)
        # 轉到當前幀的相機座標系
        X_h = np.array([X_w[0], X_w[1], X_w[2], 1.0], dtype=np.float32)
        X_c0 = Tcw_cur @ X_h
        z = float(X_c0[2])

        # 投影位置用當前幀的 pixel (track[0])
        u0 = pts_track[0]
        x = int(round(u0[0]))
        y = int(round(u0[1]))

        if not (0 <= x < W and 0 <= y < H):
            continue

        # 遮罩 & 深度範圍檢查
        if mask[y, x] == 0:
            continue
        if z <= 0 or z > max_depth:
            continue

        depth[y, x] += z
        count[y, x] += 1

    valid = count > 0
    if np.any(valid):
        depth[valid] /= count[valid]

    depth[mask == 0] = 0

    # 可選：輕微濾波
    if np.any(valid):
        depth = cv2.bilateralFilter(depth, 5, 1.0, 1.0)

    return depth


'''

import copy
import os

def save_view_dirs(view_ds: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert view_ds.ndim == 2 and view_ds.shape[1] == 3
    np.save(path, view_ds)

def load_view_dirs(path: str) -> np.ndarray:
    view_ds = np.load(path)
    if view_ds.ndim != 2 or view_ds.shape[1] != 3:
        raise RuntimeError("Invalid view_ds shape")
    return view_ds

def save_pointcloud(pc: o3d.geometry.PointCloud, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(
        path,
        pc,
        write_ascii=False,     # binary，快又小
        compressed=True        # 可選
    )


def load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    pc = o3d.io.read_point_cloud(path)
    if not pc.has_points():
        raise RuntimeError(f"Failed to load point cloud: {path}")
    return pc

def save_pose(T: np.ndarray, path: str):
    assert T.shape == (4, 4)
    np.save(path, T)

def load_pose(path: str) -> np.ndarray:
    T = np.load(path)
    if T.shape != (4, 4):
        raise RuntimeError("Invalid pose shape")
    return T


def get_line_set(pc_obs, normal_radius=0.1):
    points = np.asarray(pc_obs.points)
    normals = np.asarray(pc_obs.normals)

    scale = normal_radius * 0.5  # 視覺比例，可調
    lines = []
    line_points = []

    for i in range(len(points)):
        p = points[i]
        n = normals[i]
        line_points.append(p)
        line_points.append(p + n * scale)
        lines.append([2*i, 2*i+1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    return line_set
    

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    lines1 = get_line_set(source_temp)
    lines2 = get_line_set(target_temp)
    lines1.paint_uniform_color([1, 0.0, 0])
    lines2.paint_uniform_color([0, 0.0, 1.0])
    o3d.visualization.draw([source_temp, target_temp,lines1,lines2])


def point_to_point_icp(source, target, threshold, trans_init):
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation, "\n")
    draw_registration_result(source, target, reg_p2p.transformation)


def point_to_plane_icp(source, target, threshold, trans_init):
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation, "\n")
    draw_registration_result(source, target, reg_p2l.transformation)
    
    
def recover_object(obj:'Thing', kf:'KeyFrame'):
    
    points_3d_rel = []
    points_2d_obs = []
    
    points:set['Point'] = set()
    obj_idxs = obj.keyframes[kf]
    
    for obj_idx in obj_idxs:
        kp_idxs = list(kf.obj_to_pts[obj_idx])
        pts = kf.points[kp_idxs]
        points.update(pts)
    points.discard(None)
    
    for point in points:
        if point not in obj.static_points_relative:
            continue
            
        pt_3d_rel = obj.static_points_relative[point]
        # 點在物體座標系的位置 p_o
        
        kp_idx = kf.find_point(point)
        if kp_idx is None:
            continue
        
        pt_2d = kf.raw_kps[kp_idx]
        
        points_3d_rel.append(pt_3d_rel) 
        points_2d_obs.append(pt_2d)
    
    if len(points_3d_rel) < 4:
        return None, float('inf')
    
    points_3d_rel = np.array(points_3d_rel, dtype=np.float64)
    points_2d_obs = np.array(points_2d_obs, dtype=np.float64)
    
    
    # print(f"3D points shape: {points_3d_rel.shape}, dtype: {points_3d_rel.dtype}")
    # print(f"2D points shape: {points_2d_obs.shape}, dtype: {points_2d_obs.dtype}")
    
    # 使用 PnP 求解（點在物體座標系，求物體在相機座標系中的位姿）
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d_rel, # 點在物體座標系中的位置 p_o
        points_2d_obs, # 點在圖像中的像素位置
        kf.camera.K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    # p_pixel = K @ [R|t] @ p_object
    # p_c = Tco @ p_o
    # 其中 [R|t] 是 T_co (從物體座標系到相機座標系)
    
    if not success:
        return None, float('inf')
    
    # 物體在相機座標系中的位姿
    R_co, _ = cv2.Rodrigues(rvec)
    t_co = tvec.flatten()
    
    T_co = np.eye(4)
    T_co[:3, :3] = R_co
    T_co[:3, 3] = t_co
    
    # 轉換到世界座標系：T_wo = T_wc @ T_co
    T_wo = kf.Twc @ T_co
    # 取得物體在當前座標系的pose(位姿)
    
    # 計算重投影誤差
    points_reproj, _ = cv2.projectPoints(
        points_3d_rel[inliers],
        rvec, tvec,
        kf.camera.K,
        None
    )
    errors = np.linalg.norm(
        points_2d_obs[inliers.flatten()] - points_reproj.squeeze(),
        axis=1
    )
    mean_error = np.mean(errors)
    # print('inliers',len(inliers),inliers)
    
    return T_wo, mean_error
    
    
'''
def recover_object(self, obj:'Thing', kf:'KeyFrame', max_iterations: int = 20):
    """
    使用 g2o 最佳化物體位姿變換（基於 2D-3D 對應）
    
    Args:
        obj:
        kf: 觀測關鍵幀
        max_iterations: 優化迭代次數
        
    Returns:
        T: 物體位姿變換
        t_new: 優化後的平移向量 (3,)
        mean_error: 平均重投影誤差（像素）
    """
    
    K = kf.camera.K
    # 準備 points
    points:set['Point'] = set()
    
    obj_idxs = obj.keyframes[kf]
    
    for obj_idx in obj_idxs:
        kp_idxs = list(kf.obj_to_pts[obj_idx])
        pts = kf.points[kp_idxs]
        points.update(pts)
    points.discard(None)
        
        
    
    
    
    ba = BundleAdjustment()
    
    # 添加相機位姿節點（固定）
    ba.add_pose(kf.id, kf.pose, fixed=True)
    
    # 添加物體位姿節點（待優化）
    # 初始估計：物體位於原位置，無旋轉
    initial_pose = np.eye(4)
    initial_pose[:3, 3] = obj.static_center
    obj_pose_id = kf.id + obj.oid  # 避免 ID 衝突
    ba.add_pose(obj_pose_id, initial_pose, fixed=False)
    
    # # 添加點和邊
    # edges = []
    # for i, (pt_3d_rel, pt_2d) in enumerate(zip(points_3d_relative, points_2d_obs)):
    #     point_id = obj_pose_id + i + 1
        
    #     # 添加點節點（固定在物體座標系中）
    #     ba.add_point(point_id, pt_3d_rel, fixed=True, marginalized=False)
        
    #     # 添加重投影邊
    #     information = np.eye(2)
    #     robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    #     edge = ba.add_edge(point_id, obj_pose_id, pt_2d, K, information, robust_kernel)
    #     edges.append(edge)
    
    # 添加點和邊
    edges = []
    for point in points:
        
        pt_3d_rel = obj.static_points_relative[point]
        # 添加點節點（固定在物體座標系中）
        ba.add_point(point.id, pt_3d_rel, fixed=True, marginalized=False)
        
        kp_idx = kf.find_point(point)
        
        
        # 並根據特徵點金字塔層級設 information matrix
        octave = kf.octaves[kp_idx]
        # sigma2 = 1.2 ** octave # 根據 orb scaleFactor 設定
        # information = np.identity(2) * (1.0 / sigma2)
        information = np.identity(2) * kf.get_inv_sigma2(octave)
        pt_2d = kf.raw_kps[kp_idx]
        
        # robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        
        # 添加重投影邊
        edge = ba.add_edge(point.id, obj_pose_id, pt_2d, K, information)
        edges.append(edge)
    
    # 最佳化
    ba.optimize(max_iterations=max_iterations)
    
    # 提取優化後的物體位姿
    optimized_pose = ba.get_pose(obj_pose_id)
    # R_new = optimized_pose[:3, :3]
    # t_new = optimized_pose[:3, 3]
    
    # 計算重投影誤差
    errors = []
    for edge in edges:
        if edge.level() == 0:  # 只計算 inlier
            edge.compute_error()
            error = np.linalg.norm(edge.error())
            errors.append(error)
    
    mean_error = np.mean(errors) if errors else float('inf')
    
    T = optimized_pose
    
    return  T, mean_error
                    
        
    ########################
'''
