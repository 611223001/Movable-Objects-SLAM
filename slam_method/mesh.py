


    
# ============================================================
#  HybridMesh：統一封裝 Open3D legacy / tensor 兩套 Mesh 後端
#  - 對外只暴露統一 API（union / difference / intersection / clean / simplify / split）
#  - 內部在「需要時」自動於 legacy <-> tensor 間轉換
#  - 可選體素近似布林作為最終退路（避免整條管線中斷）
#  - 預設回傳「新物件」，也支援 in-place=True
#  - 方便你把原本 TriangleMesh 的交互，換成這個類別實例之間互相操作
# ============================================================
from __future__ import annotations
from typing import Optional
import numpy as np
import open3d as o3d
import time
from scipy.ndimage import binary_fill_holes, center_of_mass, binary_dilation, binary_erosion
from skimage.measure import marching_cubes

class Mesh():
    """
    統一 Mesh 容器：
      - 內部以 `_mesh` 指向「目前最新的網格實體」，型別可能是 legacy 或 tensor
      - 需要某種版本時，透過屬性 `.legacy` / `.tensor` 取得，若型別不同才轉換一次
      - 轉換後會更新 `_mesh` 指標與對應快取，避免之後重覆轉換
      - Open3D 0.19.0：布林建議用 tensor；清理/簡化走 legacy
    """
    
    # -------------------------
    # 建構 / 轉換
    # -------------------------
    def __init__(self, legacy_mesh: o3d.geometry.TriangleMesh = None):
        """
        初始化：
          legacy_mesh: 若提供，作為內部狀態；否則建立空網格
        """
        self.tensor_device:Optional[o3d.core.Device] = None
        
        if legacy_mesh is None:
            legacy_mesh = o3d.geometry.TriangleMesh()
        
        # 目前有效的網格物件（可能是 legacy 或 tensor）
        self._mesh:o3d.geometry.TriangleMesh|o3d.t.geometry.TriangleMesh = legacy_mesh
        # 快取：當前對應版本的拷貝（用來避免重造）
        self._legacy_cache: o3d.geometry.TriangleMesh = legacy_mesh
        self._tensor_cache: o3d.t.geometry.TriangleMesh = None  # 延遲產生
    
    @property
    def legacy(self) -> o3d.geometry.TriangleMesh:
        """取得 legacy 版本（注意：請勿外部就地修改；若修改請呼叫 self._touch()）"""
        if isinstance(self._mesh, o3d.geometry.TriangleMesh):
            lmesh = self._mesh
        else:
            lmesh = self._mesh.to_legacy()
            self._legacy_cache = lmesh
            self._tensor_cache = None
            self._mesh = lmesh
        
        # lmesh.compute_vertex_normals()
        return lmesh
    @legacy.setter
    def legacy(self,lmesh:o3d.geometry.TriangleMesh):
        self._legacy_cache = lmesh
        self._tensor_cache = None
        self._mesh = lmesh
        
    @property
    def tensor(self) -> o3d.t.geometry.TriangleMesh:
        if isinstance(self._mesh, o3d.t.geometry.TriangleMesh):
            tmesh = self._mesh
        else:
            # 避免vertex_normals 屬性導致警告
            lmesh = self._mesh
            device = self.tensor_device
            
            V = np.asarray(lmesh.vertices, dtype=np.float64, order="C")
            F = np.asarray(lmesh.triangles, dtype=np.int64,  order="C")
                        
            if device is None:
                # tmesh = o3d.t.geometry.TriangleMesh.from_legacy(
                #     self._mesh)
                tmesh = o3d.t.geometry.TriangleMesh()
                tmesh.vertex["positions"] = o3d.core.Tensor(V, dtype=o3d.core.Dtype.Float64)
                tmesh.triangle["indices"] = o3d.core.Tensor(F, dtype=o3d.core.Dtype.Int64)
            else:
                # tmesh = o3d.t.geometry.TriangleMesh.from_legacy(
                #     self._mesh, device=device)
                tmesh = o3d.t.geometry.TriangleMesh(device)
                tmesh.vertex["positions"] = o3d.core.Tensor(V, dtype=o3d.core.Dtype.Float64, device=device)
                tmesh.triangle["indices"] = o3d.core.Tensor(F, dtype=o3d.core.Dtype.Int64,  device=device)

            
            self._legacy_cache = None
            self._tensor_cache = tmesh
            self._mesh = tmesh
        return tmesh
    @tensor.setter
    def tensor(self,tmesh:o3d.t.geometry.TriangleMesh):
        self._legacy_cache = None
        self._tensor_cache = tmesh
        self._mesh = tmesh

    
    def set_tensor_device(self,device: Optional[str] = None):
        """設定轉換成 tensor 版本的device，device 可為 'CPU:0' 或 'CUDA:0' """
        if device is None:
            self.tensor_device = None
        else:
            self.tensor_device = o3d.core.Device(device)
        

    
    @staticmethod
    def from_legacy(lmesh: o3d.geometry.TriangleMesh) -> "Mesh":
        return Mesh(lmesh)
    @staticmethod
    def from_tensor(tmesh: "o3d.t.geometry.TriangleMesh") -> "Mesh":
        return Mesh(tmesh.to_legacy())
    @staticmethod
    def from_numpy(V: np.ndarray, F: np.ndarray) -> "Mesh":
        """
        由 numpy (V,F) 建立 HybridMesh
        """
        m = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64)),
            triangles=o3d.utility.Vector3iVector(np.asarray(F, dtype=np.int32))
        )
        return Mesh(m)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """輸出 (V,F)"""
        V = np.asarray(self.legacy.vertices, dtype=np.float64)
        F = np.asarray(self.legacy.triangles, dtype=np.int32)
        return V, F


    def copy(self) -> "Mesh":
        """複製一份（深拷貝 legacy 急用即可）"""
        
        V, F = self.to_numpy()
        m = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V.copy()),
            triangles=o3d.utility.Vector3iVector(F.copy())
        )
        return Mesh(m)

    @property
    def is_empty(self) -> bool:
        return len(self.legacy.vertices) == 0 or len(self.legacy.triangles) == 0

    # -------------------------
    # 清理 / 簡化（走 legacy）
    # -------------------------
    def clean(self, inplace=True) -> "Mesh":
        """
        原地清理：去重、退化、非流形、未引用、重算法向
        """
        if  inplace:
            mesh=self
        else:
            mesh=self.copy()
            
        m = mesh.legacy
        m.remove_duplicated_vertices()      # 刪除重複的頂點
        m.remove_duplicated_triangles()     # 刪除重複的三角形
        m.remove_degenerate_triangles()     # 移除退化三角形
        m.remove_unreferenced_vertices()    # 刪除未引用的頂點
        m.remove_non_manifold_edges()       # 刪除非流形邊
        m.compute_vertex_normals()          # 計算頂點法線
        mesh.legacy = m
        return mesh
    
    def check(self):
        """
        檢查並列印 legacy 網格物件的多種屬性，這些屬性用於評估網格的幾何與拓撲特性：

        - edge_manifold: 是否為邊流形（無邊界邊的流形性）。若為 True，則每條邊僅屬於最多兩個面，且不存在邊界邊。
        - edge_manifold_boundary: 是否為邊流形（允許邊界邊）。若為 True，則每條邊最多屬於兩個面，允許存在邊界邊(代表網格物件沒有體積，是一個面)
        
        - vertex_manifold: 是否為頂點流形。若為 True，則每個頂點的鄰域可展開為一個連通的平面區域。
        - not_self_inter: 是否非自相交。若為 True，則網格不存在自相交的情況（例如面與面之間穿插）。
        - watertight: 是否密封。若為 True，則網格沒有孔洞，所有邊都被恰好兩個面共享。
        - orientable: 是否可定向。若為 True，則可以為所有面分配一致的法向方向，使得網格具有一致的內外區分。
        """
        lmesh = self.legacy
        # 邊流形（不允許邊界邊）
        edge_manifold = lmesh.is_edge_manifold(allow_boundary_edges=False)
        # 邊流形（允許邊界邊）網格物件沒有體積
        # edge_manifold_boundary = lmesh.is_edge_manifold(allow_boundary_edges=True)
        
        # 頂點流形
        vertex_manifold = lmesh.is_vertex_manifold()
        # 是否自相交（True 表示沒有自相交）
        not_self_inter = not lmesh.is_self_intersecting()
        # 是否密封
        watertight = lmesh.is_watertight()
        # 是否可定向
        orientable = lmesh.is_orientable()

        print(f"  edge_manifold:          {edge_manifold}")
        # print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
        
        print(f"  vertex_manifold:        {vertex_manifold}")
        print(f"  not_self_inter:         {not_self_inter}")
        print(f"  watertight:             {watertight}")
        print(f"  orientable:             {orientable}")
        
    def simplify(self, target_tris: int, inplace: bool = True) -> "Mesh":
        """
        QEM 簡化到目標三角形數（走 legacy）
        """
        if inplace:
            mesh=self
        else:
            mesh=self.copy()
            
        m = mesh.legacy
        m2 = m.simplify_quadric_decimation(max(int(target_tris), 4))
        m2.remove_degenerate_triangles()
        m2.remove_unreferenced_vertices()
        m2.compute_vertex_normals()
        mesh.legacy = m2
        return mesh
        
    # -------------------------
    # 分割（走 legacy）
    # -------------------------
    def split_connected(self) -> list["Mesh"]:
        """
        以三角面連通群分割，回傳多個 Mesh（向量化重編索引，避免 np.vectorize 慢速迴圈）
        步驟：
        1) cluster_connected_triangles 取得每個三角面所屬群集標籤
        2) 對每個群集：
            - 蒐集該群集的三角面索引 tris
            - 將 tris 攤平成 1D 後用 np.unique(..., return_inverse=True)
            取得「使用到的頂點列表 used_vid」與「重編後索引 inverse」
            - inverse 重新 reshape 成 tris 的形狀，即為緊密的 0..(len(used_vid)-1)
            - 取出 verts_all[used_vid] 與重編三角形建立子網格
        3) 做最小清理（去退化、去未引用、重算法向），包成 Mesh 回傳
        """
        if self.is_empty:
            return []

        # 取得每個三角面標籤
        labels, counts, _ = self.legacy.cluster_connected_triangles()
        labels = np.asarray(labels)

        tris_all  = np.asarray(self.legacy.triangles, dtype=np.int32)
        verts_all = np.asarray(self.legacy.vertices, dtype=np.float64)

        parts: list[Mesh] = []
        for lab in np.unique(labels):
            tri_idx = np.where(labels == lab)[0]
            if tri_idx.size == 0:
                continue

            tris = tris_all[tri_idx]                 # (T,3)
            flat = tris.ravel()                      # 攤平成 1D

            # used_vid：該子網格實際用到的「原始頂點索引」
            # inverse：flat 中每個索引在 used_vid 的新位置（0..K-1）
            used_vid, inverse = np.unique(flat, return_inverse=True)
            tris_remap = inverse.reshape(tris.shape).astype(np.int32)  # 重編為緊密索引

            sub_verts = verts_all[used_vid]
            sub = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(sub_verts),
                triangles=o3d.utility.Vector3iVector(tris_remap)
            )

            # 最小必要清理（避免後續布林/簡化不穩）
            sub.remove_degenerate_triangles()
            sub.remove_unreferenced_vertices()
            sub.compute_vertex_normals()

            parts.append(Mesh(sub))

        return parts
    
    @staticmethod
    def _voxel_boolean(a_l: o3d.geometry.TriangleMesh,
                    b_l: o3d.geometry.TriangleMesh,
                    voxel_size: float,
                    mode: str) -> o3d.geometry.TriangleMesh:
        """
        體素域布林退路（天然密封；幾何有解析度近似）：
        mode in {"union","diff","inter"}
        步驟：
        1) 兩網格轉 VoxelGrid（同 voxel_size）
        2) 以集合運算對 voxel index 做 AND/OR/DIFF
        3) 由 VoxelGrid 還原三角網格
        """
        if voxel_size <= 0:
            raise ValueError("voxel_size 必須 > 0")
        va = o3d.geometry.VoxelGrid.create_from_triangle_mesh(a_l, voxel_size=voxel_size)
        vb = o3d.geometry.VoxelGrid.create_from_triangle_mesh(b_l, voxel_size=voxel_size)

        set_a = {tuple(v.grid_index) for v in va.get_voxels()}
        set_b = {tuple(v.grid_index) for v in vb.get_voxels()}
        
        match mode:
            case 'union':
                idx = set_a | set_b
            case 'diff':
                idx = set_a - set_b
            case 'inter':
                idx = set_a & set_b
            case _:
                idx = set_a

        vg = o3d.geometry.VoxelGrid()
        vg.voxel_size = voxel_size
        vg.origin = va.origin  # 粗略沿用 A 的原點
        # 建立體素
        voxels = [o3d.geometry.Voxel(o3d.utility.Int3Vector(list(i))) for i in idx]
        vg.voxels = o3d.utility.Vector3dVector(np.array([v.grid_index for v in voxels], dtype=np.int64))
        # 以 Open3D 介面生成三角網
        approx = o3d.geometry.TriangleMesh.create_from_voxel_grid(va)  # 先產一個骨架以取得面生成規則
        # 上面這行只是為了確保依賴存在；實際上需要用 idx 重建。簡化起見，用佔位回退：
        # 更保險的做法：將 idx 填回 vg 後直接 create_from_voxel_grid(vg)
        approx = o3d.geometry.TriangleMesh.create_from_voxel_grid(vg)

        approx.remove_degenerate_triangles()
        approx.remove_unreferenced_vertices()
        approx.compute_vertex_normals()
        return approx
    
    def _health_ok(self) -> tuple[bool, dict]:
        """
        檢查網格健康度（legacy 上檢查）：
        回傳 (是否健康, 指標字典)
        健康條件：edge_manifold(不允許邊界) 且 vertex_manifold 且 非自相交 且 watertight 且 可定向
        """
        lmesh = self.legacy
        if len(lmesh.vertices) == 0 or len(lmesh.triangles) == 0:
            return False, {"empty": True}
        stats = {
            "edge_manifold": lmesh.is_edge_manifold(allow_boundary_edges=False),
            "vertex_manifold": lmesh.is_vertex_manifold(),
            "self_intersecting": lmesh.is_self_intersecting(),
            "watertight": lmesh.is_watertight(),
            "orientable": lmesh.is_orientable(),
            "empty": False
        }
        ok = (stats["edge_manifold"] and stats["vertex_manifold"]
            and (not stats["self_intersecting"])
            and stats["watertight"] and stats["orientable"])
        return ok, stats
    
    # -------------------------
    # 統一布林：自動選後端
    #   1) 優先：legacy 若有（少數 build 會有，通常沒有）
    #   2) 其次：tensor（0.19.0 正式支援）
    #   3) 退路：體素近似
    # -------------------------
    def _boolean(self,
                other: "Mesh",
                mode: str,
                inplace: bool
                ) -> "Mesh":
        """
        op in {"union","diff","inter"}
        """
        a = self if inplace else self.copy()
        
        ta = a.tensor
        tb = other.tensor
        if tb.device != ta.device:
            tb = tb.to(ta.device) # 保證device一致
            
        match mode:
            case 'union':
                tc = ta.boolean_union(tb)
            case 'diff':
                tc = ta.boolean_difference(tb)
            case 'inter':
                tc = ta.boolean_intersection(tb)
            case _:
                tc = ta
        
        a.tensor = tc
        return a
    
    # -------------------------
    # 對外布林 API
    # -------------------------
    def union(self, mesh:"Mesh", inplace:bool=False)->"Mesh":
        """聯集"""
        return self._boolean(mesh, "union", inplace)
    
    def difference(self, mesh:"Mesh", inplace:bool=False)->"Mesh":
        """差集"""
        return self._boolean(mesh, "diff", inplace)
    
    def intersection(self, mesh:"Mesh", inplace:bool=False)->"Mesh":
        """交集"""
        return self._boolean(mesh, "inter", inplace)
    
    # -------------------------
    # 方便工具
    # -------------------------
    # @staticmethod
    # def from_cone_side(Cw: np.ndarray, ring_3d: np.ndarray)->"Mesh":
    #     """相機中心 + 遠平面環 → 側面三角帶"""
    #     V = [np.asarray(Cw, dtype=np.float64)] + [p for p in np.asarray(ring_3d, dtype=np.float64)]
    #     n = len(ring_3d)
    #     F = [[0, 1+i, 1+((i+1) % n)] for i in range(n)]
    #     lmesh = o3d.geometry.TriangleMesh(
    #         vertices=o3d.utility.Vector3dVector(np.asarray(V)),
    #         triangles=o3d.utility.Vector3iVector(np.asarray(F, dtype=np.int32))
    #     )
    #     lmesh.compute_vertex_normals()
    #     return Mesh(lmesh)
    
    @staticmethod
    def from_cone_side(Cw: np.ndarray, ring_3d: np.ndarray) -> "Mesh":
        """
        以相機中心 Cw 與遠平面多邊形 ring_3d 建立「密封錐台」：
        - 側壁：由錐頂 Cw 與遠端多邊形每條邊形成的三角形帶
        - 遠蓋：將 ring_3d 扇形三角化補上底面
        （不建立近蓋；Cw 即為錐頂）
        """
        Cw = np.asarray(Cw, dtype=np.float64).reshape(3)
        ring = np.asarray(ring_3d, dtype=np.float64)
        n = len(ring)
        if n < 3:
            return Mesh(o3d.geometry.TriangleMesh())

        # --- 1) 確保遠端多邊形繞行方向（法向朝外） ---
        # 以 ring 的質心到 Cw 的方向來決定「外側」；讓遠蓋法向大致遠離 Cw。
        cf = ring.mean(axis=0)
        dir_far = cf - Cw
        nrm_far = np.cross(ring[1] - ring[0], ring[2] - ring[0])
        if np.dot(nrm_far, dir_far) < 0.0:
            ring = ring[::-1].copy()   # 翻轉讓遠蓋法向朝外

        # --- 2) 組裝頂點：第 0 個頂點為 Cw，接著是 ring 頂點 ---
        V = np.vstack([Cw.reshape(1, 3), ring])   # index: 0 是 Cw，1..n 是遠環

        # --- 3) 側壁三角形 ---
        wall_tris = []
        for i in range(n):
            i1 = 1 + i
            j1 = 1 + ((i + 1) % n)
            # 三角形 [Cw, ring[i], ring[i+1]]
            # wall_tris.append([0, i1, j1])
            wall_tris.append([0, j1, i1])

        # --- 4) 遠蓋（扇形三角化）---
        # 新增遠蓋質心頂點，使用 (ring[i], ring[i+1], cf_idx) 的順序
        V = np.vstack([V, cf.reshape(1, 3)])
        cf_idx = V.shape[0] - 1
        cap_tris = []
        for i in range(n):
            i1 = 1 + i
            j1 = 1 + ((i + 1) % n)
            cap_tris.append([i1, j1, cf_idx])  # 方向與上方翻轉一致，法向朝外

        # --- 5) 建立 O3D Mesh 並做最小清理 ---
        F = np.asarray(wall_tris + cap_tris, dtype=np.int32)
        lmesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V),
            triangles=o3d.utility.Vector3iVector(F)
        )
        lmesh.remove_degenerate_triangles()
        lmesh.remove_unreferenced_vertices()
        lmesh.compute_vertex_normals()

        return Mesh(lmesh)
    
    @staticmethod
    def depth_map_to_mesh(depth_map: np.ndarray, K: np.ndarray, zf: float) -> o3d.geometry.TriangleMesh:
        """
        根據深度圖生成 Mesh，並限制 Mesh 的深度值不超過 zf。

        參數：
        - depth_map: np.ndarray, 深度圖 (H, W)
        - K: np.ndarray, 相機內參矩陣 (3, 3)
        - zf: float, 深度上限

        返回：
        - o3d.geometry.TriangleMesh: 生成的 Mesh
        """
        # 獲取深度圖的高和寬
        H, W = depth_map.shape

        # 生成像素網格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten()
        v = v.flatten()
        z = depth_map.flatten()

        # 過濾深度值大於 zf 的點
        mask = z <= zf
        u, v, z = u[mask], v[mask], z[mask]

        # 將像素座標轉換為 3D 空間中的點
        K_inv = np.linalg.inv(K)
        pixels = np.stack([u, v, np.ones_like(u)], axis=0)  # (3, N)
        points_3d = K_inv @ (pixels * z)  # (3, N)

        # 創建點雲
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.T)

        # 使用 Alpha Shape 生成 Mesh
        alpha = 0.05  # Alpha 值，控制 Mesh 的細節程度
        lmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        lmesh.compute_vertex_normals()
        
        return Mesh(lmesh)
    
'''        
    def solidify(self,
                eps: float = 1e-6,
                inplace: bool = True) -> "Mesh":
        """
        將目前網格「固化」：盡量修補布林後常見的小縫/懸單面，讓結果接近密封。
        流程：
        1) 近點焊接（量化合併頂點，避免微小縫隙）
        2) 偵測邊界迴圈（只屬於1個三角形的邊）並三角化「補洞」
        3) 最小清理（退化/未引用），可選擇重算法向
        參數：
        eps : 焊接公差（座標量化網格大小），可依單位調整（1e-6 ~ 1e-5 常用）
        inplace : True=就地修改；False=回傳新物件
        recompute_normals : 最後是否在 legacy 重算法向（建議 True 便於視覺化）
        備註：
        - 本方法不呼叫 remove_non_manifold_edges()，避免把剛補的薄面誤刪
        - 若邊界迴圈嚴重不共平面，簡單投影+扇形三角化可能失敗；此時可改用更強健的 2D earcut
        """
        mesh = self if inplace else self.copy()
        l = mesh.legacy

        if len(l.vertices) == 0 or len(l.triangles) == 0:
            return mesh

        # 1) 近點焊接（量化合併頂點）
        l = Mesh._merge_close_vertices_legacy(l, eps=eps)

        # 2) 找邊界迴圈並補洞
        loops = Mesh._extract_boundary_loops(l)
        if loops:
            l = Mesh._fill_loops_by_projection_and_fan(l, loops)

        # 3) 最小清理 + （可選）重算法向
        l.remove_degenerate_triangles()
        l.remove_unreferenced_vertices()


        mesh.legacy = l
        return mesh

    # ---------------- 私有輔助：近點焊接 ----------------
    @staticmethod
    def _merge_close_vertices_legacy(lmesh: o3d.geometry.TriangleMesh, eps: float = 1e-6) -> o3d.geometry.TriangleMesh:
        """
        以座標量化方式合併「非常接近」的頂點，縮小裂縫。
        eps：量化網格尺寸（越大→越多頂點會被合併）
        """
        V = np.asarray(lmesh.vertices, dtype=np.float64)
        F = np.asarray(lmesh.triangles, dtype=np.int32)
        if V.size == 0 or F.size == 0:
            return lmesh

        key = np.round(V / eps).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        V2 = (uniq.astype(np.float64) * eps)
        F2 = inv[F]  # 重新映射三角形索引

        out = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V2),
            triangles=o3d.utility.Vector3iVector(F2)
        )
        out.remove_degenerate_triangles()
        out.remove_unreferenced_vertices()
        return out

    # ---------------- 私有輔助：萃取邊界迴圈 ----------------
    @staticmethod
    def _extract_boundary_loops(lmesh: o3d.geometry.TriangleMesh, max_loop_len: int = 100000) -> list[list[int]]:
        """
        找出所有「邊界迴圈」：
        - 邊界邊定義：只被 1 個三角形使用的無向邊
        - 回傳：每個迴圈是一串頂點索引（至少3個且閉合）
        """
        V = np.asarray(lmesh.vertices, dtype=np.float64)
        F = np.asarray(lmesh.triangles, dtype=np.int32)
        if V.size == 0 or F.size == 0:
            return []

        # 聚合所有邊（無向）
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        ekey = np.sort(edges, axis=1)
        ekey_tuple = [tuple(e) for e in ekey]

        from collections import Counter, defaultdict
        cnt = Counter(ekey_tuple)
        boundary_edges = np.array([list(e) for e, c in cnt.items() if c == 1], dtype=np.int32)
        if boundary_edges.size == 0:
            return []

        # 把邊界邊串成閉合迴圈（每個節點 degree≈2）
        adj = defaultdict(list)
        for a, b in boundary_edges:
            adj[a].append(b)
            adj[b].append(a)

        loops = []
        visited = set()
        for start in adj.keys():
            if start in visited:
                continue
            loop = [start]
            visited.add(start)
            # 任取一個鄰居延展
            if not adj[start]:
                continue
            cur = adj[start][0]
            prev = start
            steps = 0
            while cur != start and steps < max_loop_len:
                loop.append(cur)
                visited.add(cur)
                nxt_candidates = [x for x in adj[cur] if x != prev]
                if not nxt_candidates:
                    # 非閉合鏈，放棄
                    loop = []
                    break
                nxt = nxt_candidates[0]
                prev, cur = cur, nxt
                steps += 1

            if loop and len(loop) >= 3:
                if loop[0] == loop[-1]:
                    loop = loop[:-1]
                if len(loop) >= 3:
                    loops.append(loop)

        return loops

    # ---------------- 私有輔助：投影+扇形三角化補洞 ----------------
    @staticmethod
    def _fill_loops_by_projection_and_fan(lmesh: o3d.geometry.TriangleMesh,
                                        loops: list[list[int]]) -> o3d.geometry.TriangleMesh:
        """
        將每個邊界迴圈近似視為「局部平面」上的閉合多邊形：
        1) 以 PCA 取得平面基底 (u, v)
        2) 投影至 2D 並確保為 CCW
        3) 以扇形三角化（v0, vi, vi+1）補洞
        備註：
        - 扇形三角化對高度凹的迴圈可能產生重疊；若遇此情況，改用 earcut 2D 更穩。
        """
        V = np.asarray(lmesh.vertices, dtype=np.float64)
        F = np.asarray(lmesh.triangles, dtype=np.int32)

        if not loops:
            return lmesh

        new_tris = []
        for loop in loops:
            P = V[loop]  # (L,3)
            if len(P) < 3:
                continue

            # PCA 估計平面（取前兩主軸為平面基底）
            P_mean = P.mean(axis=0)
            Q = P - P_mean
            # SVD（Q = U S VT），VT 的前兩行為最大主軸
            _, _, VT = np.linalg.svd(Q, full_matrices=False)
            u = VT[0]; v = VT[1]

            # 投影到 2D
            Pu = Q @ u
            Pv = Q @ v
            poly2d = np.stack([Pu, Pv], axis=1)

            # 確保 CCW（以 2D 面積判斷）
            area2 = float(np.dot(poly2d[:, 0], np.roll(poly2d[:, 1], -1)) -
                        np.dot(np.roll(poly2d[:, 0], -1), poly2d[:, 1]))
            if area2 < 0:
                loop = loop[::-1]
                poly2d = poly2d[::-1].copy()

            # 扇形三角化（選 v0 為 loop[0]）
            v0 = loop[0]
            for i in range(1, len(loop) - 1):
                new_tris.append([v0, loop[i], loop[i + 1]])

        if new_tris:
            F2 = np.vstack([F, np.asarray(new_tris, dtype=np.int32)])
            out = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(V),
                triangles=o3d.utility.Vector3iVector(F2)
            )
            out.remove_degenerate_triangles()
            out.remove_unreferenced_vertices()
            return out
        return lmesh
    
    @staticmethod
    def _voxel_boolean(a_l: o3d.geometry.TriangleMesh,
                    b_l: o3d.geometry.TriangleMesh,
                    voxel_size: float,
                    mode: str) -> o3d.geometry.TriangleMesh:
        """
        體素域布林退路（0.19 版無 create_from_voxel_grid，改用自建表面）：
        1) 兩網格各轉 VoxelGrid（同 voxel_size）
        2) 將 voxel index 做集合運算 (OR/AND/DIFF)
        3) 只在「面向空白」的體素面生成 quad（再三角化）
        4) 合併重複頂點、清理，回傳 watertight 三角網格（方塊化）
        """
        if voxel_size <= 0:
            raise ValueError("voxel_size 必須 > 0")

        va = o3d.geometry.VoxelGrid.create_from_triangle_mesh(a_l, voxel_size=voxel_size)
        vb = o3d.geometry.VoxelGrid.create_from_triangle_mesh(b_l, voxel_size=voxel_size)

        A = {tuple(v.grid_index) for v in va.get_voxels()}
        B = {tuple(v.grid_index) for v in vb.get_voxels()}

        if mode == "union":
            occ = A | B
        elif mode == "diff":
            occ = A - B
        elif mode == "inter":
            occ = A & B
        else:
            raise ValueError("mode 必須為 union/diff/inter")

        # 用體素集合生成表面網格
        origin = np.asarray(va.origin, dtype=np.float64)  # 以 A 的 origin 為準
        lmesh = Mesh._mesh_from_voxels(occ, voxel_size, origin)
        lmesh.remove_degenerate_triangles()
        lmesh.remove_unreferenced_vertices()
        lmesh.compute_vertex_normals()
        return lmesh
    
    @staticmethod
    def _mesh_from_voxels(occ: set[tuple[int,int,int]],
                        voxel_size: float,
                        origin: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        由佔據體素集合建 watertight 表面網格：
        - 每個體素是一個立方體；只有在鄰居為「空」的那一面才生成 quad
        - 將 quad 拆成兩個三角形
        - 用 (座標量化) 合併重複頂點，避免指數膨脹
        """
        if not occ:
            return o3d.geometry.TriangleMesh()
        s = float(voxel_size)

        # 六個面：法向與四個角的局部坐標（以體素格點中心為 (i+0.5, j+0.5, k+0.5)）
        faces = [
            ((-1, 0, 0), [(-0.5, -0.5, -0.5), (-0.5, -0.5,  0.5),
                        (-0.5,  0.5,  0.5), (-0.5,  0.5, -0.5)]),  # -X
            (( 1, 0, 0), [( 0.5, -0.5, -0.5), ( 0.5,  0.5, -0.5),
                        ( 0.5,  0.5,  0.5), ( 0.5, -0.5,  0.5)]),  # +X
            (( 0,-1, 0), [(-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5),
                        ( 0.5, -0.5,  0.5), (-0.5, -0.5,  0.5)]),  # -Y
            (( 0, 1, 0), [(-0.5,  0.5, -0.5), (-0.5,  0.5,  0.5),
                        ( 0.5,  0.5,  0.5), ( 0.5,  0.5, -0.5)]),  # +Y
            (( 0, 0,-1), [(-0.5, -0.5, -0.5), (-0.5,  0.5, -0.5),
                        ( 0.5,  0.5, -0.5), ( 0.5, -0.5, -0.5)]),  # -Z
            (( 0, 0, 1), [(-0.5, -0.5,  0.5), ( 0.5, -0.5,  0.5),
                        ( 0.5,  0.5,  0.5), (-0.5,  0.5,  0.5)]),  # +Z
        ]

        V = []
        F = []
        vid_map = {}  # 量化座標→頂點索引，避免重複
        qeps = 1e-12  # 量化公差

        def add_vertex(p3):
            # 量化後作為 key（避免浮點誤差導致重複點）
            key = tuple((np.round(p3 / qeps) * qeps).tolist())
            if key in vid_map:
                return vid_map[key]
            vid = len(V)
            V.append(p3)
            vid_map[key] = vid
            return vid

        occ_set = set(occ)
        for (i, j, k) in occ_set:
            center = origin + s * (np.array([i + 0.5, j + 0.5, k + 0.5], dtype=np.float64))
            for (nx, ny, nz), corners in faces:
                nb = (i + nx, j + ny, k + nz)
                if nb in occ_set:
                    continue  # 鄰居也實心→內部面，跳過
                # 生成此面（quad → 兩三角）
                vids = [add_vertex(center + s * np.array(c, dtype=np.float64)) for c in corners]
                # 面外向：corners 已按外向定序
                F.append([vids[0], vids[1], vids[2]])
                F.append([vids[0], vids[2], vids[3]])

        if not F:
            return o3d.geometry.TriangleMesh()

        lmesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64)),
            triangles=o3d.utility.Vector3iVector(np.asarray(F, dtype=np.int32))
        )
        return lmesh
'''


class VoxelGrid():
    """
    一個為 Open3D 0.19.0 設計的、用於3D體素建模的穩健高階類別，
    整合了 Open3D、NumPy 和 SciPy 的功能。

    核心設計是雙重表示法：
    1. self.grid (o3d.geometry.VoxelGrid): 用於與 Open3D 生態系統互動。
    2. self.volume (np.ndarray): 一個密集的布林陣列，作為所有內部計算的高效後端。

    布林運算現在會自動處理空間對齊，確保操作的穩健性。
    """

    def __init__(self, voxel_grid:o3d.geometry.VoxelGrid = None):
        
        # if not isinstance(voxel_grid, o3d.geometry.VoxelGrid):
        #     raise TypeError("輸入必須是 open3d.geometry.VoxelGrid 類型")
        if voxel_grid is None:
            voxel_grid = o3d.geometry.VoxelGrid()
            
        
        self.grid = voxel_grid
        self.voxel_size = voxel_grid.voxel_size
        
        # 每個 Voxel 物件記錄其自身的邊界，布林運算時會進行動態對齊
        self.origin = self.grid.get_min_bound()
        self.max_bound = self.grid.get_max_bound()
        
        self._volume = None
        self._dirty = True
    
    

    @property
    def volume(self) -> np.ndarray:
        """以 NumPy 陣列形式返回體素的密集體積表示（延遲計算）。"""
        if self._dirty or self._volume is None:
            self._grid_to_volume()
            self._dirty = False
        return self._volume
    
    def _grid_to_volume(self):
        """
        私有方法：將 VoxelGrid 轉換為 NumPy 陣列。
        """
        # 1. 根據儲存的空間錨點（self.origin, self.max_bound）計算 NumPy 陣列的維度。
        # 這確保了 NumPy 陣列的空間範圍與初始的 VoxelGrid 完全一致。
        # print(self.max_bound , self.origin , self.voxel_size)
        
        dims = np.ceil((self.max_bound - self.origin) / self.voxel_size).astype(int)
        dims = np.maximum(dims, 1) # 確保維度至少為1
        self._volume = np.zeros(dims, dtype=bool)

        if not self.grid.has_voxels():
            return

        # 2. 將 VoxelGrid 中的每個體素放置到 NumPy 陣列的正確位置。
        voxels = self.grid.get_voxels()
        for voxel in voxels:
            # 獲取體素中心的世界座標
            voxel_center = self.grid.get_voxel_center_coordinate(voxel.grid_index)
            # 計算相對於 self.origin 的索引
            relative_pos = voxel_center - self.origin
            idx = np.floor(relative_pos / self.voxel_size).astype(int)
            
            # 邊界檢查，防止索引越界
            if np.all(idx >= 0) and np.all(idx < dims):
                self._volume[tuple(idx)] = True


    def _volume_to_grid(self):
        """私有方法：將 NumPy 陣列轉換回 VoxelGrid。"""
        
        self.grid.clear()
        # 將 Open3D 網格的原點與類別自身的原點同步
        # 這是確保座標系統匹配的關鍵步驟。
        
        
        
        if self._volume is not None and np.any(self._volume):
            
            cropped_volume, new_origin, new_max_bound = self._crop_bound(self._volume,self.origin)
            self._volume = cropped_volume
            self.origin = new_origin
            self.max_bound = new_max_bound
            
            
            indices = np.argwhere(self._volume)

            for idx in indices:
                # voxel_center = self.origin + (idx + 0.5) * self.voxel_size
                # grid_index = self.grid.get_voxel(voxel_center)
                
                voxel_to_add = o3d.geometry.Voxel(idx)
                # print(grid_index)
                # print(voxel_to_add)
                self.grid.add_voxel(voxel_to_add)
            
                
            self._dirty = False
        
        self.grid.origin = self.origin
        self.grid.voxel_size = self.voxel_size
        

    def fill(self):
        """填充體素體積內部的孔洞，使其成為實心物件。"""
        self._volume = binary_fill_holes(self.volume)
        self._volume_to_grid()
        return self
    
    @staticmethod
    def from_voxel_size(voxel_size: float) -> 'VoxelGrid':
        """
        建立一個指定 voxel_size 的空 VoxelGrid。
        """
        voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.voxel_size = voxel_size
        voxel_grid.origin = np.zeros(3, dtype=np.float64)
        return VoxelGrid(voxel_grid) 
    
    @staticmethod
    def from_mesh(mesh: o3d.geometry.TriangleMesh, voxel_size: float, fill: bool = True)->'VoxelGrid':
        """從 TriangleMesh 創建一個 Voxel 物件。"""
        
        # if not mesh.is_watertight() and fill:
        #     print("警告：輸入的網格不是水密的，填充結果可能不準確。")

        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
        instance = VoxelGrid(voxel_grid)
        # print(instance.origin,instance.max_bound)
        if fill:
            instance.fill()

        return instance
    
    def to_mesh(self) -> o3d.geometry.TriangleMesh:
        """使用移動立方體演算法將體素體積轉換回 TriangleMesh。"""
        if not np.any(self.volume):
            return o3d.geometry.TriangleMesh()

        padded_volume = np.pad(self.volume, 1, mode='constant', constant_values=0)
        
        verts, faces, _, _ = marching_cubes(padded_volume, level=0.5, spacing=(self.voxel_size,)*3)
        
        verts += self.origin - self.voxel_size
        
        # FIX 2 (Corrected Syntax): 正確翻轉面的環繞順序以校正法向量。
        # 交換每個面的第二個和第三個頂點索引。
        faces = faces[:, ::-1]
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # 最佳實踐：重新計算法向量。
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh
    
    def _get_aligned_volumes(self, other: 'VoxelGrid'):
        """在內部將 self 和 other 的體積對齊到一個共享的計算空間中。"""
        # 1. 決定全域邊界
        global_origin = np.minimum(self.origin, other.origin)
        global_max_bound = np.maximum(self.max_bound, other.max_bound)

        # 2. 計算全域網格的維度
        global_dims = np.ceil((global_max_bound - global_origin) / self.voxel_size).astype(int)
        global_dims = np.maximum(global_dims, 1)

        # 3. 創建對齊後的空體積
        aligned_self_vol = np.zeros(global_dims, dtype=bool)
        aligned_other_vol = np.zeros(global_dims, dtype=bool)

        # 4. 計算偏移量（以體素為單位）
        offset_self = np.round((self.origin - global_origin) / self.voxel_size).astype(int)
        offset_other = np.round((other.origin - global_origin) / self.voxel_size).astype(int)

        shape_self = self.volume.shape
        shape_other = other.volume.shape

        # 5. 將原始體積資料繪製到對齊後的體積中
        s_s = tuple(slice(offset_self[i], offset_self[i] + shape_self[i]) for i in range(3))
        aligned_self_vol[s_s] = self.volume

        s_o = tuple(slice(offset_other[i], offset_other[i] + shape_other[i]) for i in range(3))
        aligned_other_vol[s_o] = other.volume
        
        return aligned_self_vol, aligned_other_vol, global_origin, global_max_bound

    def _crop_bound(self, volume:np.ndarray, origin:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """自動裁切有效範圍

        Args:
            volume (np.ndarray): _description_
            origin (np.ndarray): _description_

        Returns:
            cropped_volume
            new_origin
            new_max_bound
            
        TODO:修改使用時機，並增加外部api
        """
        # 找出 new_volume 的非零區域
        if np.any(volume):
            nonzero = np.argwhere(volume)
            min_idx = nonzero.min(axis=0)
            max_idx = nonzero.max(axis=0)
            # 裁切 volume
            cropped_volume = volume[
                min_idx[0]:max_idx[0]+1,
                min_idx[1]:max_idx[1]+1,
                min_idx[2]:max_idx[2]+1
            ]
            # 計算新的 origin/max_bound
            new_origin = origin + min_idx * self.voxel_size
            new_max_bound = new_origin + np.array(cropped_volume.shape) * self.voxel_size
        else:
            # 空物件
            cropped_volume = volume
            new_origin = origin
            new_max_bound = new_origin + np.array(volume.shape) * self.voxel_size
        
        return cropped_volume, new_origin, new_max_bound
    
    @property
    def is_empty(self):
        return not np.any(self.volume)
        
    def _create_result_voxel(self, new_volume:np.ndarray, origin:np.ndarray):
        """
        根據運算結果創建一個新的 Voxel 實例，並自動裁切有效範圍。
        """
        
        cropped_volume, new_origin, new_max_bound = self._crop_bound(new_volume,origin)
        
        # 建立新的 VoxelGrid
        new_grid = o3d.geometry.VoxelGrid()
        new_grid.voxel_size = self.voxel_size
        new_grid.origin = new_origin

        instance = VoxelGrid(new_grid)
        instance.max_bound = new_max_bound
        instance._volume = cropped_volume
        instance._dirty = False
        instance._volume_to_grid()
        return instance
    
    def union(self, other:VoxelGrid):
        """聯集，自動處理空間對齊。"""
        if not isinstance(other, VoxelGrid) or not np.allclose(self.voxel_size, other.voxel_size):
            raise ValueError("操作對象必須是具有相同 voxel_size 的 Voxel 實例")
        
        aligned_self, aligned_other, global_origin, max_b = self._get_aligned_volumes(other)
        new_volume = aligned_self | aligned_other
        return self._create_result_voxel(new_volume, global_origin)
    
    def inter(self, other:VoxelGrid):
        """交集 intersection，自動處理空間對齊。"""
        if not isinstance(other, VoxelGrid) or not np.allclose(self.voxel_size, other.voxel_size):
            raise ValueError("操作對象必須是具有相同 voxel_size 的 Voxel 實例")

        aligned_self, aligned_other, global_origin, max_b = self._get_aligned_volumes(other)
        new_volume = aligned_self & aligned_other
        return self._create_result_voxel(new_volume, global_origin)
    
    def diff(self, other:VoxelGrid):
        """差集 difference，自動處理空間對齊。"""
        if not isinstance(other, VoxelGrid) or not np.allclose(self.voxel_size, other.voxel_size):
            raise ValueError("操作對象必須是具有相同 voxel_size 的 Voxel 實例")

        aligned_self, aligned_other, global_origin, max_b = self._get_aligned_volumes(other)
        new_volume = aligned_self & ~aligned_other
        return self._create_result_voxel(new_volume, global_origin)
    
    def dilate(self, iterations=1):
        """對體素物件進行形態學膨脹操作。"""
        dilated_volume = binary_dilation(self.volume, iterations=iterations)
        return self._create_result_voxel(dilated_volume, self.origin)
    
    def erode(self, iterations=1):
        """對體素物件進行形態學侵蝕操作。"""
        eroded_volume = binary_erosion(self.volume, iterations=iterations)
        return self._create_result_voxel(eroded_volume, self.origin)
    
    def is_touching(self, other:VoxelGrid) -> bool:
        """檢查此物件是否與另一個物件接觸（膨脹後重疊）。"""
        aligned_self, aligned_other, _, _ = self._get_aligned_volumes(other)
        # 只要有重疊或膨脹後有新交集都算接觸
        overlap = aligned_self & aligned_other
        if np.any(overlap):
            return True
        dilated_self_vol = binary_dilation(aligned_self)
        touching = (dilated_self_vol & aligned_other)
        return np.any(touching)
    
    def is_overlapping(self, other:VoxelGrid) -> bool:
        """檢查此物件是否與另一個物件重疊。"""
        
        aligned_self, aligned_other, _, _ = self._get_aligned_volumes(other)
        return np.any(aligned_self & aligned_other)

    def get_volume(self) -> float:
        """計算體素物件的總體積。"""
        return np.sum(self.volume) * (self.voxel_size ** 3)

    def get_center_of_mass(self) -> np.ndarray:
        """計算質心的世界座標。"""
        if not np.any(self.volume):
            return self.origin + (np.array(self.volume.shape) * self.voxel_size) / 2.0
            
        com_indices = np.array(center_of_mass(self.volume))
        return self.origin + com_indices * self.voxel_size
    
    # -------------------------
    # 方便工具
    # -------------------------
    @staticmethod
    def from_cone_side(Cw: np.ndarray, ring_3d: np.ndarray, voxel_size:float, fill:bool=True) -> "VoxelGrid":
        """
        以相機中心 Cw 與遠平面多邊形 ring_3d 建立「密封錐台」：
        - 側壁：由錐頂 Cw 與遠端多邊形每條邊形成的三角形帶
        - 遠蓋：將 ring_3d 扇形三角化補上底面
        - 近蓋（可選）：在距離 Cw 的 near_cap_distance 處添加一個平面
        """
        Cw = np.asarray(Cw, dtype=np.float64).reshape(3)
        ring = np.asarray(ring_3d, dtype=np.float64)
        n = len(ring)
        if n < 3:
            # return VoxelGrid(o3d.geometry.TriangleMesh())
            return VoxelGrid.from_voxel_size(voxel_size)

        # --- 1) 確保遠端多邊形繞行方向（法向朝外） ---
        # 以 ring 的質心到 Cw 的方向來決定「外側」；讓遠蓋法向大致遠離 Cw。
        cf = ring.mean(axis=0)
        dir_far = cf - Cw
        nrm_far = np.cross(ring[1] - ring[0], ring[2] - ring[0])
        if np.dot(nrm_far, dir_far) < 0.0:
            ring = ring[::-1].copy()   # 翻轉讓遠蓋法向朝外

        # --- 2) 組裝頂點：第 0 個頂點為 Cw，接著是 ring 頂點 ---
        V = np.vstack([Cw.reshape(1, 3), ring])   # index: 0 是 Cw，1..n 是遠環

        # --- 3) 側壁三角形 ---
        wall_tris = []
        for i in range(n):
            i1 = 1 + i
            j1 = 1 + ((i + 1) % n)
            # 三角形 [Cw, ring[i], ring[i+1]]
            # wall_tris.append([0, i1, j1])
            wall_tris.append([0, j1, i1])

        # --- 4) 遠蓋（扇形三角化）---
        # 新增遠蓋質心頂點，使用 (ring[i], ring[i+1], cf_idx) 的順序
        V = np.vstack([V, cf.reshape(1, 3)])
        cf_idx = V.shape[0] - 1
        cap_tris = []
        for i in range(n):
            i1 = 1 + i
            j1 = 1 + ((i + 1) % n)
            cap_tris.append([i1, j1, cf_idx])  # 方向與上方翻轉一致，法向朝外



    # --- 6) 建立 O3D Mesh 並做最小清理 ---
        F = np.asarray(wall_tris + cap_tris, dtype=np.int32)
        lmesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V),
            triangles=o3d.utility.Vector3iVector(F)
        )
        lmesh.remove_degenerate_triangles()
        lmesh.remove_unreferenced_vertices()
        lmesh.compute_vertex_normals()

        return VoxelGrid.from_mesh(lmesh,voxel_size=voxel_size, fill=fill)
    
    
    
    @staticmethod
    def create_frustum(Cw: np.ndarray, voxel_size: float, near_ring: np.ndarray, far_ring: np.ndarray, fill: bool = True) -> "VoxelGrid":
        """
        生成一個錐台（Frustum），包含：
        - 側壁：由近平面和遠平面的邊形成的三角形帶
        - 遠平面（底面）：遠平面多邊形的扇形三角化
        - 近平面（頂面）：近平面多邊形的扇形三角化

        參數：
        - Cw: np.ndarray, 錐台的頂點（相機中心）
        - near_ring: np.ndarray, 近平面的多邊形頂點
        - far_ring: np.ndarray, 遠平面的多邊形頂點
        - voxel_size: float, 體素大小
        - fill: bool, 是否填充體素內部

        回傳：
        - VoxelGrid: 包含錐台的體素網格
        """
        Cw = np.asarray(Cw, dtype=np.float64).reshape(3)
        near_ring = np.asarray(near_ring, dtype=np.float64)
        far_ring = np.asarray(far_ring, dtype=np.float64)
        n = len(near_ring)
        if n < 3 or len(far_ring) != n:
            # time.sleep(5)
            # raise ValueError(f"近平面和遠平面必須有相同數量的頂點，且至少包含 3 個頂點{(n,len(far_ring)),far_ring}")
            print(f"近平面和遠平面必須有相同數量的頂點，且至少包含 3 個頂點{(n,len(far_ring)),far_ring}")
            return VoxelGrid.from_voxel_size(voxel_size=voxel_size)

        # --- 1) 確保遠平面多邊形繞行方向（法向朝外） ---
        cf = far_ring.mean(axis=0)
        dir_far = cf - Cw
        nrm_far = np.cross(far_ring[1] - far_ring[0], far_ring[2] - far_ring[0])
        if np.dot(nrm_far, dir_far) < 0.0:
            far_ring = far_ring[::-1].copy()
            near_ring = near_ring[::-1].copy()

        # --- 2) 組裝頂點 ---
        V = np.vstack([Cw.reshape(1, 3), near_ring, far_ring])  # 頂點順序：Cw, 近平面, 遠平面
        near_start_idx = 1
        far_start_idx = 1 + n

        # --- 3) 側壁三角形 ---
        wall_tris = []
        for i in range(n):
            i1 = near_start_idx + i
            j1 = near_start_idx + ((i + 1) % n)
            i2 = far_start_idx + i
            j2 = far_start_idx + ((i + 1) % n)
            wall_tris.append([i1, j1, j2])  # 三角形 [near[i], near[i+1], far[i+1]]
            wall_tris.append([i1, j2, i2])  # 三角形 [near[i], far[i+1], far[i]]

        # --- 4) 遠平面（扇形三角化）---
        cf_idx = V.shape[0]
        V = np.vstack([V, cf.reshape(1, 3)])  # 新增遠平面質心
        cap_tris = []
        for i in range(n):
            i1 = far_start_idx + i
            j1 = far_start_idx + ((i + 1) % n)
            cap_tris.append([i1, j1, cf_idx])  # 三角形 [far[i], far[i+1], cf]

        # --- 5) 近平面（扇形三角化）---
        cn = near_ring.mean(axis=0)
        cn_idx = V.shape[0]
        V = np.vstack([V, cn.reshape(1, 3)])  # 新增近平面質心
        near_cap_tris = []
        for i in range(n):
            i1 = near_start_idx + i
            j1 = near_start_idx + ((i + 1) % n)
            near_cap_tris.append([i1, j1, cn_idx])  # 三角形 [near[i], near[i+1], cn]

        # --- 6) 建立 O3D Mesh 並做最小清理 ---
        F = np.asarray(wall_tris + cap_tris + near_cap_tris, dtype=np.int32)
        lmesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V),
            triangles=o3d.utility.Vector3iVector(F)
        )
        lmesh.remove_degenerate_triangles()
        lmesh.remove_unreferenced_vertices()
        lmesh.compute_vertex_normals()

        return VoxelGrid.from_mesh(lmesh, voxel_size=voxel_size, fill=fill)
    
    
    
    
    
    
    def cut_by_plane(self, plane: np.ndarray, keep_positive: bool = False) -> "VoxelGrid":
        """
        以平面裁切體素網格，只保留平面一側的體素。
        plane: (4,) array，格式為 [a, b, c, d]，表示 ax+by+cz+d=0
        keep_positive: True=保留法向外側，False=保留法向內側
        回傳新的 VoxelGrid
        """
        # 取得所有體素中心座標
        volume = self.volume
        shape = volume.shape
        origin = self.origin
        voxel_size = self.voxel_size

        idxs = np.indices(shape).reshape(3, -1).T  # (N,3)
        centers = origin + (idxs + 0.5) * voxel_size  # (N,3)

        # 計算每個體素中心到平面的符號
        a, b, c, d = plane
        dists = centers @ np.array([a, b, c]) + d  # (N,)

        mask = dists >= 0 if keep_positive else dists <= 0
        mask = mask & volume.flatten()

        new_volume = np.zeros_like(volume)
        new_volume.flat[mask] = True

        return self._create_result_voxel(new_volume, origin)

