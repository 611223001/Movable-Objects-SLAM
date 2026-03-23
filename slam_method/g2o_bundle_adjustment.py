import numpy as np
import cv2
import g2o

from .point import Point
from .frame import Frame

# from https://github.com/uoip/g2opy

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        
        cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
        cam.set_id(0)
        super().add_parameter(cam)
        
        self._robust_kernels = []
        # 避免物件被 Python GC 回收
        # g2o 的 C++ 物件如果沒有被 Python 變數持有，可能會被垃圾回收，導致 segfault。
        # 例如 robust_kernel 物件要有變數持有，不要只在函式參數裡臨時建立。

    def optimize(self, max_iterations=10,level=0):
        super().initialize_optimization(level)
        super().optimize(max_iterations)
    
    
    def add_pose(self, pose_id, pose, fixed=False): # 通常原點 fixed = True 固定起始點        
        # pose_orientation = g2o.Quaternion(1, rotation_vector[0], rotation_vector[1], rotation_vector[2]) # w, x, y, z
        pose =np.linalg.inv(pose) # g2o pose 使用的是Tcw,所以要反矩陣
        se3 = g2o.SE3Quat(pose[:3,:3], pose[:3,3])
        # sbacam = g2o.SBACam(se3)
        # sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)
        # sbacam.set_cam(0.0, 0.0, 0.0, 0.0, 0.0)

        # v_se3 = g2o.VertexCam()
        # v_se3.set_id(pose_id * 2)   # internal id
        # v_se3.set_fixed(fixed)
        # v_se3.set_estimate(sbacam)
        
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 
        return v_se3

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        """marginalized: true 表示該點在解聯立系統時會被邊緣化（Schur complement）
        """
        
        point3D = point[:3] # (point4D or point3D) to point3D
        # v_p = g2o.VertexSBAPointXYZ()
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point3D)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)
        return v_p

    def add_edge(self, point_id, pose_id, 
            measurement,
            K,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI
        '''
            point_id、pose_id 必須已經存在，否則 edge 不會被添加或是optimize時出現錯誤。
        '''
        # assert self.vertex(point_id * 2 + 1) is not None, f"Point: {point_id} vertex not found"
        # assert self.vertex(pose_id * 2) is not None, f"Pose: {pose_id} vertex not found"
    
        # edge = g2o.EdgeProjectP2MC()
        # edge = g2o.EdgeProjectXYZ2UV()
        edge = g2o.EdgeSE3ProjectXYZ()
        
        edge.set_parameter_id(0, 0)
        
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        # if robust_kernel is not None:
        self._robust_kernels.append(robust_kernel)  # 保持參考
        edge.set_robust_kernel(robust_kernel)
        
        
        edge.fx = K[0,0]
        edge.fy = K[1,1]
        edge.cx = K[0,2]
        edge.cy = K[1,2]
        
        super().add_edge(edge)
        return edge

    def get_pose(self, pose_id):
        optimized_pose = self.vertex(pose_id * 2).estimate()
        pose = optimized_pose.matrix()
        # g2o pose 使用的是Tcw,所以要反矩陣回 Twc
        return np.linalg.inv(pose)

    def get_point(self, point_id):
        point4D = np.append(self.vertex(point_id * 2 + 1).estimate(),1) # 轉為齊次座標
        return point4D
    
    
    def is_depth_positive(self,edge)->bool:
        """
        用 edge 取得 frame, point 的位姿，並計算深度是否為正
        """
        # 取得 vertex
        v_point = edge.vertex(0)
        v_frame = edge.vertex(1)
        # 取得優化後的座標
        pt3d = v_point.estimate()  # shape: (3,)
        pose = v_frame.estimate().matrix()  # shape: (4,4) Tcw
        # 轉換到相機座標系
        pt_cam = pose[:3, :3] @ pt3d + pose[:3, 3]
        # 深度為 Z 軸
        return pt_cam[2] > 0
    
    
    def get_edge(self, point_id, pose_id):
        """
        edge.chi2()
        edge.level()
        """
        
        pid = point_id * 2 + 1
        fid = pose_id * 2
        
        for edge in self.edges():
            v0 = edge.vertex(0).id()
            v1 = edge.vertex(1).id()
            if (v0 == pid and v1 == fid) or (v0 == fid and v1 == pid):
                return edge
        return None
    
    
    
    
    def get_all_edges_chi2(self):
        """
        回傳所有 edge 的 (point_id, pose_id, chi2) 清單
        """
        result = []
        for edge in self.edges():
            v0 = edge.vertex(0).id()
            v1 = edge.vertex(1).id()
            chi2 = edge.chi2()
            result.append((v0, v1, chi2))
        return result
    
    def set_edge_level(self, point_id, pose_id, level):
        """
        設定 edge 的 level，level=0 參與所有最佳化，level=1 只參與粗略階段。
        """
        edge = self.get_edge(point_id, pose_id)
        if edge is not None:
            edge.set_level(level)





'''
def bundle_adjustment(frames:list[Frame],max_iterations=10):

    ba = BundleAdjustment()
    
    points:set['Point'] = set()
    # relation = {}
    
    for frame in frames:

        fixed = (frame.id == 0)
        
        ba.add_pose(frame.id,frame.pose,fixed=fixed) #固定起始Frame
        
        for point in frame.points:
            
            if point is not None:
                point:'Point'
                if len(point.frames)>1: # 只BA有2個以上觀察者的point
                    points.add(point)
                    
        
    for point in points:
        ba.add_point(point.id,point.location[:3])
        
        for frame in point.frames:
            assert  frame in frames
            kp_id = frame.find_point(point)

            ba.add_edge(point.id,frame.id,frame.kps[kp_id])
        
    # print(f"Number of vertices: {len(ba.vertices())}")
    # print(f"Number of edges: {len(ba.edges())}")
    
    ba.optimize(max_iterations=max_iterations)
        
    
    for frame in frames:
        frame.pose = ba.get_pose(frame.id)
    
    for point in points:
        point.location =ba.get_point(point.id)
        

    err = ba.active_chi2()
    return err

def local_bundle_adjustment(frames:list[Frame],local_frames:list[Frame],max_iterations=10):

    ba = BundleAdjustment()
    
    points:set['Point'] = set()
    
    for frame in frames:
        fixed = (frame.id == 0) | (frame not in local_frames)

        
        ba.add_pose(frame.id,frame.pose,fixed=fixed) #固定非local frame
        for kp_id,point in enumerate(frame.points):
            
            if point is not None:
                if len(point.frames)>2: # 只BA有三個以上觀察者的point
                    points.add(point)
        
    for point in points:
        ba.add_point(point.id,point.location[:3],fixed=False)
        for frame in point.frames:
            assert  frame in frames
            kp_id = frame.find_point(point)
            ba.add_edge(point.id,frame.id,frame.kps[kp_id])
        
        
    ba.optimize(max_iterations=max_iterations)
    
    frames_pose_optimized:list[tuple[int,any]] = [] # id, pose
    
    for frame in local_frames:
        frame.pose = ba.get_pose(frame.id)
    
    err = ba.active_chi2()
    return err
    


def pose_optimize(frames:list[Frame],max_iterations=30):
    # 只動輸入 frames 的 pose 相關 points 與 frames 固定
    ba = BundleAdjustment()
    
    points:set['Point'] = set()
    frames_ref:set['Frame'] = set()
    
    for frame in frames:
        ba.add_pose(frame.id,frame.pose,fixed=False) #固定非local frame
        for kp_id,point in enumerate(frame.points):
            
            if point is not None:
                if len(point.frames)>2: # 只BA有三個以上觀察者的point
                    points.add(point)
                    
                    # ba.add_edge(point.id,frame.id,frame.kps[kp_id])
    
    for point in points:
        
        for frame in point.frames:
            if frame in frames:
                ba.add_point(point.id,point.location[:3],fixed=True)
            else:
                ba.add_point(point.id,point.location[:3],fixed=True)
            kp_id = frame.find_point(point)
            ba.add_edge(point.id,frame.id,frame.kps[kp_id])
    #     frames_ref.update(point.frames)

    # for frame in frames_ref:
    #     if frame not in frames:
    #         ba.add_pose(frame.id,frame.pose,fixed=True)
        
    #     for kp_id,point in enumerate(frame.points):
    #         if point is not None:
    #             if point in points: 
                    
    #                 ba.add_edge(point.id,frame.id,frame.kps[kp_id])
    
    ba.optimize(max_iterations=max_iterations)
    
    for frame in frames:
        frame.pose = ba.get_pose(frame.id)

    err = ba.active_chi2()
    return err 


def points_optimize(frames:list[Frame],max_iterations=30):
    # 只動輸入 frames 的 pose 相關 points 與 frames 固定
    ba = BundleAdjustment()
    
    points:set['Point'] = set()
    frames_ref:set['Frame'] = set()
    
    for frame in frames:
        ba.add_pose(frame.id,frame.pose,fixed=True) #固定非local frame
        for kp_id,point in enumerate(frame.points):
            
            if point is not None:
                if len(point.frames)>2: # 只BA有三個以上觀察者的point
                    points.add(point)
                    
                    # ba.add_edge(point.id,frame.id,frame.kps[kp_id])
    
    i,j=0,0
    for point in points:
        ba.add_point(point.id,point.location[:3],fixed=False)
        for frame in point.frames:
            # if frame in frames:
            #     ba.add_point(point.id,point.location[:3],fixed=False)
            #     i +=1
            # else:
            #     ba.add_point(point.id,point.location[:3],fixed=True)
            #     j +=1
            
            kp_id = frame.find_point(point)
            ba.add_edge(point.id,frame.id,frame.kps[kp_id])
            
    print('unfix point',i,'fix point',j)
    print(f"Number of vertices: {len(ba.vertices())}")
    print(f"Number of edges: {len(ba.edges())}")

    # 若要分別統計 pose/point 數量，可這樣：
    pose_count = sum(1 for v in ba.vertices().values() if v.dimension() == 6)
    point_count = sum(1 for v in ba.vertices().values() if v.dimension() == 3)
    print(f"Pose vertices: {pose_count}, Point vertices: {point_count}")
    #     frames_ref.update(point.frames)

    # for frame in frames_ref:
    #     if frame not in frames:
    #         ba.add_pose(frame.id,frame.pose,fixed=True)
        
    #     for kp_id,point in enumerate(frame.points):
    #         if point is not None:
    #             if point in points: 
                    
    #                 ba.add_edge(point.id,frame.id,frame.kps[kp_id])
    
    ba.optimize(max_iterations=max_iterations)
    
    for frame in frames:
        frame.pose = ba.get_pose(frame.id)
        
    for point in points:
        point.location = ba.get_point(point.id)

    err = ba.active_chi2()
    return err 

'''

