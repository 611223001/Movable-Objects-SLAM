
import cv2
import numpy as np
from enum import Enum
from collections import defaultdict 

from .pose import triangulate,computeF12

from .point import Point
from .frame import KeyFrame,Frame
from .matcher import MapPointMatcher, KeypointMatcher

from .utils import add_ones,computeE21,compute_epipolar_error,compute_epipolar_errors,check_epipolar, show_epilines

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .slam import Slam

    


class LocalMapper():
    
    def __init__(self,slam:'Slam'):
        
        self.slam =slam
        self.map = slam.map
        self.camera = slam.camera
        self.feature_tool = slam.feature_tool
        self.vo =  slam.vo
        self.map_display =  slam.map_display
        self.state_display =  slam.state_display
        self.control =  slam.control
        
        self.tracker = slam.tracker
        
        
        

    def creat_good_points(self,kf_cur:'KeyFrame',kfs_ret:list[KeyFrame]):
        kf1 = kf_cur
        
        kfs_ret_info:dict[KeyFrame] = dict()
        for kf_ret in kfs_ret:
            
            # kp_idxs_cur, kp_idxs_ret, match_pairs = KeypointMatcher.search_for_triangulation(kf_cur,kf_ret,
            #                                                                  check_orientation=False)
            K = kf_cur.camera.K
            # kp_idxs_cur, kp_idxs_ret,ret, E, match_success  = self.vo.match_points(kf_cur,kf_ret)
            
            kp_idxs_cur, kp_idxs_ret, ret = KeypointMatcher.flannMatches(kf_cur, kf_ret)
        
                
            kps1 = kf_cur.kps[kp_idxs_cur]
            kps2 = kf_ret.kps[kp_idxs_ret]
            
            E21 = computeE21(kf_cur, kf_ret)
            # mask = check_epipolar_constraint_with_mask(kps1,kps2,E21, threshold=0.02)
            errors_E21 = compute_epipolar_errors(kps1, kps2, E21)
            
           
            
            
            # print('極約束',len(mask),np.sum(mask0),np.sum(mask),np.sum(mask2))
            # print("極線約束誤差 E  ：", np.mean(errors_E))
            # show_epilines(kf_cur.img, kf_ret.img,kf_cur.raw_kps[kp_idxs_cur],kf_ret.raw_kps[kp_idxs_ret],E  ,K)
            # print("極線約束誤差 E21：", np.mean(errors_E21))
            # show_epilines(kf_cur.img, kf_ret.img,kf_cur.raw_kps[kp_idxs_cur],kf_ret.raw_kps[kp_idxs_ret],E21,K)
            # print("極線約束誤差 E12：", np.mean(errors_E12))
            # show_epilines(kf_cur.img, kf_ret.img,kf_cur.raw_kps[kp_idxs_cur],kf_ret.raw_kps[kp_idxs_ret],E12,K)

            
            if len(ret)>8:
                # 分成兩步驟，篩選合格的特徵點配對、三角測量產生3D點
                kfs_ret_info[kf_ret] = (kp_idxs_cur, kp_idxs_ret, errors_E21)
        
        good_matchs_info = defaultdict(dict)# dictionary of matches  [kp_idx_cur] -> list[kf_ret,kp_idx_ret]
        
        # 產生point位置預測，檢查合理性
        # 輸出 good_matchs_info ，kp_idx1 -> dict[kf2,(kp_idx2,pt4d)]
        for kf2, (kp_idxs1, kp_idxs2, errors_E21) in kfs_ret_info.items():
            pts4d = triangulate(kf1.pose, kf2.pose, kf1.kps[kp_idxs1], kf2.kps[kp_idxs2]) # np.array: shape(N, 4)   N*[𝑋, 𝑌, 𝑍, 𝑊]
            
            # 檢查第四座標 W 的絕對值是否大於 0.005。 避免太接近零（除W會出現極大值）
            good_pts4d = (np.abs(pts4d[:, 3]) > 0.005)
            pts4d /= pts4d[:, 3:] # 除W
            
            visible1,  pts_proj1 = kf1.are_visible(pts4d)
            visible2,  pts_proj2 = kf2.are_visible(pts4d)
            
            
            
            
            
            good_pts4d = good_pts4d & visible1 & visible2
            
            for i, (pt4d, kp_idx1, kp_idx2) in enumerate(zip(pts4d,kp_idxs1,kp_idxs2)):
                # kp_idx1:int = kp_idxs1[i]
                # kp_idx2:int = kp_idxs2[i]
                
                # check epipolar
                epipolar_err = errors_E21[i]
                epipolar_check = check_epipolar(kf2,kp_idx2,epipolar_err)
                
                if good_pts4d[i] and kf1.points[kp_idx1] is None and epipolar_check: # good 且 kp位置沒有對應point
                    good_matchs_info[kp_idx1][kf2]=(kp_idx2,pt4d) 
        
        
        # 對 kf_cur(kf1) 的每一有 match 的 kp(kf2) 迭代
        point_counter = 0
        for kp_idx1, info_list in good_matchs_info.items(): 
            
            kfs_ret = list(info_list.keys()) # 所有批配成功且合理的kf_ret
            
            # 將kp(kf2)的每個預測位置投影到其他kf(kf3)上，檢查預測是否合理
            err_means = []
            visible_table = np.eye(len(kfs_ret),dtype=np.bool_)
            
            for i, (kf2,(kp_idx2,pt4d)) in enumerate(info_list.items()):
                
                for j,kf3 in enumerate(kfs_ret):
                    kf3:KeyFrame
                    if kf3 is kf2:continue
                    visible,  pt_proj= kf3.are_visible(pt4d[np.newaxis,:]) # [:,np.newaxis] 擴張維度
                    visible_table[i][j]=visible
                    
            
            # 取得 visible_conter 最大值的索引(可能不只一個)
            visible_conter = np.sum(visible_table,1)
            if visible_conter.max() <3 :continue # 至少有3個kf_ret觀測保證準確
            best_visible_indices = np.where(visible_conter == visible_conter.max())[0]
            
            
            err_means={}
            for i in best_visible_indices:
                kf2,(kp_idx2,pt4d) = list(info_list.items())[i]
                visible = visible_table[i]
                
                errs=[]
                for j,kf3 in enumerate(kfs_ret):
                    if kf3 is kf2:continue
                    if not visible[j]: continue # 所有 kf2:pt4d 的 visible kf3 數量一樣
                    kp_idx3 = info_list[kf3][0]
                    # pt_proj, _ = kf3.project_point(pt4d)[:2]
                    # errs.append(np.linalg.norm(kf3.kps[kp_idx3]-pt_proj[:2]))
                    pt_proj,_ = kf3.project_point_to_img(pt4d)
                    errs.append(np.linalg.norm(kf3.raw_kps[kp_idx3]-pt_proj))
                
                if any([e is None or np.isnan(e) or np.isinf(e) for e in errs]) or not errs:
                    print(f"[Warning] 異常 errs: {errs}，kp_idx1={kp_idx1}")
                    err_means[i] = 100
                    continue

                err_means[i] = np.mean(errs)
            
            
            
            # for i, (kf2,(kp_idx2,pt4d)) in enumerate(info_list.items()):
            #     errs=[]
            #     for j,kf3 in enumerate(kfs_ret):
            #         kf3:KeyFrame
            #         if kf3 is kf2:continue
            #         kp_idx3 = info_list[kf3][0]
            #         pt_proj = kf3.project_point(pt4d)
            #         if pt_proj[2]>0:
            #             errs = np.linalg.norm(kf3.kps[kp_idx3]-pt_proj[:2])
            #         else:
            #             errs = 100
            #         errs.append(errs)
            #     err_means = np.mean(errs)
            
            
                
                
            
            # 取得最好預測，建立新 map point
            sorted_items = sorted(err_means.items(), key=lambda x: x[1])
            i_best, err_mean = sorted_items[0]
            infos = list(info_list.items())
            kf2,(kp_idx2,pt4d) = infos[i_best]
            
            if err_mean>1.5:continue
            # print('err_mean',err_mean)
            
            kf2:KeyFrame
            if kf2.points[kp_idx2]is None:
                # 建立新 point
                x, y = kf1.raw_kps[kp_idx1]
                color= kf1.img[y][x][::-1] # [::-1] 會將陣列反轉順序，BGR 格式轉成 RGB
                pt = Point(self.slam, pt4d,color, frame=kf1, idx=kp_idx1)
                pt.add_frame(kf2,kp_idx2)
                # self.map.add_point_frame_relation(pt,kf2,kp_idx2)
                
                point_counter +=1
            else:
                
                # 使用追蹤中的 point
                pt = kf2.points[kp_idx2]
                pt:Point
                pt.add_frame(kf1,kp_idx1)
                
            for i, err_mean in sorted_items[1:]:
                if err_mean < 1.5:
                    kf3,(kp_idx3,pt4d) = infos[i]
                    pt.add_frame(kf3,kp_idx3)
                    
                
            
        
        return point_counter
            
        # TODO: check epipolar constraint
        # TODO: 距離測試：檢查預測點的距離是否在kp屬於的物件距離範圍內(可略超過範圍)
        
        
    def creat_points(self,kf_cur:'KeyFrame', kfs_ret:list[KeyFrame]):
        kf1 = kf_cur
        ratio_factor = 1.5*kf1.get_scale_factor(1)
        point_counter = 0
        
        for kf2 in kfs_ret:
            # TODO:修改尋找 kp對 的方法
            kp_idxs1, kp_idxs2, ret = KeypointMatcher.flannMatches(kf1, kf2)
            
            if len(ret)<=8:
                continue
            
            kps1 = kf1.kps[kp_idxs1]
            kps2 = kf2.kps[kp_idxs2]
            
            E21 = computeE21(kf1, kf2)
            # mask = check_epipolar_constraint_with_mask(kps1,kps2,E21, threshold=0.02)
            errors_E21 = compute_epipolar_errors(kps1, kps2, E21)


            
            if len(ret)<=8:
                continue
            
            
            pts4d = triangulate(kf1.pose, kf2.pose, kf1.kps[kp_idxs1], kf2.kps[kp_idxs2]) # np.array: shape(N, 4)   N*[𝑋, 𝑌, 𝑍, 𝑊]
            
            # 檢查第四座標 W 的絕對值是否大於 0.005。 避免太接近零（除W會出現極大值）
            good_pts4d = (np.abs(pts4d[:, 3]) > 0.005)
            pts4d /= pts4d[:, 3:] # 除W
            
            visible1,  pts_proj1 = kf1.are_visible(pts4d)
            visible2,  pts_proj2 = kf2.are_visible(pts4d)
            
            good_pts4d = good_pts4d & visible1 & visible2
            
            for i, (pt4d, kp_idx1, kp_idx2) in enumerate(zip(pts4d,kp_idxs1,kp_idxs2)):
                # kp_idx1:int = kp_idxs1[i]
                # kp_idx2:int = kp_idxs2[i]
                
                if kf2.points[kp_idx2]is not None and kf1.points[kp_idx1] is not None: continue # kp位置沒有對應point
                if not good_pts4d[i]:continue # good
                
                
                # check epipolar
                epipolar_err = errors_E21[i]
                epipolar_check = check_epipolar(kf2,kp_idx2,epipolar_err)
                if not epipolar_check: continue
                
                
                # 檢查視差（Parallax）
                dist1 = np.linalg.norm(pt4d[:3]-kf1.Ow)
                dist2 = np.linalg.norm(pt4d[:3]-kf2.Ow)
                if dist1==0 or dist2==0: continue
                
                ray1 = (pt4d[:3]-kf1.Ow)/dist1
                ray2 = (pt4d[:3]-kf2.Ow)/dist2
                cos_parallax = np.dot(ray1,ray2)
                # print('cos_parallax',cos_parallax)
                if(cos_parallax<0 or cos_parallax>0.9998):continue
                
                
                # 檢查重投影誤差
                (up,vp) ,_ =kf1.project_point_to_img(pt4d)
                (ui, vi) = kf1.un_kps[kp_idx1]
                proj_err1 = (up-ui)**2 + (vp-vi)**2 # 誤差平方
                sigma2 = kf1.get_sigma2(kf1.octaves[kp_idx1])
                # print('proj_err1',proj_err1)
                if proj_err1 > 5.991*sigma2: continue # sigma 平方
                
                (up,vp) ,_ =kf2.project_point_to_img(pt4d)
                (ui, vi) = kf2.un_kps[kp_idx2]
                proj_err2 = (up-ui)**2 + (vp-vi)**2 # 誤差平方
                sigma2 = kf2.get_sigma2(kf2.octaves[kp_idx2])
                # print('proj_err2',proj_err2)
                if proj_err2 > 5.991*sigma2: continue # sigma 平方
                
                
                # 檢查尺度一致性
                ratio_dist = dist1/dist2
                ratio_octave = kf1.get_scale_factor(kf1.octaves[kp_idx1])/kf2.get_scale_factor(kf2.octaves[kp_idx2])
                # if ratio_dist*ratio_factor<ratio_octave or ratio_dist>ratio_octave*ratio_factor: continue
                if not (ratio_dist*ratio_factor >= ratio_octave and ratio_dist <= ratio_octave*ratio_factor): continue
                
                
                
                # 建立新 point
                x, y = kf1.raw_kps[kp_idx1]
                color= kf1.img[y][x][::-1] # [::-1] 會將陣列反轉順序，BGR 格式轉成 RGB
                pt = Point(self.slam, pt4d,color, frame=kf1, idx=kp_idx1)
                pt.add_frame(kf2,kp_idx2)
                                
                kf1.set_track_method(kp_idx1,1)
                point_counter +=1
                
                
        
        return point_counter
        
        
    
    def show_step(self,node:str):
        # 顯示現在階段
        self.state_display.set_process_state(2,node)
        
    def local_map(self,f_cur:Frame):
        # print('kf.point_num 0',f_cur.points_num)

        
        
        # if np.sum(self.f_cur.points != None) == 0:
        #     self.vo.relocalization(self.f_cur)
        
        # creat new kf 
        
        kf_cur = KeyFrame(f_cur) # keyframe 是一種 frame
        # 更新共視圖
        self.map.update_covisibility_graph()

        # cull points
        self.show_step("cull_pts")
        
        num_culled_pt = self.map.cull_points() # 標記 point 為 bad
        self.state_display.set_info('cull_pt',num_culled_pt)
        clear_pt_num = self.map.clear_bad_points(kf_cur)
        print('cull_points  clear_pt_num:',clear_pt_num)
        
        
        
        # creat new point based on nearby kf
        self.show_step("creat_pts")
        local_kfs = self.map.get_covisible_keyframes(kf_cur,include_input=False,kfs_num=20)
        
        # print('kf.point_num 1',kf_cur.points_num)
        
        # 新增地圖點
        # 從local kf 批配 kps
        
        kfs_ret:list[KeyFrame] = []
        for kf_ret in local_kfs:
            # 拒絕沒有足夠 視差 的幀
            medianDepth = kf_ret.compute_points_median_depth()
            baseline = np.linalg.norm(kf_cur.Ow-kf_ret.Ow)
            ratioBaselineDepth = baseline/medianDepth # 最小視差根據f與場景的距離改變
            # print('ratioBaselineDepth',ratioBaselineDepth)
            if ratioBaselineDepth < 0.05:
                continue
            
            kfs_ret.append(kf_ret)
            
                
        # 使用多個相鄰kf資訊，提高新建立points的可靠性
        new_pt_num = self.creat_points(kf_cur,kfs_ret)
        
        # new_pt_num = self.create_new_map_points(kf_cur,local_kfs)
        
        self.state_display.set_info('creat_pts_num',new_pt_num)
        
        # 點融合
        self.show_step("fuse")
        fused_to_nebr, fused_to_cur = self.search_in_neighbors(kf_cur,local_kfs)
        self.state_display.set_info('fuse',(fused_to_nebr, fused_to_cur))
        # self.map.check_points_frames_consistency()
        
        
        
        # print('kf.point_num 3',kf_cur.points_num)
        self.show_step("local_BA")
        # 調整現在 keyframe 相關的 kf 及 point
        err = self.vo.local_bundle_adjustment(local_kfs+[kf_cur])
        self.state_display.set_info('lBA',err)
        
        
        err = self.vo.pose_optimize(kf_cur)
        num_clear = kf_cur.clear_outliers()
        
        
        
        # print('kf.point_num 4',kf_cur.points_num)
        
        clear_pt_num = self.map.clear_bad_points(kf_cur)
        print('fuse  clear_pt_num:',clear_pt_num)
        # for pt in kf_cur.points:
        #     if pt is None:continue
        #     assert not pt.is_bad
        
        num_culled_kf = self.map.cull_keyframes(local_kfs)
        # if num_del_kf >0:
        self.map.clear_bad_keyframes(kf_cur)
        self.map.update_covisibility_graph() # 刪除kf後要更新共識圖
        
        
        # err = self.vo.local_bundle_adjustment(local_kfs)
        
        self.state_display.set_info('pt_num',len(self.map.points))
        self.state_display.set_info('kf_num',len(self.map.keyframes))
        
        
        self.show_step("obj_pre")
        # 檢查物體移動 TODO:在BA 前 檢查並剃除動態的觀察
        self.slam.obj_tool.check_object_dynamic(kf_cur)
        
        
        
        # 使用投影重新追蹤物體
        local_things = self.map.get_covisible_things(kf_cur,20)
        # self.slam.obj_tool.retrack_objects_by_project(kf_cur,local_things) # 目前只支援things
        for thing in local_things:
            mask, missed, bPrj = self.slam.obj_tool.obj_tracker.get_info(thing)
            
            # 如果不在追蹤清單或丟失追蹤太久，建立hull並使用投影追蹤
            if mask is None or (missed >10):
                
                # thing.make_hull()
                
                # self.slam.obj_tool.retrack_objects_by_project(kf_cur,[thing]) # TODO: 修改成一個只投影單一obj的函式
                
                mask, missed, bPrj = self.slam.obj_tool.obj_tracker.get_info(thing)
                print(f'obj:{thing.oid} mask:{np.sum(mask)} missed:{missed} project:{bPrj}')
        
        self.map.clear_bad_object()
        
        
        nclean = self.slam.obj_tool.obj_tracker.clean_old_lost(self.camera.fps*2)
        print('clean_old_lost',nclean)
        self.slam.obj_tool.predict_object(kf_cur)
        
        # 基於 TSDF 的方法
        kf_flows = calc_flows(kf_cur,local_kfs)
        for obj in kf_cur.objects:
            if obj is None:continue
            obj.update_points_by_frame(kf_cur) # 更新obj.points
            
            if obj.is_thing:
                if obj.need_to_static:
                    print('re recover_object',obj.oid)
                    success = obj.static_update(kf_cur)
                    print(f're recover success:{success}')
                else:
                    
                    obj.make_mesh(kf_cur, local_kfs, kf_flows)
            
            
            # 基於體素雕刻的方法
            # if obj.is_thing:
            #     # TODO: 檢查投影與kf_cur 的 IoU 確認差異很大才切割
            #     # TODO:檢查消耗時間
            #     if obj.voxel_grid is not None:
                    
            #         obj.carve_hull(kf_cur)
        
        
        
        
       
        # for obj in kf_cur.objects:
        #     if obj is None:continue
        #     obj.update_points_by_frame(kf_cur) # 更新obj.points
        #     if obj.is_thing:
        #         # TODO:檢查消耗時間
        #         if obj.voxel_grid is  None:
        #             print('make hull',obj.oid)
        #             obj.make_hull()
                    
        #         else:
        #             print('carve',obj.oid)
        #             obj.carve_hull(kf_cur)
        
        
        # print('kf.point_num 5',kf_cur.points_num)
        
        # global mapping
        if kf_cur.kid%5 == 0:
            self.show_step("gobal_BA")
            # BA調整所有 keyframe 及 point
            err = self.vo.global_bundle_adjustment()
            self.state_display.set_info('gBA',err)
            
            # for obj in kf_cur.objects:
            #     if obj is None:continue
                
            #     if obj.is_thing and (obj.dynamic is False) and  not obj.need_to_static:
            #         obj.update_pose(kf_cur)
            
            # if kf_cur.is_clear:
            #     for obj in kf_cur.objects:
            #         if obj is not None and obj.is_thing:
            #             obj.make_hull()
        
        
        
        # print('kf.point_num 6',kf_cur.points_num)
        self.show_step(0)# 輸入無效值清空

        return kf_cur
    
    
    
    def search_in_neighbors(self, kf_cur:'KeyFrame',local_kfs:list['KeyFrame']):
        neighbor_kfs:set[KeyFrame]=set()
        for kf in local_kfs:
            if kf.is_bad:continue
            neighbor_kfs.add(kf)
            
            # 取得二階共視 KeyFrame
            second_neighbors = self.map.get_covisible_keyframes(kf, kfs_num=5, include_input=False)
            for kf2 in second_neighbors:
                if kf2.is_bad:continue
                neighbor_kfs.add(kf2)
        
        
        points_cur = [pt for pt in kf_cur.points if pt is not None]
        fused_to_nebr = 0
        for kf in neighbor_kfs:
            if kf is kf_cur:continue
            fused_to_nebr += self.map.fuse_points(kf, points_cur)
        
        
        points_ner:set[Point] = set()
        for kf in neighbor_kfs:
            if kf is kf_cur:continue
            points_ner.update(kf.points)
        # 剃除已經在kf_cur的points
        points_ner.difference_update(kf_cur.points)
        points_ner.discard(None)
        fused_to_cur = self.map.fuse_points(kf_cur, list(points_ner))
        
        
        # 更新地圖點描述子與法線
        for pt in kf_cur.points:
            if pt is not None and not pt.is_bad:
                pt:Point
                pt.update_descriptor()
                pt.update_normal()
                pt.update_depth()
        
        return fused_to_nebr, fused_to_cur
        
    
    
    
    '''
    def local_map_orb(self,f_cur:Frame):
        
        
        # creat new kf 
        self.show_step("add_kf")
        kf_cur = KeyFrame(f_cur) # keyframe 是一種 frame
        # 更新共視圖
        self.map.update_covisibility_graph()

        # cull points
        self.show_step("cull_pts")
        num_culled_pt = self.map.cull_points() # 標記 point 為 bad
        self.state_display.set_info('cull_pt',num_culled_pt)
        self.map.clear_bad_points(kf_cur)
        
        
        
        
        
        # creat new point based on nearby kf
        self.show_step("creat_pts")
        local_kfs = self.map.get_covisible_keyframes(kf_cur,include_input=False,kfs_num=20)
        
        
        # 新增地圖點
        # 從local kf 批配 kps
        new_pt_num = self.create_new_map_points(kf_cur,local_kfs)
        
        self.state_display.set_info('creat_pts_num',new_pt_num)
        
        
        # 點融合
        print('search_in_neighbors')
        self.search_in_neighbors(kf_cur,local_kfs)
        # num_fused = self.map.fuse_points(kf_cur,local_pts)
        # print(f"num_fused:{num_fused}")
        self.map.check_points_frames_consistency()
        
        # 更新共視圖連結
        self.map.update_covisibility_graph()
        local_kfs = self.map.get_covisible_keyframes(kf_cur,include_input=True)
        
        # print('kf.point_num 3',kf_cur.points_num)
        self.show_step("local_BA")
        # 調整現在 keyframe 相關的 kf 及 point
        err = self.vo.local_bundle_adjustment(local_kfs)
        self.state_display.set_info('lBA',err)
        

        
        
        self.map.clear_bad_object()
        num_culled_kf = self.map.cull_keyframes(local_kfs)
        num_del_kf = self.map.clear_bad_keyframes(kf_cur)
        if num_del_kf >0:
            self.map.update_covisibility_graph() # 刪除kf後要更新共識圖
        
        
        # err = self.vo.local_bundle_adjustment(local_kfs)
        
        self.state_display.set_info('pt_num',len(self.map.points))
        self.state_display.set_info('kf_num',len(self.map.keyframes))
        
        
        self.show_step("obj_pre")
        self.slam.obj_tool.predict_object(kf_cur)
        
        # 暫時關閉
        # for obj in kf_cur.objects:
        #     if obj is not None and obj.is_thing:
        #         # TODO:檢查消耗時間
        #         if obj.voxel_grid is  None:
        #             print('obj.voxel_grid',obj.oid,obj.voxel_grid)
        #             obj.make_hull()
        #         else:
        #             print('carve',obj.oid)
        #             obj.carve_hull(kf_cur)
        
        
        # global mapping
        if kf_cur.kid%5 == 0:
            self.show_step("gobal_BA")
            # BA調整所有 keyframe 及 point
            err = self.vo.global_bundle_adjustment()
            self.state_display.set_info('gBA',err)
            
            # 暫時關閉
            # for obj in kf_cur.objects:
            #     if obj is not None and obj.is_thing:
            #         obj.make_hull()
        
        
        
        # print('kf.point_num 6',kf_cur.points_num)
        self.show_step(0)# 輸入無效值清空

        return kf_cur
    '''
    
    
    def create_new_map_points(self, kf_cur:'KeyFrame',neighbor_kfs:list['KeyFrame']):
        """
        仿照 ORB-SLAM LocalMapping::CreateNewMapPoints
        """
        neighbor_kfs = self.map.get_covisible_keyframes(kf_cur, kfs_num=20, include_input=False)
        point_counter = 0

        for kf2 in neighbor_kfs:
            # 檢查基線與深度比例
            baseline = np.linalg.norm(kf2.Ow - kf_cur.Ow)
            median_depth = kf2.compute_points_median_depth()
            if median_depth <= 0:
                continue
            ratio_baseline_depth = baseline / median_depth
            if ratio_baseline_depth < 0.01:
                continue

            # 計算基本矩陣
            F12 = computeF12(kf_cur, kf2)

            # 用極線約束配對特徵點
            kp_idxs1, kp_idxs2, match_pairs = KeypointMatcher.search_for_triangulation(
                kf_cur, kf2, F12, distance_threshold=50, histo_length=30, check_orientation=True
            )
            if len(kp_idxs1) == 0:
                continue

            # 三角化
            pts4d = triangulate(kf_cur.pose, kf2.pose, kf_cur.kps[kp_idxs1], kf2.kps[kp_idxs2])  # (N,4)
            good_pts = (np.abs(pts4d[:, 3]) > 0.005)
            pts4d /= pts4d[:, 3:]

            # 可見性檢查
            visible1, _ = kf_cur.are_visible(pts4d)
            visible2, _ = kf2.are_visible(pts4d)
            good_pts = good_pts & visible1 & visible2

            for i, (pt4d, idx1, idx2) in enumerate(zip(pts4d, kp_idxs1, kp_idxs2)):
                if not good_pts[i]:
                    continue

                # 檢查視差
                ray1 = kf_cur.Rwc @ np.append(kf_cur.kps[idx1], 1.0)
                ray2 = kf2.Rwc @ np.append(kf2.kps[idx2], 1.0)
                cos_parallax = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
                if cos_parallax < 0 or cos_parallax > 0.9998:
                    continue

                # 檢查重投影誤差
                proj1, _ = kf_cur.project_point(pt4d)
                proj2, _ = kf2.project_point(pt4d)
                err1 = np.linalg.norm(proj1[:2] - kf_cur.kps[idx1])
                err2 = np.linalg.norm(proj2[:2] - kf2.kps[idx2])
                if err1 > 3 or err2 > 3:
                    continue

                # 檢查尺度一致性
                dist1 = np.linalg.norm(pt4d[:3] - kf_cur.Ow)
                dist2 = np.linalg.norm(pt4d[:3] - kf2.Ow)
                if dist1 == 0 or dist2 == 0:
                    continue
                ratio_dist = dist1 / dist2
                ratio_octave = 1.0  # 若有金字塔尺度可用 kf_cur.get_scale_factor(octave) / kf2.get_scale_factor(octave)
                ratio_factor = 1.5
                if ratio_dist * ratio_factor < ratio_octave or ratio_dist > ratio_octave * ratio_factor:
                    continue

                # 建立新地圖點
                if kf_cur.points[idx1] is None and kf2.points[idx2] is None:
                    x, y = kf_cur.raw_kps[idx1]
                    color = kf_cur.img[y, x][::-1]
                    pt = Point(self.slam, pt4d, color, frame=kf_cur, idx=idx1)
                    pt.add_frame(kf2, idx2)
                    point_counter += 1

        return point_counter
    
    
        
        

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

        
    
    
    
    
    
    """

    def predict_points(self,f1:'Frame',f2:'Frame',kp_idxs1,kp_idxs2):

        point_counter = 0
        
        
        # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
        #predict pose
        pts4d = triangulate(f1.pose, f2.pose, f1.kps[kp_idxs1], f2.kps[kp_idxs2])
        
        # This line normalizes the 3D points by dividing each row by its fourth coordinate W
        # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates


        # Reject points without enough "Parallax" and points behind the camera
        # checks if the absolute value of the fourth coordinate W is greater than 0.005.
        # checks if the z-coordinate of the points is positive.
        # returns, A boolean array indicating which points satisfy both criteria.
        # 檢查第四座標 W 的絕對值是否大於 0.005。 避免太接近零（除W會出現極大值）
        # 返回，一個布林陣列，表示哪些點符合這兩個條件。
        good_pts4d = (np.abs(pts4d[:, 3]) > 0.005)
        pts4d /= pts4d[:, 3:]
        
        visible1,  pts_proj1 = f1.are_visible(pts4d)
        visible2,  pts_proj2 = f2.are_visible(pts4d)

        # print('good_pts4d 1',np.sum(good_pts4d))
        good_pts4d = good_pts4d & visible1 & visible2
        # print('good_pts4d 2',np.sum(good_pts4d))
        

        for i, (p_position, kp_idx1, kp_idx2) in enumerate(zip(pts4d,kp_idxs1,kp_idxs2)):

            #  If the point is not good (i.e., good_pts4d[i] is False), the loop skips the current iteration and moves to the next point.
            if not good_pts4d[i]:continue
            
            if f1.points[kp_idx1] is not None: continue
            
            
            # # check reprojection error
            # err1 = pts_proj1[i][:2] -f1.kps[kp_idx1]
            # err2 = pts_proj2[i][:2] -f2.kps[kp_idx2]
            # err1 =np.sum(err1**2)
            # err2 =np.sum(err2**2)
            # # print('err',err1,err2)
            # if (err1 >1) | (err2>1): continue # 閾值隨便設定， TODO:修改成一個有意義的值
            
            if f2.points[kp_idx2]is None:
                # 建立新 point
                x, y = f1.raw_kps[kp_idx1]
                color= f1.img[y][x][::-1] # [::-1] 會將陣列反轉順序，BGR 格式轉成 RGB
                pt = Point(f2.slam, p_position,color, frame=f2, idx=kp_idx2)
                self.map.add_point_frame_relation(pt,f1,kp_idx1)
            else:
                # 使用追蹤中的 point
                # pt = f2.points[kp_idx2]
                # self.map.add_point_frame_relation(pt,f1,kp_idx1)
                ...
                
            point_counter +=1
            
        return point_counter
    """
    
    