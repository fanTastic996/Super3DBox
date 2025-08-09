import uuid
from dataclasses import dataclass
from typing import List, Dict, Set
from cubifyanything.instances import Instances3D
import numpy as np
import torch
import copy 


class BoxManager:
    """全局管理器（基于哈希表实现快速访问[2,5](@ref)）"""
    def __init__(self,cfg):
        # self.box_registry: Dict[str, Instances3D] = {}  # ID到对象的映射
        self.fusion_list = []  # 记录当前box由哪些per_frame_box fusion而来
        self.last_fusion_frame = [] # 记录当前box上一次fusion是什么哪一帧
        self.fusion_flag = []
        self.already_fusion = []
        self.num_record = {}
        self.cfg = cfg
        self.rotation_gap = self.cfg['association']['rotation_gap']
        self.translation_gap = self.cfg['association']['translation_gap']
        self.small_size = self.cfg['box_fusion']['small_size'] #0.35
        self.merge_log: List[Dict] = []           # 合并操作记录

    def init_new_predictions(self,box_num,all_num):
        for i in range(box_num):
            self.fusion_list.append([i+all_num])
            self.last_fusion_frame.append([0])
            self.fusion_flag.append(0)
    # def record(self, cur_id, fusion_inds):
    #     if len(self.fusion_list[cur_id]) == 1: #only the box itself
    #         self.fusion_list[cur_id] += fusion_inds
    #     else:
    #         self.fusion_list[cur_id] += fusion_inds

    def add_fusion_ind(self, idx_list):
        self.already_fusion.append(copy.deepcopy(idx_list))

    def check_if_fusion(self, idx_list):
        if idx_list in self.already_fusion:
            return True
        else:
            return False

    def record(self, cur_id, fusion_inds, init_id, cam_poses, box_size, keep, box_centers):
        '''
        Note: cur_id is consistent to 'all_pred_box', idx is according to 'per_frame_box'
        '''
        cur_box_size = box_size[cur_id,:3]
        small = False
        if np.max(cur_box_size)<self.small_size:
            small = True
        # print("fusion_inds",fusion_inds)
        for idx in fusion_inds:
            #old boxes nms new box
            if len(self.fusion_list[idx]) == 1:
                # print('cur id:',cur_id,'fusion id:',init_id[idx])
                #TODO：改成batch操作，加快判断
                '''
                baseline gap > 0.5m  | rotation gap > 30'
                '''
                #TODO：compute_pose_disparity: 应该设定一个矩阵固定放在那，算过了就不要再算了
                count = 0 
                for i in self.fusion_list[cur_id]: 
                    baseline_gap, rotation_gap, disparity_score, center_dis= self.compute_pose_center_disparity(cam_poses[i], cam_poses[init_id[idx]], box_centers[cur_id], box_centers[idx])
                    # if (baseline_gap > 0.1 or rotation_gap > 20) and not small:
                    #     count+=1
                    # if rotation_gap > 10 and small: #30
                    #     count+=1
                    # if rotation_gap > self.rotation_gap:
                    #     count+=1
                    # if (baseline_gap > self.translation_gap or rotation_gap > self.rotation_gap) and not small:
                    #     count+=1
                    # if rotation_gap > self.rotation_gap and small: #30
                    #     count+=1
                    if (baseline_gap > self.translation_gap or rotation_gap > self.rotation_gap) or center_dis>0.5:
                        count+=1



                # different from all the corresponding key boxes
                if count == len(self.fusion_list[cur_id]) and len(self.fusion_list[cur_id])<5:
                # if count >=0:
                    self.fusion_list[cur_id] += [init_id[idx]]
                    self.fusion_list[cur_id].sort()
                # print(self.fusion_list)

            #new box nms old boxes
            else:
                '''
                baseline gap > 0.5m  | rotation gap > 30'
                '''
                count = 0 
                for i in self.fusion_list[idx]: 
                    
                    baseline_gap, rotation_gap, disparity_score,center_dis = self.compute_pose_center_disparity(cam_poses[i], cam_poses[init_id[cur_id]], box_centers[cur_id], box_centers[idx])

                    if (baseline_gap > self.translation_gap or rotation_gap > self.rotation_gap) or center_dis>0.5:
                        count+=1

                    # if (baseline_gap > self.translation_gap or rotation_gap > self.rotation_gap) and not small:
                    #     count+=1
                    # if rotation_gap > self.rotation_gap and small: #30
                    #     count+=1
                    # if rotation_gap > self.rotation_gap:
                    #     count+=1
                    
                    # print('fusion list id',i,init_id[cur_id],"baseline_gap",baseline_gap,'rotation_gap',rotation_gap,'small',small,'count',count)

                # different from all the corresponding key boxes
                if count == len(self.fusion_list[idx]) and len(self.fusion_list[idx])<5:
                # if count >=0:
                    self.fusion_list[cur_id] += self.fusion_list[idx]
                    self.fusion_list[cur_id].sort()
                else:
                    if cur_id in keep:
                        print("extra remove","cur_id",cur_id,'add:',idx)
                        keep.remove(cur_id)
                        keep.append(idx)
                
                if self.fusion_flag[idx] == 1:
                    self.fusion_flag[cur_id] = 1

        return keep





    def record_corr(self, cur_id, fusion_inds, init_id, cam_poses, keep):
        '''
        Note: cur_id is consistent to 'all_pred_box', idx is according to 'per_frame_box'
        '''
        

        for idx in fusion_inds:
            #completely new boxes and no nms is valid
            if len(self.fusion_list[idx]) == 1:
                # print('cur id:',cur_id,'fusion id:',init_id[idx])
                #TODO：改成batch操作，加快判断
                '''
                baseline gap > 0.5m  | rotation gap > 30'
                '''
                count = 0 
                for i in self.fusion_list[cur_id]: 
                    baseline_gap, rotation_gap, disparity_score = self.compute_pose_disparity(cam_poses[i], cam_poses[init_id[idx]])
                    # if rotation_gap > self.rotation_gap: #10: #30
                    #     count+=1
                    if rotation_gap > self.rotation_gap or baseline_gap>self.translation_gap: #10: #30
                        count+=1
                # different from all the corresponding key boxes
                if count == len(self.fusion_list[cur_id]) and len(self.fusion_list[cur_id])<5:
                    self.fusion_list[cur_id] += [init_id[idx]]
                    self.fusion_list[cur_id].sort()
                # print(self.fusion_list)
            #new box nms old boxes
            else:
                '''
                baseline gap > 0.5m  | rotation gap > 30'
                '''
                count = 0 
                for i in self.fusion_list[idx]: 
                    baseline_gap, rotation_gap, disparity_score = self.compute_pose_disparity(cam_poses[i], cam_poses[init_id[cur_id]])
                    # if rotation_gap > self.rotation_gap: #10: #30
                    #     count+=1
                    if rotation_gap > self.rotation_gap or baseline_gap>self.translation_gap: #10: #30
                        count+=1
                # different from all the corresponding key boxes
                if count == len(self.fusion_list[idx]) and len(self.fusion_list[idx])<5:
                    self.fusion_list[cur_id] += self.fusion_list[idx]
                    self.fusion_list[cur_id].sort()
                else:
                    if cur_id in keep:
                        keep[keep==cur_id] = idx 


                if self.fusion_flag[idx] == 1:
                    self.fusion_flag[cur_id] = 1

        return keep
    
    def update(self, keep_idx):

        self.fusion_list = [self.fusion_list[i] for i in keep_idx] 
        
    def update_fusion_flag(self, idx):
        self.fusion_flag[idx] = 1

    def get_fusion_idx(self):

        fusion_idx = [idx for idx in range(len(self.fusion_flag)) if self.fusion_flag[idx]==1]

        # return np.asarray(fusion_idx)
        return fusion_idx
    
    def get_nofusion_idx(self):

        fusion_idx = [idx for idx in range(len(self.fusion_flag)) if self.fusion_flag[idx]==0]

        # return np.asarray(fusion_idx)
        return fusion_idx
    
    def check_valid_num(self, all_pred_box, count, gap):

        box_frame_ids = all_pred_box.frame_id #[N] tenor
        valid_num = all_pred_box.valid_num
        zero_boxid = torch.where((valid_num == 0) & (box_frame_ids< (count-gap)))[0] #not seen two times

        valid_boxid = torch.arange(len(all_pred_box))
        if zero_boxid.shape[0]>0:
            for idx in zero_boxid:
                valid_boxid=valid_boxid[valid_boxid!=idx]

        #update fusion_list
        self.fusion_list = [self.fusion_list[int(i)] for i in valid_boxid] 

        all_pred_box = all_pred_box[valid_boxid]
        return all_pred_box

    def compute_pose_disparity(self, pose1, pose2):
        """
        输入: 两个相机的4x4位姿矩阵 (齐次坐标形式，torch.Tensor)
        输出: 平移距离、旋转角度、综合视差指标
        """
        # 提取旋转矩阵和平移向量
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]

        # 计算平移距离（基线长度）[3,5](@ref)
        baseline = torch.norm(t2 - t1, p=2)

        # 计算相对旋转矩阵[3,5](@ref)
        R_rel = R2 @ R1.T  # R_rel = R2 * R1^T

        # 计算旋转角度（轴角表示）[1,4](@ref)
        trace = torch.clamp((torch.trace(R_rel) - 1) / 2, min=-1.0, max=1.0)  # 数值稳定处理
        rotation_angle = torch.arccos(trace) * 180 / torch.pi  # 转换为角度

        # 综合指标（加权计算）[3](@ref)
        disparity_score = 0.6 * baseline + 0.4 * rotation_angle

        return baseline, rotation_angle, disparity_score
    
    def compute_pose_center_disparity(self, pose1, pose2, center1, center2):
        """
        输入: 两个相机的4x4位姿矩阵 (齐次坐标形式，torch.Tensor)
        输出: 平移距离、旋转角度、综合视差指标
        """
        # 提取旋转矩阵和平移向量
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]

        # 计算平移距离（基线长度）[3,5](@ref)
        baseline = torch.norm(t2 - t1, p=2)

        # 计算相对旋转矩阵[3,5](@ref)
        R_rel = R2 @ R1.T  # R_rel = R2 * R1^T

        # 计算旋转角度（轴角表示）[1,4](@ref)
        trace = torch.clamp((torch.trace(R_rel) - 1) / 2, min=-1.0, max=1.0)  # 数值稳定处理
        rotation_angle = torch.arccos(trace) * 180 / torch.pi  # 转换为角度

        # 综合指标（加权计算）[3](@ref)
        disparity_score = 0.6 * baseline + 0.4 * rotation_angle

        center_dis = self.euclidean_distance_3d(center1, center2)


        return baseline, rotation_angle, disparity_score, center_dis
    
    def euclidean_distance_3d(self,point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def check_uv_bounds(self, uv_coords, W, H, ratio=1.0):
        gap_W = int((1-ratio)*W)
        gap_H = int((1-ratio)*H)
        # uv_coords: [N, 2] array
        u = uv_coords[:, 0]
        v = uv_coords[:, 1]
        mask = (u > gap_W) & (u < (W-gap_W)) & (v > gap_H) & (v < (H-gap_H))
        
        return mask#.astype(int)

    def check_floor_mask(self, box_3d, ratio=20):
        
        bos_size = box_3d[:,3:]
        max_values = torch.amax(bos_size, dim=1)  # 等价于torch.max(B, dim=1)[0]
        min_values = torch.amin(bos_size, dim=1)  # 等价于torch.min(B, dim=1)[0]
        second_values = torch.sort(bos_size, dim=1, descending=True)[0][:, 1]
        # mask = (max_values > 2) & (min_values<0.15)
        mask = (max_values/min_values > ratio)
        second_mask = (max_values/min_values > ratio/2) & (max_values/second_values > ratio/2) & (second_values/min_values<2.0) & (second_values<0.15) & (min_values<0.15)
        mask = mask | second_mask
        return mask#.astype(int)

    # def get_merge_tree(self, box_id: str) -> dict:
    #     """生成合并谱系树（参考多级链表扁平化思路[7,8](@ref)）"""
    #     box = self.box_registry[box_id]
    #     if len(box.merged_ids) == 1:
    #         return {"id": box_id, "children": []}
    #     return {
    #         "id": box_id,
    #         "children": [self.get_merge_tree(p) for p in box.merged_ids if p != box_id]
    #     }