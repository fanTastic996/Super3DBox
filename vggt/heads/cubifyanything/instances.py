# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Based on D2's Instances.
import itertools
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial import ConvexHull
import torchvision.transforms as tvf
# from mast3r.fast_nn import fast_reciprocal_NNs
# from dust3r.inference import inference
import torchvision.transforms as T
import matplotlib.pyplot as plt
from cubifyanything.utils import box_size_similarity

ImgNorm = tvf.Compose([
    # tvf.ToTensor(),  # 转换为Tensor并自动归一化到[0,1]
    tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到[-1,1]
])

def nms_3d_onlycur(boxes, scores, iou_threshold=0.5):
    """
    3D非极大值抑制实现
    参数:
        boxes: (N,7) numpy数组，格式为[x,y,z,dx,dy,dz,yaw]
        scores: (N,) numpy数组，置信度分数
        iou_threshold: 重叠阈值
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    keep = []
    
    while order.size > 0:
        # 当前最高分框
        i = order[0]
        keep.append(i)
        
        # 计算当前框与剩余框的3D IoU
        ious = calculate_3d_iou(boxes[i], boxes[order[1:]])
        
        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]  # +1因为跳过了第一个元素
        if order.size == 1:
            keep.append(order[0])
            break
    return np.array(keep)


def nms_3d(boxes, scores, iou_threshold=0.5):
    """
    3D非极大值抑制实现
    参数:
        boxes: (N,7) numpy数组，格式为[x,y,z,dx,dy,dz,yaw]
        scores: (N,) numpy数组，置信度分数
        iou_threshold: 重叠阈值
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    keep = []
    
    while order.size > 0:
        # 当前最高分框
        i = order[0]
        keep.append(i)
        
        # 计算当前框与剩余框的3D IoU
        ious = calculate_3d_iou(boxes[i], boxes[order[1:]])
        
        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]  # +1因为跳过了第一个元素
        if order.size == 1:
            keep.append(order[0])
            break
    return np.array(keep)

def nms_3d_v2(boxes, scores, framerecord, count, gap=25, iou_threshold=0.5):
    """
    3D非极大值抑制实现
    参数:
        boxes: (N,7) numpy数组，格式为[x,y,z,dx,dy,dz,yaw]
        scores: (N,) numpy数组，置信度分数
        iou_threshold: 重叠阈值
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    keep = []
    nms_box_inds = []

    while order.size > 0:
        # 当前最高分框
        i = order[0]
        keep.append(i)
        # print("i",i)
        # 计算当前框与剩余框的3D IoU
        temp_order = order[1:]
        ious = calculate_3d_iou(boxes[i], boxes[order[1:]])
        # print("debug",order)
        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        nms_inds = np.where(ious > iou_threshold)[0]
        nms_inds = np.asarray(nms_inds)
        # print("nms_inds", nms_inds, temp_order[nms_inds])
        num_global = len(framerecord.record[count-gap])
        # print("last ",num_global)

        if len(nms_inds)>0:
            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
                framerecord.add_single(count, j, i)

        order = order[inds + 1]  # +1因为跳过了第一个元素
        if order.size == 1:
            keep.append(order[0])
            break
    # print("nms",nms_box_inds)
    # exit(0)
    return np.array(keep), nms_box_inds


def nms_3d_v3(boxes, scores, framerecord, count, gap=25, iou_threshold=0.5):
    """
    3D非极大值抑制实现
    参数:
        boxes: (N,7) numpy数组，格式为[x,y,z,dx,dy,dz,yaw]
        scores: (N,) numpy数组，置信度分数
        iou_threshold: 重叠阈值
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    keep = []
    nms_box_inds = []
    # print("\n")
    while order.size > 0:
        # 当前最高分框
        i = order[0]
        keep.append(i)
        # 计算当前框与剩余框的3D IoU
        temp_order = order[1:]
        ious = calculate_obb_iou(boxes[i], boxes[order[1:]])
        # print(i,"ious:",ious,"other:",order[1:])
        # if i == 8:
        #     print("8",torch.from_numpy(boxes[i]))
        #     print("17",torch.from_numpy(boxes[17]))
        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        nms_inds = np.where(ious > iou_threshold)[0]
        nms_inds = np.asarray(nms_inds)

        if len(nms_inds)>0:
            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
                framerecord.add_single(count, j, i)

        order = order[inds + 1]  # +1因为跳过了第一个元素
        if order.size == 1:
            keep.append(order[0])
            break

    return np.array(keep), nms_box_inds

def nms_3d_v4(box_manager, boxes, scores, init_id, gap=25, iou_threshold=0.5):
    """
    3D非极大值抑制实现
    参数:
        boxes: (N,7) numpy数组，格式为[x,y,z,dx,dy,dz,yaw]
        scores: (N,) numpy数组，置信度分数
        iou_threshold: 重叠阈值
        cur2all_map: 当前all pred box每个Box到per_frame_box的id映射
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    order_init_id = init_id.tolist()

    keep = []
    # print("\n")
    while order.size > 0:
        nms_box_inds = []
        # 当前最高分框
        i = order[0]
        keep.append(i)
        # 计算当前框与剩余框的3D IoU
        temp_order = order[1:]
        ious = calculate_obb_iou(boxes[i], boxes[order[1:]])
        # print(i,"ious:",ious,"other:",order[1:])
        # if i == 8:
        #     print("8",torch.from_numpy(boxes[i]))
        #     print("17",torch.from_numpy(boxes[17]))
        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        nms_inds = np.where(ious > iou_threshold)[0]
        nms_inds = np.asarray(nms_inds)
        '''
        record the fusion history
        '''
        if len(nms_inds)>0:
            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
                # framerecord.add_single(count, j, i)
            '''
            record and update the fusion list
            '''
            # print('nms',i,nms_box_inds)
            box_manager.record(i, nms_box_inds, order_init_id)
                


        order = order[inds + 1]  # +1因为跳过了第一个元素
        if order.size == 1:
            keep.append(order[0])
            break

    return np.array(keep), nms_box_inds


def nms_3d_v5(box_manager, boxes, scores, init_id, cam_poses, box_size,  iou_threshold=0.5,merge_upper=0.7,merge_lower=0.3):
    """
    3D非极大值抑制实现
    参数:
        boxes: [N,8,3] numpy数组，格式为
        scores: [N] numpy数组，置信度分数
        iou_threshold: 重叠阈值
        cur2all_map: 当前all pred box每个Box到per_frame_box的id映射
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    order_init_id = init_id.tolist()
    # print("order",order)
    keep = []
    # print("\n")
    while order.size > 0:
        nms_box_inds = []
        # 当前最高分框
        i = order[0]
        keep.append(i)
        # 计算当前框与剩余框的3D IoU
        temp_order = order[1:]
        ious = calculate_obb_iou(boxes[i], boxes[order[1:]])
        # print(i,"ious:",ious,"other:",order[1:])
        # if i == 8:
        #     print("8",torch.from_numpy(boxes[i]))
        #     print("17",torch.from_numpy(boxes[17]))
        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        # print(i,"keep",inds)
        nms_inds = np.where((ious > iou_threshold) & (ious < merge_upper) & (ious>merge_lower))[0]
        nms_inds = np.asarray(nms_inds)
        '''
        record the fusion history
        '''
        # print("nms",i, 'len:',len(nms_inds),nms_inds)
        if len(nms_inds)>0:
            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
                # framerecord.add_single(count, j, i)
            '''
            record and update the fusion list
            '''
            # print('nms',i,nms_box_inds)
            keep = box_manager.record(i, nms_box_inds, order_init_id, cam_poses, box_size, keep)        
        # print("order in loop before",order)
        order = order[inds + 1]  # +1因为跳过了第一个元素
        # print("order in loop",order)
        if order.size == 1:
            keep.append(order[0])
            break

    keep.sort()
    return np.array(keep)

def nms_3d_v6(instance_lists, box_manager, boxes, scores, init_id, cam_poses, box_size,  iou_threshold=0.5,merge_upper=0.7,merge_lower=0.3): #merge_lower=0.3/0.1
    """
    3D非极大值抑制实现
    参数:
        boxes: [N,8,3] numpy数组，格式为
        scores: [N] numpy数组，置信度分数
        iou_threshold: 重叠阈值
        cur2all_map: 当前all pred box每个Box到per_frame_box的id映射
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    order_init_id = init_id.tolist()
    # print("order",order)
    # print("scores",scores)
    # exit(0)
    keep = []
    success_nms = []
    # print("\n")
    while order.size > 0:
        nms_box_inds = []
        # 当前最高分框
        i = order[0]
        # print("order",order)
        keep.append(i)
        # 计算当前框与剩余框的3D IoU
        temp_order = order[1:]
        ious = calculate_obb_iou(boxes[i], boxes[order[1:]])

        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        # i nms others, and is valid
        associate_inds = np.where(ious > iou_threshold)[0]
        if associate_inds.shape[0]>=1:
            instance_lists.valid_num[i] +=1
            print('nms',i,'->',order[1+associate_inds])
        # nms_inds = np.where((ious > iou_threshold) & (ious < merge_upper) & (ious>merge_lower))[0] # old priciple
        # nms_inds = np.where((ious > iou_threshold) &  (ious>merge_lower))[0]
        nms_inds = np.where((ious > iou_threshold))[0]
        nms_inds = np.asarray(nms_inds)
        # nms_inds = np.asarray(associate_inds)
        # print('nms_inds',nms_inds)
        '''
        record the fusion history
        '''
        if len(nms_inds)>0:

            success_nms.append(i)

            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
                # framerecord.add_single(count, j, i)
            '''
            record and update the fusion list
            '''
            # print('nms',i,nms_box_inds)
            keep = box_manager.record(i, nms_box_inds, order_init_id, cam_poses, box_size, keep)        

            # print("nms",i,"no associate")
        order = order[inds + 1]  # +1因为跳过了第一个元素

        if order.size == 1:
            keep.append(order[0])
            break

    keep.sort()
    success_nms.sort()
    return np.array(keep), np.array(success_nms)

def nms_3d_v8(instance_lists, box_manager, boxes, scores, init_id, cam_poses, box_size,  iou_threshold=0.5,merge_upper=0.7,merge_lower=0.3): #merge_lower=0.3/0.1
    """
    3D非极大值抑制实现
    参数:
        boxes: [N,8,3] numpy数组，格式为
        scores: [N] numpy数组，置信度分数
        iou_threshold: 重叠阈值
        cur2all_map: 当前all pred box每个Box到per_frame_box的id映射
    返回:
        keep: 保留框的索引列表
    """
    #boxes_center
    boxes_centers = np.mean(boxes,axis=1) #[N,3]

    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    order_init_id = init_id.tolist()
    # print("order",order)
    # print("scores",scores)
    # exit(0)
    keep = []
    success_nms = []
    # print("\n")
    while order.size > 0:
        nms_box_inds = []
        # 当前最高分框
        i = order[0]
        # print("order",order)
        keep.append(i)
        # 计算当前框与剩余框的3D IoU
        temp_order = order[1:]
        ious = calculate_obb_iou(boxes[i], boxes[order[1:]])

        # 保留IoU低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        # i nms others, and is valid
        associate_inds = np.where(ious > iou_threshold)[0]
        if associate_inds.shape[0]>=1:
            instance_lists.valid_num[i] +=1
            print('nms',i,'->',order[1+associate_inds])
        # nms_inds = np.where((ious > iou_threshold) & (ious < merge_upper) & (ious>merge_lower))[0] # old priciple
        # nms_inds = np.where((ious > iou_threshold) &  (ious>merge_lower))[0]
        nms_inds = np.where((ious > iou_threshold))[0]
        nms_inds = np.asarray(nms_inds)
        # nms_inds = np.asarray(associate_inds)
        # print('nms_inds',nms_inds)
        '''
        record the fusion history
        '''
        if len(nms_inds)>0:

            success_nms.append(i)

            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
                # framerecord.add_single(count, j, i)
            '''
            record and update the fusion list
            '''
            # print('nms',i,nms_box_inds)
            keep = box_manager.record(i, nms_box_inds, order_init_id, cam_poses, box_size, keep, boxes_centers)        

            # print("nms",i,"no associate")
        order = order[inds + 1]  # +1因为跳过了第一个元素

        if order.size == 1:
            keep.append(order[0])
            break

    keep.sort()
    success_nms.sort()
    return np.array(keep), np.array(success_nms)

# NMS with shape similarity, if the IoU is large but the shape is different, we can still keep them.
def nms_3d_v7(instance_lists, box_manager, boxes, scores, init_id, cam_poses, box_size,  iou_threshold=0.5,merge_upper=0.7,merge_lower=0.3):
    """
    3D非极大值抑制实现
    参数:
        boxes: [N,8,3] numpy数组，格式为
        scores: [N] numpy数组，置信度分数
        iou_threshold: 重叠阈值
        box_size: [N,3]
        cur2all_map: 当前all pred box每个Box到per_frame_box的id映射
    返回:
        keep: 保留框的索引列表
    """
    # 按置信度降序排序
    order = scores.argsort()[::-1] #index large->small
    order_init_id = init_id.tolist()
    # print("order",order)
    # print("scores",scores)
    # exit(0)
    keep = []
    # print("\n")
    while order.size > 0:
        nms_box_inds = []
        # 当前最高分框
        i = order[0]
        # print("order",order)
        keep.append(i)
        # 计算当前框与剩余框的3D IoU
        temp_order = order[1:]
        ious = calculate_obb_iou(boxes[i], boxes[order[1:]])
        shape_sim = box_size_similarity(box_size[i], box_size[order[1:]], alpha=0.2)

        # 保留IoU低于阈值的框
        inds = np.where((ious <= iou_threshold) | ((ious > iou_threshold) & (shape_sim<0.75)))[0] # number

        # i nms others, and is valid
        associate_inds = np.where((ious > iou_threshold) & (shape_sim>=0.75))[0]
        if associate_inds.shape[0]>=1:
            instance_lists.valid_num[i] +=1
            print('nms',i,'->',order[1+associate_inds])

        nms_inds = np.asarray(associate_inds)
        '''
        record the fusion history
        '''
        if len(nms_inds)>0:
            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
                # framerecord.add_single(count, j, i)
            '''
            record and update the fusion list
            '''
            # print('nms',i,nms_box_inds)
            keep = box_manager.record(i, nms_box_inds, order_init_id, cam_poses, box_size, keep)        

        order = order[inds + 1]  # +1因为跳过了第一个元素

        if order.size == 1:
            keep.append(order[0])
            break

    keep.sort()
    return np.array(keep)

def calculate_3d_iou(box_a, boxes_b):
    """
    计算一个3D框与多个3D框的IoU（简化版，未考虑旋转）
    参数:
        box_a: (7,) numpy数组
        boxes_b: (N,7) numpy数组
    返回:
        ious: (N,) numpy数组
    """
    box_a = box_a[[0,1,2,3,5,4]]
    boxes_b = boxes_b[:,[0,1,2,3,5,4]]

    # 提取坐标和尺寸
    a_min = box_a[:3] - box_a[3:6]/2
    a_max = box_a[:3] + box_a[3:6]/2
    b_mins = boxes_b[:,:3] - boxes_b[:,3:6]/2
    b_maxs = boxes_b[:,:3] + boxes_b[:,3:6]/2
    
    # 计算各轴交集
    intersect_mins = np.maximum(a_min, b_mins)
    intersect_maxs = np.minimum(a_max, b_maxs)
    intersect_whs = np.maximum(intersect_maxs - intersect_mins, 0.)
    
    # 计算交集体积
    intersection = intersect_whs[:,0] * intersect_whs[:,1] * intersect_whs[:,2]
    
    # 计算并集
    volume_a = box_a[3] * box_a[4] * box_a[5]
    volumes_b = boxes_b[:,3] * boxes_b[:,4] * boxes_b[:,5]
    union = volume_a + volumes_b - intersection
    
    # 避免除以零
    union = np.maximum(union, 1e-6)
    return intersection / union


def calculate_obb_iou(corners1, corners_others):
    """
    计算一个3D框的corners与多个3D框的corners的IoU
    参数:
        box_a: (8,3) numpy数组
        boxes_b: (N,7) numpy数组
    返回:
        ious: (N,) numpy数组
    """
    iou = [Instances3D.obb_iou(corners1,corners_others[i]) for i in range(corners_others.shape[0])]

    iou = np.asarray(iou) 
    # print("iou",iou)
    return iou

# Provides basic compatibility with D2.
class Instances3D:
    """
    This class represents a list of instances in _the world_.
    """
    def __init__(self, image_size: Tuple[int, int] = (0, 0), **kwargs: Any):
        # image_size is here for Detectron2 compatibility.
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width (note: opposite of cubifycore).

        Here for D2 compatibility. You probably shouldn't be using this.
        """
        return self._image_size            

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances3D!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances3D of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances3D":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances3D(image_size=self._image_size)
        # Copy fields that were explicitly added to this object (e.g., hidden fields)
        for name, value in self.__dict__.items():
            if (name not in ["_fields"]) and name.startswith("_"):
                setattr(ret, name, value.to(*args, **kwargs) if hasattr(value, "to") else value)
        
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)

        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances3D":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances3D` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances3D index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances3D(image_size=self.image_size)
        for name, value in self.__dict__.items():
            if (name not in ["_fields"]) and name.startswith("_"):
                setattr(ret, name, value)
        
        for k, v in self._fields.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) or hasattr(v, "tensor"):
                # assume if has .tensor, then this is piped into __getitem__.
                # Make sure to match underlying types.
                if isinstance(v, np.ndarray) and isinstance(item, torch.Tensor):
                    ret.set(k, v[item.cpu().numpy()])
                else:
                    ret.set(k, v[item])
            elif hasattr(v, "__iter__"):
                # handle non-Tensor types like lists, etc.
                if isinstance(item, np.ndarray) and (item.dtype == np.bool_):
                    ret.set(k, [v_ for i_, v_ in enumerate(v) if item[i_]])                    
                elif isinstance(item, torch.BoolTensor) or (isinstance(item, torch.Tensor) and (item.dtype == torch.bool)):
                    ret.set(k, [v_ for i_, v_ in enumerate(v) if item[i_].item()])
                elif isinstance(item, torch.LongTensor) or (isinstance(item, torch.Tensor) and (item.dtype == torch.int64)):
                    # Can this be right?
                    ret.set(k, [v[i_.item()] for i_ in item])
                elif isinstance(item, slice):
                    ret.set(k, v[item])
                else:
                    raise ValueError("Expected Bool or Long Tensor")
            else:
                raise ValueError("Not supported!")
                
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances3D does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances3D` object is not iterable!")

    def split(self, split_size_or_sections):
        indexes = torch.arange(len(self))
        splits = torch.split(indexes, split_size_or_sections)

        return [self[split] for split in splits]

    def clone(self):
        import copy

        ret = Instances3D(image_size=self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "clone"):
                v = v.clone()
            elif isinstance(v, np.ndarray):
                v = np.copy(v)
            elif isinstance(v, (str, list, tuple)):
                v = copy.copy(v)
            elif hasattr(v, "tensor"):
                v = type(v)(v.tensor.clone())
            else:
                raise NotImplementedError

            ret.set(k, v)

        return ret

    @staticmethod
    def cat(instance_lists: List["Instances3D"]) -> "Instances3D":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances3D) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        ret = Instances3D(image_size=instance_lists[0]._image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):      # 新增numpy判断
                values = np.concatenate(values, axis=0)  # 沿第一个维度拼接[1,5](@ref)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret
    
    def project_3d_boxes(self, K, H=480, W=640):
        """
        Project 3D boxes to 2D image plane.
        Returns:
            projected_boxes: (8,3) numpy array, where N is the number of boxes.
                              Each box is represented as [x1, y1, x2, y2].
        """
        
        boxes = self.get('pred_boxes_3d')
        corners = boxes.corners #[N,8,3]
        cam_pose = self.cam_pose #[N,4,4]

        N = corners.shape[0]
        # 扩展为齐次坐标 [N,8,4]
        ones = torch.ones((N, 8, 1), device=corners.device)
        boxes_homo = torch.cat([corners, ones], dim=2)  # [N,8,4]
        # 计算外参逆矩阵 [N,4,4]
        pose_inv = torch.linalg.inv(cam_pose).to(corners.device)
        # 转换到相机坐标系 [N,8,4]
        boxes_cam = torch.einsum('nij,nkj->nki', pose_inv, boxes_homo)  # 批量矩阵乘法
        # 提取相机坐标 [N,8,3]
        X = boxes_cam[..., 0]
        Y = boxes_cam[..., 1]
        Z = boxes_cam[..., 2]

        # 投影到图像平面 [N,8,2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = (fx * X / Z) + cx
        v = (fy * Y / Z) + cy

        u = torch.clamp(u,0,W)
        v = torch.clamp(v,0,H)
        
        projected_boxes = torch.stack([u, v], dim=-1) #[N,8,2]

        self.projected_boxes = projected_boxes




    #added by lyq
    # def nms(instance_lists: "Instances3D", threshold: "float", poses: "") -> "Instances3D":
    def nms(instance_lists, threshold, poses) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists


        # boxes = np.random.rand(100, 7) * 10  # [x,y,z,dx,dy,dz,yaw]
        boxes_now = instance_lists.get('pred_boxes_3d')
        # Hard-code these suffixes.
        centers_box=boxes_now.gravity_center.cpu().numpy()
        #transformed to world coordinate
        centers_box = np.squeeze(poses[:,:3,:3]@np.expand_dims(centers_box,axis=-1)) + poses[:,:3,3]

        sizes_size=boxes_now.dims.cpu().numpy()
        boxes = np.concatenate((centers_box,sizes_size),axis=-1)

        # scores = np.ones(boxes.shape[0])  # 置信度
        scores = instance_lists.scores.cpu().numpy()  # 置信度
        # print("scores",scores)
        before_nms = len(instance_lists)
        # 执行NMS
        keep = nms_3d(boxes, scores, iou_threshold=threshold)
        keep = sorted(keep)
        
        filtered_instance_lists = instance_lists[keep]
        after_nms = len(filtered_instance_lists)

        print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]


        return filtered_instance_lists, keep

    def nms_not_merge(instance_lists, threshold, poses) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists


        # boxes = np.random.rand(100, 7) * 10  # [x,y,z,dx,dy,dz,yaw]
        boxes_now = instance_lists.get('pred_boxes_3d')
        # Hard-code these suffixes.
        centers_box=boxes_now.gravity_center.cpu().numpy()
        #transformed to world coordinate
        centers_box = np.squeeze(poses[:,:3,:3]@np.expand_dims(centers_box,axis=-1)) + poses[:,:3,3]

        sizes_size=boxes_now.dims.cpu().numpy()
        boxes = np.concatenate((centers_box,sizes_size),axis=-1)

        # scores = np.ones(boxes.shape[0])  # 置信度
        scores = instance_lists.scores.cpu().numpy()  # 置信度
        # print("scores",scores)
        before_nms = len(instance_lists)
        # 执行NMS
        keep = nms_3d_v2(boxes, scores, iou_threshold=threshold)
        keep = sorted(keep)
        
        # filtered_instance_lists = instance_lists[keep]
        # after_nms = len(filtered_instance_lists)

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep

    def nms_not_merge_v2(instance_lists, threshold, poses, framerecord, count) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists


        # boxes = np.random.rand(100, 7) * 10  # [x,y,z,dx,dy,dz,yaw]
        boxes_now = instance_lists.get('pred_boxes_3d')
        # Hard-code these suffixes.
        centers_box=boxes_now.gravity_center.cpu().numpy()
        #transformed to world coordinate
        centers_box = np.squeeze(poses[:,:3,:3]@np.expand_dims(centers_box,axis=-1)) + poses[:,:3,3]

        sizes_size=boxes_now.dims.cpu().numpy()
        boxes = np.concatenate((centers_box,sizes_size),axis=-1)

        # scores = np.ones(boxes.shape[0])  # 置信度
        scores = instance_lists.scores.cpu().numpy()  # 置信度
        # print("scores",scores)
        before_nms = len(instance_lists)
        # 执行NMS
        keep, nms_inds = nms_3d_v2(boxes, scores, framerecord, count, iou_threshold=threshold)
        keep = sorted(keep)
        nms_inds = sorted(nms_inds)
        
        # filtered_instance_lists = instance_lists[keep]
        # after_nms = len(filtered_instance_lists)

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep #, nms_inds
    

    def nms_not_merge_v3(instance_lists, threshold, poses, framerecord, count) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists


        # boxes = np.random.rand(100, 7) * 10  # [x,y,z,dx,dy,dz,yaw]
        boxes_now = instance_lists.get('pred_boxes_3d')
        # Hard-code these suffixes.
        centers_box=boxes_now.gravity_center.cpu().numpy()
        #transformed to world coordinate
        centers_box = np.squeeze(poses[:,:3,:3]@np.expand_dims(centers_box,axis=-1)) + poses[:,:3,3]
 
        boxes_corners = boxes_now.corners.cpu().numpy() #[N,8,3]
        boxes_corners = poses[:,:3,:3] @ np.transpose(boxes_corners,(0,2,1)) + poses[:,:3,3:]
        boxes_corners = np.transpose(boxes_corners,(0,2,1)) #[N,8,3]




        # scores = np.ones(boxes.shape[0])  # 置信度
        scores = instance_lists.scores.cpu().numpy()  # 置信度
        # print("scores",scores)
        before_nms = len(instance_lists)
        # 执行NMS
        keep, nms_inds = nms_3d_v3(boxes_corners, scores, framerecord, count, iou_threshold=threshold)
        keep = sorted(keep)
        nms_inds = sorted(nms_inds)
        
        # filtered_instance_lists = instance_lists[keep]
        # after_nms = len(filtered_instance_lists)

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep #, nms_inds


    def nms_v4(instance_lists, threshold, poses, framerecord, count, box_manager) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists

        # boxes = np.random.rand(100, 7) * 10  # [x,y,z,dx,dy,dz,yaw]
        boxes_now = instance_lists.get('pred_boxes_3d')

        # nofusion_mask = box_manager.get_nofusion_idx()

        boxes_corners = boxes_now.corners.cpu().numpy() #[N,8,3]
        # boxes_corners[nofusion_mask] = poses[:,:3,:3] @ np.transpose(boxes_corners[nofusion_mask],(0,2,1)) + poses[:,:3,3:]
        # boxes_corners[nofusion_mask] = np.transpose(boxes_corners[nofusion_mask],(0,2,1)) #[N,8,3]

        scores = instance_lists.scores.cpu().numpy()  # 置信度
        init_id = instance_lists.init_id.cpu().numpy()  # 置信度

        before_nms = len(instance_lists)
        # 执行NMS
        keep, _ = nms_3d_v4(box_manager, boxes_corners, scores, init_id, iou_threshold=threshold)
        keep = sorted(keep)
        # nms_inds = sorted(nms_inds)
        
        # filtered_instance_lists = instance_lists[keep]
        # after_nms = len(filtered_instance_lists)

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep #, nms_inds


    def nms_v5(instance_lists, threshold, box_manager, cam_poses) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists

        boxes_now = instance_lists.get('pred_boxes_3d')

        boxes_corners = boxes_now.corners.cpu().numpy() #[N,8,3]
        box_size = boxes_now.dims.cpu().numpy()  

        scores = instance_lists.scores.cpu().numpy()  # 置信度
        init_id = instance_lists.init_id.cpu().numpy()  # 置信度

        # 执行NMS
        keep = nms_3d_v5(box_manager, boxes_corners, scores, init_id, cam_poses, box_size, iou_threshold=threshold)
        keep = sorted(keep)
 

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep #, nms_inds
    
    def nms_v6(instance_lists, threshold, box_manager, cam_poses) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists

        boxes_now = instance_lists.get('pred_boxes_3d')

        boxes_corners = boxes_now.corners.cpu().numpy() #[N,8,3]
        box_size = boxes_now.dims.cpu().numpy()  

        scores = instance_lists.scores.cpu().numpy()  # 置信度
        init_id = instance_lists.init_id.cpu().numpy()  # 置信度

        # 执行NMS
        keep, success_nms = nms_3d_v6(instance_lists, box_manager, boxes_corners, scores, init_id, cam_poses, box_size, iou_threshold=threshold)
        keep = sorted(keep)
        success_nms = sorted(success_nms)

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep,success_nms #, nms_inds

    def nms_v8(instance_lists, threshold, box_manager, cam_poses) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists

        boxes_now = instance_lists.get('pred_boxes_3d')

        boxes_corners = boxes_now.corners.cpu().numpy() #[N,8,3]
        box_size = boxes_now.dims.cpu().numpy()  

        scores = instance_lists.scores.cpu().numpy()  # 置信度
        init_id = instance_lists.init_id.cpu().numpy()  # 置信度


        # 执行NMS
        keep, success_nms = nms_3d_v8(instance_lists, box_manager, boxes_corners, scores, init_id, cam_poses, box_size, iou_threshold=threshold)
        keep = sorted(keep)
        success_nms = sorted(success_nms)

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep,success_nms #, nms_inds

    def nms_v7(instance_lists, threshold, box_manager, cam_poses) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists

        boxes_now = instance_lists.get('pred_boxes_3d')

        boxes_corners = boxes_now.corners.cpu().numpy() #[N,8,3]
        box_size = boxes_now.dims.cpu().numpy()  #[N,3]

        scores = instance_lists.scores.cpu().numpy()  # 置信度
        init_id = instance_lists.init_id.cpu().numpy()  # 置信度

        # 执行NMS
        keep = nms_3d_v7(instance_lists, box_manager, boxes_corners, scores, init_id, cam_poses, box_size, iou_threshold=threshold)
        keep = sorted(keep)
 

        # print("3D NMS:",before_nms,"--->>>", after_nms,' filter', before_nms-after_nms,'boxes' )  # 输出 [0]
        return  keep #, nms_inds

    def merge(instance_lists, cur_pred_ins, all_poses,cur_poses, threshold=0.9) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        before_merge = len(instance_lists)+len(cur_pred_ins)

        #merge id list
        merged_old = []
        merged_new = []

        # print('instance_lists',instance_lists)
        global_ins = instance_lists #.get('pred_boxes_3d')
        g_object_desc = global_ins.object_desc
        g_object_desc = g_object_desc.unsqueeze(0)
        c_object_desc = cur_pred_ins.object_desc
        c_object_desc = c_object_desc.unsqueeze(1)
        sim_matrix = F.cosine_similarity(c_object_desc, g_object_desc, dim=2) #[num of cur, num of global] 
        # 找到每行的最大值及其列索引
        max_values, col_indices = torch.max(sim_matrix, dim=1) #[num of cur]
        valid_match = max_values>threshold

        #处理多对一的情况
        for j in range(valid_match.shape[0]):
            if valid_match[j]==True:
                if j <valid_match.shape[0]-1:
                    if col_indices[j] in col_indices[j+1:]:
                        for z in range(j+1,valid_match.shape[0]):
                            if col_indices[z]==col_indices[j] and valid_match[z]==True:
                                if max_values[z] > max_values[j]:
                                    valid_match[j] = False
                                else:
                                    valid_match[z] = False
                                break

        match_cur_ind = torch.arange(sim_matrix.shape[0]).cuda()[valid_match]
        match_glo_ind = col_indices[valid_match]
        #先合并
        g_scores = global_ins.scores
        c_scores = cur_pred_ins.scores
        for i in range(match_cur_ind.shape[0]):
            g_ind = match_glo_ind[i]
            c_ind = match_cur_ind[i]
            if c_scores[c_ind] > g_scores[g_ind]: #当前的更好，进行替换
                merged_old.append(g_ind.item())
                merged_new.append(c_ind.item())
                # global_ins[g_ind] = cur_pred_ins[c_ind]
                # print("debug",3,sim_matrix[3])
                Instances3D.modify_instance(g_ind,c_ind,global_ins,cur_pred_ins)
                all_poses[g_ind] = cur_poses[c_ind]
        #再添加没有Match的
        rest_cur_pred_ins = cur_pred_ins[~valid_match]
        
        merged_instance = Instances3D.cat([global_ins, rest_cur_pred_ins])
        
        merged_poses = np.concatenate((all_poses, cur_poses[~valid_match.cpu().numpy()]),axis=0)
        
        after_merged = len(merged_instance)
        print('merged_old:',merged_old)
        print('merged_new:',merged_new)
        print("add:",torch.arange(sim_matrix.shape[0]).cuda()[~valid_match])
        print(f"Merging: global:{len(instance_lists)} + current:{len(cur_pred_ins)}","  add:",len(rest_cur_pred_ins),"  | ",before_merge,">>>>>>", after_merged )  # 输出 [0]



        return merged_instance, merged_poses

    def modify_instance(ind_old,ind_new,old_ins,new_ins):
        old_ins.scores[ind_old] = new_ins.scores[ind_new]
        old_ins.pred_classes[ind_old] = new_ins.pred_classes[ind_new]
        old_ins.pred_boxes[ind_old] = new_ins.pred_boxes[ind_new]
        old_ins.pred_logits[ind_old] = new_ins.pred_logits[ind_new]
        old_ins.pred_boxes_3d.tensor[ind_old] = new_ins.pred_boxes_3d.tensor[ind_new]
        old_ins.pred_boxes_3d.R[ind_old] = new_ins.pred_boxes_3d.R[ind_new]
        old_ins.object_desc[ind_old] = new_ins.object_desc[ind_new]
        old_ins.pred_proj_xy[ind_old] = new_ins.pred_proj_xy[ind_new]


    def merge_by_corr(last_pred_instances, pred_instances, last_kf_rgb, cur_keep_idx, global_pred_box, corr, framerecord, all_poses, frame_id, gap):
        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        lastkf_box = last_pred_instances.pred_boxes.cpu().numpy()
        last_kf_match = corr[:,:2]
        cur_match = corr[:,2:]
        global_num = len(global_pred_box)
        keep = np.arange(len(global_pred_box)+len(pred_instances)) 
        mask = np.ones_like(keep,dtype=bool)
        #record the current frame boxes info
        print("framerecord",framerecord.record)
        #当前Box的两种去路 1.与旧的关联 2.新增
        for idx in cur_keep_idx:
            cur_box = cur_2d_box[idx]
            cur_mask = Instances3D.points_in_box(cur_match,cur_box) #bool type
            if np.sum(cur_mask)>=1: #当前box有correspondence
                selected_lastkf_pixels = last_kf_match[cur_mask] #[N,2]
                # print("A",selected_lastkf_pixels,'cur_mask',cur_mask)
                lastkf_box_iou, overlap_A = Instances3D.IoU_2D(selected_lastkf_pixels,lastkf_box) #[num of lastkf box]
                # _, overlap_B = Instances3D.IoU_box(cur_match[cur_mask],cur_2d_box)

                corrponding_boxid = np.argmax(lastkf_box_iou)
                print("lastkf_box_iou",lastkf_box_iou)
                print("overlap_A",overlap_A)

                # box_features = pred_instances.box_features[idx]
                # last_box_features = last_pred_instances.box_features[corrponding_boxid]
                # semantic_similarity = F.cosine_similarity(box_features.unsqueeze(0), last_box_features.unsqueeze(0), dim=1)
                # print("cur",idx,"last",corrponding_boxid,"similarity",semantic_similarity)

                # if lastkf_box_count[corrponding_boxid] >= 5: #有关联
                if (lastkf_box_iou[corrponding_boxid] > 0.33 or overlap_A[corrponding_boxid]>0.7): #是同一Instance 有关联, 2D box in last kf overlap is >50% or the corresponding 2D box is contained in the last box
         
                    global_idx = framerecord.record[frame_id-gap][corrponding_boxid]
                    print("global_idx",global_idx,"cur idx",idx,'corrponding_boxid',corrponding_boxid)
                    print("lastkf_box_count",lastkf_box_iou)
                    print("scores",global_pred_box.scores[global_idx],pred_instances.scores[idx])
                    if global_pred_box.scores[global_idx]<pred_instances.scores[idx]: #replace and modify
                        Instances3D.modify_instance(global_idx,idx,global_pred_box,pred_instances)
                        mask[idx+global_num] = False
                        all_poses[global_idx] = all_poses[idx+global_num]
                        print("1111111111")
                    else: #abandon worse than the global one
                        mask[idx+global_num] = False
                        print("2222222222")
                    framerecord.add_single(frame_id, idx, global_idx)
                    
                else: #不是同一个instance，不够数，新增new obeserved instance 
                    print("4444444444444444")
                    mask[idx+global_num] = True
                    until_cur_mask = mask[:idx+global_num+1]
                    tmp_count = np.sum(until_cur_mask)
                    framerecord.add_single(frame_id, idx, tmp_count-1)
            else: #没有correspondence在本帧，新增new obeserved instance
                print("3333333333333")
                mask[idx+global_num] = True
                until_cur_mask = mask[:idx+global_num+1]
                tmp_count = np.sum(until_cur_mask)
                framerecord.add_single(frame_id, idx, tmp_count-1)

        keep = keep[mask]
        keep=sorted(keep)
        all_pred_box = Instances3D.cat([global_pred_box,pred_instances])
        all_pred_box = all_pred_box[keep]
        all_poses = all_poses[keep]

        return all_pred_box, all_poses
    
    def merge_by_corr_v6(cur_keep_idx, pred_instances,  global_pred_box, global_pose, all_pred_box, all_poses, frame_id, gap, mask, intrinsic,threshold=0.33):  
        '''
        查找与当前帧存在overlap的所有帧,
            1.如果当前box与所有kf都不存在cor,则认为是新Instance
            2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
            2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
                2.2.1 IoU高于阈值则认为是同一instance
                    2.2.1.1 当前box Scores更高,则替换
                    2.2.1.2 当前box scores更低,则删除当前box
                2.2.2 否则认为当前box是新的instance
        '''

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()
        cur_3d_box = pred_instances.pred_boxes_3d.dims.cpu().numpy()

        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]
        
        for idx in cur_keep_idx:
            kf_count = []
            kf_ind = []
            cur_box = cur_2d_box[idx]

            '''
            only deal with the small ones length/weight/height < 35cm
            '''
            cur_box_size = cur_3d_box[idx,:3]
            if np.max(cur_box_size)>0.35:
                continue


            #TODO:优化为只找有overlap的帧
            for kfid in range(0,frame_id,gap):
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                if corr.shape[0] == 0:
                    continue
                else:
                    cur_match = corr[:,2:]
                    cur_mask = Instances3D.points_in_box(cur_match,cur_box) #bool type
                    cur_cor_count = np.sum(cur_mask)
                    if cur_cor_count<1: #当前box有correspondence
                        continue
                    else:
                        kf_count.append(cur_cor_count)
                        kf_ind.append(kfid)
            # if len(kf_count)==0:
            #     print(frame_id,"no corr",idx,"remove",idx+N_glo)
            #     keep_idx = keep_idx[keep_idx!=(idx+N_glo)]

            '''
            NEW ADDED! remove extremely small boxes,
            '''

           
            if frame_id==550:
                print(idx,"kf_count:",kf_count)
                print(idx,"kf_ind:",kf_ind)
                print(idx,"N_glo",N_glo,"N_cur",N_cur)

            if len(kf_count)>0:

                data = np.array(kf_count)
                sorted_indices = np.argsort(data)  # 返回排序后的索引
                best_kfid = kf_ind[sorted_indices[-1]]
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{best_kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                cur_match = corr[:,2:]
                best_kf_match = corr[:,:2]
                cur_mask = Instances3D.points_in_box(cur_match,cur_box) 
                selected_bestkf_pixels = best_kf_match[cur_mask] #[N,2]
                #
                bestkf_pose =np.loadtxt(f'/media/lyq/data/dataset/ScanNet/scene0169_00/frames/pose/{best_kfid}.txt').reshape(4,4) #[4,4]
                
                boxes_global = global_pred_box.get('pred_boxes_3d')

                boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] #N,8,3
                #transformed to world coordinate
                boxes_3d = Instances3D.transform_boxes(boxes_3d,global_pose[global_keep_idx,:3,:3],global_pose[global_keep_idx,:3,3]) #[N_glo-nms,8,3]

                boxes_2d = Instances3D.project_3d_to_2d_box(boxes_3d, intrinsic.cpu().numpy(), bestkf_pose) #[N_glo-nms,4]

                box_iou, overlap_A = Instances3D.IoU_2D(selected_bestkf_pixels,boxes_2d) #[N_glo-nms]
                corrponding_boxid = np.argmax(box_iou)

                if frame_id==550:
                    print("boxes_global",len(boxes_global))
                    print("boxes_3d",boxes_3d.shape)
                    print("boxes_2d",boxes_2d.shape)
                    print("debug","frame_id:",frame_id,"cur box:",idx,"box_iou:",box_iou,'best kfid:',best_kfid)

                if (box_iou[corrponding_boxid] > threshold): #0.1 #0.33
                    corresponding_idx = global_keep_idx[corrponding_boxid]
                
                    if global_box_scores[corresponding_idx]<cur_2d_box_scores[idx]:
                        print("frame_id:",frame_id,"find corr",idx,"remove old ",corresponding_idx)
                        keep_idx = keep_idx[keep_idx!=(corresponding_idx)]
                    else:
                        print("frame_id:",frame_id,"find corr",idx,"remove new",idx+N_glo)
                        keep_idx = keep_idx[keep_idx!=(idx+N_glo)]

                if np.max(kf_count)<10 and (box_iou[corrponding_boxid])<0.01:
                    print("frame_id:",frame_id,"find corr",idx,"remove new, too small count",idx+N_glo,"count:",np.max(kf_count),'box iou:',box_iou[corrponding_boxid])
                    keep_idx = keep_idx[keep_idx!=(idx+N_glo)]


        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses


    def merge_by_corr_v7(box_manager, cur_keep_idx, pred_instances,  global_pred_box, global_pose, all_pred_box, all_poses, per_frame_ins_cam_pose, frame_id, gap, mask, intrinsic,all_kf_pose,threshold=0.33,H=480,W=640):  
        '''
        查找与当前帧存在overlap的所有帧,
            1.如果当前box与所有kf都不存在cor,则认为是新Instance
            2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
            2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
                2.2.1 IoU高于阈值则认为是同一instance
                    2.2.1.1 当前box Scores更高,则替换
                    2.2.1.2 当前box scores更低,则删除当前box
                2.2.2 否则认为当前box是新的instance
        '''

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()
        pred_instances_pred_boxes_3d = pred_instances.get("pred_boxes_3d")
        pred_box_size = pred_instances_pred_boxes_3d.dims.cpu().numpy()
        init_id = all_pred_box.init_id.cpu().numpy()  # 置信度

        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]
        
        for idx in cur_keep_idx:
            kf_count = []
            kf_ind = []
            cur_box = cur_2d_box[idx]

            '''
            only deal with the small ones length/weight/height < 35cm
            '''
            cur_box_size = pred_box_size[idx,:3]
            cur_corners = pred_instances_pred_boxes_3d.corners.cpu().numpy()[idx]

            '''
            principle 1: filter large box
            '''

            if np.max(cur_box_size)>0.35:
                continue


            #TODO:优化为只找有overlap的帧
            for kfid in range(0, frame_id, gap):
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                if corr.shape[0] == 0:
                    continue
                else:
                    cur_match = corr[:,2:]
                    cur_mask = Instances3D.points_in_box(cur_match,cur_box) #bool type
                    cur_cor_count = np.sum(cur_mask)
                    if cur_cor_count<1: #当前box有correspondence
                        continue
                    else:
 
                        valid_frame = Instances3D.check_boxes_overlap(cur_corners,all_kf_pose[kfid],intrinsic.cpu().numpy(),H,W)
                        '''
                        principle 2: filter frames that are not visible in the current box 
                        '''
                        if valid_frame:
                            kf_count.append(cur_cor_count)
                            kf_ind.append(kfid)
            # if len(kf_count)==0:
            #     print(frame_id,"no corr",idx,"remove",idx+N_glo)
            #     keep_idx = keep_idx[keep_idx!=(idx+N_glo)]
            print("kf_count", kf_count  )
            print("kf_ind", kf_ind  )
            '''
            NEW ADDED! remove extremely small boxes,
            '''

           
            # if frame_id==550:
            # if idx==4:
            #     print(idx,"kf_count:",kf_count)
            #     print(idx,"kf_ind:",kf_ind)
            #     print(idx,"N_glo",N_glo,"N_cur",N_cur)

            '''
            principle 3: only consider those have correspondence in past keyframes
            '''

            if len(kf_count)>0:

                data = np.array(kf_count)
                sorted_indices = np.argsort(data)  # 返回排序后的索引
                best_kfid = kf_ind[sorted_indices[-1]]
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{best_kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                cur_match = corr[:,2:]
                best_kf_match = corr[:,:2]
                cur_mask = Instances3D.points_in_box(cur_match,cur_box) 
                selected_bestkf_pixels = best_kf_match[cur_mask] #[N,2]
                #

                bestkf_pose =np.loadtxt(f'/media/lyq/data/dataset/ScanNet/scene0169_00/frames/pose/{best_kfid}.txt').reshape(4,4) #[4,4]

                # if idx == 4:
                #     print("bestkf_pose",bestkf_pose,'K',intrinsic.cpu().numpy())

                boxes_global = global_pred_box.get('pred_boxes_3d')
                boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] # [N,8,3]

                #transformed to world coordinate
                # boxes_3d = Instances3D.transform_boxes(boxes_3d,global_pose[global_keep_idx,:3,:3],global_pose[global_keep_idx,:3,3]) #[N_glo-nms,8,3]

    
                boxes_2d= Instances3D.project_3d_to_2d_box(boxes_3d, intrinsic.cpu().numpy(), bestkf_pose ,H, W, frame_id=frame_id) #[N_glo-nms,4]
                # if idx == 4:
                #     print(idx,'boxes_2d',boxes_2d)
                    # print(idx,'boxes_3d',boxes_3d)

                box_iou, overlap_A = Instances3D.IoU_2D(selected_bestkf_pixels,boxes_2d) #[N_glo-nms]
                # if idx == 4:
                #     print(idx,'selected_bestkf_pixels',selected_bestkf_pixels)
                corrponding_boxid = np.argmax(box_iou)

                # if frame_id==550:
                # if idx == 4:
                #     # print("boxes_global",len(boxes_global))
                #     # print("boxes_3d",boxes_3d.shape)
                #     # print("boxes_2d",boxes_2d.shape)
                #     # print("debug","frame_id:",frame_id,"cur box:",idx,"box_iou:",box_iou,'best kfid:',best_kfid)
                #     print(idx,"corrponding_boxid",corrponding_boxid,'box_iou',box_iou)

                '''
                principle 4: box with large IoU with old boxes in past keyframes
                '''
                if (box_iou[corrponding_boxid] > threshold): #0.1 #0.33
                    corresponding_idx = global_keep_idx[corrponding_boxid]
                
                    if global_box_scores[corresponding_idx]<cur_2d_box_scores[idx]:
                        print("frame_id:",frame_id,"find corr",idx,"remove old ",corresponding_idx,"better:",idx)
                        keep_idx = keep_idx[keep_idx!=(corresponding_idx)]

                        #record box manager
                        all_pred_box.valid_num[idx+N_glo] += 1
                        keep_idx = box_manager.record_corr(idx+N_glo, [corresponding_idx], init_id, per_frame_ins_cam_pose,keep_idx)
                    else:
                        print("frame_id:",frame_id,"find corr",idx,"remove new",idx+N_glo,"old:",corresponding_idx)
                        keep_idx = keep_idx[keep_idx!=(idx+N_glo)]
                        #record box manager
                        all_pred_box.valid_num[corresponding_idx] += 1
                        keep_idx = box_manager.record_corr(corresponding_idx, [idx+N_glo], init_id, per_frame_ins_cam_pose,keep_idx)

                '''
                principle 5: completely new boxes but not enough correspondence
                '''
                # if np.max(kf_count)<10 and (box_iou[corrponding_boxid])<0.01:
                #     corresponding_idx = global_keep_idx[corrponding_boxid]
                #     # print('idx',idx,box_iou)
                #     # print("corrponding_boxid in global boxes:",corrponding_boxid)
                #     # print("keep_idx:",keep_idx)
                #     print("frame_id:",frame_id,"find corr",idx,"remove new, too small","count:",np.max(kf_count),'box iou:',box_iou[corrponding_boxid],'remove',idx+N_glo,'in keep_idx','remove',idx,'in current')
                #     keep_idx = keep_idx[keep_idx!=(idx+N_glo)]

        keep_idx = np.sort(keep_idx)
        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses, keep_idx


    def preprocess_mast3r(image0, image1, device="cuda", size=256):
        '''
        image0: [3,H,W]
        image1: [3,H,W]
        '''
        ratio = size/image0.shape[1]
        # 转换为CHW格式并添加batch维度 [1,3,H,W]
        tensor_chw0 = image0.permute(2, 0, 1).unsqueeze(0)
        # 定义Resize变换（输出尺寸顺序为[H,W]）
        re_H = int(ratio*image0.shape[0])
        resize_transform0 = T.Resize((re_H, size))  # 参数顺序(height, width)
        resized_tensor0 = resize_transform0(tensor_chw0)
        # 恢复HWC格式 [192,256,3]
        image0 = resized_tensor0.squeeze(0) #.permute(1, 2, 0)
        image0 = image0/255.
        # plt.figure()    
        # plt.imshow(image0.permute(1,2,0).cpu().numpy())
        # plt.show(block=True)
        # exit(0)


        tensor_chw1 = image1.permute(2, 0, 1).unsqueeze(0)
        # 定义Resize变换（输出尺寸顺序为[H,W]）
        resize_transform1 = T.Resize((int(ratio*image1.shape[0]), size))  # 参数顺序(height, width)
        resized_tensor1 = resize_transform1(tensor_chw1)
        # 恢复HWC格式 [192,256,3]
        image1 = resized_tensor1.squeeze(0) #.permute(1, 2, 0)
        image1 = image1/255.
        # print("image0",image0.shape)
        imgs = []
        imgs.append(dict(img=ImgNorm(image0)[None], true_shape=np.int32(
            [[re_H, size]]), idx=0, instance=str(0)))
        
        imgs.append(dict(img=ImgNorm(image1)[None], true_shape=np.int32(
            [[re_H, size]]), idx=1, instance=str(1))) # idx=len(imgs), instance=str(len(imgs))))
        # print("imgs debug",imgs[0]['img'].shape,imgs[0]['true_shape'])
        return imgs

    def predict_correspondence_mast3r(imgs, match_model, ratio, device='cuda'):

        output = inference([tuple(imgs)], match_model, device, batch_size=1, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=device, dist='dot', block_size=2**13)

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
        all_matches = np.concatenate([matches_im0,matches_im1],axis=-1)

        all_matches = all_matches.astype(np.float32)
        all_matches /= ratio #return to 480*640
        all_matches = all_matches.astype(np.int16)

        return all_matches

    def merge_by_corr_v8(match_model, kf_img_dict, box_manager, cur_keep_idx, pred_instances,  global_pred_box, all_pred_box, all_poses, per_frame_ins_cam_pose, frame_id, gap, mask, intrinsic,all_kf_pose,threshold=0.33,H=480,W=640):  
        '''
        查找与当前帧存在overlap的所有帧,
            1.如果当前box与所有kf都不存在cor,则认为是新Instance
            2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
            2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
                2.2.1 IoU高于阈值则认为是同一instance
                    2.2.1.1 当前box Scores更高,则替换
                    2.2.1.2 当前box scores更低,则删除当前box
                2.2.2 否则认为当前box是新的instance
        '''

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()
        pred_instances_pred_boxes_3d = pred_instances.get("pred_boxes_3d")
        pred_box_size = pred_instances_pred_boxes_3d.dims.cpu().numpy()
        pred_corners = pred_instances_pred_boxes_3d.corners.cpu().numpy()
        init_id = all_pred_box.init_id.cpu().numpy()  # 置信度

        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]
        
        small_idx = []
        for idx in cur_keep_idx:
            cur_box_size = pred_box_size[idx,:3]
            '''
            only deal with the small ones length/weight/height < 35cm
            '''
            if np.max(cur_box_size)>0.35:
                continue
            small_idx.append(idx)
        if len(small_idx) == 0:
            keep_idx = np.sort(keep_idx)
            all_pred_box = all_pred_box[keep_idx]
            all_poses = all_poses[keep_idx]
            return all_pred_box, all_poses, keep_idx
        # valid kf check
        past_kfidx = np.arange(0,frame_id,gap) #[M,4,4]
        cur_small_corners = pred_corners[small_idx] #[N,8,3]
        
        # 提取所有 pose 并拼接
        all_kf_pose = np.stack(list(all_kf_pose.values()), axis=0)  # 形状为 [M, 4, 4]
        all_kf_pose = all_kf_pose[:-1,...]
        valid_frame = Instances3D.check_boxes_visibility(cur_small_corners, all_kf_pose, intrinsic.cpu().numpy(), H, W)   # [M] mask

        valid_kfidx = past_kfidx[valid_frame]
        valid_kfidx = past_kfidx[-1:]

        for idx in small_idx:
            kf_count = []
            kf_ind = []
            cur_box = cur_2d_box[idx]
            corr_dict = {}
            if np.sum(valid_frame)>0:
                for kfid in valid_kfidx:
                    #compute correspondence 
                    re_size = 512 #256
                    image0 = kf_img_dict[kfid]
                    image1 = kf_img_dict[frame_id]
                    ratio = re_size/kf_img_dict[frame_id].shape[1]

                    processed_imgs = Instances3D.preprocess_mast3r(image0, image1, size=re_size)
                    corr = Instances3D.predict_correspondence_mast3r(processed_imgs, match_model, ratio)

                    # corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{kfid}_{frame_id}.npy"
                    # corr = np.load(corr_path) #[N,4]

                    corr_dict[kfid] = corr

                    if corr.shape[0] == 0:
                        continue

                    cur_match = corr[:,2:] # corr: [N,4]
                    cur_mask = Instances3D.points_in_box(cur_match,cur_box) #bool type
                    cur_cor_count = np.sum(cur_mask)
                    if cur_cor_count<1: #当前box has no correspondence
                        continue
                    kf_count.append(cur_cor_count)
                    kf_ind.append(kfid)
            '''
            principle 1: only consider those have correspondence in past keyframes
            '''
            # print("kf_count", kf_count  )
            # print("kf_ind", kf_ind  )
            if len(kf_count)>0:

                data = np.array(kf_count)
                sorted_indices = np.argsort(data)  # 返回排序后的索引
                best_kfid = kf_ind[sorted_indices[-1]]
                print("best_kfid",best_kfid)
                #TODO:有没有办法可以直接找到这个best_kfid?
                #TODO:check一下是不是都是找的最近的?

                # corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{best_kfid}_{frame_id}.npy"
                # corr = np.load(corr_path) #[N,4]

                corr = corr_dict[best_kfid] #int


                # img0 = kf_img_dict[best_kfid].cpu().numpy()
                # img1 = kf_img_dict[frame_id].cpu().numpy()
                # img = np.concatenate((img0, img1), axis=1)
                # img /= 255.
      
                # plt.figure()
                # plt.imshow(img)

                # num_matches = corr.shape[0]
                # n_viz = 100 #num_matches ##100 #num_matches #20 #20
                # print("num_matches",num_matches)
                # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                # viz_matches_im0, viz_matches_im1 = corr[:,:2][match_idx_to_viz], corr[:,2:][match_idx_to_viz]
                # cmap = plt.get_cmap('jet')
                # W0 = img0.shape[1]
                # for i in range(n_viz):
                #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                #     # print("viz_matches_im1",viz_matches_im0)
                #     # print("viz_matches_im1",viz_matches_im1)

                #     plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                # plt.show(block=True)




                cur_match = corr[:,2:]
                best_kf_match = corr[:,:2]
                cur_mask = Instances3D.points_in_box(cur_match,cur_box) 
                selected_bestkf_pixels = best_kf_match[cur_mask] #[N,2]
                #
                bestkf_pose = all_kf_pose[best_kfid//gap] #[4,4]

                boxes_global = global_pred_box.get('pred_boxes_3d')
                boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] # [N,8,3]

                boxes_2d= Instances3D.project_3d_to_2d_box(boxes_3d, intrinsic.cpu().numpy(), bestkf_pose ,H, W, frame_id=frame_id) #[N_glo-nms,4]
    
                box_iou, overlap_A = Instances3D.IoU_2D(selected_bestkf_pixels,boxes_2d) #[N_glo-nms]

                corrponding_boxid = np.argmax(box_iou)

                '''
                principle 2: box with large IoU with old boxes in past keyframes
                '''
                if (box_iou[corrponding_boxid] > threshold): #0.1 #0.33
                    corresponding_idx = global_keep_idx[corrponding_boxid]
                
                    if global_box_scores[corresponding_idx]<cur_2d_box_scores[idx]:
                        print("frame_id:",frame_id,"find corr",idx,"remove old ",corresponding_idx,"better:",idx)
                        keep_idx = keep_idx[keep_idx!=(corresponding_idx)]

                        #record box manager
                        all_pred_box.valid_num[idx+N_glo] += 1
                        keep_idx = box_manager.record_corr(idx+N_glo, [corresponding_idx], init_id, per_frame_ins_cam_pose,keep_idx)
                    else:
                        print("frame_id:",frame_id,"find corr",idx,"remove new",idx+N_glo,"old:",corresponding_idx)
                        keep_idx = keep_idx[keep_idx!=(idx+N_glo)]
                        #record box manager
                        all_pred_box.valid_num[corresponding_idx] += 1
                        keep_idx = box_manager.record_corr(corresponding_idx, [idx+N_glo], init_id, per_frame_ins_cam_pose,keep_idx)


        keep_idx = np.sort(keep_idx)
        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses, keep_idx

    def merge_by_corr_v9(cfg, box_manager, cur_keep_idx, cur_success_nms,pred_instances,  global_pred_box, all_pred_box, all_poses, per_frame_ins_cam_pose, frame_id, mask, intrinsic, all_kf_pose,threshold=0.33,H=480,W=640):  

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()
        pred_instances_pred_boxes_3d = pred_instances.get("pred_boxes_3d")
        pred_box_size = pred_instances_pred_boxes_3d.dims.cpu().numpy()

        init_id = all_pred_box.init_id.cpu().numpy()  # 置信度

        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]
        # print("global_keep_idx",global_keep_idx)
        small_idx = []
        # print("cur_success_nms",cur_success_nms)
        for idx in cur_keep_idx:
            cur_box_size = pred_box_size[idx,:3]
            '''
            only deal with the small ones length/weight/height < 35cm
            '''
            if np.max(cur_box_size)>cfg['box_fusion']['small_size'] or idx in cur_success_nms:
                continue
            small_idx.append(idx)
        # print("small_idx",small_idx)

        #
        # small_idx=[]

        if len(small_idx) == 0:
            keep_idx = np.sort(keep_idx)
            all_pred_box = all_pred_box[keep_idx]
            all_poses = all_poses[keep_idx]
            return all_pred_box, all_poses, keep_idx

        cur_pose = all_kf_pose[frame_id] #[4,4]

        for idx in small_idx:
   
            boxes_global = global_pred_box.get('pred_boxes_3d')
            boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] # [N,8,3]


            boxes_2d= Instances3D.project_3d_to_2d_box(boxes_3d, intrinsic.cpu().numpy(), cur_pose ,H, W, frame_id=frame_id) #[N_glo-nms,4]

            cur_small_box_2d = cur_2d_box[idx]
            if len(boxes_2d) == 0:
                continue
            box_iou = Instances3D.IoU_2D_box(cur_small_box_2d,boxes_2d) #[N_glo-nms]

            boxes_3d_dims = boxes_global.dims.cpu().numpy()[global_keep_idx,...] # #[N_glo-nms, 3]
            global_small_mask = np.max(boxes_3d_dims, axis=1) < cfg['box_fusion']['small_size']+0.1 # [N_glo-nms]
            box_iou = box_iou * global_small_mask

            corrponding_boxid = np.argmax(box_iou)
            # print("idx in cur",idx,corrponding_boxid,box_iou[corrponding_boxid],box_iou)
            '''
            principle 2: box with large IoU with old boxes in past keyframes
            '''
            if (box_iou[corrponding_boxid] > threshold): #0.1 #0.33
                corresponding_idx = global_keep_idx[corrponding_boxid]
                
                if global_box_scores[corresponding_idx]<cur_2d_box_scores[idx]:
                    print("frame_id:",frame_id,"find corr",idx,"remove old ",corresponding_idx,'iou:',box_iou[corrponding_boxid],"better:",idx)
                    keep_idx = keep_idx[keep_idx!=(corresponding_idx)]

                    #record box manager
                    all_pred_box.valid_num[idx+N_glo] += 1
                    keep_idx = box_manager.record_corr(idx+N_glo, [corresponding_idx], init_id, per_frame_ins_cam_pose,keep_idx)
                else:
                    print("frame_id:",frame_id,"find corr",idx,"remove new",idx+N_glo,"old:",corresponding_idx,'worse')
                    keep_idx = keep_idx[keep_idx!=(idx+N_glo)]
                    #record box manager
                    all_pred_box.valid_num[corresponding_idx] += 1
                    keep_idx = box_manager.record_corr(corresponding_idx, [idx+N_glo], init_id, per_frame_ins_cam_pose,keep_idx)


        keep_idx = np.sort(keep_idx)
        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses, keep_idx
    
    def merge_by_corr_v10(depth_gt, box_manager, cur_keep_idx, pred_instances,  global_pred_box, all_pred_box, all_poses, per_frame_ins_cam_pose, frame_id, gap, mask, intrinsic,all_kf_pose,threshold=0.33,H=480,W=640):  

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()
        pred_instances_pred_boxes_3d = pred_instances.get("pred_boxes_3d")
        pred_box_size = pred_instances_pred_boxes_3d.dims.cpu().numpy()

        init_id = all_pred_box.init_id.cpu().numpy()  # 置信度

        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]
        
        small_idx = []
        for idx in cur_keep_idx:
            cur_box_size = pred_box_size[idx,:3]
            '''
            only deal with the small ones length/weight/height < 35cm
            '''
            if np.max(cur_box_size)>0.35:
                continue
            small_idx.append(idx)
        if len(small_idx) == 0:
            keep_idx = np.sort(keep_idx)
            all_pred_box = all_pred_box[keep_idx]
            all_poses = all_poses[keep_idx]
            return all_pred_box, all_poses, keep_idx

        cur_pose = all_kf_pose[frame_id] #[4,4]

        for idx in small_idx:
   
            boxes_global = global_pred_box.get('pred_boxes_3d')
            boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] # [N,8,3]

            boxes_2d= Instances3D.project_3d_to_2d_box_depth(depth_gt, boxes_3d, intrinsic.cpu().numpy(), cur_pose ,H, W, frame_id=frame_id) #[N_glo-nms,4]

            cur_small_box_2d = cur_2d_box[idx]
            box_iou = Instances3D.IoU_2D_box(cur_small_box_2d,boxes_2d) #[N_glo-nms]

            corrponding_boxid = np.argmax(box_iou)

            '''
            principle 2: box with large IoU with old boxes in past keyframes
            '''
            if (box_iou[corrponding_boxid] > threshold): #0.1 #0.33
                corresponding_idx = global_keep_idx[corrponding_boxid]
            
                if global_box_scores[corresponding_idx]<cur_2d_box_scores[idx]:
                    print("frame_id:",frame_id,"find corr",idx,"remove old ",corresponding_idx,"better:",idx)
                    keep_idx = keep_idx[keep_idx!=(corresponding_idx)]

                    #record box manager
                    all_pred_box.valid_num[idx+N_glo] += 1
                    keep_idx = box_manager.record_corr(idx+N_glo, [corresponding_idx], init_id, per_frame_ins_cam_pose,keep_idx)
                else:
                    print("frame_id:",frame_id,"find corr",idx,"remove new",idx+N_glo,"old:",corresponding_idx)
                    keep_idx = keep_idx[keep_idx!=(idx+N_glo)]
                    #record box manager
                    all_pred_box.valid_num[corresponding_idx] += 1
                    keep_idx = box_manager.record_corr(corresponding_idx, [idx+N_glo], init_id, per_frame_ins_cam_pose,keep_idx)


        keep_idx = np.sort(keep_idx)
        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses, keep_idx


    def merge_by_corr_v6_v2(cur_keep_idx, pred_instances,  global_pred_box, global_pose, all_pred_box, all_poses, frame_id, gap, mask, intrinsic,all_kf_pose,threshold=0.33,H=480,W=640):  
        '''
        查找与当前帧存在overlap的所有帧,
            1.如果当前box与所有kf都不存在cor,则认为是新Instance
            2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
            2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
                2.2.1 IoU高于阈值则认为是同一instance
                    2.2.1.1 当前box Scores更高,则替换
                    2.2.1.2 当前box scores更低,则删除当前box
                2.2.2 否则认为当前box是新的instance
        '''

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()
        pred_instances_pred_boxes_3d = pred_instances.get("pred_boxes_3d")
        pred_box_size = pred_instances_pred_boxes_3d.dims.cpu().numpy()
        
        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]
        
        for idx in cur_keep_idx:
            kf_count = []
            kf_ind = []
            cur_box = cur_2d_box[idx]

            '''
            only deal with the small ones length/weight/height < 35cm
            '''
            cur_box_size = pred_box_size[idx,:3]
            cur_corners = pred_instances_pred_boxes_3d.corners.cpu().numpy()[idx]

            '''
            principle 1: filter large box
            '''

            if np.max(cur_box_size)>0.35:
                continue


            #TODO:优化为只找有overlap的帧
            for kfid in range(0, frame_id, gap):
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                if corr.shape[0] == 0:
                    continue
                else:
                    cur_match = corr[:,2:]
                    cur_mask = Instances3D.points_in_box(cur_match,cur_box) #bool type
                    cur_cor_count = np.sum(cur_mask)
                    if cur_cor_count<1: #当前box有correspondence
                        continue
                    else:
 
                        valid_frame = Instances3D.check_boxes_overlap(cur_corners,all_poses[-1,:3,:3], all_poses[-1,:3,3],all_kf_pose[kfid],intrinsic.cpu().numpy(),H,W)

                        '''
                        principle 2: filter frames that are not visible in the current box 
                        '''

                        if valid_frame:
                            kf_count.append(cur_cor_count)
                            kf_ind.append(kfid)
            # if len(kf_count)==0:
            #     print(frame_id,"no corr",idx,"remove",idx+N_glo)
            #     keep_idx = keep_idx[keep_idx!=(idx+N_glo)]

            '''
            NEW ADDED! remove extremely small boxes,
            '''

           
            # if frame_id==550:
            # if idx==4:
            #     print(idx,"kf_count:",kf_count)
            #     print(idx,"kf_ind:",kf_ind)
            #     print(idx,"N_glo",N_glo,"N_cur",N_cur)

            '''
            principle 3: only consider those have correspondence in past keyframes
            '''

            if len(kf_count)>0:

                data = np.array(kf_count)
                sorted_indices = np.argsort(data)  # 返回排序后的索引
                best_kfid = kf_ind[sorted_indices[-1]]
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{best_kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                cur_match = corr[:,2:]
                best_kf_match = corr[:,:2]
                cur_mask = Instances3D.points_in_box(cur_match,cur_box) 
                selected_bestkf_pixels = best_kf_match[cur_mask] #[N,2]
                #

                bestkf_pose =np.loadtxt(f'/media/lyq/data/dataset/ScanNet/scene0169_00/frames/pose/{best_kfid}.txt').reshape(4,4) #[4,4]

                # if idx == 4:
                #     print("bestkf_pose",bestkf_pose,'K',intrinsic.cpu().numpy())

                boxes_global = global_pred_box.get('pred_boxes_3d')
                boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] #N,8,3
                # print("global_keep_idx",global_keep_idx)
                #transformed to world coordinate
                boxes_3d = Instances3D.transform_boxes(boxes_3d,global_pose[global_keep_idx,:3,:3],global_pose[global_keep_idx,:3,3]) #[N_glo-nms,8,3]

    
                boxes_2d= Instances3D.project_3d_to_2d_box(boxes_3d, intrinsic.cpu().numpy(), bestkf_pose ,H, W, frame_id=frame_id) #[N_glo-nms,4]
                # if idx == 4:
                #     print(idx,'boxes_2d',boxes_2d)
                    # print(idx,'boxes_3d',boxes_3d)

                box_iou, overlap_A = Instances3D.IoU_2D(selected_bestkf_pixels,boxes_2d) #[N_glo-nms]
                # if idx == 4:
                #     print(idx,'selected_bestkf_pixels',selected_bestkf_pixels)
                corrponding_boxid = np.argmax(box_iou)



                # if frame_id==550:
                # if idx == 4:
                #     # print("boxes_global",len(boxes_global))
                #     # print("boxes_3d",boxes_3d.shape)
                #     # print("boxes_2d",boxes_2d.shape)
                #     # print("debug","frame_id:",frame_id,"cur box:",idx,"box_iou:",box_iou,'best kfid:',best_kfid)
                #     print(idx,"corrponding_boxid",corrponding_boxid,'box_iou',box_iou)

                '''
                principle 4: box with large IoU with old boxes in past keyframes
                '''
                if (box_iou[corrponding_boxid] > threshold): #0.1 #0.33
                    corresponding_idx = global_keep_idx[corrponding_boxid]
                
                    if global_box_scores[corresponding_idx]<cur_2d_box_scores[idx]:
                        print("frame_id:",frame_id,"find corr",idx,"remove old ",corresponding_idx,"better:",idx)
                        keep_idx = keep_idx[keep_idx!=(corresponding_idx)]
                    else:
                        print("frame_id:",frame_id,"find corr",idx,"remove new",idx+N_glo)
                        keep_idx = keep_idx[keep_idx!=(idx+N_glo)]

                '''
                TODO:有个Bug，如果当前box对应的global box被nms掉了，你的iou就肯定是0
                '''
                '''
                principle 5: completely new boxes but not enough correspondence
                '''
                # if np.max(kf_count)<10 and (box_iou[corrponding_boxid])<0.01:
                #     corresponding_idx = global_keep_idx[corrponding_boxid]
                #     # print('idx',idx,box_iou)
                #     # print("corrponding_boxid in global boxes:",corrponding_boxid)
                #     # print("keep_idx:",keep_idx)
                #     print("frame_id:",frame_id,"find corr",idx,"remove new, too small","count:",np.max(kf_count),'box iou:',box_iou[corrponding_boxid],'remove',idx+N_glo,'in keep_idx','remove',idx,'in current')
                #     keep_idx = keep_idx[keep_idx!=(idx+N_glo)]


        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses

    def merge_by_corr_v5(cur_keep_idx, pred_instances,  global_pred_box, global_pose, all_pred_box, all_poses, frame_id, gap, mask, intrinsic,threshold=0.33):  
        '''
        查找与当前帧存在overlap的所有帧,
            1.如果当前box与所有kf都不存在cor,则认为是新Instance
            2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
            2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
                2.2.1 IoU高于阈值则认为是同一instance
                    2.2.1.1 当前box Scores更高,则替换
                    2.2.1.2 当前box scores更低,则删除当前box
                2.2.2 否则认为当前box是新的instance
        '''

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()



        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]
        
        for idx in cur_keep_idx:
            kf_count = []
            kf_ind = []
            cur_box = cur_2d_box[idx]
            #TODO:优化为只找有overlap的帧
            for kfid in range(0,frame_id,gap):
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                if corr.shape[0] == 0:
                    continue
                else:
                    cur_match = corr[:,2:]
                    cur_mask = Instances3D.points_in_box(cur_match,cur_box) #bool type
                    cur_cor_count = np.sum(cur_mask)
                    if cur_cor_count<1: #当前box有correspondence
                        continue
                    else:
                        kf_count.append(cur_cor_count)
                        kf_ind.append(kfid)
            # if len(kf_count)==0:
            #     print(frame_id,"no corr",idx,"remove",idx+N_glo)
            #     keep_idx = keep_idx[keep_idx!=(idx+N_glo)]

            # if frame_id==475:
            #     print(idx,"kf_count:",kf_count)
            #     print(idx,"kf_ind:",kf_ind)

            if len(kf_count)>0:

                data = np.array(kf_count)
                sorted_indices = np.argsort(data)  # 返回排序后的索引
                best_kfid = kf_ind[sorted_indices[-1]]
                corr_path = f"/media/lyq/data/dataset/ScanNet/scene0169_00/frames/matches/{best_kfid}_{frame_id}.npy"
                corr = np.load(corr_path) #[N,4]
                cur_match = corr[:,2:]
                best_kf_match = corr[:,:2]
                cur_mask = Instances3D.points_in_box(cur_match,cur_box) 
                selected_bestkf_pixels = best_kf_match[cur_mask] #[N,2]
                #
                bestkf_pose =np.loadtxt(f'/media/lyq/data/dataset/ScanNet/scene0169_00/frames/pose/{best_kfid}.txt').reshape(4,4) #[4,4]
                
                boxes_global = global_pred_box.get('pred_boxes_3d')

                boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] #N,8,3
                #transformed to world coordinate
                boxes_3d = Instances3D.transform_boxes(boxes_3d,global_pose[global_keep_idx,:3,:3],global_pose[global_keep_idx,:3,3]) #[N_glo,8,3]

                boxes_2d = Instances3D.project_3d_to_2d_box(boxes_3d, intrinsic.cpu().numpy(), bestkf_pose) #[N_glo,4]

                box_iou, overlap_A = Instances3D.IoU_2D(selected_bestkf_pixels,boxes_2d) #[N_glo]
                corrponding_boxid = np.argmax(box_iou)

                # if frame_id==475:
                #     print("debug","frame_id:",frame_id,"cur box:",idx,"box_iou:",box_iou)

                if (box_iou[corrponding_boxid] > threshold): #0.1 #0.33
                    corresponding_idx = global_keep_idx[corrponding_boxid]
                
                    if global_box_scores[corresponding_idx]<cur_2d_box_scores[idx]:
                        print("frame_id:",frame_id,"find corr",idx,"remove old ",corresponding_idx)
                        keep_idx = keep_idx[keep_idx!=(corresponding_idx)]
                    else:
                        print("frame_id:",frame_id,"find corr",idx,"remove new",idx+N_glo)
                        keep_idx = keep_idx[keep_idx!=(idx+N_glo)]


        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses

    def augment_vertices(corners):
        # 边列表（每条边由两个顶点索引定义）
        edges = [
            [0, 1], [0, 4], [1, 5], [4, 5],
            [2, 3], [2, 6], [6, 7], [3, 7],
            [0, 3], [4, 7], [1, 2], [5, 6]
        ]

        # 计算每条边的中点
        midpoints = []
        for edge in edges:
            v1 = corners[edge[0]]
            v2 = corners[edge[1]]
            midpoint = (v1 + v2) / 2
            midpoints.append(midpoint)

        # 合并原始顶点与中点
        combined = np.vstack([corners, midpoints])
        
        return combined

    def check_intersection(corners1, corners2):
        """
        判断两个box是否存在交集
        :param corners1: 凸包顶点 (8x3数组)
        :param corners2: 凸包顶点 (8x3数组)
        :return: 布尔数组 True or False
        """
        # 计算凸包方程
        hull1 = ConvexHull(corners1)
        hull2 = ConvexHull(corners2)

        corners1 = Instances3D.augment_vertices(corners1)
        corners2 = Instances3D.augment_vertices(corners2)

        equations1 = hull1.equations  # 形状 [K,4]，每行对应ax + by + cz + d <= 0
        equations2 = hull2.equations 
        

        # 判断corners1是否有在corners2构成的凸包内
        dot_products1 = np.dot(corners1, equations2[:, :3].T) + equations2[:, 3]  # 形状 [N, K]
        # 判断每个点是否在所有面的负半空间
        mask1 = np.all(dot_products1 <= 1e-6, axis=1)  # 形状 [N,]

        # 判断corners2是否有在corners1构成的凸包内
        dot_products2 = np.dot(corners2, equations1[:, :3].T) + equations1[:, 3]  # 形状 [N, K]
        # 判断每个点是否在所有面的负半空间
        mask2 = np.all(dot_products2 <= 1e-6, axis=1)  # 形状 [N,]

        sum_of_mask = np.sum(mask1) + np.sum(mask2)

        if sum_of_mask > 0:
            return True
        else:
            return False
    


    def batch_in_convex_hull_3d(points, corners):
        """
        批量判断点是否在3D凸包内
        :param points: 待检测点集 (Nx3数组)
        :param corners: 凸包顶点 (Mx3数组)
        :return: 布尔数组 (N,)
        """
        # 计算凸包方程
        hull = ConvexHull(corners)
        equations = hull.equations  # 形状 [K,4]，每行对应ax + by + cz + d <= 0
        # print("equations",equations)
        # 向量化计算：利用广播机制同时处理所有点
        dot_products = np.dot(points, equations[:, :3].T) + equations[:, 3]  # 形状 [N, K]
        
        # 判断每个点是否在所有面的负半空间
        mask = np.all(dot_products <= 1e-6, axis=1)  # 形状 [N,]
        
        return mask

    def obb_iou(corners1,corners2):
        # 执行检测和可视化
        # print("checking")
        results = Instances3D.check_intersection(corners1,corners2)

        if results:
            # print("case1")
            # 假设box1_corners和box2_corners是形状为[8,3]的数组
            all_corners = np.concatenate([corners1, corners2], axis=0)

            xmin, ymin, zmin = np.min(all_corners, axis=0)
            xmax, ymax, zmax = np.max(all_corners, axis=0)

            # print(f"包围范围: x∈[{xmin}, {xmax}], y∈[{ymin}, {ymax}], z∈[{zmin}, {zmax}]")

            # 定义每个轴的采样数量
            num_samples_per_axis = 10

            # 生成各轴的采样点
            x_samples = np.linspace(xmin, xmax, num_samples_per_axis)
            y_samples = np.linspace(ymin, ymax, num_samples_per_axis)
            z_samples = np.linspace(zmin, zmax, num_samples_per_axis)

            # 构建三维网格
            xx, yy, zz = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')

            # 展平并组合采样点
            sampled_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

            # print("采样点形状:", sampled_points.shape)  # 输出: (8000, 3)
            # print("采样点:", sampled_points)  # 输出: (8000, 3)
            # print("corners1",corners1.shape,"corners2",corners2.shape)
            #TODO:容易卡在这段，如果点数较多的话
            mask1 = Instances3D.batch_in_convex_hull_3d(sampled_points, corners1)
            mask2 = Instances3D.batch_in_convex_hull_3d(sampled_points, corners2)
            common_mask = mask1 * mask2
            count1 = np.sum(mask1)
            count2 = np.sum(mask2)
            common_count = np.sum(common_mask)
            # print("mask1",mask1,np.sum(mask1))
            # print("mask2",mask2,np.sum(mask2))
            # print("common_mask",common_mask,np.sum(common_mask))
            #compute IoU according to the mask
            IoU = common_count/(count1+count2-common_count+1e-6)
            # print("IoU",IoU)
            return IoU
        else:
            # print("case2")
            IoU = 0.0
            # print("IoU",IoU)
            return IoU

    
    def check_boxes_visibility(boxes, projected_poses, K, H, W):
        """
        boxes: [N,8,3]    # N个3D框的角点
        projected_poses: [M,4,4]  # M个相机的位姿矩阵
        K: 内参矩阵 [3,3]
        H, W: 图像高度和宽度
        Returns: [M]的布尔掩码，True表示该图像中至少有一个3D框可见
        """
        N, M = boxes.shape[0], projected_poses.shape[0]
        
        # 1. 转换为齐次坐标 [N,8,4]
        boxes_homo = np.concatenate([boxes, np.ones((N,8,1))], axis=-1)  # [N,8,4]
        
        # 2. 转换到每个相机坐标系 [M,N,8,3]
        pose_inv = np.linalg.inv(projected_poses)  # [M,4,4]
        #[M,4,4] @ [N,4,8]
        boxes_cam = np.einsum('mij,njk->mnik', pose_inv, boxes_homo.transpose(0,2,1))  #[M,N,4,8]
        boxes_cam = boxes_cam.transpose(0,1,3,2)
        boxes_cam = boxes_cam[..., :3]# [M,N,8,3]

        # 3. 投影到图像平面 [M,N,8,2]
        X, Y, Z = boxes_cam[...,0], boxes_cam[...,1], boxes_cam[...,2]
        u = (K[0,0] * X / Z) + K[0,2]  # [M,N,8]
        v = (K[1,1] * Y / Z) + K[1,2]
        
        # 4. 计算有效掩码
        valid_mask = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)  # [M,N,8]
        
        # 5. 聚合每个相机的可见性（任何框的任意角点有效）
        visibility_mask = np.any(valid_mask.reshape(M, N*8), axis=1)  # [M]
        
        return visibility_mask

    def check_boxes_overlap(boxes, projected_pose, K, H, W):
        """
        boxes: [8, 3]   # N个3D框的8个角点坐标
        rotation: [3, 3]    # 每个框对应的旋转矩阵
        translation: [3] # 每个框对应的平移向量
        返回: [8, 3]     # 变换后的3D框坐标
        """
 
        boxes_3d = boxes[None,...]
  

        """
        boxes_3d: [N, 8, 3]  (世界坐标系下的3D框角点)
        K: [3, 3]             (内参矩阵)
        pose: [4, 4]          (外参矩阵，世界坐标系到相机坐标系)
        返回: [N, 4]          (2D包围框: [x_min, y_min, x_max, y_max])
        """
        N = boxes_3d.shape[0]
        boxes_2d = np.zeros((N, 4))
        
        # 扩展为齐次坐标 [N, 8, 4]
        ones = np.ones((N, 8, 1))
        boxes_homo = np.concatenate([boxes_3d, ones], axis=2)
        
        # 转换到相机坐标系 [N, 8, 4]
        pose_inv = np.linalg.inv(projected_pose)
        boxes_cam = np.dot(boxes_homo, pose_inv.T)
        
        # 提取相机坐标系下的坐标 [N, 8, 3]
        X = boxes_cam[..., 0]
        Y = boxes_cam[..., 1]
        Z = boxes_cam[..., 2]
        
        # 投影到图像平面 [N, 8, 2]
        u = (K[0, 0] * X / Z) + K[0, 2]
        v = (K[1, 1] * Y / Z) + K[1, 2]
        
        # 处理无效点（Z <= 0）
        valid_mask = (Z > 0) * (u>0) * (u<W) * (v>0) * (v<H)
        mask=False
        for i in range(N):
            valid_u = u[i][valid_mask[i]]
            valid_v = v[i][valid_mask[i]]
            # print("valid_u",valid_u)
            # print("valid_v",valid_v)
            if len(valid_u) >0:
                mask = True

        return mask

    # def check_boxes_overlap(boxes, rotation, translation, projected_pose, K, H, W):
    #     """
    #     boxes: [8, 3]   # N个3D框的8个角点坐标
    #     rotation: [3, 3]    # 每个框对应的旋转矩阵
    #     translation: [3] # 每个框对应的平移向量
    #     返回: [8, 3]     # 变换后的3D框坐标
    #     """
    #     # 1. 应用旋转变换
    #     rotated_boxes = np.einsum('ij,kj->ki', rotation, boxes)  # 矩阵乘法
        
    #     # 2. 应用平移变换
    #     transformed_boxes = rotated_boxes + translation[np.newaxis, :]  # 广播加法 

    #     boxes_3d = transformed_boxes[None,...]
  

    #     """
    #     boxes_3d: [N, 8, 3]  (世界坐标系下的3D框角点)
    #     K: [3, 3]             (内参矩阵)
    #     pose: [4, 4]          (外参矩阵，世界坐标系到相机坐标系)
    #     返回: [N, 4]          (2D包围框: [x_min, y_min, x_max, y_max])
    #     """
    #     N = boxes_3d.shape[0]
    #     boxes_2d = np.zeros((N, 4))
        
    #     # 扩展为齐次坐标 [N, 8, 4]
    #     ones = np.ones((N, 8, 1))
    #     boxes_homo = np.concatenate([boxes_3d, ones], axis=2)
        
    #     # 转换到相机坐标系 [N, 8, 4]
    #     pose_inv = np.linalg.inv(projected_pose)
    #     boxes_cam = np.dot(boxes_homo, pose_inv.T)
        
    #     # 提取相机坐标系下的坐标 [N, 8, 3]
    #     X = boxes_cam[..., 0]
    #     Y = boxes_cam[..., 1]
    #     Z = boxes_cam[..., 2]
        
    #     # 投影到图像平面 [N, 8, 2]
    #     u = (K[0, 0] * X / Z) + K[0, 2]
    #     v = (K[1, 1] * Y / Z) + K[1, 2]
        
    #     # 处理无效点（Z <= 0）
    #     valid_mask = (Z > 0) * (u>0) * (u<W) * (v>0) * (v<H)
    #     mask=False
    #     for i in range(N):
    #         valid_u = u[i][valid_mask[i]]
    #         valid_v = v[i][valid_mask[i]]
    #         # print("valid_u",valid_u)
    #         # print("valid_v",valid_v)
    #         if len(valid_u) >0:
    #             mask = True

    #     return mask


    def transform_boxes(boxes, rotation, translation):
        """
        boxes: [N, 8, 3]   # N个3D框的8个角点坐标
        pose: [N, 3, 3]    # 每个框对应的旋转矩阵
        translation: [N, 3] # 每个框对应的平移向量
        返回: [N, 8, 3]     # 变换后的3D框坐标
        """
        # 1. 应用旋转变换
        rotated_boxes = np.einsum('nij,nkj->nki', rotation, boxes)  # 矩阵乘法
        
        # 2. 应用平移变换
        transformed_boxes = rotated_boxes + translation[:, np.newaxis, :]  # 广播加法
        
        return transformed_boxes

    def merge_by_corr_v4(last_pred_instances, pred_instances,  global_pred_box, corr, framerecord, all_pred_box, all_poses, frame_id, gap): 
        '''
        查找与当前帧存在overlap的所有帧,
            1.如果当前box与所有kf都不存在cor,则认为是新Instance
            2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
            2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
                2.2.1 IoU高于阈值则认为是同一instance
                    2.2.1.1 当前box Scores更高,则替换
                    2.2.1.2 当前box scores更低,则删除当前box
                2.2.2 否则认为当前box是新的instance
        '''
        boxes_now = all_pred_box.get('pred_boxes_3d')
        # Hard-code these suffixes.
        centers_box=boxes_now.gravity_center.cpu().numpy()
        #transformed to world coordinate
        centers_box = np.squeeze(all_poses[:,:3,:3]@np.expand_dims(centers_box,axis=-1)) + all_poses[:,:3,3]
        sizes_size=boxes_now.dims.cpu().numpy()
        boxes = np.concatenate((centers_box,sizes_size),axis=-1) #[N_glo+N_cur]

        N_glo = len(global_pred_box)
        N_cur = len(pred_instances)

        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        lastkf_2d_box = last_pred_instances.pred_boxes.cpu().numpy()
        last_kf_match = corr[:,:2]
        cur_match = corr[:,2:]

        global_pred_box_scores = global_pred_box.scores.cpu().numpy()
        cur_pred_box_scores = pred_instances.scores.cpu().numpy()
        all_pred_box_scores = all_pred_box.scores.cpu().numpy()

        cur_l2s_inds = cur_pred_box_scores.argsort()[::-1]
        print("cur_l2s_inds:",cur_l2s_inds,cur_pred_box_scores)

        keep = np.arange(N_glo+N_cur) 
        mask = np.ones_like(keep,dtype=bool)
        '''
        当前Box的两种去路 1.新增 2.删除 3.替换 
        1.当前box与所有global box算IoU
        1.1 若存在于某个 box IoU大于阈值
        (a) scores比这些box中最高的还高,则替换 所有低于这个box的
            (a-1) 看看当前预测的剩余box里有没有和自己IoU很高的, 若存在比较scores,若cur box高,则用其替换掉所有上述box, 若cur box低, 则用那个新box替换掉所有上述box
            (a-2) 若不存在, 则替换 所有低于这个box的
        (b) scores比该box低,则删除
        1.2 若不存在任何一个box IoU大于阈值,考虑可能要新增,但要check correspondence
        (a) 如果与任何一个可见帧该box都不存在correspondence, 则新增
        (b) 如果存在与可见帧存在correspondence, 则挑选corr最多的那一帧, 找到该帧该box的global box, 判断该当前box的scores是否高于该global box, 如果高于, 则替换，如果低于, 则删除
        '''
        print("\n")
        # for idx in np.arange(len(pred_instances)):
        for idx in cur_l2s_inds:
            if mask[N_glo+idx] == False:
                continue
            cur_box_2d = cur_2d_box[idx]
            cur_box_3d = boxes[idx+N_glo]
            # ious = calculate_3d_iou(cur_box_3d,boxes[:N_glo])
            ious = calculate_3d_iou(cur_box_3d,boxes)

            iou_ind = np.where(ious>0.01)[0]
            iou_ind = [i for i in iou_ind if i !=idx+N_glo]
            if len(iou_ind)>0:
                #1.1
                if len(iou_ind) == 1:
                    max_scores_ind =  iou_ind[0]
                else:
                    iou_ind_scores = [all_pred_box_scores[temp_id] for temp_id in iou_ind]
                    max_scores_ind = iou_ind[np.argmax(iou_ind_scores)]
  
                #see the scores to determine which case
                print("max_scores_ind",max_scores_ind,all_pred_box_scores.shape,cur_pred_box_scores.shape)
                if all_pred_box_scores[max_scores_ind]<cur_pred_box_scores[idx]:
                #1.1-(a)
                    to_be_replaced = iou_ind
                    mask[to_be_replaced] = False
                    all_poses[max_scores_ind] = all_poses[idx+N_glo]
                    print("frame-",frame_id, "box-",idx,"1.1-(a) ","to_be_replaced:",to_be_replaced)
                else:
                #1.1-(b)
                    iou_ind.append(idx+N_glo)
                    #remove the best one from deleting list
                    iou_ind.remove(max_scores_ind)
                    to_be_replaced = iou_ind
                    mask[to_be_replaced] = False
                    print("frame-",frame_id,"box-",idx,"1.1-(b) ", "to_be_replaced:",to_be_replaced,max_scores_ind)
                framerecord.add_single(frame_id, idx, max_scores_ind)
            else:
                #1.2
                print("frame-",frame_id,"box-",idx,"case 1.2")
                mask[idx+N_glo] = True
                until_cur_mask = mask[:(idx+N_glo+1)]
                tmp_count = np.sum(until_cur_mask)
                framerecord.add_single(frame_id, idx, tmp_count-1)


        keep = keep[mask]
        keep=sorted(keep)
        all_pred_box = Instances3D.cat([global_pred_box,pred_instances])
        all_pred_box = all_pred_box[keep]
        all_poses = all_poses[keep]

        return all_pred_box, all_poses


    # def merge_by_corr_v3(last_pred_instances, pred_instances,  global_pred_box, corr, framerecord, all_pred_box, all_poses, frame_id, gap): 
    #     '''
    #     查找与当前帧存在overlap的所有帧,
    #         1.如果当前box与所有kf都不存在cor,则认为是新Instance
    #         2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
    #         2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
    #             2.2.1 IoU高于阈值则认为是同一instance
    #                 2.2.1.1 当前box Scores更高,则替换
    #                 2.2.1.2 当前box scores更低,则删除当前box
    #             2.2.2 否则认为当前box是新的instance
    #     '''
    #     boxes_now = all_pred_box.get('pred_boxes_3d')
    #     # Hard-code these suffixes.
    #     centers_box=boxes_now.gravity_center.cpu().numpy()
    #     #transformed to world coordinate
    #     centers_box = np.squeeze(all_poses[:,:3,:3]@np.expand_dims(centers_box,axis=-1)) + all_poses[:,:3,3]
    #     sizes_size=boxes_now.dims.cpu().numpy()
    #     boxes = np.concatenate((centers_box,sizes_size),axis=-1) #[N_glo+N_cur]

    #     N_glo = len(global_pred_box)
    #     N_cur = len(pred_instances)

    #     cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
    #     lastkf_2d_box = last_pred_instances.pred_boxes.cpu().numpy()
    #     last_kf_match = corr[:,:2]
    #     cur_match = corr[:,2:]

    #     global_pred_box_scores = global_pred_box.scores.cpu().numpy()
    #     cur_pred_box_scores = pred_instances.scores.cpu().numpy()

    #     keep = np.arange(N_glo+N_cur) 
    #     mask = np.ones_like(keep,dtype=bool)
    #     '''
    #     当前Box的两种去路 1.新增 2.删除 3.替换 
    #     1.当前box与所有global box算IoU
    #     1.1 若存在于某个 box IoU大于阈值
    #     (a) scores比该box高,则替换
    #     (b) scores比该box低,则删除
    #     1.2 若不存在任何一个box IoU大于阈值,考虑可能要新增,但要check correspondence
    #     (a) 如果与任何一个可见帧该box都不存在correspondence, 则新增
    #     (b) 如果存在与可见帧存在correspondence, 则挑选corr最多的那一帧, 找到该帧该box的global box, 判断该当前box的scores是否高于该global box, 如果高于, 则替换，如果低于, 则删除
    #     '''
    #     print("\n")
    #     for idx in np.arange(len(pred_instances)):
    #         cur_box_2d = cur_2d_box[idx]
    #         cur_box_3d = boxes[idx+N_glo]
    #         ious = calculate_3d_iou(cur_box_3d,boxes[:N_glo])
    #         iou_ind = np.where(ious>0.01)[0]
    #         if iou_ind.shape[0]>0:
    #             #1.1
    #             if iou_ind.shape[0] == 1:
    #                 max_scores_ind =  iou_ind[0]
    #             else:
    #                 valid_scores = [global_pred_box_scores[temp_id] for temp_id in iou_ind]
    #                 max_scores_ind = iou_ind[np.argmax(valid_scores)]
    #             #see the scores to determine which case
    #             if global_pred_box_scores[max_scores_ind]<cur_pred_box_scores[idx]:
    #             #1.1-(a)
    #                 Instances3D.modify_instance(max_scores_ind,idx,global_pred_box,pred_instances)
    #                 mask[idx+N_glo] = False
    #                 all_poses[max_scores_ind] = all_poses[idx+N_glo]
    #                 print("frame-",frame_id, "box-",idx,"1.1-(a) ","global idx:",max_scores_ind)
    #             else:
    #                 mask[idx+N_glo] = False
    #                 print("frame-",frame_id,"box-",idx,"1.1-(b) ", "global idx:",max_scores_ind)
    #             framerecord.add_single(frame_id, idx, max_scores_ind)
    #         else:
    #             #1.2
    #             print("frame-",frame_id,"box-",idx,"case 1.2")
    #             mask[idx+N_glo] = True
    #             until_cur_mask = mask[:(idx+N_glo+1)]
    #             tmp_count = np.sum(until_cur_mask)
    #             framerecord.add_single(frame_id, idx, tmp_count-1)

        

    #     keep = keep[mask]
    #     keep=sorted(keep)
    #     all_pred_box = Instances3D.cat([global_pred_box,pred_instances])
    #     all_pred_box = all_pred_box[keep]
    #     all_poses = all_poses[keep]

    #     return all_pred_box, all_poses



    def merge_by_corr_v2(last_pred_instances, pred_instances, last_kf_rgb, cur_keep_idx, global_pred_box, corr, framerecord, all_poses, frame_id, gap): 
        '''
        查找与当前帧存在overlap的所有帧,
            1.如果当前box与所有kf都不存在cor,则认为是新Instance
            2.1 如果当前box与所有kf的cor 自身overlap低于一个阈值,则认为是新instance  
            2.2 否则即认为这个cor是合理的,可能是同一instance,还需要剔除一个额外情况,即cor box可能存在被包含的情况,metric打分需要考虑2D box IoU, 
                2.2.1 IoU高于阈值则认为是同一instance
                    2.2.1.1 当前box Scores更高,则替换
                    2.2.1.2 当前box scores更低,则删除当前box
                2.2.2 否则认为当前box是新的instance
        '''
        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        lastkf_box = last_pred_instances.pred_boxes.cpu().numpy()
        last_kf_match = corr[:,:2]
        cur_match = corr[:,2:]
        global_num = len(global_pred_box)
        keep = np.arange(len(global_pred_box)+len(pred_instances)) 
        mask = np.ones_like(keep,dtype=bool)
        #record the current frame boxes info
        # print("framerecord",framerecord.record)
        #当前Box的两种去路 1.与旧的关联 2.新增
        '''
        overlap B应该根据所有历史帧计算
        '''
        print("\n")
        for idx in cur_keep_idx:
            cur_box = cur_2d_box[idx]
            cur_mask = Instances3D.points_in_box(cur_match,cur_box) #bool type
            if np.sum(cur_mask)>=1: #当前box有correspondence
                selected_lastkf_pixels = last_kf_match[cur_mask] #[N,2]

                lastkf_box_iou, overlap_A = Instances3D.IoU_2D(selected_lastkf_pixels,lastkf_box) #[num of lastkf box]
                _, overlap_B = Instances3D.IoU_box(cur_match[cur_mask],cur_2d_box)
                print(idx,"overlap_B",overlap_B)
                print(idx,"overlap_A",overlap_A)
                # print(pred_instances.pred_boxes)
                if overlap_B[idx] < 0.1:# and overlap_A[corrponding_boxid]<0.7:
                    print(frame_id,idx,"case 2.1")
                    mask[idx+global_num] = True
                    until_cur_mask = mask[:idx+global_num+1]
                    tmp_count = np.sum(until_cur_mask)
                    framerecord.add_single(frame_id, idx, tmp_count-1)
                else:
                    
                    corrponding_boxid = np.argmax(lastkf_box_iou)
                    print(idx,"lastkf_box_iou",lastkf_box_iou)
                    # print("overlap_A",overlap_A)

                    global_idx = framerecord.record[frame_id-gap][corrponding_boxid]
                    if (lastkf_box_iou[corrponding_boxid] > 0.22) :
 
                        # print("global_idx",global_idx,"cur idx",idx,'corrponding_boxid',corrponding_boxid)
                        # print("lastkf_box_count",lastkf_box_iou)
                        # print("scores",global_pred_box.scores[global_idx],pred_instances.scores[idx])
                        if global_pred_box.scores[global_idx]<pred_instances.scores[idx]: #replace and modify
                            Instances3D.modify_instance(global_idx,idx,global_pred_box,pred_instances)
                            mask[idx+global_num] = False
                            all_poses[global_idx] = all_poses[idx+global_num]
                            print(frame_id,idx,"case 2.2.1.1 ","global idx:",global_idx)
                        else: #abandon worse than the global one
                            mask[idx+global_num] = False
                            print(frame_id,idx,"case 2.2.1.2 ", "global idx:",global_idx)
                        framerecord.add_single(frame_id, idx, global_idx)
                    elif overlap_A[corrponding_boxid]>0.78:
                        mask[idx+global_num] = False
                        print(frame_id,idx,"case 2.2.1.3 ","global idx:",global_idx)
                        framerecord.add_single(frame_id, idx, global_idx,)
                    else: #不是同一个instance，不够数，新增new obeserved instance 
                        print(frame_id,idx,"case 2.2.2 ")
                        mask[idx+global_num] = True
                        until_cur_mask = mask[:idx+global_num+1]
                        tmp_count = np.sum(until_cur_mask)
                        framerecord.add_single(frame_id, idx, tmp_count-1)
            else: #没有correspondence在本帧，新增new obeserved instance
                print(frame_id,idx,"case 1")
                mask[idx+global_num] = True
                until_cur_mask = mask[:idx+global_num+1]
                tmp_count = np.sum(until_cur_mask)
                framerecord.add_single(frame_id, idx, tmp_count-1)

        keep = keep[mask]
        keep=sorted(keep)
        all_pred_box = Instances3D.cat([global_pred_box,pred_instances])
        all_pred_box = all_pred_box[keep]
        all_poses = all_poses[keep]

        return all_pred_box, all_poses

    def count_pixels_in_boxes(A, B):
        """
        统计每个box包含的像素点数量（闭区间）
        
        参数:
            A: numpy数组[N,2]，表示像素点坐标
            B: numpy数组[K,4]，每行为[x_min, y_min, x_max, y_max]
        
        返回:
            C: numpy数组[K]，每个元素表示对应box包含的像素点数量
        """
        x_coords = A[:, 0]  # 提取所有x坐标 [N]
        y_coords = A[:, 1]  # 提取所有y坐标 [N]
        
        # 将B拆分为四个边界数组 [K]
        x_min = B[:, 0]
        y_min = B[:, 1]
        x_max = B[:, 2]
        y_max = B[:, 3]
        
        # 广播比较生成布尔矩阵 [K, N]
        x_in = (x_coords >= x_min[:, None]) & (x_coords <= x_max[:, None])
        y_in = (y_coords >= y_min[:, None]) & (y_coords <= y_max[:, None])
        
        # 组合条件并统计数量 [K]
        return np.sum(x_in & y_in, axis=1) / ((x_max-x_min)*(y_max-y_min))

    def IoU_2D(A, B):
        """
        统计每个box包含的像素点数量（闭区间）
        
        参数:
            A: numpy数组[N,2]，表示像素点坐标
            B: numpy数组[K,4]，每行为[x_min, y_min, x_max, y_max]
        
        返回:
            C: numpy数组[K]，每个元素表示对应IoU
        """
        A = A.astype(np.float64)
        x_min_A, y_min_A = np.min(A, axis=0)
        x_max_A, y_max_A = np.max(A, axis=0)
        area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)

        x_min_B = B[:, 0]
        y_min_B = B[:, 1]
        x_max_B = B[:, 2]
        y_max_B = B[:, 3]
        area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)

        x_min_inter = np.maximum(x_min_A, x_min_B)
        y_min_inter = np.maximum(y_min_A, y_min_B)
        x_max_inter = np.minimum(x_max_A, x_max_B)
        y_max_inter = np.minimum(y_max_A, y_max_B)

        inter_width = np.maximum(0, x_max_inter - x_min_inter)
        inter_height = np.maximum(0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        union_area = area_A + area_B - inter_area
        iou = inter_area / (union_area + 1e-6)
        overlap_A = inter_area / (area_A+1e-6)
        return iou, overlap_A
    
    def IoU_2D_box(A, B):
        """
        统计每个box包含的像素点数量（闭区间）
        
        参数:
            A: numpy数组[4]，表示像素点坐标[x_min, y_min, x_max, y_max]
            B: numpy数组[K,4]，每行为[x_min, y_min, x_max, y_max]
        
        返回:
            C: numpy数组[K]，每个元素表示对应IoU
        """
        A = A.astype(np.float64)
        x_min_A, y_min_A, x_max_A, y_max_A  = A[0],A[1],A[2],A[3]

        area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)

        x_min_B = B[:, 0]
        y_min_B = B[:, 1]
        x_max_B = B[:, 2]
        y_max_B = B[:, 3]
        area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)

        x_min_inter = np.maximum(x_min_A, x_min_B)
        y_min_inter = np.maximum(y_min_A, y_min_B)
        x_max_inter = np.minimum(x_max_A, x_max_B)
        y_max_inter = np.minimum(y_max_A, y_max_B)

        inter_width = np.maximum(0, x_max_inter - x_min_inter)
        inter_height = np.maximum(0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        union_area = area_A + area_B - inter_area
        iou = inter_area / (union_area + 1e-6)
        # overlap_A = inter_area / (area_A+1e-6)
        return iou

    def project_3d_to_2d_box(boxes_3d, K, pose, H, W, frame_id=None):
        """
        boxes_3d: [N, 8, 3]  (世界坐标系下的3D框角点)
        K: [3, 3]             (内参矩阵)
        pose: [4, 4]          (外参矩阵，世界坐标系到相机坐标系)
        H: 480
        W: 640
        返回: [N, 4]          (2D包围框: [x_min, y_min, x_max, y_max])
        """
        N = boxes_3d.shape[0]
        boxes_2d = np.zeros((N, 4))
        
        # 扩展为齐次坐标 [N, 8, 4]
        ones = np.ones((N, 8, 1))
        boxes_homo = np.concatenate([boxes_3d, ones], axis=2)
        
        # 转换到相机坐标系 [N, 8, 4]
        pose_inv = np.linalg.inv(pose)
        boxes_cam = np.dot(boxes_homo, pose_inv.T)
        
        # 提取相机坐标系下的坐标 [N, 8, 3]
        X = boxes_cam[..., 0]
        Y = boxes_cam[..., 1]
        Z = boxes_cam[..., 2]
        
        # 投影到图像平面 [N, 8, 2]
        u = (K[0, 0] * X / Z) + K[0, 2]
        v = (K[1, 1] * Y / Z) + K[1, 2]
        
        # 处理无效点（Z <= 0）以及超出图像边界的点
        valid_mask = (Z > 0) * (u>0) * (u<W) * (v>0) * (v<H)

        for i in range(N):
            # if frame_id is not None:
            #     print(frame_id, i,"valid_u",u[i])
            #     print(frame_id, i,"valid_v",v[i])
            valid_u = u[i][valid_mask[i]]
            valid_v = v[i][valid_mask[i]]
            if len(valid_u) == 0:
                boxes_2d[i] = [0, 0, 0, 0]  # 标记为无效
            else:
                valid_z = (Z > 0) * (Z<8) #newly added
                if len(valid_z) == 0:
                    boxes_2d[i] = [0, 0, 0, 0]  # 标记为无效
                else:
                    valid_u = u[i][valid_z[i]]
                    valid_v = v[i][valid_z[i]]
                    if len(valid_u) == 0 or len(valid_v)==0:
                        boxes_2d[i] = [0, 0, 0, 0]
                    else:
                        valid_u = np.clip(valid_u, 0, W)
                        valid_v = np.clip(valid_v, 0, H)
                        # print(frame_id, i,"valid_u",valid_u)
                        # print(frame_id, i,"valid_v",valid_v)
                        boxes_2d[i] = [np.min(valid_u), np.min(valid_v), 
                                        np.max(valid_u), np.max(valid_v)]
        
        return boxes_2d
    
    def project_3d_to_2d_box_depth(depth_gt, boxes_3d, K, pose, H, W, frame_id=None, depth_bound = 0.5):
        """
        boxes_3d: [N, 8, 3]  (世界坐标系下的3D框角点)
        K: [3, 3]             (内参矩阵)
        pose: [4, 4]          (外参矩阵，世界坐标系到相机坐标系)
        H: 480
        W: 640
        返回: [N, 4]          (2D包围框: [x_min, y_min, x_max, y_max])
        """
        N = boxes_3d.shape[0]
        boxes_2d = np.zeros((N, 4))
        
        # 扩展为齐次坐标 [N, 8, 4]
        ones = np.ones((N, 8, 1))
        boxes_homo = np.concatenate([boxes_3d, ones], axis=2)
        
        # 转换到相机坐标系 [N, 8, 4]
        pose_inv = np.linalg.inv(pose)
        boxes_cam = np.dot(boxes_homo, pose_inv.T)
        
        # 提取相机坐标系下的坐标 [N, 8, 3]
        X = boxes_cam[..., 0] 
        Y = boxes_cam[..., 1]
        Z = boxes_cam[..., 2]
        
        # 投影到图像平面 [N, 8, 2]
        u = (K[0, 0] * X / Z) + K[0, 2]
        v = (K[1, 1] * Y / Z) + K[1, 2]
        u = u.astype(int) #[N,8]
        v = v.astype(int)
        # [H,W]
        # 处理无效点（Z <= 0）以及超出图像边界的点
        valid_mask = (Z > 0) * (u>0) * (u<W) * (v>0) * (v<H) 

        u_mask =u * valid_mask
        v_mask =v * valid_mask
        z_mask = Z * valid_mask

        d_uv = depth_gt[v_mask,u_mask]
        further_mask = (z_mask<d_uv+depth_bound) * (z_mask>d_uv-depth_bound)
        valid_mask = valid_mask * further_mask


        for i in range(N):
            # if frame_id is not None:
            #     print(frame_id, i,"valid_u",u[i])
            #     print(frame_id, i,"valid_v",v[i])
            valid_u = u[i][valid_mask[i]]
            valid_v = v[i][valid_mask[i]]
            if len(valid_u) == 0:
                boxes_2d[i] = [0, 0, 0, 0]  # 标记为无效
            else:
                valid_z = valid_mask #Z > 0
                valid_u = u[i][valid_z[i]]
                valid_v = v[i][valid_z[i]]
                valid_u = np.clip(valid_u, 0, W)
                valid_v = np.clip(valid_v, 0, H)
                # print(frame_id, i,"valid_u",valid_u)
                # print(frame_id, i,"valid_v",valid_v)
                boxes_2d[i] = [np.min(valid_u), np.min(valid_v), 
                                np.max(valid_u), np.max(valid_v)]
        
        return boxes_2d



    def IoU_box(A, B):
        """
        统计每个box包含的像素点数量（闭区间）
        
        参数:
            A: numpy数组[N,2]，表示像素点坐标
            B: numpy数组[K,4]，每行为[x_min, y_min, x_max, y_max]
        
        返回:
            C: numpy数组[K]，每个元素表示对应IoU
        """
        A = A.astype(np.float64)
        x_min_A, y_min_A = np.min(A, axis=0)
        x_max_A, y_max_A = np.max(A, axis=0)
        area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)

        x_min_B = B[:, 0]
        y_min_B = B[:, 1]
        x_max_B = B[:, 2]
        y_max_B = B[:, 3]
        area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)

        x_min_inter = np.maximum(x_min_A, x_min_B)
        y_min_inter = np.maximum(y_min_A, y_min_B)
        x_max_inter = np.minimum(x_max_A, x_max_B)
        y_max_inter = np.minimum(y_max_A, y_max_B)

        inter_width = np.maximum(0, x_max_inter - x_min_inter)
        inter_height = np.maximum(0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        union_area = area_A + area_B - inter_area
        iou = inter_area / (union_area + 1e-6)
        overlap_A = inter_area / (area_B+1e-6)
        return iou, overlap_A

    def points_in_box(points, box):
        """
        判断二维点集是否在矩形框内（闭区间）
        
        参数:
            points: numpy数组[N,2]，表示N个点的(x,y)坐标
            box: 元组或列表[x_min, y_min, x_max, y_max]
        
        返回:
            mask: 布尔数组[N]，True表示对应点在框内
        """
        x_coords = points[:, 0]  # 提取所有x坐标
        y_coords = points[:, 1]  # 提取所有y坐标
        
        # 解包边界坐标
        x_min, y_min, x_max, y_max = box
        
        # 生成布尔掩码（使用向量化运算）
        x_inside = (x_coords >= x_min) & (x_coords <= x_max)
        y_inside = (y_coords >= y_min) & (y_coords <= y_max)
        
        return x_inside & y_inside


    def translate(self, translation):
        # in-place.
        for field_name, field in self._fields.items():
            if hasattr(field, "translate"):
                field.translate(translation)

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__

