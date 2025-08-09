import numpy as np
import torch



class FrameRecord:
    def __init__(self):
        self.record = {}
    
    def add(self, frame_id, cur_boxes_id, glo_boxes_id):
        if frame_id not in self.record.keys():
            self.record[frame_id] = {}
        for i in range(cur_boxes_id.shape[0]):
            cur_id = cur_boxes_id[i]
            glo_id = glo_boxes_id[i]
            self.record[frame_id][cur_id] = glo_id
    
    def add_single(self, frame_id, cur_boxes_id, glo_boxes_id):
        if frame_id not in self.record.keys():
            self.record[frame_id] = {}
        self.record[frame_id][cur_boxes_id] = glo_boxes_id

    def neg(self,frame_id, cur_boxes_id):
        if frame_id not in self.record.keys():
            self.record[frame_id] = {}
        for i in range(cur_boxes_id.shape[0]):
            cur_id = cur_boxes_id[i]
            self.record[frame_id][cur_id] = -1
        
