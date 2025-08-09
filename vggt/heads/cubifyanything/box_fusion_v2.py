import torch
import numpy as np

import cv2
from torch.optim import Adam
from kornia.geometry.conversions import rotation_matrix_to_angle_axis as matrix_to_axis_angle
from kornia.geometry.conversions import angle_axis_to_rotation_matrix as at_to_transform_matrix
import os
from sklearn.cluster import DBSCAN
import copy
try:
  import pycuda.driver as cuda
  import pycuda.autoprimaryctx
  from pycuda.compiler import SourceModule
  import pycuda.gpuarray as gpuarray
  GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  GPU_MODE = 0

class Holder(cuda.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()
    
class BoxFusion(object):
    def __init__(self, cfg) -> None:
        super(BoxFusion, self).__init__()
        self.cfg = cfg
        self.PST_path = cfg["box_fusion"]["pst_path"]
        self.PST = np.ascontiguousarray(cv2.imread(self.PST_path, -1)) #[3072,6]
        
        self.basedir = cfg['data']['datadir']
        # print('self.basedir',self.basedir)
        # exit(0)
        if 'ScanNet' in self.basedir or 'BS3D' in self.basedir or 'large' in self.basedir or cfg["dataset"] == 'online':
            self.K = np.array([[cfg['cam']['fx'], 0.0, cfg['cam']['cx'],0.0],
                            [0.0, cfg['cam']['fy'], cfg['cam']['cy'],0.0],
                            [0.0,0.0,1.0,0.0],
                            [0.0,0.0,0.0,1.0]])
            self.H=cfg["cam"]["H"] #l
            self.W=cfg["cam"]["W"] #s

        else:
            # depth_intric = np.loadtxt(os.path.join(self.basedir, 'K_depth.txt')).reshape(3,3)
            depth_intric = np.loadtxt(os.path.join(self.basedir, 'K_rgb.txt')).reshape(3,3)
            self.K = np.array([[depth_intric[0,0], 0.0, depth_intric[0,2],0.0],
                            [0.0, depth_intric[1,1], depth_intric[1,2],0.0],
                            [0.0,0.0,1.0,0.0],
                            [0.0,0.0,0.0,1.0]])
            if self.K[0,2] < self.K[1,2]:
                self.H=cfg["cam"]["W"] #l
                self.W=cfg["cam"]["H"] #s
            else:
                self.H=cfg["cam"]["H"]
                self.W=cfg["cam"]["W"]
        self.update_K_flag=False

        self.fusion_iters = cfg["box_fusion"]["iters"]
        self.pst_size = cfg["box_fusion"]["pst_size"]
        self.center_init_size = cfg["box_fusion"]["random_opt"]["center_init_size"]
        self.center_scaling_coefficient = cfg["box_fusion"]["random_opt"]["center_scaling_coefficient"]
        self.shape_init_size = cfg["box_fusion"]["random_opt"]["shape_init_size"]
        self.shape_scaling_coefficient = cfg["box_fusion"]["random_opt"]["shape_scaling_coefficient"]



        self.cuda_src_mod = SourceModule("""
            #include <curand_kernel.h>
            #include <algorithm>
            extern "C" {       

            struct Point {
                float x, y;
                __device__ Point(float x=0, float y=0) : x(x), y(y) {}
            };
                                
            // 向量叉积计算                    
            __device__ float cross(const Point& o, const Point& a, const Point& b) {
                return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
            }
                                
            __device__ float array_max(float* data, int n) {
                float max_val = data[0];
                for (int i = 1; i < n; i++) {
                    max_val = max(max_val, data[i]);
                }
                return max_val;
            }
                                
            __device__ float array_min(float* data, int n) {
                float min_val = data[0];
                for (int i = 1; i < n; i++) {
                    min_val = min(min_val, data[i]);
                }
                return min_val;
            }
                                
            // Andrew凸包算法
            __device__ void convex_hull(Point* in_points, int in_size, 
                            Point* out_points, int& out_size) {
                if (in_size == 0) {
                    out_size = 0;
                    return;
                }
                
                // 排序（模拟网页3的std::sort）
                for(int i=0; i<in_size-1; ++i){
                    for(int j=i+1; j<in_size; ++j){
                        if(in_points[i].x > in_points[j].x || 
                        (in_points[i].x == in_points[j].x && in_points[i].y > in_points[j].y)){
                            Point tmp = in_points[i];
                            in_points[i] = in_points[j];
                            in_points[j] = tmp;
                        }
                    }
                }

                // 下凸包（参考网页5的数组操作）
                Point* lower = new Point[in_size];
                int lower_size = 0;
                for(int i=0; i<in_size; ++i){
                    while(lower_size >= 2 && 
                        cross(lower[lower_size-2], lower[lower_size-1], in_points[i]) <= 0){
                        lower_size--;
                    }
                    lower[lower_size++] = in_points[i];
                }

                // 上凸包
                Point* upper = new Point[in_size];
                int upper_size = 0;
                for(int i=in_size-1; i>=0; --i){
                    while(upper_size >= 2 && 
                        cross(upper[upper_size-2], upper[upper_size-1], in_points[i]) <= 0){
                        upper_size--;
                    }
                    upper[upper_size++] = in_points[i];
                }

                // 合并结果（参考网页5的数组拼接）
                lower_size--;
                upper_size--;
                out_size = lower_size + upper_size;
                for(int i=0; i<lower_size; ++i) out_points[i] = lower[i];
                for(int i=0; i<upper_size; ++i) out_points[lower_size+i] = upper[i];
                
                delete[] lower;
                delete[] upper;
            }
                                
            // 鞋带公式计算面积
            __device__ float polygon_area(Point* poly, int n) {
                float area = 0.0;
                for(int i=0; i<n; ++i){
                    Point& p1 = poly[i];
                    Point& p2 = poly[(i+1)%n];
                    area += p1.x * p2.y - p2.x * p1.y;
                }
                return fabs(area) / 2.0;
            }
                                
            // 线段交点计算
            __device__ Point* line_intersection(const Point& a1, const Point& a2, 
                                const Point& b1, const Point& b2) {
                double dx1 = a2.x - a1.x;
                double dy1 = a2.y - a1.y;
                double dx2 = b2.x - b1.x;
                double dy2 = b2.y - b1.y;
                
                double denominator = dx1 * dy2 - dy1 * dx2;
                if (std::abs(denominator) < 1e-8) return nullptr;
                
                double t = (dx2*(a1.y - b1.y) + dy2*(b1.x - a1.x)) / denominator;
                double s = (dx1*(a1.y - b1.y) + dy1*(b1.x - a1.x)) / denominator;
                
                if (t >= -1e-8 && t <= 1.00000001 && 
                    s >= -1e-8 && s <= 1.00000001) {
                    return new Point(a1.x + t*dx1, a1.y + t*dy1);
                }
                return nullptr;
            }
                                
            // 射线法点包含检测
            __device__ bool point_in_polygon(const Point& p, Point* poly, int poly_size) {
                const float x = p.x;
                const float y = p.y;
                bool inside = false;

                for(int i = 0; i < poly_size; ++i) {
                    const Point& p1 = poly[i];
                    const Point& p2 = poly[(i+1) % poly_size];

                    // 完全保持Python的条件判断逻辑
                    if( (p1.y > y) != (p2.y > y) ) {
                        // 严格使用原始Python公式
                        const float x_inters = ( (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) ) + p1.x;
                        if(x < x_inters) {
                            inside = !inside;
                        }
                    }
                }
                return inside;
            }
                                
            // 多边形交集计算
            __device__ void polygon_intersection(
                Point* poly1, int poly1_size,
                Point* poly2, int poly2_size,
                Point* candidates, int& cand_size) 
            {
                cand_size = 0;
                
                // 收集内部顶点（参考网页7的数组遍历）
                for(int i=0; i<poly1_size; ++i){
                    if(point_in_polygon(poly1[i], poly2, poly2_size)){
                        candidates[(cand_size)++] = poly1[i];
                    }
                }
                for(int i=0; i<poly2_size; ++i){
                    if(point_in_polygon(poly2[i], poly1, poly1_size)){
                        candidates[(cand_size)++] = poly2[i];
                    }
                }
                
                // 边交点计算（参考网页6的循环嵌套）
                for(int i=0; i<poly1_size; ++i){
                    Point a1 = poly1[i];
                    Point a2 = poly1[(i+1)%poly1_size];
                    
                    for(int j=0; j<poly2_size; ++j){
                        Point b1 = poly2[j];
                        Point b2 = poly2[(j+1)%poly2_size];
                        
                        Point* pt = line_intersection(a1, a2, b1, b2);
                        if(pt){
                            candidates[(cand_size)++] = *pt;
                            delete pt;
                        }
                    }
                }
                
                // 极角排序（参考网页3的排序逻辑）
                if(cand_size > 0){
                    float cx=0, cy=0;
                    for(int i=0; i<cand_size; ++i){
                        cx += candidates[i].x;
                        cy += candidates[i].y;
                    }
                    cx /= cand_size;
                    cy /= cand_size;
                    
                    // 冒泡排序替代std::sort
                    for(int i=0; i<cand_size-1; ++i){
                        for(int j=0; j<cand_size-i-1; ++j){
                            float angle1 = atan2(candidates[j].y-cy, candidates[j].x-cx);
                            float angle2 = atan2(candidates[j+1].y-cy, candidates[j+1].x-cx);
                            if(angle1 > angle2){
                                Point tmp = candidates[j];
                                candidates[j] = candidates[j+1];
                                candidates[j+1] = tmp;
                            }
                        }
                    }
                }
            }


            __global__ void compute_iou_value(float * box_3d,
                                    float * t_c,
                                    float * scores,    
                                    float * transform_candidate,
                                    float * box_rot,
                                    float * cam_poses,
                                    float * K,
                                    float * search_size,
                                    float * search_value,
                                    float * search_count,
                                    float * other_params
                                ){
           
            int node=blockDim.x*blockIdx.x+threadIdx.x;
            
            
            float img_h = other_params[0];
            float img_w = other_params[1];
            float node_size = other_params[2];
            int num_boxes = (int) other_params[3];
            
            if (node>=node_size){
                return;
            }

            float x3d = box_3d[0];
            float y3d = box_3d[1];
            float z3d = box_3d[2];
            float w3d = box_3d[5];
            float h3d = box_3d[4];
            float l3d = box_3d[3];
                            
            x3d = x3d + transform_candidate[node*6+0] * search_size[0];
            y3d = y3d + transform_candidate[node*6+1] * search_size[1];
            z3d = z3d + transform_candidate[node*6+2] * search_size[2];
            w3d = w3d + transform_candidate[node*6+5] * search_size[5];
            h3d = h3d + transform_candidate[node*6+4] * search_size[4];
            l3d = l3d + transform_candidate[node*6+3] * search_size[3];
            
            float xyz[3] = {x3d,y3d,z3d};

            w3d = max(w3d, 0.01f); // 防止w3d为0
            h3d = max(h3d, 0.01f); // 防止h3d为0
            l3d = max(l3d, 0.01f); // 防止l3d为0                                     


            float verts[8][3] = {
            {-l3d / 2, -h3d / 2, -w3d / 2},
            {l3d / 2, -h3d / 2, -w3d / 2},                
            {l3d / 2, h3d / 2, -w3d / 2},
            {-l3d / 2, h3d / 2, -w3d / 2},
            {-l3d / 2, -h3d / 2, w3d / 2},
            {l3d / 2, -h3d / 2, w3d / 2},
            {l3d / 2, h3d / 2, w3d / 2},
            {-l3d / 2, h3d / 2, w3d / 2},
            };
                            
            
            float corners[8][3] = {0}; // 结果矩阵

            for (int i =0; i<8; ++i){          // 遍历3x3矩阵的行
                for (int j=0; j<3; ++j){       // 遍历8x3数组的列（转置后的行）
                    for (int k=0; k<3; ++k){  // 累加计算点积
                        corners[i][j] += box_rot[j*3+k] * verts[i][k];        
                    } 
                    corners[i][j] += xyz[j];
                }  
            } 

            
            
            //project pts in world cordinate into 2D planes and get [u,v] -> [N,8,2]
                            
            int i=(blockDim.y*blockIdx.y+threadIdx.y);
                            
            if (i>=num_boxes){
                return;
            }
                                         
            float score_box = scores[i];
                           
            float uv[8][2] = {0};
                        
            for (int j=0; j<8; ++j){ 
                float vertex_x = corners[j][0]-cam_poses[i*16+3];
                float vertex_y = corners[j][1]-cam_poses[i*16+7];
                float vertex_z = corners[j][2]-cam_poses[i*16+11];
                
                float cam_x = cam_poses[i*16+0]*vertex_x+cam_poses[i*16+4]*vertex_y+cam_poses[i*16+8]*vertex_z ;
                float cam_y = cam_poses[i*16+1]*vertex_x+cam_poses[i*16+5]*vertex_y+cam_poses[i*16+9]*vertex_z ;
                float cam_z = cam_poses[i*16+2]*vertex_x+cam_poses[i*16+6]*vertex_y+cam_poses[i*16+10]*vertex_z ;

                float pixel_x = ((cam_x*K[0])/cam_z+K[2]);
                float pixel_y = ((cam_y*K[5])/cam_z+K[6]);
                
                uv[j][0] = (pixel_x > img_w) ? img_w : (pixel_x < 0) ? 0 : pixel_x;
                uv[j][1] = (pixel_y > img_h) ? img_h : (pixel_y < 0) ? 0 : pixel_y;
            }
                            

            Point corners0[8] = {Point(uv[0][0], uv[0][1]),Point(uv[1][0], uv[1][1]),Point(uv[2][0], uv[2][1]),Point(uv[3][0], uv[3][1]),Point(uv[4][0], uv[4][1]),Point(uv[5][0], uv[5][1]),Point(uv[6][0], uv[6][1]),Point(uv[7][0], uv[7][1])};  
            

            Point t_corners0[8] = {Point(t_c[i*16+0], t_c[i*16+1]),Point(t_c[i*16+2], t_c[i*16+3]),Point(t_c[i*16+4], t_c[i*16+5]),Point(t_c[i*16+6], t_c[i*16+7]),Point(t_c[i*16+8], t_c[i*16+9]),Point(t_c[i*16+10], t_c[i*16+11]),Point(t_c[i*16+12], t_c[i*16+13]),Point(t_c[i*16+14], t_c[i*16+15])};

            Point convex_0[8]; 
            int out_size_0 = 8;
            Point convex_t[8]; 
            int out_size_t = 8;

            convex_hull(corners0, 8, convex_0, out_size_0);
            convex_hull(t_corners0, 8, convex_t, out_size_t);


            Point corners_i[36]; 
            int corners_i_size = 8;
            polygon_intersection(convex_0,out_size_0,convex_t,out_size_t,corners_i,corners_i_size);
            
            Point convex_inter[8]; 
            int out_size_inter = 8;
            convex_hull(corners_i, corners_i_size, convex_inter, out_size_inter);
                            
                            
            float inter_area = polygon_area(convex_inter, out_size_inter);
            float area0 = polygon_area(convex_0, out_size_0);
            float area_t = polygon_area(convex_t, out_size_t);
            
                
            float union_area = area0 + area_t - inter_area;
            float iou = 0;         
            if (union_area>0){
                iou =  inter_area / (union_area+0.00001);  
                
                iou = iou; //* max(score_box,1.0);
            }
            
            atomicAdd_system(search_value+node,abs(1-iou));
            atomicAdd_system(search_count+node,1);
                        
            return;

        }
}
         """, no_extern_c=True)

        self.cuda_compute_iou_value = self.cuda_src_mod.get_function("compute_iou_value") 



    def evaluate_iou(self, 
                    box_3d,
                    corners_2d,
                    box_rot,
                    scores_box,
                    camera_poses,
                    search_size,
                    num_of_boxes,
                    verbose=False):
        '''
        box_2d: [N, xyzlwh]
        transform_candidate: [N_particle]
        '''


        search_value=np.zeros((self.PST.shape[0])).astype(np.float32)
        search_count=np.zeros((self.PST.shape[0])).astype(np.float32)
        # if verbose:
        #     print("box_3d",box_3d)
        #     print("corners_2d",corners_2d)
        #     print("self.PST",self.PST)
        #     print("box_rot",box_rot)
        #     print("camera_poses",camera_poses)
        #     print("search_size",search_size)

        self.cuda_compute_iou_value(
                        cuda.In(box_3d.reshape(-1).astype(np.float32)),
                        cuda.In(corners_2d.reshape(-1).astype(np.float32)),
                        cuda.In(scores_box.reshape(-1).astype(np.float32)),
                        cuda.In(self.PST.reshape(-1).astype(np.float32)),
                        cuda.In(box_rot.reshape(-1).astype(np.float32)),
                        cuda.In(camera_poses.reshape(-1).astype(np.float32)),
                        cuda.In(self.K.reshape(-1).astype(np.float32)),
                        cuda.In(search_size),
                        cuda.InOut(search_value),
                        cuda.InOut(search_count),
                        cuda.In(np.asarray([
                                        self.H,
                                        self.W,
                                        self.pst_size,
                                        num_of_boxes
                                        ], np.float32)),
                        # block=(32*32,1,1),  
                        # grid=( int(node_size/(32*32)),1,1)  # 3,1      
                        block=(32,1,1),  
                        grid=( int(self.pst_size/(32)),num_of_boxes,1)  # 3,1      
                        )
        
        fitness = search_value/(search_count+1e-6)

        if verbose:
            # print("box_3d",box_3d)
            print("search value",search_value, search_value.shape, np.sum(search_value), 'last best iou:',1-fitness[0])


        return fitness

    def update_intrinsics(self,size,K):
        self.H=size[1]
        self.W=size[0]
        self.K[:3,:3] = K

    def init_searchsize(self):
        self.search_size=np.zeros((6),dtype=np.float32)
        self.previous_search_size =np.zeros((6),dtype=np.float32)
        self.search_size[:3] = self.center_init_size
        self.search_size[3:] = self.shape_init_size
        # self.search_size[...] = 0.01 #self.scaling_coefficient

    def cal_transform(self,search_value,search_size):
        #calculate the mean_transform result:
        mean_transform = np.zeros((6),dtype=np.float32) 
        origin_iou = search_value[0]
        #init sum value
        sum_tx = 0.0
        sum_ty = 0.0
        sum_tz = 0.0
        sum_l = 0.0
        sum_w = 0.0
        sum_h = 0.0
        sum_weight = 0.0
        sum_iou = 0.0
        count_search = 0

        for j in range(1,len(search_value)):
        # for j in arr:
            if search_value[j]<origin_iou:
                tx = self.PST[j][0]
                ty = self.PST[j][1]
                tz = self.PST[j][2]
                qx = self.PST[j][3]
                qy = self.PST[j][4]
                qz = self.PST[j][5]
                cur_fit = search_value[j]
                weight = origin_iou - cur_fit

                sum_tx +=tx*weight
                sum_ty +=ty*weight
                sum_tz +=tz*weight
                sum_l +=qx*weight
                sum_w +=qy*weight
                sum_h +=qz*weight
                


                sum_weight +=weight
                sum_iou +=cur_fit*weight
                count_search +=1

                
                if count_search== 200: #test过了，没影响 #ori:200
                    break 
                
        #如果所有粒子都一直比0号粒子差，那么就跳过这一轮，如果都差，那就整个这一帧跳过，保持上一帧的best_pose
        if count_search <= 0:
            success = False
            min_iou = origin_iou #* DIVSHORTMAX
            return False,min_iou,mean_transform
        #计算各种均值
        mean_iou = sum_iou / sum_weight
        mean_transform[0] = (sum_tx / sum_weight)*search_size[0]
        mean_transform[1] = (sum_ty / sum_weight)*search_size[1]
        mean_transform[2] = (sum_tz / sum_weight)*search_size[2]
    

        mean_transform[3] = (sum_l / sum_weight)*search_size[3]
        mean_transform[4] = (sum_w / sum_weight)*search_size[4]
        mean_transform[5] = (sum_h / sum_weight)*search_size[5]

        min_tsdf = mean_iou #* DIVSHORTMAX

        return True,min_tsdf,mean_transform

    def update_PST(self, iou,mean_transform,min_scale=1e-3,center_scale=0.5, shape_scale=0.5): #min_scale=1e-3
        
        s_tx =abs(mean_transform[0])+min_scale
        s_ty =abs(mean_transform[1])+min_scale
        s_tz =abs(mean_transform[2])+min_scale
        
        s_qx =abs(mean_transform[3])+min_scale
        s_qy =abs(mean_transform[4])+min_scale
        s_qz =abs(mean_transform[5])+min_scale
        
        trans_norm = np.sqrt(s_tx*s_tx+s_ty*s_ty+s_tz*s_tz+s_qx*s_qx+s_qy*s_qy+s_qz*s_qz)
        
        normal_tx=s_tx/trans_norm
        normal_ty=s_ty/trans_norm
        normal_tz=s_tz/trans_norm 
        normal_qx=s_qx/trans_norm
        normal_qy=s_qy/trans_norm
        normal_qz=s_qz/trans_norm
        #0.09   + 1e-3

        self.search_size[3] = shape_scale * iou * normal_qx+min_scale
        self.search_size[4] = shape_scale * iou * normal_qy+min_scale
        self.search_size[5] = shape_scale * iou * normal_qz+min_scale
        self.search_size[0] = center_scale * iou * normal_tx+min_scale
        self.search_size[1] = center_scale * iou * normal_ty+min_scale
        self.search_size[2] = center_scale * iou * normal_tz+min_scale
        # print('self.search_size',self.search_size)

    def remove_outlier(self, box_3d, per_boxes_3d_scores, eps=0.3, min_samples=3):
        """
        简化版6D聚类异常值检测
        参数：
            box_3d: numpy数组，形状[N,6]，每行[x,y,z,l,w,h]
            per_boxes_3d_scores： scores数组，形状[N,]，每个box的置信度分数
            eps: 邻域半径（默认0.2）
            min_samples: 核心点最小邻居数（默认3）
        返回：
            mask: 布尔数组，True表示有效box（非异常值）
        """
        # 标准化6D特征
        # scaled = (boxes - np.mean(boxes, axis=0)) / np.std(boxes, axis=0)

        best_box = np.argmax(per_boxes_3d_scores)
        best_box_size = box_3d[best_box, 3:]
        sorted_indices = np.argsort(best_box_size)  # 输出[1,0,2]，对应[小,中,大]
        index_0 = np.where(sorted_indices == 0)[0][0]
        index_1 = np.where(sorted_indices == 1)[0][0]
        index_2 = np.where(sorted_indices == 2)[0][0]
        get_indices = [index_0,index_1,index_2]
        # 步骤2：按索引重组B的列顺序
        B_sorted = np.sort(box_3d[:,3:], axis=1) #[N,3] s->l
        B_sorted = B_sorted[:, get_indices]

        transformed_boxes = np.zeros_like(box_3d)
        transformed_boxes[:,:3] = box_3d[:,:3]  # 保留位置
        transformed_boxes[:,3:] = B_sorted  # 替换尺寸

        # 密度聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(transformed_boxes)
        
        valid_mask = clustering.labels_ != -1
        transformed_boxes = transformed_boxes[valid_mask]  # 仅保留有效数据
        if np.sum(valid_mask) != box_3d.shape[0]:
            print("**********deleting***********", valid_mask)
        return valid_mask, transformed_boxes

    def init_opt_params(self,box_3d,per_boxes_3d_R,per_boxes_3d_scores,verbose=False):
        '''
        box_3d: [N,6]
        per_boxes_3d_R: [N,3,3] 
        per_boxes_3d_scores: [N] 
        '''
        best_box = np.argmax(per_boxes_3d_scores) 

        mean_xyzlwh = np.zeros(6)
        box_center = box_3d[:,:3]
        mean_xyz = np.mean(box_center, axis=0) #[3]
        mean_xyzlwh[:3] = mean_xyz
        
        best_box_size = box_3d[best_box, 3:]
        sorted_indices = np.argsort(best_box_size)  # 输出[1,0,2]，对应[小,中,大]
        index_0 = np.where(sorted_indices == 0)[0][0]
        index_1 = np.where(sorted_indices == 1)[0][0]
        index_2 = np.where(sorted_indices == 2)[0][0]
        get_indices = [index_0,index_1,index_2]
        # 步骤2：按索引重组B的列顺序
        B_sorted = np.sort(box_3d[:,3:], axis=1) #[N,3] s->l
        B_sorted = B_sorted[:, get_indices]
        if verbose:
            print('best_box_size',best_box_size)
            print("per_boxes_3d_scores",per_boxes_3d_scores)
            print("best_box",best_box)
            print("sorted_indices",sorted_indices)
            print('box_3d',box_3d)
            print('B_sorted',B_sorted)
        mean_xyzlwh[3:6] = np.mean(B_sorted,axis=0) #[3]
       

        mean_rot = per_boxes_3d_R[best_box] #[3,3]

        return mean_xyzlwh, mean_rot
    
    def init_opt_params_v2(self,box_3d,per_boxes_3d_R,per_boxes_3d_scores,verbose=False):
        '''
        box_3d: [N,6]
        per_boxes_3d_R: [N,3,3] 
        per_boxes_3d_scores: [N] 
        '''
        best_box = np.argmax(per_boxes_3d_scores) 

        mean_xyzlwh = np.zeros(6)
        box_center = box_3d[:,:3]
        mean_xyz = np.mean(box_center, axis=0) #[3]
        mean_xyzlwh[:3] = mean_xyz
        
        mean_xyzlwh[3:6] = np.mean(box_3d[:,3:],axis=0) #[3]
       
        mean_rot = per_boxes_3d_R[best_box] #[3,3]

        return mean_xyzlwh, mean_rot

    
    def boxfusion(self, all_pred_box, per_frame_box, box_manager, beta=0.9):
        N_box = len(all_pred_box)
        per_cam_pose = per_frame_box.cam_pose.cpu().numpy()
        per_boxes_3d = per_frame_box.pred_boxes_3d.tensor.cpu().numpy()
        per_boxes_3d_R = per_frame_box.get("pred_boxes_3d").R.cpu().numpy()
        per_boxes_3d_scores = per_frame_box.scores.cpu().numpy()

        per_boxes_2d = per_frame_box.pred_boxes.cpu().numpy()
        per_boxes_2d_cor = per_frame_box.projected_boxes.cpu().numpy()
        # print("self.already_fusion",box_manager.already_fusion)
        verbose=False
        for i in range(N_box):

            #TODO:增加视差判断，在添加fusion list时判断两帧之间的视差是否足够大，否则不添加
            if len(box_manager.fusion_list[i])<3 or box_manager.check_if_fusion(box_manager.fusion_list[i]): 
                continue

            # if i>=1:
            #     continue

            '''
            prepare the data used for fusion
            '''
            fusion_idx = box_manager.fusion_list[i]
            num_of_boxes = len(fusion_idx)
            print(f"fusing {i} box, fusion list is ",fusion_idx, 'len:', num_of_boxes)



            cam_poses = per_cam_pose[fusion_idx] #[N,4,4]
            # print("cam_poses",cam_poses)
            box_3d = per_boxes_3d[fusion_idx] #[N,6] 

            corners_2d = per_boxes_2d_cor[fusion_idx] 

            # if i==28:
            #     print("box_3d ",box_3d) 
            #     print("box_3d R ",per_boxes_3d_R[fusion_idx]) 
            #     print("corners_2d",corners_2d)
            #     print("cam_poses",cam_poses)
            # mean_xyz = np.mean(box_3d[:,:3], axis=0) #[3]
            # mean_xyzlwh = np.zeros(6)
            # mean_xyzlwh[:3] = mean_xyz
            # mean_xyzlwh[3:6] = np.mean(box_3d[:,3:],axis=0) 

            # if i == 17:
            #     print("box_3d:",box_3d)
            #     print("R:",per_frame_box.get("pred_boxes_3d").R[fusion_idx])
            #     print('corners:',per_frame_box.get("pred_boxes_3d").corners[fusion_idx])



            # rot = per_boxes_3d_R[fusion_idx] #[N,3,3]
            # rot = torch.from_numpy(rot)
            # r_axis = matrix_to_axis_angle(rot)
            # r_axis = torch.mean(r_axis,dim=0)
            # mean_rot = at_to_transform_matrix(r_axis.unsqueeze(0)).squeeze().cpu().numpy()
            # if i == 22:
            #     mean_xyzlwh, mean_rot = self.init_opt_params(box_3d, per_boxes_3d_R[fusion_idx], per_boxes_3d_scores[fusion_idx],verbose=True)
            # else:



            mean_xyzlwh, mean_rot = self.init_opt_params(box_3d, per_boxes_3d_R[fusion_idx], per_boxes_3d_scores[fusion_idx],verbose=False)

            scores_box = per_boxes_3d_scores[fusion_idx] #* 2
            # sum_of_scores = np.sum(scores_box)
            # scores_box /= sum_of_scores

            # print("mean rot", mean_rot)

            global_xyzlwh = mean_xyzlwh #initialize the parameters to be optimized
            before_opt = copy.deepcopy(global_xyzlwh)
            # if i==28:
            #     print('global_xyzlwh',global_xyzlwh)
            self.init_searchsize()

            need_update = False
            previous_success = False
            fail_count = 0

            # if i == 17:
            verbose=False

            # if i==22:
            #     verbose=True
            #     self.fusion_iters = 1
            
#             if i==28:
#                global_xyzlwh = np.array([-1.31018125e+00,  1.53088125e+00,  1.65228788e+02,  1.40244993e-01,
#   1.63229273e-01,  1.86234708e-01])

            for n in range(self.fusion_iters):
                # print("optimizing global_xyzlwh",global_xyzlwh)
                search_value = self.evaluate_iou(global_xyzlwh, 
                                                corners_2d,
                                                mean_rot, 
                                                scores_box,
                                                cam_poses,
                                                self.search_size,
                                                num_of_boxes,
                                                verbose=verbose)

                success,min_iou,mean_transform = self.cal_transform(search_value, 
                self.search_size)
                
                #update PST
                self.update_PST(min_iou,
                                mean_transform,
                                center_scale = self.center_scaling_coefficient,
                                shape_scale = self.shape_scaling_coefficient)
                                #scale=0.5) 
                
                if previous_success and success:
                    self.search_size[0] = beta*self.search_size[0]+(1-beta)*self.previous_search_size[0]
                    self.search_size[1] = beta*self.search_size[1]+(1-beta)*self.previous_search_size[1]
                    self.search_size[2] = beta*self.search_size[2]+(1-beta)*self.previous_search_size[2]
                    self.search_size[3] = beta*self.search_size[3]+(1-beta)*self.previous_search_size[3]
                    self.search_size[4] = beta*self.search_size[4]+(1-beta)*self.previous_search_size[4]
                    self.search_size[5] = beta*self.search_size[5]+(1-beta)*self.previous_search_size[5]

                #update global xyzlwh
                if success:
                    need_update = True
                    previous_success = True 
                    fail_count = 0
                    # print("each transform",mean_transform)
                    global_xyzlwh += mean_transform 
                    # print("global_xyzlwh",global_xyzlwh)
                    self.previous_search_size[0] = self.search_size[0]
                    self.previous_search_size[1] = self.search_size[1]
                    self.previous_search_size[2] = self.search_size[2]
                    self.previous_search_size[3] = self.search_size[3]
                    self.previous_search_size[4] = self.search_size[4]
                    self.previous_search_size[5] = self.search_size[5]

                else:
                    fail_count+=1
                    previous_success=False

                #shut down optimization if covergence
                if fail_count>=3:
                    break
                
            if need_update:
                #update tensor xyzlwh
                # print("final global_xyzlwh",global_xyzlwh)
                # print("final mean_transform",mean_transform)
                # print("final difference",(before_opt-global_xyzlwh))
                global_lwh = global_xyzlwh[3:]
                global_lwh[global_lwh<0.01]=0.01
                global_xyzlwh[3:] = global_lwh
                all_pred_box.pred_boxes_3d.tensor[i]= torch.from_numpy(global_xyzlwh).to(all_pred_box.pred_boxes_3d.tensor[i].device)#[3:]
                # print("debug",all_pred_box.pred_boxes_3d.tensor[i])
                #update fusion flag
                box_manager.update_fusion_flag(i)
                box_manager.add_fusion_ind(fusion_idx)
