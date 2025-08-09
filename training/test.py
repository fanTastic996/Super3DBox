import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import l1_loss

# ---------- Chamfer Loss ----------
def chamfer_loss(corners_pred, corners_gt):
    """
    corners_pred: [N_pos, 8, 3]
    corners_gt  : [N_pos, 8, 3]
    return scalar
    """
    if corners_pred.shape[0] == 0:
        return torch.tensor(0.0, device=corners_pred.device, requires_grad=True)

    dist = torch.cdist(corners_pred, corners_gt)          # [N_pos, 8, 8]
    chamfer = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
    return chamfer

# ---------- Hungarian 匹配 ----------
def hungarian_2d_matching(cost_matrix):
    """
    cost_matrix: [N_pred, N_gt]
    return pred_indices, gt_indices  (长度 = min(N_pred, N_gt))
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu())
    return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)

# ---------- 1) 构造 3D Chamfer 代价矩阵 ----------
def build_3d_cost(pred_corners, gt_corners):
    """
    pred_corners: [N_pred, 8, 3]
    gt_corners  : [N_gt, 8, 3]
    return      : [N_pred, N_gt]  每个元素是 (pred_i, gt_j) 的 Chamfer 距离
    """
    N_pred, N_gt = pred_corners.shape[0], gt_corners.shape[0]
    cost = torch.zeros(N_pred, N_gt, device=pred_corners.device)

    for i in range(N_pred):
        for j in range(N_gt):
            # 两个 8×3 点集之间的 Chamfer
            dist = torch.cdist(pred_corners[i], gt_corners[j])   # [8,8]
            chamfer_ij = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
            cost[i, j] = chamfer_ij
    return cost

# ---------- 总接口 ----------
def compute_box_loss(
        pred_corners,   # [N_pred, 8, 3]
        pred_2d,        # [N_pred, 4]  用于匹配
        gt_corners,     # [N_gt, 8, 3]
        gt_2d,          # [N_gt, 4]
        cost_bbox_weight=5.0):
    """
    1. 用 2D box 做匈牙利匹配
    2. 匹配成功的角点用 Chamfer 损失
    """
    N_pred, N_gt = pred_corners.shape[0], gt_corners.shape[0]
    if N_pred == 0 or N_gt == 0:
        return torch.tensor(0.0, device=pred_corners.device, requires_grad=True)

    # 1) 构造 2D L1 代价矩阵
    # cost_bbox = torch.cdist(pred_2d, gt_2d, p=1)  # [N_pred, N_gt]
    cost_bbox = build_3d_cost(pred_corners, gt_corners)
    
    # 2) 匈牙利匹配
    pred_idx, gt_idx = hungarian_2d_matching(cost_bbox)

    # 3) 取出匹配上的角点
    matched_pred = pred_corners[pred_idx]
    matched_gt   = gt_corners[gt_idx]

    # 4) Chamfer Loss
    loss = chamfer_loss(matched_pred, matched_gt)
    return loss


# 假设模型已经输出角点和 2D box
pred_corners = torch.randn(100, 8, 3, requires_grad=True)  # 100 个预测
pred_2d      = torch.randn(100, 4)

gt_corners   = torch.randn(20, 8, 3)  # 20 个真值
gt_2d        = torch.randn(20, 4)

loss = compute_box_loss(pred_corners, pred_2d,
                              gt_corners,   gt_2d)
print("loss:",loss)
loss.backward()