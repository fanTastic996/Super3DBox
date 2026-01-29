#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from sklearn.neighbors import KDTree

import open3d as o3d
import pickle
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from itertools import combinations


import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from itertools import combinations
import torch

def _as_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _is_torch(x):
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _group_box_planes_from_hull(corners, round_decimals=6):
    hull = ConvexHull(corners)
    eq = hull.equations  # (F,4): n·x + d <= 0

    normals = eq[:, :3]
    ds = eq[:, 3]

    nrm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals_u = normals / nrm
    ds_u = ds / nrm[:, 0]

    keys = np.concatenate([normals_u, ds_u[:, None]], axis=1)
    keys_r = np.round(keys, round_decimals)

    uniq = {}
    for row, rowr in zip(keys, keys_r):
        k = tuple(rowr.tolist())
        if k not in uniq:
            uniq[k] = row

    planes = np.stack(list(uniq.values()), axis=0)
    if planes.shape[0] > 6:
        pr = np.round(planes, 4)
        uniq2 = {}
        for row, rowr in zip(planes, pr):
            k = tuple(rowr.tolist())
            if k not in uniq2:
                uniq2[k] = row
        planes = np.stack(list(uniq2.values()), axis=0)

    return planes


def _poly_intersection_volume(planes, eps=1e-9):
    P = planes.shape[0]
    if P < 4:
        return 0.0

    A = planes[:, :3]
    b = -planes[:, 3]  # n·x <= b

    verts = []
    for i, j, k in combinations(range(P), 3):
        M = np.stack([A[i], A[j], A[k]], axis=0)
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        rhs = np.array([b[i], b[j], b[k]], dtype=np.float64)
        x = np.linalg.solve(M, rhs)
        if np.all(A @ x - b <= eps):
            verts.append(x)

    if len(verts) < 4:
        return 0.0

    verts = np.unique(np.round(np.stack(verts, axis=0), 10), axis=0)
    if verts.shape[0] < 4:
        return 0.0

    try:
        hull = ConvexHull(verts)
        return float(hull.volume)
    except QhullError:
        return 0.0


def obb_iou_corners(c1, c2, pre1=None, pre2=None, eps=1e-9):
    c1 = np.asarray(c1, dtype=np.float64).reshape(8, 3)
    c2 = np.asarray(c2, dtype=np.float64).reshape(8, 3)

    if pre1 is None:
        try:
            v1 = float(ConvexHull(c1).volume)
            p1 = _group_box_planes_from_hull(c1)
        except QhullError:
            return 0.0
    else:
        v1 = float(pre1["vol"])
        p1 = pre1["planes"]

    if pre2 is None:
        try:
            v2 = float(ConvexHull(c2).volume)
            p2 = _group_box_planes_from_hull(c2)
        except QhullError:
            return 0.0
    else:
        v2 = float(pre2["vol"])
        p2 = pre2["planes"]

    if v1 <= 0.0 or v2 <= 0.0 or p1 is None or p2 is None:
        return 0.0

    planes = np.concatenate([p1, p2], axis=0)
    inter = _poly_intersection_volume(planes, eps=eps)
    if inter <= 0.0:
        return 0.0

    union = v1 + v2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)

def _sizes_similar(s1, s2, size_ratio_thresh=1.25, eps=1e-12, order_invariant=False):
    s1 = np.asarray(s1, dtype=np.float64).reshape(3)
    s2 = np.asarray(s2, dtype=np.float64).reshape(3)
    if order_invariant:
        s1 = np.sort(s1)
        s2 = np.sort(s2)
    r = np.maximum(s1, s2) / (np.minimum(s1, s2) + eps)
    return bool(np.all(r <= size_ratio_thresh))

def _sizes_similar_test(s1, s2, size_ratio_thresh=1.25, eps=1e-12, order_invariant=False):
    s1 = np.asarray(s1, dtype=np.float64).reshape(3)
    s2 = np.asarray(s2, dtype=np.float64).reshape(3)
    if order_invariant:
        s1 = np.sort(s1)
        s2 = np.sort(s2)
    r = np.maximum(s1, s2) / (np.minimum(s1, s2) + eps)
    print('ratio:', r)
    return bool(np.all(r <= size_ratio_thresh))

def obb_nms_corners_sizeaware_v2(
    box_corners,
    scores,
    box_sizes,
    iou_thresh=0.10,          # 低阈值：超过它才可能触发“size-aware” suppress
    hard_iou_thresh=0.50,     # 高阈值：超过它直接 suppress（忽略size）
    size_ratio_thresh=1.25,
    order_invariant_sizes=True,
    eps=1e-9,
    return_iou_matrix=True,
    iou_matrix_torch=False,
):
    """
    New rule:
      1) if IoU > hard_iou_thresh: suppress directly
      2) elif IoU > iou_thresh: suppress only if sizes are similar
      3) else: keep

    Returns:
      - if return_iou_matrix: (keep_inds, iou_mat[K,K])
      - else: keep_inds
    """
    bc = _as_numpy(box_corners).astype(np.float64)
    sc = _as_numpy(scores).astype(np.float64).reshape(-1)
    bs = _as_numpy(box_sizes).astype(np.float64)

    N = bc.shape[0]
    assert bc.shape == (N, 8, 3), f"box_corners should be (N,8,3), got {bc.shape}"
    assert sc.shape[0] == N, f"scores should be (N,), got {sc.shape}"
    assert bs.shape == (N, 3), f"box_sizes should be (N,3), got {bs.shape}"

    order = np.argsort(-sc)

    # precompute planes & volume (for faster IoU)
    pre = []
    for i in range(N):
        c = bc[i]
        try:
            vol = float(ConvexHull(c).volume)
            planes = _group_box_planes_from_hull(c)
        except QhullError:
            vol = 0.0
            planes = None
        pre.append({"vol": vol, "planes": planes})

    iou_cache = {}  # (min_i, max_i) -> iou
    keep = []

    for idx in order:
        if pre[idx]["planes"] is None or pre[idx]["vol"] <= 0:
            continue

        suppressed = False
        for kept in keep:
            a, b = (idx, kept) if idx < kept else (kept, idx)
            key = (a, b)

            if key in iou_cache:
                iou = iou_cache[key]
            else:
                iou = obb_iou_corners(bc[idx], bc[kept], pre1=pre[idx], pre2=pre[kept], eps=eps)
                iou_cache[key] = iou

            # ---------- NEW RULE ----------
            if iou > hard_iou_thresh:
                suppressed = True
                break

            if iou > iou_thresh:
                # only then check size similarity
                if _sizes_similar(
                    bs[idx], bs[kept],
                    size_ratio_thresh=size_ratio_thresh,
                    order_invariant=order_invariant_sizes,
                ):
                    suppressed = True
                    break
                # else: do NOT suppress (containment allowed)
            # else: iou too small -> no suppress
            # ------------------------------

        if not suppressed:
            keep.append(int(idx))

    if not return_iou_matrix:
        return keep

    # ---- IoU matrix among kept ----
    K = len(keep)
    iou_mat = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        iou_mat[i, i] = 1.0

    for ii in range(K):
        for jj in range(ii + 1, K):
            gi, gj = keep[ii], keep[jj]
            a, b = (gi, gj) if gi < gj else (gj, gi)
            key = (a, b)
            if key in iou_cache:
                iou = iou_cache[key]
            else:
                iou = obb_iou_corners(bc[gi], bc[gj], pre1=pre[gi], pre2=pre[gj], eps=eps)
                iou_cache[key] = iou
            iou_mat[ii, jj] = iou_mat[jj, ii] = np.float32(iou)

    if iou_matrix_torch or _is_torch(box_corners):
        iou_mat = torch.from_numpy(iou_mat)  # cpu tensor

    return keep, iou_mat

def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

def apply_sim3_np(pts, c, R, t):
    """y = c*(x@R.T) + t ; pts: [...,3]"""
    p = np.asarray(pts, dtype=np.float64)
    return c * (p @ R.T) + t





def apply_sim3_to_pcd(pcd: o3d.geometry.PointCloud, c: float, R: np.ndarray, t: np.ndarray):
    """
    Apply Sim(3): x' = c * R * x + t  to an Open3D point cloud (in-place).

    Args:
        pcd: open3d PointCloud
        c: scalar scale
        R: (3,3)
        t: (3,) or (3,1) or (1,3)
    """
    pts = np.asarray(pcd.points)  # (N,3)

    t = np.asarray(t).reshape(3,)           # ensure (3,)
    R = np.asarray(R).reshape(3, 3)
    c = float(c)

    # (N,3) -> (N,3)
    pts_new = (c * (pts @ R.T)) + t[None, :]
    pcd.points = o3d.utility.Vector3dVector(pts_new)

    # 如果你有 normals，也可以按 R 旋转（不缩放）
    if pcd.has_normals():
        n = np.asarray(pcd.normals)
        n_new = n @ R.T
        pcd.normals = o3d.utility.Vector3dVector(n_new)

    return pcd

import random
def _triangles_from_corners_via_hull(corners8: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    给定 [8,3] corners（顺序任意），通过凸包推断三角面。
    返回 triangles: [T,3]，索引是 0..7（对应 corners8 的索引）。
    """
    corners8 = np.asarray(corners8, dtype=np.float64)
    if corners8.shape != (8, 3):
        raise ValueError(f"corners8 shape must be [8,3], got {corners8.shape}")

    # 去重检查（避免 hull 因重复点崩）
    # 用 rounding 做鲁棒去重
    key = np.round(corners8 / max(eps, 1e-12)).astype(np.int64)
    _, unique_idx = np.unique(key, axis=0, return_index=True)
    if unique_idx.size < 8:
        raise ValueError(f"Degenerate corners: only {unique_idx.size}/8 unique points")

    # 用 Open3D 做凸包
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(corners8))
    hull_mesh, _ = pcd.compute_convex_hull()
    hull_mesh.remove_duplicated_triangles()
    hull_mesh.remove_degenerate_triangles()
    hull_mesh.remove_duplicated_vertices()
    hull_mesh.remove_unreferenced_vertices()

    hv = np.asarray(hull_mesh.vertices)      # hull vertices（应该是 8 个，但顺序可能变）
    ht = np.asarray(hull_mesh.triangles)     # hull triangles（对盒子应该 ~12 个）

    if hv.shape[0] < 4 or ht.shape[0] < 4:
        raise ValueError("Hull seems invalid (too few vertices/triangles)")

    # hull_mesh 的顶点顺序 != 原 corners8 的顺序
    # 我们把 hull vertices 映射回 corners8 的索引（最近邻匹配）
    # 因为 hull 顶点就是 corners8 的子集（对于盒子通常是全部 8 个）
    # 用距离矩阵做 argmin 即可（8x8 很小）
    d2 = ((hv[:, None, :] - corners8[None, :, :]) ** 2).sum(-1)  # [Nh, 8]
    map_hv_to_c = np.argmin(d2, axis=1)  # [Nh] -> corners index 0..7

    # 额外做个 sanity：最近距离不能太大
    min_d = np.sqrt(np.min(d2, axis=1))
    if np.max(min_d) > 1e-4:
        # 如果你的坐标单位非常大/非常小可适当放宽阈值
        raise ValueError(f"Hull->corner mapping seems off, max nearest dist = {np.max(min_d)}")

    # 把 hull triangles 顶点索引（指向 hv）换成 corners8 索引
    tri = map_hv_to_c[ht]  # [T,3] in 0..7

    # 去掉可能的重复三角形（无向）
    tri_sorted = np.sort(tri, axis=1)
    _, keep = np.unique(tri_sorted, axis=0, return_index=True)
    tri = tri[keep]

    # 盒子应当是 12 个三角形；但如果数据噪声/重复点处理，可能略有出入
    # 这里不强制等于12，只要能构成闭合凸包即可
    return tri.astype(np.int32)


def boxes3d_to_ply(corners: np.ndarray, output_path: str, seed: int = 0):
    """
    corners: [N,8,3] numpy array，8个角点顺序任意
    output_path: 输出 ply
    seed: 固定随机种子使颜色可复现
    """
    corners = np.asarray(corners)
    assert corners.ndim == 3 and corners.shape[1:] == (8, 3), f"corners must be [N,8,3], got {corners.shape}"
    N = corners.shape[0]

    random.seed(seed)

    # 顶点：每个 box 8 个
    vertices = np.zeros((N * 8, 3), dtype=np.float32)
    vertex_colors = np.zeros((N * 8, 3), dtype=np.float32)

    all_faces = []  # 变长累计，最后再 stack

    for i in range(N):
        start_idx = i * 8
        end_idx = (i + 1) * 8

        # 写入顶点
        corners_i = corners[i].astype(np.float32)
        vertices[start_idx:end_idx] = corners_i

        # 给每个 box 一个颜色（你也可以换成更“离散”的调色板）
        color = [random.random(), random.random(), random.random()]
        vertex_colors[start_idx:end_idx] = color

        # 根据坐标自动推断三角面
        try:
            tri_local = _triangles_from_corners_via_hull(corners_i)  # [T,3] in 0..7
        except Exception as e:
            # fallback：如果 hull 失败，就退回到“原模板”(假设角点顺序正确)
            # 你如果不想 fallback，可以直接 raise
            print(f"[WARN] box {i}: hull failed ({e}), fallback to template faces.")
            tri_local = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7]
            ], dtype=np.int32)

        # 转全局索引
        tri_global = tri_local + start_idx
        all_faces.append(tri_global)

    faces = np.vstack(all_faces).astype(np.int32)

    # 创建 mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.astype(np.float64))

    # 可选：计算法线（方便渲染）
    mesh.compute_vertex_normals()

    # 保存
    o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
    print(f"成功保存 {len(corners)} 个带颜色的立方体到: {output_path}")
    




def cam0_to_world_boxes_by_w2c0(boxes_cam0: np.ndarray, w2c0: np.ndarray) -> np.ndarray:
    """
    boxes_cam0: [N,8,3] in cam0
    Keep all-zero padded boxes unchanged.
    """
    boxes_cam0 = np.asarray(boxes_cam0, dtype=np.float64)
    assert boxes_cam0.ndim == 3 and boxes_cam0.shape[1:] == (8, 3), boxes_cam0.shape

    bbox_sum = boxes_cam0.sum(axis=(-2, -1))    # [N]
    padding_mask = (bbox_sum == 0.0)

    out = boxes_cam0.copy()
    if (~padding_mask).any():
        valid = out[~padding_mask]              # [Nv,8,3]
        valid_w = cam0_to_world_points_by_w2c0(valid.reshape(-1, 3), w2c0).reshape(valid.shape)
        out[~padding_mask] = valid_w
    return out

def cam0_to_world_points_by_w2c0(points_cam0: np.ndarray, w2c0: np.ndarray) -> np.ndarray:
    """
    points_cam0: [...,3] in cam0 frame
    w2c0: [4,4] world->cam0 extrinsic (w2c)
    return points_world: [...,3]
    Using your convention:
        x_cam = x_world @ R^T + t
        => x_world = (x_cam - t) @ R
    """
    w2c0 = np.asarray(w2c0, dtype=np.float64)
    R = w2c0[:3, :3]   # [3,3]
    t = w2c0[:3, 3]    # [3]
    p = np.asarray(points_cam0, dtype=np.float64)
    return (p - t[None, :]) @ R


def transform_ply_with_plyfile(in_ply: str, out_ply: str, s: float, R: np.ndarray, t: np.ndarray):
    # pip install plyfile
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(in_ply)
    if "vertex" not in ply:
        raise ValueError("PLY has no 'vertex' element")

    v = ply["vertex"].data  # structured array
    names = v.dtype.names
    for k in ("x", "y", "z"):
        if k not in names:
            raise ValueError(f"PLY vertex missing '{k}'")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    xyz2 = apply_sim3_np(xyz, s, R, t)

    # create new vertex structured array with same dtype, replace xyz
    v2 = np.empty(v.shape, dtype=v.dtype)
    for n in names:
        v2[n] = v[n]
    v2["x"] = xyz2[:, 0].astype(v["x"].dtype, copy=False)
    v2["y"] = xyz2[:, 1].astype(v["y"].dtype, copy=False)
    v2["z"] = xyz2[:, 2].astype(v["z"].dtype, copy=False)

    vertex_el = PlyElement.describe(v2, "vertex")

    # keep other elements (face, etc.) unchanged
    elements = [vertex_el]
    for el in ply.elements:
        if el.name != "vertex":
            elements.append(el)

    out = PlyData(elements, text=ply.text)
    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
    out.write(out_ply)


def transform_ply_with_open3d(in_ply: str, out_ply: str, s: float, R: np.ndarray, t: np.ndarray):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(in_ply)
    pts = np.asarray(pcd.points, dtype=np.float64)
    pts2 = apply_sim3_np(pts, s, R, t)
    pcd.points = o3d.utility.Vector3dVector(pts2)

    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
    ok = o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False)
    if not ok:
        raise RuntimeError(f"open3d failed to write: {out_ply}")

def transform_ply(in_ply: str, out_ply: str, s: float, R: np.ndarray, t: np.ndarray):
    # try open3d first
    try:
        transform_ply_with_open3d(in_ply, out_ply, s, R, t)
        return "open3d"
    except ImportError:
        pass
    except Exception as e:
        # open3d exists but failed; fallback to plyfile
        open3d_err = e
    else:
        open3d_err = None

    try:
        transform_ply_with_plyfile(in_ply, out_ply, s, R, t)
        return "plyfile"
    except ImportError:
        raise ImportError(
            "Need either open3d or plyfile to read/write PLY.\n"
            "Install one of them:\n"
            "  pip install open3d\n"
            "  pip install plyfile"
        )
    except Exception as e:
        if open3d_err is not None:
            raise RuntimeError(f"open3d failed: {open3d_err}\nplyfile also failed: {e}")
        raise

def apply_se3_to_corners(corners: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    corners: (N,8,3)
    T: (4,4) SE(3)
    return: (N,8,3)
    """
    assert corners.ndim == 3 and corners.shape[2] == 3, corners.shape
    assert T.shape == (4, 4), T.shape

    N, K, _ = corners.shape  # K=8
    pts = corners.reshape(-1, 3)  # (N*8, 3)

    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)  # (M,4)
    pts_h2 = (pts_h @ T.T)  # (M,4)  行向量写法
    pts2 = pts_h2[:, :3]

    return pts2.reshape(N, K, 3)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=str, required=True, help="pred poses .npy, shape [F,4,4]")
    ap.add_argument("--pred_gt_root", type=str, required=True, help="gt root")
    ap.add_argument("--box_threshold", type=float, default=0.8, help="box threshold")
    args = ap.parse_args()
    gt_root = '/data1/lyq/CA1M-dataset/CA1M-dataset/test/'
    
    
    scenes = [
    "42446540", "42897501", "42897521", "42897538", "42897545",
    "42897552", "42897561", "42897599", "42897647", "42897688",
    "42897692", "42898486", "42898521", "42898538", "42898570",
    "42898811", "42898849", "42898867", "42899459", "42899611",
    "42899617", "42899679", "42899691", "42899698", "42899712",
    "42899725", "42899729", "42899736",
    "43896260", "43896321", "43896330",
    "44358442", "44358451",
    "45260854", "45260898", "45260903", "45260920",
    "45261121", "45261133", "45261143", "45261179",
    "45261575", "45261587", "45261615", "45261631",
    "45662921", "45662942", "45662970", "45662981",
    "45663113", "45663149", "45663164",
    "47115452", "47115469", "47115525", "47115543",
    "47204552", "47204559", "47204573", "47204605",
    "47331068", "47331262", "47331311", "47331319",
    "47331651", "47331661", "47331963", "47331971",
    "47331988", "47332000", "47332885", "47332893",
    "47332915", "47333431", "47333440", "47333452",
    "47333898", "47333916", "47333923", "47333927",
    "47333934", "47334107", "47334115", "47334234",
    "47334239", "47334256",
    "47430475", "47430485",
    "47895341", "47895364", "47895534", "47895542", "47895552",
    "48018345", "48018367", "48018375", "48018382",
    "48018559", "48018566",
    "48018730", "48018737", "48018947",
    "48458415", "48458427", "48458481", "48458647", "48458654",
    ]
    # scenes = ['43896260']
    import os
    for seq in scenes:
        for i in range(2):
            
            # if i != 1:
                # continue
            
            pred_root = os.path.join(args.pred_root, seq, str(i))
            gt_root = os.path.join(args.pred_gt_root, seq, str(i))
            filterd_pred_ply = os.path.join(args.pred_root,'ply', seq + '_' + str(i) + '_pred.ply')
            pcd_other = o3d.io.read_point_cloud(filterd_pred_ply)
            group_id = str(i)
            print(f'running on {seq} {group_id}')

            pred_pkl_path = os.path.join(args.pred_root, seq +"_"+ str(i)+'_pred.pkl')
            with open(pred_pkl_path, 'rb') as f:
                pred_data = pickle.load(f)
            
            pred_box = pred_data['box_result']['corners'] #np.load(os.path.join(pred_root, 'all_bboxes.npy'))    # [N,8,3]
            gt_data = np.load(os.path.join(gt_root, 'gt_data.npz'))

            
            # if os.path.exists(os.path.join(pred_root, 'all_bboxes_aligned.npy')) and os.path.exists(os.path.join(pred_root, 'all_bboxes_aligned.ply')):
            #     print(f'{pred_root} already exists')
            #     continue
            
            pred_pts = pred_data['world_points']
            gt_pts = gt_data['gt_world_points']
            # 将 valid_mask 扩展成 shape [N, h, w]，N = gt_pts.shape[0]
            valid_mask = gt_data['valid_mask']

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            print('c', c)
            print('R', R)
            print('t', t)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T
            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)
            
            t = np.asarray(t).reshape(1, 1, 3)   # -> [1,1,3]，方便broadcast
            R = np.asarray(R).reshape(3, 3)

            pred_box_ue = c * np.einsum('nkj,ij->nki', pred_box, R) + t   # [N,8,3]
            
            import os 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            
            # option 2
            # 估计法向量（半径一般取 2~3 倍 voxel）
            radius = 0.02 * 3.0
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
            pcd_gt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

            init_T = np.eye(4)
            max_corr = 0.02 * 3.0

            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd, pcd_gt,
                max_corr, init_T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)
            
            pcd_other = apply_sim3_to_pcd(pcd_other, c, R, t)
            # 检查 ICP transform 的平移距离，如果超过 5m 则不应用
            translation = transformation[0:3, 3]
            translation_distance = np.linalg.norm(translation)
            if translation_distance <= 5.0:
                pcd_other.transform(transformation)
                pred_box_ue = apply_se3_to_corners(pred_box_ue, reg_p2p.transformation)

            save_ply_path = os.path.join(args.pred_root, 'ply', seq + '_' + str(i) + '_ours_scene_aligned.ply')
            print('save_ply_path', save_ply_path)
            ok = o3d.io.write_point_cloud(save_ply_path, pcd_other)

            
            # threshold filtering pred_boxes
 
            box_mask = pred_data['box_result']['scores'] > args.box_threshold
            all_scores = pred_data['box_result']['scores'][box_mask]
            all_sizes  = pred_data['box_result']['size'][box_mask]
            pred_box_ue = pred_box_ue[box_mask]


            
            # keep = obb_nms_corners_sizeaware(all_corners, all_scores, all_sizes, iou_thresh=0.1, size_ratio_thresh=1.25)  # keep 是 index 列表
            keep, iou_mat = obb_nms_corners_sizeaware_v2(
                                pred_box_ue, all_scores, all_sizes,
                                iou_thresh=0.01, #0.1,
                                hard_iou_thresh=0.50,
                                size_ratio_thresh=2.0,#1.5,
                                return_iou_matrix=True,
                                iou_matrix_torch=True,   # 想要torch就开
                            )
            pred_box_ue = pred_box_ue[keep]
            
            # print('pred_box_aligned', pred_box_aligned.shape)
            box_dir = os.path.join(args.pred_root, 'box')
            os.makedirs(box_dir, exist_ok=True)
            boxes3d_to_ply(pred_box_ue, os.path.join(box_dir, seq+"_"+str(i)+'_bboxes_aligned.ply'))
            np.save(os.path.join(os.path.join(box_dir, seq+"_"+str(i)+'_bboxes_aligned.npy')), pred_box_ue)
    






if __name__ == "__main__":
    main()


'''
python pred_alignment_ca1m.py   --pred_root /data1/lyq/CA1M_results/  --pred_gt_root /data1/lyq/CA1M_results/gt_data/
'''