import numpy as np


def box_size_similarity(A, B, alpha=0.2):
    """
    计算3D Box尺寸相似度
    参数：
        A: 目标尺寸数组，形状(3,)
        B: 候选尺寸数组，形状(M,3)
        alpha: 余弦相似度权重(0-1)
    返回：
        sim_array: 相似度数组，形状(M,)
    """
    # 1. 特征预处理
    sorted_A = np.sort(A)
    sorted_B = np.sort(B, axis=1)
    A_norm = preprocess_size(sorted_A)
    B_norm = preprocess_size(sorted_B)
    
    # 2. 余弦相似度（比例相似性）
    cos_sim = np.dot(B_norm, A_norm)
    
    # 3. 欧氏距离相似度（绝对尺寸）
    # 使用广播机制计算所有组合的差异[7,8](@ref)
    log_A = np.log(sorted_A + 1e-8)
    log_B = np.log(sorted_B + 1e-8)
    euclidean_dist = np.linalg.norm(log_B - log_A, axis=1)
    euclidean_sim = np.exp(-euclidean_dist / 3)  # 转换为相似度[0-1]
    
    # 4. 综合相似度
    return alpha * cos_sim + (1-alpha) * euclidean_sim

def preprocess_size(lwh):
    """尺寸预处理：对数变换 + 归一化"""
    log_lwh = np.log(lwh + 1e-8)  # 防止log(0)
    return log_lwh / np.linalg.norm(log_lwh, axis=-1, keepdims=True)
