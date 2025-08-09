import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        # 投影层：将特征B的维度从2048降到256（与A对齐）
        self.proj_b = nn.Linear(2048, d_model)
        
        # PyTorch官方多头注意力模块（设置batch_first适配[N, seq, feat]格式）
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, # 输入/输出维度
            num_heads=num_heads, # 头数（256÷8=32，符合整除要求）
            dropout=dropout, # 输入格式为 [N, seq, feat]
            batch_first=True  # 关键：匹配输入维度顺序
        )
        
        # 层归一化（稳定训练）
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, A, B):
        """
        Args:
            A: [N, 1024, 256] 作为目标特征（Query）
            B: [N, 306, 2048] 作为源特征（Key/Value）
        Returns:
            aligned_B: [N, 1024, 256] 与A对齐的特征
        """
        # 1. 投影特征B至A的维度空间 [N, 306, 2048] → [N, 306, 256]
        B_proj = self.proj_b(B)
        
        # 2. 跨模态注意力计算（A作为Query，B_proj作为Key/Value）
        attn_output, _ = self.multihead_attn(
            query=A, # [N, 1024, 256]
            key=B_proj, # [N, 306, 256]
            value=B_proj
        )
        
        # 3. 残差连接 + 层归一化（保留原始A的信息）
        return self.layer_norm(attn_output + A) # [N, 1024, 256]
    
if __name__ == "__main__":
    # 初始化模块
    cmmha = CrossModalMultiHeadAttention(d_model=256, num_heads=8)

    # 模拟输入
    A = torch.randn(4, 1024, 256)  # 特征A [N, 1024, 256]
    B = torch.randn(4, 306, 2048)   # 特征B [N, 306, 2048]

    # 特征对齐
    aligned_B = cmmha(A, B)  # 输出 [4, 1024, 256]

    print("对齐后特征形状:", aligned_B.shape)  # torch.Size([4, 1024, 256])