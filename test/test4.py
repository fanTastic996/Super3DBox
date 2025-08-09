import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels=1024):
        super().__init__()
        # 多尺度卷积分支（1x1, 3x3, 5x5）
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, in_channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1)  # 通道压缩

    def forward(self, x):
        # x: [3, 1024, 256] → 重组为 [1024, 3, 256]
        x = x.permute(1, 0, 2).unsqueeze(0)  # [1, 1024, 3, 256]
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        cat = torch.cat([c1, c2, c3], dim=1)  # [1, 3 * 1024, 3, 256]
        fused = self.fuse(cat).squeeze(0)     # [1024, 3, 256] → 压缩为 [1024, 256]
        return fused
    
if __name__ == "__main__":
    # 初始化模块
    cmmha = CrossModalMultiHeadAttention(d_model=256, num_heads=8)

    # 模拟输入
    A = torch.randn(4, 1024, 256)  # 特征A [N, 1024, 256]
    B = torch.randn(4, 306, 2048)   # 特征B [N, 306, 2048]

    # 特征对齐
    aligned_B = cmmha(A, B)  # 输出 [4, 1024, 256]

    print("对齐后特征形状:", aligned_B.shape)  # torch.Size([4, 1024, 256])