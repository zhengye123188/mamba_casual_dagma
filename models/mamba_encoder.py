"""
MambaCausal v2 - 模块A：Mamba 选择性因果编码器

职责：将多变量时序数据编码为特征矩阵，供 DAGMA 进行因果结构学习。

与 v1 的关键区别：
  v1: 每个目标指标独立训练一个编码器，输出预测值 (B,1,1)
  v2: 一次性处理所有指标，输出特征矩阵 (n_samples, d) 供 DAGMA 使用

Mamba 在此框架中的角色：
  - 将原始时序数据中的时间滞后因果信息编码到特征表示中
  - 弥补 DAGMA 不处理时序滞后的不足
  - Mamba 解决 "when"（什么时候的信息重要），DAGMA 解决 "what"（谁导致了谁）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureTorchMambaBlock(nn.Module):
    """
    纯 PyTorch 实现的 Mamba 选择性状态空间模型。
    选择性参数 Δ, B, C 随输入动态变化，自动决定保留/遗忘历史信息。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch_size, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbc = self.x_proj(x_conv)
        dt_input = x_dbc[..., :1]
        B = x_dbc[..., 1:1 + self.d_state]
        C = x_dbc[..., 1 + self.d_state:]
        dt = self.dt_proj(dt_input)

        # 选择性扫描
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        output = self.out_proj(y) + residual
        return output


class OfficialMambaBlock(nn.Module):
    """基于 mamba-ssm 官方库的实现（需 CUDA）"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.mamba(self.norm(x)) + x


def get_mamba_block(d_model, d_state=16, d_conv=4, expand=2, use_official=False):
    if use_official:
        try:
            return OfficialMambaBlock(d_model, d_state, d_conv, expand)
        except ImportError:
            print("WARNING: mamba-ssm 不可用，回退到纯 PyTorch 实现")
    return PureTorchMambaBlock(d_model, d_state, d_conv, expand)


class MambaFeatureEncoder(nn.Module):
    """
    Mamba 特征编码器：将多变量时序数据编码为特征矩阵。

    输入：原始时序数据 X ∈ R^{T × N}（T个时间步，N个指标）
    输出：特征矩阵 Z ∈ R^{n_samples × N}（编码后的特征，保留N个指标维度）

    DAGMA 需要的输入是 (n_samples, d) 的截面数据。
    Mamba 的作用是将时序信息编码到这个截面表示中，
    使得 DAGMA 虽然不直接处理时序，但拿到的输入已蕴含时序因果信息。

    编码策略：
      1. 用滑动窗口将时序数据切分为多个子序列
      2. 每个子序列通过 Mamba 编码，取最后时间步的输出作为该样本的特征
      3. 输出 n_samples 个特征向量，每个向量 N 维（对应 N 个指标）
    """

    def __init__(self, n_metrics, d_model=64, d_state=16, d_conv=4,
                 expand=2, n_layers=2, use_official_mamba=False):
        super().__init__()
        self.n_metrics = n_metrics
        self.d_model = d_model

        # 输入投影: N -> d_model
        self.input_proj = nn.Linear(n_metrics, d_model)

        # Mamba 层堆叠
        self.mamba_layers = nn.ModuleList([
            get_mamba_block(d_model, d_state, d_conv, expand, use_official_mamba)
            for _ in range(n_layers)
        ])

        # 输出投影: d_model -> N（投射回指标空间，供 DAGMA 使用）
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, n_metrics)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, N) 滑窗切分后的多变量时序子序列
        Returns:
            z: (batch, N) 每个子序列的编码特征（取最后时间步）
        """
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        for layer in self.mamba_layers:
            h = layer(h)  # (batch, seq_len, d_model)
        h = self.output_norm(h)  # (batch, seq_len, d_model)
        h_last = h[:, -1, :]  # (batch, d_model) 取最后时间步
        z = self.output_proj(h_last)  # (batch, N)
        return z

    def encode_timeseries(self, data, seq_len=32, stride=1, device='cpu'):
        """
        将完整时序数据编码为 DAGMA 所需的特征矩阵。

        Args:
            data: numpy array (T, N) 完整时序数据
            seq_len: 滑动窗口长度
            stride: 滑动步长
            device: 计算设备

        Returns:
            Z: numpy array (n_samples, N) 编码后的特征矩阵
        """
        import numpy as np

        T, N = data.shape

        # 滑动窗口切分
        windows = []
        for i in range(0, T - seq_len + 1, stride):
            windows.append(data[i:i + seq_len])
        windows = np.array(windows)  # (n_samples, seq_len, N)

        # 分批编码
        self.eval()
        batch_size = 256
        all_features = []

        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = torch.FloatTensor(windows[start:start + batch_size]).to(device)
                features = self.forward(batch)  # (batch, N)
                all_features.append(features.cpu().numpy())

        Z = np.concatenate(all_features, axis=0)  # (n_samples, N)
        return Z
