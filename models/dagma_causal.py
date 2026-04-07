"""
MambaCausal v2 - 模块B：DAGMA 全局因果图学习

职责：从 Mamba 编码后的特征矩阵中，一次性学出完整的因果 DAG。

替换 RUN 的三步串行流程：
  RUN: 逐指标注意力学习 + 固定阈值 H=0.5 截断 + 皮尔逊相关剪枝去环
  v2:  DAGMA 全局优化 → 自动保证 DAG，无需后处理

基于 DAGMA (NeurIPS 2022) 的 DagmaMLP + DagmaNonlinear。
"""

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import copy
from tqdm.auto import tqdm
import typing
import math


# ============================================================
# LocallyConnected 层（来自 DAGMA 官方代码）
# ============================================================

class LocallyConnected(nn.Module):
    """Conv1dLocal() with filter size 1."""

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            out += self.bias
        return out


# ============================================================
# DagmaMLP：建模结构方程的 MLP（来自 DAGMA 官方代码）
# ============================================================

class DagmaMLP(nn.Module):
    """
    用 MLP 建模每个变量的结构方程。
    每个变量 x_j = f_j(x_parents(j)) + noise_j
    """

    def __init__(self, dims, bias=True, dtype=torch.double):

        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self, s=1.0):
        """log-det 无环约束：h(W) = -log det(sI - A) + d*log(s)"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I.to(A.device) - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self):
        """L1 正则化"""
        return torch.sum(torch.abs(self.fc1.weight))

    @torch.no_grad()
    def fc1_to_adj(self):
        """从 fc1 权重提取加权邻接矩阵 W"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W


# ============================================================
# DagmaNonlinear：DAGMA 优化器（基于官方代码，适配 MambaCausal）
# ============================================================

class DagmaNonlinear:
    """
    DAGMA 非线性因果结构学习。

    核心优化问题：
      min_{W(Θ) ∈ W^s}  μ · Q(Θ; X) + h(W(Θ))
    其中 Q 是重建损失，h 是 log-det 无环约束。

    通过逐步减小 μ（中心路径），在极限处保证返回 DAG。
    """

    def __init__(self, model, verbose=False, dtype=torch.double):
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype

    def log_mse_loss(self, output, target):
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def minimize(self, max_iter, lr, lambda1, lambda2, mu, s,
                 lr_decay=False, tol=1e-6, pbar=None):
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(self.model.parameters(), lr=lr,
                               betas=(.99, .999), weight_decay=mu * lambda2)
        if lr_decay:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16

        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            X_hat = self.model(self.X)
            score = self.log_mse_loss(X_hat, self.X)
            l1_reg = lambda1 * self.model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()

            if lr_decay and (i + 1) % 1000 == 0:
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / max(abs(obj_prev), 1e-10)) <= tol:
                    if pbar:
                        pbar.update(max_iter - i)
                    break
                obj_prev = obj_new
            if pbar:
                pbar.update(1)
        return True

    def fit(self, X, lambda1=.02, lambda2=.005, T=4, mu_init=.1,
            mu_factor=.1, s=1.0, warm_iter=5e4, max_iter=8e4,
            lr=.0002, w_threshold=0.3, checkpoint=1000, device='cpu'):
        """
        运行 DAGMA 算法，从数据 X 中学习因果 DAG。

        Args:
            X: (n, d) 数据矩阵（Mamba 编码后的特征矩阵）
            lambda1: L1 正则化系数
            lambda2: L2 正则化系数
            T: DAGMA 外循环迭代次数
            mu_init: μ 初始值
            mu_factor: μ 衰减因子
            s: M-矩阵域控制参数
            warm_iter: 预热迭代次数
            max_iter: 最大迭代次数
            lr: 学习率
            w_threshold: 边权重阈值（小于此值的边被删除）
            checkpoint: 打印间隔

        Returns:
            W_est: (d, d) 加权邻接矩阵（DAG）
        """
        torch.set_default_dtype(self.dtype)
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype).to(device)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype).to(device)
        else:
            raise ValueError("X should be numpy array or torch Tensor.")

        self.checkpoint = checkpoint
        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]

        with tqdm(total=int((T - 1) * warm_iter + max_iter), desc="DAGMA") as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i + 1} -- mu: {mu}')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_cur = lr
                lr_decay = False

                while success is False:
                    success = self.minimize(inner_iter, lr_cur, lambda1, lambda2,
                                            mu, s_cur, lr_decay, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr_cur *= 0.5
                        lr_decay = True
                        if lr_cur < 1e-10:
                            break
                        s_cur = 1
                mu *= mu_factor

        W_est = self.model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est


# ============================================================
# 便捷接口：一键学习因果 DAG
# ============================================================

def learn_causal_dag(feature_matrix, n_metrics, hidden_dim=10,
                     lambda1=0.02, lambda2=0.005, T=4,
                     w_threshold=0.3, lr=0.0002,
                     warm_iter=5000, max_iter=8000,
                     verbose=False, device='cpu'):
    """
    从特征矩阵中学习因果 DAG 的便捷接口。

    Args:
        feature_matrix: numpy array (n_samples, d)，Mamba 编码后的特征
        n_metrics: 指标数量 d
        hidden_dim: DAGMA MLP 隐藏层维度
        lambda1: L1 正则化
        lambda2: L2 正则化
        T: DAGMA 外循环次数
        w_threshold: 边权重阈值
        lr: 学习率
        warm_iter: 预热迭代次数
        max_iter: 最大迭代次数
        verbose: 是否打印详细信息
        device: 计算设备

    Returns:
        W: numpy array (d, d) 加权邻接矩阵（DAG）
    """
    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # 构建 DAGMA 模型
    eq_model = DagmaMLP(dims=[n_metrics, hidden_dim, 1], bias=True, dtype=torch.float)
    eq_model = eq_model.to(device)

    model = DagmaNonlinear(eq_model, verbose=verbose, dtype=torch.double)

    # 运行 DAGMA
    W_est = model.fit(
        X_scaled,
        lambda1=lambda1, lambda2=lambda2,
        T=T, lr=lr,
        warm_iter=warm_iter, max_iter=max_iter,
        w_threshold=w_threshold,
        device=device
    )

    return W_est
