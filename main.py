"""
MambaCausal v2 - 主入口

完整流程：
1. 加载时序数据
2. Mamba 编码器训练 + 特征提取
3. DAGMA 全局因果图学习
4. RobustScorer + PageRank 根因评分
5. 输出结果

用法：
# 完整 MambaCausal v2
python main.py --root_path ./data --data_path sock_shop_cpu.csv --root_cause target_metric

# 消融：不用 Mamba（直接用原始数据喂 DAGMA）
python main.py --root_path ./data --data_path sock_shop_cpu.csv --root_cause target --no_mamba

# 消融：不用 RobustScorer（用固定个性化向量）
python main.py --root_path ./data --data_path sock_shop_cpu.csv --root_cause target --no_scorer

# 粗颗粒度实验（只用 latency 列，根因为服务名）
python main.py --root_path ./data --data_path RE1-SS/carts_cpu/1 --root_cause auto --coarse_grained

# 细颗粒度实验（只用指定服务的资源指标列）
python main.py --root_path ./data --data_path RE1-SS/carts_cpu/1 --root_cause auto --fine_grained_service carts
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.mamba_encoder import MambaFeatureEncoder
from models.dagma_causal import learn_causal_dag, DagmaMLP, DagmaNonlinear
from models.root_cause_scorer import root_cause_ranking, evaluate_ranking, build_causal_graph

import networkx as nx


def load_data(root_path, data_path):
    """
    加载数据，支持两种格式：
    1. 直接 CSV 文件: --data_path some_file.csv
    2. RE1 案例目录: --data_path RE1-SS/carts_cpu/1
       (自动读取 simple_data.csv 和 inject_time.txt)
    """
    filepath = os.path.join(root_path, data_path)

    if os.path.isdir(filepath):
        # RE1 目录格式
        csv_file = os.path.join(filepath, 'simple_data.csv')
        if not os.path.exists(csv_file):
            csv_file = os.path.join(filepath, 'data.csv')

        df = pd.read_csv(csv_file)

        # 读取故障注入时间
        inject_time = None
        inject_file = os.path.join(filepath, 'inject_time.txt')
        if os.path.exists(inject_file):
            with open(inject_file, 'r') as f:
                inject_time = int(f.read().strip())

        # 去掉 time 列
        time_col = None
        if 'time' in df.columns:
            time_col = df['time'].values
            df = df.drop('time', axis=1)

        columns = list(df.columns)
        data = df.values.astype(np.float32)

        # 用 inject_time 精确划分故障前后
        fault_idx = None
        if inject_time is not None and time_col is not None:
            fault_idx = np.searchsorted(time_col, inject_time)

        print(f"数据加载完成: {data.shape[0]} 时间步 × {data.shape[1]} 指标")
        if inject_time:
            print(f"  故障注入时间: {inject_time}, 对应行: {fault_idx}")

        return data, columns, fault_idx

    else:
        # 普通 CSV 文件
        df = pd.read_csv(filepath)
        if 'time' in df.columns:
            df = df.drop('time', axis=1)
        columns = list(df.columns)
        data = df.values.astype(np.float32)
        print(f"数据加载完成: {data.shape[0]} 时间步 × {data.shape[1]} 指标")
        return data, columns, None


def train_mamba_encoder(data, args):
    """
    阶段1：训练 Mamba 编码器。
    使用自监督方式训练：用滑窗切出的子序列，让 Mamba 学习预测下一时间步。
    训练完成后，用编码器将整个时序数据编码为特征矩阵。
    """
    # ★ 关键修复：强制恢复全局 dtype 为 float32
    #   因为 DAGMA 的 fit() 会把全局 dtype 改成 double，
    #   在 run_batch.py 的循环中，后续案例创建 Mamba 编码器时
    #   全局 dtype 已经是 double，导致 nn.Linear 权重变成 double，
    #   但输入数据是 FloatTensor(float32) → dtype 不匹配报错
    torch.set_default_dtype(torch.float32)

    T, N = data.shape
    device = args.device

    print(f"\n{'='*50}")
    print(f"阶段1: 训练 Mamba 编码器")
    print(f"  指标数: {N}, 时序长度: {T}")
    print(f"  d_model: {args.d_model}, n_layers: {args.n_layers}")
    print(f"{'='*50}")

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 构建滑窗训练数据
    seq_len = args.seq_len
    windows_x = []  # 输入: [t, t+seq_len)
    windows_y = []  # 目标: t+seq_len 时刻的值

    for i in range(T - seq_len):
        windows_x.append(data_scaled[i:i+seq_len])
        windows_y.append(data_scaled[i+seq_len])

    X_train = torch.FloatTensor(np.array(windows_x)).to(device)
    Y_train = torch.FloatTensor(np.array(windows_y)).to(device)  # (n, N)

    print(f"  训练样本数: {len(X_train)}")

    # 构建模型
    encoder = MambaFeatureEncoder(
        n_metrics=N,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        use_official_mamba=args.use_official_mamba
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.mamba_lr)

    # 训练：自监督预测
    encoder.train()
    batch_size = min(args.batch_size, len(X_train))
    n_batches = len(X_train) // batch_size

    for epoch in range(1, args.mamba_epochs + 1):
        total_loss = 0
        indices = torch.randperm(len(X_train))
        for b in range(n_batches):
            idx = indices[b*batch_size:(b+1)*batch_size]
            batch_x = X_train[idx]
            batch_y = Y_train[idx]

            optimizer.zero_grad()
            pred = encoder(batch_x)  # (batch, N)
            loss = torch.nn.functional.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{args.mamba_epochs}, Loss: {avg_loss:.6f}")

    # 编码：将整个时序数据转换为特征矩阵
    print(f"  编码时序数据为特征矩阵...")
    encoder.eval()
    Z = encoder.encode_timeseries(
        data_scaled, seq_len=seq_len, stride=args.stride, device=device
    )
    print(f"  特征矩阵: {Z.shape}")

    return Z, encoder, scaler


def run_dagma(feature_matrix, n_metrics, args):
    """
    阶段2：DAGMA 全局因果图学习。
    输入 Mamba 编码后的特征矩阵，输出因果 DAG。
    """
    print(f"\n{'='*50}")
    print(f"阶段2: DAGMA 因果图学习")
    print(f"  输入特征: {feature_matrix.shape}")
    print(f"  hidden_dim: {args.dagma_hidden}, T: {args.dagma_T}")
    print(f"{'='*50}")

    W = learn_causal_dag(
        feature_matrix=feature_matrix,
        n_metrics=n_metrics,
        hidden_dim=args.dagma_hidden,
        lambda1=args.dagma_lambda1,
        lambda2=args.dagma_lambda2,
        T=args.dagma_T,
        w_threshold=args.dagma_threshold,
        lr=args.dagma_lr,
        warm_iter=args.dagma_warm_iter,
        max_iter=args.dagma_max_iter,
        verbose=args.verbose,
        device=args.device
    )

    # ★ 关键修复：DAGMA 完成后强制恢复 float32，防止污染后续 Mamba
    torch.set_default_dtype(torch.float32)

    # 统计因果图信息
    n_edges = np.sum(W > 0)
    print(f"  学到的因果边数: {n_edges}")
    print(f"  邻接矩阵非零率: {n_edges / (n_metrics * n_metrics):.4f}")

    return W


def run_scoring(W, columns, data, args, fault_idx=None):
    """
    阶段3：RobustScorer + PageRank 根因评分。
    """
    T, N = data.shape

    # 用 inject_time 精确划分，否则用 fault_ratio
    if fault_idx is not None and fault_idx > 0:
        pre_fault = data[:fault_idx]
        post_fault = data[fault_idx:]
    else:
        fault_split = int(T * (1 - args.fault_ratio))
        pre_fault = data[:fault_split]
        post_fault = data[fault_split:]

    print(f"\n{'='*50}")
    print(f"阶段3: 根因评分")
    print(f"  故障前数据: {pre_fault.shape}, 故障后数据: {post_fault.shape}")
    print(f"  评分方法: {args.scorer_method}")
    print(f"{'='*50}")

    if args.no_scorer:
        print(f"  [消融] 跳过 RobustScorer，使用纯 PageRank")
        G = build_causal_graph(W, columns)
        personalization = {}
        dangling = [n for n in G.nodes() if G.out_degree(n) == 0]
        for node in G.nodes():
            personalization[node] = 1.0 if node in dangling else 0.5
        try:
            pr = nx.pagerank(G, personalization=personalization, weight='weight')
        except:
            pr = personalization
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        ranked_nodes = [n for n, _ in ranked]
        info = {'graph_edges': G.number_of_edges(), 'is_dag': nx.is_directed_acyclic_graph(G)}
    else:
        ranked_nodes, info = root_cause_ranking(
            W, columns, pre_fault, post_fault,
            scorer_method=args.scorer_method,
            alpha=args.pagerank_alpha
        )

    return ranked_nodes, info


def main(args):
    """主流程"""
    start_time = time.time()

    # ====== 加载数据 ======
    data, columns, fault_idx = load_data(args.root_path, args.data_path)
    T, N = data.shape

    # ====== 自动推断根因标签 ======
    # auto 模式：从路径推断故障服务，再根据粗/细颗粒度决定标签粒度
    if args.root_cause == 'auto':
        parts = args.data_path.strip('/').split('/')
        if len(parts) >= 2:
            fault_type = parts[-2]          # 如 carts_cpu
            service = fault_type.rsplit('_', 1)[0]   # 如 carts
            metric_suffix = fault_type.rsplit('_', 1)[-1]  # 如 cpu

            if args.coarse_grained:
                # 粗颗粒度：根因为服务名，如 carts
                args.root_cause = service
                print(f"  [粗颗粒度] 自动推断根因服务: {args.root_cause}")
            else:
                # 细颗粒度：根因为具体指标名，如 carts_cpu
                args.root_cause = fault_type
                print(f"  [细颗粒度] 自动推断根因指标: {args.root_cause}")

    # ====== 粗颗粒度：只保留各服务的 latency-50 列（每个服务一个节点） ======
    if args.coarse_grained:
        # 优先用 latency-50（P50中位数），每个服务只保留一个延迟指标
        latency_cols = [c for c in columns if 'latency-50' in c]
        if len(latency_cols) == 0:
            # 回退：尝试所有 latency/response 列
            latency_cols = [c for c in columns if 'latency' in c or 'response' in c]
        if len(latency_cols) == 0:
            print("  WARNING: 未找到 latency 列，将使用全部列")
        else:
            col_idx = [columns.index(c) for c in latency_cols]
            data = data[:, col_idx]
            columns = latency_cols
            print(f"  [粗颗粒度] 保留 {len(columns)} 个服务延迟列: {columns}")

    # ====== 细颗粒度：只保留指定服务的资源指标列 ======
    if args.fine_grained_service:
        svc = args.fine_grained_service
        svc_cols = [c for c in columns if c.startswith(svc + '_') and 'latency' not in c and 'response' not in c]
        if len(svc_cols) == 0:
            print(f"  WARNING: 未找到服务 {svc} 的资源指标列，将使用全部列")
        else:
            col_idx = [columns.index(c) for c in svc_cols]
            data = data[:, col_idx]
            columns = svc_cols
            print(f"  [细颗粒度] 过滤后保留列: {columns}")

    # ====== 阶段1: Mamba 编码 ======
    if args.no_mamba:
        print("\n[消融] 跳过 Mamba 编码，使用原始数据")
        scaler = StandardScaler()
        Z = scaler.fit_transform(data)
    else:
        Z, encoder, scaler = train_mamba_encoder(data, args)

    # ====== 阶段2: DAGMA 因果图 ======
    W = run_dagma(Z, data.shape[1], args)

    # ====== 阶段3: 根因评分 ======
    ranked_nodes, info = run_scoring(W, columns, data, args, fault_idx=fault_idx)

    # ====== 输出结果 ======
    elapsed = time.time() - start_time

    # 构建结果字典
    result_data = {
        'data_path': args.data_path,
        'root_cause': args.root_cause,
        'n_metrics': data.shape[1],
        'n_samples': T,
        'elapsed_seconds': round(elapsed, 2),
        'graph_edges': info.get('graph_edges', -1),
        'is_dag': info.get('is_dag', False),
        'use_mamba': not args.no_mamba,
        'use_scorer': not args.no_scorer,
        'coarse_grained': args.coarse_grained,
        'fine_grained_service': args.fine_grained_service,
        'top10_ranking': ranked_nodes[:10],
        'full_ranking': ranked_nodes,
    }

    if args.root_cause and args.root_cause != 'None':
        eval_result = evaluate_ranking(ranked_nodes, args.root_cause)
        result_data.update(eval_result)

    if 'anomaly_scores' in info:
        result_data['anomaly_scores'] = info['anomaly_scores']
    if 'pagerank_scores' in info:
        result_data['pagerank_scores'] = info['pagerank_scores']

    # 保存结果到 JSON
    import json
    os.makedirs('results', exist_ok=True)
    data_name = os.path.splitext(args.data_path)[0].replace('/', '_')
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    config_tag = 'full'
    if args.coarse_grained:
        config_tag = 'coarse'
    elif args.fine_grained_service:
        config_tag = f'fine_{args.fine_grained_service}'
    elif args.no_mamba and args.no_scorer:
        config_tag = 'no_mamba_no_scorer'
    elif args.no_mamba:
        config_tag = 'no_mamba'
    elif args.no_scorer:
        config_tag = 'no_scorer'

    result_file = f'results/{data_name}_{config_tag}_{timestamp}.json'

    # 将 numpy/torch 类型转为 Python 原生类型
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False, default=convert)

    # 同时保存邻接矩阵
    adj_file = f'results/{data_name}_{config_tag}_{timestamp}_adj.npy'
    np.save(adj_file, W)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"MambaCausal v2 结果")
    print(f"{'='*60}")
    print(f"  因果图: {info.get('graph_edges', '?')} 条边, DAG: {info.get('is_dag', '?')}")
    print(f"  运行时间: {elapsed:.1f}s")

    if args.root_cause and args.root_cause != 'None':
        result = evaluate_ranking(ranked_nodes, args.root_cause)
        print(f"  真实根因: {args.root_cause}")
        print(f"  排名位置: {result['rank']}")
        print(f"  AC@1: {result['AC@1']}, AC@2: {result['AC@2']}, AC@3: {result['AC@3']}, AC@4: {result['AC@4']}, AC@5: {result['AC@5']}")
        print(f"  Avg@5: {result['Avg@5']:.4f}")

    print(f"\n--- Top-10 根因排序 ---")
    for i, node in enumerate(ranked_nodes[:10]):
        matched = args.root_cause and (node == args.root_cause or node.startswith(args.root_cause + '_'))
        marker = "  <<<" if matched else ""
        print(f"  {i+1}. {node}{marker}")

    print(f"\n  结果已保存: {result_file}")
    print(f"  邻接矩阵已保存: {adj_file}")
    print(f"{'='*60}")

    return ranked_nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MambaCausal v2')

    # 数据参数
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--root_cause', type=str, default='None')
    parser.add_argument('--fault_ratio', type=float, default=0.25)

    # ====== 新增：粗/细颗粒度开关 ======
    parser.add_argument('--coarse_grained', action='store_true',
                        help='粗颗粒度实验：只用 latency 列，根因为服务名')
    parser.add_argument('--fine_grained_service', type=str, default=None,
                        help='细颗粒度实验：指定服务名（如 carts），只用该服务的资源指标列')

    # Mamba 参数
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--mamba_epochs', type=int, default=50)
    parser.add_argument('--mamba_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--use_official_mamba', action='store_true')

    # DAGMA 参数
    parser.add_argument('--dagma_hidden', type=int, default=10)
    parser.add_argument('--dagma_lambda1', type=float, default=0.05,
                        help='L1稀疏系数，越大因果图越稀疏')
    parser.add_argument('--dagma_lambda2', type=float, default=0.005)
    parser.add_argument('--dagma_T', type=int, default=5)
    parser.add_argument('--dagma_lr', type=float, default=0.0002)
    parser.add_argument('--dagma_threshold', type=float, default=0.5,
                        help='边权重阈值，越大保留的边越少')
    parser.add_argument('--dagma_warm_iter', type=int, default=5000)
    parser.add_argument('--dagma_max_iter', type=int, default=8000)

    # 评分参数
    parser.add_argument('--scorer_method', type=str, default='modified_zscore',
                        choices=['modified_zscore', 'iqr', 'mannwhitney'])
    parser.add_argument('--pagerank_alpha', type=float, default=0.85)

    # 消融开关
    parser.add_argument('--no_mamba', action='store_true', help='消融: 不用 Mamba 编码')
    parser.add_argument('--no_scorer', action='store_true', help='消融: 不用 RobustScorer')

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if not torch.cuda.is_available() and 'cuda' in args.device:
        print("WARNING: CUDA 不可用，切换到 CPU")
        args.device = 'cpu'

    main(args)