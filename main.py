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
    """加载 CSV 数据"""
    filepath = os.path.join(root_path, data_path)
    df = pd.read_csv(filepath)

    # 去掉 time 列（如果有）
    if 'time' in df.columns:
        df = df.drop('time', axis=1)

    columns = list(df.columns)
    data = df.values.astype(np.float32)

    print(f"数据加载完成: {data.shape[0]} 时间步 × {data.shape[1]} 指标")
    return data, columns


def train_mamba_encoder(data, args):
    """
    阶段1：训练 Mamba 编码器。

    使用自监督方式训练：用滑窗切出的子序列，让 Mamba 学习预测下一时间步。
    训练完成后，用编码器将整个时序数据编码为特征矩阵。
    """
    T, N = data.shape
    device = args.device

    print(f"\n{'=' * 50}")
    print(f"阶段1: 训练 Mamba 编码器")
    print(f"  指标数: {N}, 时序长度: {T}")
    print(f"  d_model: {args.d_model}, n_layers: {args.n_layers}")
    print(f"{'=' * 50}")

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 构建滑窗训练数据
    seq_len = args.seq_len
    windows_x = []  # 输入: [t, t+seq_len)
    windows_y = []  # 目标: t+seq_len 时刻的值

    for i in range(T - seq_len):
        windows_x.append(data_scaled[i:i + seq_len])
        windows_y.append(data_scaled[i + seq_len])

    X_train = torch.FloatTensor(np.array(windows_x)).to(device)  # (n, seq_len, N)
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
            idx = indices[b * batch_size:(b + 1) * batch_size]
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
    print(f"\n{'=' * 50}")
    print(f"阶段2: DAGMA 因果图学习")
    print(f"  输入特征: {feature_matrix.shape}")
    print(f"  hidden_dim: {args.dagma_hidden}, T: {args.dagma_T}")
    print(f"{'=' * 50}")

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

    # 统计因果图信息
    n_edges = np.sum(W > 0)
    print(f"  学到的因果边数: {n_edges}")
    print(f"  邻接矩阵非零率: {n_edges / (n_metrics * n_metrics):.4f}")

    return W


def run_scoring(W, columns, data, args):
    """
    阶段3：RobustScorer + PageRank 根因评分。
    """
    T, N = data.shape
    fault_split = int(T * (1 - args.fault_ratio))
    pre_fault = data[:fault_split]
    post_fault = data[fault_split:]

    print(f"\n{'=' * 50}")
    print(f"阶段3: 根因评分")
    print(f"  故障前数据: {pre_fault.shape}, 故障后数据: {post_fault.shape}")
    print(f"  评分方法: {args.scorer_method}")
    print(f"{'=' * 50}")

    if args.no_scorer:
        # 消融：不用 RobustScorer，用固定个性化向量
        G = build_causal_graph(W, columns)
        dangling = [n for n, d in G.out_degree() if d == 0]
        personalization = {}
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
    data, columns = load_data(args.root_path, args.data_path)
    T, N = data.shape

    # ====== 阶段1: Mamba 编码 ======
    if args.no_mamba:
        # 消融：跳过 Mamba，直接用原始数据
        print("\n[消融] 跳过 Mamba 编码，使用原始数据")
        scaler = StandardScaler()
        Z = scaler.fit_transform(data)
    else:
        Z, encoder, scaler = train_mamba_encoder(data, args)

    # ====== 阶段2: DAGMA 因果图 ======
    W = run_dagma(Z, N, args)

    # ====== 阶段3: 根因评分 ======
    ranked_nodes, info = run_scoring(W, columns, data, args)

    # ====== 输出结果 ======
    elapsed = time.time() - start_time

    # 构建结果字典
    result_data = {
        'data_path': args.data_path,
        'root_cause': args.root_cause,
        'n_metrics': N,
        'n_samples': T,
        'elapsed_seconds': round(elapsed, 2),
        'graph_edges': info.get('graph_edges', -1),
        'is_dag': info.get('is_dag', False),
        'use_mamba': not args.no_mamba,
        'use_scorer': not args.no_scorer,
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
    if args.no_mamba and args.no_scorer:
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
    print(f"\n{'=' * 60}")
    print(f"MambaCausal v2 结果")
    print(f"{'=' * 60}")
    print(f"  因果图: {info.get('graph_edges', '?')} 条边, DAG: {info.get('is_dag', '?')}")
    print(f"  运行时间: {elapsed:.1f}s")

    if args.root_cause and args.root_cause != 'None':
        result = evaluate_ranking(ranked_nodes, args.root_cause)
        print(f"  真实根因: {args.root_cause}")
        print(f"  排名位置: {result['rank']}")
        print(f"  AC@1: {result['AC@1']}, AC@3: {result['AC@3']}, AC@5: {result['AC@5']}")

    print(f"\n--- Top-10 根因排序 ---")
    for i, node in enumerate(ranked_nodes[:10]):
        marker = " <<<" if args.root_cause and node == args.root_cause else ""
        print(f"  {i + 1}. {node}{marker}")

    print(f"\n  结果已保存: {result_file}")
    print(f"  邻接矩阵已保存: {adj_file}")
    print(f"{'=' * 60}")
    return ranked_nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MambaCausal v2')

    # 数据参数
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--root_cause', type=str, default='None')
    parser.add_argument('--fault_ratio', type=float, default=0.25)

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
    parser.add_argument('--dagma_lambda1', type=float, default=0.02)
    parser.add_argument('--dagma_lambda2', type=float, default=0.005)
    parser.add_argument('--dagma_T', type=int, default=4)
    parser.add_argument('--dagma_lr', type=float, default=0.0002)
    parser.add_argument('--dagma_threshold', type=float, default=0.3)
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