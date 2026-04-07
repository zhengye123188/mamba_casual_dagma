"""
批量运行 RE1 数据集的所有故障案例（实时输出版）。

用法:
  # 粗颗粒度（只用 latency 列，根因为服务名）
  python run_batch.py --dataset RE1-SS --coarse_grained --device cuda:0 2>&1 | tee batch_ss_coarse.log

  # 细颗粒度（已知故障服务，只用该服务资源指标）
  python run_batch.py --dataset RE1-SS --device cuda:0 2>&1 | tee batch_ss_fine.log

  # 细颗粒度（全量指标）
  python run_batch.py --dataset RE1-SS --fine_all --device cuda:0 2>&1 | tee batch_ss_fine_all.log

  python run_batch.py --dataset all --device cuda:0 2>&1 | tee batch_all.log
  python run_batch.py --dataset RE1-SS --fault_type cpu --device cuda:0
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
from datetime import datetime

# 直接 import main 里的函数，不走子进程
from main import load_data, train_mamba_encoder, run_dagma, run_scoring
from models.root_cause_scorer import evaluate_ranking
from sklearn.preprocessing import StandardScaler


def find_cases(data_root, dataset, fault_type=None):
    """扫描数据目录，找到所有故障案例。"""
    dataset_dir = os.path.join(data_root, dataset)
    if not os.path.exists(dataset_dir):
        print(f"ERROR: 数据集目录不存在: {dataset_dir}")
        return []

    cases = []
    for fault_dir in sorted(os.listdir(dataset_dir)):
        fault_path = os.path.join(dataset_dir, fault_dir)
        if not os.path.isdir(fault_path):
            continue
        parts = fault_dir.rsplit('_', 1)
        if len(parts) != 2:
            continue
        service, ftype = parts
        if fault_type and ftype != fault_type:
            continue
        for repeat in sorted(os.listdir(fault_path)):
            repeat_path = os.path.join(fault_path, repeat)
            if os.path.isdir(repeat_path):
                case_path = f"{dataset}/{fault_dir}/{repeat}"
                cases.append((case_path, service, ftype, repeat))
    return cases


def run_single_case(data_root, case_path, root_cause, args, service=None):
    """
    直接调用 main 的各阶段函数，实时输出到终端。
    返回排序结果、评估指标、邻接矩阵、列名。
    """
    # 加载数据
    data, columns, fault_idx = load_data(data_root, case_path)
    T, N = data.shape

    # ====== 粗颗粒度：只保留各服务的 latency 列 ======
    if args.coarse_grained:
        latency_cols = [c for c in columns if 'latency' in c or 'response' in c]
        if len(latency_cols) == 0:
            print(f"  WARNING: 未找到 latency/response 列，将使用全部列")
        else:
            col_idx = [columns.index(c) for c in latency_cols]
            data = data[:, col_idx]
            columns = latency_cols
            print(f"  [粗颗粒度] 保留列数: {len(columns)}")

    # ====== 细颗粒度：只保留指定服务的资源指标列 ======
    elif not args.fine_all and service is not None:
        svc_cols = [c for c in columns if c.startswith(service + '_') and 'latency' not in c and 'response' not in c]
        if len(svc_cols) == 0:
            print(f"  WARNING: 未找到服务 {service} 的资源指标列，将使用全部列")
        else:
            col_idx = [columns.index(c) for c in svc_cols]
            data = data[:, col_idx]
            columns = svc_cols
            print(f"  [细颗粒度] 保留列数: {len(columns)}, 列: {columns}")

    # fine_all 模式：不做任何列过滤，使用全量指标

    # 阶段1: Mamba 编码
    if args.no_mamba:
        print(f"  [消融] 跳过 Mamba 编码")
        scaler = StandardScaler()
        Z = scaler.fit_transform(data)
    else:
        Z, encoder, scaler = train_mamba_encoder(data, args)

    # 阶段2: DAGMA 因果图
    W = run_dagma(Z, data.shape[1], args)

    # 阶段3: 根因评分
    ranked_nodes, info = run_scoring(W, columns, data, args, fault_idx=fault_idx)

    # 评估
    result = evaluate_ranking(ranked_nodes, root_cause)

    return ranked_nodes, result, info, W, columns


def main():
    parser = argparse.ArgumentParser(description='批量运行 RE1 实验')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='RE1-SS')
    parser.add_argument('--fault_type', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')

    # ====== 粗/细颗粒度开关 ======
    parser.add_argument('--coarse_grained', action='store_true',
                        help='粗颗粒度：只用 latency 列，根因为服务名')
    parser.add_argument('--fine_all', action='store_true',
                        help='细颗粒度全量：使用所有指标，根因为具体指标名')
    # 不加这两个开关时，默认为细颗粒度（已知服务）：只用故障服务的资源指标列

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
    parser.add_argument('--dagma_lambda1', type=float, default=0.05)
    parser.add_argument('--dagma_lambda2', type=float, default=0.005)
    parser.add_argument('--dagma_T', type=int, default=5)
    parser.add_argument('--dagma_lr', type=float, default=0.0002)
    parser.add_argument('--dagma_threshold', type=float, default=0.5)
    parser.add_argument('--dagma_warm_iter', type=int, default=5000)
    parser.add_argument('--dagma_max_iter', type=int, default=8000)

    # 评分参数
    parser.add_argument('--scorer_method', type=str, default='modified_zscore')
    parser.add_argument('--pagerank_alpha', type=float, default=0.85)
    parser.add_argument('--fault_ratio', type=float, default=0.25)

    # 消融开关
    parser.add_argument('--no_mamba', action='store_true')
    parser.add_argument('--no_scorer', action='store_true')

    # ====== 补全 main.py 里有但之前缺失的参数 ======
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # 确定数据集
    if args.dataset == 'all':
        datasets = ['RE1-OB', 'RE1-SS', 'RE1-TT']
    else:
        datasets = [args.dataset]

    all_cases = []
    for ds in datasets:
        all_cases.extend(find_cases(args.data_root, ds, args.fault_type))

    # 打印实验模式
    if args.coarse_grained:
        mode_str = '粗颗粒度（latency列，根因=服务名）'
    elif args.fine_all:
        mode_str = '细颗粒度-全量指标（根因=service_ftype）'
    else:
        mode_str = '细颗粒度-已知服务（根因=service_ftype）'

    print(f"\n{'#'*60}")
    print(f"MambaCausal v2 批量实验")
    print(f"数据集: {datasets}, 故障类型: {args.fault_type or '全部'}")
    print(f"实验模式: {mode_str}")
    print(f"总案例数: {len(all_cases)}")
    print(f"{'#'*60}")

    # 结果目录
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_tag = 'full'
    if args.coarse_grained: config_tag = 'coarse'
    elif args.fine_all: config_tag = 'fine_all'
    if args.no_mamba: config_tag += '_no_mamba'
    if args.no_scorer: config_tag += '_no_scorer'
    if args.no_mamba and args.no_scorer: config_tag = config_tag.replace('_no_mamba_no_scorer', '_no_both')

    # numpy/torch 类型转换
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj

    # 逐个运行
    all_results = []
    ac_sums = {'AC@1': 0, 'AC@2': 0, 'AC@3': 0, 'AC@4': 0, 'AC@5': 0}

    for i, (case_path, service, ftype, repeat) in enumerate(all_cases):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(all_cases)}] {case_path}")
        print(f"  根因服务: {service}, 故障类型: {ftype}, 重复: {repeat}")
        print(f"{'='*60}")

        # 根因标签：粗颗粒度用服务名，细颗粒度用具体指标名
        if args.coarse_grained:
            root_cause = service                  # 如 carts
        else:
            root_cause = f"{service}_{ftype}"     # 如 carts_cpu

        start = time.time()
        try:
            ranked_nodes, r, info, W, columns = run_single_case(
                args.data_root, case_path, root_cause, args, service=service
            )
            elapsed = time.time() - start

            for k in ac_sums:
                ac_sums[k] += r[k]

            print(f"\n  >>> 结果: rank={r['rank']}, AC@1={r['AC@1']}, AC@3={r['AC@3']}, AC@5={r['AC@5']}, Avg@5={r['Avg@5']:.2f} ({elapsed:.1f}s)")

            case_record = {
                'case': case_path,
                'service': service,
                'fault_type': ftype,
                'repeat': repeat,
                'elapsed': round(elapsed, 2),
                'status': 'success',
                'root_cause': root_cause,
                'n_metrics': len(columns),
                'columns': columns,
                'graph_edges': info.get('graph_edges', -1),
                'is_dag': info.get('is_dag', False),
                'top10': ranked_nodes[:10],
                'full_ranking': ranked_nodes,
                **r,
            }
            # anomaly_scores 和 pagerank_scores 与单例保持一致
            if 'anomaly_scores' in info:
                case_record['anomaly_scores'] = info['anomaly_scores']
            if 'pagerank_scores' in info:
                case_record['pagerank_scores'] = info['pagerank_scores']

            all_results.append(case_record)

            # 保存单个案例结果 JSON（与单例运行格式完全一致）
            case_name = case_path.replace('/', '_')
            case_file = f'results/{case_name}_{config_tag}_{timestamp}.json'
            with open(case_file, 'w', encoding='utf-8') as f:
                json.dump(case_record, f, indent=2, ensure_ascii=False, default=convert)

            # 保存单个案例邻接矩阵
            adj_file = f'results/{case_name}_{config_tag}_{timestamp}_adj.npy'
            np.save(adj_file, W)

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  >>> FAILED: {str(e)} ({elapsed:.1f}s)")
            traceback.print_exc()
            all_results.append({
                'case': case_path, 'service': service, 'fault_type': ftype,
                'repeat': repeat, 'elapsed': round(elapsed, 2), 'status': 'failed',
                'root_cause': root_cause, 'error': str(e),
                'AC@1': 0, 'AC@2': 0, 'AC@3': 0, 'AC@4': 0, 'AC@5': 0, 'Avg@5': 0, 'rank': -1
            })

    # ====== 汇总 ======
    n = len(all_results)
    n_success = sum(1 for r in all_results if r['status'] == 'success')

    print(f"\n\n{'#'*60}")
    print(f"汇总结果")
    print(f"{'#'*60}")
    print(f"  总案例: {n}, 成功: {n_success}")
    if n_success > 0:
        for k in ['AC@1', 'AC@2', 'AC@3', 'AC@4', 'AC@5']:
            print(f"  {k}: {ac_sums[k]}/{n_success} = {ac_sums[k]/n_success:.4f}")
        avg5 = sum(ac_sums[f'AC@{i}'] for i in range(1, 6)) / (n_success * 5)
        print(f"  Avg@5: {avg5:.4f}")

    # 按故障类型分组
    fault_types = sorted(set(r['fault_type'] for r in all_results))
    print(f"\n--- 按故障类型分组 ---")
    print(f"  {'类型':<8} {'n':<5} {'AC@1':<8} {'AC@3':<8} {'AC@5':<8} {'Avg@5':<8}")
    for ft in fault_types:
        ft_results = [r for r in all_results if r['fault_type'] == ft and r['status'] == 'success']
        if ft_results:
            ft_n = len(ft_results)
            ft_ac1 = sum(r['AC@1'] for r in ft_results) / ft_n
            ft_ac3 = sum(r['AC@3'] for r in ft_results) / ft_n
            ft_ac5 = sum(r['AC@5'] for r in ft_results) / ft_n
            ft_avg5 = sum(r.get('Avg@5', 0) for r in ft_results) / ft_n
            print(f"  {ft:<8} {ft_n:<5} {ft_ac1:<8.4f} {ft_ac3:<8.4f} {ft_ac5:<8.4f} {ft_avg5:<8.4f}")

    # 保存汇总 JSON
    avg5_overall = sum(ac_sums[f'AC@{i}'] for i in range(1, 6)) / max(n_success * 5, 1)

    summary_file = f'results/batch_{args.dataset}_{config_tag}_{timestamp}.json'
    summary = {
        'dataset': args.dataset,
        'fault_type': args.fault_type,
        'mode': mode_str,
        'config': config_tag,
        'timestamp': timestamp,
        'total_cases': n,
        'success_cases': n_success,
        'AC@1': ac_sums['AC@1'] / max(n_success, 1),
        'AC@2': ac_sums['AC@2'] / max(n_success, 1),
        'AC@3': ac_sums['AC@3'] / max(n_success, 1),
        'AC@4': ac_sums['AC@4'] / max(n_success, 1),
        'AC@5': ac_sums['AC@5'] / max(n_success, 1),
        'Avg@5': avg5_overall,
        'by_fault_type': {},
        'cases': all_results,
    }
    for ft in fault_types:
        ft_results = [r for r in all_results if r['fault_type'] == ft and r['status'] == 'success']
        if ft_results:
            ft_n = len(ft_results)
            summary['by_fault_type'][ft] = {
                'n': ft_n,
                'AC@1': sum(r['AC@1'] for r in ft_results) / ft_n,
                'AC@3': sum(r['AC@3'] for r in ft_results) / ft_n,
                'AC@5': sum(r['AC@5'] for r in ft_results) / ft_n,
                'Avg@5': sum(r.get('Avg@5', 0) for r in ft_results) / ft_n,
            }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=convert)

    print(f"\n  汇总结果已保存: {summary_file}")
    print(f"  各案例结果已保存至 results/ 目录")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()