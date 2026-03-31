"""
MambaCausal v2 消融实验脚本

配置矩阵：
  Full:       Mamba + DAGMA + RobustScorer  (完整方法)
  -Mamba:     原始数据 + DAGMA + RobustScorer  (验证 Mamba 编码器贡献)
  -Scorer:    Mamba + DAGMA + 固定向量  (验证 RobustScorer 贡献)
  -Both:      原始数据 + DAGMA + 固定向量  (只有 DAGMA)
"""

import subprocess
import sys
import json
import time
import argparse
from datetime import datetime


CONFIGS = {
    "Full_MambaCausal": {
        "flags": [],
        "desc": "完整方法: Mamba + DAGMA + RobustScorer"
    },
    "Ablation_no_Mamba": {
        "flags": ["--no_mamba"],
        "desc": "消融Mamba: 原始数据 + DAGMA + RobustScorer"
    },
    "Ablation_no_Scorer": {
        "flags": ["--no_scorer"],
        "desc": "消融Scorer: Mamba + DAGMA + 固定向量"
    },
    "Ablation_no_Both": {
        "flags": ["--no_mamba", "--no_scorer"],
        "desc": "消融两者: 原始数据 + DAGMA + 固定向量"
    },
}


def main():
    parser = argparse.ArgumentParser(description='MambaCausal v2 消融实验')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--root_cause', type=str, default='None')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--configs', type=str, nargs='*', default=None)
    args = parser.parse_args()
    
    configs = {k: v for k, v in CONFIGS.items() if not args.configs or k in args.configs}
    
    print(f"\nMambaCausal v2 消融实验")
    print(f"数据: {args.root_path}/{args.data_path}")
    print(f"配置数: {len(configs)}\n")
    
    results = []
    for name, cfg in configs.items():
        print(f"\n{'='*50}")
        print(f"实验: {name} — {cfg['desc']}")
        print(f"{'='*50}")
        
        cmd = [
            sys.executable, "main.py",
            "--root_path", args.root_path,
            "--data_path", args.data_path,
            "--root_cause", args.root_cause,
            "--device", args.device,
        ] + cfg["flags"]
        
        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            elapsed = time.time() - start
            results.append({
                "config": name, "desc": cfg["desc"],
                "stdout": result.stdout, "elapsed": elapsed,
                "status": "success" if result.returncode == 0 else "failed"
            })
            # 打印最后 15 行
            for line in result.stdout.strip().split('\n')[-15:]:
                print(f"  {line}")
        except subprocess.TimeoutExpired:
            results.append({"config": name, "status": "timeout", "elapsed": 7200})
    
    # 汇总
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"ablation_v2_{ts}.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"{'配置':<25} {'状态':<10} {'耗时(s)':<10}")
    for r in results:
        print(f"{r['config']:<25} {r['status']:<10} {r.get('elapsed',0):<10.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
