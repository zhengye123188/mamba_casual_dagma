"""
MambaCausal v2 - 模块C：RobustScorer + PageRank 根因评分

职责：在 DAGMA 学出的因果 DAG 上，融合统计异常信号，输出根因排序。

融合三种信息：
  1. 因果拓扑（谁导致了谁）—— 来自 DAGMA 的 DAG 结构
  2. 因果强度（因果关系有多强）—— 来自 DAGMA 的边权重 W[i][j]
  3. 异常程度（指标有多异常）—— 来自 RobustScorer 的异常分数
"""

import numpy as np
import networkx as nx
from scipy import stats


class RobustScorer:
    """
    基于 BARO 的 RobustScorer：计算每个指标的统计异常分数。
    使用中位数和 MAD 的非参数方法，对极端值鲁棒。
    """
    def __init__(self, method='modified_zscore'):
        self.method = method
    
    def compute_scores(self, pre_fault_data, post_fault_data):
        """
        Args:
            pre_fault_data: (T1, N) 故障前数据
            post_fault_data: (T2, N) 故障后数据
        Returns:
            scores: (N,) 异常分数
        """
        n_metrics = pre_fault_data.shape[1]
        scores = np.zeros(n_metrics)
        
        for i in range(n_metrics):
            pre = pre_fault_data[:, i]
            post = post_fault_data[:, i]
            
            if self.method == 'modified_zscore':
                scores[i] = self._modified_zscore(pre, post)
            elif self.method == 'iqr':
                scores[i] = self._iqr_score(pre, post)
            elif self.method == 'mannwhitney':
                scores[i] = self._mannwhitney_score(pre, post)
            else:
                scores[i] = self._modified_zscore(pre, post)
        
        return scores
    
    def _modified_zscore(self, pre, post):
        median_pre = np.median(pre)
        mad = np.median(np.abs(pre - median_pre))
        if mad < 1e-10:
            std_pre = np.std(pre)
            if std_pre < 1e-10:
                return np.abs(np.median(post) - median_pre)
            return np.abs(np.median(post) - median_pre) / std_pre
        median_post = np.median(post)
        return np.abs(0.6745 * (median_post - median_pre) / mad)
    
    def _iqr_score(self, pre, post):
        q1, q3 = np.percentile(pre, 25), np.percentile(pre, 75)
        iqr = q3 - q1
        if iqr < 1e-10:
            return np.abs(np.median(post) - np.median(pre))
        return np.abs(np.median(post) - np.median(pre)) / iqr
    
    def _mannwhitney_score(self, pre, post):
        try:
            _, p_value = stats.mannwhitneyu(pre, post, alternative='two-sided')
            if p_value < 1e-300:
                p_value = 1e-300
            return -np.log(p_value)
        except ValueError:
            return 0.0


def build_causal_graph(W, columns):
    """
    从 DAGMA 的加权邻接矩阵构建 NetworkX 有向图。
    
    关键改进：使用 DAGMA 的边权重作为图的边权重，
    而非 RUN 原始代码中所有边权重相同的做法。
    
    Args:
        W: (d, d) DAGMA 输出的加权邻接矩阵，W[i][j] > 0 表示 j→i
        columns: 指标名称列表
    
    Returns:
        G: NetworkX DiGraph，带边权重
    """
    d = W.shape[0]
    G = nx.DiGraph()
    
    for i, col in enumerate(columns):
        G.add_node(col)
    
    for i in range(d):
        for j in range(d):
            if W[i][j] > 0:
                # W[i][j] > 0 表示 j→i 的因果关系，权重为因果强度
                G.add_edge(columns[j], columns[i], weight=W[i][j])
    
    return G


def root_cause_ranking(W, columns, pre_fault_data, post_fault_data,
                        scorer_method='modified_zscore',
                        alpha=0.85, max_iter=100):
    """
    MambaCausal 的根因评分：融合 DAGMA 因果图和 RobustScorer 异常分数。
    
    Args:
        W: (d, d) DAGMA 加权邻接矩阵
        columns: 指标名称列表
        pre_fault_data: (T1, N) 故障前数据
        post_fault_data: (T2, N) 故障后数据
        scorer_method: 异常评分方法
        alpha: PageRank 阻尼因子
        max_iter: PageRank 最大迭代次数
    
    Returns:
        ranked_nodes: 按根因可能性排序的节点列表
        info: 包含中间结果的字典
    """
    # ====== 步骤1：构建带权因果图 ======
    G = build_causal_graph(W, columns)
    
    # ====== 步骤2：计算异常分数 ======
    scorer = RobustScorer(method=scorer_method)
    anomaly_scores = scorer.compute_scores(pre_fault_data, post_fault_data)
    
    # ====== 步骤3：构建个性化向量 ======
    smoothing = 1e-6
    personalization = {}
    total = 0.0
    for i, col in enumerate(columns):
        score = anomaly_scores[i] + smoothing
        personalization[col] = score
        total += score
    for col in personalization:
        personalization[col] /= total
    
    # ====== 步骤4：运行 PageRank ======
    try:
        pagerank_scores = nx.pagerank(
            G, alpha=alpha, personalization=personalization,
            max_iter=max_iter, weight='weight'  # 使用 DAGMA 边权重
        )
    except nx.PowerIterationFailedConvergence:
        pagerank_scores = personalization
    
    # ====== 步骤5：排序 ======
    ranked = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_nodes = [node for node, _ in ranked]
    
    info = {
        'anomaly_scores': {columns[i]: anomaly_scores[i] for i in range(len(columns))},
        'pagerank_scores': pagerank_scores,
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'is_dag': nx.is_directed_acyclic_graph(G),
    }
    
    return ranked_nodes, info


def evaluate_ranking(ranking, root_cause):
    """评估根因排序结果。"""
    result = {'AC@1': 0, 'AC@3': 0, 'AC@5': 0, 'rank': -1}
    
    for i, node in enumerate(ranking):
        if node == root_cause:
            result['rank'] = i + 1
            if i < 1: result['AC@1'] = 1
            if i < 3: result['AC@3'] = 1
            if i < 5: result['AC@5'] = 1
            break
    
    return result
