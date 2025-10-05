import time
import math
from typing import Dict, Optional
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from hardware import HardwareParams
from quantum_chip import QuantumChip
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from qiskit.visualization import dag_drawer


class TimeAwareCompiler:
    """
      1) 初始映射（基于交互图的启发式方法）
      2) 窗口调度 + 贪心映射 + 重用跟踪
      3) 通过简单的时空近似（或距离）估算交换成本
      4) 可选的局部CP-SAT优化窗口

    输入:
       - circuit: QuantumCircuit（读取qasm或qiskit电路）
       - topo: Topology（物理硬件图）
       - hw: HardwareParams
       - params: 权重/阈值（lambda_makespan, lambda_swap, lambda_idle, ...）
    输出:
       - 调度序列（简单表示）
       - 指标
    """

    def __init__(self, circuit: QuantumCircuit, topo: QuantumChip, hw: HardwareParams, params: Optional[Dict] = None):
        self.quantum_circuit = circuit
        self.topo = topo
        self.hw = hw
        self.params = params if params is not None else {}
        # 默认权重
        self.lambda_makespan = self.params.get("lambda_makespan", 1.0)
        self.lambda_swap = self.params.get("lambda_swap", 1.0)
        self.lambda_idle = self.params.get("lambda_idle", 1.0)
        self.cp_sat_window_threshold = self.params.get(
            "cp_sat_window_threshold", 1e-2)
        # 运行时结构
        self.mapping = {}  # logical_qubit -> physical_node
        self.reverse_mapping = {}  # physical_node -> logical or None
        # 每个物理节点被占用直到的时间
        self.occupied_until = {p: 0.0 for p in self.topo.nodes()}
        # 指标记录 (physical_node, released_time, reused_time) 列表，用于计算等待时间
        self.reuse_events = []
        self.timeline_ops = []  # 调度的操作：包含时间和类型的字典

    # 辅助方法
    def build_interaction_graph(self, dag=None):
        """
        从电路构建逻辑量子比特的交互图（两量子比特操作的频率）。
        返回：networkx图，逻辑量子比特节点，边上的权重=#两量子比特操作。
        """
        if dag is None:
            dag = circuit_to_dag(self.quantum_circuit)
        G = nx.Graph()
        num_q = self.quantum_circuit.num_qubits
        for i in range(num_q):
            G.add_node(i)
        for node in dag.op_nodes():  # qiskit DAGNode
            if node.op.num_qubits > 1:
                qargs = [self.quantum_circuit.find_bit(
                    q).index for q in node.qargs]
                if len(qargs) == 2:
                    u, v = qargs
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
                else:
                    # 对于k量子比特门，添加完全图计数
                    for i in range(len(qargs)):
                        for j in range(i+1, len(qargs)):
                            u, v = qargs[i], qargs[j]
                            if G.has_edge(u, v):
                                G[u][v]['weight'] += 1
                            else:
                                G.add_edge(u, v, weight=1)
        return G

    def initial_mapping_by_interaction(self):
        """
        简单的贪心映射：
         - 按交互图中的度数对逻辑量子比特排序
         - 将高度数的逻辑量子比特填充到连接良好的物理节点上（按度数）
        这是基线；可以用更好的映射替换（例如，SABRE / VF2匹配等）
        """
        inter = self.build_interaction_graph()
        logical_sorted = sorted(inter.degree, key=lambda x: -x[1])
        physical_sorted = sorted([(n, self.topo.graph.degree(n)) for n in self.topo.nodes()],
                                 key=lambda x: -x[1])
        self.mapping = {}
        self.reverse_mapping = {}
        for idx, (l, _) in enumerate(logical_sorted):
            if idx < len(physical_sorted):
                p = physical_sorted[idx][0]
                self.mapping[l] = p
                self.reverse_mapping[p] = l
            else:
                # 没有物理量子比特 => 暂时保持未映射（如果节点足够多则不应发生）
                self.mapping[l] = None
        # 初始化任何剩余的物理量子比特为未映射
        for p in self.topo.nodes():
            if p not in self.reverse_mapping:
                self.reverse_mapping[p] = None
        return self.mapping

    def shortest_path_distance(self, p, q):
        try:
            return nx.shortest_path_length(self.topo.graph, p, q)
        except nx.NetworkXNoPath:
            return math.inf

    def estimate_swap_time_distance(self, p, q):
        """
        简化的 swap_time 估计：按图距离 * avg t_2q（或具体边的总和）。
        更精确的可替换为 space-time A*（见接口 space_time_swap_estimate）
        """
        d = self.shortest_path_distance(p, q)
        if d == math.inf:
            return math.inf
        # 为了使两个量子比特相邻，需要 ~d-1 个交换（取决于方案）。
        # 近似总交换时间：
        avg_t2q = self.hw.time_2q
        # 因子3作为保守的SWAP成本（硬件相关）
        return max(0, d - 1) * avg_t2q * 3.0

    def space_time_swap_estimate(self, p, q, t_now, depth_limit=50):
        """
        接口/简化的时空估计：
        - 可以在时空图中运行有限的A*来估计两个量子比特可以相邻的最早时间。
        - 这里我们返回（estimated_extra_time, swap_count_estimate）
        这是一个占位符，使用简单模型：swap_time = estimate_swap_time_distance
        """
        est = self.estimate_swap_time_distance(p, q)
        swap_count = max(0, self.shortest_path_distance(p, q) - 1)
        return est, swap_count

    # 调度
    def schedule(self, strategy="windowed_greedy", window_size=5):
        """
        主入口：使用指定策略调度电路。
        实现的策略：'windowed_greedy'（简单），如果OR-Tools可用则可选调用本地CP-SAT。
        返回指标字典和时间线。
        """
        t_start = time.time()

        dag = circuit_to_dag(self.quantum_circuit)
        self.initial_mapping_by_interaction()
        # 我们将按层进行简单的拓扑扫描：
        layers = self.dag_to_layers(dag)

        # 重置结构
        self.occupied_until = {p: 0.0 for p in self.topo.nodes()}
        self.mapping = self.mapping  # 保持初始映射
        self.reverse_mapping = {p: self.mapping.get(
            l) for l, p in self.mapping.items()} if self.mapping else self.reverse_mapping
        # 确保反向映射一致：
        self.reverse_mapping = {p: None for p in self.topo.nodes()}
        for l, p in self.mapping.items():
            if p is not None:
                self.reverse_mapping[p] = l
        self.reuse_events = []
        self.timeline_ops = []

        current_time = 0.0
        # 我们将维护一个简单的"ASAP"逐层方案；在一层内，按原始顺序处理门。
        for layer_idx, layer_nodes in enumerate(layers):
            # 根据前驱完成时间确定层可以开始的最早时间（非常粗糙：计算占用时间的最大值）
            # 在这个骨架中，我们只是在每层中顺序调度门，同时尊重occupied_until。
            for node in layer_nodes:
                opname = node.name
                qargs = [self.quantum_circuit.find_bit(
                    q).index for q in node.qargs]
                # 基于前驱完成时间计算最早就绪时间：
                pred_finish = 0.0
                for pred in dag.predecessors(node):
                    # 查找pred的调度完成时间（扫描时间线）-> 我们没有记录每个门的完成时间；为了保持代码紧凑，
                    # 我们用current_time近似
                    pred_finish = max(pred_finish, current_time)
                ready_time = max(current_time, pred_finish)
                # 确保操作数已映射；如果没有，从空闲池中分配（简单）
                phys_args = []
                for lq in qargs:
                    if self.mapping.get(lq) is None:
                        # 选择最早可用时间的空闲物理量子比特
                        chosen = min(self.occupied_until.items(),
                                     key=lambda kv: kv[1])[0]
                        self.mapping[lq] = chosen
                        self.reverse_mapping[chosen] = lq
                    phys_args.append(self.mapping[lq])

                # 按操作码大小分支
                if len(phys_args) == 1:
                    p = phys_args[0]
                    start = max(ready_time, self.occupied_until[p])
                    duration = self.hw.get_t1q(p)
                    # 调度
                    self.timeline_ops.append({"type": "1Q", "op": opname, "qargs": qargs, "p": [
                                             p], "start": start, "dur": duration})
                    self.occupied_until[p] = start + duration
                    # 我们允许门在其开始时间启动（ASAP）
                    current_time = start
                elif len(phys_args) == 2:
                    p1, p2 = phys_args
                    # 如果不相邻，估计交换时间并可选择插入交换
                    if not self.topo.graph.has_edge(p1, p2):
                        est_swap_time, swap_count = self.space_time_swap_estimate(
                            p1, p2, ready_time)
                        # 选择插入交换或重新映射一个操作数到附近的空闲量子比特
                        # 最简单策略：如果有完全空闲的节点使其廉价地相邻，则重新映射；否则插入交换
                        # 查找现在可用的空闲池节点
                        free_candidates = [u for u, occ in self.occupied_until.items(
                        ) if occ <= ready_time and self.reverse_mapping[u] is None]
                        # 尝试将第二个量子比特重新映射到与p1相邻的候选节点
                        remapped = False
                        for cand in free_candidates:
                            if self.topo.graph.has_edge(p1, cand):
                                # 重新映射 l2 -> cand
                                l2 = None
                                # 反向查找p2的逻辑量子比特
                                l2 = self.reverse_mapping.get(p2, None)
                                if l2 is not None:
                                    # 释放p2然后映射l2->cand
                                    self.reverse_mapping[p2] = None
                                    self.mapping[l2] = cand
                                    self.reverse_mapping[cand] = l2
                                    p2 = cand
                                    remapped = True
                                    break
                        if not remapped:
                            # 插入交换延迟
                            start = max(
                                ready_time, self.occupied_until[p1], self.occupied_until[p2])
                            # 我们保守地添加est_swap_time
                            start_after_swap = start + est_swap_time
                            # 在交换后调度2Q操作
                            start = start_after_swap
                        else:
                            start = max(
                                ready_time, self.occupied_until[p1], self.occupied_until[p2])
                    else:
                        start = max(
                            ready_time, self.occupied_until[p1], self.occupied_until[p2])
                    duration = self.hw.get_t2q(p1, p2)
                    self.timeline_ops.append({"type": "2Q", "op": opname, "qargs": qargs, "p": [
                                             p1, p2], "start": start, "dur": duration})
                    self.occupied_until[p1] = start + duration
                    self.occupied_until[p2] = start + duration
                    current_time = start
                else:
                    # k量子比特门：近似为多个2Q + 1Q片段或块
                    # 这里我们使用朴素模型：选择共同时间=max占用，持续时间求和（近似）
                    phys = phys_args
                    start = max([self.occupied_until[p]
                                for p in phys] + [ready_time])
                    dur = self.hw.time_2q * (len(phys)-1)  # 粗略
                    self.timeline_ops.append(
                        {"type": "kQ", "op": opname, "qargs": qargs, "p": phys, "start": start, "dur": dur})
                    for p in phys:
                        self.occupied_until[p] = start + dur
                    current_time = start

                # 测量处理：将测量+重置视为阻塞物理量子比特直到开始+测量+重置
                if node.op.name.lower().startswith("measure"):
                    # 对于qiskit DAG测量节点，qargs指量子寄存器；假设一个目标
                    p = self.mapping[qargs[0]]
                    meas_t = self.hw.get_tmeas(p) + self.hw.get_treset(p)
                    # 记录释放/重用事件（如果后续重用）
                    release_time = self.occupied_until[p]
                    # 我们标记物理节点被阻塞直到release_time
                    self.timeline_ops[-1]["dur"] = meas_t
                    self.timeline_ops[-1]["p"] = [p]
                    self.occupied_until[p] = self.timeline_ops[-1]["start"] + meas_t
                    # 取消映射逻辑量子比特（重置后逻辑槽可用于重用）
                    l = qargs[0]
                    self.reverse_mapping[self.mapping[l]] = None
                    self.mapping[l] = None
                    # 标记释放（用于指标）
                    self.reuse_events.append(
                        {"p": p, "released_at": self.occupied_until[p], "reused_at": None})

            # 处理完一层后，可选择对前几层运行本地CP-SAT（窗口）
            # 占位符：如果OR-Tools可用且满足阈值，则运行本地优化
            # （留作扩展）
            # current_time = max(current_time, max(self.occupied_until.values()))

        elapsed = time.time() - t_start
        metrics = self.compute_metrics(elapsed)
        return {"timeline": self.timeline_ops, "metrics": metrics}

    def dag_to_layers(self, dag):
        """
        将DAG转换为层列表（节点列表的列表），类似于层次化。
        简单的BFS-like层次化：依赖关系中相同深度的节点。
        """

        # dag_drawer(dag, filename="my_dag.png")

        indeg = {n: len(list(dag.predecessors(n)))
                 for n in dag.topological_op_nodes()}
        # 注意：qiskit DAG节点处理；如果可用，我们将使用dag.layers()
        try:
            layers = []
            for layer in dag.layers():
                # 每层有'ops'作为节点
                layers.append([nd for nd in layer['graph'].op_nodes()])
            return layers
        except Exception:
            # 备用方案：计算层级
            levels = {}
            for n in dag.topological_op_nodes():
                levels[n] = 0
            changed = True
            while changed:
                changed = False
                for n in dag.topological_op_nodes():
                    preds = list(dag.predecessors(n))
                    if preds:
                        newlevel = max(levels[p] for p in preds) + 1
                        if newlevel != levels[n]:
                            levels[n] = newlevel
                            changed = True
            maxlevel = max(levels.values()) if levels else 0
            layers = [[] for _ in range(maxlevel+1)]
            for n, lvl in levels.items():
                layers[lvl].append(n)
            return layers

    # 指标
    def compute_metrics(self, compile_time_sec: float):
        """
        计算：
         - makespan（近似）：max occupied_until
         - SWAP估计：从时间线估计的交换数量（不精确）
         - 重用率：被重用的物理节点比例（有重用事件）
         - 平均重用等待时间：对于被重用的重用事件，平均（reused_at - released_at）
         - 深度：近似电路深度（层数）
        """
        makespan = max(self.occupied_until.values()
                       ) if self.occupied_until else 0.0
        # 近似SWAP总时间作为非预先存在的相邻性2Q间隙的总和
        swap_time_total = 0.0
        for op in self.timeline_ops:
            if op["type"] == "2Q":
                p1, p2 = op["p"]
                if not self.topo.graph.has_edge(p1, p2):
                    # 如果在调度时非相邻 -> 我们估计了交换，包括估计
                    est, sc = self.space_time_swap_estimate(
                        p1, p2, op["start"])
                    swap_time_total += est
        # 重用指标：计算有多少重用事件后来设置了reused_at
        reused_events = [e for e in self.reuse_events if e.get(
            "reused_at") is not None]
        reuse_rate = len(reused_events) / max(1, len(self.reuse_events))
        avg_reuse_wait = np.mean([e["reused_at"] - e["released_at"]
                                 for e in reused_events]) if reused_events else None
        approx_depth = len(self.dag_to_layers(
            circuit_to_dag(self.quantum_circuit)))
        metrics = {
            "compile_time_sec": compile_time_sec,
            "makespan_ns": makespan,
            "estimated_swap_time_ns": swap_time_total,
            "reuse_rate": reuse_rate,
            "avg_reuse_wait_ns": avg_reuse_wait,
            "approx_depth": approx_depth,
            "num_physical_nodes": len(self.topo.nodes()),
        }
        return metrics
