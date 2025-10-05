import time
import math
import copy
from typing import Dict, Tuple, List, Optional, Any

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

# Optional: CP-SAT
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

from hardware import HardwareParams
from quantum_chip import QuantumChip


class TimeAwareCompiler:
    """
    编译器骨架：
      1) initial mapping (heuristic based on interaction graph)
      2) windowed scheduling + greedy mapping + reuse tracking
      3) swap cost estimates via a simple space-time approximation (or distance)
      4) optional local CP-SAT optimize on windows

    输入:
       - circuit_mgr: CircuitManager (reads qasm or qiskit circuit)
       - topo: Topology (physical hardware graph)
       - hw: HardwareParams
       - params: dict of weights/thresholds (lambda_makespan, lambda_swap, lambda_idle, ...)
    输出:
       - scheduled sequence (a simple representation)
       - metrics
    """

    def __init__(self, circuit: QuantumCircuit, topo: QuantumChip, hw: HardwareParams, params: Optional[Dict] = None):
        self.circuit_mgr = circuit
        self.topo = topo
        self.hw = hw
        self.params = params if params is not None else {}
        # default weights
        self.lambda_makespan = self.params.get("lambda_makespan", 1.0)
        self.lambda_swap = self.params.get("lambda_swap", 1.0)
        self.lambda_idle = self.params.get("lambda_idle", 1.0)
        self.cp_sat_window_threshold = self.params.get(
            "cp_sat_window_threshold", 1e-2)
        # runtime structures
        self.mapping = {}  # logical_qubit -> physical_node
        self.reverse_mapping = {}  # physical_node -> logical or None
        # time until each physical node is blocked
        self.occupied_until = {p: 0.0 for p in self.topo.nodes()}
        # book-keeping for metrics
        # list of (physical_node, released_time, reused_time) for computing wait time
        self.reuse_events = []
        self.timeline_ops = []  # scheduled operations: dicts with timing & type

    ####################
    # Helper methods
    ####################
    def build_interaction_graph(self, dag=None):
        """
        Build interaction graph of logical qubits from circuit (frequency of 2Q ops).
        Return: networkx Graph with logical qubit nodes and weight on edges = # two-qubit ops.
        """
        if dag is None:
            dag = self.circuit_mgr.get_dag()
        G = nx.Graph()
        num_q = self.circuit_mgr.num_qubits()
        for i in range(num_q):
            G.add_node(i)
        for node in dag.op_nodes():  # qiskit DAGNode
            if node.op.num_qubits > 1:
                qargs = [self.circuit_mgr.qc.find_bit(
                    q).index for q in node.qargs]
                if len(qargs) == 2:
                    u, v = qargs
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
                else:
                    # for k-qubit gates, add clique counts
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
        A simple greedy mapping:
         - sort logical qubits by degree in interaction graph
         - fill high-degree logical qubits onto well-connected physical nodes (by degree)
        This is baseline; can replace with better mapping (e.g., SABRE / VF2 matching, etc.)
        """
        inter = self.build_interaction_graph()
        logical_sorted = sorted(inter.degree, key=lambda x: -x[1])
        physical_sorted = sorted([(n, self.topo.G.degree(n)) for n in self.topo.nodes()],
                                 key=lambda x: -x[1])
        self.mapping = {}
        self.reverse_mapping = {}
        for idx, (l, _) in enumerate(logical_sorted):
            if idx < len(physical_sorted):
                p = physical_sorted[idx][0]
                self.mapping[l] = p
                self.reverse_mapping[p] = l
            else:
                # no physical => leave unmapped for now (should not happen if we have enough nodes)
                self.mapping[l] = None
        # initialize any remaining physical as unmapped
        for p in self.topo.nodes():
            if p not in self.reverse_mapping:
                self.reverse_mapping[p] = None
        return self.mapping

    def shortest_path_distance(self, p, q):
        try:
            return nx.shortest_path_length(self.topo.G, p, q)
        except nx.NetworkXNoPath:
            return math.inf

    def estimate_swap_time_distance(self, p, q):
        """
        简化的 swap_time 估计：按图距离 * avg t_2q (或具体边的 sum).
        更精确的可替换为 space-time A*（见接口 space_time_swap_estimate）
        """
        d = self.shortest_path_distance(p, q)
        if d == math.inf:
            return math.inf
        # For swaps to bring two qubits adjacent, need ~d-1 swaps (depends on scheme).
        # approximate total swap time:
        avg_t2q = self.hw.time_2q
        # factor 3 as conservative SWAP cost (hardware dependent)
        return max(0, d - 1) * avg_t2q * 3.0

    def space_time_swap_estimate(self, p, q, t_now, depth_limit=50):
        """
        Interface / simplified space-time estimate:
        - Could run a limited A* in space-time graph to estimate earliest time two qubits can be adjacent.
        - Here we return (estimated_extra_time, swap_count_estimate)
        This is a placeholder with a simple model: swap_time = estimate_swap_time_distance
        """
        est = self.estimate_swap_time_distance(p, q)
        swap_count = max(0, self.shortest_path_distance(p, q) - 1)
        return est, swap_count

    ####################
    # Scheduling core
    ####################
    def schedule(self, strategy="windowed_greedy", window_size=5):
        """
        Main entry: schedule the circuit using specified strategy.
        Implemented strategy: 'windowed_greedy' (simple), optionally call local CP-SAT if OR-Tools available.
        Returns metrics dict and chronological timeline.
        """
        t_start = time.time()

        dag = self.circuit_mgr.get_dag()
        self.initial_mapping_by_interaction()
        # We'll do a naive topological scan by layers:
        layers = self._dag_to_layers(dag)

        # reset structures
        self.occupied_until = {p: 0.0 for p in self.topo.nodes()}
        self.mapping = self.mapping  # keep initial mapping
        self.reverse_mapping = {p: self.mapping.get(
            l) for l, p in self.mapping.items()} if self.mapping else self.reverse_mapping
        # ensure reverse mapping consistent:
        self.reverse_mapping = {p: None for p in self.topo.nodes()}
        for l, p in self.mapping.items():
            if p is not None:
                self.reverse_mapping[p] = l
        self.reuse_events = []
        self.timeline_ops = []

        current_time = 0.0
        # We'll maintain a simple "ASAP" layer-by-layer scheme; within a layer, process gates in original order.
        for layer_idx, layer_nodes in enumerate(layers):
            # determine earliest time layer can start based on predecessor finishes (very crude: compute max of occupied times)
            # In this skeleton, we'll just schedule gates sequentially in each layer while respecting occupied_until.
            for node in layer_nodes:
                opname = node.name
                qargs = [self.circuit_mgr.qc.find_bit(
                    q).index for q in node.qargs]
                # compute earliest ready time based on predecessor finish times:
                pred_finish = 0.0
                for pred in dag.predecessors(node):
                    # find scheduled finish of pred (scan timeline) -> we didn't record per-gate finishes; to keep code compact,
                    # we approximate by current_time
                    pred_finish = max(pred_finish, current_time)
                ready_time = max(current_time, pred_finish)
                # ensure operands mapped; if not, assign from free pool (simple)
                phys_args = []
                for lq in qargs:
                    if self.mapping.get(lq) is None:
                        # pick a free physical qubit with earliest available time
                        chosen = min(self.occupied_until.items(),
                                     key=lambda kv: kv[1])[0]
                        self.mapping[lq] = chosen
                        self.reverse_mapping[chosen] = lq
                    phys_args.append(self.mapping[lq])

                # Branch by opcode size
                if len(phys_args) == 1:
                    p = phys_args[0]
                    start = max(ready_time, self.occupied_until[p])
                    duration = self.hw.get_t1q(p)
                    # schedule
                    self.timeline_ops.append({"type": "1Q", "op": opname, "qargs": qargs, "p": [
                                             p], "start": start, "dur": duration})
                    self.occupied_until[p] = start + duration
                    # we allow gates to start at their start time (ASAP)
                    current_time = start
                elif len(phys_args) == 2:
                    p1, p2 = phys_args
                    # if not adjacent, estimate swap time and optionally insert swaps
                    if not self.topo.G.has_edge(p1, p2):
                        est_swap_time, swap_count = self.space_time_swap_estimate(
                            p1, p2, ready_time)
                        # choose to either insert swaps or remap one operand to a free nearby qubit
                        # simplest policy: if there's an entirely free node that makes them adjacent cheaply, remap; otherwise insert swaps
                        # find free pool nodes available now
                        free_candidates = [u for u, occ in self.occupied_until.items(
                        ) if occ <= ready_time and self.reverse_mapping[u] is None]
                        # try remapping second qubit to a candidate adjacent to p1
                        remapped = False
                        for cand in free_candidates:
                            if self.topo.G.has_edge(p1, cand):
                                # remap l2 -> cand
                                l2 = None
                                # reverse lookup logical for p2
                                l2 = self.reverse_mapping.get(p2, None)
                                if l2 is not None:
                                    # free p2 then map l2->cand
                                    self.reverse_mapping[p2] = None
                                    self.mapping[l2] = cand
                                    self.reverse_mapping[cand] = l2
                                    p2 = cand
                                    remapped = True
                                    break
                        if not remapped:
                            # Insert swap delay
                            start = max(
                                ready_time, self.occupied_until[p1], self.occupied_until[p2])
                            # we conservatively add est_swap_time
                            start_after_swap = start + est_swap_time
                            # schedule the 2Q op after swaps
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
                    # k-qubit gate: approximate as multiple 2Q + 1Q pieces or block
                    # here we do a naive model: choose common time = max occupied, sum durations (approx)
                    phys = phys_args
                    start = max([self.occupied_until[p]
                                for p in phys] + [ready_time])
                    dur = self.hw.time_2q * (len(phys)-1)  # crude
                    self.timeline_ops.append(
                        {"type": "kQ", "op": opname, "qargs": qargs, "p": phys, "start": start, "dur": dur})
                    for p in phys:
                        self.occupied_until[p] = start + dur
                    current_time = start

                # measurement handling: treat measurement + reset as blocking the physical qubit until start + meas+reset
                if node.op.name.lower().startswith("measure"):
                    # For qiskit DAG measure node, qargs refer to quantum regs; assume one target
                    p = self.mapping[qargs[0]]
                    meas_t = self.hw.get_tmeas(p) + self.hw.get_treset(p)
                    # record release/reuse event if later reused
                    release_time = self.occupied_until[p]
                    # We mark physical node blocked until release_time
                    self.timeline_ops[-1]["dur"] = meas_t
                    self.timeline_ops[-1]["p"] = [p]
                    self.occupied_until[p] = self.timeline_ops[-1]["start"] + meas_t
                    # unmap logical qubit (logical slot is free for reuse after reset)
                    l = qargs[0]
                    self.reverse_mapping[self.mapping[l]] = None
                    self.mapping[l] = None
                    # mark release (for metrics)
                    self.reuse_events.append(
                        {"p": p, "released_at": self.occupied_until[p], "reused_at": None})

            # after processing a layer, optionally run local CP-SAT for the previous few layers (window)
            # Placeholder: if OR-Tools available and threshold met, run local optimize
            # (left as extension)
            # current_time = max(current_time, max(self.occupied_until.values()))

        elapsed = time.time() - t_start
        metrics = self.compute_metrics(elapsed)
        return {"timeline": self.timeline_ops, "metrics": metrics}

    def _dag_to_layers(self, dag):
        """
        Convert DAG into list-of-layers (list of lists of nodes) similar to levelization.
        A simple BFS-like levelization: nodes at same depth w.r.t dependencies.
        """
        indeg = {n: len(list(dag.predecessors(n)))
                 for n in dag.topological_op_nodes()}
        # careful: qiskit DAG node handling; we'll use dag.layers() if available
        try:
            layers = []
            for layer in dag.layers():
                # each layer has 'ops' as nodes
                layers.append([nd for nd in layer['graph'].op_nodes()])
            return layers
        except Exception:
            # fallback: compute levels
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

    ####################
    # Metrics
    ####################
    def compute_metrics(self, compile_time_sec: float):
        """
        Compute:
         - makespan (approx): max occupied_until
         - SWAP estimate: how many swaps estimated from timeline (not exact)
         - reuse rate: fraction of physical nodes that were reused (had a reuse event)
         - average reuse wait time: average (reused_at - released_at) for reuse events that got reused
         - depth: approximate circuit depth (# layers)
        """
        makespan = max(self.occupied_until.values()
                       ) if self.occupied_until else 0.0
        # approximate SWAP total time as sum of 2Q gaps where adjacency was not preexisting
        swap_time_total = 0.0
        for op in self.timeline_ops:
            if op["type"] == "2Q":
                p1, p2 = op["p"]
                if not self.topo.G.has_edge(p1, p2):
                    # if non-adjacent at scheduling time -> we estimated swaps, include estimate
                    est, sc = self.space_time_swap_estimate(
                        p1, p2, op["start"])
                    swap_time_total += est
        # reuse metrics: compute how many reuse_events later had reused_at set
        reused_events = [e for e in self.reuse_events if e.get(
            "reused_at") is not None]
        reuse_rate = len(reused_events) / max(1, len(self.reuse_events))
        avg_reuse_wait = np.mean([e["reused_at"] - e["released_at"]
                                 for e in reused_events]) if reused_events else None
        approx_depth = len(self._dag_to_layers(self.circuit_mgr.get_dag()))
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

    ####################
    # Optional: Local CP-SAT window (placeholder)
    ####################
    def run_local_cp_sat(self, window_nodes):
        """
        If OR-Tools available, model a local window as CP-SAT to optimize mapping decisions exactly.
        This is a placeholder showing how to build a small model; complete modeling must
        implement variables/constraints discussed in earlier design.

        Return: best local action (to be integrated).
        """
        if not ORTOOLS_AVAILABLE:
            raise RuntimeError(
                "OR-Tools not available; install ortools to use CP-SAT local window solver")
        model = cp_model.CpModel()
        # Example: for a small set of logical qubits and physical candidate nodes, build assignment variables
        # ... (the full model may mimic parts of the ILP earlier)
        # For framework, we return None (user fills in with specific window modeling).
        return None
