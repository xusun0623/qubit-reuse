from typing import Dict, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from hardware import HardwareParams
from quantum_chip import QuantumChip
import pandas as pd
import networkx as nx


class QRMapMatrixElement:
    def __init__(self, gate_id: int):
        self.gate_id = gate_id
        self.logic_qubit_id = 0
        self.idle_status = 0  # 0-可用 -1-占用


class QRMapCompiler:

    def __init__(self, quantum_circuit: QuantumCircuit, quantum_chip: QuantumChip,
                 hardware_params: HardwareParams = None, params: Optional[Dict] = None):
        """
        quantum_circuit: 待优化的量子电路
        quantum_chip: 量子芯片信息
        hardware_params: 硬件参数
        params: 编译器参数
        """
        self.quantum_circuit = quantum_circuit
        self.quantum_chip = quantum_chip
        self.hardware_params = hardware_params

        self.params = params or {}
        self.idling_threshold = self.params.get(
            'idling_threshold', None
        )  # 空闲时间阈值，用于判断何时可以重用量子比特
        self.use_most_qubit_rule = self.params.get(
            'use_most_qubit_rule', True
        )  # 是否使用最多量子位规则选择pivot列
        self.qr_map = None  # 原始QR-Map数据结构
        self.optimized_map = None  # 优化后的QR-Map数据结构
        self.gate_dependencies = None  # 门依赖关系
        self.qubit_reuse_count = 0  # 量子比特重用次数

    def compile_program(self):
        shrinked_matrix = self.explore_qubit_reuse()
        self.scatter_to_chip(shrinked_matrix)

    def scatter_to_chip(self, shrinked_matrix):
        # 将优化后的矩阵，投到量子芯片上
        # 输入矩阵，输出映射数据结构
        # self.export_matrix_to_csv(shrinked_matrix, "./output/shrinked_matrix")
        # 构建 shrinked_matrix 上的热力图
        _q_num = 0  # 所需要的物理比特的数量
        _q_idxs = []
        for col_idx in range(shrinked_matrix.shape[1]):
            col_sum = 0
            for row_idx in range(shrinked_matrix.shape[0]):
                col_sum += shrinked_matrix[row_idx, col_idx].gate_id
            if (col_sum != 0):
                _q_num += 1
                _q_idxs.append(col_idx)

        heat_map = np.array([[0] * _q_num for _ in range(_q_num)])

        def get_two_col_shared_bit_num(col_idx_a, col_idx_b):
            count = 0
            for row_idx in range(shrinked_matrix.shape[0]):
                gate_id_a = shrinked_matrix[row_idx, col_idx_a].gate_id
                gate_id_b = shrinked_matrix[row_idx, col_idx_b].gate_id
                if (gate_id_a != 0 and gate_id_a == gate_id_b):
                    count += 1
            return count

        for i in range(_q_num):
            for j in range(i + 1, _q_num):
                col_idx_a, col_idx_b = _q_idxs[i], _q_idxs[j]
                heat_map[i, j] = get_two_col_shared_bit_num(
                    col_idx_a, col_idx_b)
                heat_map[j, i] = heat_map[i, j]

        # 构建逻辑量子比特交互图
        interaction_graph = nx.Graph()
        for i in range(_q_num):
            interaction_graph.add_node(i)

        for i in range(_q_num):
            for j in range(i + 1, _q_num):
                if heat_map[i, j] > 0:
                    interaction_graph.add_edge(i, j, weight=heat_map[i, j])

        # 获取芯片拓扑结构
        chip_graph = self.quantum_chip.graph

        # 使用简单的贪心算法进行映射
        # 1. 按照逻辑量子比特在交互图中的度数排序
        logical_sorted = sorted(interaction_graph.degree, key=lambda x: -
                                x[1]) if interaction_graph.edges() else [(i, 0) for i in range(_q_num)]

        # 2. 按照物理量子比特在芯片图中的度数排序
        physical_sorted = sorted([(n, chip_graph.degree(n)) for n in chip_graph.nodes()],
                                 key=lambda x: -x[1])

        # 3. 创建映射
        self.qubit_mapping = {}  # 逻辑量子比特 -> 物理量子比特
        self.reverse_qubit_mapping = {}  # 物理量子比特 -> 逻辑量子比特

        # 映射逻辑量子比特到物理量子比特
        for idx, (logical_qubit, _) in enumerate(logical_sorted):
            if idx < len(physical_sorted):
                physical_qubit = physical_sorted[idx][0]
                self.qubit_mapping[logical_qubit] = physical_qubit
                self.reverse_qubit_mapping[physical_qubit] = logical_qubit
            else:
                # 如果物理量子比特不够，保持未映射
                self.qubit_mapping[logical_qubit] = None

        # 初始化剩余的物理量子比特为未映射
        for p in chip_graph.nodes():
            if p not in self.reverse_qubit_mapping:
                self.reverse_qubit_mapping[p] = None

        print(f"逻辑量子比特到物理量子比特的映射: {self.qubit_mapping}")

    def explore_qubit_reuse(self) -> np.ndarray:
        # 抽取矩阵；收缩，输出优化后的矩阵
        # 1. 每一行的相同数字，进行连线（黑线）
        # 2. 每一个纵列的所有数字，进行连线（红线）
        # 3. 黑线可以收缩/扩展，但是红线必须协同左右移动
        # 4. 任意两条红线，不能交叉
        # 5. 最小化横向宽度
        # Qubit Reuse会有一个, (q_3 -> q_2)[g_2], (q_4 -> q_2)[g_1]
        # 红线：[g_0, g_1, q_4, q_4], 假设移动q_4到q_2, 则表示为 [g_0, g_1, q_2, q_4]（原始的在q_4上）
        # another：[g_0, g_4, q_0, q_0]，移动到q_3，则表示为 [g_0, g_4, q_3, q_0]
        object_matrix = self.extract_matrix()

        self.export_matrix_to_csv(object_matrix)

        init_qubit_num = self.get_not_all_zero_col_count(object_matrix)

        def get_pivot_idx():
            # 选取最多非0数字的列idx作为pivot
            non_zero_counts = np.array([
                sum(1 for row in range(object_matrix.shape[0])
                    if object_matrix[row, col].gate_id != 0)
                for col in range(object_matrix.shape[1])
            ])
            return np.argmax(non_zero_counts)

        pivot = get_pivot_idx()
        direction = 1  # 0-向左 1-向右
        mid_column = object_matrix.shape[1] / 2

        def can_be_pulled(from_col_idx, to_col_idx, from_logic_qubit_id):
            # 判断 object_matrix 中两个列之间，是否可以拉
            for row_idx in range(object_matrix.shape[0]):
                if (object_matrix[row_idx, from_col_idx].logic_qubit_id != from_logic_qubit_id):
                    continue
                if object_matrix[row_idx, from_col_idx].idle_status + object_matrix[row_idx, to_col_idx].idle_status == -2:
                    return False
            return True

        def pull_it(from_col_idx, to_col_idx, from_logic_qubit_id):
            # 将 from_col_idx 中所有 logic_qubit_id 为 from_logic_qubit_id 的元素，移动到 to_col_idx 列

            # 首先检查是否可以拉动
            if not can_be_pulled(from_col_idx, to_col_idx, from_logic_qubit_id):
                return False

            # 执行拉动操作
            for row_idx in range(object_matrix.shape[0]):
                # 如果当前元素属于要移动的逻辑量子比特
                if object_matrix[row_idx, from_col_idx].logic_qubit_id == from_logic_qubit_id:
                    # 将元素从源列移动到目标列
                    # 更新目标列的元素属性
                    object_matrix[row_idx, to_col_idx].gate_id = object_matrix[row_idx,
                                                                               from_col_idx].gate_id
                    object_matrix[row_idx,
                                  to_col_idx].logic_qubit_id = from_logic_qubit_id
                    object_matrix[row_idx, to_col_idx].idle_status = object_matrix[row_idx,
                                                                                   from_col_idx].idle_status

                    # 清空源列的元素（设置为默认状态）
                    object_matrix[row_idx, from_col_idx].gate_id = 0
                    object_matrix[row_idx, from_col_idx].logic_qubit_id = -1
                    object_matrix[row_idx, from_col_idx].idle_status = 0

            return True

        def gate_pulling(pivot_idx, direction):
            # 从pivot所在列的gate开始拉取，将其他所有的门拉到自己附近
            # 规则：
            # 1. 每一列中logic_qubit_id的所有元素，必须在横向协同移动
            # 2. 仅可以拉到idle_status为0的目标位置上，如果一列上有任何一个不为-1 logic_qubit_id冲突了，都不能拉

            # 先拉自己有的Gate的
            for row_idx in range(object_matrix.shape[0]):
                if object_matrix[row_idx, pivot_idx].gate_id != 0:
                    # [row_id, pivot_idx]
                    for col_idx in range(object_matrix.shape[1]):
                        if (pivot_idx == col_idx):
                            continue
                        if (object_matrix[row_idx, col_idx].gate_id == object_matrix[row_idx, pivot_idx].gate_id):
                            # 将 col_idx 指向的列，拉到 direction 指向的附近
                            if direction == 1:  # 向右搜索
                                for tmp_col_idx in range(pivot_idx + 1, object_matrix.shape[1]):
                                    ret = pull_it(
                                        col_idx, tmp_col_idx, object_matrix[row_idx, col_idx].logic_qubit_id)
                                    if ret:
                                        break
                            else:  # 向左搜索
                                for tmp_col_idx in range(pivot_idx - 1, 0, -1):
                                    ret = pull_it(
                                        col_idx, tmp_col_idx, object_matrix[row_idx, col_idx].logic_qubit_id)
                                    if ret:
                                        break
            # 再拉自己没有的Gate的
            for row_idx in range(object_matrix.shape[0]):
                if object_matrix[row_idx, pivot_idx].gate_id == 0:
                    # [row_id, pivot_idx]
                    for col_idx in range(object_matrix.shape[1]):
                        if (pivot_idx == col_idx):
                            continue
                        if (object_matrix[row_idx, col_idx].gate_id != 0):
                            # 将 col_idx 指向的列，拉到 direction 指向的附近
                            if direction == 1:  # 向右搜索
                                for tmp_col_idx in range(pivot_idx + 1, object_matrix.shape[1]):
                                    ret = pull_it(
                                        col_idx, tmp_col_idx, object_matrix[row_idx, col_idx].logic_qubit_id)
                                    if ret:
                                        break
                            else:  # 向左搜索
                                for tmp_col_idx in range(pivot_idx - 1, 0, -1):
                                    ret = pull_it(
                                        col_idx, tmp_col_idx, object_matrix[row_idx, col_idx].logic_qubit_id)
                                    if ret:
                                        break

        while True:
            direction = 1 if pivot < mid_column else 0
            gate_pulling(pivot, direction)
            tmp_pivot = get_pivot_idx()
            if (pivot == tmp_pivot):
                break
            else:
                pivot = tmp_pivot

        after_qubit_num = self.get_not_all_zero_col_count(object_matrix)

        # self.export_matrix_to_csv(
        #     object_matrix, base_filename="./output/qubit_matrix_optimized")
        print(f"[{self.params['circuit_type']}, {self.params['qubit_num']}]: {init_qubit_num} → {after_qubit_num}")

        return object_matrix

    def get_not_all_zero_col_count(self, object_matrix):
        """ 计算矩阵中gate_id非零列的数量 """
        count = 0
        for col_idx in range(object_matrix.shape[1]):
            col_sum = 0
            for row_idx in range(object_matrix.shape[0]):
                col_sum += object_matrix[row_idx, col_idx].gate_id
            if (col_sum != 0):
                count += 1
        return count

    def export_matrix_to_csv(self, object_matrix, base_filename="./output/qubit_matrix"):
        """
        导出三个矩阵到CSV文件，分别包含gate_id、logic_qubit_id和idle_status

        参数:
        object_matrix: 包含QRMapMatrixElement对象的numpy数组
        base_filename: 基础文件名路径，默认为"./output/qubit_matrix"
        """
        if object_matrix is None or object_matrix.size == 0:
            return

        # 提取三个属性矩阵
        gate_id_matrix = np.zeros(object_matrix.shape, dtype=int)
        logic_qubit_id_matrix = np.zeros(object_matrix.shape, dtype=int)
        idle_status_matrix = np.zeros(object_matrix.shape, dtype=int)

        # 填充三个矩阵
        for i in range(object_matrix.shape[0]):
            for j in range(object_matrix.shape[1]):
                gate_id_matrix[i, j] = object_matrix[i, j].gate_id
                logic_qubit_id_matrix[i,
                                      j] = object_matrix[i, j].logic_qubit_id
                idle_status_matrix[i, j] = object_matrix[i, j].idle_status

        # 导出 gate_id 矩阵
        gate_df = pd.DataFrame(gate_id_matrix)
        gate_df.to_csv(f"{base_filename}_gate_id.csv",
                       index=False, header=False)

        # 导出 logic_qubit_id 矩阵
        logic_qubit_df = pd.DataFrame(logic_qubit_id_matrix)
        logic_qubit_df.to_csv(
            f"{base_filename}_logic_qubit_id.csv", index=False, header=False)

        # 导出 idle_status 矩阵
        idle_status_df = pd.DataFrame(idle_status_matrix)
        idle_status_df.to_csv(
            f"{base_filename}_idle_status.csv", index=False, header=False)

    def extract_matrix(self) -> np.ndarray:
        # 抽取矩阵
        qc = self.quantum_circuit
        n = qc.num_qubits
        if n == 0:
            return [], 0
        next_free = [0] * n
        multigate_records = []
        next_gid = 1
        instr_layers = []
        for instr, qargs, _ in qc.data:
            qidxs = [qc.qubits.index(q) for q in qargs] if qargs else []
            if not qidxs:
                instr_layers.append(0)
                continue
            layer = max(next_free[i] for i in qidxs)
            instr_layers.append(layer)
            if len(qidxs) > 1:
                gid = next_gid
                multigate_records.append((gid, layer, qidxs))
                next_gid += 1
            for i in qidxs:
                next_free[i] = layer + 1
        computed_depth = (max(instr_layers) + 1) if instr_layers else 0
        total_layers = computed_depth
        mat = [[0 for _ in range(total_layers)] for _ in range(n)]
        for gid, layer, qidxs in multigate_records:
            for i in qidxs:
                mat[i][layer] = gid

        np_mat = np.array(mat)
        np_mat = np_mat.T

        object_matrix = np.empty(np_mat.shape, dtype=object)
        # 为每个元素创建对象
        for i in range(np_mat.shape[0]):
            for j in range(np_mat.shape[1]):
                object_matrix[i, j] = QRMapMatrixElement(int(np_mat[i, j]))
        for col_idx in range(object_matrix.shape[1]):
            # 找到当前列中非零元素的最小和最大行索引
            non_zero_rows = []
            for row_idx in range(object_matrix.shape[0]):
                if object_matrix[row_idx, col_idx].gate_id != 0:
                    non_zero_rows.append(row_idx)
            if non_zero_rows:  # 如果当前列有非零元素
                min_row = min(non_zero_rows)
                max_row = max(non_zero_rows)
                for row_idx in range(object_matrix.shape[0]):
                    if min_row <= row_idx <= max_row:
                        # 区间内的元素
                        object_matrix[
                            row_idx, col_idx].logic_qubit_id = col_idx
                        object_matrix[
                            row_idx, col_idx].idle_status = -1
                    else:
                        # 区间外的元素
                        object_matrix[row_idx, col_idx].logic_qubit_id = -1
                        object_matrix[row_idx,
                                      col_idx].idle_status = 0   # 可用状态
            else:
                # 如果当前列全为零元素，则所有元素都标记为区间外
                for row_idx in range(object_matrix.shape[0]):
                    object_matrix[row_idx, col_idx].logic_qubit_id = -1
                    object_matrix[row_idx, col_idx].idle_status = 0

        # self.export_matrix_to_csv(object_matrix)

        return object_matrix
