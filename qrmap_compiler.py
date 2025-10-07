from typing import Dict, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from hardware import HardwareParams
from quantum_chip import QuantumChip
import pandas as pd


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

    # 调度，输入优化后的矩阵
    def schedule(self):
        # 抽取的矩阵
        # 1. 每一行的相同数字，进行连线（黑线）
        # 2. 每一个纵列的所有数字，进行连线（红线）
        # 3. 黑线可以收缩/扩展，但是红线必须协同左右移动
        # 4. 任意两条红线，不能交叉
        # 5. 最小化横向宽度
        # Qubit Reuse会有一个, (q_3 -> q_2)[g_2], (q_4 -> q_2)[g_1]
        # 红线：[g_0, g_1, q_4, q_4], 假设移动q_4到q_2, 则表示为 [g_0, g_1, q_2, q_4]（原始的在q_4上）
        # another：[g_0, g_4, q_0, q_0]，移动到q_3，则表示为 [g_0, g_4, q_3, q_0]
        matrix = self.extract_matrix()

        class QRMapMatrixElement:
            def __init__(self, gate_id):
                self.gate_id = gate_id
                self.logic_qubit_id = 0
                self.idle_status = 0  # 0-可用 -1-占用
        object_matrix = np.empty(matrix.shape, dtype=object)
        # 为每个元素创建对象
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                object_matrix[i, j] = QRMapMatrixElement(matrix[i, j])
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
                        object_matrix[row_idx,
                                      col_idx].logic_qubit_id = col_idx
                        object_matrix[row_idx,
                                      col_idx].idle_status = -1
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

        non_zero_counts = np.count_nonzero(matrix, axis=0)
        pivot = np.argmax(non_zero_counts)  # 选取最多非0数字的列idx作为pivot
        direction = 1  # 0-向左 1-向右
        mid_column = len(matrix[0]) / 2
        while True:
            direction = 1 if pivot < mid_column else 0

    def export_matrix_to_csv(self, mat, filename="./output/qubit_matrix.csv"):
        # 导出矩阵到CSV文件，用于可视化和调试
        df = pd.DataFrame(mat)
        df.to_csv(filename, index=False, header=False)

    def extract_matrix(self) -> np.ndarray:
        # 抽取的矩阵表示
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
        self.export_matrix_to_csv(np_mat)
        return np_mat
