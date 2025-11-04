from typing import Dict, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from hardware import HardwareParams
from quantum_chip import QuantumChip
import pandas as pd
import networkx as nx
from copy import deepcopy
import math
from qrmove_ds import QRMoveMatrix


class QRMoveCompiler:
    def __init__(
        self,
        quantum_circuit: QuantumCircuit,
        quantum_chip: QuantumChip,
        hardware_params: HardwareParams = None,
    ):
        """
        quantum_circuit: 待优化的量子电路，类型为 QuantumCircuit
        quantum_chip: 量子芯片，包括芯片形状，芯片大小
        hardware_params: 硬件参数
        """
        self.quantum_circuit: QuantumCircuit = quantum_circuit
        self.quantum_chip: QuantumChip = quantum_chip
        self.hardware_params: HardwareParams = hardware_params
        self.circuit_matrix: QRMoveMatrix = QRMoveMatrix(
            quantum_circuit, quantum_chip, hardware_params
        )
        self.compile_program()

    def compile_program(self):
        # 编译程序，三阶段优化
        self.pull_to_min_width()  # Stage 1：拆分并组合电路、拉取以最小化电路宽度
        self.eliminate_idle_period()  # Stage 2：消除气泡
        self.compress_depth_with_extra_qubit()  # Stage 3：通过额外的量子比特，来进行深度压缩

    def compress_depth_with_extra_qubit(self):
        # Stage 3：通过额外的量子比特，来进行深度压缩
        pass

    def eliminate_idle_period(self):
        # Stage 2：消除气泡
        pass

    def pull_to_min_width(self):
        # Stage 1：拆分并组合电路、拉取以最小化电路宽度，将电路的宽度拉到极致

        # 抽取矩阵；收缩，输出优化后的矩阵
        # 1. 每一行的相同数字，进行连线（黑线）
        # 2. 每一个纵列的所有数字，进行连线（红线）
        # 3. 黑线可以收缩/扩展，但是红线必须协同左右移动
        # 4. 任意两条红线，不能交叉
        # 5. 最小化横向宽度
        # Qubit Reuse会有一个, (q_3 -> q_2)[g_2], (q_4 -> q_2)[g_1]
        # 红线：[g_0, g_1, q_4, q_4], 假设移动q_4到q_2, 则表示为 [g_0, g_1, q_2, q_4]（原始的在q_4上）
        # another：[g_0, g_4, q_0, q_0]，移动到q_3，则表示为 [g_0, g_4, q_3, q_0]
        object_matrix = self.circuit_matrix.matrix

        def get_pivot_idx():
            # 选取最多非0数字的列idx作为pivot
            non_zero_counts = np.array(
                [
                    sum(
                        1
                        for row in range(object_matrix.shape[0])
                        if object_matrix[row, col].gate_id != 0
                    )
                    for col in range(object_matrix.shape[1])
                ]
            )
            return np.argmax(non_zero_counts)

        pivot = get_pivot_idx()
        direction = 1  # 0-向左 1-向右
        mid_column = object_matrix.shape[1] / 2

        def can_be_pulled(from_col_idx, to_col_idx, from_logic_qubit_id):
            # 判断 object_matrix 中两个列之间，是否可以拉
            for row_idx in range(object_matrix.shape[0]):
                if (
                    object_matrix[row_idx, from_col_idx].logic_qubit_id
                    != from_logic_qubit_id
                ):
                    continue
                if (
                    object_matrix[row_idx, from_col_idx].idle_status
                    + object_matrix[row_idx, to_col_idx].idle_status
                    == -2
                ):
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
                if (
                    object_matrix[row_idx, from_col_idx].logic_qubit_id
                    == from_logic_qubit_id
                ):
                    # 将元素从源列移动到目标列
                    # 更新目标列的元素属性
                    object_matrix[row_idx, to_col_idx].gate_id = object_matrix[
                        row_idx, from_col_idx
                    ].gate_id
                    object_matrix[row_idx, to_col_idx].logic_qubit_id = (
                        from_logic_qubit_id
                    )
                    object_matrix[row_idx, to_col_idx].idle_status = object_matrix[
                        row_idx, from_col_idx
                    ].idle_status

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
                        if pivot_idx == col_idx:
                            continue
                        if (
                            object_matrix[row_idx, col_idx].gate_id
                            == object_matrix[row_idx, pivot_idx].gate_id
                        ):
                            # 将 col_idx 指向的列，拉到 direction 指向的附近
                            if direction == 1:  # 向右搜索
                                for tmp_col_idx in range(
                                    pivot_idx + 1, object_matrix.shape[1]
                                ):
                                    ret = pull_it(
                                        col_idx,
                                        tmp_col_idx,
                                        object_matrix[row_idx, col_idx].logic_qubit_id,
                                    )
                                    if ret:
                                        break
                            else:  # 向左搜索
                                for tmp_col_idx in range(pivot_idx - 1, 0, -1):
                                    ret = pull_it(
                                        col_idx,
                                        tmp_col_idx,
                                        object_matrix[row_idx, col_idx].logic_qubit_id,
                                    )
                                    if ret:
                                        break
            # 再拉自己没有的Gate的
            for row_idx in range(object_matrix.shape[0]):
                if object_matrix[row_idx, pivot_idx].gate_id == 0:
                    # [row_id, pivot_idx]
                    for col_idx in range(object_matrix.shape[1]):
                        if pivot_idx == col_idx:
                            continue
                        if object_matrix[row_idx, col_idx].gate_id != 0:
                            # 将 col_idx 指向的列，拉到 direction 指向的附近
                            if direction == 1:  # 向右搜索
                                for tmp_col_idx in range(
                                    pivot_idx + 1, object_matrix.shape[1]
                                ):
                                    ret = pull_it(
                                        col_idx,
                                        tmp_col_idx,
                                        object_matrix[row_idx, col_idx].logic_qubit_id,
                                    )
                                    if ret:
                                        break
                            else:  # 向左搜索
                                for tmp_col_idx in range(pivot_idx - 1, 0, -1):
                                    ret = pull_it(
                                        col_idx,
                                        tmp_col_idx,
                                        object_matrix[row_idx, col_idx].logic_qubit_id,
                                    )
                                    if ret:
                                        break

        while True:
            direction = 1 if pivot < mid_column else 0
            gate_pulling(pivot, direction)
            tmp_pivot = get_pivot_idx()
            if pivot == tmp_pivot:
                break
            else:
                pivot = tmp_pivot
        self.circuit_matrix.get_lqubit_num()
        pass  # 完成拉取
