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

        matrix = self.circuit_matrix
        pivot_idx = matrix.get_pivot_idx()  # 枢轴

        def near_col(col_idx):
            # 获取 col_idx 列的相邻列
            col_num = len(matrix.circuit_dag.matrix_column)
            near_col_idx = []
            for i in range(1, col_num):
                if col_idx - i >= 0:
                    near_col_idx.append(col_idx - i)
                if col_idx + i < col_num:
                    near_col_idx.append(col_idx + i)
            return near_col_idx

        while True:
            row_num, col_num = matrix.matrix.shape
            pulled_logic_qid = []
            for i in range(row_num):
                # 先拉取带 gate_id 的列
                gid = matrix.matrix[i, pivot_idx].gate_id
                if gid != 0:
                    for j in near_col(pivot_idx):
                        ij_gid = matrix.matrix[i, j].gate_id
                        ij_logic_id = matrix.matrix[i, j].logic_qubit_id
                        if (
                            ij_gid != 0
                            and ij_logic_id != -1
                            and gid == ij_gid
                            and j != pivot_idx
                            and (ij_logic_id not in pulled_logic_qid)
                        ):
                            pulled_logic_qid.append(ij_logic_id)
                            matrix.try_pull_block(j, ij_logic_id, pivot_idx)

            # 再拉取不带 gate_id 的列
            for j in near_col(pivot_idx):
                for i in range(row_num):
                    ij_logic_id = matrix.matrix[i, j].logic_qubit_id
                    if ij_logic_id == -1:
                        continue
                    if ij_logic_id not in pulled_logic_qid:
                        pulled_logic_qid.append(ij_logic_id)
                        matrix.try_pull_block(j, ij_logic_id, pivot_idx)

            # 重新计算枢轴
            tmp_pivot = matrix.get_pivot_idx()
            if tmp_pivot == pivot_idx:
                break
