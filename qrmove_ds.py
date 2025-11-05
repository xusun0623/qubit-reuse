import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
import math
from hardware import HardwareParams
from quantum_chip import QuantumChip


class QRMoveDAGNode:
    # DAG的Gate节点
    def __init__(self):
        # 将原来的矩阵元素属性复制过来
        self.depth = 0
        self.gate_id: int = None
        self.logic_qid_a: int = None
        self.logic_qid_b: int = None
        # 指向下一个节点
        self.next_nodes: list[QRMoveDAGNode] = []
        self.last_nodes: list[QRMoveDAGNode] = []


class QRMoveDAGBlock:
    # DAG的CP块
    def __init__(self):
        # 当前块的起点，应该连到节点列表第一个节点
        # 节点列表的最后一个节点，应该连接到块的终点
        # 节点的列表
        self.start_depth = 0
        self.end_depth = 0
        self.column_id: int = None
        self.nodes: list[QRMoveDAGNode] = []
        # 下一个块
        self.next_blocks: list[QRMoveDAGBlock] = []
        self.last_blocks: list[QRMoveDAGBlock] = []


class QRMoveDAG:
    # DAG的CP块
    def __init__(self, matrix: np.ndarray, mrp_time):
        self.mrp_time = mrp_time
        self.matrix: np.ndarray = matrix
        self.dag_root: QRMoveDAGBlock = QRMoveDAGBlock()
        self.dag_leaf: QRMoveDAGBlock = QRMoveDAGBlock()
        self.build_dag()

    def get_circuit_depth(self):
        # 获取当前电路的最大深度
        depth = self.dag_leaf.start_depth
        return depth

    def is_col_empty(self, col_idx):
        # 判断某个列是否为空
        for row_idx in range(self.matrix.shape[0]):
            if self.matrix[row_idx, col_idx].gate_id != 0:
                return False
        return True

    def build_dag(self):
        added_node: dict[int, QRMoveDAGNode] = {}  # 用于记录已经添加的节点
        matrix = self.matrix
        dag_root = self.dag_root

        dag_root.depth = 0

        row_num, col_num = matrix.shape
        for j in range(col_num):
            # 获取某个列
            if self.is_col_empty(j):
                continue
            block: QRMoveDAGBlock = QRMoveDAGBlock()
            block.column_id = j

            # 双向链表
            dag_root.next_blocks.append(block)
            block.last_blocks.append(dag_root)

            block.next_blocks.append(self.dag_leaf)
            self.dag_leaf.last_blocks.append(block)

            for i in range(row_num):
                # 获取对应的行
                _gate_id = matrix[i, j].gate_id
                _logic_qubit_id = matrix[i, j].logic_qubit_id
                if _gate_id != 0:
                    # 块的深度
                    if i - 1 < 0 or matrix[i - 1, j].gate_id == 0:
                        block.start_depth = i + 1
                    if i + 1 >= row_num or matrix[i + 1, j].gate_id == 0:
                        block.end_depth = i + 1
                    if _gate_id not in added_node:
                        # 没有添加过
                        _node = QRMoveDAGNode()
                        _node.depth = i + 1  # 赋予节点深度
                        _node.gate_id = _gate_id
                        _node.logic_qid_a = _logic_qubit_id
                        block.nodes.append(_node)
                        added_node[_gate_id] = _node
                    else:
                        # 添加过
                        _node = added_node[_gate_id]
                        _node.logic_qid_b = _logic_qubit_id
                        block.nodes.append(_node)

            nodes = block.nodes
            for node_idx in range(len(nodes) - 1):
                # 获取两个节点
                node_a, node_b = nodes[node_idx], nodes[node_idx + 1]
                node_a.next_nodes.append(node_b)
                node_b.last_nodes.append(node_a)
        leaf_last_blocks = self.dag_leaf.last_blocks
        max_depth = max([block.end_depth for block in leaf_last_blocks])
        self.dag_leaf.start_depth = max_depth + self.mrp_time
        self.dag_leaf.end_depth = max_depth + self.mrp_time


class QRMoveMatrixElement:
    def __init__(
        self,
        gate_id: int,
        logic_qubit_id: int = 0,
        idle_status: int = 0,
        is_mrp: bool = False,
    ):
        self.gate_id = gate_id  # 所属量子CNOT门的ID
        self.logic_qubit_id = logic_qubit_id  # 逻辑量子比特的ID
        self.idle_status = idle_status  # 0-可用 -1-占用
        self.is_mrp = is_mrp  # 是否为 测量-重置 阶段


class QRMoveMatrix:
    # 传入：量子电路、硬件参数
    # 矩阵形式的量子电路
    def __init__(
        self,
        circuit: QuantumCircuit,
        quantum_chip: QuantumChip,
        hardware_params: HardwareParams,
    ):
        self.quantum_circuit: QuantumCircuit = circuit
        self.quantum_chip: QuantumChip = quantum_chip
        self.hardware_params = hardware_params
        self.matrix: np.ndarray = None
        self.extract_matrix()
        self.construct_dag()
        pass

    def try_pull_block(self, from_col_idx, logic_qid, to_col_idx):
        # from_col_idx: 源列索引，to_col_idx: 目标列索引，logic_qid: 逻辑量子比特ID
        # 需要多次尝试，直到拉到最近的量子比特为止

        pass

    def get_column_gate_ids(self, col_idx):
        # 获取某个列的CNOT门ID列表
        ids = []
        row_num = self.matrix.shape[0]
        for i in range(row_num):
            _gate_id = self.matrix[i, col_idx].gate_id
            if _gate_id != 0:
                ids.append(_gate_id)
        return ids

    def get_circuit_depth(self):
        # 获取当前电路的最大深度
        row_num, col_num = self.matrix.shape
        max_depth = 0
        for j in range(col_num):
            for i in range(row_num):
                if self.matrix[i, j].gate_id != 0:
                    if i + 1 > max_depth:
                        max_depth = i + 1
        return max_depth

    def get_pivot_idx(self) -> int:
        # 获取矩阵的枢轴
        matrix = self.matrix
        x, y = matrix.shape
        pivot_idx = -1
        pivot_gate_sum = 0
        for j in range(y):
            count = 0
            for i in range(x):
                if matrix[i, j].gate_id != 0:
                    count += 1
            if count > pivot_gate_sum:
                pivot_gate_sum = count
                pivot_idx = j
        # 返回枢轴的idx
        return pivot_idx

    def construct_dag(self):
        # 将现有的矩阵表示转化为双重DAG表示，方便计算
        hp = self.hardware_params
        # 计算「测量-重置时间」和「双比特门」时间的比值
        mrp_time = math.ceil((hp.time_meas + hp.time_reset) / hp.time_2q)
        self.circuit_dag: QRMoveDAG = QRMoveDAG(self.matrix, mrp_time)

    def get_lqubit_num(self, circuit_matrix=None):
        """获取逻辑比特的数量"""
        object_matrix = self.matrix if circuit_matrix is None else circuit_matrix
        count = 0
        for col_idx in range(object_matrix.shape[1]):
            col_sum = 0
            for row_idx in range(object_matrix.shape[0]):
                col_sum += object_matrix[row_idx, col_idx].gate_id
            if col_sum != 0:
                count += 1
        print("logical qubit num", count)
        return count

    def export_matrix_to_csv(
        self, object_matrix=None, base_filename="./output/qubit_matrix"
    ):
        """
        导出三个矩阵到CSV文件，分别包含gate_id、logic_qubit_id和idle_status

        参数:
        object_matrix: 包含QRMoveMatrixElement对象的numpy数组
        base_filename: 基础文件名路径，默认为"./output/qubit_matrix"
        """
        if object_matrix is None or object_matrix.size == 0:
            object_matrix = self.matrix
            return

        # 提取三个属性矩阵
        gate_id_matrix = np.zeros(object_matrix.shape, dtype=int)
        logic_qubit_id_matrix = np.zeros(object_matrix.shape, dtype=int)
        idle_status_matrix = np.zeros(object_matrix.shape, dtype=int)

        # 填充三个矩阵
        for i in range(object_matrix.shape[0]):
            for j in range(object_matrix.shape[1]):
                gate_id_matrix[i, j] = object_matrix[i, j].gate_id
                logic_qubit_id_matrix[i, j] = object_matrix[i, j].logic_qubit_id
                idle_status_matrix[i, j] = object_matrix[i, j].idle_status

        # 导出 gate_id 矩阵
        gate_df = pd.DataFrame(gate_id_matrix)
        gate_df.to_csv(f"{base_filename}_gate_id.csv", index=False, header=False)

        # 导出 logic_qubit_id 矩阵
        logic_qubit_df = pd.DataFrame(logic_qubit_id_matrix)
        logic_qubit_df.to_csv(
            f"{base_filename}_logic_qubit_id.csv", index=False, header=False
        )

        # 导出 idle_status 矩阵
        idle_status_df = pd.DataFrame(idle_status_matrix)
        idle_status_df.to_csv(
            f"{base_filename}_idle_status.csv", index=False, header=False
        )

    def get_mrp_matrix(self, circuit_matrix=None) -> np.ndarray:
        """获取测量-重置的状态矩阵"""
        object_matrix = self.matrix if circuit_matrix is None else circuit_matrix
        tmp_object_matrix = np.empty(object_matrix.shape)
        for i in range(object_matrix.shape[0]):
            for j in range(object_matrix.shape[1]):
                tmp_object_matrix[i, j] = object_matrix[i, j].is_mrp
        return tmp_object_matrix

    def get_idle_status_matrix(self, circuit_matrix=None) -> np.ndarray:
        """获取比特状态的矩阵"""
        object_matrix = self.matrix if circuit_matrix is None else circuit_matrix
        tmp_object_matrix = np.empty(object_matrix.shape)
        for i in range(object_matrix.shape[0]):
            for j in range(object_matrix.shape[1]):
                tmp_object_matrix[i, j] = object_matrix[i, j].idle_status
        return tmp_object_matrix

    def get_logical_qubit_id_matrix(self, circuit_matrix=None) -> np.ndarray:
        """获取逻辑ID的矩阵"""
        object_matrix = self.matrix if circuit_matrix is None else circuit_matrix
        tmp_object_matrix = np.empty(object_matrix.shape)
        for i in range(object_matrix.shape[0]):
            for j in range(object_matrix.shape[1]):
                tmp_object_matrix[i, j] = object_matrix[i, j].logic_qubit_id
        return tmp_object_matrix

    def get_gate_id_matrix(self, circuit_matrix=None) -> np.ndarray:
        """获取门ID的矩阵"""
        object_matrix = self.matrix if circuit_matrix is None else circuit_matrix
        tmp_object_matrix = np.empty(object_matrix.shape)
        for i in range(object_matrix.shape[0]):
            for j in range(object_matrix.shape[1]):
                tmp_object_matrix[i, j] = object_matrix[i, j].gate_id
        return tmp_object_matrix

    def extract_matrix(self) -> np.ndarray:
        # 抽取矩阵，向矩阵中增加MRP Phase
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

        hp = self.hardware_params
        # 计算「测量-重置时间」和「双比特门」时间的比值
        mrp_2q_ratio = math.ceil((hp.time_meas + hp.time_reset) / hp.time_2q)

        object_matrix = np.empty(np_mat.shape, dtype=object)
        # 为每个元素创建对象
        for i in range(np_mat.shape[0]):
            for j in range(np_mat.shape[1]):
                object_matrix[i, j] = QRMoveMatrixElement(int(np_mat[i, j]))
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
                        object_matrix[row_idx, col_idx].logic_qubit_id = col_idx
                        object_matrix[row_idx, col_idx].idle_status = -1  # 不可用状态
                    else:
                        # 区间外的元素
                        object_matrix[row_idx, col_idx].logic_qubit_id = -1
                        object_matrix[row_idx, col_idx].idle_status = 0  # 可用状态
            else:
                # 如果当前列全为零元素，则所有元素都标记为区间外
                for row_idx in range(object_matrix.shape[0]):
                    object_matrix[row_idx, col_idx].logic_qubit_id = -1
                    object_matrix[row_idx, col_idx].idle_status = 0

        rows, cols = object_matrix.shape
        for _ in range(1000):
            new_row = np.array(
                [[QRMoveMatrixElement(int(0), int(-1), int(0)) for i in range(cols)]]
            )
            object_matrix = np.vstack((object_matrix, new_row))

        # 对于处于MRP阶段的元素：
        #    门ID           gate_id         置为0
        #    逻辑比特ID      logic_qubit_id  置为列索引
        #    空闲状态        idle_status     置为-1
        #    是否为MRP阶段   is_mrp          置为True
        for j in range(cols):
            find_idle_status = False
            inserted_row = 0
            for i in range(rows + 1000):
                if object_matrix[i, j].idle_status == 0 and (not find_idle_status):
                    continue
                if object_matrix[i, j].idle_status == -1:
                    find_idle_status = True
                    continue
                if find_idle_status and inserted_row < mrp_2q_ratio:
                    object_matrix[i, j].idle_status = -1
                    object_matrix[i, j].logic_qubit_id = j
                    object_matrix[i, j].is_mrp = True
                    inserted_row += 1
        self.matrix = object_matrix
