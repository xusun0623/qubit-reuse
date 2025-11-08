import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
import math
from hardware import HardwareParams
from quantum_chip import QuantumChip
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


class QRMoveDAGNode:
    # DAG的Gate节点
    def __init__(self):
        # 将原来的矩阵元素属性复制过来
        self.depth = 0
        self.gate_id: int = None
        self.logic_qid_a: int = None
        self.belong_block_a: QRMoveDAGBlock = None
        self.logic_qid_b: int = None
        self.belong_block_b: QRMoveDAGBlock = None
        # 指向下一个节点
        self.next_nodes: list[QRMoveDAGNode] = []
        self.last_nodes: list[QRMoveDAGNode] = []
        # 下一个节点的最小间隙，一般用于表示两个双比特中间夹的单比特门数量
        self.next_node_interval = 0


class QRMoveDAGBlock:
    # DAG的CP块
    def __init__(self):
        # 当前块的起点，应该连到节点列表第一个节点
        # 节点列表的最后一个节点，应该连接到块的终点
        # 节点的列表
        self.start_depth = None
        self.end_depth = None
        self.column_id: int = None
        self.logic_qid: int = None
        self.nodes: list[QRMoveDAGNode] = []
        self.tag = ""
        # 下一个块
        self.next_blocks: list[QRMoveDAGBlock] = []
        self.last_blocks: list[QRMoveDAGBlock] = []


class QRMoveDAG:
    # DAG的CP块
    def __init__(self, matrix: np.ndarray, mrp_time):
        self.mrp_time = mrp_time
        self.matrix: np.ndarray = matrix
        self.dag_root: QRMoveDAGBlock = QRMoveDAGBlock()
        self.dag_root.tag = "root"
        self.dag_leaf: QRMoveDAGBlock = QRMoveDAGBlock()
        self.dag_leaf.tag = "leaf"
        self.matrix_column: list[QRMoveDAGBlock] = []
        self.build_dag()

    def visualize_dag(self):
        G = nx.DiGraph()  # 创建一个有向无环图(DAG)
        queue = deque([self.dag_root])  # 使用队列来进行bfs
        visited = set()
        node_labels = {}  # 存储节点标签
        node_positions = {}  # 存储节点位置信息

        while queue:
            # ⭐️ 取出一个Block
            current_block = queue.popleft()
            if current_block in visited:
                continue
            visited.add(current_block)

            # 设置节点标签，包含start_depth和end_depth信息
            if current_block.tag == "root":
                node_id = "root"
                node_labels[node_id] = (
                    f"root\n[{current_block.start_depth},{current_block.end_depth}]"
                )
            elif current_block.tag == "leaf":
                node_id = "leaf"
                node_labels[node_id] = (
                    f"leaf\n[{current_block.start_depth},{current_block.end_depth}]"
                )
            else:
                node_id = current_block.logic_qid
                # 添加start_depth和end_depth信息到节点标签
                node_labels[node_id] = (
                    f"{current_block.logic_qid}\n[{current_block.start_depth},{current_block.end_depth}]"
                )

            for next_block in current_block.next_blocks:
                c_qid = current_block.logic_qid
                n_qid = next_block.logic_qid
                G.add_edge(
                    c_qid if c_qid != None else "root",
                    n_qid if n_qid != None else "leaf",
                )
                queue.append(next_block)

        plt.figure(figsize=(10, 12))
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=1200,  # 增大节点大小以容纳更多信息
            node_color="lightblue",
            edgecolors="black",
            linewidths=1,
        )
        nx.draw_networkx_edges(
            G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=1.5
        )
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, font_size=10, font_weight="bold"
        )
        plt.title("Circuit DAG", fontsize=16, pad=20)
        plt.axis("off")  # 关闭坐标轴
        plt.tight_layout()
        plt.show()
        pass

    def get_block_by_lqid(self, logic_qid, col_idx: int = None) -> QRMoveDAGBlock:
        # 应该非常鲁棒才对，不应该有拉取的块不在、列是空的，这种逻辑错误
        matrix_column = self.matrix_column
        block_pointer: QRMoveDAGBlock = matrix_column[col_idx]
        while logic_qid != block_pointer.logic_qid:
            block_pointer = block_pointer.next_blocks[0]
        if block_pointer.logic_qid != logic_qid:
            print("查找失败，需要严密地检查代码逻辑")
            return False
        # 定位要拉取的目标块
        return block_pointer

    def get_blocks_by_column_id(self, column_id) -> list[QRMoveDAGBlock]:
        # 根据Column_id获取所有块
        blocks = []
        matrix_column = self.matrix_column
        block_pointer: QRMoveDAGBlock = matrix_column[column_id]
        if len(block_pointer.next_blocks) == 0:
            # print("所在的列没有块")
            return []
        while (
            len(block_pointer.next_blocks) != 0
            and block_pointer.next_blocks[0].logic_qid != None
        ):
            block_pointer = block_pointer.next_blocks[0]
            blocks.append(block_pointer)
        return blocks

    def can_be_pulled(self, from_col_idx, logic_qid, to_col_idx, src_block):
        """源列idx, 源块qid, 目标列idx, 源块"""
        # 是否可以将一个块拉到目标列
        to_col_gate_ids = []  # 目标列的所有 gate_id

        to_column_blocks: list[QRMoveDAGBlock] = self.get_blocks_by_column_id(
            to_col_idx
        )
        for block in to_column_blocks:
            for node in block.nodes:
                # 获取目标列的所有 gate_id
                to_col_gate_ids.append(node.gate_id)

        for node in src_block.nodes:
            node_gate_id = node.gate_id
            # 如果有门ID在目标列中，则返回False
            if node_gate_id in to_col_gate_ids:
                return False

        return True

    def try_pull_block(self, from_col_idx, logic_qid, to_col_idx, src_block):
        # 定位要拉取的目标块
        # src_block = self.get_block_by_lqid(logic_qid, from_col_idx)
        # if src_block == False:
        #     return False

        def near_col(col_idx):
            # 获取 col_idx 列的附近的列
            col_num = self.matrix.shape[1]
            near_col_idx = []
            near_col_idx.append(col_idx)
            for i in range(1, col_num):
                if col_idx - i >= 0:
                    near_col_idx.append(col_idx - i)
                if col_idx + i < col_num:
                    near_col_idx.append(col_idx + i)
            return near_col_idx

        pulled = False
        actual_pull_col = None  # 实际拉到的列
        for i in near_col(to_col_idx):
            # 拉远了可不行
            if abs(i - to_col_idx) > abs(to_col_idx - from_col_idx):
                continue
            if i == from_col_idx:
                continue
            can_pull = self.can_be_pulled(from_col_idx, logic_qid, i, src_block)
            if can_pull:
                pulled = True
                actual_pull_col = i
                break

        # 拉取 block 块到目标列
        if pulled:
            self.confirm_pull(from_col_idx, logic_qid, actual_pull_col, src_block)

    def confirm_pull(
        self, from_col_idx, logic_qid, to_col_idx, src_block: QRMoveDAGBlock
    ):
        """确认拉取, from_col_idx 源列index, logic_qid 源块qid, to_col_idx 目标列index, src_block源 块"""
        to_col_blocks = self.get_blocks_by_column_id(to_col_idx)
        # 实际插入的位置
        actual_insert_pos = None  # ⭐️ 拉取块所在目标列的块index
        src_block_middle_depth = (
            src_block.start_depth + src_block.end_depth
        ) / 2  # 居中的位置
        if (
            len(to_col_blocks) == 0
            or src_block_middle_depth <= to_col_blocks[0].start_depth
        ):
            actual_insert_pos = -1
        if actual_insert_pos == None:
            for i in range(0, len(to_col_blocks) - 1):
                depth_range = [
                    to_col_blocks[i].start_depth,
                    to_col_blocks[i + 1].start_depth,
                ]
                if (
                    src_block_middle_depth >= depth_range[0]
                    and src_block_middle_depth <= depth_range[1]
                ):
                    actual_insert_pos = i
        
        if actual_insert_pos == None:
            actual_insert_pos = len(to_col_blocks) - 1

        actual_pulled_pos = -1  # ⭐️ 拉取块所在源列的块index
        all_blocks_of_pulled_column = self.get_blocks_by_column_id(from_col_idx)
        for idx, item in enumerate(all_blocks_of_pulled_column):
            if item.logic_qid == logic_qid:
                actual_pulled_pos = idx
                break
        
        # Gemini： actual_pulled_pos
        # Gemini： actual_insert_pos

        if actual_pulled_pos == 0:
            if src_block.next_blocks[0].tag == "leaf":
                # 后面只有叶子节点了，需要断掉源列的头指针
                self.matrix_column[from_col_idx].next_blocks = []
            else:
                # 后面还有一个节点，把这个节点赋给头指针.next
                self.matrix_column[from_col_idx].next_blocks = [
                    src_block.next_blocks[0]
                ]

        if actual_insert_pos == -1:
            # 被拉取块的目的位置，在头指针后面一个，需要更新目标列的头指针
            self.matrix_column[to_col_idx].next_blocks = [src_block]

        # 拉取的参数：
        #   from_col_idx - 被拉取的列index
        #   to_col_idx - 拉取到的列index
        #   src_block - 被拉取的块
        #   logic_qid - 逻辑比特ID（暂时未用）
        #   actual_insert_pos - 实际插入的位置

        # 完成 Move 之后，需要更新 depth 的块
        to_update_blocks: list[QRMoveDAGBlock] = []

        # 新增一个跨越 src_block 的双向指针
        if not (
            src_block.last_blocks[0].tag == "root"
            and src_block.next_blocks[0].tag == "leaf"
        ):
            src_block.last_blocks[0].next_blocks.append(src_block.next_blocks[0])
            src_block.next_blocks[0].last_blocks.append(src_block.last_blocks[0])
        else:
            to_update_blocks.append(src_block.next_blocks[0])

        # 删除 src_block 的上下四指针
        self.remove_blocks(src_block.last_blocks[0].next_blocks, src_block)
        self.remove_blocks(src_block.last_blocks, src_block.last_blocks[0])
        self.remove_blocks(src_block.next_blocks[0].last_blocks, src_block)
        self.remove_blocks(src_block.next_blocks, src_block.next_blocks[0])

        blocks_of_to_col = [self.dag_root] + to_col_blocks + [self.dag_leaf]

        # 将 src_block 插入到目标列，更新指针
        add_start_block = blocks_of_to_col[actual_insert_pos + 1]
        add_end_block = blocks_of_to_col[actual_insert_pos + 2]

        # print("actual ", add_start_block.tag)

        # 先断开跨越指针
        self.remove_blocks(add_start_block.next_blocks, add_end_block)
        self.remove_blocks(add_end_block.last_blocks, add_start_block)

        # 连上中间指针
        add_start_block.next_blocks.append(src_block)
        src_block.last_blocks.append(add_start_block)
        src_block.next_blocks.append(add_end_block)
        add_end_block.last_blocks.append(src_block)

        to_update_blocks.append(src_block)

        for block in self.get_blocks_by_column_id(to_col_idx):
            if block.column_id != to_col_idx:
                block.column_id = to_col_idx

        # if src_block.column_id == None:
        #     print(111)

        # self.remove_blocks(self.dag_root.next_blocks, self.dag_leaf)

        # print("dag_root_next_blocks", len(self.dag_root.next_blocks))

        for i in to_update_blocks:
            self.update_depth(i)

    def update_depth(self, block: QRMoveDAGBlock):
        """更新块、块内节点及之后所级联的深度"""

        # 使用队列来进行bfs
        queue = deque([block])
        visited = set()

        while len(queue) > 0:
            # ⭐️ 取出一个Block
            current_block = queue.popleft()
            if current_block in visited:
                continue
            visited.add(current_block)
            # 取出上一个节点的最大深度
            last_block_end_depth = 0
            if current_block == None:
                pass
            for last_block in current_block.last_blocks:
                tmp_last_end_depth = (
                    last_block.end_depth if last_block.end_depth != None else 0
                )
                last_block_end_depth = max(tmp_last_end_depth, last_block_end_depth)
            if last_block_end_depth > current_block.start_depth:
                # 更新块的上界
                current_block.start_depth = last_block_end_depth
                nodes_in_block = current_block.nodes
                current_node_depth = current_block.start_depth
                for node_idx, node_item in enumerate(nodes_in_block):
                    current_node_depth += 1
                    if node_item.depth < current_node_depth:
                        other_block = (
                            node_item.belong_block_a
                            if node_item.logic_qid_b == current_block.logic_qid
                            else node_item.belong_block_b
                        )
                        if other_block == None:
                            continue
                        # 将关联的另一个block添加进队列里
                        queue.append(other_block)
                        node_item.depth = current_node_depth
                    else:
                        current_node_depth = node_item.depth
                if current_node_depth + self.mrp_time > current_block.end_depth:
                    current_block.end_depth = current_node_depth + self.mrp_time
                if current_block.end_depth > current_block.next_blocks[0].start_depth:
                    current_block.next_blocks[0].start_depth = current_block.end_depth
                    queue.append(current_block.next_blocks[0])

    def remove_blocks(
        self, total_blocks: list[QRMoveDAGBlock], remove_block: QRMoveDAGBlock
    ):
        idx = -1
        remove_success = False
        for i in range(len(total_blocks)):
            b = total_blocks[i]
            if b.logic_qid == remove_block.logic_qid and b.tag == remove_block.tag:
                idx = i
                remove_success = True
                break
        if remove_success:
            total_blocks.pop(idx)

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

        dag_root.start_depth = 0
        dag_root.end_depth = 0

        row_num, col_num = matrix.shape
        self.matrix_column = [QRMoveDAGBlock() for _ in range(col_num)]

        for j in range(col_num):
            # 获取某个列
            if self.is_col_empty(j):
                continue
            block: QRMoveDAGBlock = QRMoveDAGBlock()
            self.matrix_column[j].next_blocks.append(block)
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
                if block.logic_qid == None and _logic_qubit_id != -1:
                    block.logic_qid = _logic_qubit_id
                if _gate_id != 0:
                    if i == 0 or matrix[i - 1, j].gate_id == 0:
                        # 为块设置开始深度
                        if block.start_depth == None:
                            block.start_depth = i + 1
                    if i + 1 >= row_num or matrix[i + 1, j].gate_id == 0:
                        # 为块设置结束深度
                        block.end_depth = i + 1
                    if _gate_id not in added_node:
                        # 没有添加过
                        _node = QRMoveDAGNode()
                        _node.depth = i + 1  # 赋予节点深度
                        _node.gate_id = _gate_id
                        _node.logic_qid_a = _logic_qubit_id
                        if block.column_id == None:
                            print(111)
                        _node.belong_block_a = block
                        block.nodes.append(_node)
                        added_node[_gate_id] = _node
                    else:
                        # 添加过
                        _node = added_node[_gate_id]
                        _node.logic_qid_b = _logic_qubit_id
                        _node.belong_block_b = block
                        block.nodes.append(_node)

            block.end_depth += self.mrp_time

            nodes = block.nodes
            for node_idx in range(len(nodes) - 1):
                # 获取两个节点
                node_a, node_b = nodes[node_idx], nodes[node_idx + 1]
                node_a.next_nodes.append(node_b)
                node_b.last_nodes.append(node_a)

        leaf_last_blocks = self.dag_leaf.last_blocks
        if leaf_last_blocks == []:
            print(222)
        max_depth = max([block.end_depth for block in leaf_last_blocks])
        self.dag_leaf.start_depth = max_depth
        self.dag_leaf.end_depth = max_depth


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
        self.circuit_dag: QRMoveDAG = None
        self.extract_matrix()
        self.construct_dag()

    def restore_matrix(self):
        dag = self.circuit_dag
        pass

    def try_pull_block(self, from_col_idx, logic_qid, to_col_idx, src_block):
        """from_col_idx: 源列索引，logic_qid: 逻辑量子比特ID，to_col_idx: 目标列索引，src_block: 源块"""
        # 需要多次尝试，直到拉到最近的量子比特为止
        circuit_dag = self.circuit_dag
        return circuit_dag.try_pull_block(
            from_col_idx, logic_qid, to_col_idx, src_block
        )

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
        circuit_dag = self.circuit_dag
        col_num = len(circuit_dag.matrix_column)
        pivot_idx = -1
        pivot_gate_sum = 0
        for col_idx in range(col_num):
            count = 0
            blocks = circuit_dag.get_blocks_by_column_id(col_idx)
            for block in blocks:
                count += len(block.nodes)
            if count > pivot_gate_sum:
                pivot_gate_sum = count
                pivot_idx = col_idx
        return pivot_idx

    def visual_dag(self):
        self.circuit_dag.visualize_dag()

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
        # SINGLE_GATE_ID = 1000000

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
            if len(qidxs) >= 1:
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

        # rows, cols = object_matrix.shape
        # for _ in range(1000):
        #     new_row = np.array(
        #         [[QRMoveMatrixElement(int(0), int(-1), int(0)) for i in range(cols)]]
        #     )
        #     object_matrix = np.vstack((object_matrix, new_row))

        # 对于处于MRP阶段的元素：
        #    门ID           gate_id         置为0
        #    逻辑比特ID      logic_qubit_id  置为列索引
        #    空闲状态        idle_status     置为-1
        #    是否为MRP阶段   is_mrp          置为True
        # for j in range(cols):
        #     find_idle_status = False
        #     inserted_row = 0
        #     for i in range(rows + 1000):
        #         if object_matrix[i, j].idle_status == 0 and (not find_idle_status):
        #             continue
        #         if object_matrix[i, j].idle_status == -1:
        #             find_idle_status = True
        #             continue
        #         if find_idle_status and inserted_row < mrp_2q_ratio:
        #             object_matrix[i, j].idle_status = -1
        #             object_matrix[i, j].logic_qubit_id = j
        #             object_matrix[i, j].is_mrp = True
        #             inserted_row += 1
        self.matrix = object_matrix
        pass
