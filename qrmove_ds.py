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
import random


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
    def __init__(self, matrix: np.ndarray, mrp_time, quantum_chip: QuantumChip):
        self.mrp_time = mrp_time
        self.matrix: np.ndarray = matrix
        self.quantum_chip: QuantumChip = quantum_chip
        self.dag_root: QRMoveDAGBlock = QRMoveDAGBlock()
        self.dag_root.tag = "root"
        self.dag_leaf: QRMoveDAGBlock = QRMoveDAGBlock()
        self.dag_leaf.tag = "leaf"
        self.matrix_column: list[QRMoveDAGBlock] = []
        self.build_dag()

    def compress_depth_with_extra_qubit(self):
        """
        通过添加额外的量子比特列，使用模拟退火算法优化电路深度
        """
        # 1. 创建一个新的列用于额外量子比特
        new_col_idx = len(self.matrix_column)
        self.matrix_column.append(QRMoveDAGBlock())  # 添加新的列头
        new_col_head = self.matrix_column[new_col_idx]
        new_col_head.column_id = new_col_idx
        
        # 2. 设置模拟退火参数
        initial_temperature = 1.0
        final_temperature = 0.01
        cooling_rate = 0.95
        max_iterations = 200
        temperature = initial_temperature
        iteration = 0
        
        # 3. 收集所有可能移动到新列的块
        candidate_blocks = []
        for col_idx in range(new_col_idx):  # 不包括新添加的列
            blocks = self.get_blocks_by_column_id(col_idx)
            for block_idx, block in enumerate(blocks):
                # 计算块的深度跨度和重要性
                depth_span = block.end_depth - block.start_depth
                gate_count = len(block.nodes)
                # 将块及其信息添加到候选列表
                candidate_blocks.append((depth_span, gate_count, col_idx, block_idx, block))
        
        # 如果没有候选块，直接返回
        if not candidate_blocks:
            return
        
        # 4. 按深度跨度排序，优先考虑跨度大的块
        candidate_blocks.sort(key=lambda x: x[0], reverse=True)
        
        # 5. 模拟退火主循环
        best_depth = self.get_circuit_depth()
        best_configuration = []  # 保存最佳配置
        
        while iteration < max_iterations and temperature > final_temperature:
            # 5.1 根据温度决定选择策略
            if temperature > 0.3:  # 高温阶段，更多随机性
                # 随机选择一些候选块
                num_to_select = max(1, min(5, len(candidate_blocks) // 10))
                selected_blocks = random.sample(candidate_blocks, num_to_select)
            else:  # 低温阶段，选择深度跨度大的块
                num_to_select = max(1, min(3, len(candidate_blocks) // 20))
                selected_blocks = candidate_blocks[:num_to_select]
            
            # 5.2 评估移动这些块到新列的成本
            current_depth = self.get_circuit_depth()
            best_move = None
            best_delta = float('inf')
            
            for depth_span, gate_count, col_idx, block_idx, block in selected_blocks:
                # 评估移动这个块到新列的成本
                before_depth = self.get_circuit_depth()
                
                # 检查可行性
                if not self.feasible_by_dag_after_pull(
                    col_idx, 
                    block.logic_qid, 
                    new_col_idx, 
                    block_idx, 
                    -1,  # 插入到新列的开始位置
                    block
                ):
                    continue
                
                # 临时移动块
                self.confirm_pull(
                    col_idx,
                    block.logic_qid,
                    new_col_idx,
                    block_idx,
                    -1,  # 插入到新列的头
                    block
                )
                
                # 计算移动后的新深度
                new_depth = self.get_circuit_depth()
                delta = new_depth - before_depth
                
                # 恢复原状，以便评估下一个移动
                self.confirm_pull(
                    new_col_idx,
                    block.logic_qid,
                    col_idx,
                    0,  # 假设在新列中是第一个块
                    block_idx - 1 if block_idx > 0 else -1,
                    block
                )
                
                # 5.3 使用模拟退火接受准则
                if delta < best_delta or (temperature > 0 and random.random() < math.exp(-delta / temperature)):
                    best_delta = delta
                    best_move = (col_idx, block.logic_qid, new_col_idx, block_idx, -1, block)
            
            # 5.4 执行最佳移动
            if best_move is not None and (best_delta < 0 or (temperature > 0.1 and random.random() < math.exp(-abs(best_delta) / temperature))):
                self.confirm_pull(*best_move)
                
                # 检查是否找到了更好的配置
                current_depth = self.get_circuit_depth()
                if current_depth < best_depth:
                    best_depth = current_depth
                    # 保存当前配置
                    best_configuration = [(block.column_id, block.logic_qid) for col in range(len(self.matrix_column)) for block in self.get_blocks_by_column_id(col)]
            
            # 5.5 降温
            temperature *= cooling_rate
            iteration += 1
        
        # 6. 评估是否添加新列确实减少了深度
        final_depth = self.get_circuit_depth()
        print(f"Original depth: {best_depth}, Final depth with extra qubit: {final_depth}")
        
        # 7. 如果没有改善，移除新列
        if final_depth >= best_depth:
            # 恢复到最佳配置
            # 这部分实现可以根据需要完善，这里简化处理
            print("No improvement with extra qubit, removing the new column")
            # 清空新列中的所有块
            blocks_in_new_col = self.get_blocks_by_column_id(new_col_idx)
            for block in blocks_in_new_col:
                # 将块移回原来的列
                original_col = next((col for col in range(new_col_idx) 
                                if any(b.logic_qid == block.logic_qid for b in self.get_blocks_by_column_id(col))), None)
                if original_col is not None:
                    self.confirm_pull(
                        new_col_idx,
                        block.logic_qid,
                        original_col,
                        self.get_block_idx_by_col_qid(new_col_idx, block.logic_qid),
                        -1,  # 简单地放回原列开头
                        block
                    )
            
            # 移除新列
            self.matrix_column.pop(new_col_idx)
        else:
            print(f"Successfully reduced depth by {best_depth - final_depth} using an extra qubit")
        
        # 8. 更新深度
        self.update_depth()
    
    
    def get_pivot_idx(self) -> int:
        # 获取矩阵的枢轴
        col_num = len(self.matrix_column)
        pivot_idx = -1
        pivot_gate_sum = 0
        for col_idx in range(col_num):
            count = 0
            blocks = self.get_blocks_by_column_id(col_idx)
            for block in blocks:
                count += len(block.nodes)
            if count > pivot_gate_sum:
                pivot_gate_sum = count
                pivot_idx = col_idx
        return pivot_idx

    def cal_swap_cost(self):
        # square, hexagon, heavy_square, heavy_hexagon
        chip_type = self.quantum_chip.chip_type
        all_gates = []
        for col_idx, col in enumerate(self.matrix_column):
            blocks = self.get_blocks_by_column_id(col_idx)
            if len(blocks) == 0:
                continue
            for block in blocks:
                for node in block.nodes:
                    if node.belong_block_b == None:
                        continue
                    two_col = [
                        node.belong_block_a.column_id,
                        node.belong_block_b.column_id,
                    ]
                    two_col.sort()
                    if two_col not in all_gates:
                        all_gates.append(two_col)
        total_cost = 0
        for gate in all_gates:
            degree_map = {
                "square": 4,
                "hexagon": 3,
                "heavy_square": 3,
                "heavy_hexagon": 2.5,
            }
            avg_degree = degree_map[chip_type]
            total_cost += math.floor((gate[0] - gate[1]) / avg_degree)
        return total_cost

    def cost_b_to_i(
        self,
        from_col_idx,  # 源列index
        logic_qid,  # 源块qid
        to_col_idx,  # 目标列index
        actual_pulled_pos,  # 源块所在源列的块index
        actual_insert_pos,  # 目标块所在目标列的块index
        src_block: QRMoveDAGBlock,
    ):
        # 将 b 移动到 i 槽的成本
        before_swap = self.cal_swap_cost()
        before_depth = self.get_circuit_depth()
        if not self.feasible_by_dag_after_pull(
            from_col_idx,
            logic_qid,
            to_col_idx,
            actual_pulled_pos,
            actual_insert_pos,
            src_block,
        ):
            return 10000
        # 移动一下
        self.confirm_pull(
            from_col_idx,
            logic_qid,
            to_col_idx,
            actual_pulled_pos,
            actual_insert_pos,
            src_block,
        )
        after_swap = self.cal_swap_cost()
        after_depth = self.get_circuit_depth()
        # 再给它移回去
        self.confirm_pull(
            to_col_idx,
            logic_qid,
            from_col_idx,
            actual_insert_pos + 1,
            actual_pulled_pos - 1,
            src_block,
        )
        SWAP_TO_CNOT = 3
        return (after_swap - before_swap) * SWAP_TO_CNOT * 2 + (
            after_depth - before_depth
        )

    def get_no_none_col(self) -> list[int]:
        # 获取非空列
        col_num = len(self.matrix_column)
        no_none_col = []
        for col_idx in range(col_num):
            blocks = self.get_blocks_by_column_id(col_idx)
            if len(blocks) > 0:
                no_none_col.append(col_idx)
        return no_none_col

    def compress_depth_with_existing_qubit(self):
        # 通过已有的量子比特，对深度进行压缩
        pivot_idx = self.get_pivot_idx()  # 先获取枢轴
        count = 300
        K = 10  # 选择被移动的块数量
        initial_temperature = 1.0
        final_temperature = 0.01
        cooling_rate = 0.995

        temperature = initial_temperature

        while count > 0 and temperature > final_temperature:
            # 计算当前温度（随迭代次数下降）
            temperature = initial_temperature * (cooling_rate ** (300 - count))

            # ⭐️ 获取候选被移动的块（根据温度决定随机程度）
            candidate_blocks = []
            pivot_blocks = self.get_blocks_by_column_id(pivot_idx)
            for block_idx, block in enumerate(pivot_blocks):
                start_depth = block.start_depth
                end_depth = block.end_depth
                gap_depth = end_depth - start_depth
                depth_range = [start_depth, end_depth]
                candidate_blocks.append((gap_depth, depth_range, block, block_idx))

            # 根据温度决定选择策略
            if temperature > 0.5:  # 高温阶段，更多随机性
                # 随机选择K个块
                if len(candidate_blocks) <= K:
                    top_k_blocks = candidate_blocks
                else:
                    top_k_blocks = random.sample(candidate_blocks, K)
            else:  # 低温阶段，选择TopK
                # 按gap_depth降序排列
                candidate_blocks.sort(key=lambda x: x[0], reverse=True)
                top_k_blocks = candidate_blocks[:K]

                # 在低温阶段，也以一定概率接受较差的解（模拟退火特性）
                if random.random() < temperature and len(candidate_blocks) > K:
                    # 以温度为概率，从剩余块中随机选择一个替换TopK中的一个
                    remaining_blocks = candidate_blocks[K:]
                    if remaining_blocks:
                        # 随机替换TopK中的一个块
                        replace_idx = random.randint(0, len(top_k_blocks) - 1)
                        replacement = random.choice(remaining_blocks)
                        top_k_blocks[replace_idx] = replacement

            if not top_k_blocks:
                count -= 1
                continue

            now_cost = 10000
            final_move = None

            # ⭐️ 获取目标位置
            for gap_depth, depth_range, target_block, target_block_idx in top_k_blocks:
                candidate_target = []
                # [[col_idx, insert_idx, depth_range], [col_idx, insert_idx, depth_range], ...]
                for col_idx in self.get_no_none_col():
                    if col_idx == pivot_idx:
                        continue
                    # 需要统计一下可以移动到的目标位置，然后初筛-复筛-用cost function最后筛
                    # 然后确认执行移动操作
                    # 已有的：
                    # from_col_idx: pivot_idx
                    # logic_qid: target_block.logic_qid
                    # to_col_idx: col_idx
                    # actual_pulled_pos: target_block_idx
                    # actual_insert_pos: 待选择
                    blocks_of_col = self.get_blocks_by_column_id(col_idx)
                    candidate_target.append(
                        (
                            col_idx,
                            -1,
                            [0, blocks_of_col[0].start_depth],
                        )
                    )
                    for block_idx, block in enumerate(blocks_of_col):
                        start_depth = None
                        end_depth = None
                        if block_idx == len(self.get_blocks_by_column_id(col_idx)) - 1:
                            start_depth = block.end_depth
                            end_depth = self.get_circuit_depth()
                            candidate_target.append(
                                (
                                    col_idx,
                                    block_idx,
                                    [start_depth, end_depth],
                                )
                            )
                            continue
                        start_depth = block.end_depth
                        end_depth = blocks_of_col[block_idx + 1].start_depth
                        candidate_target.append(
                            (
                                col_idx,
                                block_idx,
                                [start_depth, end_depth],
                            )
                        )

                # 根据温度调整目标选择策略
                final_candidate_target = self.cal_match_level(
                    (gap_depth, depth_range, target_block, target_block_idx),
                    candidate_target,
                )

                if len(final_candidate_target) > 0:
                    # 在高温阶段随机选择目标，在低温阶段选择最优目标
                    if temperature > 0.5 and len(final_candidate_target) > 1:
                        # 高温阶段，随机选择部分目标进行评估
                        sample_size = min(5, len(final_candidate_target))
                        selected_targets = random.sample(
                            final_candidate_target, sample_size
                        )
                    else:
                        # 低温阶段，评估所有候选目标
                        selected_targets = final_candidate_target

                    for _target in selected_targets:
                        tmp_cost = self.cost_b_to_i(
                            pivot_idx,
                            target_block.logic_qid,
                            _target[0],
                            target_block_idx,
                            _target[1],
                            target_block,
                        )

                        # 模拟退火接受准则
                        if tmp_cost < now_cost or (
                            temperature > 0.01
                            and random.random()
                            < math.exp(-(tmp_cost - now_cost) / max(temperature, 0.001))
                        ):
                            now_cost = tmp_cost
                            final_move = (
                                pivot_idx,
                                target_block.logic_qid,
                                _target[0],
                                target_block_idx,
                                _target[1],
                                target_block,
                            )

            # 执行移动（如果找到合适的移动）
            if final_move is not None and now_cost < 0:
                self.confirm_pull(
                    final_move[0],
                    final_move[1],
                    final_move[2],
                    final_move[3],
                    final_move[4],
                    final_move[5],
                )
            elif final_move is not None and temperature > 0.1:
                # 即使不是改进，高温时也以一定概率接受（模拟退火特性）
                acceptance_probability = math.exp(-abs(now_cost) / temperature)
                if random.random() < acceptance_probability:
                    self.confirm_pull(
                        final_move[0],
                        final_move[1],
                        final_move[2],
                        final_move[3],
                        final_move[4],
                        final_move[5],
                    )

            count -= 1

    def cal_match_level(self, src_param, des_params):
        # 判断深度匹配的权重
        # src_param: (gap_depth, depth_range, block, block_idx)
        # des_params: [(col_idx, insert_idx, depth_range), ...]
        src_depth_range = src_param[1]
        ret_target = []
        for i in des_params:
            i_depth_range = i[2]
            if i[2][1] > src_depth_range[0] or i[2][0] < src_depth_range[1]:
                ret_target.append(i)
        return ret_target

    def cal_cross_depth_weight(self, src_range, target_range):

        pass

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

    def near_col(self, col_idx):
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

    def can_be_pulled(self, to_col_idx, src_block):
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

    def get_block_idx_by_col_qid(self, col_idx, logic_qid):
        actual_pulled_pos = -1  # ⭐️ 拉取块所在源列的块index
        all_blocks_of_pulled_column = self.get_blocks_by_column_id(col_idx)
        for idx, item in enumerate(all_blocks_of_pulled_column):
            if item.logic_qid == logic_qid:
                actual_pulled_pos = idx
                break
        return actual_pulled_pos

    def can_actual_be_pulled(
        self, to_col_idx, from_col_idx, logic_qid, src_block: QRMoveDAGBlock
    ):
        # 已经第一次筛选过了col_idx
        # 是否可以将一个块实际拉到目标列
        # 需要返回拉到目标列的块index，不行就返回-1
        # 这个或许需要 DAG 辅助判断一下

        actual_insert_pos = None  # ⭐️ 拉取块所在目标列的块index
        to_col_blocks = self.get_blocks_by_column_id(to_col_idx)  # 目标列的所有块
        block_num = len(to_col_blocks)

        up_down_depth = src_block.start_depth + src_block.end_depth
        mid_depth = up_down_depth / 2  # 居中的位置
        if block_num == 0 or mid_depth <= to_col_blocks[0].start_depth:
            actual_insert_pos = -1
        if actual_insert_pos == None:
            for i in range(0, block_num - 1):
                block_start_depth = to_col_blocks[i].start_depth
                block_end_depth = to_col_blocks[i + 1].start_depth
                depth_range = [block_start_depth, block_end_depth]
                if mid_depth >= depth_range[0] and mid_depth <= depth_range[1]:
                    actual_insert_pos = i
        if actual_insert_pos == None:
            actual_insert_pos = block_num - 1

        def near_block_idx(idx):
            # 遍历块周围的idx，范围是 [-1, block_num - 1]
            a_list = [idx]
            for i in range(1, 20):
                a_list.append(idx + i)
                a_list.append(idx - i)
            # a_list = [idx - 1, idx - 2, idx, idx + 1, idx + 2]
            b_list = [-1] + [i for i in range(block_num)]
            intersection = [x for x in a_list if x in b_list]
            return intersection

        # 做第一次filter，以概率接收
        near_interval = near_block_idx(to_col_idx)

        all_deep_range = []
        all_deep_range_idx = []

        for interval_idx in near_interval:
            depth_range = [0, 0]
            if interval_idx == -1:
                if len(to_col_blocks) == 0:
                    all_deep_range.append([0, 0])
                    all_deep_range_idx.append(interval_idx)
                    continue
                depth_range = [0, to_col_blocks[0].start_depth]
                all_deep_range.append(depth_range)
                all_deep_range_idx.append(interval_idx)
                continue
            if interval_idx == block_num - 1:
                max_depth = to_col_blocks[block_num - 1].end_depth
                depth_range = [max_depth, max_depth + self.mrp_time]
                all_deep_range.append(depth_range)
                all_deep_range_idx.append(interval_idx)
                continue
            depth_range = [
                to_col_blocks[interval_idx].end_depth,
                to_col_blocks[interval_idx + 1].start_depth,
            ]
            all_deep_range.append(depth_range)
            all_deep_range_idx.append(interval_idx)

        _x = src_block.start_depth
        _y = src_block.end_depth

        weights = []

        for i in all_deep_range:
            x, y = i[0], i[1]
            weight = 0
            if y > _x:
                weight = 1 - math.exp(_x - y)
                weights.append(weight)
                continue
            if _y > x:
                weight = 1 - math.exp(x - _y)
                weights.append(weight)
                continue
            x_y_depths = [x, _x, y, _y]
            x_y_depths.sort()
            weight = 2 - math.exp(x_y_depths[1] - x_y_depths[2])
            weights.append(weight)

        arr = np.array(weights)
        sorted_indices_desc = np.argsort(arr)[::-1]
        top_indices_desc = sorted_indices_desc[:100]

        for i in top_indices_desc:
            actual_insert_pos = all_deep_range_idx[i]
            if self.feasible_by_dag_after_pull(
                from_col_idx,
                logic_qid,
                to_col_idx,
                self.get_block_idx_by_col_qid(from_col_idx, logic_qid),
                actual_insert_pos,
                src_block,
            ):
                # 最后再模拟一次插入，如果结果是Feasible的
                return actual_insert_pos

        return None

    def traverse_near(self, to_col_idx, from_col_idx, logic_qid, src_block):
        if to_col_idx != 0:
            pass
        if logic_qid == 16:
            pass
        for i in self.near_col(to_col_idx):
            # 拉远了可不行
            if abs(i - to_col_idx) > abs(to_col_idx - from_col_idx):
                continue
            if i == from_col_idx:
                continue
            # 第一次判断是否可以拉取
            can_pull_col = self.can_be_pulled(i, src_block)
            if can_pull_col:
                # 第二次判断是否可以拉取
                # 输入：列的idx
                # 输出：实际拉到的所在列的块idx
                # 不能拉取：返回-1
                pull_to_block_idx = self.can_actual_be_pulled(
                    i, from_col_idx, logic_qid, src_block
                )
                if pull_to_block_idx != None:
                    return i, pull_to_block_idx
        return None, None

    def try_pull_block(self, from_col_idx, logic_qid, to_col_idx, src_block):
        # 定位要拉取的目标块

        pull_to_col, pull_to_block_idx = self.traverse_near(
            to_col_idx, from_col_idx, logic_qid, src_block
        )
        # 遍历周围的列，先看门冲突，再看 DAG 冲突

        if pull_to_col != None:
            # 可以拉取，执行拉动操作
            pulled_pos = self.get_block_idx_by_col_qid(from_col_idx, logic_qid)
            self.confirm_pull(
                from_col_idx,
                logic_qid,
                pull_to_col,
                pulled_pos,
                pull_to_block_idx,
                src_block,
            )

    def feasible_by_dag_after_pull(
        self,
        from_col_idx,  # 源列index
        logic_qid,  # 源块qid
        to_col_idx,  # 目标列index
        actual_pulled_pos,  # 源块所在源列的块index
        actual_insert_pos,  # 目标块所在目标列的块index
        src_block: QRMoveDAGBlock,
    ):
        """模拟拉取, 返回是否feasible
        from_col_idx 源列index,
        logic_qid 源块qid,
        to_col_idx 目标列index,
        src_block 源块
        actual_insert_pos 实际插入的位置"""

        gate_dag = nx.DiGraph()

        def insert_nodes_by_block(tmpBlock: QRMoveDAGBlock, last_gate_id, cross_block):
            for node in tmpBlock.nodes:
                if last_gate_id == None:
                    last_gate_id = node.gate_id
                else:
                    gate_dag.add_edge(
                        last_gate_id,
                        node.gate_id,
                        weight=(self.mrp_time if cross_block else 1),
                    )
                    last_gate_id = node.gate_id
                    cross_block = False
            cross_block = True
            return last_gate_id, cross_block

        if logic_qid == 16:
            pass

        col_num = len(self.matrix_column)
        for col_idx in range(col_num):
            from_flag = col_idx == from_col_idx
            from_idx = actual_pulled_pos
            to_flag = col_idx == to_col_idx
            to_idx = actual_insert_pos
            blocks_of_col = self.get_blocks_by_column_id(col_idx)
            _last_gate_id = None
            _cross_block = False

            if to_flag and len(blocks_of_col) == 0:
                # 目标列为空
                _last_gate_id, _cross_block = insert_nodes_by_block(
                    src_block, _last_gate_id, _cross_block
                )
            for block_idx, block in enumerate(blocks_of_col):
                if from_flag and block_idx == from_idx:  # 略过已经移走的块
                    continue
                if to_flag and to_idx == -1 and block_idx == 0:  # 头插
                    _last_gate_id, _cross_block = insert_nodes_by_block(
                        src_block, _last_gate_id, _cross_block
                    )
                _last_gate_id, _cross_block = insert_nodes_by_block(
                    block, _last_gate_id, _cross_block
                )
                if to_flag and to_idx == block_idx:  # 尾插
                    _last_gate_id, _cross_block = insert_nodes_by_block(
                        src_block, _last_gate_id, _cross_block
                    )
        try:
            # nx.topological_sort(gate_dag) # 不报错
            topological_order = list(nx.topological_sort(gate_dag))  # 报错
        except Exception as e:
            # print(e)
            return False

        return True

    def confirm_pull(
        self,
        from_col_idx,
        logic_qid,
        to_col_idx,
        actual_pulled_pos,
        actual_insert_pos,
        src_block: QRMoveDAGBlock,
    ):
        """确认拉取
        from_col_idx 源列index
        logic_qid 源块qid
        to_col_idx 目标列index
        actual_pulled_pos 实际拉取的位置
        actual_insert_pos 实际插入的位置
        src_block 源块"""

        # 目标列的所有块
        to_col_blocks = self.get_blocks_by_column_id(to_col_idx)
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

        # 新增一个跨越 src_block 的双向指针
        if not (
            src_block.last_blocks[0].tag == "root"
            and src_block.next_blocks[0].tag == "leaf"
        ):
            src_block.last_blocks[0].next_blocks.append(src_block.next_blocks[0])
            src_block.next_blocks[0].last_blocks.append(src_block.last_blocks[0])

        # 删除 src_block 的上下四指针
        self.remove_blocks(src_block.last_blocks[0].next_blocks, src_block)
        self.remove_blocks(src_block.last_blocks, src_block.last_blocks[0])
        self.remove_blocks(src_block.next_blocks[0].last_blocks, src_block)
        self.remove_blocks(src_block.next_blocks, src_block.next_blocks[0])

        blocks_of_to_col = [self.dag_root] + to_col_blocks + [self.dag_leaf]

        # 将 src_block 插入到目标列，更新指针
        add_start_block = blocks_of_to_col[actual_insert_pos + 1]
        add_end_block = blocks_of_to_col[actual_insert_pos + 2]

        # 先断开跨越指针
        self.remove_blocks(add_start_block.next_blocks, add_end_block)
        self.remove_blocks(add_end_block.last_blocks, add_start_block)

        # 连上中间指针
        add_start_block.next_blocks.append(src_block)
        src_block.last_blocks.append(add_start_block)
        src_block.next_blocks.append(add_end_block)
        add_end_block.last_blocks.append(src_block)

        for block in self.get_blocks_by_column_id(to_col_idx):
            if block.column_id != to_col_idx:
                block.column_id = to_col_idx

        # 最后一步，更新所有块和节点的深度
        self.update_depth()

    def visual_gate_dag(self, gate_dag: nx.DiGraph):
        """可视化 gate_dag 的拓扑结构"""
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(gate_dag, k=3, iterations=50)
        nx.draw_networkx_nodes(
            gate_dag, pos, node_size=400, node_color="lightblue", alpha=0.9
        )
        nx.draw_networkx_edges(
            gate_dag, pos, arrowstyle="->", arrowsize=60, edge_color="gray"
        )
        labels = {
            node: f"{node}\n{gate_dag.nodes[node]['qid']}" for node in gate_dag.nodes()
        }
        nx.draw_networkx_labels(gate_dag, pos, labels, font_size=10)
        edge_labels = nx.get_edge_attributes(gate_dag, "weight")
        nx.draw_networkx_edge_labels(gate_dag, pos, edge_labels, font_size=8)
        plt.title("Gate Dependency DAG Topology", size=15)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def update_depth(self):
        """计算所有块、节点的深度"""
        dag_root = self.dag_root
        gate_dag = nx.DiGraph()
        for col_idx in range(len(self.matrix_column)):
            blocks_of_col = self.get_blocks_by_column_id(col_idx)
            for block_idx, block in enumerate(blocks_of_col):
                for node in block.nodes:
                    gate_dag.add_node(
                        node.gate_id, qid=f"{node.logic_qid_a}, {node.logic_qid_b}"
                    )
        for col_idx in range(len(self.matrix_column)):
            blocks_of_col = self.get_blocks_by_column_id(col_idx)
            last_gate_id = None
            cross_block = False
            for block_idx, block in enumerate(blocks_of_col):
                for node in block.nodes:
                    if last_gate_id == None:
                        last_gate_id = node.gate_id
                    else:
                        # 保留大的那个权重
                        if gate_dag.has_edge(last_gate_id, node.gate_id):
                            existing_weight = gate_dag[last_gate_id][node.gate_id][
                                "weight"
                            ]
                            new_weight = self.mrp_time if cross_block else 1
                            gate_dag[last_gate_id][node.gate_id]["weight"] = max(
                                existing_weight, new_weight
                            )
                        else:
                            gate_dag.add_edge(
                                last_gate_id,
                                node.gate_id,
                                weight=(self.mrp_time if cross_block else 1),
                            )
                        last_gate_id = node.gate_id
                        cross_block = False
                cross_block = True

        # self.visualize_dag()
        # self.visual_gate_dag(gate_dag)

        # 拓扑排序
        topological_order = list(nx.topological_sort(gate_dag))
        gate_depths = {gate_id: 0 for gate_id in topological_order}

        # 按照拓扑顺序计算depth
        for gate_id in topological_order:
            current_depth = gate_depths[gate_id]

            if gate_dag.in_degree(gate_id) == 0:
                gate_depths[gate_id] = 1
                current_depth = 1

            # 更新后续节点的depth
            for successor in gate_dag.successors(gate_id):
                edge_weight = gate_dag[gate_id][successor]["weight"]
                new_depth = current_depth + edge_weight
                if new_depth > gate_depths[successor]:
                    gate_depths[successor] = new_depth

        max_depth = 0
        for col in range(len(self.matrix_column)):
            for block in self.get_blocks_by_column_id(col):
                nodes = block.nodes
                block.start_depth = gate_depths[nodes[0].gate_id]
                block.end_depth = gate_depths[nodes[-1].gate_id] + self.mrp_time
                max_depth = max(max_depth, block.end_depth)

        self.dag_leaf.start_depth = max_depth
        self.dag_leaf.end_depth = max_depth

    def print_block_depth(self):
        for col_idx, col in enumerate(self.matrix_column):
            print(f"------------ 列{col_idx} -------------")
            for block in self.get_blocks_by_column_id(col_idx):
                print(f"{block.logic_qid}: {block.start_depth} {block.end_depth}")

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
        # self.visual_dag()

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
        return self.circuit_dag.get_circuit_depth()

    def get_pivot_idx(self) -> int:
        # 获取矩阵的枢轴
        return self.circuit_dag.get_pivot_idx()

    def visual_dag(self):
        self.circuit_dag.visualize_dag()

    def construct_dag(self):
        # 将现有的矩阵表示转化为双重DAG表示，方便计算
        hp = self.hardware_params
        # 计算「测量-重置时间」和「双比特门」时间的比值
        mrp_time = math.ceil((hp.time_meas + hp.time_reset) / hp.time_2q)
        self.circuit_dag: QRMoveDAG = QRMoveDAG(
            self.matrix, mrp_time, self.quantum_chip
        )

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

        self.matrix = object_matrix
        pass
