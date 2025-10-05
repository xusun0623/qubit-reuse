from typing import Dict, Optional, List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from hardware import HardwareParams
from quantum_chip import QuantumChip
import pandas as pd
from collections import defaultdict
import copy


class QRMapCompiler:
    """QR-Map编译器"""

    def __init__(self, quantum_circuit: QuantumCircuit, quantum_chip: QuantumChip, 
                 hardware_params: HardwareParams = None, params: Optional[Dict] = None):
        self.quantum_circuit = quantum_circuit
        self.quantum_chip = quantum_chip
        self.hardware_params = hardware_params
        self.params = params or {}
        
        # QR-Map参数
        self.idling_threshold = self.params.get('idling_threshold', None)  # 空闲时间阈值
        self.use_most_qubit_rule = self.params.get('use_most_qubit_rule', True)  # 是否使用最多量子位规则
        
        # 内部状态
        self.qr_map = None
        self.optimized_map = None
        self.gate_dependencies = None
        self.qubit_reuse_count = 0

    def export_matrix_to_csv(self, mat, filename="qubit_matrix.csv"):
        """导出矩阵到CSV文件"""
        df = pd.DataFrame(mat)
        df.to_csv(filename, index=False, header=False)

    def schedule(self):
        """主调度函数"""
        print("开始QR-Map调度...")
        
        # 1. 提取两量子比特门
        two_qubit_gates = self.extract_two_qubit_gates()
        print(f"提取到 {len(two_qubit_gates)} 个两量子比特门")
        
        # 2. 构建QR-Map
        self.build_qr_map(two_qubit_gates)
        
        # 3. 应用Tapering算法
        self.apply_tapering()
        
        # 4. 恢复量子电路
        optimized_circuit = self.restore_to_circuit()
        
        print(f"优化完成: 量子位重用次数={self.qubit_reuse_count}")
        return optimized_circuit

    def extract_two_qubit_gates(self):
        """提取两量子比特门及其相关单量子比特门"""
        two_qubit_gates = []
        current_gate_groups = defaultdict(list)
        
        for instruction in self.quantum_circuit.data:
            gate = instruction.operation
            qubits = [self.quantum_circuit.qubits.index(q) for q in instruction.qubits]
            
            if len(qubits) == 2:
                # 两量子比特门 - 创建新的门组
                gate_info = {
                    'type': 'two_qubit',
                    'gate': gate,
                    'qubits': qubits,
                    'single_qubit_gates': {},
                    'name': gate.name
                }
                
                # 添加相关的单量子比特门
                for qubit in qubits:
                    if qubit in current_gate_groups:
                        gate_info['single_qubit_gates'][qubit] = current_gate_groups[qubit].copy()
                        current_gate_groups[qubit].clear()
                
                two_qubit_gates.append(gate_info)
                
            elif len(qubits) == 1:
                # 单量子比特门 - 添加到当前门组
                qubit = qubits[0]
                current_gate_groups[qubit].append({
                    'gate': gate,
                    'qubit': qubit,
                    'name': gate.name
                })
        
        return two_qubit_gates

    def build_qr_map(self, two_qubit_gates):
        """构建QR-Map数据结构"""
        num_gates = len(two_qubit_gates)
        num_qubits = self.quantum_circuit.num_qubits
        
        # 初始化QR-Map
        self.qr_map = {
            'array': np.zeros((num_gates, num_qubits), dtype=int),
            'gate_info': [],
            'vertical_lines': np.zeros((num_gates, num_qubits), dtype=bool),
            'qubit_usage': [set() for _ in range(num_qubits)]
        }
        
        # 填充门信息
        for gate_idx, gate_info in enumerate(two_qubit_gates):
            qubits = gate_info['qubits']
            
            # 在数组中标记门位置 (使用门索引+1)
            for qubit in qubits:
                self.qr_map['array'][gate_idx, qubit] = gate_idx + 1
            
            # 记录门信息
            self.qr_map['gate_info'].append(gate_info)
            
            # 更新量子位使用情况
            for qubit in qubits:
                self.qr_map['qubit_usage'][qubit].add(gate_idx)
        
        # 构建垂直依赖线
        self._build_vertical_lines()
        
        print("QR-Map构建完成")
        self.export_matrix_to_csv(self.qr_map['array'], "initial_qr_map.csv")

    def _build_vertical_lines(self):
        """构建垂直依赖线"""
        num_gates, num_qubits = self.qr_map['array'].shape
        
        for qubit in range(num_qubits):
            used_gates = sorted(self.qr_map['qubit_usage'][qubit])
            if not used_gates:
                continue
                
            # 从第一个使用到最后一个使用之间都是活跃的
            first_use = used_gates[0]
            last_use = used_gates[-1]
            
            for gate_idx in range(first_use, last_use + 1):
                self.qr_map['vertical_lines'][gate_idx, qubit] = True

    def apply_tapering(self):
        """应用Tapering算法"""
        if self.qr_map is None:
            raise ValueError("QR-Map未构建")
            
        print("开始Tapering算法...")
        
        # 创建工作副本
        working_map = copy.deepcopy(self.qr_map)
        iteration = 0
        
        while True:
            iteration += 1
            print(f"迭代 {iteration}")
            
            # 选择pivot和方向
            pivot = self._select_pivot(working_map)
            direction = self._select_direction(pivot, working_map['array'].shape[1])
            
            print(f"Pivot: {pivot}, Direction: {direction}")
            
            # 门拉动阶段
            gates_moved = self._gate_pulling(working_map, pivot, direction)
            
            # 检查收敛
            new_pivot = self._select_pivot(working_map)
            if new_pivot == pivot and not gates_moved:
                break
                
            pivot = new_pivot
            
            if iteration > 100:  # 防止无限循环
                print("达到最大迭代次数")
                break
        
        self.optimized_map = working_map
        self.export_matrix_to_csv(self.optimized_map['array'], "optimized_qr_map.csv")
        print("Tapering算法完成")

    def _select_pivot(self, qr_map):
        """选择pivot列"""
        array = qr_map['array']
        num_columns = array.shape[1]
        
        if self.use_most_qubit_rule:
            # 选择门数量最多的列
            gate_counts = [np.count_nonzero(array[:, col]) for col in range(num_columns)]
            return np.argmax(gate_counts)
        else:
            # 选择门数量最少的列（用于比较）
            gate_counts = [np.count_nonzero(array[:, col]) for col in range(num_columns)]
            return np.argmin(gate_counts)

    def _select_direction(self, pivot, num_columns):
        """选择移动方向"""
        if pivot < num_columns / 2:
            return "right"
        else:
            return "left"

    def _gate_pulling(self, qr_map, pivot, direction):
        """门拉动阶段"""
        array = qr_map['array']
        vertical_lines = qr_map['vertical_lines']
        num_rows, num_columns = array.shape
        
        gates_moved = 0
        
        # 处理顺序: 先处理包含pivot的门，再处理不包含的
        for phase in ["contain", "not_contain"]:
            for row in range(num_rows):
                if (phase == "contain" and array[row, pivot] > 0) or \
                   (phase == "not_contain" and array[row, pivot] <= 0):
                    
                    for col in range(num_columns):
                        if col != pivot and array[row, col] > 0:
                            # 找到可以移动的门
                            moving_col = col
                            if self._try_move_column(qr_map, moving_col, pivot, direction, row):
                                gates_moved += 1
        
        return gates_moved > 0

    def _try_move_column(self, qr_map, moving_col, pivot, direction, trigger_row):
        """尝试移动列"""
        array = qr_map['array']
        vertical_lines = qr_map['vertical_lines']
        num_rows = array.shape[0]
        
        # 生成移动偏好顺序
        preferences = self._generate_move_preferences(moving_col, pivot, direction)
        
        for target_offset in preferences:
            target_col = pivot + target_offset
            
            # 检查目标列是否有效
            if target_col < 0 or target_col >= array.shape[1]:
                continue
                
            # 检查是否可以移动（无冲突）
            if self._can_move_column(qr_map, moving_col, target_col):
                # 执行移动
                self._move_column(qr_map, moving_col, target_col)
                return True
        
        return False

    def _generate_move_preferences(self, moving_col, pivot, direction):
        """生成移动偏好顺序"""
        distance = abs(moving_col - pivot)
        preferences = [0]  # 首先尝试不移动
        
        for i in range(1, distance + 1):
            if direction == "right":
                preferences.extend([i, -i])
            else:
                preferences.extend([-i, i])
        
        return preferences

    def _can_move_column(self, qr_map, moving_col, target_col):
        """检查是否可以移动列"""
        array = qr_map['array']
        vertical_lines = qr_map['vertical_lines']
        num_rows = array.shape[0]
        
        # 检查每一行是否有冲突
        for row in range(num_rows):
            # 如果目标列在该行有门或垂直依赖，且移动列也有内容，则冲突
            if (array[row, target_col] > 0 or vertical_lines[row, target_col]) and \
               (array[row, moving_col] > 0 or vertical_lines[row, moving_col]):
                return False
        
        return True

    def _move_column(self, qr_map, moving_col, target_col):
        """移动列数据"""
        array = qr_map['array']
        vertical_lines = qr_map['vertical_lines']
        num_rows = array.shape[0]
        
        # 移动数组数据
        for row in range(num_rows):
            if array[row, moving_col] > 0:
                array[row, target_col] = array[row, moving_col]
                array[row, moving_col] = 0
            
            if vertical_lines[row, moving_col]:
                vertical_lines[row, target_col] = True
                vertical_lines[row, moving_col] = False
        
        # 更新量子位使用情况
        self._update_qubit_usage_after_move(qr_map, moving_col, target_col)
        
        print(f"移动列 {moving_col} -> {target_col}")

    def _update_qubit_usage_after_move(self, qr_map, from_col, to_col):
        """移动后更新量子位使用情况"""
        qubit_usage = qr_map['qubit_usage']
        
        # 将from_col的使用转移到to_col
        if from_col < len(qubit_usage) and to_col < len(qubit_usage):
            qubit_usage[to_col].update(qubit_usage[from_col])
            qubit_usage[from_col].clear()

    def restore_to_circuit(self):
        """将优化后的QR-Map恢复为量子电路"""
        if self.optimized_map is None:
            raise ValueError("没有优化后的QR-Map")
            
        print("恢复量子电路...")
        
        # 映射逻辑量子位到物理量子位
        qubit_mapping = self._create_qubit_mapping()
        num_physical_qubits = max(qubit_mapping.values()) + 1 if qubit_mapping else self.quantum_circuit.num_qubits
        
        # 创建新电路
        optimized_circuit = QuantumCircuit(num_physical_qubits)
        
        # 按顺序添加门
        for gate_idx, gate_info in enumerate(self.optimized_map['gate_info']):
            original_qubits = gate_info['qubits']
            physical_qubits = [qubit_mapping[q] for q in original_qubits]
            
            # 检查是否有重复的物理量子比特
            if len(physical_qubits) != len(set(physical_qubits)):
                raise ValueError(f"门 {gate_idx} 的物理量子比特存在重复: {physical_qubits}")
            
            # 添加单量子比特门
            for qubit, single_gates in gate_info['single_qubit_gates'].items():
                physical_qubit = qubit_mapping[qubit]
                for single_gate in single_gates:
                    optimized_circuit.append(single_gate['gate'], [physical_qubit])
            
            # 添加两量子比特门
            optimized_circuit.append(gate_info['gate'], physical_qubits)
            
            # 检查是否需要插入测量和重置（量子位重用点）
            self._insert_measure_reset_if_needed(optimized_circuit, gate_idx, qubit_mapping)
        
        # 计算量子位重用次数
        self.qubit_reuse_count = self._calculate_qubit_reuse()
        
        print(f"电路恢复完成: 物理量子位数={num_physical_qubits}, 重用次数={self.qubit_reuse_count}")
        return optimized_circuit

    def _count_physical_qubits(self):
        """计算所需的物理量子位数"""
        if self.optimized_map is None:
            return self.quantum_circuit.num_qubits
            
        array = self.optimized_map['array']
        num_columns = array.shape[1]
        
        # 统计有内容的列数
        used_columns = 0
        for col in range(num_columns):
            if np.any(array[:, col] > 0) or np.any(self.optimized_map['vertical_lines'][:, col]):
                used_columns += 1
        
        return used_columns

    def _create_qubit_mapping(self):
        """创建逻辑量子比特到物理量子比特的映射"""
        if self.optimized_map is None:
            return {i: i for i in range(self.quantum_circuit.num_qubits)}
        
        # 创建一个跟踪每个门中逻辑量子比特到物理量子比特的映射
        mapping = {}
        physical_qubit_id = 0
        
        # 遍历所有门，确保每个多量子比特门中的量子比特都映射到不同的物理量子比特
        for gate_info in self.optimized_map['gate_info']:
            gate_qubits = gate_info['qubits']
            gate_mapping = {}  # 当前门的临时映射
            
            # 为此门中的每个逻辑量子比特分配物理量子比特
            for logical_qubit in gate_qubits:
                if logical_qubit not in mapping:
                    mapping[logical_qubit] = physical_qubit_id
                    physical_qubit_id += 1
        
        return mapping

    def _insert_measure_reset_if_needed(self, circuit, gate_idx, qubit_mapping):
        """在需要时插入测量和重置操作"""
        # 简化的实现：在实际应用中需要更复杂的逻辑来判断重用点
        pass

    def _calculate_qubit_reuse(self):
        """计算量子位重用次数"""
        if self.optimized_map is None:
            return 0
            
        original_qubits = self.quantum_circuit.num_qubits
        optimized_qubits = self._count_physical_qubits()
        
        return original_qubits - optimized_qubits

    def extract_matrix(self):
        """抽取的矩阵表示（原有功能）"""
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
                
        self.export_matrix_to_csv(mat)
        np_mat = np.array(mat)
        print(np.array2string(np_mat, separator=', ', prefix=''))
        return mat

    def get_optimization_metrics(self):
        """获取优化指标"""
        if self.optimized_map is None:
            return {}
            
        original_qubits = self.quantum_circuit.num_qubits
        optimized_qubits = self._count_physical_qubits()
        
        return {
            'original_qubits': original_qubits,
            'optimized_qubits': optimized_qubits,
            'qubit_reduction': original_qubits - optimized_qubits,
            'reduction_rate': (original_qubits - optimized_qubits) / original_qubits * 100,
            'qubit_reuse_count': self.qubit_reuse_count
        }

