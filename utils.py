from qiskit import QuantumCircuit, qasm3

import random
import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile


class CircuitGenerator:
    def __init__(self, n_qubits: int):
        self.qc: QuantumCircuit = None
        self.n_qubits = n_qubits

    def generateAll(self):
        self.bv()
        self.vqa()
        self.qaoa()
        self.qft()
        self.mod()
        self.mul()
        self.qram()
        self.rd()
        self.sym()
        self.xor()

    def saveCircuitFig(self, qc: QuantumCircuit, save_name: str):
        qc_img = qc.draw(output="mpl")
        qc_img.savefig(f"./circuit_imgs/{save_name}.png")

    # 保存电路到文件
    def saveCircuitToFile(self, qc: QuantumCircuit, save_name: str = "tmp"):
        basis = ["u", "cx", "id", "measure"]
        qc_t = transpile(qc, basis_gates=basis, optimization_level=3)
        qasm_string = qasm3.dumps(qc_t)
        with open(f"./circuits/{save_name}.qasm", "wb") as fd:
            fd.write(qasm_string.encode())

    # 隐藏线性布尔函数算法
    def bv(self):
        def build_bv(percentage=0.5) -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)
            all_possible_gates = [(0, i) for i in range(1, qubit_num)]
            gates = list(
                set(
                    random.sample(
                        all_possible_gates, k=int(len(all_possible_gates) * percentage)
                    )
                )
            )
            if sorted:
                gates.sort()
            for c, t in gates:
                qc.cx(c, t)
            return qc

        bv_circuit = build_bv()
        self.saveCircuitToFile(bv_circuit, f"bv_{self.n_qubits}")

    # 变分量子算法
    def vqa(self):
        def build_vqa() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)
            # 完成这里的电路构建
            for i in range(qubit_num):
                for j in range(i + 1, qubit_num):
                    qc.cx(i, j)
            # print('{} gates'.format(qubit_num**2/2))
            return qc

        vqa_circuit = build_vqa()
        self.saveCircuitToFile(vqa_circuit, f"vqa_{self.n_qubits}")

    # 量子近似优化算法
    def qaoa(self, percentage=0.5, sorted=True):
        def build_qaoa() -> QuantumCircuit:
            qubit_num = self.n_qubits
            all_possible_gates = [
                (i, j) for i in range(qubit_num) for j in range(i + 1, qubit_num)
            ]
            gates = list(
                set(
                    random.sample(
                        all_possible_gates, k=int(len(all_possible_gates) * percentage)
                    )
                )
            )
            if sorted:
                gates.sort()
            # print("{}/{} gates selected : {} ... {}".format(len(gates), len(all_possible_gates), gates[:10], gates[-10:]))
            qiskit_circuit = QuantumCircuit(qubit_num)
            for c, t in gates:
                qiskit_circuit.cx(c, t)
                qiskit_circuit.rz(np.pi / 3, t)
                qiskit_circuit.cx(c, t)
            return qiskit_circuit

        qaoa_circuit = build_qaoa()
        self.saveCircuitToFile(qaoa_circuit, f"qaoa_{self.n_qubits}")

    # 量子傅里叶算法
    def qft(self):
        def build_qft() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qiskit_circuit = QuantumCircuit(qubit_num)
            for c in range(qubit_num):
                for t in range(c + 1, qubit_num):
                    qiskit_circuit.rz(np.pi / 3, t)
                    qiskit_circuit.cx(c, t)
                    qiskit_circuit.rz(np.pi / 3, t)
                    qiskit_circuit.cx(c, t)
            return qiskit_circuit

        qft_circuit = build_qft()
        self.saveCircuitToFile(qft_circuit, f"qft_{self.n_qubits}")

    # x mod y 算法
    def mod(self):
        def build_mod() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)

            # 实现 x mod y 算法的基本结构
            # 这里采用简化版本，实际实现可能需要更多细节
            for i in range(qubit_num // 2):
                if 2 * i + 1 < qubit_num:
                    qc.cx(2 * i, 2 * i + 1)
                    qc.rz(np.pi / 4, 2 * i + 1)
                    qc.cx(2 * i, 2 * i + 1)

            return qc

        mod_circuit = build_mod()
        self.saveCircuitToFile(mod_circuit, f"mod_{self.n_qubits}")

    # Mul_n 算法 (乘法)
    def mul(self):
        def build_mul() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)

            # 实现乘法运算的量子电路
            # 控制位和目标位的组合
            for i in range(qubit_num):
                for j in range(i + 1, min(i + 3, qubit_num)):  # 限制连接范围
                    qc.cx(i, j)
                    qc.rz(np.pi / 2, j)
                    qc.cx(i, j)

            return qc

        mul_circuit = build_mul()
        self.saveCircuitToFile(mul_circuit, f"mul_{self.n_qubits}")

    # QRAM (Quantum Random Access Memory)
    def qram(self):
        def build_qram() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)

            # 实现 QRAM 的基本结构
            address_qubits = qubit_num // 2
            data_qubits = qubit_num - address_qubits

            # 地址编码
            for i in range(min(address_qubits, data_qubits)):
                addr_idx = i
                data_idx = address_qubits + i
                if data_idx < qubit_num:
                    qc.cx(addr_idx, data_idx)
                    qc.rz(np.pi / 3, data_idx)
                    qc.cx(addr_idx, data_idx)

            return qc

        qram_circuit = build_qram()
        self.saveCircuitToFile(qram_circuit, f"qram_{self.n_qubits}")

    # RD_n 算法
    def rd(self):
        def build_rd() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)

            # 实现 RD_n 算法的基本结构
            # 这是一种随机化算法的简化实现
            connections = [(i, (i + 2) % qubit_num) for i in range(qubit_num - 1)]
            selected_connections = random.sample(
                connections, k=max(1, len(connections) // 2)
            )

            for c, t in selected_connections:
                qc.cx(c, t)
                qc.ry(np.pi / 4, t)
                qc.cx(c, t)

            return qc

        rd_circuit = build_rd()
        self.saveCircuitToFile(rd_circuit, f"rd_{self.n_qubits}")

    # Sym 算法 (对称函数)
    def sym(self):
        def build_sym() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)

            # 实现对称函数算法
            # 对所有可能的量子比特对应用对称操作
            for i in range(qubit_num):
                for j in range(i + 1, qubit_num):
                    # 添加对称操作
                    qc.cx(i, j)
                    qc.rz(np.pi / 2, j)
                    qc.cx(i, j)
                    # 反向操作保持对称性
                    qc.cx(j, i)
                    qc.rz(np.pi / 2, i)
                    qc.cx(j, i)

            return qc

        sym_circuit = build_sym()
        self.saveCircuitToFile(sym_circuit, f"sym_{self.n_qubits}")

    # XOR 算法
    def xor(self):
        def build_xor() -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)

            # 实现 XOR 算法 - 量子比特之间的异或操作
            for i in range(0, qubit_num - 1, 2):  # 成对处理
                qc.cx(i, i + 1)

            # 添加一些额外的纠缠操作
            for i in range(qubit_num):
                j = (i + qubit_num // 2) % qubit_num
                if i != j:
                    qc.cx(i, j)
                    qc.rz(np.pi / 4, j)
                    qc.cx(i, j)

            return qc

        xor_circuit = build_xor()
        self.saveCircuitToFile(xor_circuit, f"xor_{self.n_qubits}")


def get_quantum_circuit(quantum_alg, qubit_num) -> QuantumCircuit:
    quantum_circuit: QuantumCircuit = None
    try:
        quantum_circuit = qasm3.load(
            # 从qasm文件中加载电路
            f"./circuits/{quantum_alg}_{qubit_num}.qasm"
        )
    except:
        # 如果加载失败，则生成相应类型的电路
        generator = CircuitGenerator(qubit_num)
        method_map = {
            "bv": generator.bv,
            "vqa": generator.vqa,
            "qaoa": generator.qaoa,
            "qft": generator.qft,
            "mod": generator.mod,
            "mul": generator.mul,
            "qram": generator.qram,
            "rd": generator.rd,
            "sym": generator.sym,
            "xor": generator.xor,
        }

        # 调用对应的方法生成电路
        if quantum_alg in method_map:
            method_map[quantum_alg]()

        # 尝试再次加载生成的电路文件
        try:
            quantum_circuit = qasm3.load(f"./circuits/{quantum_alg}_{qubit_num}.qasm")
        except:
            pass

    return quantum_circuit

