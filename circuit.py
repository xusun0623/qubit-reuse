import random
import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.converters import circuit_to_dag


class QuantumCircuitManager:
    def __init__(self, n_qubits: int):
        self.qc: QuantumCircuit = None
        self.n_qubits = n_qubits

    def from_qasm_file(self, path: str):
        self.qc = qasm3.load(path)

    def saveCircuitFig(self, qc: QuantumCircuit, save_name: str):
        qc_img = qc.draw(output='mpl')
        qc_img.savefig(f"./circuit_imgs/{save_name}.png")

    # 保存电路到文件
    def saveCircuitToFile(self, qc: QuantumCircuit, save_name: str = "tmp"):
        basis = ['u', 'cx', 'id', 'measure']
        qc_t = transpile(qc, basis_gates=basis, optimization_level=3)
        qasm_string = qasm3.dumps(qc_t)
        with open(f"./circuits/{save_name}.qasm", 'wb') as fd:
            fd.write(qasm_string.encode())

    # 隐藏线性布尔函数算法
    def bv(self):
        def build_bv(percentage=0.5) -> QuantumCircuit:
            qubit_num = self.n_qubits
            qc = QuantumCircuit(qubit_num)
            all_possible_gates = [(0, i) for i in range(1, qubit_num)]
            gates = list(set(random.sample(all_possible_gates,
                         k=int(len(all_possible_gates) * percentage))))
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
                for j in range(i+1, qubit_num):
                    qc.cx(i, j)
            # print('{} gates'.format(qubit_num**2/2))
            return qc
        vqa_circuit = build_vqa()
        self.saveCircuitToFile(vqa_circuit, f"vqa_{self.n_qubits}")

    # 量子近似优化算法
    def qaoa(self, percentage=0.5, sorted=True):
        def build_qaoa() -> QuantumCircuit:
            qubit_num = self.n_qubits
            all_possible_gates = [(i, j) for i in range(qubit_num)
                                  for j in range(i+1, qubit_num)]
            gates = list(set(random.sample(all_possible_gates,
                         k=int(len(all_possible_gates)*percentage))))
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
                for t in range(c+1, qubit_num):
                    qiskit_circuit.rz(np.pi/3, t)
                    qiskit_circuit.cx(c, t)
                    qiskit_circuit.rz(np.pi/3, t)
                    qiskit_circuit.cx(c, t)
            return qiskit_circuit
        qft_circuit = build_qft()
        self.saveCircuitToFile(qft_circuit, f"qft_{self.n_qubits}")

