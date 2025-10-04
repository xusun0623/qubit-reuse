from typing import Optional
from qiskit import QuantumCircuit, qasm2
from qiskit.converters import circuit_to_dag


class CircuitManager:
    """读取/写入QASM和一些操作"""

    def __init__(self, qc: Optional[QuantumCircuit] = None):
        self.qc = qc

    def from_qasm_file(path: str) -> "CircuitManager":
        qc = qasm2.load(path)
        return CircuitManager(qc)

    def from_qiskit_circuit(qc: QuantumCircuit) -> "CircuitManager":
        return CircuitManager(qc)

    def write_qasm(self, path: str):
        if self.qc is None:
            raise ValueError("No circuit loaded")
        with open(path, "w") as f:
            qasm2.dump(self.qc, f)

    def get_dag(self):
        if self.qc is None:
            raise ValueError("No circuit loaded")
        return circuit_to_dag(self.qc)

    def num_qubits(self):
        return 0 if self.qc is None else self.qc.num_qubits

    def to_qiskit(self):
        return self.qc
