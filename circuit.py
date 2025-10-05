from qiskit import QuantumCircuit, qasm3


class Circuit:
    def __init__(self, path: str):
        self.qc: QuantumCircuit = self.from_qasm_file(path)

    def from_qasm_file(self, path: str):
        self.qc = qasm3.load(path)
