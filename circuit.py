import random
import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.converters import circuit_to_dag


class Circuit:
    def __init__(self, path: str):
        self.qc: QuantumCircuit = self.from_qasm_file(path)

    def from_qasm_file(self, path: str):
        self.qc = qasm3.load(path)
