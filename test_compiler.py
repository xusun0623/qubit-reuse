from qiskit import QuantumCircuit, qasm3
from quantum_chip import QuantumChip
from hardware import HardwareParams
from qrmove_compiler import QRMoveCompiler
from utils import get_quantum_circuit

_type = "bv"
_qubit_num = 10 # 395行 * 100列

quantum_circuit: QuantumCircuit = get_quantum_circuit(_type, _qubit_num)

# quantum_circuit.draw(output="mpl", filename="./output/qaoa_10.png")
quantum_chip = QuantumChip(
    # square / hexagon / heavy_square / heavy_hexagon
    "square",
    50,
)
# quantum_chip.visualize()
hardware_param = HardwareParams(
    time_1q=50.0,  # 单比特门时间
    time_2q=300.0,  # 双比特门时间
    time_meas=4000.0,  # 测量时间
    time_reset=1000.0,  # 重置时间
)
qmc = QRMoveCompiler(
    quantum_circuit,
    quantum_chip,
    hardware_param,
)