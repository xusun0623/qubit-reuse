from qiskit import QuantumCircuit

from circuit import QuantumCircuitManager
from quantum_chip import QuantumChip
from hardware import HardwareParams
from compiler import TimeAwareCompiler


def example_usage():
    qc = QuantumCircuit.from_qasm_str(
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    measure q[2] -> c[2];
    """
    )
    circuit_manager: QuantumCircuitManager = QuantumCircuitManager.from_qiskit_circuit(qc)
    circuit_manager.get_dag()
    circuit_manager.write_qasm("./data/example_circ.qasm")

    topo = QuantumChip.square_chip(8, 1)
    topo.visualize()

    # topo.save_json("./data/topo_3line.json")

    heavy = QuantumChip.heavy_square_chip(2, 2)
    heavy.visualize()
    # heavy.save_json("./data/heavy_square_2x2.json")

    hardware_param = HardwareParams(
        time_1q=50.0,
        time_2q=300.0,
        time_meas=4000.0,
        time_reset=1000.0,
    )

    compiler = TimeAwareCompiler(
        circuit_manager,
        topo,
        hardware_param,
        params={
            "lambda_makespan": 1.0,
            "lambda_swap": 1.0,
            "lambda_idle": 0.5
        }
    )
    result = compiler.schedule(strategy="windowed_greedy")
    print("Metrics:", result["metrics"])
    return result


if __name__ == "__main__":
    example_usage()
