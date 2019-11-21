import pytest 

from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits
from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter

from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute
from qiskit import Aer

from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import partial_trace
import numpy as np

from qiskit.quantum_info.states import DensityMatrix, Statevector

@pytest.mark.parametrize("n", range(3,7))
@pytest.mark.parametrize("nshots", [1000, 5000])
def test_pairwise_fitter(n, nshots):
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    psi = np.random.rand(2**n) + 1j * np.random.rand(2**n)
    psi /= np.linalg.norm(psi)

    qc.initialize(psi, q)

    #rho = Statevector.from_instruction(qc).data
    rho = np.outer(psi.conj().T, psi)
    print("Density matrix evaluated")

    circ = pairwise_state_tomography_circuits(qc, q)

    job = execute(circ, Aer.get_backend("qasm_simulator"), shots=nshots)
    
    print("Simulation done")
    fitter = PairwiseStateTomographyFitter(job.result(), circ, q)

    np.set_printoptions(suppress=True)

    result = fitter.fit()
    print("Results fit")
    for (k, v) in result.items():
        trace_qubits = list(range(n))
        trace_qubits.remove(k[0])
        trace_qubits.remove(k[1])
        rhok = partial_trace(rho, trace_qubits)
        print(k, 1 - state_fidelity(v, rhok))
        assert 1 - state_fidelity(v, rhok) < 1 / np.sqrt(nshots)