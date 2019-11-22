
# pylint: disable=missing-docstring
import unittest

import numpy as np
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import partial_trace
from qiskit.quantum_info.states import DensityMatrix

from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits
from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter

n_list = [2, 4]
nshots = 5000

    
class TestPairwiseStateTomography(unittest.TestCase):
    def test_pairwise_tomography(self):
        for n in n_list:
            with self.subTest():
                self.tomography_random_circuit(n)
    
    def tomography_random_circuit(self, n):
        q = QuantumRegister(n)
        qc = QuantumCircuit(q)

        psi = ((2 * np.random.rand(2 ** n) - 1) 
            + 1j * (2 *np.random.rand(2 ** n) - 1))
        psi /= np.linalg.norm(psi)

        qc.initialize(psi, q)
        rho = DensityMatrix.from_instruction(qc).data

        circ = pairwise_state_tomography_circuits(qc, q)
        job = execute(circ, Aer.get_backend("qasm_simulator"), shots=nshots)
        fitter = PairwiseStateTomographyFitter(job.result(), circ, q)
        result = fitter.fit()

        # Compare the tomography matrices with the partial trace of 
        # the original state using fidelity
        for (k, v) in result.items():
            trace_qubits = list(range(n))
            trace_qubits.remove(k[0])
            trace_qubits.remove(k[1])
            rhok = partial_trace(rho, trace_qubits)
            f = state_fidelity(v, rhok)
            try:
                self.assertAlmostEqual(f, 1, delta=1 / np.sqrt(nshots))
            except AssertionError:
                print(k, f)
                raise
            
    def test_meas_qubit_specification(self):
        n = 4

        q = QuantumRegister(n)
        qc = QuantumCircuit(q)

        psi = ((2 * np.random.rand(2 ** n) - 1) 
            + 1j * (2 *np.random.rand(2 ** n) - 1))
        psi /= np.linalg.norm(psi)

        qc.initialize(psi, q)
        rho = DensityMatrix.from_instruction(qc).data

        measured_qubits = q#[q[0], q[1], q[2]]
        circ = pairwise_state_tomography_circuits(qc, measured_qubits)
        job = execute(circ, Aer.get_backend("qasm_simulator"), shots=nshots)
        fitter = PairwiseStateTomographyFitter(job.result(), circ, measured_qubits)
        result = fitter.fit()

        # Compare the tomography matrices with the partial trace of 
        # the original state using fidelity
        for (k, v) in result.items():
            trace_qubits = list(range(n))
            trace_qubits.remove(measured_qubits[k[0]].index)
            trace_qubits.remove(measured_qubits[k[1]].index)
            rhok = partial_trace(rho, trace_qubits)
            f = state_fidelity(v, rhok)
            try:
                self.assertAlmostEqual(f, 1, delta=1 / np.sqrt(nshots))
            except AssertionError:
                print(k, f)
                raise

    def test_multiple_registers(self):
        n = 4

        q = QuantumRegister(2)
        p = QuantumRegister(2)

        qc = QuantumCircuit(q, p)
        
        qc.h(q[0])
        qc.rx(np.pi/4, q[1])
        qc.cx(q[0], p[0])
        qc.cx(q[1], p[1])

        rho = DensityMatrix.from_instruction(qc).data

        measured_qubits = q#[q[0], q[1], q[2]]
        circ = pairwise_state_tomography_circuits(qc, measured_qubits)
        job = execute(circ, Aer.get_backend("qasm_simulator"), shots=nshots)
        fitter = PairwiseStateTomographyFitter(job.result(), circ, measured_qubits)
        result = fitter.fit()

        # Compare the tomography matrices with the partial trace of 
        # the original state using fidelity
        for (k, v) in result.items():
            trace_qubits = list(range(n))
            trace_qubits.remove(measured_qubits[k[0]].index)
            trace_qubits.remove(measured_qubits[k[1]].index)
            rhok = partial_trace(rho, trace_qubits)
            f = state_fidelity(v, rhok)
            try:
                self.assertAlmostEqual(f, 1, delta=1 / np.sqrt(nshots))
            except AssertionError:
                print(k, f)
                raise

if __name__ == '__main__':
    unittest.main()
