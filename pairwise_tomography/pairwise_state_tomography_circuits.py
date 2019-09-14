"""
Pairwise tomography circuit generation
"""
import copy
import numpy as np

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

def pairwise_state_tomography_circuits(circuit, measured_qubits):
    """
    Generates a minimal set of circuits for pairwise state tomography.

    This performs measurement in the Pauli-basis resulting in 
    circuits for an n-qubit state tomography experiment.

    Args:
        circuit (QuantumCircuit): the state preparation circuit to be
                                  tomographed.
        measured_qubits (list): list of the indices of qubits to be measured
    Returns:
        A list of QuantumCircuit objects containing the original circuit
        with state tomography measurements appended at the end.
    """

    ### Initialisation stuff

    #TODO: measured_qubits should be like in the ignis tomography functions, 
    # i.e. it should be a QuantumRegister or a list of QuantumRegisters
    
    ordered_qubit_list = sorted(measured_qubits)
    N = len(measured_qubits)
    
    cr = ClassicalRegister(len(measured_qubits))
    qr = circuit.qregs[0]
    
    ### Uniform measurement settings
    X = copy.deepcopy(circuit)
    Y = copy.deepcopy(circuit)
    Z = copy.deepcopy(circuit)
    
    X.add_register(cr)
    Y.add_register(cr)
    Z.add_register(cr)
    
    X.name = str(('X',)*N)
    Y.name = str(('Y',)*N)
    Z.name = str(('Z',)*N)

    for bit_index in range(len(ordered_qubit_list)):

        qubit_index = ordered_qubit_list[bit_index]

        X.h(qr[qubit_index])
        Y.sdg(qr[qubit_index])
        Y.h(qr[qubit_index])
        
        X.measure(qr[qubit_index], cr[bit_index])
        Y.measure(qr[qubit_index], cr[bit_index])
        Z.measure(qr[qubit_index], cr[bit_index])
    
    output_circuit_list = [X, Y, Z]
    
    ### Heterogeneous measurement settings
    # Generation of six possible sequences
    sequences = []
    meas_bases = ['X', 'Y', 'Z']
    for i in range(3):
        for j in range(2):
            meas_bases_copy = meas_bases[:]
            sequence = [meas_bases_copy[i]]
            meas_bases_copy.remove(meas_bases_copy[i])
            sequence.append(meas_bases_copy[j])
            meas_bases_copy.remove(meas_bases_copy[j])
            sequence.append(meas_bases_copy[0])
            sequences.append(sequence)
    
    # Qubit colouring
    nlayers = int(np.ceil(np.log(float(N))/np.log(3.0)))
    pairs = {}
    for layout in range(nlayers):
        for sequence in sequences:
            meas_layout = copy.deepcopy(circuit)
            meas_layout.add_register(cr)
            meas_layout.name = ()
            for bit_index in range(N):
                qubit_index = ordered_qubit_list[bit_index]
                local_basis = sequence[int(float(bit_index)/float(3**layout))%3]
                if local_basis == 'Y':
                    meas_layout.sdg(qr[qubit_index])
                if local_basis != 'Z':
                    meas_layout.h(qr[qubit_index])
                meas_layout.measure(qr[qubit_index], cr[bit_index])
                meas_layout.name += (local_basis,)
            meas_layout.name = str(meas_layout.name)
            output_circuit_list.append(meas_layout)
    
    return output_circuit_list
