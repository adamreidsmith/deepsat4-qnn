import random
from math import pi

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
import torch

random.seed(25)

DEFAULT_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (15, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
]


class Quanvolution:
    def __init__(self, edges, nfilters, kernel_size):
        self.nfilters = nfilters
        self.nnodes = max([e[0] for e in edges] + [e[1] for e in edges]) + 1  # number of nodes in the graph
        assert kernel_size**2 == len(edges) + 1
        self.kernel_size = kernel_size
        self.edges: list[tuple[int, int, float]] = self.generate_random_edge_weights(edges)
        self.backend = Aer.get_backend('qasm_simulator')
        self.filters: list[tuple[QuantumCircuit, ParameterVector]] = [
            self.create_qaoa_circuit(self.edges[i], p=1) for i in range(nfilters)
        ]

    def generate_random_edge_weights(self, edges):
        '''Add random edge weights to the graph defined by `edges`'''

        # k - number of filters
        edge_list = []
        for _ in range(self.nfilters):
            edge_list.append([(n1, n2, 2 * pi * random.random()) for n1, n2 in edges])
        return edge_list

    def create_qaoa_circuit(self, edges, p=1):
        '''Create the QAOA circuit parameters and get the circuit corresponding to the max-cut problem'''

        # Assign weights of 1 to edges with no weight
        edges = [(*e, 1.0) if len(e) == 2 else e for e in edges]

        # One gamma parameter for each edge, and one beta parameter for the mixer hamiltonian
        theta = (
            tuple([ParameterVector(f'$\\theta_{{{i}}}$', len(edges) + 1) for i in range(p)])
            if p > 1
            else ParameterVector(f'$\\theta$', len(edges) + 1)
        )

        qc = self.define_qaoa_circuit(edges, theta, p)
        return qc, theta

    def define_mixer_circuit(self, beta):
        '''Define the mixer circuit'''

        qc_mix = QuantumCircuit(self.nnodes)
        for i in range(self.nnodes):
            qc_mix.rx(beta, i)
        return qc_mix

    def define_problem_circuit(self, edges, gamma):
        '''Define the problem circuit'''

        qc_p = QuantumCircuit(self.nnodes)
        for i, (n1, n2, w) in enumerate(edges):  # pairs of nodes and edge weights
            qc_p.rzz(w * gamma[i], n1, n2)
            # qc_p.barrier()
        return qc_p

    def define_qaoa_circuit(self, edges, theta, p):
        '''Compose the QAOA circuit corresponding to the max-cut problem'''

        qc = QuantumCircuit(self.nnodes)

        # The intital state is the uniform superposition
        qc.h(range(self.nnodes))

        # Define the mixer and problem circuits
        for layer in range(p):
            qc_p = self.define_problem_circuit(edges, theta[layer][1:] if isinstance(theta, list) else theta[1:])
            qc_mix = self.define_mixer_circuit(theta[layer][0] if isinstance(theta, list) else theta[0])

            # Append the circuits to the master quantum circuit
            qc.compose(qc_p, inplace=True)
            qc.barrier()
            qc.compose(qc_mix, inplace=True)

        # Measure all qubits
        qc.measure_all()
        return qc

    def maxcut_obj(self, bit_string, edges):
        '''
        Objective function of the max-cut problem

        Returns the number of edges that connect oppositely labeled nodes in the graph
        '''
        obj = 0
        for i, j, _ in edges:
            if bit_string[i] != bit_string[j]:
                obj += 1
        return obj

    def compute_expectation(self, counts, edges):
        '''
        Computes expectation value of the problem Hamiltonian based on measurement results
        Args:
            counts: (dict) key as bit string, val as count
            graph: networkx graph
        Returns:
            avg: float
                expectation value
        '''
        avg = 0
        sum_count = 0
        for bit_string, count in counts.items():
            obj = self.maxcut_obj(bit_string, edges)
            avg += obj * count
            sum_count += count
        return avg / sum_count

    def run_circuit(self, qc, theta, theta_vals, edges, shots=1024):
        '''Run the quantum circuit and return the expectation value of the problem Hamiltonian'''

        bound_qc = qc.bind_parameters({param: val for param, val in zip(theta, theta_vals)})

        counts = execute(bound_qc, self.backend, shots=shots).result().get_counts()
        return self.compute_expectation(counts, edges)

    def __call__(self, t: torch.Tensor):
        '''Perform Quanvolution on a batched image input tensors'''

        t = torch.mean(t, dim=1)

        ks2 = self.kernel_size**2
        imax = t.shape[1] - self.kernel_size + 1
        jmax = t.shape[2] - self.kernel_size + 1

        out = torch.empty([t.shape[0], self.nfilters, imax, jmax])
        print(t.shape[0] * self.nfilters * imax * jmax)
        for batch_index in range(t.shape[0]):
            for filter_index in range(self.nfilters):
                for i in range(imax):
                    for j in range(jmax):
                        window = (
                            t[batch_index, i : i + self.kernel_size, j : j + self.kernel_size].reshape(ks2).tolist()
                        )
                        expectation = self.run_circuit(
                            self.filters[filter_index][0],
                            self.filters[filter_index][1],
                            window,
                            self.edges[filter_index],
                        )
                        print(expectation)
                        out[batch_index, filter_index, i, j] = expectation
        return out


def main():
    img = torch.rand(2, 4, 5, 5)
    Q1 = Quanvolution(DEFAULT_EDGES, nfilters=1, kernel_size=5)
    print(Q1(img))


if __name__ == '__main__':
    main()
