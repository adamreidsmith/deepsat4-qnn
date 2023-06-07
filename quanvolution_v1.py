import random
from math import pi

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
import torch

from constants import *

random.seed(25)


class Quanvolution:
    def __init__(self, edges=DEFAULT_EDGES, nfilters=1, kernel_size=5, manual_filters=None, max_cores=5):
        # Number of quantvolutional filters
        self.nfilters = nfilters

        # Number of nodes in the graph assuming the nodes are labeled 0, 1, ..., n
        self.nnodes = max([e[0] for e in edges] + [e[1] for e in edges]) + 1

        self.kernel_size = kernel_size

        # The kernel has one parameter for each edge (from the problem hamiltonian), and one more parameter (from the mixer hamiltonian)
        assert kernel_size**2 == len(edges) + 1

        # Generate random filters if no filters are supplied.
        if manual_filters is None:
            self.edges: list[tuple[int, int, float]] = self.generate_random_edge_weights(edges)
        else:
            self.edges = manual_filters

        # Run tasks in parallel on the local simulator
        self.backend = Aer.get_backend('qasm_simulator')
        self.backend.set_options(max_parallel_threads=max_cores)

        # A filter is a parameterized quantum circuit with kernel_size**2 parameters
        self.filters: list[tuple[QuantumCircuit, ParameterVector]] = [
            self.create_qaoa_circuit(self.edges[i], p=1) for i in range(nfilters)
        ]

    def generate_random_edge_weights(self, edges):
        '''Add random edge weights to the graph defined by `edges`'''

        edge_list = []
        for _ in range(self.nfilters):
            # Weights are initialized randomly on the interval [0, 2pi] (they correspond to rotations)
            edge_list.append([(n1, n2, 2 * pi * random.random()) for n1, n2 in edges])
        return edge_list

    def create_qaoa_circuit(self, edges, p=1):
        '''Create the QAOA circuit parameters and get the circuit corresponding to the max-cut problem.
        `p` is the number of 'layers', i.e. the number of successive problem and mixer hamiltonitans.
        Usually p > 1 for a QAOA circuit, but we default to p = 1 here to reduce computational complexity.
        '''

        # Assign weights of 1 to edges with no weight (if any)
        edges = [(*e, 1.0) if len(e) == 2 else e for e in edges]

        # One gamma parameter for each edge, and one beta parameter for the mixer hamiltonian
        theta = (
            tuple([ParameterVector(f'$\\theta_{{{i}}}$', len(edges) + 1) for i in range(p)])
            if p > 1
            else ParameterVector(f'$\\theta$', len(edges) + 1)
        )

        qc = self.define_qaoa_circuit(edges, theta, p)
        return qc, theta

    def define_mixer_circuit(self, beta: Parameter):
        '''Define the mixer circuit.
        This circuit has a single parameter `beta` and applies a x-rotation by `beta` to each qubit:

        |q_0> -- Rx(beta) --
        |q_1> -- Rx(beta) --
        ...
        |q_n> -- Rx(beta) --
        '''

        qc_mix = QuantumCircuit(self.nnodes)
        for i in range(self.nnodes):
            qc_mix.rx(beta, i)
        return qc_mix

    def define_problem_circuit(self, edges, gamma: ParameterVector):
        '''Define the problem circuit.
        This circuit applies RZZ operations between pairs of qubits specified by the graph edges.
        For example, for a graph with edges [(0, 1), (1, 2), (0, 2)] and corresponding weights
        [a, b, c], the following 3-qubit circuit is created:

        |q_0> -- * ------------------------------------- * -----------------
                 | RZZ(a * gamma_0)                      | RZZ(c * gamma_2)
        |q_1> -- * ----------------- * -------------------------------------
                                     | RZZ(b * gamma_1)  |
        |q_2> ---------------------- * ----------------- * -----------------

        where the gamma_i are parameters.
        '''

        qc_p = QuantumCircuit(self.nnodes)
        for i, (n1, n2, w) in enumerate(edges):
            qc_p.rzz(w * gamma[i], n1, n2)
        return qc_p

    def define_qaoa_circuit(self, edges, theta, p):
        '''Compose the QAOA circuit corresponding to the max-cut problem.
        This circuit initializes all qubits in the |+> state and then applies
        `p` consecutive problem and mixer circuits.
        '''

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
        '''Objective function of the max-cut problem
        Returns the number of edges that connect oppositely labeled nodes in the graph
        '''

        obj = 0
        for i, j, _ in edges:
            if bit_string[i] != bit_string[j]:
                obj += 1
        return obj

    def compute_expectation(self, counts, edges):
        '''Computes expectation value of the problem Hamiltonian based on measurement results'''

        avg = 0
        sum_count = 0
        for bit_string, count in counts.items():
            obj = self.maxcut_obj(bit_string, edges)
            avg += obj * count
            sum_count += count
        return avg / sum_count

    def run_circuit(self, qc: QuantumCircuit, theta: ParameterVector, theta_vals: list[float], edges, shots=1024):
        '''Run the quantum circuit and return the expectation value of the problem Hamiltonian'''

        bound_qc = qc.bind_parameters({param: val for param, val in zip(theta, theta_vals)})
        counts = execute(bound_qc, self.backend, shots=shots).result().get_counts()
        return self.compute_expectation(counts, edges)

    def forward(self, t: torch.Tensor):
        '''Perform Quanvolution on batched image input tensors of shape (batch_size, channels, n, m).
        To keep the filters two dimensional, we take a channel-wise mean of the input'''

        t = torch.mean(t, dim=1, keepdim=True)  # Must keep the dimension so that torch.unfold works properly

        bs = t.shape[0]
        ks2 = self.kernel_size**2
        iout = t.shape[2] - self.kernel_size + 1
        jout = t.shape[3] - self.kernel_size + 1
        # Output tensor has shape (bs, self.nfilters, iout, jout)

        # Old method. New method below avoids the ugly nested for loops.
        # out = torch.empty([bs, self.nfilters, iout, jout])
        # for batch_index in range(t.shape[0]):
        #     for filter_index in range(self.nfilters):
        #         for i in range(iout):
        #             for j in range(jout):
        #                 window = (
        #                     t.squeeze()[batch_index, i : i + self.kernel_size, j : j + self.kernel_size]
        #                     .reshape(ks2)
        #                     .tolist()
        #                 )
        #                 expectation = self.run_circuit(
        #                     *self.filters[filter_index],
        #                     window,
        #                     self.edges[filter_index],
        #                 )
        #                 out[batch_index, filter_index, i, j] = expectation

        # Unfold the input tensor to obtain all blocks on which the quanvolution operation operates
        t_blocks = t.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        t_blocks = t_blocks.reshape(-1, self.kernel_size, self.kernel_size)

        # Define a helper function to run the circuit
        run_circuit_reduced = lambda t, i: self.run_circuit(*self.filters[i], t.reshape(ks2).tolist(), self.edges[i])

        return torch.stack(
            [
                torch.Tensor([run_circuit_reduced(t, i) for t in t_blocks]).reshape(bs, iout, jout)
                for i in range(self.nfilters)
            ],
            dim=1,
        )

    def forward_single_block(self, t):
        '''Perform the forward operation for a single block of shape (kernel_size, kernel_size)'''

        return [
            self.run_circuit(*self.filters[i], t.reshape(self.kernel_size**2).tolist(), self.edges[i])
            for i in range(self.nfilters)
        ]

    def __call__(self, t):
        assert not t.isnan().any()
        if len(t.shape) == 2:
            return self.forward_single_block(t)
        return self.forward(t)


def main():
    img = torch.rand(5, 5)
    for _ in range(10):
        Q1 = Quanvolution(nfilters=5, kernel_size=5, manual_filters=FILTERS)
        print(Q1(img))


if __name__ == '__main__':
    main()
