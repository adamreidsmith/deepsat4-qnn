import random
from math import pi

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
import torch

from constants import DEFAULT_EDGES

random.seed(25)


class Quanvolution:
    def __init__(self, edges=DEFAULT_EDGES, nfilters=1, kernel_size=5, manual_filters=None, max_cores=5):
        self.nfilters = nfilters
        self.nnodes = max([e[0] for e in edges] + [e[1] for e in edges]) + 1  # number of nodes in the graph
        self.kernel_size = kernel_size
        assert kernel_size**2 == len(edges) + 1

        if manual_filters is None:
            self.edges: list[tuple[int, int, float]] = self.generate_random_edge_weights(edges)
        else:
            self.edges = manual_filters

        self.backend = Aer.get_backend('qasm_simulator')
        self.backend.set_options(max_parallel_threads=max_cores)

        self.filters: list[tuple[QuantumCircuit, ParameterVector]] = [
            self.create_qaoa_circuit(self.edges[i], p=1) for i in range(nfilters)
        ]

    def generate_random_edge_weights(self, edges, weights=None):
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

    def forward(self, t: torch.Tensor):
        '''Perform Quanvolution on a batched image input tensors'''

        t = torch.mean(t, dim=1, keepdim=True)  # Must keep the dimension so that torch.unfold works properly

        bs = t.shape[0]
        ks2 = self.kernel_size**2
        iout = t.shape[2] - self.kernel_size + 1
        jout = t.shape[3] - self.kernel_size + 1
        # Output tensor has shape (bs, self.nfilters, iout, jout)

        # Old method. New method below avoids the ugly nested for loops.
        # out = torch.empty([bs, self.nfilters, iout, jout])
        # t0=time.time()
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
        run_circuit = lambda t, i: self.run_circuit(*self.filters[i], t.reshape(ks2).tolist(), self.edges[i])

        return torch.stack(
            [
                torch.Tensor([run_circuit(t, i) for t in t_blocks]).reshape(bs, iout, jout)
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
    img = torch.rand(2, 4, 5, 5)
    Q1 = Quanvolution(DEFAULT_EDGES, nfilters=2, kernel_size=5)
    print(Q1(img))


if __name__ == '__main__':
    main()
