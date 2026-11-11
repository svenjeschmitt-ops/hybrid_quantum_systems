"""Exact constant Lindblad propagation using independent sparse-generator blocks.

Larger Hilbert spaces make dense diagonalization of the entire Liouvillian
expensive. Connected components give exact invariant blocks, without discarding
small populations or normalizing away loss/leakage.
"""
from types import SimpleNamespace
import numpy as np
import qutip as qt
import simulation_stop as stopping
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply


class ParkedSolver:
    def __init__(self, hamiltonian, collapse):
        self.generator = qt.liouvillian(hamiltonian, collapse).to("CSR").data.as_scipy()
        count, labels = connected_components(self.generator, directed=False)
        self.blocks = []
        for label in range(count):
            indices = np.flatnonzero(labels == label)
            matrix = self.generator[indices][:, indices]
            self.blocks.append((indices, matrix.toarray() if len(indices) <= 64 else matrix))

    def run(self, initial, times):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError("Storage times must be finite, nonempty and ordered.")
        rho = initial.proj() if initial.isket else initial
        vector = qt.operator_to_vector(rho)
        values = vector.full().ravel()
        active = [(indices, matrix) for indices, matrix in self.blocks
                  if np.any(values[indices] != 0)]
        stopping.begin()
        stopping.sample(times[0], rho)
        states = [rho.copy()]
        # Cache repeated step sizes only within one run, bounding retained memory.
        propagators = {}
        for dt in np.diff(times):
            for index, (indices, matrix) in enumerate(active):
                if len(indices) == 1:
                    values[indices] *= np.exp(matrix[0, 0]*dt)
                elif isinstance(matrix, np.ndarray):
                    key = (index, float(dt))
                    if key not in propagators:
                        propagators[key] = expm(matrix*dt)
                    values[indices] = propagators[key] @ values[indices]
                else:
                    values[indices] = expm_multiply(matrix*dt, values[indices])
            state_vector = qt.Qobj(values[:, None], dims=vector.dims)
            states.append(qt.vector_to_operator(state_vector))
            stopping.sample(times[len(states)-1], states[-1])
        return SimpleNamespace(states=states, times=times)
