import math

import numpy as np


class QuantumState:
    def __init__(self, amplitudes: np.ndarray):
        assert len(amplitudes.shape) == 1, "amplitudes should be a one dimensional array"
        assert np.log2(len(amplitudes)).is_integer(), "length of amplitudes should be 2^n"
        ms_sum = np.sum(np.square(np.abs(amplitudes)))
        assert np.isclose(np.sum(np.square(np.abs(amplitudes))),
                          1), f"sum of mod square of amplitudes needs to be 1, but was {ms_sum}"
        self.__amplitudes = amplitudes

    def apply_gate(self, gate_matrix):
        new_amplitudes = np.matmul(gate_matrix, self.__amplitudes)
        return QuantumState(new_amplitudes)

    def amplitudes(self):
        return self.__amplitudes

    def multiply_with_qubit(self, qs):
        new_amplitudes = np.kron(qs.amplitudes(), self.amplitudes())
        return QuantumState(new_amplitudes)

    def drawing_data(self):
        amplitude_lengths = len(self.__amplitudes)
        result = []
        rotation = 2 * np.pi / (amplitude_lengths / 2)
        for idx, amplitude in enumerate(self.__amplitudes):
            vect = [[0, np.angle(amplitude)], [0, np.abs(amplitude)]]
            sign_multiplier = 1 if idx % 2 == 1 else -1

            probability = np.square(np.abs(amplitude))
            if sign_multiplier == -1:
                theta = np.linspace(math.floor(idx / 2) * rotation + (-np.pi / amplitude_lengths),
                                    math.floor(idx / 2) * rotation + (np.pi / amplitude_lengths), 50)
            else:
                theta = np.linspace(math.floor(idx / 2) * rotation + (np.pi / amplitude_lengths),
                                    math.floor(idx / 2) * rotation + (3 * np.pi / amplitude_lengths),
                                    50)
            result.append((vect, (theta, probability)))
        return result


class QuantumGate:
    def __init__(self, gate_matrix):
        assert np.allclose(np.eye(gate_matrix.shape[0]), gate_matrix.H * gate_matrix), "gate matrix needs to be unitary"

    def drawing_data(self):
        pass
