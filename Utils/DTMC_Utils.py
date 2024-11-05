import numpy as np


class DTMC_Utils:

    @staticmethod
    def build_transition_matrix(graph, S):
        """Method to build the transition matrix with an equal probability to go up, down, left or right in the grid"""
        t_m = np.matrix(np.zeros([S, S]))  # Usa np.matrix per creare una matrice
        edges = graph.edges()
        for row in range(S):
            p_ij = 1 / len([edge for edge in edges if edge[0] == row or edge[1] == row])
            t_m[row, row] = p_ij
            for col in range(S):
                if row != col and (row, col) in edges:
                    t_m[row, col] = p_ij
        DTMC_Utils.defined_check(t_m)
        return t_m

    @staticmethod
    def defined_check(M: np.matrix) -> None:
        """ It will check if the given transition matrix is well-defined """
        # Check if the matrix is squared
        if np.shape(M)[0] != np.shape(M)[1]:
            raise RuntimeError("Number of rows is different from the number of columns")

        for i in range(len(M)):
            s = 0

            for j in range(len(M)):
                # Check if the values are actually probabilities
                if M[i, j] > 1 or M[i, j] < 0:
                    raise RuntimeError(
                        "An element of the transition matrix is not a probability: " + str(i) + "," + str(j))

                s = s + M[i, j]

            # Check if the rows sum up to 1
            if s - 1 > 0.00000001:
                raise RuntimeError(
                    "The matrix is not well defined. Goes over 1 in the row " + str(i) + " with a value " + str(s))

    @staticmethod
    def __cumulativeTransformer(p0: np.array) -> np.array:
        """
        Given a probability array this function will compute the cumulative of such distribution.
        """
        p = np.zeros(len(p0), dtype=float)
        p[0] = p0[0]

        for i in range(1, len(p)):
            p[i] = p0[i] + p[i - 1]

        return p

    @staticmethod
    def obtainState(p0: np.array) -> int:
        """
        Given a probability distribution of the state this function will
        produce a weighted random state.
        """
        p = DTMC_Utils.__cumulativeTransformer(p0)

        random_state = np.random.uniform()
        s = 0
        for i in range(len(p0)):
            if (random_state < p[i]) and ((p[i] - p[i - 1]) > 0.0000001):
                s = i
                break
        return s