import numpy as np


class DTMC_Utils:

    @staticmethod
    def build_transition_matrix(graph, S):
        """Method to build the transition matrix with an equal probability to go up, down, left or right in the grid"""
        t_m = np.matrix(np.zeros([S, S]))  # Usa np.matrix per creare una matrice
        edges = graph.edges()
        for row in range(S):
            for col in range(S):
                if row != col and (row, col) in edges:
                    t_m[row, col] = 1 / len([edge for edge in edges if edge[0] == row or edge[1] == row])
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
