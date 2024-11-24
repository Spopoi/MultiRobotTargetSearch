import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from Utils.Agent import Agent


class DTMC_Utils:

    @staticmethod
    def build_equal_probability_transition_matrix(graph, S):
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
        # print("matrix : ", t_m)
        return t_m

    @staticmethod
    def build_preferred_direction_transition_matrix(graph, S, north_preference=0.4):
        """
        Metodo per costruire la matrice di transizione con una direzione preferita (ad esempio, probabilità più alta di andare a nord).
        Usa esplicitamente np.matrix per rappresentare la matrice.

        Args:
            graph: Grafo connesso che rappresenta i nodi e i collegamenti.
            S: Numero di stati/nodi.
            north_preference: Probabilità preferita di muoversi verso nord.

        Returns:
            np.matrix: Matrice di transizione normalizzata.
        """
        t_m = np.matrix(np.zeros((S, S)))

        edges = graph.edges()

        for row in range(S):
            neighbors = [edge[1] if edge[0] == row else edge[0] for edge in edges if row in edge]

            if not neighbors:
                t_m[row, row] = 1
                continue

            num_neighbors = len(neighbors)

            preferred_prob = north_preference
            other_prob = (1 - north_preference) / (num_neighbors - 1) if num_neighbors > 1 else 0

            for col in neighbors:
                if DTMC_Utils.is_north(row, col, S):
                    t_m[row, col] = preferred_prob
                else:
                    t_m[row, col] = other_prob

            t_m[row, :] /= np.sum(t_m[row, :])

        DTMC_Utils.defined_check(t_m)
        # print(t_m)
        return t_m

    @staticmethod
    def is_north(row, col, num_of_nodes):
        """Placeholder function to determine if col is north of row"""
        # Define logic for grid layout to check if col is north of row
        # For example, if nodes are arranged in an NxN grid:
        N = int(num_of_nodes ** 0.5)  # assuming a square grid
        return col == row - N

    @staticmethod
    def is_probabilistic_transition_matrix(matrix):
        """
        Verifica se una matrice è una matrice di transizione probabilistica per una DTMC.
        Args:
            matrix (numpy.ndarray): Matrice da verificare.
        Returns:
            bool: True se è una matrice di transizione valida, False altrimenti.
        """
        # Verifica non negatività
        if not np.all(matrix >= 0):
            print("La matrice contiene valori negativi.")
            return False

        # Verifica che la somma di ogni riga sia 1 (entro una tolleranza)
        row_sums = np.sum(matrix, axis=1)
        if not np.allclose(row_sums, 1):
            print("Le somme delle righe non sono tutte uguali a 1.")
            print("Somme delle righe:", row_sums)
            return False

        # Se entrambe le condizioni sono soddisfatte
        return True

    @staticmethod
    def initAgents(number_of_agents, number_of_nodes):
        agent_list = []
        for i in range(number_of_agents):
            initial_node = np.random.randint(0, number_of_nodes)
            agent = Agent(i + 1, initial_node)
            agent_list.append(agent)
        return agent_list

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
    def plot_average_execution_time(agents_number_list, average_execution_time_list, std_deviation_list, x_label, title):
        plt.figure(figsize=(8, 6))

        plt.errorbar(
            agents_number_list, average_execution_time_list, yerr=std_deviation_list,
            fmt='o-', color='b', ecolor='gray', elinewidth=2, capsize=4, label='Average Execution Time'
        )

        plt.xlabel(x_label)
        plt.ylabel("Average Execution Time (iterations)")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

    @staticmethod
    def plot_average_execution_time_comparison(agents_number_list, avg_time_list_1, std_dev_list_1, avg_time_list_2,
                                               std_dev_list_2, x_label):
        # Configurazione del grafico
        plt.figure(figsize=(10, 6))

        # Comunicazione nello stesso nodo
        plt.errorbar(
            agents_number_list, avg_time_list_1, yerr=std_dev_list_1,
            fmt='o-', color='b', ecolor='b', elinewidth=2, capsize=4, label='Same Node Communication'
        )
        # Comunicazione in nodi contigui
        plt.errorbar(
            agents_number_list, avg_time_list_2, yerr=std_dev_list_2,
            fmt='o-', color='r', ecolor='r', elinewidth=2, capsize=4, label='Contiguous Node Communication'
        )

        # Etichette e titoli
        plt.xlabel(x_label)
        plt.ylabel("Average Execution Time (iterations)")
        plt.title("Average Execution Time on a 5x5 grid over 1000 Monte Carlo Simulations:\n Comparison Between Same "
                  "Node Communication vs Contiguous Node Communication.")
        plt.legend()

        # Griglia e formattazione dell'asse
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Mostra il grafico
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_separate_execution_time_comparison(agents_number_list, avg_time_list_1, std_dev_list_1, avg_time_list_2,
                                                std_dev_list_2, x_label):
        # Configurazione dei subplot
        fig, axs = plt.subplots(1, 2, figsize=(14, 6),
                                sharey=True)  # Condividiamo l'asse y per mantenere la stessa scala

        # Primo subplot: Comunicazione nello stesso nodo
        axs[0].errorbar(
            agents_number_list, avg_time_list_1, yerr=std_dev_list_1,
            fmt='o-', color='b', ecolor='b', elinewidth=2, capsize=4
        )
        axs[0].set_title("Same Node Communication")
        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel("Average Execution Time (iterations)")
        axs[0].grid(True, alpha=0.3)
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Secondo subplot: Comunicazione in nodi contigui
        axs[1].errorbar(
            agents_number_list, avg_time_list_2, yerr=std_dev_list_2,
            fmt='o-', color='r', ecolor='r', elinewidth=2, capsize=4
        )
        axs[1].set_title("Contiguous Node Communication")
        axs[1].set_xlabel(x_label)
        axs[1].grid(True, alpha=0.3)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Miglior layout
        plt.tight_layout()
        plt.show()

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
