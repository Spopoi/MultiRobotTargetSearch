import time
import numpy as np
from matplotlib import pyplot as plt

from Utils.DTMC_Utils import DTMC_Utils as Utils


class MultiRobotTargetSearch:

    def __init__(self, _agents, _graph, _reference_information_state, _Zr, _alpha, _same_node_communication=True,
                 _random_target=False, _preferred_direction=False, _noisy_measure=False):
        self.eps = 0.01
        self.agents = _agents
        self.graph = _graph
        self.reference_information_state = _reference_information_state
        self.k = 0
        self.N = len(self.agents)
        self.alpha = _alpha
        self.S = self.graph.getNodesNumber()
        if _random_target:
            self.set_random_target()
        else:
            self.Zr = _Zr
        if _preferred_direction:
            self.P = Utils.build_preferred_direction_transition_matrix(self.graph, self.S)
        else:
            self.P = Utils.build_equal_probability_transition_matrix(self.graph, self.S)

        self.iterations = 0
        self.consensus_time = np.full(self.N, np.nan)
        self.same_node_communication = _same_node_communication
        self.noisy_measure = _noisy_measure

    def run(self):
        while not self.check_consensus():
            # self.plot()
            self.update_agents_state()
            self.update_agents_position()

            self.k += 1
        self.iterations = self.k

    def plot(self):
        self.graph.plot_graph_with_agents(self.agents)

    def getIterationNumber(self):
        return self.iterations

    def getMeanConsensusTime(self):
        return np.mean(self.consensus_time)

    def getInformationStateVector(self):
        return self.build_augmented_information_state_vector()

    def writeAgentTrajectories(self):
        file_name = "trajectories.txt"
        with open(file_name, 'w') as file:
            stringa = ""
            for agent in self.agents:
                stringa += (
                    f"Agent {agent.id_number}: \n State trajectory: {str(agent.getTrajectory())} \n Information state [k]:"
                    f" {np.array(agent.getInformationStateTrajectory()).__str__()} \n\n")
            file.write(stringa)

    def check_consensus(self):
        check = True
        for ag in self.agents:
            agent_index = ag.getID() - 1
            if abs(ag.getInformationState() - self.reference_information_state) > self.eps:
                check = False  # at least one agent doesn't reach the reference state
            elif np.isnan(self.consensus_time[agent_index]):
                self.consensus_time[agent_index] = self.k
        return check

    def update_agents_position(self):
        for agent in self.agents:
            actual_state = agent.getPosition()
            new_state = Utils.obtainState(self.P[actual_state].A1)
            agent.updatePosition(new_state)

    def update_agents_state(self):
        actual_information_vector = self.getInformationStateVector()
        new_information_vector = []
        for (i, agent) in enumerate(self.agents):
            sum1 = 0
            sum2 = 0
            if self.same_node_communication:
                neighbors = agent.getNeighbors(self.agents)
            else:
                neighbors = agent.getNeighborsContiguousNodes(self.agents, self.graph)

            agent_information_state = actual_information_vector[i]

            for neighbor in neighbors:
                sum1 += np.maximum(0, self.alpha * (neighbor.getInformationState() - agent_information_state))

            if agent.getPosition() in self.Zr:
                reference_information_state = (
                    np.random.normal(self.reference_information_state, 0.02)
                    if self.noisy_measure else self.reference_information_state
                )
                sum2 = -(agent_information_state - reference_information_state)

            new_information_vector.append(min(agent_information_state + sum1 + sum2, [self.reference_information_state]))
        self.update_agents_information_state(new_information_vector)

    def update_agents_information_state(self, new_information_state):
        for (i, information_state) in enumerate(new_information_state):
            self.agents[i].updateInformationState(information_state[0])

    def build_augmented_information_state_vector(self):
        vector = []
        for a in self.agents:
            vector.append(a.getInformationState())
        vector.append(self.reference_information_state)
        column_vector = np.array(vector).reshape(-1, 1)
        return column_vector

    def plot_agents_information_state_trajectories(self):
        agents_information_state_trajectories = [
            agent.getInformationStateTrajectory() for agent in self.agents
        ] if self.agents else []

        num_instants = len(agents_information_state_trajectories[0])

        x_values = range(num_instants)

        plt.figure(figsize=(10, 6))

        for i, trajectory in enumerate(agents_information_state_trajectories):
            plt.step(x_values, trajectory, label=f'Agente {i + 1}', where='post')

        plt.title("Traiettorie degli stati di informazione per ogni agente")
        plt.xlabel("Istanti temporali (k)")
        plt.ylabel("Stato di informazione")

        plt.legend()
        plt.grid(True)

        plt.show()

    def set_random_target(self):
        """
        Genera un target casuale all'interno di S e aggiorna il vettore Zr.
        Zr[0]: nodo target
        Zr[1], Zr[2]: nodi vicini (precedente e successivo) rispetto a target_node
        """
        target_node = np.random.randint(1, self.S)
        self.Zr = np.zeros(3)
        self.Zr[0] = target_node - 1
        self.Zr[1] = target_node
        self.Zr[2] = target_node + 1

