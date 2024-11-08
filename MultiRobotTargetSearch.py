import time
import numpy as np
from matplotlib import pyplot as plt

from Utils.DTMC_Utils import DTMC_Utils as Utils


class MultiRobotTargetSearch:

    def __init__(self, _agents, _graph, _reference_information_state, _Zr, _alpha, same_node_communication=True):
        self.eps = 0.01
        self.agents = _agents
        self.graph = _graph
        self.reference_information_state = _reference_information_state
        self.k = 0
        self.N = len(self.agents)
        self.alpha = _alpha
        self.Zr = _Zr
        self.S = self.graph.getNodesNumber()
        self.P = Utils.build_equal_probability_transition_matrix(self.graph, self.S)
        # self.P = Utils.build_preferred_direction_transition_matrix(self.graph, self.S)
        # self.execution_time = 0
        self.iterations = 0
        # self.consensus_time = np.zeros(self.N)
        self.consensus_time = np.full(self.N, np.nan)
        self.same_node_communication = same_node_communication

    def run(self):
        # timer = 1
        # start_time = time.time()
        while not self.check_consensus():
            # self.plot()
            # print(self.k)
            # H_k = self.build_transition_information_states_matrix()
            # actual_information_state_vector = self.build_augmented_information_state_vector()
            # print(f"{self.k}: information state \n {actual_information_state_vector}")

            # new_information_state_vector = np.matmul(H_k, actual_information_state_vector)
            self.update_agents_state()
            self.update_agents_position()
            # self.update_agents_information_state(new_information_state_vector[:-1, :])

            self.k += 1
            # timer += 1
        # self.iterations = timer
        self.iterations = self.k
        # print("last iteration: ", self.iterations)
        # print(f"Finished. Information State Vector: {self.build_augmented_information_state_vector()}")
        # end_time = time.time()
        # self.execution_time = end_time - start_time
        # print(f"{timer} iterazioni in {self.execution_time} secondi")

    def plot(self):
        self.graph.plot_graph_with_agents(self.agents)

    def getIterationNumber(self):
        return self.iterations

    def getMeanConsensusTime(self):
        # print("mean consensus time: ", np.mean(self.consensus_time))
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

    # def getExecutionTime(self):
    #     return self.execution_time

    def check_consensus(self):
        # for ag in self.agents:
        #     agent_index = ag.getID()-1
        #     if abs(ag.getInformationState() - self.reference_information_state) > self.eps:
        #         return False  # at least one agent doesn't reach the reference state
        #     elif np.isnan(self.consensus_time[agent_index]):
        #         self.consensus_time[agent_index] = self.k
        # return True
        check = True
        for ag in self.agents:
            agent_index = ag.getID() - 1
            if abs(ag.getInformationState() - self.reference_information_state) > self.eps:
                check = False  # at least one agent doesn't reach the reference state
            elif np.isnan(self.consensus_time[agent_index]):
                # print(f"update agent {agent_index} at time {self.k}")
                self.consensus_time[agent_index] = self.k
        return check

    def update_agents_position(self):
        for agent in self.agents:
            actual_state = agent.getPosition()
            new_state = Utils.obtainState(self.P[actual_state].A1)
            # print(f"agent {agent.id_number} from {actual_state} -> {new_state}")
            agent.updatePosition(new_state)

    # def update_agents_state(self):
    #     actual_information_vector = self.getInformationStateVector()
    #     new_information_vector = []
    #     for (i, agent) in enumerate(self.agents):
    #         sum1 = 0
    #         sum2 = 0
    #         # neighbors = agent.getNeighbors(self.agents)
    #         neighbors = agent.getNeighborsContiguousNodes(self.agents, self.graph)
    #         agent_information_state = actual_information_vector[i]
    #         for neighbor in neighbors:
    #             # sum1 += self.alpha * abs(agent_information_state - neighbor.getInformationState())
    #             sum1 -= self.alpha * (agent_information_state - neighbor.getInformationState())
    #             # print(f"Exchange information from agent {agent.getID()} to {neighbor.getID()}")
    #         if agent.getPosition() in self.Zr:
    #             sum2 = -(agent_information_state - self.reference_information_state)
    #             # print(f"agent {agent.getID()} finds the feature")
    #         # agent.updateInformationState(agent_information_state + sum1 + sum2)
    #         # print(f"sum1 = {sum1}, sum2 = {sum2}, total sum = {agent_information_state + sum1 + sum2}")
    #         new_information_vector.append(agent_information_state + sum1 + sum2)
    #     self.update_agents_information_state(new_information_vector)
    #     # print(f"old state: {actual_information_vector} \n new state: {new_information_vector}")

    def update_agents_state(self):
        actual_information_vector = self.getInformationStateVector()
        new_information_vector = []
        for (i, agent) in enumerate(self.agents):
            sum1 = 0
            sum2 = 0
            if self.same_node_communication:
                neighbors = agent.getNeighbors(self.agents)
            else:
                neighbors = agent.getNeighborsContiguousNodes(self.agents)

            # neighbors = agent.getNeighborsContiguousNodes(self.agents, self.graph)
            agent_information_state = actual_information_vector[i]

            for neighbor in neighbors:
                sum1 += self.alpha * np.maximum(0, neighbor.getInformationState() - agent_information_state)

            if agent.getPosition() in self.Zr:
                sum2 = -(agent_information_state - self.reference_information_state)

            new_information_vector.append(agent_information_state + sum1 + sum2)
        self.update_agents_information_state(new_information_vector)

    def update_agents_information_state(self, new_information_state):
        for (i, information_state) in enumerate(new_information_state):
            # print(f"(i, information_state) = ({i},{information_state[0]})")
            self.agents[i].updateInformationState(information_state[0])
            # self.agents[i].updateInformationState(max(information_state[0],0))

    def build_augmented_information_state_vector(self):
        vector = []
        for a in self.agents:
            vector.append(a.getInformationState())
        vector.append(self.reference_information_state)
        column_vector = np.array(vector).reshape(-1, 1)
        # print("information vector: ", column_vector)
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

    # def build_graph_laplacian(self):
    #     matrix = np.zeros((self.N, self.N))
    #     for actor in self.agents:
    #         position = actor.getPosition()
    #         id = actor.getID()
    #
    #         neighbors = actor.getNeighbors(self.agents)
    #         matrix[id - 1][id - 1] = len(neighbors)
    #
    #         for neighbor in neighbors:
    #             if position == neighbor.getPosition():
    #                 matrix[id - 1][neighbor.getID() - 1] = -1
    #     # print("Laplacian: \n", matrix)
    #     return matrix
        # for ii in range(self.N):
        #     matrix[ii][ii] = len(self.agents[ii].getNeighbors(self.agents))
        # for j in range(self.N):
        #     for l in range(self.N):
        #         if j != l:
        #             if self.agents[j].getPosition() == self.agents[l].getPosition():
        #                 matrix[j][l] = -1
        #             else:
        #                 matrix[j][l] = 0
        # # print("Laplacian: ", matrix)
        # return matrix

    # def build_transition_information_states_matrix(self):
    #     L = self.build_graph_laplacian()
    #     I = np.eye(self.N)
    #
    #     # build d -> agent who finds the target
    #     d = []
    #     for index in range(self.N):
    #         if self.agents[index].getPosition() in self.Zr:
    #             d.append(1)
    #         else:
    #             d.append(0)
    #
    #     # print("d = ", d)
    #     # element_11 = I - alpha*L + np.diag(d)
    #     element_11 = I - self.alpha * L - np.diag(d)
    #     # element_12 = [-x for x in d]
    #     element_12 = [x for x in d]
    #     element_12_T = np.array(element_12).reshape(-1, 1)
    #     element_21 = np.zeros((1, self.N))
    #     element_22 = np.array([[1]])
    #
    #     H = np.block([
    #         [element_11, element_12_T], [element_21, element_22]
    #     ])
    #
    #     # print("H = ", H)
    #     return H
