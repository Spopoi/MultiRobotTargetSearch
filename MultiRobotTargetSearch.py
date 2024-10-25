import time
import numpy as np
from Utils.DTMC_Utils import DTMC_Utils as Utils


class MultiRobotTargetSearch:

    def __init__(self, _agents, _graph, _reference_information_state, _Zr, _alpha):
        self.eps = 0.01
        self.agents = _agents
        self.graph = _graph
        self.reference_information_state = _reference_information_state
        self.k = 0
        self.N = len(self.agents)
        self.alpha = _alpha
        self.Zr = _Zr
        self.S = self.graph.getNodesNumber()
        self.P = Utils.build_transition_matrix(self.graph, self.S)

    def run(self):
        timer = 0
        start_time = time.time()
        while not self.check_consensus():
            H_k = self.build_transition_information_states_matrix()
            actual_information_state_vector = self.build_augmented_information_state_vector()

            new_information_state_vector = np.matmul(H_k, actual_information_state_vector)
            self.update_agents_position()
            self.update_agents_information_state(new_information_state_vector[:-1, :])

            timer += 1

        print(f"Finished. Information State Vector: {self.build_augmented_information_state_vector()}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{timer} iterazioni in {execution_time} secondi")

        file_name = "trajectories.txt"
        with open(file_name, 'w') as file:
            stringa = ""
            for agent in self.agents:
                stringa += (
                    f"Agent {agent.id_number}: \n State trajectory: {str(agent.getTrajectory())} \n Information state [k]:"
                    f" {np.array(agent.getInformationStateTrajectory()).__str__()} \n\n")
            file.write(stringa)

            # for _agent in self.agents:
            #     sum1 = 0
            #     sum2 = 0
            #     N_ka = getNeighbors(_agent, agents)
            #     for b in N_ka:
            #         None # formula
            #     if _agent.getPosition() in Z_r:
            #         sum2 = -(_agent.getInformationState() - reference_information_state)
            #
            #     information_state_k = _agent.getInformationState()
            #     _agent.updateInformationState(information_state_k + sum1 + sum2)
            #     print(build_graph_laplacian(self.agents))
            #     self.k += 1 # time

    def plot(self):
        self.graph.plot_graph_with_agents(self.agents)

    def check_consensus(self):
        for ag in self.agents:
            if abs(ag.getInformationState() - self.reference_information_state) > self.eps:
                return False  # at least one agent doesn't reach the reference state
        return True

    def update_agents_position(self):
        for agent in self.agents:
            actual_state = agent.getPosition()
            new_state = Utils.obtainState(self.P[actual_state].A1)
            agent.updatePosition(new_state)

    def update_agents_information_state(self, new_information_state):
        for (i, information_state) in enumerate(new_information_state):
            # print(f"(i, information_state) = ({i},{information_state})")
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

    def build_graph_laplacian(self):
        matrix = np.zeros((self.N, self.N))
        for ii in range(self.N):
            matrix[ii][ii] = len(self.agents[ii].getNeighbors(self.agents))
        for j in range(self.N):
            for l in range(self.N):
                if j != l:
                    if self.agents[j].getPosition() == self.agents[l].getPosition():
                        matrix[j][l] = -1
                    else:
                        matrix[j][l] = 0
        # print("Laplacian: ", matrix)
        return matrix

    def build_transition_information_states_matrix(self):
        L = self.build_graph_laplacian()
        I = np.eye(self.N)

        # build d -> agent who finds the target
        d = []
        for index in range(self.N):
            if self.agents[index].getPosition() in self.Zr:
                d.append(1)
            else:
                d.append(0)

        # print("d = ", d)
        # element_11 = I - alpha*L - np.diag(d)
        element_11 = I - self.alpha * L - np.diag(d)
        # element_12 = [-x for x in d]
        element_12 = [x for x in d]
        element_12_T = np.array(element_12).reshape(-1, 1)
        element_21 = np.zeros((1, self.N))
        element_22 = np.array([[1]])

        H = np.block([
            [element_11, element_12_T], [element_21, element_22]
        ])

        # print("H = ", H)
        return H
