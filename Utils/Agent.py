import numpy as np


class Agent:
    def __init__(self, id_number, initial_position):
        self.id_number = id_number
        self.node_position = initial_position
        self.information_state = np.random.uniform()
        self.information_state_k = [self.information_state]
        # self.information_state = 0
        self.trajectory = [initial_position]

    def __str__(self):
        return str(f"Agent ID: {self.id_number}, \n"
                   f"Position: {self.node_position} \n"
                   f"Information state: {self.information_state}\n")

    def __eq__(self, other):
        return self.id_number == other.id_number

    def getID(self):
        return self.id_number

    def getPosition(self):
        return self.node_position

    def getInformationState(self):
        return self.information_state

    def updateInformationState(self, new_information_state):
        self.information_state_k.append(new_information_state)
        self.information_state = new_information_state

    def updatePosition(self, new_position):
        self.trajectory.append(new_position)
        self.node_position = new_position

    def getTrajectory(self):
        return self.trajectory

    def getInformationStateTrajectory(self):
        return self.information_state_k

    def getNeighbors(self, _agents):
        neighbors = []
        for a in _agents:
            if a != self and a.getPosition() == self.node_position:
                neighbors.append(a)
        return neighbors

    def getNeighborsContiguousNodes(self, _agents, graph):
        neighbors = []
        agent_position = self.node_position
        agent_edges = [edge for edge in graph.edges() if edge[0] == agent_position or edge[1] == agent_position]

        for a in _agents:
            if a != self:
                other_agent_position = a.getPosition()
                if (
                        other_agent_position == agent_position
                        or (other_agent_position, agent_position) in agent_edges
                        or (agent_position, other_agent_position) in agent_edges
                ):
                    neighbors.append(a)
        # print(f"Agent {a.getID()} neighbors: \n {neighbors}")
        return neighbors

