import networkx as nx
import matplotlib.pyplot as plt


class GridGraph:
    def __init__(self, n, m):
        """Constructor to initialize the grid dimensions and create the graph."""
        self.n = n  # Number of rows
        self.m = m  # Number of columns
        self.graph = self._create_graph()
        self.node_positions = self._create_node_positions()

    def _create_graph(self):
        """Private method to create the grid graph."""
        G = nx.Graph()  # Create an undirected graph

        # Add nodes and edges based on the grid
        for i in range(self.n):
            for j in range(self.m):
                node = i * self.m + j
                G.add_node(node)

                # Connect to neighbors: above, below, left, right
                if i > 0:  # Above
                    above = (i - 1) * self.m + j
                    G.add_edge(node, above)
                if i < self.n - 1:  # Below
                    below = (i + 1) * self.m + j
                    G.add_edge(node, below)
                if j > 0:  # Left
                    left = i * self.m + (j - 1)
                    G.add_edge(node, left)
                if j < self.m - 1:  # Right
                    right = i * self.m + (j + 1)
                    G.add_edge(node, right)

        return G

    def _create_node_positions(self):
        """Private method to create a dictionary of node positions (i, j)."""
        positions = {}
        for i in range(self.n):
            for j in range(self.m):
                node = i * self.m + j
                positions[node] = (i, j)
        return positions

    def edges(self):
        return self.graph.edges()

    def get_position(self, node):
        """Method to get the (i, j) position of a node."""
        return self.node_positions.get(node, None)

    def plot_graph(self):
        """Method to plot the graph using matplotlib."""
        # Define the node positions in a grid layout
        pos = {i * self.m + j: (j, -i) for i in range(self.n) for j in range(self.m)}

        # Draw the graph
        nx.draw(self.graph, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=10)
        plt.show()
