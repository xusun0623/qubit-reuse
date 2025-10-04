import json
import math
from typing import Optional
from matplotlib import pyplot as plt
import networkx as nx


class QuantumChip:
    def __init__(self, chip_type: str = "square", size: int = 8):
        self.graph = nx.Graph()
        self.chip_type = chip_type
        self.size = size
        if (chip_type == "square"):
            self.build_square_chip(size)
        if (chip_type == "hexagon"):
            self.build_hexagon_chip(size)
        if (chip_type == "heavy_square"):
            self.build_heavy_square_chip(size)
        if (chip_type == "heavy_hexagon"):
            self.build_heavy_hexagon_chip(size)

    def nodes(self):
        return list(self.graph.nodes())

    def edges(self):
        return list(self.graph.edges())

    def build_square_chip(self, size):
        self.graph = nx.grid_2d_graph(size, size)
        for local_idx, node in enumerate(self.graph.nodes()):
            unique_id = local_idx
            self.graph.nodes[node]['id'] = unique_id

    def build_hexagon_chip(self, size):
        width, height = size, size
        self.graph = nx.hexagonal_lattice_graph(width, height)
        for local_idx, node in enumerate(self.graph.nodes()):
            unique_id = local_idx
            self.graph.nodes[node]['id'] = unique_id

    def build_heavy_square_chip(self, size):
        G = nx.grid_2d_graph(size, size)
        mid_edge_nodes = []
        edges_list = list(G.edges())
        for (u, v) in edges_list:
            x1, y1 = u
            x2, y2 = v
            mid_node = ((x1 + x2)/2, (y1 + y2)/2)
            mid_edge_nodes.append(mid_node)
            G.add_node(mid_node)
            G.add_edge(u, mid_node)
            G.add_edge(v, mid_node)

        self.graph = G
        for local_idx, node in enumerate(self.graph.nodes()):
            unique_id = local_idx
            self.graph.nodes[node]['id'] = unique_id

    def build_heavy_hexagon_chip(self, size):
        width, height = size, size
        G = nx.hexagonal_lattice_graph(width, height)
        edges_list = list(G.edges())
        for (u, v) in edges_list:
            x1, y1 = u
            x2, y2 = v
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            mid_node = (mid_x, mid_y)
            G.remove_edge(u, v)
            G.add_node(mid_node)
            G.add_edge(u, mid_node)
            G.add_edge(mid_node, v)
        self.graph = G
        for local_idx, node in enumerate(self.graph.nodes()):
            unique_id = local_idx
            self.graph.nodes[node]['id'] = unique_id

    def visualize(self):
        chip_type = self.chip_type
        size = self.size
        figsize_width = 6
        if (chip_type == "square"):
            figsize_width = size
        elif (chip_type == "hexagon"):
            figsize_width = size * 2
        elif (chip_type == "heavy_square"):
            figsize_width = size * 2
        elif (chip_type == "heavy_square"):
            figsize_width = size * 4
        plt.figure(figsize=(figsize_width, figsize_width))
        pos = {(x, y): (y, -x) for (x, y) in self.graph.nodes()}
        labels = {}
        for node in self.graph.nodes():
            x, y = node
            if x == int(x) and y == int(y):
                labels[node] = str(node)
        nx.draw(self.graph, pos=pos, with_labels=False,
                node_size=50, font_size=8)
        nx.draw_networkx_labels(self.graph, pos=pos,
                                labels=labels, font_size=8)
        plt.title(f"{self.chip_type} Graph")
        plt.show()
