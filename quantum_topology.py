import json
import math
from typing import Optional
import networkx as nx


class ChipTopology:
    def __init__(self, graph: Optional[nx.Graph] = None, name: str = "custom"):
        self.G = graph if graph is not None else nx.Graph()
        self.name = name

    def nodes(self):
        return list(self.G.nodes())

    def edges(self):
        return list(self.G.edges())

    def save_json(self, path: str):
        data = {
            "name": self.name,
            "nodes": [],
            "edges": []
        }
        for n, attr in self.G.nodes(data=True):
            data["nodes"].append({"id": n, "attr": attr})
        for u, v, attr in self.G.edges(data=True):
            data["edges"].append({"u": u, "v": v, "attr": attr})
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(path: str):
        with open(path, "r") as f:
            data = json.load(f)
        G = nx.Graph()
        for node in data["nodes"]:
            G.add_node(node["id"], **node.get("attr", {}))
        for e in data["edges"]:
            G.add_edge(e["u"], e["v"], **e.get("attr", {}))
        topo = ChipTopology(G, name=data.get("name", "loaded"))
        return topo

    def square_grid(width: int, height: int, start_index: int = 0, data_nodes=True):
        G = nx.Graph()
        def idx(x, y): return start_index + x + y*width
        for y in range(height):
            for x in range(width):
                G.add_node(idx(x, y))
                if x < width - 1:
                    G.add_edge(idx(x, y), idx(x+1, y))
                if y < height - 1:
                    G.add_edge(idx(x, y), idx(x, y+1))
        return ChipTopology(G, name=f"square_{width}x{height}")

    def heavy_square(width: int, height: int):
        base = ChipTopology.square_grid(width, height)
        G = nx.Graph()
        for n, d in base.G.nodes(data=True):
            G.add_node(n)
        ancilla_idx = max(base.G.nodes()) + 1 if base.G.nodes() else 0
        for u, v in base.G.edges():
            anc = ancilla_idx
            ancilla_idx += 1
            G.add_node(anc)
            G.add_edge(anc, u)
            G.add_edge(anc, v)
        return ChipTopology(G, name=f"heavy_square_{width}x{height}")

    def hex_lattice(radius: int = 2):
        G = nx.Graph()
        idx = 0
        coords = {}
        for q in range(-radius, radius+1):
            r1 = max(-radius, -q-radius)
            r2 = min(radius, -q+radius)
            for r in range(r1, r2+1):
                G.add_node(idx, q=q, r=r)
                coords[(q, r)] = idx
                idx += 1
        for (q, r), i in coords.items():
            neighbors = [(q+1, r), (q-1, r), (q, r+1),
                         (q, r-1), (q+1, r-1), (q-1, r+1)]
            for nb in neighbors:
                if nb in coords:
                    G.add_edge(i, coords[nb])
        return ChipTopology(G, name=f"hex_radius{radius}")

    def heavy_hex(radius: int = 2):
        base = ChipTopology.hex_lattice(radius)
        G = nx.Graph()
        for n, d in base.G.nodes(data=True):
            G.add_node(n, **d)
        ancilla_idx = max(base.G.nodes()) + 1 if base.G.nodes() else 0
        for u, v in base.G.edges():
            anc = ancilla_idx
            ancilla_idx += 1
            G.add_node(anc)
            G.add_edge(anc, u)
            G.add_edge(anc, v)
        return ChipTopology(G, name=f"heavy_hex_radius{radius}")

    def visualize(self, figsize=(8, 8), node_size=300, font_size=8, with_labels=True):
        """可视化芯片拓扑结构"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)
        nx.draw_networkx_nodes(
            self.G, pos, node_size=node_size, node_color='lightblue')
        if with_labels:
            nx.draw_networkx_labels(self.G, pos, font_size=font_size)
        plt.title(f"Chip Topology: {self.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
