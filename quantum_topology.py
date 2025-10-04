"""
Physical topology representation for quantum devices.
"""

import json
import math
from typing import Dict, Tuple, List, Optional, Any

import networkx as nx


class ChipTopology:
    """
    统一的拓扑表示（基于 networkx.Graph）。
    Node attributes:
       - 'type': 'data' | 'ancilla' (可选)
       - any other per-node attrs
    Edge attributes:
       - weight or other attrs
    提供生成常见拓扑函数与 JSON persist/load
    """

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

    @staticmethod
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

    @staticmethod
    def square_grid(width: int, height: int, start_index: int = 0, data_nodes=True):
        G = nx.Graph()
        def idx(x, y): return start_index + x + y*width
        for y in range(height):
            for x in range(width):
                G.add_node(idx(x, y), role='data' if data_nodes else 'node')
                if x < width - 1:
                    G.add_edge(idx(x, y), idx(x+1, y))
                if y < height - 1:
                    G.add_edge(idx(x, y), idx(x, y+1))
        return ChipTopology(G, name=f"square_{width}x{height}")

    @staticmethod
    def heavy_square(width: int, height: int):
        # Heavy-square: each data qubit connected via ancilla between horizontal/vertical edges
        # This function provides a simplified heavy-square generation: insert ancilla nodes on edges.
        base = ChipTopology.square_grid(width, height)
        G = nx.Graph()
        # copy data nodes
        for n, d in base.G.nodes(data=True):
            G.add_node(n, role='data')
        ancilla_idx = max(base.G.nodes()) + 1 if base.G.nodes() else 0
        for u, v in base.G.edges():
            anc = ancilla_idx
            ancilla_idx += 1
            G.add_node(anc, role='ancilla')
            G.add_edge(anc, u)
            G.add_edge(anc, v)
        return ChipTopology(G, name=f"heavy_square_{width}x{height}")

    @staticmethod
    def hex_lattice(radius: int = 2):
        # Create a hex-like lattice using networkx hexagonal_graph or manual construction.
        # For small radius, use networkx hexagonal_graph approximation.
        # networkx.generators.classic.hexagonal_lattice_graph may not exist in all versions; we create a small hex.
        G = nx.Graph()
        # quick approximate hex layout using axial coords
        idx = 0
        coords = {}
        for q in range(-radius, radius+1):
            r1 = max(-radius, -q-radius)
            r2 = min(radius, -q+radius)
            for r in range(r1, r2+1):
                G.add_node(idx, role='data', q=q, r=r)
                coords[(q, r)] = idx
                idx += 1
        for (q, r), i in coords.items():
            neighbors = [(q+1, r), (q-1, r), (q, r+1),
                         (q, r-1), (q+1, r-1), (q-1, r+1)]
            for nb in neighbors:
                if nb in coords:
                    G.add_edge(i, coords[nb])
        return ChipTopology(G, name=f"hex_radius{radius}")

    @staticmethod
    def heavy_hex(radius: int = 2):
        # simplified heavy-hex: build hex lattice then place ancilla on each edge similar to heavy-square
        base = ChipTopology.hex_lattice(radius)
        G = nx.Graph()
        for n, d in base.G.nodes(data=True):
            G.add_node(n, **d)
        ancilla_idx = max(base.G.nodes()) + 1 if base.G.nodes() else 0
        for u, v in base.G.edges():
            anc = ancilla_idx
            ancilla_idx += 1
            G.add_node(anc, role='ancilla')
            G.add_edge(anc, u)
            G.add_edge(anc, v)
        return ChipTopology(G, name=f"heavy_hex_radius{radius}")
