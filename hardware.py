"""
Hardware parameters for quantum computing devices.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any


@dataclass
class HardwareParams:
    """
    时间参数可以是：
      - 全局标量： t_1q, t_2q, t_meas, t_reset
      - 或者按节点/边字典覆盖： t_1q_map[node] 或 t_2q_map[(u,v)]
    使用 get_* 接口统一读取最终值。
    """
    t_1q: float = 50.0       # ns default single-qubit gate time
    t_2q: float = 300.0      # ns default two-qubit gate time
    t_meas: float = 4000.0   # ns measurement
    t_reset: float = 1000.0  # ns reset after measurement
    # optional per-node/edge overrides:
    t_1q_map: Dict[Any, float] = field(default_factory=dict)
    t_2q_map: Dict[Tuple[Any, Any], float] = field(default_factory=dict)
    t_meas_map: Dict[Any, float] = field(default_factory=dict)
    t_reset_map: Dict[Any, float] = field(default_factory=dict)

    def get_t1q(self, node):
        return self.t_1q_map.get(node, self.t_1q)

    def get_t2q(self, u, v):
        # symmetric lookup
        key = (u, v)
        if key in self.t_2q_map:
            return self.t_2q_map[key]
        key2 = (v, u)
        return self.t_2q_map.get(key2, self.t_2q)

    def get_tmeas(self, node):
        return self.t_meas_map.get(node, self.t_meas)

    def get_treset(self, node):
        return self.t_reset_map.get(node, self.t_reset)
