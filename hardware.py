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
    time_1q: float = 50.0       # ns default single-qubit gate time
    time_2q: float = 300.0      # ns default two-qubit gate time
    time_meas: float = 4000.0   # ns measurement
    time_reset: float = 1000.0  # ns reset after measurement
    # optional per-node/edge overrides:
    time_1q_map: Dict[Any, float] = field(default_factory=dict)
    time_2q_map: Dict[Tuple[Any, Any], float] = field(default_factory=dict)
    time_meas_map: Dict[Any, float] = field(default_factory=dict)
    time_reset_map: Dict[Any, float] = field(default_factory=dict)

    def get_t1q(self, node):
        return self.time_1q_map.get(node, self.time_1q)

    def get_t2q(self, u, v):
        # symmetric lookup
        key = (u, v)
        if key in self.time_2q_map:
            return self.time_2q_map[key]
        key2 = (v, u)
        return self.time_2q_map.get(key2, self.time_2q)

    def get_tmeas(self, node):
        return self.time_meas_map.get(node, self.time_meas)

    def get_treset(self, node):
        return self.time_reset_map.get(node, self.time_reset)
