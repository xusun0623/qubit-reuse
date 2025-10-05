from dataclasses import dataclass, field
from typing import Dict, Tuple, Any


@dataclass
class HardwareParams:
    # 时间参数
    time_1q: float = 50.0       # 单比特门的耗时
    time_2q: float = 300.0      # 双比特门的耗时
    time_meas: float = 4000.0   # 测量的耗时
    time_reset: float = 1000.0  # 测量后的重置耗时

    time_1q_map: Dict[Any, float] = field(default_factory=dict)
    time_2q_map: Dict[Tuple[Any, Any], float] = field(default_factory=dict)
    time_meas_map: Dict[Any, float] = field(default_factory=dict)
    time_reset_map: Dict[Any, float] = field(default_factory=dict)

    def get_t1q(self, node):
        return self.time_1q_map.get(node, self.time_1q)

    def get_t2q(self, u, v):
        key = (u, v)
        if key in self.time_2q_map:
            return self.time_2q_map[key]
        key2 = (v, u)
        return self.time_2q_map.get(key2, self.time_2q)

    def get_tmeas(self, node):
        return self.time_meas_map.get(node, self.time_meas)

    def get_treset(self, node):
        return self.time_reset_map.get(node, self.time_reset)
