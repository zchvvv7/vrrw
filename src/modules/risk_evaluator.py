"""
文件名: risk_evaluator.py
用途: 根据障碍物距离和位置输出风险等级
作者:
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

from typing import List
from typing import Tuple

from src.interface.schemas import DetectedObject
from src.interface.schemas import UnknownRegion


class RiskEvaluator:
    """负责输出 safe、notice、warning、danger 风险等级"""

    def __init__(self, warning_distance: float, danger_distance: float):
        self.warning_distance = warning_distance
        self.danger_distance = danger_distance

    # TODO: 判断当前帧风险等级，以下为占位代码
    def evaluate(
        self,
        known_objects: List[DetectedObject],
        unknown_regions: List[UnknownRegion],
    ) -> Tuple[str, str]:
        return "safe", "no_obstacle"