"""Game of Life dataset plugin."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from . import Example, IOList, register_plugin, DatasetResult


def _make_gol_test_case(left_column: int, center_column: int, right_column: int, alive: bool) -> Tuple[IOList, IOList]:
    inputs: IOList = [
        left_column % 2 != 0,
        left_column // 2 % 2 != 0,
        center_column % 2 != 0,
        center_column // 2 % 2 != 0,
        right_column % 2 != 0,
        right_column // 2 % 2 != 0,
        alive,
    ]

    sum = left_column + center_column + right_column

    outputs: IOList = [
        sum == 3 or (sum == 2 and alive)
    ]
    return inputs, outputs


def _build_from_config(cfg: Dict[str, Any]) -> DatasetResult:
    num_inputs = 7
    num_outputs = 1

    gol_cfg = cfg.get("gol", {})
    left_range = int(gol_cfg.get("left_range", 4))
    center_range = int(gol_cfg.get("center_range", 3))
    right_range = int(gol_cfg.get("right_range", 4))
    include_alive = bool(gol_cfg.get("include_alive", True))

    examples: List[Example] = []
    for left in range(left_range):
        for center in range(center_range):
            for right in range(right_range):
                alive_values = [True, False] if include_alive else [False]
                for alive in alive_values:
                    ins, outs = _make_gol_test_case(left, center, right, alive)
                    examples.append({"inputs": ins, "outputs": outs})

    return examples, num_inputs, num_outputs


register_plugin("gol", _build_from_config)
