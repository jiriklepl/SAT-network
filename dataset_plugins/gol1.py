"""Game of Life dataset plugin."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from . import Example, IOList, register_plugin, DatasetResult


def _make_gol2_test_case(left_column: int, center_column: int, right_column: int, alive: bool) -> Tuple[IOList, IOList]:
    sum = left_column + center_column + right_column
    carry = (left_column % 2 + center_column % 2 + right_column % 2) // 2

    assert carry in (0, 1)
    assert sum <= 8

    inputs: IOList = [
        left_column % 2 != 0,
        left_column // 2 % 2 != 0,
        center_column % 2 != 0,
        center_column // 2 % 2 != 0,
        right_column % 2 != 0,
        right_column // 2 % 2 != 0,
        alive,
    ]

    outputs: IOList = [
        (sum | alive) % 2 == 1,
        carry == 1,
    ]
    return inputs, outputs


def _build_from_config(cfg: Dict[str, Any]) -> DatasetResult:
    num_inputs = 7
    num_outputs = 2

    left_range = 4
    center_range = 3
    right_range = 4

    examples: List[Example] = []
    for left in range(left_range):
        for center in range(center_range):
            for right in range(right_range):
                for alive in [True, False]:
                    ins, outs = _make_gol2_test_case(left, center, right, alive)
                    examples.append({"inputs": ins, "outputs": outs})

    return examples, num_inputs, num_outputs


register_plugin("gol1", _build_from_config)
