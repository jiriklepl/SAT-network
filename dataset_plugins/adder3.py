"""3-bit adder dataset plugin."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from . import register_plugin, DatasetResult


def _make_3_bits_adder_test_case(left_bit: bool, center_bit: bool, right_bit: bool) -> Tuple[List[bool], List[bool]]:
    inputs: List[bool] = [left_bit, center_bit, right_bit]

    left = 1 if left_bit else 0
    center = 1 if center_bit else 0
    right = 1 if right_bit else 0
    total = left + center + right

    outputs: List[bool] = [
        total % 2 != 0,        # least significant bit
        total // 2 % 2 != 0,   # carry bit
    ]
    return inputs, outputs


def _build_from_config(cfg: Dict[str, Any]) -> DatasetResult:
    num_inputs = int(cfg.get("num_inputs", 3))
    num_outputs = int(cfg.get("num_outputs", 2))

    examples: List[Dict[str, List[bool]]] = []
    for left in [True, False]:
        for center in [True, False]:
            for right in [True, False]:
                ins, outs = _make_3_bits_adder_test_case(left, center, right)
                examples.append({"inputs": ins, "outputs": outs})

    return examples, num_inputs, num_outputs


register_plugin("adder3", _build_from_config)
