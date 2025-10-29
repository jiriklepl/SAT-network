"""3-bit adder dataset plugin."""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

from . import Example, IOList, register_plugin, DatasetResult


def _make_sloppy_adder_test_case(left: int, right: int) -> Tuple[IOList, IOList]:

    assert 0 <= left < 4
    assert 0 <= right < 4

    l0 = (left & 1) != 0
    l1 = (left & 2) != 0

    r0 = (right & 1) != 0
    r1 = (right & 2) != 0

    result = left + right

    inputs: IOList = [l1, l0, r1, r0]

    result_bit0: Optional[bool] = (result & 1) != 0
    result_bit1: Optional[bool] = (result & 2) != 0
    carry_bit: Optional[bool] = result >= 4

    if carry_bit:
        result_bit0 = None
        result_bit1 = None

    outputs: IOList = [carry_bit, result_bit1, result_bit0]

    return inputs, outputs


def _build_from_config(cfg: Dict[str, Any]) -> DatasetResult:
    num_inputs = int(cfg.get("num_inputs", 4))
    num_outputs = int(cfg.get("num_outputs", 3))

    sloppy_adder_cfg = cfg.get("sloppy_adder", {})
    left_range = int(sloppy_adder_cfg.get("left_range", 4))
    right_range = int(sloppy_adder_cfg.get("right_range", 4))

    examples: List[Example] = []
    for left in range(left_range):
        for right in range(right_range):
            ins, outs = _make_sloppy_adder_test_case(left, right)
            examples.append({"inputs": ins, "outputs": outs})

    return examples, num_inputs, num_outputs


register_plugin("sloppy-adder", _build_from_config)
