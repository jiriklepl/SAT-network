"""3-bit adder dataset plugin."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from . import Example, IOList, register_plugin, DatasetResult


def _make_bit_adder_test_case(bits: List[bool], num_outputs: int) -> Tuple[IOList, IOList]:
    inputs: IOList = [bit for bit in bits]

    total = sum(bits)

    outputs: IOList = []

    for i in range(num_outputs):
        outputs.append((total & (1 << i)) != 0)

    return inputs, outputs


def _build_from_config(cfg: Dict[str, Any]) -> DatasetResult:
    num_inputs = int(cfg.get("num_inputs", 3))
    num_outputs = 0

    while (1 << num_outputs) < num_inputs + 1:
        num_outputs += 1

    examples: List[Example] = []
    for i in range(2**num_inputs):
        bits = [(i & (1 << j)) != 0 for j in range(num_inputs)]
        inputs, outputs = _make_bit_adder_test_case(bits, num_outputs)
        examples.append({"inputs": inputs, "outputs": outputs})

    return examples, num_inputs, num_outputs


register_plugin("adder", _build_from_config)
