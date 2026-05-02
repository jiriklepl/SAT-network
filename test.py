#!/usr/bin/env python3

import io
import unittest
from contextlib import redirect_stdout

from main import (
    OP_BY_LABEL,
    EncodingOptions,
    ProgramSpec,
    _build_assumptions_from_file,
    _build_dataset_from_config,
    _build_program,
    _build_test,
    _emit_program,
    _operator_sort_key,
    _post_process_program,
)
from build_assumptions import build_assumption_lines
from dataset_plugins import available_plugins, get_plugin, get_plugin_config


def _bits(value: int, width: int) -> list[bool]:
    return [(value & (1 << bit)) != 0 for bit in range(width)]


def _state_bits(states: list[int], width: int) -> list[bool]:
    result: list[bool] = []
    for state in states:
        result.extend(_bits(state, width))
    return result


def _eval_assumption_lines(lines: list[str], inputs: list[bool], num_outputs: int) -> list[bool]:
    values: dict[str, bool] = {"1": True}
    values.update({f"I{idx}": value for idx, value in enumerate(inputs)})
    outputs: list[bool | None] = [None] * num_outputs

    for line in lines:
        if not line or line.startswith("#"):
            continue
        lhs, rhs = [part.strip() for part in line.split(":", 1)]
        if lhs.startswith("OUT"):
            outputs[int(lhs[3:])] = values[rhs]
            continue

        op, args_part = rhs.split("(", 1)
        left_name, right_name = [arg.strip() for arg in args_part.rstrip(")").split(",", 1)]
        left = values[left_name]
        right = values[right_name]
        if op == "AND":
            values[lhs] = left and right
        elif op == "OR":
            values[lhs] = left or right
        elif op == "XOR":
            values[lhs] = left != right
        else:
            raise AssertionError(f"Unexpected operator: {op}")

    if any(output is None for output in outputs):
        raise AssertionError("Missing output in assumption lines")
    return [bool(output) for output in outputs]


class MainEncodingTests(unittest.TestCase):
    def test_explicit_examples_preserve_output_dont_cares(self) -> None:
        examples, num_inputs, num_outputs, instructions = _build_dataset_from_config({
            "num_inputs": 1,
            "num_outputs": 1,
            "instructions": 0,
            "examples": [{"inputs": [False], "outputs": [None]}],
        })

        self.assertEqual(num_inputs, 1)
        self.assertEqual(num_outputs, 1)
        self.assertEqual(instructions, 0)
        self.assertIsNone(examples[0]["outputs"][0])

    def test_assumptions_reject_out_of_range_inputs(self) -> None:
        spec = ProgramSpec(num_inputs=2, num_outputs=1, program_length=1)

        with self.assertRaisesRegex(ValueError, "Input index out of range"):
            _build_assumptions_from_file(io.StringIO("T0: AND(I2, I0)\n"), spec)

    def test_assumptions_reject_out_of_range_temporaries_and_outputs(self) -> None:
        spec = ProgramSpec(num_inputs=2, num_outputs=1, program_length=1)

        with self.assertRaisesRegex(ValueError, "Temporary index out of range"):
            _build_assumptions_from_file(io.StringIO("T1: AND(I0, I1)\n"), spec)

        with self.assertRaisesRegex(ValueError, "Output index out of range"):
            _build_assumptions_from_file(io.StringIO("OUT1: I0\n"), spec)

    def test_duplicate_xor_emits_blif_constant_zero(self) -> None:
        buf = io.StringIO()
        xor = OP_BY_LABEL["XOR"].code

        with redirect_stdout(buf):
            _emit_program(
                instrs=[(xor, 1, 1)],
                outputs=[2],
                num_inputs=1,
                num_outputs=1,
                output_blif=True,
            )

        self.assertIn(".names T0\n.names T0 OUT0", buf.getvalue())

    def test_operator_sort_key_is_explicit(self) -> None:
        self.assertLess(_operator_sort_key(OP_BY_LABEL["XOR"].code), _operator_sort_key(OP_BY_LABEL["AND"].code))
        self.assertLess(_operator_sort_key(OP_BY_LABEL["AND"].code), _operator_sort_key(OP_BY_LABEL["OR"].code))

    def test_post_process_does_not_mutate_output_selectors(self) -> None:
        xor = OP_BY_LABEL["XOR"].code
        examples = [
            {"inputs": [False, False], "outputs": [False]},
            {"inputs": [False, True], "outputs": [True]},
            {"inputs": [True, False], "outputs": [True]},
            {"inputs": [True, True], "outputs": [False]},
        ]
        outputs = [3]

        instrs, processed_outputs = _post_process_program(
            [(xor, 1, 2)],
            num_inputs=2,
            num_outputs=1,
            examples=examples,
            outputs=outputs,
        )

        self.assertEqual(instrs, [(xor, 1, 2)])
        self.assertEqual(processed_outputs, [3])
        self.assertEqual(outputs, [3])

    def test_post_process_rejects_invalid_program_instead_of_asserting(self) -> None:
        and_op = OP_BY_LABEL["AND"].code
        examples = [{"inputs": [True], "outputs": [False]}]

        with self.assertRaisesRegex(ValueError, "incorrect program"):
            _post_process_program(
                [(and_op, 0, 1)],
                num_inputs=1,
                num_outputs=1,
                examples=examples,
                outputs=[2],
            )

    def test_post_process_afterburner_accepts_only_final_size_reductions(self) -> None:
        xor = OP_BY_LABEL["XOR"].code
        or_op = OP_BY_LABEL["OR"].code
        examples = [
            {"inputs": [False, False], "outputs": [False]},
            {"inputs": [False, True], "outputs": [True]},
            {"inputs": [True, False], "outputs": [True]},
            {"inputs": [True, True], "outputs": [False]},
        ]

        instrs, outputs = _post_process_program(
            [
                (xor, 1, 2),
                (xor, 1, 2),
                (or_op, 3, 4),
            ],
            num_inputs=2,
            num_outputs=1,
            examples=examples,
            outputs=[5],
        )

        self.assertEqual(instrs, [(xor, 1, 2)])
        self.assertEqual(outputs, [3])

    def test_boolean_encoding_uses_output_selector_guards(self) -> None:
        spec = ProgramSpec(num_inputs=1, num_outputs=1, program_length=1)
        options = EncodingOptions(encode_boolean=True)

        structure_constraints = _build_program(spec, options)
        batch_constraints, outputs = _build_test(1, [0], "b0", spec, options)

        structure_text = "\n".join(str(c) for c in structure_constraints)
        batch_text = "\n".join(str(c) for c in batch_constraints)

        self.assertIn("OUT_0_eq_0", structure_text)
        self.assertIn("OUTVAL_b0_0", str(outputs[0]))
        self.assertIn("OUT_0_eq_0", batch_text)

    def test_build_assumptions_dnf_matches_examples_and_parser(self) -> None:
        examples = [
            {"inputs": [False, False], "outputs": [False]},
            {"inputs": [False, True], "outputs": [True]},
            {"inputs": [True, False], "outputs": [True]},
            {"inputs": [True, True], "outputs": [False]},
        ]
        lines, required_instructions = build_assumption_lines(examples, 2, 1)

        spec = ProgramSpec(num_inputs=2, num_outputs=1, program_length=required_instructions)
        constraints = _build_assumptions_from_file(io.StringIO("\n".join(lines)), spec)

        self.assertGreater(required_instructions, 0)
        self.assertGreater(len(constraints), 0)
        for ex in examples:
            self.assertEqual(_eval_assumption_lines(lines, ex["inputs"], 1), ex["outputs"])

    def test_build_assumptions_auto_uses_compact_life_circuit(self) -> None:
        examples, num_inputs, num_outputs = get_plugin("life-compressed")(get_plugin_config("life-compressed"))
        lines, required_instructions = build_assumption_lines(
            examples,
            num_inputs,
            num_outputs,
            dataset_name="life-compressed",
            strategy="auto",
        )

        spec = ProgramSpec(num_inputs=num_inputs, num_outputs=num_outputs, program_length=required_instructions)
        constraints = _build_assumptions_from_file(io.StringIO("\n".join(lines)), spec)

        self.assertEqual(required_instructions, 14)
        self.assertEqual(lines[-1], "OUT0: T13")
        self.assertGreater(len(constraints), 0)
        for ex in examples:
            self.assertEqual(_eval_assumption_lines(lines, ex["inputs"], num_outputs), ex["outputs"])

    def test_cellular_automata_plugins_are_registered(self) -> None:
        plugins = available_plugins()
        for name in [
            "life",
            "life-compressed",
            "maze",
            "maze-compressed",
            "brian",
            "brian-compressed",
            "fire",
            "fire-compressed",
            "wire",
            "wire-compressed",
            "excitable",
            "excitable-compressed",
            "cyclic",
            "cyclic-compressed",
            "fluid",
            "critters",
            "traffic",
        ]:
            self.assertIn(name, plugins)
            examples, num_inputs, num_outputs = get_plugin(name)(get_plugin_config(name))
            self.assertGreater(len(examples), 0)
            self.assertGreater(num_inputs, 0)
            self.assertGreater(num_outputs, 0)

    def test_life_raw_plugin_birth_rule(self) -> None:
        examples, _, _ = get_plugin("life")(get_plugin_config("life"))
        by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in examples}

        states = [0, 1, 1, 1, 0, 0, 0, 0, 0]
        self.assertEqual(by_input[tuple(_state_bits(states, 1))], [True])

    def test_life_compressed_birth_rule(self) -> None:
        examples, num_inputs, num_outputs = get_plugin("life-compressed")(get_plugin_config("life-compressed"))
        by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in examples}

        self.assertEqual((len(examples), num_inputs, num_outputs), (96, 7, 1))
        self.assertEqual(by_input[tuple(_bits(1, 2) + _bits(1, 2) + _bits(1, 2) + [False])], [True])

    def test_fire_plugin_spread_and_ash_rules(self) -> None:
        examples, _, _ = get_plugin("fire")(get_plugin_config("fire"))
        by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in examples}

        tree_with_fire_neighbor = [1, 2, 0, 0, 0]
        ash_without_fire_neighbor = [3, 0, 0, 0, 0]

        self.assertEqual(by_input[tuple(_state_bits(tree_with_fire_neighbor, 2))], _bits(2, 2))
        self.assertEqual(by_input[tuple(_state_bits(ash_without_fire_neighbor, 2))], _bits(0, 2))

    def test_fire_compressed_spread_and_ash_rules(self) -> None:
        examples, num_inputs, num_outputs = get_plugin("fire-compressed")(get_plugin_config("fire-compressed"))
        by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in examples}

        self.assertEqual((len(examples), num_inputs, num_outputs), (48, 6, 2))
        self.assertEqual(by_input[tuple(_bits(1, 1) + _bits(0, 2) + _bits(0, 1) + _bits(1, 2))], _bits(2, 2))
        self.assertEqual(by_input[tuple(_bits(0, 1) + _bits(0, 2) + _bits(0, 1) + _bits(3, 2))], _bits(0, 2))

    def test_wire_compressed_head_count_rule(self) -> None:
        examples, _, _ = get_plugin("wire-compressed")(get_plugin_config("wire-compressed"))
        by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in examples}

        self.assertEqual(by_input[tuple(_bits(1, 2) + _bits(0, 2) + _bits(1, 2) + _bits(1, 2))], _bits(2, 2))
        self.assertEqual(by_input[tuple(_bits(1, 2) + _bits(1, 2) + _bits(1, 2) + _bits(1, 2))], _bits(1, 2))

    def test_excitable_and_cyclic_compressed_predicates(self) -> None:
        excitable_examples, excitable_inputs, excitable_outputs = get_plugin("excitable-compressed")(get_plugin_config("excitable-compressed"))
        cyclic_examples, cyclic_inputs, cyclic_outputs = get_plugin("cyclic-compressed")(get_plugin_config("cyclic-compressed"))

        excitable_by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in excitable_examples}
        cyclic_by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in cyclic_examples}

        self.assertEqual((len(excitable_examples), excitable_inputs, excitable_outputs), (384, 9, 3))
        self.assertEqual(excitable_by_input[tuple(_bits(1, 2) + _bits(0, 2) + _bits(0, 2) + _bits(0, 3))], _bits(1, 3))

        self.assertEqual((len(cyclic_examples), cyclic_inputs, cyclic_outputs), (1536, 11, 5))
        self.assertEqual(cyclic_by_input[tuple(_bits(1, 2) + _bits(0, 2) + _bits(0, 2) + _bits(31, 5))], _bits(0, 5))

    def test_critters_plugin_block_rules(self) -> None:
        examples, _, _ = get_plugin("critters")(get_plugin_config("critters"))
        by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in examples}

        self.assertEqual(by_input[(True, True, False, False)], [True, True, False, False])
        self.assertEqual(by_input[(True, True, True, False)], [True, False, False, False])

    def test_traffic_plugin_right_moving_substep(self) -> None:
        examples, _, _ = get_plugin("traffic")(get_plugin_config("traffic"))
        by_input = {tuple(ex["inputs"]): ex["outputs"] for ex in examples}

        right_phase = False
        right_car = 1
        empty = 0

        incoming_right_car = [right_phase] + _state_bits([right_car, empty, empty], 2)
        leaving_right_car = [right_phase] + _state_bits([empty, right_car, empty], 2)

        self.assertEqual(by_input[tuple(incoming_right_car)], _bits(right_car, 2))
        self.assertEqual(by_input[tuple(leaving_right_car)], _bits(empty, 2))


if __name__ == "__main__":
    unittest.main()
