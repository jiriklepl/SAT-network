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
)


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
        self.assertLess(_operator_sort_key(OP_BY_LABEL["AND"].code), _operator_sort_key(OP_BY_LABEL["XOR"].code))
        self.assertLess(_operator_sort_key(OP_BY_LABEL["XOR"].code), _operator_sort_key(OP_BY_LABEL["OR"].code))

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


if __name__ == "__main__":
    unittest.main()
