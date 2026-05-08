#!/usr/bin/env python3

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from dataset_plugins import get_plugin, get_plugin_config
from main import OP_BY_LABEL, _verify_program


def _cpp_binary() -> Path | None:
    env_path = os.environ.get("SAT_SYNTH_CPP")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        Path("build/sat_synth_cpp"),
        Path("cmake-build-debug/sat_synth_cpp"),
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _parse_source(raw: str, num_inputs: int) -> int:
    if raw == "1":
        return 0
    if raw.startswith("I"):
        return int(raw[1:]) + 1
    if raw.startswith("T"):
        return num_inputs + 1 + int(raw[1:])
    raise AssertionError(f"Unexpected source: {raw}")


def _parse_program(text: str, num_inputs: int, num_outputs: int):
    instrs = []
    outputs = [None] * num_outputs
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lhs, rhs = [part.strip() for part in line.split(":", 1)]
        if lhs.startswith("T"):
            op, args = rhs.split("(", 1)
            left, right = [part.strip() for part in args.rstrip(")").split(",", 1)]
            instrs.append((OP_BY_LABEL[op].code, _parse_source(left, num_inputs), _parse_source(right, num_inputs)))
        elif lhs.startswith("OUT"):
            outputs[int(lhs[3:])] = _parse_source(rhs, num_inputs)
    if any(output is None for output in outputs):
        raise AssertionError(f"Missing outputs in program:\n{text}")
    return instrs, outputs


@unittest.skipIf(_cpp_binary() is None, "sat_synth_cpp binary is not built; set SAT_SYNTH_CPP or build/sat_synth_cpp")
class CppSolverIntegrationTests(unittest.TestCase):
    def run_cpp_args(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        binary = _cpp_binary()
        assert binary is not None
        return subprocess.run(
            [str(binary), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_cpp_args_with_input(self, args: list[str], input_text: str) -> subprocess.CompletedProcess[str]:
        binary = _cpp_binary()
        assert binary is not None
        return subprocess.run(
            [str(binary), *args],
            input=input_text,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_cpp(self, config: dict, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as file:
            json.dump(config, file)
            file.flush()
            return self.run_cpp_args(["--config", file.name, *args])

    def assert_cpp_solves(self, config: dict, *args: str) -> None:
        proc = self.run_cpp(config, *args)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        instrs, outputs = _parse_program(proc.stdout, config["num_inputs"], config["num_outputs"])
        examples = [
            {
                "inputs": [bool(value) for value in ex["inputs"]],
                "outputs": [None if value is None else bool(value) for value in ex["outputs"]],
            }
            for ex in config["examples"]
        ]
        self.assertEqual(_verify_program(instrs, outputs, examples, config["num_outputs"]), [])

    def assert_cpp_solves_examples(
        self,
        proc: subprocess.CompletedProcess[str],
        examples: list[dict],
        num_inputs: int,
        num_outputs: int,
    ) -> None:
        self.assertEqual(proc.returncode, 0, proc.stderr)
        instrs, outputs = _parse_program(proc.stdout, num_inputs, num_outputs)
        self.assertEqual(_verify_program(instrs, outputs, examples, num_outputs), [])

    def dump_cpp_dataset(self, config: dict) -> dict:
        proc = self.run_cpp(config, "--dump-dataset")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def assert_cpp_rejects_dataset(self, config: dict, expected_error: str) -> None:
        proc = self.run_cpp(config, "--dump-dataset")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(expected_error, proc.stderr)

    def assert_cpp_dataset_matches_python(self, config: dict) -> None:
        cpp_dataset = self.dump_cpp_dataset(config)
        examples, num_inputs, num_outputs = get_plugin(config["type"])(config)
        self.assertEqual(cpp_dataset["num_inputs"], num_inputs)
        self.assertEqual(cpp_dataset["num_outputs"], num_outputs)
        self.assertEqual(cpp_dataset["examples"], examples)

    def test_xor_with_cegis(self) -> None:
        config = {
            "num_inputs": 2,
            "num_outputs": 1,
            "instructions": 1,
            "examples": [
                {"inputs": [False, False], "outputs": [False]},
                {"inputs": [False, True], "outputs": [True]},
                {"inputs": [True, False], "outputs": [True]},
                {"inputs": [True, True], "outputs": [False]},
            ],
        }
        self.assert_cpp_solves(config, "--cegis", "--cegis-initial-size", "1", "--cegis-counterexamples", "1", "--no-shuffle")

    def test_dont_care_output(self) -> None:
        config = {
            "num_inputs": 1,
            "num_outputs": 1,
            "instructions": 0,
            "examples": [{"inputs": [False], "outputs": [None]}],
        }
        self.assert_cpp_solves(config)

    def test_encoding_flags(self) -> None:
        config = {
            "num_inputs": 2,
            "num_outputs": 1,
            "instructions": 1,
            "examples": [
                {"inputs": [False, False], "outputs": [False]},
                {"inputs": [False, True], "outputs": [True]},
                {"inputs": [True, False], "outputs": [True]},
                {"inputs": [True, True], "outputs": [True]},
            ],
        }
        self.assert_cpp_solves(config, "--force-ordered")
        self.assert_cpp_solves(config, "--force-useful")
        self.assert_cpp_solves(config, "--encode-boolean")
        self.assert_cpp_solves(config, "--balanced-select")

    def test_unsat_returns_nonzero(self) -> None:
        config = {
            "num_inputs": 1,
            "num_outputs": 1,
            "instructions": 0,
            "examples": [
                {"inputs": [False], "outputs": [False]},
                {"inputs": [False], "outputs": [True]},
            ],
        }
        proc = self.run_cpp(config)
        self.assertNotEqual(proc.returncode, 0)

    def test_generated_adder_config(self) -> None:
        config = {"type": "adder", "num_inputs": 2, "instructions": 2}
        examples, num_inputs, num_outputs = get_plugin("adder")(config)
        proc = self.run_cpp(config, "--no-shuffle")
        self.assert_cpp_solves_examples(proc, examples, num_inputs, num_outputs)

    def test_assume_file_solves_xor(self) -> None:
        config = {
            "num_inputs": 2,
            "num_outputs": 1,
            "instructions": 1,
            "examples": [
                {"inputs": [False, False], "outputs": [False]},
                {"inputs": [False, True], "outputs": [True]},
                {"inputs": [True, False], "outputs": [True]},
                {"inputs": [True, True], "outputs": [False]},
            ],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as assume_file:
            assume_file.write("# fixed xor\nT0: XOR(I0, I1)\nOUT0: T0\n")
            assume_file.flush()
            self.assert_cpp_solves(config, "--assume", assume_file.name)

    def test_assume_stdin_solves_xor(self) -> None:
        config = {
            "num_inputs": 2,
            "num_outputs": 1,
            "instructions": 1,
            "examples": [
                {"inputs": [False, False], "outputs": [False]},
                {"inputs": [False, True], "outputs": [True]},
                {"inputs": [True, False], "outputs": [True]},
                {"inputs": [True, True], "outputs": [False]},
            ],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as file:
            json.dump(config, file)
            file.flush()
            proc = self.run_cpp_args_with_input(
                ["--config", file.name, "--assume", "-"],
                "T0: XOR(I0, I1)\nOUT0: T0\n",
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        instrs, outputs = _parse_program(proc.stdout, config["num_inputs"], config["num_outputs"])
        self.assertEqual(_verify_program(instrs, outputs, config["examples"], config["num_outputs"]), [])

    def test_contradictory_assumptions_return_nonzero(self) -> None:
        config = {
            "num_inputs": 2,
            "num_outputs": 1,
            "instructions": 1,
            "examples": [
                {"inputs": [False, False], "outputs": [False]},
                {"inputs": [False, True], "outputs": [True]},
                {"inputs": [True, False], "outputs": [True]},
                {"inputs": [True, True], "outputs": [False]},
            ],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as assume_file:
            assume_file.write("T0: AND(I0, I0)\nOUT0: T0\n")
            assume_file.flush()
            proc = self.run_cpp(config, "--assume", assume_file.name)
        self.assertNotEqual(proc.returncode, 0)

    def test_missing_assume_file_returns_error(self) -> None:
        config = {
            "num_inputs": 1,
            "num_outputs": 1,
            "instructions": 0,
            "examples": [{"inputs": [False], "outputs": [False]}],
        }
        proc = self.run_cpp(config, "--assume", "/tmp/missing-sat-synth-assume.txt")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Assume file not found", proc.stderr)

    def test_make_smt2_outputs_smtlib(self) -> None:
        config = {
            "num_inputs": 2,
            "num_outputs": 1,
            "instructions": 1,
            "examples": [
                {"inputs": [False, False], "outputs": [False]},
                {"inputs": [False, True], "outputs": [True]},
                {"inputs": [True, False], "outputs": [True]},
                {"inputs": [True, True], "outputs": [False]},
            ],
        }
        proc = self.run_cpp(config, "--make-smt2", "--batch-size", "2", "--no-shuffle")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OP_0", proc.stdout)
        self.assertIn("OUT_0_idx", proc.stdout)
        self.assertIn("assert", proc.stdout)
        self.assertIn("b0", proc.stdout)
        self.assertIn("b1", proc.stdout)
        self.assertNotIn("T0:", proc.stdout)

    def test_make_smt2_accepts_assumptions(self) -> None:
        config = {
            "num_inputs": 2,
            "num_outputs": 1,
            "instructions": 1,
            "examples": [
                {"inputs": [False, False], "outputs": [False]},
                {"inputs": [False, True], "outputs": [True]},
                {"inputs": [True, False], "outputs": [True]},
                {"inputs": [True, True], "outputs": [False]},
            ],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as assume_file:
            assume_file.write("T0: XOR(I0, I1)\nOUT0: T0\n")
            assume_file.flush()
            proc = self.run_cpp(config, "--make-smt2", "--assume", assume_file.name)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OP_0", proc.stdout)
        self.assertIn("OUT_0_idx", proc.stdout)

    def test_cegis_make_smt2_is_rejected(self) -> None:
        config = {
            "num_inputs": 1,
            "num_outputs": 1,
            "instructions": 0,
            "examples": [{"inputs": [False], "outputs": [False]}],
        }
        proc = self.run_cpp(config, "--cegis", "--make-smt2")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--cegis cannot be combined with --make-smt2", proc.stderr)

    def test_dataset_flag_uses_builtin_config(self) -> None:
        cfg = get_plugin_config("adder")
        examples, num_inputs, num_outputs = get_plugin("adder")(cfg)
        proc = self.run_cpp_args(["--dataset", "adder", "--cegis", "--cegis-initial-size", "2", "--cegis-counterexamples", "2", "--no-shuffle"])
        self.assert_cpp_solves_examples(proc, examples, num_inputs, num_outputs)

    def test_generated_datasets_match_python_plugins(self) -> None:
        for name in [
            "adder",
            "gol",
            "gol1",
            "gol2",
            "sloppy-adder",
            "sloppy-adder3",
            "life",
            "life-compressed",
            "maze",
            "maze-compressed",
            "brian",
            "brian-compressed",
            "fire",
            "fire-compressed",
            "wire-compressed",
            "excitable-compressed",
            "cyclic-compressed",
            "critters",
            "traffic",
        ]:
            with self.subTest(name=name):
                self.assert_cpp_dataset_matches_python(get_plugin_config(name))

    def test_compact_generated_datasets_match_python_plugins(self) -> None:
        configs = [
            {
                "type": "wire",
                "instructions": 24,
                "wire": {"max_examples": 4**9},
            },
            {
                "type": "fluid",
                "instructions": 18,
                "fluid": {"max_examples": 16**4},
            },
            {
                "type": "excitable",
                "instructions": 32,
                "excitable": {"states": 3},
            },
            {
                "type": "cyclic",
                "instructions": 48,
                "cyclic": {"states": 3},
            },
            {
                "type": "excitable-compressed",
                "instructions": 12,
                "excitable-compressed": {"states": 3},
            },
            {
                "type": "cyclic-compressed",
                "instructions": 18,
                "cyclic-compressed": {"states": 3},
            },
        ]
        for config in configs:
            with self.subTest(name=config["type"]):
                self.assert_cpp_dataset_matches_python(config)

    def test_sampled_state_space_configs_are_rejected(self) -> None:
        for config in [
            {"type": "life", "life": {"max_examples": 4, "seed": 0}},
            {"type": "excitable", "excitable": {"states": 3, "max_examples": 4, "seed": 0}},
            {"type": "fluid", "fluid": {"max_examples": 4, "seed": 0}},
        ]:
            with self.subTest(name=config["type"]):
                self.assert_cpp_rejects_dataset(config, "max_examples sampling is not supported")

    def test_list_datasets_includes_python_builtins(self) -> None:
        proc = self.run_cpp_args(["--list-datasets"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        names = set(proc.stdout.splitlines())
        self.assertIn("adder", names)
        self.assertIn("gol", names)
        self.assertIn("traffic", names)


if __name__ == "__main__":
    unittest.main()
