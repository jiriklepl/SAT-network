#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataset_plugins import Example, IOList, available_plugins, get_plugin_config
from main import _build_dataset_from_config


@dataclass(frozen=True)
class Source:
    name: str
    idx: int


class ProgramBuilder:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.instructions: List[str] = []
        self.not_inputs: Dict[int, Source] = {}
        self.and_cache: Dict[Tuple[int, int], Source] = {}
        self.or_cache: Dict[Tuple[int, int], Source] = {}
        self.false_source: Optional[Source] = None

    def input(self, input_idx: int) -> Source:
        return Source(f"I{input_idx}", input_idx + 1)

    def one(self) -> Source:
        return Source("1", 0)

    def false(self) -> Source:
        if self.false_source is None:
            self.false_source = self._emit("XOR", self.one(), self.one())
        return self.false_source

    def literal(self, input_idx: int, value: bool) -> Source:
        if value:
            return self.input(input_idx)
        if input_idx not in self.not_inputs:
            self.not_inputs[input_idx] = self._emit("XOR", self.one(), self.input(input_idx))
        return self.not_inputs[input_idx]

    def and_all(self, sources: Iterable[Source]) -> Source:
        items = list(sources)
        if not items:
            return self.one()
        result = items[0]
        for item in items[1:]:
            result = self._cached_commutative("AND", result, item, self.and_cache)
        return result

    def or_all(self, sources: Iterable[Source]) -> Source:
        items = list(sources)
        if not items:
            return self.false()
        result = items[0]
        for item in items[1:]:
            result = self._cached_commutative("OR", result, item, self.or_cache)
        return result

    def _cached_commutative(self, op: str, left: Source, right: Source, cache: Dict[Tuple[int, int], Source]) -> Source:
        if left.idx == right.idx:
            return left
        first, second = sorted((left, right), key=lambda source: source.idx)
        key = (first.idx, second.idx)
        if key not in cache:
            cache[key] = self._emit(op, first, second)
        return cache[key]

    def _emit(self, op: str, left: Source, right: Source) -> Source:
        # All supported operations are commutative. Keep operands in source-index order
        # so the generated assumptions satisfy main.py's structural constraints.
        first, second = sorted((left, right), key=lambda source: source.idx)
        instr_idx = len(self.instructions)
        self.instructions.append(f"T{instr_idx}: {op}({first.name}, {second.name})")
        return Source(f"T{instr_idx}", self.num_inputs + 1 + instr_idx)


GOL_COMPRESSED_ASSUMPTIONS = [
    "T0: XOR(I0, I2)",
    "T1: XOR(I0, I4)",
    "T2: OR(I1, I5)",
    "T3: XOR(I1, I5)",
    "T4: XOR(I2, T1)",
    "T5: AND(T0, T1)",
    "T6: OR(I6, T4)",
    "T7: XOR(I0, T5)",
    "T8: OR(I3, T7)",
    "T9: XOR(I3, T7)",
    "T10: XOR(T2, T8)",
    "T11: OR(T3, T9)",
    "T12: AND(T6, T10)",
    "T13: AND(T11, T12)",
    "OUT0: T13",
]


def _bits_to_int(values: Sequence[bool]) -> int:
    return sum(1 << idx for idx, value in enumerate(values) if value)


def _gol_compressed_output(inputs: Sequence[bool]) -> bool:
    left_count = _bits_to_int(inputs[0:2])
    center_count = _bits_to_int(inputs[2:4])
    right_count = _bits_to_int(inputs[4:6])
    alive = inputs[6]
    neighbor_count = left_count + center_count + right_count
    return neighbor_count == 3 or (alive and neighbor_count == 2)


def _maybe_named_assumptions(
    dataset_name: str,
    examples: List[Example],
    num_inputs: int,
    num_outputs: int,
) -> Optional[Tuple[List[str], int]]:
    if dataset_name not in {"gol", "life-compressed"} or num_inputs != 7 or num_outputs != 1:
        return None

    for ex in examples:
        inputs = ex["inputs"]
        outputs = ex["outputs"]
        if len(inputs) != 7 or any(value is None for value in inputs):
            return None
        if outputs[0] is not None and bool(outputs[0]) != _gol_compressed_output([bool(value) for value in inputs]):
            return None

    return list(GOL_COMPRESSED_ASSUMPTIONS), 14


def _load_examples(args: argparse.Namespace) -> Tuple[List[Example], int, int, int]:
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as file:
            cfg = json.load(file)
    else:
        cfg = get_plugin_config(args.dataset)

    examples, num_inputs, num_outputs, cfg_instructions = _build_dataset_from_config(cfg)
    instructions = cfg_instructions
    if args.instructions is not None:
        instructions = args.instructions
    return examples, num_inputs, num_outputs, instructions


def _check_examples(examples: List[Example], num_inputs: int, num_outputs: int) -> Dict[Tuple[bool, ...], IOList]:
    by_input: Dict[Tuple[bool, ...], IOList] = {}
    for ex in examples:
        inputs = ex["inputs"]
        outputs = ex["outputs"]
        if len(inputs) != num_inputs or len(outputs) != num_outputs:
            raise ValueError("Example length does not match declared input/output sizes")
        if any(value is None for value in inputs):
            raise ValueError("Cannot generate assumptions for examples with input don't-cares")

        key = tuple(bool(value) for value in inputs)
        normalized_outputs: IOList = [None if value is None else bool(value) for value in outputs]
        previous = by_input.get(key)
        if previous is not None:
            for out_idx, (old, new) in enumerate(zip(previous, normalized_outputs)):
                if old is not None and new is not None and old != new:
                    raise ValueError(f"Conflicting outputs for duplicate input {key} at OUT{out_idx}")
            by_input[key] = [old if new is None else new if old is None else old for old, new in zip(previous, normalized_outputs)]
        else:
            by_input[key] = normalized_outputs
    return by_input


def build_assumption_lines(
    examples: List[Example],
    num_inputs: int,
    num_outputs: int,
    *,
    dataset_name: Optional[str] = None,
    strategy: str = "dnf",
) -> Tuple[List[str], int]:
    if strategy not in {"auto", "dnf"}:
        raise ValueError(f"Unknown assumption strategy: {strategy}")
    if strategy == "auto" and dataset_name is not None:
        named = _maybe_named_assumptions(dataset_name, examples, num_inputs, num_outputs)
        if named is not None:
            return named

    by_input = _check_examples(examples, num_inputs, num_outputs)
    builder = ProgramBuilder(num_inputs)
    minterm_cache: Dict[Tuple[bool, ...], Source] = {}

    def minterm(inputs: Tuple[bool, ...]) -> Source:
        if inputs not in minterm_cache:
            minterm_cache[inputs] = builder.and_all(builder.literal(idx, value) for idx, value in enumerate(inputs))
        return minterm_cache[inputs]

    output_sources: List[Source] = []
    for out_idx in range(num_outputs):
        true_terms = [
            minterm(inputs)
            for inputs, outputs in sorted(by_input.items())
            if outputs[out_idx] is True
        ]
        output_sources.append(builder.or_all(true_terms))

    lines = list(builder.instructions)
    for out_idx, source in enumerate(output_sources):
        lines.append(f"OUT{out_idx}: {source.name}")
    return lines, len(builder.instructions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an assumption file from a registered dataset truth table")
    parser.add_argument("--dataset", choices=list(available_plugins().keys()), default="life-compressed", help="Dataset plugin to encode")
    parser.add_argument("--config", type=str, default=None, help="Path to a custom JSON config")
    parser.add_argument("--instructions", type=int, default=None, help="Instruction budget to report in the header")
    parser.add_argument("--strategy", choices=["auto", "dnf"], default="auto", help="Use a named circuit when available, otherwise fall back to DNF")
    parser.add_argument("--output", type=str, default=None, help="Write assumptions to this file instead of stdout")
    parser.add_argument("--max-generated-instructions", type=int, default=20000, help="Abort if the generated assumption program exceeds this size")
    args = parser.parse_args()

    examples, num_inputs, num_outputs, configured_instructions = _load_examples(args)
    dataset_name = None if args.config else args.dataset
    lines, required_instructions = build_assumption_lines(
        examples,
        num_inputs,
        num_outputs,
        dataset_name=dataset_name,
        strategy=args.strategy,
    )
    if required_instructions > args.max_generated_instructions:
        raise SystemExit(
            f"Generated {required_instructions} instructions, exceeding --max-generated-instructions={args.max_generated_instructions}"
        )

    output_lines = [
        f"# dataset: {args.config if args.config else args.dataset}",
        f"# examples: {len(examples)}",
        f"# inputs: {num_inputs}",
        f"# outputs: {num_outputs}",
        f"# strategy: {args.strategy}",
        f"# required-instructions: {required_instructions}",
    ]
    if configured_instructions != required_instructions:
        output_lines.append(f"# run with: --instructions {required_instructions} --assume <this-file>")
    else:
        output_lines.append("# run with: --assume <this-file>")
    output_lines.extend(lines)

    output = "\n".join(output_lines) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
