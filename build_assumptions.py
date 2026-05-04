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
        self.xor_cache: Dict[Tuple[int, int], Source] = {}
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
            result = self.and_(result, item)
        return result

    def or_all(self, sources: Iterable[Source]) -> Source:
        items = list(sources)
        if not items:
            return self.false()
        result = items[0]
        for item in items[1:]:
            result = self.or_(result, item)
        return result

    def not_(self, source: Source) -> Source:
        if source.name == "1":
            return self.false()
        if self.false_source is not None and source.idx == self.false_source.idx:
            return self.one()
        if source.name.startswith("I"):
            return self.literal(int(source.name[1:]), False)
        return self._cached_commutative("XOR", self.one(), source, self.xor_cache, same_is_left=False)

    def and_(self, left: Source, right: Source) -> Source:
        if left.name == "1":
            return right
        if right.name == "1":
            return left
        if (self.false_source is not None and left.idx == self.false_source.idx) or (
            self.false_source is not None and right.idx == self.false_source.idx
        ):
            return self.false()
        return self._cached_commutative("AND", left, right, self.and_cache, same_is_left=True)

    def or_(self, left: Source, right: Source) -> Source:
        if left.name == "1" or right.name == "1":
            return self.one()
        if self.false_source is not None and left.idx == self.false_source.idx:
            return right
        if self.false_source is not None and right.idx == self.false_source.idx:
            return left
        return self._cached_commutative("OR", left, right, self.or_cache, same_is_left=True)

    def xor(self, left: Source, right: Source) -> Source:
        if left.idx == right.idx:
            return self.false()
        if left.name == "1":
            return self.not_(right)
        if right.name == "1":
            return self.not_(left)
        if self.false_source is not None and left.idx == self.false_source.idx:
            return right
        if self.false_source is not None and right.idx == self.false_source.idx:
            return left
        return self._cached_commutative("XOR", left, right, self.xor_cache, same_is_left=False)

    def _cached_commutative(
        self,
        op: str,
        left: Source,
        right: Source,
        cache: Dict[Tuple[int, int], Source],
        *,
        same_is_left: bool,
    ) -> Source:
        if left.idx == right.idx and same_is_left:
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


def _input_bits(builder: ProgramBuilder, start: int, width: int) -> List[Source]:
    return [builder.input(start + bit) for bit in range(width)]


def _eq_const(builder: ProgramBuilder, bits: Sequence[Source], value: int) -> Source:
    if value >= (1 << len(bits)):
        return builder.false()
    return builder.and_all(
        bit if value & (1 << bit_idx) else builder.not_(bit)
        for bit_idx, bit in enumerate(bits)
    )


def _eq_any_const(builder: ProgramBuilder, bits: Sequence[Source], values: Iterable[int]) -> Source:
    return builder.or_all(_eq_const(builder, bits, value) for value in values)


def _lt_const(builder: ProgramBuilder, bits: Sequence[Source], value: int) -> Source:
    return _eq_any_const(builder, bits, range(max(0, value)))


def _gt_zero(builder: ProgramBuilder, bits: Sequence[Source]) -> Source:
    return builder.or_all(bits)


def _add_vectors(builder: ProgramBuilder, left: Sequence[Source], right: Sequence[Source]) -> List[Source]:
    width = max(len(left), len(right))
    carry = builder.false()
    result: List[Source] = []
    for bit_idx in range(width):
        a = left[bit_idx] if bit_idx < len(left) else builder.false()
        b = right[bit_idx] if bit_idx < len(right) else builder.false()
        a_xor_b = builder.xor(a, b)
        result.append(builder.xor(a_xor_b, carry))
        carry = builder.or_(builder.and_(a, b), builder.and_(carry, a_xor_b))
    result.append(carry)
    return result


def _sum_vectors(builder: ProgramBuilder, vectors: Sequence[Sequence[Source]]) -> List[Source]:
    terms = [list(vector) for vector in vectors]
    if not terms:
        return [builder.false()]
    while len(terms) > 1:
        next_terms: List[List[Source]] = []
        for idx in range(0, len(terms), 2):
            if idx + 1 < len(terms):
                next_terms.append(_add_vectors(builder, terms[idx], terms[idx + 1]))
            else:
                next_terms.append(terms[idx])
        terms = next_terms
    return terms[0]


def _mux(builder: ProgramBuilder, condition: Source, when_true: Source, when_false: Source) -> Source:
    return builder.or_(builder.and_(condition, when_true), builder.and_(builder.not_(condition), when_false))


def _state_eq(builder: ProgramBuilder, cell_idx: int, bits_per_state: int, value: int) -> Source:
    return _eq_const(builder, _input_bits(builder, cell_idx * bits_per_state, bits_per_state), value)


def _raw_neighbor_count(
    builder: ProgramBuilder,
    *,
    bits_per_state: int,
    cell_count: int,
    counted_state: int,
) -> List[Source]:
    return _sum_vectors(
        builder,
        [[_state_eq(builder, cell_idx, bits_per_state, counted_state)] for cell_idx in range(1, cell_count)]
    )


def _raw_neighbor_any(
    builder: ProgramBuilder,
    *,
    bits_per_state: int,
    cell_count: int,
    counted_state: int,
) -> Source:
    return builder.or_all(
        _state_eq(builder, cell_idx, bits_per_state, counted_state) for cell_idx in range(1, cell_count)
    )


def _compressed_moore_count(builder: ProgramBuilder) -> List[Source]:
    return _sum_vectors(
        builder,
        [
            _input_bits(builder, 0, 2),
            _input_bits(builder, 2, 2),
            _input_bits(builder, 4, 2),
        ],
    )


def _compressed_fire_count(builder: ProgramBuilder) -> List[Source]:
    return _sum_vectors(
        builder,
        [
            _input_bits(builder, 0, 1),
            _input_bits(builder, 1, 2),
            _input_bits(builder, 3, 1),
        ],
    )


def _or_state_terms(
    builder: ProgramBuilder,
    *,
    output_width: int,
    terms: Iterable[Tuple[Source, int]],
) -> List[Source]:
    outputs: List[List[Source]] = [[] for _ in range(output_width)]
    for condition, value in terms:
        for bit_idx in range(output_width):
            if value & (1 << bit_idx):
                outputs[bit_idx].append(condition)
    return [builder.or_all(bit_terms) for bit_terms in outputs]


def _life_outputs(builder: ProgramBuilder, center_alive: Source, count_bits: Sequence[Source]) -> List[Source]:
    born = _eq_const(builder, count_bits, 3)
    survives = builder.and_(center_alive, _eq_const(builder, count_bits, 2))
    return [builder.or_(born, survives)]


def _maze_outputs(builder: ProgramBuilder, center_alive: Source, count_bits: Sequence[Source]) -> List[Source]:
    return [builder.or_(_eq_const(builder, count_bits, 3), builder.and_(center_alive, _lt_const(builder, count_bits, 6)))]


def _brian_outputs(builder: ProgramBuilder, center_bits: Sequence[Source], count_bits: Sequence[Source]) -> List[Source]:
    center_dead = _eq_const(builder, center_bits, 0)
    center_alive = _eq_const(builder, center_bits, 1)
    birth = builder.and_(center_dead, _eq_const(builder, count_bits, 2))
    return [birth, center_alive]


def _fire_outputs(builder: ProgramBuilder, center_bits: Sequence[Source], has_fire: Source) -> List[Source]:
    center_tree = _eq_const(builder, center_bits, 1)
    center_fire = _eq_const(builder, center_bits, 2)
    center_ash = _eq_const(builder, center_bits, 3)
    ash_stays = builder.and_(center_ash, has_fire)
    tree_ignites = builder.and_(center_tree, has_fire)
    tree_stays = builder.and_(center_tree, builder.not_(has_fire))
    return [
        builder.or_all([tree_stays, center_fire, ash_stays]),
        builder.or_all([tree_ignites, center_fire, ash_stays]),
    ]


def _wire_outputs(builder: ProgramBuilder, center_bits: Sequence[Source], count_bits: Sequence[Source]) -> List[Source]:
    center_conductor = _eq_const(builder, center_bits, 1)
    center_head = _eq_const(builder, center_bits, 2)
    center_tail = _eq_const(builder, center_bits, 3)
    one_or_two_heads = _eq_any_const(builder, count_bits, [1, 2])
    conductor_stays = builder.and_(center_conductor, builder.not_(one_or_two_heads))
    conductor_heads = builder.and_(center_conductor, one_or_two_heads)
    return [
        builder.or_all([center_head, center_tail, conductor_stays]),
        builder.or_(center_head, conductor_heads),
    ]


def _excitable_outputs(
    builder: ProgramBuilder,
    *,
    center_bits: Sequence[Source],
    has_excited_neighbor: Source,
    state_count: int,
) -> List[Source]:
    terms: List[Tuple[Source, int]] = [(builder.and_(_eq_const(builder, center_bits, 0), has_excited_neighbor), 1)]
    for center in range(1, state_count - 1):
        terms.append((_eq_const(builder, center_bits, center), center + 1))
    return _or_state_terms(builder, output_width=len(center_bits), terms=terms)


def _cyclic_outputs(
    builder: ProgramBuilder,
    *,
    center_bits: Sequence[Source],
    successor_has_neighbor: Sequence[Source] | Source,
    state_count: int,
) -> List[Source]:
    output_terms: List[List[Source]] = [[] for _ in range(len(center_bits))]
    for center in range(state_count):
        successor = (center + 1) % state_count
        center_match = _eq_const(builder, center_bits, center)
        has_neighbor = (
            successor_has_neighbor[center]
            if isinstance(successor_has_neighbor, list)
            else successor_has_neighbor
        )
        selected_bits = [
            _mux(
                builder,
                has_neighbor,
                builder.one() if successor & (1 << bit_idx) else builder.false(),
                center_bits[bit_idx],
            )
            for bit_idx in range(len(center_bits))
        ]
        for bit_idx, selected in enumerate(selected_bits):
            output_terms[bit_idx].append(builder.and_(center_match, selected))
    return [builder.or_all(terms) for terms in output_terms]


def _build_ca_circuit(dataset_name: str, num_inputs: int, num_outputs: int) -> Optional[List[str]]:
    builder = ProgramBuilder(num_inputs)
    outputs: Optional[List[Source]] = None

    if dataset_name == "gol":
        dataset_name = "life-compressed"

    if dataset_name == "life" and (num_inputs, num_outputs) == (9, 1):
        outputs = _life_outputs(builder, builder.input(0), _raw_neighbor_count(builder, bits_per_state=1, cell_count=9, counted_state=1))
    elif dataset_name == "life-compressed" and (num_inputs, num_outputs) == (7, 1):
        outputs = _life_outputs(builder, builder.input(6), _compressed_moore_count(builder))
    elif dataset_name == "maze" and (num_inputs, num_outputs) == (9, 1):
        outputs = _maze_outputs(builder, builder.input(0), _raw_neighbor_count(builder, bits_per_state=1, cell_count=9, counted_state=1))
    elif dataset_name == "maze-compressed" and (num_inputs, num_outputs) == (7, 1):
        outputs = _maze_outputs(builder, builder.input(6), _compressed_moore_count(builder))
    elif dataset_name == "brian" and (num_inputs, num_outputs) == (18, 2):
        outputs = _brian_outputs(builder, _input_bits(builder, 0, 2), _raw_neighbor_count(builder, bits_per_state=2, cell_count=9, counted_state=1))
    elif dataset_name == "brian-compressed" and (num_inputs, num_outputs) == (8, 2):
        outputs = _brian_outputs(builder, _input_bits(builder, 6, 2), _compressed_moore_count(builder))
    elif dataset_name == "fire" and (num_inputs, num_outputs) == (10, 2):
        outputs = _fire_outputs(builder, _input_bits(builder, 0, 2), _raw_neighbor_any(builder, bits_per_state=2, cell_count=5, counted_state=2))
    elif dataset_name == "fire-compressed" and (num_inputs, num_outputs) == (6, 2):
        outputs = _fire_outputs(builder, _input_bits(builder, 4, 2), _gt_zero(builder, _compressed_fire_count(builder)))
    elif dataset_name == "wire" and (num_inputs, num_outputs) == (18, 2):
        outputs = _wire_outputs(builder, _input_bits(builder, 0, 2), _raw_neighbor_count(builder, bits_per_state=2, cell_count=9, counted_state=2))
    elif dataset_name == "wire-compressed" and (num_inputs, num_outputs) == (8, 2):
        outputs = _wire_outputs(builder, _input_bits(builder, 6, 2), _compressed_moore_count(builder))
    elif dataset_name == "excitable" and num_inputs == 27 and num_outputs == 3:
        outputs = _excitable_outputs(
            builder,
            center_bits=_input_bits(builder, 0, 3),
            has_excited_neighbor=_raw_neighbor_any(builder, bits_per_state=3, cell_count=9, counted_state=1),
            state_count=8,
        )
    elif dataset_name == "excitable-compressed" and num_inputs == 9 and num_outputs == 3:
        outputs = _excitable_outputs(
            builder,
            center_bits=_input_bits(builder, 6, 3),
            has_excited_neighbor=_gt_zero(builder, _compressed_moore_count(builder)),
            state_count=8,
        )
    elif dataset_name == "cyclic" and num_inputs == 45 and num_outputs == 5:
        has_successor = [
            _raw_neighbor_any(builder, bits_per_state=5, cell_count=9, counted_state=(center + 1) % 32)
            for center in range(32)
        ]
        outputs = _cyclic_outputs(
            builder,
            center_bits=_input_bits(builder, 0, 5),
            successor_has_neighbor=has_successor,
            state_count=32,
        )
    elif dataset_name == "cyclic-compressed" and num_inputs == 11 and num_outputs == 5:
        outputs = _cyclic_outputs(
            builder,
            center_bits=_input_bits(builder, 6, 5),
            successor_has_neighbor=_gt_zero(builder, _compressed_moore_count(builder)),
            state_count=32,
        )

    if outputs is None:
        return None

    lines = list(builder.instructions)
    lines.extend(f"OUT{out_idx}: {source.name}" for out_idx, source in enumerate(outputs))
    return lines


def _eval_assumption_lines(lines: Sequence[str], inputs: Sequence[bool], num_outputs: int) -> List[bool]:
    values: Dict[str, bool] = {"1": True}
    values.update({f"I{idx}": bool(value) for idx, value in enumerate(inputs)})
    outputs: List[Optional[bool]] = [None] * num_outputs

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
            raise ValueError(f"Unexpected operator in generated assumptions: {op}")

    if any(output is None for output in outputs):
        raise ValueError("Generated assumptions did not assign every output")
    return [bool(output) for output in outputs]


def _maybe_circuit_assumptions(
    dataset_name: str,
    examples: List[Example],
    num_inputs: int,
    num_outputs: int,
) -> Optional[Tuple[List[str], int]]:
    lines = _build_ca_circuit(dataset_name, num_inputs, num_outputs)
    if lines is None:
        return None

    for ex in examples:
        inputs = ex["inputs"]
        outputs = ex["outputs"]
        if any(value is None for value in inputs):
            return None
        generated = _eval_assumption_lines(lines, [bool(value) for value in inputs], num_outputs)
        for out_idx, expected in enumerate(outputs):
            if expected is not None and generated[out_idx] != bool(expected):
                return None

    return lines, sum(1 for line in lines if line.startswith("T"))


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
    strategy: str = "circuit",
) -> Tuple[List[str], int]:
    if strategy not in {"circuit", "dnf"}:
        raise ValueError(f"Unknown assumption strategy: {strategy}")
    if strategy == "circuit" and dataset_name is not None:
        circuit = _maybe_circuit_assumptions(dataset_name, examples, num_inputs, num_outputs)
        if circuit is not None:
            return circuit

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
    parser.add_argument("--strategy", choices=["circuit", "dnf"], default="circuit", help="Use a cellular automata circuit when available, otherwise fall back to exact DNF")
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
