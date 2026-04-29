#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional, TextIO, Tuple, List, Dict, Any, Literal, TypeVar, Union

from z3 import (
    BitVec,
    BitVecVal,
    If,
    Or,
    And,
    Tactic,
    Then,
    ULE,
    ULT,
    Bool,
    Implies,
    Solver,
    SolverFor,
    BoolRef,
    BitVecRef,
    Goal,
    BitVecNumRef,
)

from dataset_plugins import Example, IOList, get_plugin, get_plugin_config, available_plugins

DEFAULT_PROGRAM_LENGTH = 16


@dataclass(frozen=True)
class LogicOperator:
    code: int
    label: str
    apply: Callable[[Any, Any], Any]
    blif_rows: Tuple[str, ...]
    canonical_rank: int


LOGIC_OPERATORS: Tuple[LogicOperator, ...] = (
    LogicOperator(0, 'AND', lambda left, right: left & right, ("11 1",), 0),
    LogicOperator(1, 'XOR', lambda left, right: left ^ right, ("10 1", "01 1"), 1),
    LogicOperator(2, 'OR', lambda left, right: left | right, ("10 1", "01 1", "11 1"), 2),
)

OP_BY_CODE = {op.code: op for op in LOGIC_OPERATORS}
OP_BY_LABEL = {op.label: op for op in LOGIC_OPERATORS}
OP_RANK_BY_CODE = {op.code: op.canonical_rank for op in LOGIC_OPERATORS}
OP_BITS = max(1, max(op.code for op in LOGIC_OPERATORS).bit_length())
OP_RANK_BITS = max(1, max(op.canonical_rank for op in LOGIC_OPERATORS).bit_length())


def _op_label(op_code: int) -> str:
    operator = OP_BY_CODE.get(op_code)
    return operator.label if operator else '?'


def _apply_operator(op_code: int, left: Any, right: Any, default: Any) -> Any:
    operator = OP_BY_CODE.get(op_code)
    if operator is None:
        return default
    return operator.apply(left, right)


def _operator_constraint(op: BitVecRef) -> BoolRef:
    return Or(*(op == BitVecVal(operator.code, OP_BITS) for operator in LOGIC_OPERATORS))


def _operator_expr(op: BitVecRef, left: Any, right: Any) -> Any:
    if not LOGIC_OPERATORS:
        raise ValueError("At least one logic operator must be defined")

    expr = LOGIC_OPERATORS[-1].apply(left, right)
    for operator in reversed(LOGIC_OPERATORS[:-1]):
        expr = If(op == BitVecVal(operator.code, OP_BITS), operator.apply(left, right), expr)
    return expr


def _operator_rank_expr(op: BitVecRef) -> BitVecRef:
    if not LOGIC_OPERATORS:
        raise ValueError("At least one logic operator must be defined")

    expr = BitVecVal(LOGIC_OPERATORS[-1].canonical_rank, OP_RANK_BITS)
    for operator in reversed(LOGIC_OPERATORS[:-1]):
        expr = If(
            op == BitVecVal(operator.code, OP_BITS),
            BitVecVal(operator.canonical_rank, OP_RANK_BITS),
            expr,
        )
    return expr


def _operator_sort_key(op_code: int) -> int:
    try:
        return OP_RANK_BY_CODE[op_code]
    except KeyError as exc:
        raise ValueError(f"Unknown operation code: {op_code}") from exc


@dataclass(frozen=True)
class ProgramSpec:
    num_inputs: int
    num_outputs: int
    program_length: int

    @property
    def total_sources(self) -> int:
        return self.num_inputs + 1 + self.program_length

    @property
    def idx_bits(self) -> int:
        return max(1, (self.total_sources - 1).bit_length())


@dataclass(frozen=True)
class EncodingOptions:
    encode_boolean: bool = False
    force_ordered: bool = False
    force_useful: bool = False


T = TypeVar('T')
def _select_bv(values: List[T], idx_var: Union[BitVecNumRef, BitVecRef], bits: int) -> T:
    if not values or len(values) == 0:
        raise ValueError("values must be a non-empty list")

    result : T = values[0]
    for idx, value in enumerate(values):
        if idx == 0:
            continue
        result = If(idx_var == BitVecVal(idx, bits), value, result)

    return result


def _build_program(spec: ProgramSpec, options: EncodingOptions) -> List[BoolRef]:
    """Build SSA-style straight-line program constraints (no examples).

    Returns a list of constraints that define the program structure.
    """
    if spec.num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    if spec.num_outputs <= 0:
        raise ValueError("num_outputs must be positive")
    if spec.program_length < 0:
        raise ValueError("program_length must be non-negative")

    constraints: List[BoolRef] = []

    for instr in range(spec.program_length):
        idx = spec.num_inputs + 1 + instr  # inputs + const1 + previous temps
        max_idx = idx - 1
        max_idx_bv = BitVecVal(max_idx, spec.idx_bits)

        op = BitVec(f"OP_{instr}", OP_BITS)
        src1 = BitVec(f"S1_{instr}", spec.idx_bits)
        src2 = BitVec(f"S2_{instr}", spec.idx_bits)

        # Force uniqueness of (op, src1, src2) tuples
        if options.force_ordered:
            if instr > 0:
                pre_op = BitVec(f"OP_{instr-1}", OP_BITS)
                pre_src1 = BitVec(f"S1_{instr-1}", spec.idx_bits)
                pre_src2 = BitVec(f"S2_{instr-1}", spec.idx_bits)
                pre_rank = _operator_rank_expr(pre_op)
                rank = _operator_rank_expr(op)

                constraints.append(ULE(pre_src2, src2))
                constraints.append(Implies(pre_src2 == src2, ULE(pre_src1, src1)))
                constraints.append(Implies(And(pre_src2 == src2, pre_src1 == src1), ULT(pre_rank, rank)))

        # Force usefulness of each instruction
        if options.force_useful:
            srcs = [BitVec(f"OUT_{out_idx}_idx", spec.idx_bits) == BitVecVal(idx, spec.idx_bits) for out_idx in range(spec.num_outputs)]

            for next_instr in range(instr + 1, spec.program_length):
                next_src1 = BitVec(f"S1_{next_instr}", spec.idx_bits)
                next_src2 = BitVec(f"S2_{next_instr}", spec.idx_bits)

                srcs.append(next_src1 == BitVecVal(idx, spec.idx_bits))
                srcs.append(next_src2 == BitVecVal(idx, spec.idx_bits))

            constraints.append(Or(*srcs))

        # Encode boolean selection of src1 and src2
        if options.encode_boolean:
            for idx in range(max_idx + 1):
                src1_idx = Bool(f"S1_{instr}_eq_{idx}")
                src2_idx = Bool(f"S2_{instr}_eq_{idx}")
                constraints.append(src1_idx == (src1 == BitVecVal(idx, spec.idx_bits)))
                constraints.append(src2_idx == (src2 == BitVecVal(idx, spec.idx_bits)))

        constraints.append(_operator_constraint(op))
        constraints.append(ULE(src1, max_idx_bv))
        constraints.append(ULE(src2, max_idx_bv))
        xor_operator = OP_BY_LABEL.get('XOR')
        if xor_operator is None:
            constraints.append(ULT(src1, src2))
        else:
            constraints.append(Or(ULT(src1, src2), And(op == BitVecVal(xor_operator.code, OP_BITS), src1 == src2)))

    max_total_idx = BitVecVal(spec.total_sources - 1, spec.idx_bits)
    for out_idx in range(spec.num_outputs):
        selector = BitVec(f"OUT_{out_idx}_idx", spec.idx_bits)
        constraints.append(ULE(selector, max_total_idx))
        if options.encode_boolean:
            for idx in range(spec.total_sources):
                selector_idx = Bool(f"OUT_{out_idx}_eq_{idx}")
                constraints.append(selector_idx == (selector == BitVecVal(idx, spec.idx_bits)))

    return constraints


def _build_assumptions_from_file(file: TextIO, spec: ProgramSpec) -> List[Union[BoolRef, Literal[False]]]:
    """Build assumptions from a file with assumed program bits.

    Example file format (3-bit adder):
    ```
    T0: XOR(I0, I2)
    T1: XOR(I0, I1)
    T2: XOR(I2, T1)
    T3: OR(T0, T1)
    T4: XOR(T2, T3)
    OUT0: T2
    OUT1: T4
    ```
    """

    constraints: List[Union[BoolRef, Literal[False]]] = []

    for line in file.readlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if ':' not in line:
            raise ValueError(f"Invalid assumption line: {line}")

        lhs, rhs = line.split(':', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        def parse_index(kind: str, raw: str) -> int:
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid {kind} index in assumption: {kind}{raw}") from exc

        def translate_arg(arg: str) -> int:
            if arg == '1':
                return 0
            elif arg.startswith('I'):
                input_idx = parse_index('I', arg[1:])
                if not 0 <= input_idx < spec.num_inputs:
                    raise ValueError(f"Input index out of range in assumption: {arg}")
                return input_idx + 1
            elif arg.startswith('T'):
                temp_idx = parse_index('T', arg[1:])
                if not 0 <= temp_idx < spec.program_length:
                    raise ValueError(f"Temporary index out of range in assumption: {arg}")
                return spec.num_inputs + 1 + temp_idx
            else:
                raise ValueError(f"Unknown argument in assumption: {arg}")

        if lhs.startswith('T'):
            instr_idx = parse_index('T', lhs[1:])
            if not 0 <= instr_idx < spec.program_length:
                raise ValueError(f"Temporary index out of range in assumption: {lhs}")

            op_part, args_part = rhs.split('(', 1)
            args_part = args_part.rstrip(')')
            arg1_str, arg2_str = [s.strip() for s in args_part.split(',', 1)]

            operator = OP_BY_LABEL.get(op_part.strip())
            if operator is None:
                raise ValueError(f"Unknown operation in assumption: {op_part}")

            constraints.append(BitVec(f"OP_{instr_idx}", OP_BITS) == BitVecVal(operator.code, OP_BITS))
            constraints.append(BitVec(f"S1_{instr_idx}", spec.idx_bits) == BitVecVal(translate_arg(arg1_str), spec.idx_bits))
            constraints.append(BitVec(f"S2_{instr_idx}", spec.idx_bits) == BitVecVal(translate_arg(arg2_str), spec.idx_bits))

        elif lhs.startswith('OUT'):
            out_idx = parse_index('OUT', lhs[3:])
            if not 0 <= out_idx < spec.num_outputs:
                raise ValueError(f"Output index out of range in assumption: {lhs}")
            arg_idx = translate_arg(rhs.strip())

            constraints.append(BitVec(f"OUT_{out_idx}_idx", spec.idx_bits) == BitVecVal(arg_idx, spec.idx_bits))

        else:
            raise ValueError(f"Unknown LHS in assumption: {lhs}")

    return constraints


def _build_test(
    width: int,
    input_vals: List[int],
    tag: str,
    spec: ProgramSpec,
    options: EncodingOptions,
) -> Tuple[List[Union[BoolRef, Literal[False]]], List[Union[BitVecRef, BitVecNumRef]]]:
    """Build SSA-style straight-line program constraints for a batch.

    Returns a tuple (constraints, outputs) where outputs is a list of
    bit-vector expressions representing the program outputs for this batch.
    """
    if spec.program_length < 0:
        raise ValueError("program_length must be non-negative")

    constraints: List[Union[BoolRef, Literal[False]]] = []

    # Seed values with the batch input truth tables
    values: List[Union[BitVecNumRef, BitVecRef]] = [BitVecVal(input_vals[j], width) for j in range(spec.num_inputs)]
    values = [~BitVecVal(0, width)] + values  # constant 1

    for instr in range(spec.program_length):
        op = BitVec(f"OP_{instr}", OP_BITS)
        src1 = BitVec(f"S1_{instr}", spec.idx_bits)
        src2 = BitVec(f"S2_{instr}", spec.idx_bits)
        val = BitVec(f"VAL_{tag}_{instr}", width)

        if options.encode_boolean:
            left_expr = BitVec(f"LEFT_{tag}_{instr}", width)
            right_expr = BitVec(f"RIGHT_{tag}_{instr}", width)

            for idx, value in enumerate(values):
                src1_idx = Bool(f"S1_{instr}_eq_{idx}")
                src2_idx = Bool(f"S2_{instr}_eq_{idx}")

                constraints.append(Implies(src1_idx, left_expr == value))
                constraints.append(Implies(src2_idx, right_expr == value))
        else:
            left_expr = _select_bv(values, src1, spec.idx_bits)
            right_expr = _select_bv(values, src2, spec.idx_bits)

        gate_expr = _operator_expr(op, left_expr, right_expr)

        constraints.append(val == gate_expr)
        values.append(val)

    outputs: List[Union[BitVecRef, BitVecNumRef]] = []
    for out_idx in range(spec.num_outputs):
        selector = BitVec(f"OUT_{out_idx}_idx", spec.idx_bits)
        if options.encode_boolean:
            out_expr = BitVec(f"OUTVAL_{tag}_{out_idx}", width)
            for idx, value in enumerate(values):
                selector_idx = Bool(f"OUT_{out_idx}_eq_{idx}")
                constraints.append(Implies(selector_idx, out_expr == value))
            outputs.append(out_expr)
        else:
            outputs.append(_select_bv(values, selector, spec.idx_bits))

    return constraints, outputs


def _pack_examples_to_bitvectors(examples: List[Example], num_inputs: int, num_outputs: int) -> Tuple[int, List[int], List[int], List[int]]:
    width = len(examples)
    input_vals = [0] * num_inputs
    output_vals = [0] * num_outputs
    output_masks = [0] * num_outputs

    for t_idx, ex in enumerate(examples):
        ins = ex["inputs"]
        outs = ex["outputs"]

        if len(ins) != num_inputs or len(outs) != num_outputs:
            raise ValueError("Example length does not match num_inputs/num_outputs")

        for j in range(num_inputs):
            if ins[j] is None:
                raise ValueError("Input values cannot be None")
            elif bool(ins[j]):
                input_vals[j] |= (1 << t_idx)

        for j in range(num_outputs):
            if outs[j] is None:
                output_masks[j] |= (1 << t_idx)
            elif bool(outs[j]):
                output_vals[j] |= (1 << t_idx)

    return width, input_vals, output_vals, output_masks


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_dataset_from_config(cfg: Dict[str, Any]) -> Tuple[List[Example], int, int, int]:
    """Returns (examples, num_inputs, num_outputs, instructions)."""
    ctype = cfg.get("type")

    def _collect_examples(data: List[Dict[str, Any]], num_inputs: int, num_outputs: int) -> List[Example]:
        result: List[Example] = []
        for ex in data:
            ins: IOList = [bool(v) for v in ex["inputs"]]
            outs: IOList = [None if v is None else bool(v) for v in ex["outputs"]]
            if len(ins) != num_inputs or len(outs) != num_outputs:
                raise ValueError("Example length does not match declared input/output sizes")
            result.append({"inputs": ins, "outputs": outs})
        return result

    if "examples" in cfg:
        num_inputs = int(cfg["num_inputs"])  # required
        num_outputs = int(cfg["num_outputs"])  # required
        examples: List[Example] = _collect_examples(cfg["examples"], num_inputs, num_outputs)
    elif ctype:
        try:
            plugin = get_plugin(ctype)
        except KeyError as exc:
            raise ValueError(f"Unsupported config type or format: {ctype}") from exc
        examples, num_inputs, num_outputs = plugin(cfg)
    else:
        raise ValueError(f"Unsupported config type or format: {ctype}")

    instructions = cfg.get("instructions")
    if instructions is None:
        instructions = DEFAULT_PROGRAM_LENGTH
    instructions = int(instructions)
    if instructions < 0:
        raise ValueError("instructions must be non-negative")

    return examples, num_inputs, num_outputs, instructions


def _make_solver(solver_choice: str) -> Solver:
    if solver_choice == 'z3':
        return SolverFor('QF_BV')

    if solver_choice == 'simple-tactic':
        simple_tactic: Tactic = Then(
            Tactic('simplify'),
            Tactic('propagate-values'),
            Tactic('bit-blast'),
            Tactic('sat'),
        )
        return simple_tactic.solver()

    if solver_choice == 'ctx-simplify-tactic':
        ctx_simplify_tactic: Tactic = Then(
            Tactic('ctx-simplify'),
            Tactic('propagate-values'),
            Tactic('bit-blast'),
            Tactic('sat'),
        )
        return ctx_simplify_tactic.solver()

    raise ValueError(f"Unsupported solver choice: {solver_choice}")


def _export_blif(examples: List[Example], num_inputs: int, num_outputs: int) -> None:
    """Export the problem as a BLIF file to stdout."""

    print(f".model synth_program")
    print(f'.inputs {" ".join(f"I{i}" for i in range(num_inputs))}')
    print(f'.outputs {" ".join(f"OUT{o}" for o in range(num_outputs))}')


    for out_idx in range(num_outputs):
        print(".names " + " ".join(f"I{i}" for i in range(num_inputs)) + f" OUT{out_idx}")
        for ex in examples:
            ins = ex["inputs"]
            outs = ex["outputs"]
            if outs[out_idx] is None:
                raise ValueError("Cannot export BLIF with don't-care outputs")
            input_line = "".join('1' if bool(v) else '0' for v in ins)
            if bool(outs[out_idx]):
                print(f"{input_line} 1")

    print(".end")


def _format_source(idx: int, num_inputs: int) -> str:
    if idx == 0:
        return "1"
    if idx <= num_inputs:
        return f"I{idx - 1}"
    return f"T{idx - num_inputs - 1}"


def _evaluate_program(
    instrs: List[Tuple[Optional[int], int, int]],
    output_selectors: List[int],
    inputs: IOList,
) -> List[int]:
    values: List[int] = [1] + [int(bool(v)) for v in inputs]

    for op, s1_idx, s2_idx in instrs:
        if op is None:
            val = 0
        else:
            left = values[s1_idx]
            right = values[s2_idx]
            val = _apply_operator(op, left, right, 0)
        values.append(val)

    return [values[selector] for selector in output_selectors]


def _verify_program(
    instrs: List[Tuple[Optional[int], int, int]],
    output_selectors: List[int],
    examples: List[Example],
    num_outputs: int,
) -> List[Tuple[int, IOList, List[int], List[int]]]:
    mismatches: List[Tuple[int, IOList, List[int], List[int]]] = []

    for idx, ex in enumerate(examples):
        ins = ex["inputs"]
        outs = ex["outputs"]

        expected_outs_mask = [int(v is None) for v in outs]
        expected_outs = [int(bool(v)) for v in outs]
        actual_outs = _evaluate_program(instrs, output_selectors, ins)

        for j in range(num_outputs):
            if expected_outs_mask[j]:
                actual_outs[j] = -1  # don't care
                expected_outs[j] = -1  # don't care

        if actual_outs != expected_outs:
            mismatches.append((idx, ins, expected_outs, actual_outs))

    return mismatches


def _emit_program(
    instrs: List[Tuple[Optional[int], int, int]],
    outputs: List[int],
    num_inputs: int,
    num_outputs: int,
    output_blif: bool,
) -> None:
    if output_blif:
        print(f".model spec")
        print(f'.inputs {" ".join(f"I{i}" for i in range(num_inputs))}')
        print(f'.outputs {" ".join(f"OUT{o}" for o in range(num_outputs))}')

    for instr_idx, (op_val, s1_val, s2_val) in enumerate(instrs):
        operator = OP_BY_CODE.get(op_val) if op_val is not None else None
        label = _op_label(op_val) if op_val is not None else '?'

        if output_blif:
            if operator is None:
                raise ValueError("Unsupported operation in BLIF output")
            if s1_val == s2_val:
                if operator.label != 'XOR':
                    raise ValueError("Only XOR may use duplicate sources in BLIF output")
                print(f".names T{instr_idx}")
            else:
                print(f".names {_format_source(s1_val, num_inputs)} {_format_source(s2_val, num_inputs)} T{instr_idx}")
                for row in operator.blif_rows:
                    print(row)
        else:
            print(f"T{instr_idx}: {label}({_format_source(s1_val, num_inputs)}, {_format_source(s2_val, num_inputs)})")

    for out_idx, sel in enumerate(outputs):
        if output_blif:
            print(f".names {_format_source(sel, num_inputs)} OUT{out_idx}")
            print(f"1 1")
        else:
            print(f"OUT{out_idx}: {_format_source(sel, num_inputs)}")

    if output_blif:
        print(".end")


def _post_process_program(instrs: List[Tuple[Optional[int], int, int]], num_inputs: int, num_outputs: int, examples: List[Example], outputs: List[int]) -> Tuple[List[Tuple[Optional[int], int, int]], List[int]]:
    """Post-process the synthesized program"""
    logger = logging.getLogger(__name__)

    class DAGNode:
        def __init__(self, op: int, s1: int, s2: int):
            self.op = op
            self.s1 = s1
            self.s2 = s2
            self.users: set[int] = set()

        def __repr__(self) -> str:
            return f"({self.op}, {self.s1}, {self.s2})"

    dag: Dict[int, DAGNode] = {}

    for i, (op, s1, s2) in enumerate(instrs):
        idx = num_inputs + 1 + i
        if op is None:
            raise ValueError("Cannot post-process program with unknown operations")

        dag[idx] = DAGNode(op, s1, s2)

    for i, (op, s1, s2) in enumerate(instrs):
        idx = num_inputs + 1 + i
        if s1 in dag:
            dag[s1].users.add(idx)
        if s2 in dag:
            dag[s2].users.add(idx)

    XOR = OP_BY_LABEL['XOR'].code

    def fmt_source(idx: int) -> str:
        return _format_source(idx, num_inputs)

    def check() -> bool:
        for ex in examples:
            ins = ex["inputs"]
            outs = ex["outputs"]

            values: Dict[int, bool] = {i + 1: bool(v) for i, v in enumerate(ins)}
            values[0] = True  # constant 1

            for idx, node in dag.items():
                if node.s1 not in values:
                    raise ValueError(f"Missing value for source {node.s1} of node {idx} in {dag} with outputs {outputs}")
                if node.s2 not in values:
                    raise ValueError(f"Missing value for source {node.s2} of node {idx} in {dag} with outputs {outputs}")
                left = values[node.s1]
                right = values[node.s2]
                val = _apply_operator(node.op, left, right, False)
                values[idx] = val

            for out_idx in range(num_outputs):
                sel_idx = outputs[out_idx]
                if outs[out_idx] is None:
                    continue
                expected = bool(outs[out_idx])
                actual = values[sel_idx]
                if expected != actual:
                    return False

        return True

    assert check(), "Post-processing started with incorrect program"

    updated = True
    while updated:
        updated = False
        logger.info("Starting post-processing iteration")

        # Ensure src1 < src2
        for idx, node in dag.items():
            if node.s1 > node.s2:
                logger.info("Swapping sources of instruction %s to maintain s1 < s2", fmt_source(idx))
                node.s1, node.s2 = node.s2, node.s1

        # Ensure that instructions are ordered by (s2, s1, op)
        sorted_items = sorted(dag.items(), key=lambda item: (item[1].s2, item[1].s1, _operator_sort_key(item[1].op)))
        if list(dag.keys()) != [k for k, _ in sorted_items]:
            logger.info("Re-ordering instructions to maintain canonical order")
            updated = True
            new_idxs: Dict[int, int] = {}
            for new_idx, (old_idx, node) in enumerate(sorted_items):
                new_idxs[old_idx] = num_inputs + 1 + new_idx

            new_dag: Dict[int, DAGNode] = {}
            for old_idx, node in sorted_items:
                new_node = DAGNode(node.op, new_idxs.get(node.s1, node.s1), new_idxs.get(node.s2, node.s2))
                new_dag[new_idxs[old_idx]] = new_node

            for out_idx in range(num_outputs):
                sel_idx = outputs[out_idx]
                outputs[out_idx] = new_idxs.get(sel_idx, sel_idx)

            dag = new_dag

            if updated:
                continue

        # Update users
        for node in dag.values():
            node.users.clear()

        for idx, node in dag.items():
            if node.s1 in dag:
                dag[node.s1].users.add(idx)
            if node.s2 in dag:
                dag[node.s2].users.add(idx)

        # Ensure that if an instruction has one user with the same op, then s1, s2 < user.s1, user.s2
        for idx, node in dag.items():
            if len(node.users) != 1:
                continue

            user_idx = next(iter(node.users))
            user_node = dag[user_idx]

            if node.op != user_node.op:
                continue

            assert idx == user_node.s1 or idx == user_node.s2, f"Node {idx} is not a source of its user {user_idx} in {dag}"

            if not (node.s1 < user_node.s1 and node.s2 < user_node.s1 and node.s1 < user_node.s2 and node.s2 < user_node.s2):
                logger.info(
                    "Adjusting instruction %s to satisfy user %s constraints",
                    fmt_source(idx),
                    fmt_source(user_idx),
                )
                sources = sorted([node.s1, node.s2, user_node.s1, user_node.s2])
                node.s1, node.s2, user_node.s1, user_node.s2 = sources
                updated = True
                break

        accessed: set[int] = set()
        new_accessed: List[int] = []

        for idx in outputs:
            accessed.add(idx)
            new_accessed.append(idx)

        while len(new_accessed) > 0:
            curr = new_accessed.pop()
            nnode = dag.get(curr)
            if nnode is None:
                continue

            for src in (nnode.s1, nnode.s2):
                if src not in accessed:
                    accessed.add(src)
                    new_accessed.append(src)

        # Eliminate unused nodes
        for idx in list(dag.keys()):
            if idx not in accessed:
                logger.info("Eliminating unused instruction %s", fmt_source(idx))
                del dag[idx]

        # Try to replace operands with earlier equivalent nodes
        for idx, node in dag.items():
            s1_idx = node.s1
            s2_idx = node.s2
            old_op = node.op

            possible_idxs = [x for x in list(range(num_inputs + 1)) + list(dag.keys()) if x < idx]

            alt_ops = [operator.code for operator in LOGIC_OPERATORS]

            # cartesian product: (op, s1, s2)
            product = [
                (op, s1, s2)
                for op in alt_ops
                for s1 in possible_idxs
                for s2 in possible_idxs
                if (
                    (s1 < s2 or (op == XOR and s1 == s2))
                    and ((s2, s1, _operator_sort_key(op)) < (s2_idx, s1_idx, _operator_sort_key(old_op)))
                )
            ]
            product.sort(key=lambda x: (x[2], x[1], _operator_sort_key(x[0])))  # sort by (s2, s1, op rank)

            for op, cidx1, cidx2 in product:
                node.op = op
                node.s1 = cidx1
                node.s2 = cidx2

                if not check():
                    node.op = old_op
                    node.s1 = s1_idx
                    node.s2 = s2_idx
                    continue

                logger.info(
                    "Replacing instruction %s from %s(%s, %s) to %s(%s, %s)",
                    fmt_source(idx),
                    _op_label(old_op),
                    fmt_source(s1_idx),
                    fmt_source(s2_idx),
                    _op_label(op),
                    fmt_source(cidx1),
                    fmt_source(cidx2),
                )
                updated = True
                break


        # Try to use earlier equivalent nodes as outputs
        for out_idx in range(num_outputs):
            sel_idx = outputs[out_idx]

            for cidx in list(range(num_inputs + 1)) + list(dag.keys()):
                if cidx >= sel_idx:
                    continue

                outputs[out_idx] = cidx

                if not check():
                    outputs[out_idx] = sel_idx
                    continue

                logger.info("Replacing output OUT%d from %s to %s", out_idx, fmt_source(sel_idx), fmt_source(cidx))
                updated = True
                break

    # Re-number instructions to be contiguous
    new_idxs = {}
    for new_idx, old_idx in enumerate(sorted(dag.keys())):
        new_idxs[old_idx] = new_idx + (num_inputs + 1)

    new_dag = {}
    for idx, node in dag.items():
        new_node = DAGNode(node.op, new_idxs.get(node.s1, node.s1), new_idxs.get(node.s2, node.s2))
        new_dag[new_idxs[idx]] = new_node

    for out_idx in range(num_outputs):
        sel_idx = outputs[out_idx]
        outputs[out_idx] = new_idxs.get(sel_idx, sel_idx)

    dag = new_dag

    instrs = []
    for idx in dag.keys():
        node = dag[idx]
        instrs.append((node.op, node.s1, node.s2))

    logger.info("Post-processing reduced program to %d instructions", len(instrs))

    return instrs, outputs


def main() -> None:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Synthesize a logic program with z3")

    parser.add_argument("--dataset", choices=list(available_plugins().keys()), default="gol", help="Choose a built-in dataset config")
    parser.add_argument("--config", type=str, default=None, help="Path to a custom JSON config")

    parser.add_argument("--assume", type=str, default=None, help="Path to a file with assumed program bits")

    parser.add_argument("--post-process", action="store_true", help="Post-process the synthesized program, attempting to minimize instructions and simplify")

    parser.add_argument("--instructions", type=int, default=None, help="Override number of SSA instructions")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of examples to add to each bit-vector-encoded batch (default: all examples)")

    parser.add_argument("--make-smt2", action="store_true", help="Output the problem in SMT-LIB2 format and exit")
    parser.add_argument("--make-dimacs", action="store_true", help="Output the problem in DIMACS CNF format and exit (uses bit-blasting followed by Tseitin transformation)")
    parser.add_argument("--make-blif", action="store_true", help="Output the problem specification in BLIF format and exit")

    parser.add_argument("--solver", type=str, choices=["z3", "simple-tactic", "ctx-simplify-tactic"], default="simple-tactic", help="Choose the Z3 solver or tactic to use")

    parser.add_argument("--encode-boolean", action="store_true", help="Enable boolean encoding")
    parser.add_argument("--force-ordered", action="store_true", help="Enable constraint on ordered instructions")
    parser.add_argument("--force-useful", action="store_true", help="Enable constraint on useful instructions")

    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of examples")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling examples (None means random)")

    parser.add_argument("--do-all", action="store_true", help="Synthesize for all example batches at once")

    parser.add_argument("--output-blif", action="store_true", help="Output the found program in BLIF format")

    parser.add_argument("--quiet", action="store_true", help="Suppress informational output")

    args = parser.parse_args()

    options = EncodingOptions(
        encode_boolean=args.encode_boolean,
        force_ordered=args.force_ordered,
        force_useful=args.force_useful,
    )

    if args.quiet:
        logger.setLevel(logging.WARNING)
        logging.getLogger("z3").setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger("z3").setLevel(logging.INFO)


    # print the chosen configuration
    logger.info("Using configuration: %s", vars(args))

    if args.config:
        cfg = _load_config(Path(args.config))
    elif args.dataset:
        cfg = get_plugin_config(args.dataset)
    else:
        raise SystemExit("Either --dataset or --config must be specified")

    examples, num_inputs, num_outputs, cfg_instructions = _build_dataset_from_config(cfg)
    if not examples:
        raise SystemExit("Dataset contains no examples")

    instructions = cfg_instructions
    if args.instructions is not None:
        instructions = args.instructions
        if instructions < 0:
            raise SystemExit("--instructions must be non-negative")

    spec = ProgramSpec(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        program_length=instructions,
    )

    batch_size = args.batch_size if args.batch_size else len(examples)
    if batch_size <= 0:
        raise SystemExit("Batch size must be positive")

    if args.make_blif:
        _export_blif(examples, spec.num_inputs, spec.num_outputs)
        return

    s = _make_solver(args.solver)

    s.add(*_build_program(spec, options))

    if args.assume:
        if args.assume == "-":
            s.add(*_build_assumptions_from_file(sys.stdin, spec))
        else:
            assume_path = Path(args.assume)
            if not assume_path.is_file():
                raise SystemExit(f"Assume file not found: {assume_path}")
            with assume_path.open("r", encoding="utf-8") as file:
                s.add(*_build_assumptions_from_file(file, spec))

    logger.info("Built program structure with %d instructions", spec.program_length)
    logger.info("Solver has %d assertions", len(s.assertions()))

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(examples)

    start = time.time()
    result = None
    for batch_idx, offset in enumerate(range(0, len(examples), batch_size)):
        batch = examples[offset: offset + batch_size]
        width, input_vals, output_vals, output_masks = _pack_examples_to_bitvectors(batch, spec.num_inputs, spec.num_outputs)
        constraints, outputs = _build_test(width, input_vals, tag=f"b{batch_idx}", spec=spec, options=options)
        s.add(*constraints)
        for j in range(spec.num_outputs):
            if output_masks[j] != 0:
                s.add((outputs[j] | output_masks[j]) == (BitVecVal(output_vals[j], width) | output_masks[j]))
            else:
                s.add(outputs[j] == BitVecVal(output_vals[j], width))

        logger.info("Solver has %d assertions after batch %d", len(s.assertions()), batch_idx + 1)

        if not args.make_smt2 and not args.make_dimacs and not args.do_all:
            result = s.check()

            if str(result) != 'sat':
                break

            progress = min(offset + batch_size, len(examples))
            logger.info("Processed %d/%d examples", progress, len(examples))

    if args.make_smt2:
        print(s.to_smt2())
        return

    if args.make_dimacs:
        goal = Goal()

        for c in s.assertions():
            goal.add(c)

        cnf_result = Then(
            Tactic('simplify'),
            Tactic('propagate-values'),
            Tactic('bit-blast'),
            Tactic('tseitin-cnf'),
        )(goal)

        assert len(cnf_result) == 1

        cnf_goal = cnf_result[0]
        print(cnf_goal.dimacs())
        return

    if args.do_all:
        result = s.check()

    elapsed = time.time() - start

    if str(result) == 'sat':
        logger.info("SAT in %.3f seconds", elapsed)

        # Pretty-print a compact architecture summary: per-instruction (op, src indices)
        m = s.model()

        instrs: List[Tuple[Optional[int], int, int]] = []

        def eval_bv_as_long(name: str, ref: BitVecRef) -> int:
            value = m.evaluate(ref, model_completion=True)
            if not isinstance(value, BitVecNumRef):
                raise RuntimeError(f"Model did not provide a concrete value for {name}: {value}")
            return value.as_long()

        for instr in range(spec.program_length):
            op_ref = BitVec(f"OP_{instr}", OP_BITS)
            s1_ref = BitVec(f"S1_{instr}", spec.idx_bits)
            s2_ref = BitVec(f"S2_{instr}", spec.idx_bits)
            op_val = eval_bv_as_long(f"OP_{instr}", op_ref)
            s1_val = eval_bv_as_long(f"S1_{instr}", s1_ref)
            s2_val = eval_bv_as_long(f"S2_{instr}", s2_ref)
            operator = OP_BY_CODE.get(op_val)

            if operator is not None:
                instrs.append((op_val, s1_val, s2_val))
            else:
                instrs.append((None, s1_val, s2_val))

        outputs: List[int] = []
        for out_idx in range(spec.num_outputs):
            selector = BitVec(f"OUT_{out_idx}_idx", spec.idx_bits)
            outputs.append(eval_bv_as_long(f"OUT_{out_idx}_idx", selector))

        if len(outputs) != spec.num_outputs:
            raise RuntimeError(f"Model provided {len(outputs)} outputs, expected {spec.num_outputs}")

        # Post-process the synthesized program
        if args.post_process:
            logger.info("Post-processing synthesized program")
            instrs, outputs = _post_process_program(instrs, spec.num_inputs, spec.num_outputs, examples, outputs)

        _emit_program(instrs, outputs, spec.num_inputs, spec.num_outputs, args.output_blif)

        # Verify the synthesized program against all examples
        mismatches = _verify_program(instrs, outputs, examples, spec.num_outputs)
        if not mismatches:
            logger.info("All examples matched successfully")
        else:
            for idx, ins, expected, actual in mismatches:
                logger.error("Mismatch on example %d with inputs %s: expected %s, got %s", idx, ins, expected, actual)
            logger.error("Total mismatches: %d", len(mismatches))
            exit(1)
    elif str(result) == 'unsat':
        logger.info("UNSAT in %.3f seconds", elapsed)
        exit(1)
    else:
        logger.info("UNKNOWN result: %s in %.3f seconds", result, elapsed)
        exit(1)


if __name__ == "__main__":
    main()
