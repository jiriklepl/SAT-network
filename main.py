#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
import logging
import itertools
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, TextIO, Tuple, List, Dict, Any, Literal, TypeVar, Union

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
    LogicOperator(0, 'AND', lambda left, right: left & right, ("11 1",), 1),
    LogicOperator(1, 'XOR', lambda left, right: left ^ right, ("10 1", "01 1"), 0),
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


def _post_process_program(
    instrs: List[Tuple[Optional[int], int, int]],
    num_inputs: int,
    num_outputs: int,
    examples: List[Example],
    outputs: List[int],
    enable_afterburner: bool = True,
    post_process_beam_width: int = 1,
    post_process_beam_rounds: int = 0,
    post_process_beam_candidates: int = 0,
    post_process_replace_patience: int = 50,
    post_process_resynthesis_maxnodes: int = 6,
    post_process_resynthesis_patience: int = 10,
    post_process_score: Optional[List[Union[str, List[str]]]] = None,
) -> Tuple[List[Tuple[Optional[int], int, int]], List[int]]:
    """Post-process the synthesized program"""
    logger = logging.getLogger(__name__)
    outputs = list(outputs)
    if post_process_replace_patience < 0:
        raise ValueError("post_process_replace_patience must be non-negative")
    if post_process_resynthesis_maxnodes < 2:
        raise ValueError("post_process_resynthesis_maxnodes must be at least 2")
    if post_process_resynthesis_patience < 0:
        raise ValueError("post_process_resynthesis_patience must be non-negative")
    if not post_process_score:
        score_phases = [["program-length"]]
    elif all(isinstance(metric, str) for metric in post_process_score):
        score_phases = [list(post_process_score)]
    else:
        score_phases = [list(phase) for phase in post_process_score if not isinstance(phase, str)]
    if not score_phases or any(not phase for phase in score_phases):
        raise ValueError("Post-process score must specify at least one metric per phase")
    score_phase_specs = [
        [(metric[1:], True) if metric.startswith("-") else (metric, False) for metric in phase]
        for phase in score_phases
    ]
    active_score_metric_specs = score_phase_specs[0]

    if len(outputs) != num_outputs:
        raise ValueError(f"Expected {num_outputs} output selectors, got {len(outputs)}")

    class DAGNode:
        def __init__(self, op: int, s1: int, s2: int):
            self.op = op
            self.s1 = s1
            self.s2 = s2
            self.users: set[int] = set()

        def __repr__(self) -> str:
            return f"({self.op}, {self.s1}, {self.s2})"

    Instruction = Tuple[Optional[int], int, int]
    Program = List[Instruction]
    ProgramKey = Tuple[Tuple[Instruction, ...], Tuple[int, ...]]
    Snapshot = Tuple[Dict[int, DAGNode], List[int]]

    dag: Dict[int, DAGNode] = {}

    for i, (op, s1, s2) in enumerate(instrs):
        idx = num_inputs + 1 + i
        if op is None:
            raise ValueError("Cannot post-process program with unknown operations")
        if op not in OP_BY_CODE:
            raise ValueError(f"Cannot post-process program with unsupported operation code: {op}")
        if not 0 <= s1 < idx or not 0 <= s2 < idx:
            raise ValueError(f"Instruction {idx} references an invalid source")

        dag[idx] = DAGNode(op, s1, s2)

    max_source = num_inputs + len(instrs)
    for out_idx, sel_idx in enumerate(outputs):
        if not 0 <= sel_idx <= max_source:
            raise ValueError(f"Output OUT{out_idx} selector out of range: {sel_idx}")

    for i, (op, s1, s2) in enumerate(instrs):
        idx = num_inputs + 1 + i
        if s1 in dag:
            dag[s1].users.add(idx)
        if s2 in dag:
            dag[s2].users.add(idx)

    XOR = OP_BY_LABEL['XOR'].code
    logic_operator_codes = [operator.code for operator in LOGIC_OPERATORS]

    def fmt_source(idx: int) -> str:
        return _format_source(idx, num_inputs)

    example_count = len(examples)
    all_examples_mask = (1 << example_count) - 1
    input_masks = [0] * num_inputs
    expected_output_masks = [0] * num_outputs
    expected_output_values = [0] * num_outputs

    for example_idx, ex in enumerate(examples):
        bit = 1 << example_idx
        ins = ex["inputs"]
        outs = ex["outputs"]

        for input_idx in range(num_inputs):
            if bool(ins[input_idx]):
                input_masks[input_idx] |= bit

        for out_idx in range(num_outputs):
            if outs[out_idx] is None:
                continue
            expected_output_masks[out_idx] |= bit
            if bool(outs[out_idx]):
                expected_output_values[out_idx] |= bit

    def topological_dag_order() -> List[int]:
        ordered_idxs: List[int] = []
        visit_state: Dict[int, int] = {}

        def visit(idx: int) -> None:
            state = visit_state.get(idx, 0)
            if state == 2:
                return
            if state == 1:
                raise ValueError(f"Cycle detected while ordering instruction {fmt_source(idx)}")
            node = dag.get(idx)
            if node is None:
                return

            visit_state[idx] = 1
            visit(node.s1)
            visit(node.s2)
            visit_state[idx] = 2
            ordered_idxs.append(idx)

        for idx in sorted(dag.keys()):
            visit(idx)

        return ordered_idxs

    def topological_positions() -> Dict[int, int]:
        return {idx: pos for pos, idx in enumerate(topological_dag_order())}

    def source_topological_key(idx: int, positions: Optional[Dict[int, int]] = None) -> Tuple[int, int]:
        if idx not in dag:
            return (0, idx)
        if positions is None:
            positions = topological_positions()
        return (1, positions[idx])

    def ordered_sources(srcs: List[int]) -> List[int]:
        positions = topological_positions()
        return sorted(srcs, key=lambda src: source_topological_key(src, positions))

    def evaluate_masks() -> Dict[int, int]:
        values: Dict[int, int] = {0: all_examples_mask}
        for input_idx, input_mask in enumerate(input_masks):
            values[input_idx + 1] = input_mask

        for idx in topological_dag_order():
            node = dag[idx]
            if node.s1 not in values:
                raise ValueError(f"Missing value for source {node.s1} of node {idx} in {dag} with outputs {outputs}")
            if node.s2 not in values:
                raise ValueError(f"Missing value for source {node.s2} of node {idx} in {dag} with outputs {outputs}")
            values[idx] = _apply_operator(node.op, values[node.s1], values[node.s2], 0) & all_examples_mask

        return values

    def check() -> bool:
        values = evaluate_masks()

        for out_idx in range(num_outputs):
            sel_idx = outputs[out_idx]
            if sel_idx not in values:
                raise ValueError(f"Missing value for output OUT{out_idx} selector {sel_idx}")
            care_mask = expected_output_masks[out_idx]
            if (values[sel_idx] & care_mask) != (expected_output_values[out_idx] & care_mask):
                return False

        return True

    def rebuild_users() -> None:
        for node in dag.values():
            node.users.clear()

        for idx, node in dag.items():
            if node.s1 in dag:
                dag[node.s1].users.add(idx)
            if node.s2 in dag:
                dag[node.s2].users.add(idx)

    def clear_dag() -> None:
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

        for idx in list(dag.keys()):
            if idx not in accessed:
                del dag[idx]

        rebuild_users()

    if not check():
        raise ValueError("Post-processing started with incorrect program")

    def materialize_program() -> Tuple[Program, List[int]]:
        ordered_idxs = topological_dag_order()
        new_idxs = {}
        for new_idx, old_idx in enumerate(ordered_idxs):
            new_idxs[old_idx] = new_idx + (num_inputs + 1)

        materialized_instrs: Program = []
        for idx in ordered_idxs:
            node = dag[idx]
            materialized_instrs.append((node.op, new_idxs.get(node.s1, node.s1), new_idxs.get(node.s2, node.s2)))

        materialized_outputs = [new_idxs.get(sel_idx, sel_idx) for sel_idx in outputs]
        return materialized_instrs, materialized_outputs

    def load_materialized_program(materialized_instrs: Program, materialized_outputs: List[int]) -> None:
        nonlocal dag, outputs
        outputs = list(materialized_outputs)
        dag = {}
        for instr_idx, (op, s1, s2) in enumerate(materialized_instrs):
            if op is None:
                raise ValueError("Cannot load materialized program with unknown operations")
            dag[num_inputs + 1 + instr_idx] = DAGNode(op, s1, s2)
        rebuild_users()

    Score = Tuple[Any, ...]

    def instruction_depths(candidate_instrs: Program) -> Dict[int, int]:
        depths: Dict[int, int] = {idx: 0 for idx in range(num_inputs + 1)}
        for instr_idx, (_op, s1, s2) in enumerate(candidate_instrs):
            idx = num_inputs + 1 + instr_idx
            if s1 not in depths or s2 not in depths:
                raise ValueError(f"Cannot score output depth for non-topological program at {fmt_source(idx)}")
            depths[idx] = max(depths[s1], depths[s2]) + 1
        return depths

    def instruction_tree_sizes(candidate_instrs: Program) -> Dict[int, int]:
        sizes: Dict[int, int] = {idx: 0 for idx in range(num_inputs + 1)}
        for instr_idx, (_op, s1, s2) in enumerate(candidate_instrs):
            idx = num_inputs + 1 + instr_idx
            if s1 not in sizes or s2 not in sizes:
                raise ValueError(f"Cannot score tree size for non-topological program at {fmt_source(idx)}")
            sizes[idx] = sizes[s1] + sizes[s2] + 1
        return sizes

    def output_depths(candidate_instrs: Program, candidate_outputs: List[int]) -> Tuple[int, ...]:
        depths = instruction_depths(candidate_instrs)
        return tuple(depths[sel_idx] for sel_idx in candidate_outputs)

    def output_cone_sizes(candidate_instrs: Program, candidate_outputs: List[int]) -> Tuple[int, ...]:
        sources_by_idx: Dict[int, Tuple[int, int]] = {}
        for instr_idx, (_op, s1, s2) in enumerate(candidate_instrs):
            sources_by_idx[num_inputs + 1 + instr_idx] = (s1, s2)

        def collect_cone(idx: int, cone: set[int]) -> None:
            sources = sources_by_idx.get(idx)
            if sources is None or idx in cone:
                return
            cone.add(idx)
            collect_cone(sources[0], cone)
            collect_cone(sources[1], cone)

        sizes = []
        for sel_idx in candidate_outputs:
            cone: set[int] = set()
            collect_cone(sel_idx, cone)
            sizes.append(len(cone))
        return tuple(sizes)

    def instruction_fanouts(candidate_instrs: Program, candidate_outputs: List[int]) -> Tuple[int, ...]:
        first_instr_idx = num_inputs + 1
        fanouts = {first_instr_idx + instr_idx: 0 for instr_idx in range(len(candidate_instrs))}
        for _op, s1, s2 in candidate_instrs:
            if s1 in fanouts:
                fanouts[s1] += 1
            if s2 in fanouts:
                fanouts[s2] += 1
        for sel_idx in candidate_outputs:
            if sel_idx in fanouts:
                fanouts[sel_idx] += 1
        return tuple(fanouts[first_instr_idx + instr_idx] for instr_idx in range(len(candidate_instrs)))

    def independent_instruction_pairs(candidate_instrs: Program) -> int:
        sources_by_idx: Dict[int, Tuple[int, int]] = {}
        for instr_idx, (_op, s1, s2) in enumerate(candidate_instrs):
            sources_by_idx[num_inputs + 1 + instr_idx] = (s1, s2)

        dependency_cache: Dict[Tuple[int, int], bool] = {}

        def candidate_depends_on(source_idx: int, target_idx: int) -> bool:
            cached = dependency_cache.get((source_idx, target_idx))
            if cached is not None:
                return cached
            if source_idx == target_idx:
                dependency_cache[(source_idx, target_idx)] = True
                return True
            sources = sources_by_idx.get(source_idx)
            if sources is None:
                dependency_cache[(source_idx, target_idx)] = False
                return False
            result = candidate_depends_on(sources[0], target_idx) or candidate_depends_on(sources[1], target_idx)
            dependency_cache[(source_idx, target_idx)] = result
            return result

        instr_idxs = [num_inputs + 1 + instr_idx for instr_idx in range(len(candidate_instrs))]
        count = 0
        for left_pos, left_idx in enumerate(instr_idxs):
            for right_idx in instr_idxs[left_pos + 1:]:
                if not candidate_depends_on(left_idx, right_idx) and not candidate_depends_on(right_idx, left_idx):
                    count += 1
        return count

    def operator_cost(candidate_instrs: Program) -> int:
        costs = {
            OP_BY_LABEL["AND"].code: 1,
            OP_BY_LABEL["OR"].code: 1,
            OP_BY_LABEL["XOR"].code: 2,
        }
        return sum(costs.get(op, 1) for op, _s1, _s2 in candidate_instrs if op is not None)

    def node_value_entropy(candidate_instrs: Program) -> float:
        sample_count = len(candidate_instrs) * example_count
        if sample_count == 0:
            return 0.0

        values: Dict[int, int] = {0: all_examples_mask}
        for input_idx, input_mask in enumerate(input_masks):
            values[input_idx + 1] = input_mask

        true_count = 0
        for instr_idx, (op, s1, s2) in enumerate(candidate_instrs):
            idx = num_inputs + 1 + instr_idx
            if op is None:
                raise ValueError("Cannot score entropy for unknown operation")
            if s1 not in values or s2 not in values:
                raise ValueError(f"Cannot score entropy for non-topological program at {fmt_source(idx)}")
            value = _apply_operator(op, values[s1], values[s2], 0) & all_examples_mask
            values[idx] = value
            true_count += value.bit_count()

        false_count = sample_count - true_count
        entropy = 0.0
        for count in (false_count, true_count):
            if count == 0:
                continue
            probability = count / sample_count
            entropy -= probability * math.log2(probability)
        return entropy

    def reverse_score_value(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(reverse_score_value(item) for item in value)
        return -value

    def program_score(candidate_instrs: Program, candidate_outputs: List[int]) -> Score:
        score_parts: List[Any] = []
        for metric, reverse_sort in active_score_metric_specs:
            if metric == "program-length":
                value = len(candidate_instrs)
            elif metric == "output-depth":
                value = output_depths(candidate_instrs, candidate_outputs)
            elif metric == "max-output-depth":
                depths = output_depths(candidate_instrs, candidate_outputs)
                value = max(depths, default=0)
            elif metric == "sum-output-depth":
                depths = output_depths(candidate_instrs, candidate_outputs)
                value = sum(depths)
            elif metric == "total-node-depth":
                depths_by_idx = instruction_depths(candidate_instrs)
                value = sum(depths_by_idx[num_inputs + 1 + instr_idx] for instr_idx in range(len(candidate_instrs)))
            elif metric == "total-tree-size":
                tree_sizes_by_idx = instruction_tree_sizes(candidate_instrs)
                value = sum(tree_sizes_by_idx[num_inputs + 1 + instr_idx] for instr_idx in range(len(candidate_instrs)))
            elif metric == "operator-cost":
                value = operator_cost(candidate_instrs)
            elif metric == "xor-count":
                value = sum(1 for op, _s1, _s2 in candidate_instrs if op == OP_BY_LABEL["XOR"].code)
            elif metric == "output-cone-size":
                value = output_cone_sizes(candidate_instrs, candidate_outputs)
            elif metric == "max-output-cone-size":
                cone_sizes = output_cone_sizes(candidate_instrs, candidate_outputs)
                value = max(cone_sizes, default=0)
            elif metric == "sum-output-cone-size":
                cone_sizes = output_cone_sizes(candidate_instrs, candidate_outputs)
                value = sum(cone_sizes)
            elif metric == "fanout":
                value = instruction_fanouts(candidate_instrs, candidate_outputs)
            elif metric == "max-fanout":
                fanouts = instruction_fanouts(candidate_instrs, candidate_outputs)
                value = max(fanouts, default=0)
            elif metric == "sum-fanout":
                fanouts = instruction_fanouts(candidate_instrs, candidate_outputs)
                value = sum(fanouts)
            elif metric == "one-fanout-count":
                fanouts = instruction_fanouts(candidate_instrs, candidate_outputs)
                value = sum(1 for fanout in fanouts if fanout == 1)
            elif metric == "independent-pairs":
                value = independent_instruction_pairs(candidate_instrs)
            elif metric == "entropy":
                value = node_value_entropy(candidate_instrs)
            elif metric == "random":
                value = random.random()
            else:
                raise ValueError(f"Unsupported post-process score metric: {metric}")
            score_parts.append(reverse_score_value(value) if reverse_sort else value)

        score_parts.extend([
            tuple(candidate_outputs),
            tuple(candidate_instrs),
        ])
        return tuple(score_parts)

    def compute_signatures() -> Dict[int, int]:
        return evaluate_masks()

    def depends_on(source_idx: int, target_idx: int, cache: Dict[int, bool]) -> bool:
        cached = cache.get(source_idx)
        if cached is not None:
            return cached
        if source_idx == target_idx:
            cache[source_idx] = True
            return True
        source_node = dag.get(source_idx)
        if source_node is None:
            cache[source_idx] = False
            return False
        result = depends_on(source_node.s1, target_idx, cache) or depends_on(source_node.s2, target_idx, cache)
        cache[source_idx] = result
        return result

    def replacement_preserves_outputs(
        modified_idx: int,
        replacement_mask: int,
        current_masks: Dict[int, int],
        dependency_cache: Dict[int, bool],
    ) -> bool:
        memo: Dict[int, int] = {modified_idx: replacement_mask}
        visiting: set[int] = set()

        def hypothetical_value(source_idx: int) -> int:
            memoized = memo.get(source_idx)
            if memoized is not None:
                return memoized
            if source_idx not in dag:
                return current_masks[source_idx]
            if not depends_on(source_idx, modified_idx, dependency_cache):
                return current_masks[source_idx]
            if source_idx in visiting:
                raise ValueError(f"Cycle detected while checking replacement for {fmt_source(modified_idx)}")

            visiting.add(source_idx)
            node = dag[source_idx]
            value = _apply_operator(node.op, hypothetical_value(node.s1), hypothetical_value(node.s2), 0) & all_examples_mask
            visiting.remove(source_idx)
            memo[source_idx] = value
            return value

        for out_idx, sel_idx in enumerate(outputs):
            care_mask = expected_output_masks[out_idx]
            if (hypothetical_value(sel_idx) & care_mask) != (expected_output_values[out_idx] & care_mask):
                return False
        return True

    def snapshot_dag() -> Snapshot:
        return (
            {snap_idx: DAGNode(node.op, node.s1, node.s2) for snap_idx, node in dag.items()},
            list(outputs),
        )

    def program_key(candidate_instrs: Program, candidate_outputs: List[int]) -> ProgramKey:
        return (tuple(candidate_instrs), tuple(candidate_outputs))

    def topology_sorted_nodes(positions: Optional[Dict[int, int]] = None) -> List[int]:
        if positions is None:
            positions = topological_positions()
        return sorted(dag, key=lambda source_idx: source_topological_key(source_idx, positions))

    def topology_sorted_sources(positions: Optional[Dict[int, int]] = None) -> List[int]:
        return list(range(num_inputs + 1)) + topology_sorted_nodes(positions)

    def is_ordered_operand_pair(op: int, s1: int, s2: int, positions: Dict[int, int]) -> bool:
        if op == XOR and s1 == s2:
            return True
        return source_topological_key(s1, positions) < source_topological_key(s2, positions)

    def ordered_pair(s1: int, s2: int, positions: Dict[int, int]) -> Tuple[int, int]:
        if source_topological_key(s1, positions) <= source_topological_key(s2, positions):
            return s1, s2
        return s2, s1

    def existing_node_by_op_sources(positions: Dict[int, int]) -> Dict[Tuple[int, int, int], int]:
        result: Dict[Tuple[int, int, int], int] = {}
        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            s1, s2 = ordered_pair(node.s1, node.s2, positions)
            result.setdefault((node.op, s1, s2), idx)
        return result

    def redirect_source(old_idx: int, new_idx: int) -> None:
        for out_idx, sel_idx in enumerate(outputs):
            if sel_idx == old_idx:
                outputs[out_idx] = new_idx

        for node in dag.values():
            if node.s1 == old_idx:
                node.s1 = new_idx
            if node.s2 == old_idx:
                node.s2 = new_idx

    def restore_snapshot(snapshot: Snapshot) -> None:
        nonlocal dag, outputs
        dag = snapshot[0]
        outputs = snapshot[1]
        rebuild_users()

    def canonicalize_dag() -> None:
        positions = topological_positions()
        for node in dag.values():
            if source_topological_key(node.s1, positions) > source_topological_key(node.s2, positions):
                node.s1, node.s2 = node.s2, node.s1
        rebuild_users()

    active_rejection_counts: Optional[Dict[str, int]] = None

    def record_rejection(reason: str) -> None:
        if active_rejection_counts is None:
            return
        active_rejection_counts[reason] = active_rejection_counts.get(reason, 0) + 1

    def process_materialized_candidate(
        strategy: str,
        description: str,
        base_score: Score,
        candidate_instrs: Program,
        candidate_outputs: List[int],
    ) -> Optional[Tuple[Score, str, Program, List[int]]]:
        snapshot = snapshot_dag()
        finalization_failed = False
        try:
            load_materialized_program(candidate_instrs, candidate_outputs)
            canonicalize_dag()
            clear_dag()
            processed_instrs, processed_outputs = materialize_program()
            valid = check()
        except ValueError as exc:
            record_rejection(f"{strategy}:invalid-finalization:{type(exc).__name__}")
            finalization_failed = True
            valid = False
            processed_instrs, processed_outputs = candidate_instrs, candidate_outputs
        restore_snapshot(snapshot)
        if not valid:
            if not finalization_failed:
                record_rejection(f"{strategy}:incorrect")
            return None
        score = program_score(processed_instrs, processed_outputs)
        if score >= base_score:
            return None
        return (score, f"{strategy}:{description}", processed_instrs, processed_outputs)

    Candidate = Tuple[Score, str, Program, List[int]]

    def generate_afterburner_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        if not enable_afterburner or not examples:
            return

        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        signatures = compute_signatures()
        representative_by_signature: Dict[int, int] = {}

        for idx in topology_sorted_sources():
            signature = signatures[idx]
            representative = representative_by_signature.get(signature)
            if representative is None:
                representative_by_signature[signature] = idx
                continue
            if idx not in dag:
                continue

            snapshot = snapshot_dag()
            redirect_source(idx, representative)
            candidate_instrs, candidate_outputs = materialize_program()
            restore_snapshot(snapshot)

            candidate = process_materialized_candidate("afterburner", f"{fmt_source(idx)}->{fmt_source(representative)}", base_score, candidate_instrs, candidate_outputs)
            if candidate is not None:
                consider_candidate(candidate)

    def generate_replacement_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        current_masks = evaluate_masks()

        positions = topological_positions()
        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            dependency_cache: Dict[int, bool] = {}
            possible_idxs = [
                x
                for x in list(range(num_inputs + 1)) + list(dag.keys())
                if x != idx and not depends_on(x, idx, dependency_cache)
            ]
            product = [
                (op, s1, s2)
                for op in logic_operator_codes
                for s1 in possible_idxs
                for s2 in possible_idxs
                if is_ordered_operand_pair(op, s1, s2, positions)
            ]
            random.shuffle(product)

            patience = post_process_replace_patience
            valid_replacement_mask_cache: Dict[int, bool] = {current_masks[idx]: True}
            for op, s1, s2 in product:
                candidate_mask = _apply_operator(op, current_masks[s1], current_masks[s2], 0) & all_examples_mask
                valid_mask = valid_replacement_mask_cache.get(candidate_mask)
                if valid_mask is None:
                    valid_mask = replacement_preserves_outputs(idx, candidate_mask, current_masks, dependency_cache)
                    valid_replacement_mask_cache[candidate_mask] = valid_mask
                if not valid_mask:
                    continue

                snapshot = snapshot_dag()
                dag[idx].op = op
                dag[idx].s1 = s1
                dag[idx].s2 = s2
                try:
                    candidate_instrs, candidate_outputs = materialize_program()
                except ValueError as exc:
                    record_rejection(f"replace:invalid-materialize:{type(exc).__name__}")
                    continue
                finally:
                    restore_snapshot(snapshot)

                candidate = process_materialized_candidate("replace", f"{fmt_source(idx)}", base_score, candidate_instrs, candidate_outputs)
                if candidate is not None:
                    consider_candidate(candidate)

                if post_process_replace_patience > 0:
                    patience -= 1
                if post_process_replace_patience > 0 and patience <= 0:
                    break

    def generate_adjustment_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)

        positions = topological_positions()
        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            if len(node.users) != 1 or idx in outputs:
                continue
            user_idx = next(iter(node.users))
            user_node = dag[user_idx]
            if node.op != user_node.op:
                continue

            if user_node.s1 == idx:
                user_other_source = user_node.s2
            elif user_node.s2 == idx:
                user_other_source = user_node.s1
            else:
                raise ValueError(f"Node {idx} is not a source of its user {user_idx} in {dag}")

            original_inner_sources = ordered_sources([node.s1, node.s2])
            original_remaining_source = user_other_source
            seen_regroupings: set[Tuple[int, int, int]] = set()

            for inner_a, inner_b, remaining_source in itertools.permutations([node.s1, node.s2, user_other_source], 3):
                inner_sources = ordered_sources([inner_a, inner_b])
                grouping_key = (inner_sources[0], inner_sources[1], remaining_source)
                if grouping_key in seen_regroupings:
                    continue
                seen_regroupings.add(grouping_key)
                if inner_sources == original_inner_sources and remaining_source == original_remaining_source:
                    continue
                dependency_cache: Dict[int, bool] = {}
                if any(depends_on(source, idx, dependency_cache) for source in inner_sources):
                    continue

                snapshot = snapshot_dag()
                dag[idx].s1, dag[idx].s2 = inner_sources[0], inner_sources[1]
                try:
                    current_positions = topological_positions()
                except ValueError as exc:
                    record_rejection(f"adjust:cycle:{type(exc).__name__}")
                    restore_snapshot(snapshot)
                    continue
                if source_topological_key(idx, current_positions) < source_topological_key(remaining_source, current_positions):
                    dag[user_idx].s1, dag[user_idx].s2 = idx, remaining_source
                else:
                    dag[user_idx].s1, dag[user_idx].s2 = remaining_source, idx
                try:
                    candidate_instrs, candidate_outputs = materialize_program()
                except ValueError as exc:
                    record_rejection(f"adjust:invalid-materialize:{type(exc).__name__}")
                    restore_snapshot(snapshot)
                    continue
                restore_snapshot(snapshot)

                candidate = process_materialized_candidate("adjust", f"{fmt_source(idx)}->{fmt_source(user_idx)}", base_score, candidate_instrs, candidate_outputs)
                if candidate is not None:
                    consider_candidate(candidate)

    def generate_existing_regroup_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        positions = topological_positions()
        existing_nodes = existing_node_by_op_sources(positions)

        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            for child_idx, remaining_source in ((node.s1, node.s2), (node.s2, node.s1)):
                child_node = dag.get(child_idx)
                if child_node is None or child_node.op != node.op:
                    continue

                seen_regroupings: set[Tuple[int, int, int]] = set()
                sources = [child_node.s1, child_node.s2, remaining_source]
                for inner_a, inner_b, outer_source in itertools.permutations(sources, 3):
                    inner_s1, inner_s2 = ordered_pair(inner_a, inner_b, positions)
                    grouping_key = (inner_s1, inner_s2, outer_source)
                    if grouping_key in seen_regroupings:
                        continue
                    seen_regroupings.add(grouping_key)

                    existing_inner = existing_nodes.get((node.op, inner_s1, inner_s2))
                    if existing_inner is None or existing_inner == child_idx:
                        continue
                    dependency_cache: Dict[int, bool] = {}
                    if depends_on(existing_inner, idx, dependency_cache) or depends_on(outer_source, idx, dependency_cache):
                        continue

                    new_s1, new_s2 = ordered_pair(existing_inner, outer_source, positions)
                    if (new_s1, new_s2) == ordered_pair(node.s1, node.s2, positions):
                        continue

                    snapshot = snapshot_dag()
                    dag[idx].s1, dag[idx].s2 = new_s1, new_s2
                    try:
                        candidate_instrs, candidate_outputs = materialize_program()
                    except ValueError as exc:
                        record_rejection(f"existing-regroup:invalid-materialize:{type(exc).__name__}")
                        restore_snapshot(snapshot)
                        continue
                    restore_snapshot(snapshot)

                    candidate = process_materialized_candidate("existing-regroup", f"{fmt_source(idx)}->{fmt_source(existing_inner)}", base_score, candidate_instrs, candidate_outputs)
                    if candidate is not None:
                        consider_candidate(candidate)

    def generate_single_use_bypass_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        values = evaluate_masks()
        positions = topological_positions()
        sources_by_topology = topology_sorted_sources(positions)

        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            if len(node.users) != 1:
                continue
            user_idx = next(iter(node.users))
            user_mask = values[user_idx]

            for replacement in sources_by_topology:
                if replacement == user_idx or values[replacement] != user_mask:
                    continue
                dependency_cache: Dict[int, bool] = {}
                if depends_on(replacement, user_idx, dependency_cache):
                    continue

                snapshot = snapshot_dag()
                redirect_source(user_idx, replacement)
                try:
                    candidate_instrs, candidate_outputs = materialize_program()
                except ValueError as exc:
                    record_rejection(f"single-use-bypass:invalid-materialize:{type(exc).__name__}")
                    restore_snapshot(snapshot)
                    continue
                restore_snapshot(snapshot)

                candidate = process_materialized_candidate("single-use-bypass", f"{fmt_source(user_idx)}->{fmt_source(replacement)}", base_score, candidate_instrs, candidate_outputs)
                if candidate is not None:
                    consider_candidate(candidate)

    def generate_xor_common_cancellation_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        positions = topological_positions()

        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            if node.op != XOR:
                continue
            left_node = dag.get(node.s1)
            right_node = dag.get(node.s2)
            if left_node is None or right_node is None or left_node.op != XOR or right_node.op != XOR:
                continue

            for common_source in set((left_node.s1, left_node.s2)).intersection((right_node.s1, right_node.s2)):
                left_remaining = left_node.s2 if left_node.s1 == common_source else left_node.s1
                right_remaining = right_node.s2 if right_node.s1 == common_source else right_node.s1
                new_s1, new_s2 = ordered_pair(left_remaining, right_remaining, positions)
                if (new_s1, new_s2) == ordered_pair(node.s1, node.s2, positions):
                    continue

                snapshot = snapshot_dag()
                dag[idx].s1, dag[idx].s2 = new_s1, new_s2
                try:
                    candidate_instrs, candidate_outputs = materialize_program()
                except ValueError as exc:
                    record_rejection(f"xor-common-cancel:invalid-materialize:{type(exc).__name__}")
                    restore_snapshot(snapshot)
                    continue
                restore_snapshot(snapshot)

                candidate = process_materialized_candidate("xor-common-cancel", f"{fmt_source(idx)}:{fmt_source(common_source)}", base_score, candidate_instrs, candidate_outputs)
                if candidate is not None:
                    consider_candidate(candidate)

    def generate_equivalent_operand_swap_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        values = evaluate_masks()
        positions = topological_positions()
        sources_by_topology = topology_sorted_sources(positions)

        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            original_sources = (node.s1, node.s2)
            for operand_idx, old_source in enumerate(original_sources):
                old_key = source_topological_key(old_source, positions)
                for replacement in sources_by_topology:
                    if replacement == old_source or values[replacement] != values[old_source]:
                        continue
                    if source_topological_key(replacement, positions) >= old_key:
                        continue
                    dependency_cache: Dict[int, bool] = {}
                    if depends_on(replacement, idx, dependency_cache):
                        continue

                    new_sources = [node.s1, node.s2]
                    new_sources[operand_idx] = replacement
                    new_s1, new_s2 = ordered_pair(new_sources[0], new_sources[1], positions)

                    snapshot = snapshot_dag()
                    dag[idx].s1, dag[idx].s2 = new_s1, new_s2
                    try:
                        candidate_instrs, candidate_outputs = materialize_program()
                    except ValueError as exc:
                        record_rejection(f"equiv-operand:invalid-materialize:{type(exc).__name__}")
                        restore_snapshot(snapshot)
                        continue
                    restore_snapshot(snapshot)

                    candidate = process_materialized_candidate("equiv-operand", f"{fmt_source(idx)}:{fmt_source(old_source)}->{fmt_source(replacement)}", base_score, candidate_instrs, candidate_outputs)
                    if candidate is not None:
                        consider_candidate(candidate)

    def generate_local_resynthesis_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        if example_count == 0:
            return

        @dataclass(frozen=True)
        class ResynthesisWindow:
            strategy: str
            nodes: Tuple[int, ...]
            outputs: Tuple[int, ...]

            @property
            def description(self) -> str:
                return f"{fmt_source(self.nodes[0])}->{fmt_source(self.outputs[-1])}"

        def canonical_window_nodes(nodes: Iterable[int]) -> Tuple[int, ...]:
            return tuple(sorted(set(nodes), key=lambda idx: source_topological_key(idx, positions)))

        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        positions = topological_positions()
        values = evaluate_masks()

        def try_resynthesize_window(
            window: ResynthesisWindow,
            tag: str,
        ) -> Tuple[Optional[Candidate], bool]:
            window_nodes = list(window.nodes)
            window_outputs = list(window.outputs)
            if len(window_nodes) < 2:
                return None, False

            window_set = set(window_nodes)
            external_sources = sorted(
                {
                    src
                    for idx in window_nodes
                    for src in (dag[idx].s1, dag[idx].s2)
                    if src not in window_set
                },
                key=lambda src: source_topological_key(src, positions),
            )
            if not external_sources:
                return None, False

            local_length = len(window_nodes) - 1
            spec = ProgramSpec(
                num_inputs=len(external_sources),
                num_outputs=len(window_outputs),
                program_length=local_length,
            )
            solver = _make_solver("simple-tactic")
            solver.add(*_build_program(spec, EncodingOptions()))
            local_input_masks = [values[src] for src in external_sources]
            constraints, local_outputs = _build_test(example_count, local_input_masks, tag=tag, spec=spec, options=EncodingOptions())
            solver.add(*constraints)
            for out_idx, output_node in enumerate(window_outputs):
                solver.add(local_outputs[out_idx] == BitVecVal(values[output_node], example_count))

            result = solver.check()
            if str(result) != "sat":
                return None, False

            model = solver.model()

            def eval_bv_as_long(name: str, var: BitVecRef) -> int:
                value = model.eval(var, model_completion=True)
                if not isinstance(value, BitVecNumRef):
                    raise ValueError(f"Model did not assign {name}: {value}")
                return value.as_long()

            local_instrs: Program = []
            for instr_idx in range(local_length):
                op = eval_bv_as_long(f"OP_{instr_idx}", BitVec(f"OP_{instr_idx}", OP_BITS))
                s1 = eval_bv_as_long(f"S1_{instr_idx}", BitVec(f"S1_{instr_idx}", spec.idx_bits))
                s2 = eval_bv_as_long(f"S2_{instr_idx}", BitVec(f"S2_{instr_idx}", spec.idx_bits))
                local_instrs.append((op, s1, s2))

            local_output_selectors = [
                eval_bv_as_long(f"OUT_{out_idx}_idx", BitVec(f"OUT_{out_idx}_idx", spec.idx_bits))
                for out_idx in range(len(window_outputs))
            ]
            fresh_base_idx = max(dag.keys(), default=num_inputs) + 1
            fresh_idxs = [fresh_base_idx + instr_idx for instr_idx in range(local_length)]

            def translate_local_source(source_idx: int) -> int:
                if source_idx == 0:
                    return 0
                if 1 <= source_idx <= len(external_sources):
                    return external_sources[source_idx - 1]
                local_instr_idx = source_idx - len(external_sources) - 1
                if not 0 <= local_instr_idx < len(fresh_idxs):
                    raise ValueError(f"Local synthesized source out of range: {source_idx}")
                return fresh_idxs[local_instr_idx]

            snapshot = snapshot_dag()
            try:
                for local_idx, (op, s1, s2) in enumerate(local_instrs):
                    dag[fresh_idxs[local_idx]] = DAGNode(op, translate_local_source(s1), translate_local_source(s2))
                for output_node, local_output in zip(window_outputs, local_output_selectors):
                    redirect_source(output_node, translate_local_source(local_output))
                candidate_instrs, candidate_outputs = materialize_program()
            except ValueError as exc:
                record_rejection(f"{window.strategy}:invalid-materialize:{type(exc).__name__}")
                restore_snapshot(snapshot)
                return None, True
            restore_snapshot(snapshot)

            return process_materialized_candidate(window.strategy, window.description, base_score, candidate_instrs, candidate_outputs), True

        output_uses: Dict[int, int] = {}
        for sel_idx in outputs:
            if sel_idx in dag:
                output_uses[sel_idx] = output_uses.get(sel_idx, 0) + 1

        def total_fanout(idx: int) -> int:
            return len(dag[idx].users) + output_uses.get(idx, 0)

        component_nodes: set[int] = set()
        for idx in dag:
            if total_fanout(idx) != 1:
                continue
            node = dag[idx]
            if (node.s1 in dag and total_fanout(node.s1) == 1) or (node.s2 in dag and total_fanout(node.s2) == 1):
                component_nodes.add(idx)

        parent: Dict[int, int] = {}

        def find(idx: int) -> int:
            parent.setdefault(idx, idx)
            if parent[idx] != idx:
                parent[idx] = find(parent[idx])
            return parent[idx]

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for idx in component_nodes:
            find(idx)
        for idx in component_nodes:
            node = dag[idx]
            for src in (node.s1, node.s2):
                if src in component_nodes:
                    union(idx, src)

        components_by_root: Dict[int, List[int]] = {}
        for idx in component_nodes:
            components_by_root.setdefault(find(idx), []).append(idx)

        def component_output(component: set[int]) -> Optional[int]:
            exits = [
                idx
                for idx in component
                if any(user not in component for user in dag[idx].users) or output_uses.get(idx, 0) > 0
            ]
            if len(exits) != 1:
                return None
            return exits[0]

        def closed_subcomponents(component: set[int], size: int) -> List[Tuple[int, ...]]:
            result: set[Tuple[int, ...]] = set()

            def dependencies_inside(idx: int) -> List[int]:
                node = dag[idx]
                return [src for src in (node.s1, node.s2) if src in component]

            for root_idx in component:
                stack: List[Tuple[frozenset[int], Tuple[int, ...]]] = [(frozenset([root_idx]), tuple(dependencies_inside(root_idx)))]
                while stack:
                    selected, frontier = stack.pop()
                    if len(selected) == size:
                        result.add(canonical_window_nodes(selected))
                        continue
                    if len(selected) > size or not frontier:
                        continue
                    candidate_idx = frontier[0]
                    rest_frontier = frontier[1:]

                    stack.append((selected, rest_frontier))

                    new_selected = set(selected)
                    new_selected.add(candidate_idx)
                    new_frontier = list(rest_frontier)
                    for dep_idx in dependencies_inside(candidate_idx):
                        if dep_idx not in new_selected and dep_idx not in new_frontier:
                            new_frontier.append(dep_idx)
                    stack.append((frozenset(new_selected), tuple(new_frontier)))

            return sorted(result, key=lambda nodes: (source_topological_key(nodes[-1], positions), nodes))

        windows: List[ResynthesisWindow] = []
        seen_windows: set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()

        def add_window(nodes: Tuple[int, ...], output_idx: int) -> None:
            if len(nodes) < 2:
                return
            key = (nodes, (output_idx,))
            if key in seen_windows:
                return
            seen_windows.add(key)
            windows.append(ResynthesisWindow("component-sat", nodes, (output_idx,)))

        for component in components_by_root.values():
            component_tuple = canonical_window_nodes(component)
            component_set = set(component_tuple)
            if len(component_tuple) < 2:
                continue
            if len(component_tuple) <= post_process_resynthesis_maxnodes:
                output_idx = component_output(component_set)
                if output_idx is not None:
                    add_window(component_tuple, output_idx)
                continue

            for nodes in closed_subcomponents(component_set, post_process_resynthesis_maxnodes):
                output_idx = component_output(set(nodes))
                if output_idx is not None:
                    add_window(nodes, output_idx)

        random.shuffle(windows)

        sat_results = 0
        for window_idx, window in enumerate(windows):
            if post_process_resynthesis_patience > 0 and sat_results >= post_process_resynthesis_patience:
                break

            candidate, was_sat = try_resynthesize_window(
                window,
                f"resynth{window_idx}",
            )
            if was_sat:
                sat_results += 1
            if candidate is not None:
                consider_candidate(candidate)

    def generate_simplification_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        values = evaluate_masks()
        positions = topological_positions()
        false_sources = [idx for idx, value in values.items() if value == 0]
        false_source = min(false_sources, key=lambda source_idx: source_topological_key(source_idx, positions)) if false_sources else None

        def other_source(node: DAGNode, source: int) -> Optional[int]:
            if node.s1 == source:
                return node.s2
            if node.s2 == source:
                return node.s1
            return None

        def propose(idx: int, replacement: Optional[int], reason: str) -> None:
            if replacement is None or replacement == idx:
                return
            snapshot = snapshot_dag()
            redirect_source(idx, replacement)
            try:
                candidate_instrs, candidate_outputs = materialize_program()
            except ValueError as exc:
                record_rejection(f"simplify:invalid-materialize:{type(exc).__name__}")
                restore_snapshot(snapshot)
                return
            restore_snapshot(snapshot)

            candidate = process_materialized_candidate("simplify", f"{fmt_source(idx)}:{reason}", base_score, candidate_instrs, candidate_outputs)
            if candidate is not None:
                consider_candidate(candidate)

        for idx in topology_sorted_nodes(positions):
            node = dag[idx]
            s1 = node.s1
            s2 = node.s2

            if node.op == OP_BY_LABEL["AND"].code:
                if s1 == s2:
                    propose(idx, s1, "idempotent")
                if s1 == 0:
                    propose(idx, s2, "identity")
                if s2 == 0:
                    propose(idx, s1, "identity")
                if s1 == false_source or s2 == false_source:
                    propose(idx, false_source, "annihilator")

                s1_node = dag.get(s1)
                s2_node = dag.get(s2)
                if s1_node is not None and s1_node.op == OP_BY_LABEL["OR"].code:
                    absorbed = other_source(s1_node, s2)
                    if absorbed is not None:
                        propose(idx, s2, "absorption")
                if s2_node is not None and s2_node.op == OP_BY_LABEL["OR"].code:
                    absorbed = other_source(s2_node, s1)
                    if absorbed is not None:
                        propose(idx, s1, "absorption")

            elif node.op == OP_BY_LABEL["OR"].code:
                if s1 == s2:
                    propose(idx, s1, "idempotent")
                if s1 == false_source:
                    propose(idx, s2, "identity")
                if s2 == false_source:
                    propose(idx, s1, "identity")
                if s1 == 0 or s2 == 0:
                    propose(idx, 0, "annihilator")

                s1_node = dag.get(s1)
                s2_node = dag.get(s2)
                if s1_node is not None and s1_node.op == OP_BY_LABEL["AND"].code:
                    absorbed = other_source(s1_node, s2)
                    if absorbed is not None:
                        propose(idx, s2, "absorption")
                if s2_node is not None and s2_node.op == OP_BY_LABEL["AND"].code:
                    absorbed = other_source(s2_node, s1)
                    if absorbed is not None:
                        propose(idx, s1, "absorption")

            elif node.op == XOR:
                if s1 == s2:
                    propose(idx, false_source, "self-cancel")
                if s1 == false_source:
                    propose(idx, s2, "identity")
                if s2 == false_source:
                    propose(idx, s1, "identity")

                s1_node = dag.get(s1)
                s2_node = dag.get(s2)
                if s1_node is not None and s1_node.op == XOR:
                    remaining = other_source(s1_node, s2)
                    if remaining is not None:
                        propose(idx, remaining, "cancel")
                if s2_node is not None and s2_node.op == XOR:
                    remaining = other_source(s2_node, s1)
                    if remaining is not None:
                        propose(idx, remaining, "cancel")

    def generate_output_candidates(consider_candidate: Callable[[Candidate], None]) -> None:
        base_instrs, base_outputs = materialize_program()
        base_score = program_score(base_instrs, base_outputs)
        values = evaluate_masks()
        positions = topological_positions()
        sources_by_topology = topology_sorted_sources(positions)

        for out_idx in range(num_outputs):
            sel_idx = outputs[out_idx]
            care_mask = expected_output_masks[out_idx]
            for cidx in sources_by_topology:
                if source_topological_key(cidx, positions) >= source_topological_key(sel_idx, positions):
                    continue
                if (values[cidx] & care_mask) != (expected_output_values[out_idx] & care_mask):
                    continue
                candidate_outputs = list(outputs)
                candidate_outputs[out_idx] = cidx
                snapshot = snapshot_dag()
                outputs[out_idx] = cidx
                candidate_instrs, candidate_outputs = materialize_program()
                restore_snapshot(snapshot)
                candidate = process_materialized_candidate("output", f"OUT{out_idx}", base_score, candidate_instrs, candidate_outputs)
                if candidate is not None:
                    consider_candidate(candidate)

    def generate_beam_candidates(
        candidate_limit: int,
        seen_programs: set[ProgramKey],
    ) -> List[Candidate]:
        result: List[Candidate] = []
        result_keys: Dict[ProgramKey, int] = {}

        def rebuild_result_keys() -> None:
            result_keys.clear()
            for idx, item in enumerate(result):
                result_keys[program_key(item[2], item[3])] = idx

        def consider_candidate(candidate: Candidate) -> None:
            key = program_key(candidate[2], candidate[3])
            if key in seen_programs:
                return
            existing_idx = result_keys.get(key)
            if existing_idx is not None:
                if candidate[0] >= result[existing_idx][0]:
                    return
                result[existing_idx] = candidate
                result.sort(key=lambda item: item[0])
                rebuild_result_keys()
                return

            if candidate_limit <= 0:
                result.append(candidate)
                result_keys[key] = len(result) - 1
                return

            if len(result) < candidate_limit:
                result.append(candidate)
                result.sort(key=lambda item: item[0])
                rebuild_result_keys()
                return

            if candidate[0] >= result[-1][0]:
                return

            del result_keys[program_key(result[-1][2], result[-1][3])]
            result[-1] = candidate
            result.sort(key=lambda item: item[0])
            rebuild_result_keys()

        for generator in (
            generate_afterburner_candidates,
            generate_replacement_candidates,
            generate_adjustment_candidates,
            generate_existing_regroup_candidates,
            generate_single_use_bypass_candidates,
            generate_xor_common_cancellation_candidates,
            generate_equivalent_operand_swap_candidates,
            generate_local_resynthesis_candidates,
            generate_simplification_candidates,
            generate_output_candidates,
        ):
            generator(consider_candidate)
        result.sort(key=lambda item: item[0])
        return result

    def log_beam_output_candidate(round_idx: int, beam_idx: int, candidate: Candidate) -> None:
        score, description, candidate_instrs, _candidate_outputs = candidate
        logger.info(
            "Post-process beam round %d state %d output candidate %s score=%s length=%d",
            round_idx,
            beam_idx,
            description,
            score[:len(active_score_metric_specs)],
            len(candidate_instrs),
        )

    post_process_interrupted = False

    def run_beam(phase_idx: int) -> Optional[Tuple[Program, List[int]]]:
        nonlocal active_rejection_counts, post_process_interrupted
        canonicalize_dag()
        clear_dag()
        start_instrs, start_outputs = materialize_program()
        start_key = program_key(start_instrs, start_outputs)
        best_instrs = start_instrs
        best_outputs = start_outputs
        best_score = program_score(best_instrs, best_outputs)
        start_score = best_score
        beam: List[Tuple[Program, List[int]]] = [(start_instrs, start_outputs)]
        seen = {start_key}
        round_idx = 0
        next_states: List[Tuple[Score, Program, List[int]]] = []

        try:
            while True:
                if post_process_beam_rounds > 0 and round_idx >= post_process_beam_rounds:
                    break
                round_idx += 1
                next_states = []
                output_counts_by_strategy: Dict[str, int] = {}
                active_rejection_counts = {}

                for beam_idx, (state_instrs, state_outputs) in enumerate(beam):
                    load_materialized_program(state_instrs, state_outputs)
                    for candidate in generate_beam_candidates(post_process_beam_candidates, seen):
                        score, description, candidate_instrs, candidate_outputs = candidate
                        key = program_key(candidate_instrs, candidate_outputs)
                        if key in seen:
                            continue
                        seen.add(key)
                        log_beam_output_candidate(round_idx, beam_idx, candidate)
                        strategy = description.split(":", 1)[0]
                        output_counts_by_strategy[strategy] = output_counts_by_strategy.get(strategy, 0) + 1
                        next_states.append((score, candidate_instrs, candidate_outputs))

                if active_rejection_counts:
                    logger.info(
                        "Post-process beam round %d rejected candidate counts: %s",
                        round_idx,
                        dict(sorted(active_rejection_counts.items())),
                    )
                active_rejection_counts = None

                if not next_states:
                    break

                logger.info(
                    "Post-process beam round %d output candidate counts by strategy: %s",
                    round_idx,
                    dict(sorted(output_counts_by_strategy.items())),
                )

                next_states.sort(key=lambda item: item[0])
                beam = [(candidate_instrs, candidate_outputs) for _score, candidate_instrs, candidate_outputs in next_states[:post_process_beam_width]]

                if next_states[0][0] < best_score:
                    best_score, best_instrs, best_outputs = next_states[0]
        except KeyboardInterrupt:
            active_rejection_counts = None
            post_process_interrupted = True
            if next_states:
                next_states.sort(key=lambda item: item[0])
                if next_states[0][0] < best_score:
                    best_score, best_instrs, best_outputs = next_states[0]
            logger.warning(
                "Post-processing interrupted during beam phase %d; returning best candidate found so far",
                phase_idx,
            )

        load_materialized_program(start_instrs, start_outputs)
        if best_score < start_score:
            logger.info(
                "Accepted post-process beam phase %d reducing program from %d to %d instructions",
                phase_idx,
                len(start_instrs),
                len(best_instrs),
            )
            return best_instrs, best_outputs

        if post_process_interrupted:
            return start_instrs, start_outputs

        return None

    for phase_idx, phase_metrics in enumerate(score_phases, start=1):
        active_score_metric_specs = score_phase_specs[phase_idx - 1]
        logger.info(
            "Starting post-process beam phase %d/%d with score metrics %s",
            phase_idx,
            len(score_phases),
            phase_metrics,
        )
        beam_result = run_beam(phase_idx)
        if beam_result is not None:
            load_materialized_program(*beam_result)
        if post_process_interrupted:
            break

    instrs, outputs = materialize_program()

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
    parser.add_argument("--post-process-beam-width", type=int, default=1, help="Beam width for post-process neighbor exploration")
    parser.add_argument("--post-process-beam-rounds", type=int, default=0, help="Maximum post-process beam search rounds (0 means until no better candidates are found)")
    parser.add_argument("--post-process-beam-candidates", type=int, default=0, help="Maximum post-process neighbor candidates generated per beam state (0 means unlimited)")
    parser.add_argument("--post-process-replace-patience", type=int, default=50, help="Replacement attempts accepted per modified node before moving to the next node (0 means unlimited)")
    parser.add_argument("--post-process-resynthesis-maxnodes", type=int, default=5, help="Maximum one-fanout component window size considered by local SAT resynthesis")
    parser.add_argument("--post-process-resynthesis-patience", type=int, default=1, help="Maximum SAT calls made by local resynthesis per beam state (0 means unlimited)")
    parser.add_argument(
        "--post-process-score",
        type=str,
        default="program-length",
        help=(
            "Comma-separated lexicographic post-process score metrics: "
            "program-length, output-depth, max-output-depth, sum-output-depth, "
            "total-node-depth, total-tree-size, operator-cost, xor-count, output-cone-size, "
            "max-output-cone-size, sum-output-cone-size, fanout, max-fanout, "
            "sum-fanout, one-fanout-count, independent-pairs, entropy, random. "
            "Prefix a metric with '-' to sort it descending. Separate phases "
            "with ';' to continue beam search with the next score after the "
            "previous phase finishes."
        ),
    )

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

    if args.post_process_beam_width < 1:
        raise SystemExit("--post-process-beam-width must be at least 1")
    if args.post_process_beam_rounds < 0:
        raise SystemExit("--post-process-beam-rounds must be non-negative")
    if args.post_process_beam_candidates < 0:
        raise SystemExit("--post-process-beam-candidates must be non-negative")
    if args.post_process_replace_patience < 0:
        raise SystemExit("--post-process-replace-patience must be non-negative")
    if args.post_process_resynthesis_maxnodes < 2:
        raise SystemExit("--post-process-resynthesis-maxnodes must be at least 2")
    if args.post_process_resynthesis_patience < 0:
        raise SystemExit("--post-process-resynthesis-patience must be non-negative")
    post_process_score = [
        [metric.strip() for metric in phase.split(",") if metric.strip()]
        for phase in args.post_process_score.split(";")
    ]
    valid_post_process_score_metrics = {
        "program-length",
        "output-depth",
        "max-output-depth",
        "sum-output-depth",
        "total-node-depth",
        "total-tree-size",
        "operator-cost",
        "xor-count",
        "output-cone-size",
        "max-output-cone-size",
        "sum-output-cone-size",
        "fanout",
        "max-fanout",
        "sum-fanout",
        "one-fanout-count",
        "independent-pairs",
        "entropy",
        "random",
    }
    if not post_process_score:
        raise SystemExit("--post-process-score must specify at least one metric")
    for phase in post_process_score:
        if not phase:
            raise SystemExit("--post-process-score contains an empty phase")
        for metric in phase:
            metric_name = metric[1:] if metric.startswith("-") else metric
            if not metric_name:
                raise SystemExit("--post-process-score contains an empty metric")
            if metric_name not in valid_post_process_score_metrics:
                raise SystemExit(f"Unsupported --post-process-score metric: {metric}")

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
    interrupted_during_generation = False
    try:
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
    except KeyboardInterrupt:
        interrupted_during_generation = True

    elapsed = time.time() - start

    if interrupted_during_generation:
        logger.info("UNKNOWN result: interrupted during program generation in %.3f seconds", elapsed)
        print("# UNKNOWN: interrupted during program generation; result is not conclusive")
        exit(1)

    if args.make_smt2:
        try:
            print(s.to_smt2())
            return
        except KeyboardInterrupt:
            logger.info("UNKNOWN result: interrupted during SMT-LIB export in %.3f seconds", time.time() - start)
            print("# UNKNOWN: interrupted during SMT-LIB export; result is not conclusive")
            exit(1)

    if args.make_dimacs:
        try:
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
        except KeyboardInterrupt:
            logger.info("UNKNOWN result: interrupted during DIMACS export in %.3f seconds", time.time() - start)
            print("c UNKNOWN: interrupted during DIMACS export; result is not conclusive")
            exit(1)

    if args.do_all:
        try:
            result = s.check()
        except KeyboardInterrupt:
            interrupted_during_generation = True

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
            instrs, outputs = _post_process_program(
                instrs,
                spec.num_inputs,
                spec.num_outputs,
                examples,
                outputs,
                post_process_beam_width=args.post_process_beam_width,
                post_process_beam_rounds=args.post_process_beam_rounds,
                post_process_beam_candidates=args.post_process_beam_candidates,
                post_process_replace_patience=args.post_process_replace_patience,
                post_process_resynthesis_maxnodes=args.post_process_resynthesis_maxnodes,
                post_process_resynthesis_patience=args.post_process_resynthesis_patience,
                post_process_score=post_process_score,
            )

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
