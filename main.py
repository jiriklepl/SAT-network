#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
import time
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any, Literal, TypeVar

from z3 import (
    BitVec,
    BitVecVal,
    If,
    Or,
    Tactic,
    Then,
    ULE,
    ULT,
    Bool,
    Implies,
    Solver,
    BoolRef,
    BitVecRef,
    Goal,
    BitVecNumRef,
    Tactic,
)

from dataset_plugins import get_plugin, available_plugins
import dataset_plugins.gol  # ensures GoL plugin registration
import dataset_plugins.adder3  # ensures 3-bit adder plugin registration

# Defaults (overridden by config/CLI)
NUM_INPUTS = 7
NUM_OUTPUTS = 1
PROGRAM_LENGTH = 16

force_unique = True
encode_boolean = True

OP_BITS = 2  # encode OR, AND, XOR
OP_LABELS = {0: 'OR', 1: 'AND', 2: 'XOR'}


T = TypeVar('T')
def _select_bv(values: List[T], idx_var: BitVecNumRef | BitVecRef, bits: int) -> T:
    if not values or len(values) == 0:
        raise ValueError("values must be a non-empty list")

    result : T = values[0]
    for idx, value in enumerate(values):
        if idx == 0:
            continue
        result = If(idx_var == BitVecVal(idx, bits), value, result)

    return result


def build_program(num_inputs: int, num_outputs: int, program_length: int) -> List[BoolRef]:
    """Build SSA-style straight-line program constraints (no examples).

    Returns a list of constraints that define the program structure.
    """
    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    if num_outputs <= 0:
        raise ValueError("num_outputs must be positive")
    if program_length < 0:
        raise ValueError("program_length must be non-negative")

    total_sources = num_inputs + 2 + program_length  # inputs + const0 + const1 + temps
    idx_bits = max(1, (total_sources - 1).bit_length())

    op_or = BitVecVal(0, OP_BITS)
    op_and = BitVecVal(1, OP_BITS)
    op_xor = BitVecVal(2, OP_BITS)

    constraints: List[BoolRef] = []

    for instr in range(program_length):
        max_idx = num_inputs + 1 + instr  # inputs + const0 + const1 + previous temps
        max_idx_bv = BitVecVal(max_idx, idx_bits)

        op = BitVec(f"OP_{instr}", OP_BITS)
        src1 = BitVec(f"S1_{instr}", idx_bits)
        src2 = BitVec(f"S2_{instr}", idx_bits)


        # Force uniqueness of (op, src1, src2) tuples
        if force_unique:
            for pre_instr in range(instr):
                pre_op = BitVec(f"OP_{pre_instr}", OP_BITS)
                pre_src1 = BitVec(f"S1_{pre_instr}", idx_bits)
                pre_src2 = BitVec(f"S2_{pre_instr}", idx_bits)
                constraints.append(Or(pre_op != op, pre_src1 != src1, pre_src2 != src2))

        # Encode boolean selection of src1 and src2
        if encode_boolean:
            for idx in range(max_idx + 1):
                src1_idx = Bool(f"S1_{instr}_eq_{idx}")
                src2_idx = Bool(f"S2_{instr}_eq_{idx}")
                constraints.append(src1_idx == (src1 == BitVecVal(idx, idx_bits)))
                constraints.append(src2_idx == (src2 == BitVecVal(idx, idx_bits)))

        constraints.append(Or(op == op_or, op == op_and, op == op_xor))
        constraints.append(ULE(src1, max_idx_bv))
        constraints.append(ULE(src2, max_idx_bv))
        constraints.append(ULT(src1, src2))

    max_total_idx = BitVecVal(total_sources - 1, idx_bits)
    for out_idx in range(num_outputs):
        selector = BitVec(f"OUT_{out_idx}_idx", idx_bits)
        constraints.append(ULE(selector, max_total_idx))

    return constraints

def build_test(width: int, input_vals: List[int], tag: str) -> Tuple[List[BoolRef | Literal[False]], List[BitVecRef | BitVecNumRef]]:
    """Build SSA-style straight-line program constraints for a batch.

    Returns a tuple (constraints, outputs) where outputs is a list of
    bit-vector expressions representing the program outputs for this batch.
    """
    if PROGRAM_LENGTH < 0:
        raise ValueError("PROGRAM_LENGTH must be non-negative")

    total_sources = NUM_INPUTS + 2 + PROGRAM_LENGTH  # inputs + const0 + const1 + temps
    idx_bits = max(1, (total_sources - 1).bit_length())

    op_or = BitVecVal(0, OP_BITS)
    op_and = BitVecVal(1, OP_BITS)

    constraints: List[BoolRef | Literal[False]] = []

    # Seed values with the batch input truth tables
    values: List[BitVecNumRef | BitVecRef] = [BitVecVal(input_vals[j], width) for j in range(NUM_INPUTS)]

    values.append(BitVecVal(0, width))  # constant 0
    values.append(~BitVecVal(0, width))  # constant 1

    for instr in range(PROGRAM_LENGTH):
        op = BitVec(f"OP_{instr}", OP_BITS)
        src1 = BitVec(f"S1_{instr}", idx_bits)
        src2 = BitVec(f"S2_{instr}", idx_bits)
        val = BitVec(f"VAL_{tag}_{instr}", width)

        if encode_boolean:
            left_expr = BitVec(f"LEFT_{tag}_{instr}", width)
            right_expr = BitVec(f"RIGHT_{tag}_{instr}", width)

            for idx, value in enumerate(values):
                src1_idx = Bool(f"S1_{instr}_eq_{idx}")
                src2_idx = Bool(f"S2_{instr}_eq_{idx}")

                constraints.append(Implies(src1_idx, left_expr == value))
                constraints.append(Implies(src2_idx, right_expr == value))


            left_expr_ = _select_bv(values, src1, idx_bits)
            right_expr_ = _select_bv(values, src2, idx_bits)

            constraints.append(left_expr == left_expr_)
            constraints.append(right_expr == right_expr_)
        else:
            left_expr = _select_bv(values, src1, idx_bits)
            right_expr = _select_bv(values, src2, idx_bits)

        gate_expr = If(
            op == op_or,
            left_expr | right_expr, # OR
            If(
                op == op_and,
                left_expr & right_expr, # AND
                left_expr ^ right_expr # XOR
            ),
        )

        constraints.append(val == gate_expr)
        values.append(val)

    outputs: List[BitVecRef | BitVecNumRef] = []
    for out_idx in range(NUM_OUTPUTS):
        selector = BitVec(f"OUT_{out_idx}_idx", idx_bits)
        outputs.append(_select_bv(values, selector, idx_bits))

    return constraints, outputs


def _pack_examples_to_bitvectors(examples: List[Dict[str, Any]], num_inputs: int, num_outputs: int) -> Tuple[int, List[int], List[int]]:
    width = len(examples)
    input_vals = [0] * num_inputs
    output_vals = [0] * num_outputs
    for t_idx, ex in enumerate(examples):
        ins = ex["inputs"]
        outs = ex["outputs"]
        if len(ins) != num_inputs or len(outs) != num_outputs:
            raise ValueError("Example length does not match num_inputs/num_outputs")
        for j in range(num_inputs):
            if bool(ins[j]):
                input_vals[j] |= (1 << t_idx)
        for j in range(num_outputs):
            if bool(outs[j]):
                output_vals[j] |= (1 << t_idx)
    return width, input_vals, output_vals


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_dataset_from_config(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Returns (examples, num_inputs, num_outputs, instructions)."""
    ctype = cfg.get("type")

    def _collect_examples(data: List[Dict[str, Any]], num_inputs: int, num_outputs: int) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for ex in data:
            ins = [bool(v) for v in ex["inputs"]]
            outs = [bool(v) for v in ex["outputs"]]
            if len(ins) != num_inputs or len(outs) != num_outputs:
                raise ValueError("Example length does not match declared input/output sizes")
            result.append({"inputs": ins, "outputs": outs})
        return result

    if "examples" in cfg:
        num_inputs = int(cfg["num_inputs"])  # required
        num_outputs = int(cfg["num_outputs"])  # required
        examples: List[Dict[str, List[bool]]] = _collect_examples(cfg["examples"], num_inputs, num_outputs)
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
        instructions = PROGRAM_LENGTH
    instructions = int(instructions)
    if instructions < 0:
        raise ValueError("instructions must be non-negative")

    return examples, num_inputs, num_outputs, instructions


def make_solver() -> Solver:
    # return SolverFor('QF_BV')
    tactic: Tactic = Then(
        Tactic('simplify'),
        Tactic('propagate-values'),
        Tactic('bit-blast'),
        Tactic('sat'),
    )
    return tactic.solver()


def main() -> None:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the program")

    parser = argparse.ArgumentParser(description="Synthesize a logic program with z3")

    parser.add_argument("--dataset", choices=list(available_plugins().keys()), default="gol", help="Choose a built-in dataset config")
    parser.add_argument("--config", type=str, default=None, help="Path to a custom JSON config")

    parser.add_argument("--instructions", type=int, default=None, help="Override number of SSA instructions")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of examples to add per incremental batch")
    parser.add_argument("--make-smt2", action="store_true", help="Output SMT-LIB2 format and exit")
    parser.add_argument("--make-dimacs", action="store_true", help="Output DIMACS CNF format and exit (uses bit-blasting followed by Tseitin transformation)")

    parser.add_argument("--no-encode-boolean", action="store_true", help="Disable boolean encoding optimizations")
    parser.add_argument("--no-force-unique", action="store_true", help="Disable uniqueness constraints")

    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of examples")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling examples (None means random)")

    args = parser.parse_args()

    if args.no_encode_boolean:
        global encode_boolean
        encode_boolean = False

    if args.no_force_unique:
        global force_unique
        force_unique = False

    config_path = Path("configs")
    if args.config:
        config_path = Path(args.config)
    elif args.dataset:
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', args.dataset)
        config_path = config_path / f"{sanitized}.json"
    else:
        raise SystemExit("Either --dataset or --config must be specified")

    cfg = _load_config(config_path)
    examples, num_inputs, num_outputs, cfg_instructions = _build_dataset_from_config(cfg)
    if not examples:
        raise SystemExit("Dataset contains no examples")

    instructions = cfg_instructions
    if args.instructions is not None:
        instructions = args.instructions
        if instructions < 0:
            raise SystemExit("--instructions must be non-negative")

    # Update globals
    global NUM_INPUTS, NUM_OUTPUTS, PROGRAM_LENGTH
    NUM_INPUTS = num_inputs
    NUM_OUTPUTS = num_outputs
    PROGRAM_LENGTH = instructions

    batch_size = args.batch_size if args.batch_size else len(examples)
    if batch_size <= 0:
        raise SystemExit("Batch size must be positive")

    s = make_solver()

    s.add(*build_program(NUM_INPUTS, NUM_OUTPUTS, PROGRAM_LENGTH))

    if not args.make_smt2 and not args.make_dimacs:
        s.check()

    logger.info("Built program structure with %d instructions", PROGRAM_LENGTH)
    logger.info("Solver has %d assertions", len(s.assertions()))

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(examples)

    start = time.time()
    result = None
    for batch_idx, offset in enumerate(range(0, len(examples), batch_size)):
        batch = examples[offset: offset + batch_size]
        width, input_vals, output_vals = _pack_examples_to_bitvectors(batch, NUM_INPUTS, NUM_OUTPUTS)
        constraints, outputs = build_test(width, input_vals, tag=f"b{batch_idx}")
        s.add(*constraints)
        for j in range(NUM_OUTPUTS):
            s.add(outputs[j] == BitVecVal(output_vals[j], width))

        logger.info("Solver has %d assertions after batch %d", len(s.assertions()), batch_idx + 1)

        if not args.make_smt2 and not args.make_dimacs:
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

    elapsed = time.time() - start

    if str(result) == 'sat':
        print(f"SAT in {elapsed:.3f} seconds")

        # Pretty-print a compact architecture summary: per-instruction (op, src indices)
        m = s.model()

        total_sources = NUM_INPUTS + 2 + PROGRAM_LENGTH  # inputs + const0 + const1 + temps
        idx_bits = max(1, (total_sources - 1).bit_length())

        instrs: List[Tuple[int | None, int, int]] = []

        def fmt_src(idx: int) -> str:
            if idx < NUM_INPUTS:
                return f"I{idx}"
            if idx == NUM_INPUTS:
                return "0"
            if idx == NUM_INPUTS + 1:
                return "1"
            return f"T{idx - NUM_INPUTS - 2}"

        for instr in range(PROGRAM_LENGTH):
            op_ref = BitVec(f"OP_{instr}", OP_BITS)
            s1_ref = BitVec(f"S1_{instr}", idx_bits)
            s2_ref = BitVec(f"S2_{instr}", idx_bits)
            op_val_ref = m[op_ref]
            s1_val_ref = m[s1_ref]
            s2_val_ref = m[s2_ref]
            if op_val_ref is None or s1_val_ref is None or s2_val_ref is None:
                continue
            op_val = op_val_ref.as_long()
            s1_val = s1_val_ref.as_long()
            s2_val = s2_val_ref.as_long()
            label = OP_LABELS.get(op_val, '?')
            if op_val in (0, 1, 2):  # binary
                print(f"T{instr}: {label}({fmt_src(s1_val)}, {fmt_src(s2_val)})")
                instrs.append((op_val, s1_val, s2_val))
            else:
                print(f"T{instr}: ?({fmt_src(s1_val)}, {fmt_src(s2_val)})")
                instrs.append((None, s1_val, s2_val))

        outputs: List[int | None] = []
        for out_idx in range(NUM_OUTPUTS):
            selector = BitVec(f"OUT_{out_idx}_idx", idx_bits)
            sel_val = m[selector]
            if sel_val is None:
                continue
            print(f"OUT{out_idx}: {fmt_src(sel_val.as_long())}")
            outputs.append(sel_val.as_long())

        for idx, ex in enumerate(examples):
            ins = ex["inputs"]
            expected_outs = [int(bool(v)) for v in ex["outputs"]]

            # Evaluate the synthesized program on this example
            values: List[int] = [int(bool(v)) for v in ins]
            values.append(0)  # constant 0
            values.append(1)  # constant 1

            for instr in range(PROGRAM_LENGTH):
                op, s1_idx, s2_idx = instrs[instr]
                if op is None:
                    val = 0
                else:
                    left = values[s1_idx]
                    right = values[s2_idx]
                    if op == 0:  # OR
                        val = left | right
                    elif op == 1:  # AND
                        val = left & right
                    elif op == 2:  # XOR
                        val = left ^ right
                    else:
                        val = 0
                values.append(val)

            actual_outs: List[int] = []
            for out_idx in range(NUM_OUTPUTS):
                sel_idx = outputs[out_idx]
                if sel_idx is None:
                    actual_outs.append(0)
                else:
                    actual_outs.append(values[sel_idx])

            if actual_outs != expected_outs:
                print(f"Mismatch on example {idx} with inputs {ins}: expected {expected_outs}, got {actual_outs}")
    else:
        print(f"UNSAT in {elapsed:.3f} seconds")

if __name__ == "__main__":
    main()
