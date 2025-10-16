#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

from z3 import (
    BitVec,
    BitVecVal,
    If,
    Or,
    Not,
    Tactic,
    Then,
    ULE,
    ULT,
    Implies,
)

def make_gol_test_case(left_column: int, center_column: int, right_column: int, alive: bool) -> Tuple[list[bool], list[bool]]:
    inputs: list[bool] = []
    inputs.append(left_column % 2 != 0)
    inputs.append(left_column // 2 % 2 != 0)
    inputs.append(center_column % 2 != 0)
    inputs.append(center_column // 2 % 2 != 0)
    inputs.append(right_column % 2 != 0)
    inputs.append(right_column // 2 % 2 != 0)
    inputs.append(alive)

    # Game of Life rules
    outputs: list[bool] = []
    outputs.append(
        left_column + center_column + right_column == 3
        or (left_column + center_column + right_column == 2 and alive)
    )
    return inputs, outputs

def make_3_bits_adder_test_case(left_bit: bool, center_bit: bool, right_bit: bool) -> Tuple[list[bool], list[bool]]:
    inputs: list[bool] = []
    inputs.append(left_bit)
    inputs.append(center_bit)
    inputs.append(right_bit)
    
    left = 1 if left_bit else 0
    center = 1 if center_bit else 0
    right = 1 if right_bit else 0
    
    sum = left + center + right

    outputs: list[bool] = []
    outputs.append(sum % 2 != 0)  # bottom bit
    outputs.append(sum // 2 % 2 != 0)  # top bit
    return inputs, outputs

# Defaults (overridden by config/CLI)
NUM_INPUTS = 7
NUM_OUTPUTS = 1
PROGRAM_LENGTH = 16

OP_BITS = 2  # encode OR, AND, XOR
OP_LABELS = {0: 'OR', 1: 'AND', 2: 'XOR'}


def _select_bv(values: List, idx_var, bits: int, width: int):
    result = BitVecVal(0, width)
    for idx, value in enumerate(values):
        result = If(idx_var == BitVecVal(idx, bits), value, result)
    return result


def build_program(width: int, input_vals: List[int], tag: str) -> Tuple[List, List]:
    """Build SSA-style straight-line program constraints for a batch.

    Returns a tuple (constraints, outputs) where outputs is a list of
    bit-vector expressions representing the program outputs for this batch.
    """
    if PROGRAM_LENGTH < 0:
        raise ValueError("PROGRAM_LENGTH must be non-negative")

    total_sources = NUM_INPUTS + PROGRAM_LENGTH
    idx_bits = max(1, total_sources.bit_length())

    op_or = BitVecVal(0, OP_BITS)
    op_and = BitVecVal(1, OP_BITS)
    op_xor = BitVecVal(2, OP_BITS)

    constraints: List = []

    # Seed values with the batch input truth tables
    values: List = [BitVecVal(input_vals[j], width) for j in range(NUM_INPUTS)]

    values.append(BitVecVal(1, width))  # constant 1

    for instr in range(PROGRAM_LENGTH):
        max_idx = NUM_INPUTS + instr
        clamp = NUM_INPUTS if max_idx < NUM_INPUTS else max_idx
        max_idx_bv = BitVecVal(clamp, idx_bits)

        op = BitVec(f"OP_{instr}", OP_BITS)
        src1 = BitVec(f"S1_{instr}", idx_bits)
        src2 = BitVec(f"S2_{instr}", idx_bits)
        val = BitVec(f"VAL_{tag}_{instr}", width)

        constraints.append(Or(op == op_or, op == op_and, op == op_xor))
        constraints.append(ULE(src1, max_idx_bv))
        constraints.append(ULE(src2, max_idx_bv))

        left_expr = _select_bv(values, src1, idx_bits, width)
        right_expr = _select_bv(values, src2, idx_bits, width)


        constraints.append(ULE(src1, src2))

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

    outputs: List = []
    max_total_idx = BitVecVal(total_sources, idx_bits)
    for out_idx in range(NUM_OUTPUTS):
        selector = BitVec(f"OUT_{out_idx}_idx", idx_bits)
        constraints.append(ULE(selector, max_total_idx))
        outputs.append(_select_bv(values, selector, idx_bits, width))

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
        examples = _collect_examples(cfg["examples"], num_inputs, num_outputs)
    elif ctype == "gol":
        num_inputs = int(cfg.get("num_inputs", 7))
        num_outputs = int(cfg.get("num_outputs", 1))
        gol = cfg.get("gol", {})
        left_range = int(gol.get("left_range", 4))
        center_range = int(gol.get("center_range", 3))
        right_range = int(gol.get("right_range", 4))
        include_alive = bool(gol.get("include_alive", True))

        examples = []
        for i in range(left_range):
            for j in range(center_range):
                for k in range(right_range):
                    for alive in ([True, False] if include_alive else [False]):
                        ins, outs = make_gol_test_case(i, j, k, alive)
                        examples.append({"inputs": ins, "outputs": outs})
    elif ctype in ("adder3", "adder"):
        num_inputs = int(cfg.get("num_inputs", 3))
        num_outputs = int(cfg.get("num_outputs", 2))
        examples = []
        for left in [True, False]:
            for center in [True, False]:
                for right in [True, False]:
                    ins, outs = make_3_bits_adder_test_case(left, center, right)
                    examples.append({"inputs": ins, "outputs": outs})
    else:
        raise ValueError(f"Unsupported config type or format: {ctype}")

    instructions = cfg.get("instructions")
    if instructions is None:
        hidden_layers = cfg.get("hidden_layers")
        if isinstance(hidden_layers, list):
            instructions = sum(int(x) for x in hidden_layers)
        elif hidden_layers is not None:
            instructions = int(hidden_layers)
    if instructions is None:
        instructions = PROGRAM_LENGTH
    instructions = int(instructions)
    if instructions < 0:
        raise ValueError("instructions must be non-negative")

    return examples, num_inputs, num_outputs, instructions


def make_solver():
    try:
        tactic = Then(
            Tactic('simplify'),
            Tactic('propagate-values'),
            Tactic('bit-blast'),
            Tactic('sat'),
        )
        return tactic.solver()
    except Exception:  # pragma: no cover
        from z3 import SolverFor

        return SolverFor('QF_BV')


def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the program")

    parser = argparse.ArgumentParser(description="Synthesize a logic program with z3")
    parser.add_argument("--dataset", choices=["gol", "adder3"], default=None, help="Choose a built-in dataset config")
    parser.add_argument("--config", type=str, default=None, help="Path to a dataset JSON config")
    parser.add_argument("--instructions", type=int, default=None, help="Override number of SSA instructions")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of examples to add per incremental batch")
    args = parser.parse_args()

    # Resolve config path
    default_configs = {"gol": Path("configs/gol.json"), "adder3": Path("configs/adder3.json")}
    config_path: Path
    if args.config:
        config_path = Path(args.config)
    elif args.dataset:
        config_path = default_configs[args.dataset]
    else:
        config_path = default_configs["gol"]

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

    start = time.time()
    result = None
    for batch_idx, offset in enumerate(range(0, len(examples), batch_size)):
        batch = examples[offset: offset + batch_size]
        width, input_vals, output_vals = _pack_examples_to_bitvectors(batch, NUM_INPUTS, NUM_OUTPUTS)
        constraints, outputs = build_program(width, input_vals, tag=f"b{batch_idx}")
        s.add(*constraints)
        for j in range(NUM_OUTPUTS):
            s.add(outputs[j] == BitVecVal(output_vals[j], width))
        result = s.check()
        if str(result) != 'sat':
            break

    elapsed = time.time() - start

    if str(result) == 'sat':
        print("SAT in {:.3f} seconds".format(elapsed))

        # Pretty-print a compact architecture summary: per-instruction (op, src indices)
        m = s.model()
        idx_bits = max(1, (NUM_INPUTS + PROGRAM_LENGTH).bit_length())

        def fmt_src(idx: int) -> str:
            if idx < NUM_INPUTS:
                return f"I{idx}"
            elif idx == NUM_INPUTS:
                return "1"
            else:
                return f"T{idx - NUM_INPUTS - 1}"

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
            else:
                print(f"T{instr}: BUF({fmt_src(s1_val)})")

        for out_idx in range(NUM_OUTPUTS):
            selector = BitVec(f"OUT_{out_idx}_idx", idx_bits)
            sel_val = m[selector]
            if sel_val is None:
                continue
            print(f"OUT{out_idx}: {fmt_src(sel_val.as_long())}")
    else:
        print("UNSAT in {:.3f} seconds".format(elapsed))

if __name__ == "__main__":
    main()
