#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

from z3 import (
    And,
    Bool,
    BoolVal,
    BitVec,
    BitVecVal,
    Implies,
    Int,
    Not,
    Or,
    PbEq,
    Solver,
    Xor,
    If,
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

# Defaults (can be overridden by config/CLI)
HIDDEN_LAYERS = [8, 6, 3, 2]
NUM_INPUTS = 7
NUM_OUTPUTS = 1

LAYERS = [NUM_INPUTS] + HIDDEN_LAYERS + [NUM_OUTPUTS]  # recomputed in main()


def _pb_exactly(vars_list: List, count: int):
    if not vars_list:
        return BoolVal(count == 0)
    return PbEq([(var, 1) for var in vars_list], count)


# -------- Original boolean-based builder (kept for reference) --------


def make_structure() -> List:
    constraints: List = []

    # Input layer nodes are active
    for i in range(LAYERS[0]):
        active = Bool(f"L_0_{i}_active")
        constraints.append(active)

    # Output layer nodes are active
    for i in range(LAYERS[-1]):
        active = Bool(f"L_{len(LAYERS) - 1}_{i}_active")
        constraints.append(active)

    # Node activity depends on connections from the next layer
    for i in range(0, len(LAYERS) - 1):
        for j in range(LAYERS[i]):
            active = Bool(f"L_{i}_{j}_active")
            ors = []
            for k in range(LAYERS[i + 1]):
                next_active = Bool(f"L_{i + 1}_{k}_active")
                left_ij = Bool(f"L_{i + 1}_{k}_left_{j}")
                right_ij = Bool(f"L_{i + 1}_{k}_right_{j}")
                ors.append(Or(And(left_ij, next_active), And(right_ij, next_active)))
            constraints.append(active == Or(*ors))

    # Operator selection and wiring constraints
    for i in range(1, len(LAYERS)):
        for j in range(LAYERS[i]):
            binary = Bool(f"L_{i}_{j}_binary")
            unary = Bool(f"L_{i}_{j}_unary")
            active = Bool(f"L_{i}_{j}_active")

            hor = Bool(f"L_{i}_{j}_or")
            hand = Bool(f"L_{i}_{j}_and")
            hxor = Bool(f"L_{i}_{j}_xor")

            hnot = Bool(f"L_{i}_{j}_not")
            hnop = Bool(f"L_{i}_{j}_nop")

            constraints.append(active == Or(binary, unary))
            constraints.append(Implies(binary, Not(unary)))
            constraints.append(Implies(unary, Not(binary)))

            constraints.append(binary == Or(hor, hand, hxor))
            constraints.append(unary == Or(hnot, hnop))

            constraints.append(Implies(binary, _pb_exactly([hor, hand, hxor], 1)))
            constraints.append(Implies(Not(binary), _pb_exactly([hor, hand, hxor], 0)))
            constraints.append(Implies(unary, _pb_exactly([hnot, hnop], 1)))
            constraints.append(Implies(Not(unary), _pb_exactly([hnot, hnop], 0)))

            lefts = [Bool(f"L_{i}_{j}_left_{k}") for k in range(LAYERS[i - 1])]
            rights = [Bool(f"L_{i}_{j}_right_{k}") for k in range(LAYERS[i - 1])]
            constraints.append(Implies(active, _pb_exactly(lefts, 1)))
            constraints.append(Implies(Not(active), _pb_exactly(lefts, 0)))
            constraints.append(Implies(binary, _pb_exactly(rights, 1)))
            constraints.append(Implies(Not(binary), _pb_exactly(rights, 0)))

            for k in range(LAYERS[i - 1]):
                lk = Bool(f"L_{i}_{j}_left_{k}")
                rk = Bool(f"L_{i}_{j}_right_{k}")
                constraints.append(Not(And(lk, rk)))

    return constraints

def make_test(counter: int, inputs: list, outputs: list):
    '''
    Make a test case for the neural network.
    counter is used to generate unique names for the variables in the test case.
    inputs is a list of inputs to the neural network (logic values)
    output is a list of outputs from the neural network (logic values)
    '''
    constraints: List = []
    
    assert len(inputs) == NUM_INPUTS
    assert len(outputs) == NUM_OUTPUTS

    # Encode the test case (truth table for the inputs and the expected outputs)
    # V_counter_i_j
    # Assert the expected truth values for each input
    for j, value in enumerate(inputs):
        atom = Bool(f"V_{counter}_0_{j}")
        constraints.append(atom if value else Not(atom))

    # Assert the expected truth values for each output
    for j, value in enumerate(outputs):
        atom = Bool(f"V_{counter}_{len(LAYERS) - 1}_{j}")
        constraints.append(atom if value else Not(atom))

    # Encode the evaluation of the neural network
    #  for each node, the value of the left operand and the right operand is copied to new variables
    #  then, for every possible configuration of the node, the value of the node is calculated according to the values of the operands and the chosen operation (e.g., L_i_j_or)
    # V_counter_i_j
    for i in range(1, len(LAYERS)):
        for j in range(LAYERS[i]):
            active = Bool(f"L_{i}_{j}_active")

            hor = Bool(f"L_{i}_{j}_or")
            hand = Bool(f"L_{i}_{j}_and")
            hxor = Bool(f"L_{i}_{j}_xor")

            hnot = Bool(f"L_{i}_{j}_not")
            hnop = Bool(f"L_{i}_{j}_nop")

            prev_vals = [Bool(f"V_{counter}_{i - 1}_{k}") for k in range(LAYERS[i - 1])]
            left_choices = [Bool(f"L_{i}_{j}_left_{k}") for k in range(LAYERS[i - 1])]
            right_choices = [Bool(f"L_{i}_{j}_right_{k}") for k in range(LAYERS[i - 1])]

            left_terms = [And(sel, val) for sel, val in zip(left_choices, prev_vals)]
            right_terms = [And(sel, val) for sel, val in zip(right_choices, prev_vals)]

            left_expr = Or(*left_terms) if left_terms else BoolVal(False)
            right_expr = Or(*right_terms) if right_terms else BoolVal(False)

            out_val = Bool(f"V_{counter}_{i}_{j}")
            constraints.append(Implies(active, And(
                Implies(hor, out_val == Or(left_expr, right_expr)),
                Implies(hand, out_val == And(left_expr, right_expr)),
                Implies(hxor, out_val == Xor(left_expr, right_expr)),
                Implies(hnot, out_val == Not(left_expr)),
                Implies(hnop, out_val == left_expr),
            )))

    return constraints


# -------- BitVector-based, index-encoded builder (optimized) --------

def _build_dataset_gol() -> Tuple[int, List[int], List[int]]:
    tests: List[Tuple[List[bool], List[bool]]] = []
    for i in range(4):
        for j in range(3):
            for k in range(4):
                for alive in [True, False]:
                    inputs, outputs = make_gol_test_case(i, j, k, alive)
                    tests.append((inputs, outputs))
    width = len(tests)
    input_vals = [0] * NUM_INPUTS
    output_vals = [0] * NUM_OUTPUTS
    for t_idx, (ins, outs) in enumerate(tests):
        for j in range(NUM_INPUTS):
            if ins[j]:
                input_vals[j] |= (1 << t_idx)
        for j in range(NUM_OUTPUTS):
            if outs[j]:
                output_vals[j] |= (1 << t_idx)
    return width, input_vals, output_vals


def _build_dataset_adder() -> Tuple[int, List[int], List[int]]:
    tests: List[Tuple[List[bool], List[bool]]] = []
    for left in [True, False]:
        for center in [True, False]:
            for right in [True, False]:
                tests.append(make_3_bits_adder_test_case(left, center, right))
    width = len(tests)
    input_vals = [0] * NUM_INPUTS
    output_vals = [0] * NUM_OUTPUTS
    for t_idx, (ins, outs) in enumerate(tests):
        for j in range(NUM_INPUTS):
            if ins[j]:
                input_vals[j] |= (1 << t_idx)
        for j in range(NUM_OUTPUTS):
            if outs[j]:
                output_vals[j] |= (1 << t_idx)
    return width, input_vals, output_vals


def _select_bv(prev_list: List, idx_var, width: int):
    expr = BitVecVal(0, width)
    for k in range(len(prev_list)):
        expr = If(idx_var == k, prev_list[k], expr)
    return expr


def build_network_bitvec(width: int, input_vals: List[int]) -> Tuple[List[List], List[List]]:
    """Build bitvector-valued network variables and constraints.
    Returns (constraints, node_bv_values) where node_bv_values[i][j] is the BV value for layer i node j.
    """
    constraints: List = []
    # Initialize input layer BV constants
    layer_values: List[List] = []
    input_bvs = [BitVecVal(input_vals[j], width) for j in range(LAYERS[0])]
    layer_values.append(input_bvs)

    # For layers > 0, create BitVec vars and index/op Int vars
    for i in range(1, len(LAYERS)):
        prev = layer_values[i - 1]
        curr_vals: List = []
        for j in range(LAYERS[i]):
            out = BitVec(f"BV_{i}_{j}", width)
            left_idx = Int(f"L_{i}_{j}_left_idx")
            right_idx = Int(f"L_{i}_{j}_right_idx")
            op = Int(f"L_{i}_{j}_op")  # 0=OR,1=AND,2=XOR,3=NOT,4=NOP

            # Index domains
            constraints.append(And(left_idx >= 0, left_idx < len(prev)))
            constraints.append(And(right_idx >= 0, right_idx < len(prev)))
            constraints.append(And(op >= 0, op <= 4))

            left_expr = _select_bv(prev, left_idx, width)
            right_expr = _select_bv(prev, right_idx, width)

            # Symmetry breaking and well-formed binary wiring
            is_binary = Or(op == 0, op == 1, op == 2)
            constraints.append(Implies(is_binary, left_idx != right_idx))
            constraints.append(Implies(is_binary, left_idx < right_idx))

            # Gate semantics
            gate_expr = If(
                op == 0, left_expr | right_expr,
                If(op == 1, left_expr & right_expr,
                   If(op == 2, left_expr ^ right_expr,
                      If(op == 3, ~left_expr,  # NOT
                         left_expr))))  # NOP
            constraints.append(out == gate_expr)
            curr_vals.append(out)
        layer_values.append(curr_vals)

    return constraints, layer_values


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


def _build_dataset_from_config(cfg: Dict[str, Any]) -> Tuple[int, List[int], List[int], int, int, List[int]]:
    """Returns (width, input_vals, output_vals, num_inputs, num_outputs, hidden_layers)."""
    ctype = cfg.get("type")
    hidden_layers = cfg.get("hidden_layers", HIDDEN_LAYERS)

    if "examples" in cfg:
        num_inputs = int(cfg["num_inputs"])  # required
        num_outputs = int(cfg["num_outputs"])  # required
        width, input_vals, output_vals = _pack_examples_to_bitvectors(cfg["examples"], num_inputs, num_outputs)
        return width, input_vals, output_vals, num_inputs, num_outputs, hidden_layers

    if ctype == "gol":
        num_inputs = int(cfg.get("num_inputs", 7))
        num_outputs = int(cfg.get("num_outputs", 1))
        gol = cfg.get("gol", {})
        left_range = int(gol.get("left_range", 4))
        center_range = int(gol.get("center_range", 3))
        right_range = int(gol.get("right_range", 4))
        include_alive = bool(gol.get("include_alive", True))

        examples: List[Dict[str, Any]] = []
        for i in range(left_range):
            for j in range(center_range):
                for k in range(right_range):
                    for alive in ([True, False] if include_alive else [False]):
                        ins, outs = make_gol_test_case(i, j, k, alive)
                        examples.append({"inputs": ins, "outputs": outs})
        width, input_vals, output_vals = _pack_examples_to_bitvectors(examples, num_inputs, num_outputs)
        return width, input_vals, output_vals, num_inputs, num_outputs, hidden_layers

    if ctype in ("adder3", "adder"):
        num_inputs = int(cfg.get("num_inputs", 3))
        num_outputs = int(cfg.get("num_outputs", 2))
        examples: List[Dict[str, Any]] = []
        for left in [True, False]:
            for center in [True, False]:
                for right in [True, False]:
                    ins, outs = make_3_bits_adder_test_case(left, center, right)
                    examples.append({"inputs": ins, "outputs": outs})
        width, input_vals, output_vals = _pack_examples_to_bitvectors(examples, num_inputs, num_outputs)
        return width, input_vals, output_vals, num_inputs, num_outputs, hidden_layers

    raise ValueError(f"Unsupported config type or format: {ctype}")

def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the program")

    parser = argparse.ArgumentParser(description="Synthesize a logic network with z3")
    parser.add_argument("--dataset", choices=["gol", "adder3"], default=None, help="Choose a built-in dataset config")
    parser.add_argument("--config", type=str, default=None, help="Path to a dataset JSON config")
    parser.add_argument("--hidden-layers", type=str, default=None, help="Comma-separated hidden layer sizes, overrides config")
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
    width, input_vals, output_vals, num_inputs, num_outputs, cfg_hidden = _build_dataset_from_config(cfg)

    # Override hidden layers if requested
    hidden_layers = cfg_hidden
    if args.hidden_layers:
        try:
            hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x.strip()]
        except Exception as e:
            raise SystemExit(f"Invalid --hidden-layers: {e}")

    # Update globals
    global NUM_INPUTS, NUM_OUTPUTS, HIDDEN_LAYERS, LAYERS
    NUM_INPUTS = num_inputs
    NUM_OUTPUTS = num_outputs
    HIDDEN_LAYERS = hidden_layers
    LAYERS = [NUM_INPUTS] + HIDDEN_LAYERS + [NUM_OUTPUTS]

    constraints, layer_values = build_network_bitvec(width, input_vals)

    # Constrain outputs to match dataset
    last_layer = layer_values[-1]
    for j in range(NUM_OUTPUTS):
        constraints.append(last_layer[j] == BitVecVal(output_vals[j], width))

    s = Solver()
    s.add(*constraints)

    # Solve the formula
    start = time.time()
    result = s.check()
    elapsed = time.time() - start

    if str(result) == 'sat':
        print("SAT in {:.3f} seconds".format(elapsed))

        # Pretty-print a compact architecture summary: per-node (op, left_idx, right_idx)
        m = s.model()
        for i in range(1, len(LAYERS)):
            summaries: List[str] = []
            for j in range(LAYERS[i]):
                op = m[Int(f"L_{i}_{j}_op")]
                li = m[Int(f"L_{i}_{j}_left_idx")]
                ri = m[Int(f"L_{i}_{j}_right_idx")]
                op_map = {0: 'OR', 1: 'AND', 2: 'XOR', 3: 'NOT', 4: 'NOP'}
                op_val = op.as_long() if op is not None else -1
                li_val = li.as_long() if li is not None else -1
                ri_val = ri.as_long() if ri is not None else -1
                if op_val in (0, 1, 2):
                    summaries.append(f"n{j}={op_map.get(op_val,'?')}(L{li_val},R{ri_val})")
                elif op_val == 3:
                    summaries.append(f"n{j}=NOT(L{li_val})")
                else:
                    summaries.append(f"n{j}=NOP(L{li_val})")
            print(f"Layer {i}: " + ", ".join(summaries))
    else:
        print("UNSAT in {:.3f} seconds".format(elapsed))

if __name__ == "__main__":
    main()
