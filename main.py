#!/usr/bin/env python3

import logging
import sys
import time
from typing import Tuple, List

from z3 import (
    And,
    Bool,
    Implies,
    Not,
    Or,
    Solver,
    Xor,
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

# # Game of Life neural network
HIDDEN_LAYERS = [8, 8, 4, 2]
NUM_INPUTS = 7
NUM_OUTPUTS = 1

# 3 bits adder neural network
# HIDDEN_LAYERS = [3, 2]
# NUM_INPUTS = 3
# NUM_OUTPUTS = 2

LAYERS = [NUM_INPUTS] + HIDDEN_LAYERS + [NUM_OUTPUTS]  # add output layer to hidden layers


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

            constraints.append(Implies(hor, And(Not(hand), Not(hxor))))
            constraints.append(Implies(hand, And(Not(hor), Not(hxor))))
            constraints.append(Implies(hxor, And(Not(hor), Not(hand))))

            constraints.append(Implies(hnot, Not(hnop)))
            constraints.append(Implies(hnop, Not(hnot)))

            # At least one left operand when active
            lefts = [Bool(f"L_{i}_{j}_left_{k}") for k in range(LAYERS[i - 1])]
            rights = [Bool(f"L_{i}_{j}_right_{k}") for k in range(LAYERS[i - 1])]
            constraints.append(Implies(active, Or(*lefts)))
            # At least one right operand iff binary
            constraints.append(Implies(binary, Or(*rights)))
            # No right operand when unary
            constraints.append(Implies(unary, And(*[Not(x) for x in rights])))

            # Mutual exclusion within lefts/rights and between matching left/right
            for k in range(LAYERS[i - 1]):
                lk = Bool(f"L_{i}_{j}_left_{k}")
                rk = Bool(f"L_{i}_{j}_right_{k}")
                constraints.append(Implies(lk, Not(rk)))
                constraints.append(Implies(rk, Not(lk)))

                for l in range(LAYERS[i - 1]):
                    if k != l:
                        lk2 = Bool(f"L_{i}_{j}_left_{l}")
                        rk2 = Bool(f"L_{i}_{j}_right_{l}")
                        constraints.append(Implies(lk, Not(lk2)))
                        constraints.append(Implies(rk, Not(rk2)))

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
            # Copy the values of the left operand and the right operand to new variables
            left_operand = Bool(f"V_left_{counter}_{i}_{j}")
            right_operand = Bool(f"V_right_{counter}_{i}_{j}")
            active = Bool(f"L_{i}_{j}_active")

            local_exprs: List = []

            hor = Bool(f"L_{i}_{j}_or")
            hand = Bool(f"L_{i}_{j}_and")
            hxor = Bool(f"L_{i}_{j}_xor")

            hnot = Bool(f"L_{i}_{j}_not")
            hnop = Bool(f"L_{i}_{j}_nop")

            for k in range(LAYERS[i - 1]):
                left_sel = Bool(f"L_{i}_{j}_left_{k}")
                right_sel = Bool(f"L_{i}_{j}_right_{k}")
                prev_val = Bool(f"V_{counter}_{i - 1}_{k}")
                local_exprs.append(Implies(left_sel, left_operand == prev_val))
                local_exprs.append(Implies(right_sel, right_operand == prev_val))

            # Calculate the value of the node according to the chosen operation
            out_val = Bool(f"V_{counter}_{i}_{j}")
            local_exprs.append(Implies(hor, out_val == Or(left_operand, right_operand)))
            local_exprs.append(Implies(hand, out_val == And(left_operand, right_operand)))
            local_exprs.append(Implies(hxor, out_val == Xor(left_operand, right_operand)))

            local_exprs.append(Implies(hnot, out_val == Not(left_operand)))
            local_exprs.append(Implies(hnop, out_val == left_operand))

            constraints.append(Implies(active, And(*local_exprs)))

    return constraints

def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the program")
    
    constraints: List = []

    structure = make_structure()
    constraints += structure
    logger.info("Structure constraints: {}".format(len(structure)))

    # # Generate test cases for the Game of Life
    counter = 0
    for i in range(4):
        for j in range(3):
            for k in range(4):
                for alive in [True, False]:
                    inputs, outputs = make_gol_test_case(i, j, k, alive)
                    test_case = make_test(counter, inputs, outputs)
                    constraints += test_case
                    logger.info("Test case formula: {}".format(test_case))

                    counter += 1

    # Generate test cases for the 3 bits adder
    # counter = 0
    # for left in [True, False]:
    #     for center in [True, False]:
    #         for right in [True, False]:
    #             inputs, outputs = make_3_bits_adder_test_case(left, center, right)
    #             test_case = make_test(counter, inputs, outputs)
    #             constraints += test_case
    #             logger.info("Test case constraints: {}".format(len(test_case)))

    #             counter += 1

    s = Solver()
    s.add(*constraints)

    # Solve the formula
    start = time.time()
    result = s.check()
    elapsed = time.time() - start

    if str(result) == 'sat':
        layers = [None for _ in range(len(LAYERS))]
        print("SAT in {:.3f} seconds".format(elapsed))

        m = s.model()
        for d in m.decls():
            name = d.name()
            val = m[d]
            # Only list positive truths for layer wiring/ops (exclude helper flags)
            if name.startswith("L_") and str(val) == 'True' and ("active" not in name and "binary" not in name and "unary" not in name):
                parts = name.split("_")
                if len(parts) > 1 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    if layers[layer_idx] is None:
                        layers[layer_idx] = []
                    layers[layer_idx].append(name)

        for i in range(len(layers)):
            if layers[i] is not None:
                layers[i].sort()
                print("Layer {}: {}".format(i, layers[i]))
    else:
        print("UNSAT in {:.3f} seconds".format(elapsed))

if __name__ == "__main__":
    main()
