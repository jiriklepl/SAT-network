#!/usr/bin/env python3

import logging
import sys
from typing import Tuple

from pysat.solvers import Solver
from pysat.formula import *

def make_gol_test_case(left_column : int, center_column : int, right_column : int, alive : bool) -> Tuple[list, list]:
    inputs = []
    inputs.append(left_column % 2 != 0)
    inputs.append(left_column // 2 % 2 != 0)
    inputs.append(center_column % 2 != 0)
    inputs.append(center_column // 2 % 2 != 0)
    inputs.append(right_column % 2 != 0)
    inputs.append(right_column // 2 % 2 != 0)
    inputs.append(alive)

    # Game of Life rules
    outputs = []
    outputs.append(left_column + center_column + right_column == 3 or (left_column + center_column + right_column == 2 and alive))
    return inputs, outputs

def make_3_bits_adder_test_case(left_bit : bool, center_bit : bool, right_bit : bool) -> Tuple[list, list]:
    inputs = []
    inputs.append(left_bit)
    inputs.append(center_bit)
    inputs.append(right_bit)
    
    left = 1 if left_bit else 0
    center = 1 if center_bit else 0
    right = 1 if right_bit else 0
    
    sum = left + center + right

    outputs = []
    outputs.append(sum % 2 != 0) # bottom bit
    outputs.append(sum // 2 % 2 != 0) # top bit
    return inputs, outputs

# # Game of Life neural network
# HIDDEN_LAYERS = [8, 8, 4, 2]
# NUM_INPUTS = 7
# NUM_OUTPUTS = 1

# 3 bits adder neural network
HIDDEN_LAYERS = [3, 2]
NUM_INPUTS = 3
NUM_OUTPUTS = 2

LAYERS = [NUM_INPUTS] + HIDDEN_LAYERS + [NUM_OUTPUTS] # add output layer to hidden layers

def make_structure():
    formula = PYSAT_TRUE
    
    for i in range(LAYERS[0]):
        active = Atom(f"L_0_{i}_active")
        formula = formula & active
    
    for i in range(LAYERS[-1]):
        active = Atom(f"L_{len(LAYERS) - 1}_{i}_active")
        formula = formula & active

    for i in range(0, len(LAYERS) - 1):
        for j in range(LAYERS[i]):
            active = Atom(f"L_{i}_{j}_active")

            formula = formula & Equals(active, Or(*[(Atom(f"L_{i + 1}_{k}_left_{j}") & Atom(f"L_{i + 1}_{k}_active")) | 
                                                  (Atom(f"L_{i + 1}_{k}_right_{j}") & Atom(f"L_{i + 1}_{k}_active"))
                                                  for k in range(LAYERS[i + 1])], merge=True))

    for i in range(1, len(LAYERS)):
        for j in range(LAYERS[i]):
            binary = Atom('L_' + str(i) + '_' + str(j) + '_binary')
            unary = Atom('L_' + str(i) + '_' + str(j) + '_unary')
            active = Atom('L_' + str(i) + '_' + str(j) + '_active')

            hor = Atom('L_' + str(i) + '_' + str(j) + '_or')
            hand = Atom('L_' + str(i) + '_' + str(j) + '_and')
            hxor = Atom('L_' + str(i) + '_' + str(j) + '_xor')

            hnot = Atom('L_' + str(i) + '_' + str(j) + '_not')
            hnop = Atom('L_' + str(i) + '_' + str(j) + '_nop')

            formula = formula & Equals(active, binary | unary)
            formula = formula & Implies(binary, ~unary)
            formula = formula & Implies(unary, ~binary)

            formula = formula & Equals(binary, hor | hand | hxor)
            formula = formula & Equals(unary, hnot | hnop)

            formula = formula & Implies(hor, ~hand & ~hxor)
            formula = formula & Implies(hand, ~hor & ~hxor)
            formula = formula & Implies(hxor, ~hor & ~hand)

            formula = formula & Implies(hnot, ~hnop)
            formula = formula & Implies(hnop, ~hnot)

            formula = formula & (active >> Or(*[Atom(f"L_{i}_{j}_left_{k}") for k in range(LAYERS[i - 1])], merge=True))
            formula = formula & (binary >> Or(*[Atom(f"L_{i}_{j}_right_{k}") for k in range(LAYERS[i - 1])], merge=True))
            formula = formula & (unary >> And(*[~Atom(f"L_{i}_{j}_right_{k}") for k in range(LAYERS[i - 1])], merge=True))

            for k in range(LAYERS[i - 1]):
                formula = formula & Implies(Atom(f"L_{i}_{j}_left_{k}"), ~Atom(f"L_{i}_{j}_right_{k}"))
                formula = formula & Implies(Atom(f"L_{i}_{j}_right_{k}"), ~Atom(f"L_{i}_{j}_left_{k}"))

                for l in range(LAYERS[i - 1]):
                    if k != l:
                        formula = formula & Implies(Atom(f"L_{i}_{j}_left_{k}"), ~Atom(f"L_{i}_{j}_left_{l}"))
                        formula = formula & Implies(Atom(f"L_{i}_{j}_right_{k}"), ~Atom(f"L_{i}_{j}_right_{l}"))

    return formula

def make_test(counter : int, inputs : list, outputs : list):
    '''
    Make a test case for the neural network.
    counter is used to generate unique names for the variables in the test case.
    inputs is a list of inputs to the neural network (logic values)
    output is a list of outputs from the neural network (logic values)
    '''
    formula = PYSAT_TRUE
    
    assert len(inputs) == NUM_INPUTS
    assert len(outputs) == NUM_OUTPUTS

    # Encode the test case (truth table for the inputs and the expected outputs)
    # V_counter_i_j
    # Assert the expected truth values for each input
    for j, value in enumerate(inputs):
        atom = Atom(f"V_{counter}_0_{j}")
        formula = formula & (atom if value else ~atom)

    # Assert the expected truth values for each output
    for j, value in enumerate(outputs):
        atom = Atom(f"V_{counter}_{len(LAYERS) - 1}_{j}")
        formula = formula & (atom if value else ~atom)

    # Encode the evaluation of the neural network
    #  for each node, the value of the left operand and the right operand is copied to new variables
    #  then, for every possible configuration of the node, the value of the node is calculated according to the values of the operands and the chosen operation (e.g., L_i_j_or)
    # V_counter_i_j
    for i in range(1, len(LAYERS)):
        for j in range(LAYERS[i]):
            # Copy the values of the left operand and the right operand to new variables
            left_operand = Atom(f"V_left_{counter}_{i}_{j}")
            right_operand = Atom(f"V_right_{counter}_{i}_{j}")
            active = Atom(f"L_{i}_{j}_active")
            
            sub_formula = PYSAT_TRUE

            hor = Atom(f"L_{i}_{j}_or")
            hand = Atom(f"L_{i}_{j}_and")
            hxor = Atom(f"L_{i}_{j}_xor")

            hnot = Atom(f"L_{i}_{j}_not")
            hnop = Atom(f"L_{i}_{j}_nop")
            
            for k in range(LAYERS[i - 1]):
                sub_formula = sub_formula & Implies(Atom(f"L_{i}_{j}_left_{k}"), Equals(Atom(f"V_{counter}_{i - 1}_{k}"), left_operand))
                sub_formula = sub_formula & Implies(Atom(f"L_{i}_{j}_right_{k}"), Equals(Atom(f"V_{counter}_{i - 1}_{k}"), right_operand))

            # Calculate the value of the node according to the chosen operation
            sub_formula = sub_formula & Implies(hor, Equals(Atom(f"V_{counter}_{i}_{j}"), left_operand | right_operand))
            sub_formula = sub_formula & Implies(hand, Equals(Atom(f"V_{counter}_{i}_{j}"), left_operand & right_operand))
            sub_formula = sub_formula & Implies(hxor, Equals(Atom(f"V_{counter}_{i}_{j}"), left_operand ^ right_operand))

            sub_formula = sub_formula & Implies(hnot, Equals(Atom(f"V_{counter}_{i}_{j}"), ~left_operand))
            sub_formula = sub_formula & Implies(hnop, Equals(Atom(f"V_{counter}_{i}_{j}"), left_operand))
            
            formula = formula & (active >> sub_formula)

    return formula

def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the program")
    
    formula = PYSAT_TRUE

    structure = make_structure()
    formula = formula & structure
    logger.info("Structure formula: {}".format(structure))

    # # Generate test cases for the Game of Life
    # counter = 0
    # for i in range(4):
    #     for j in range(3):
    #         for k in range(4):
    #             for alive in [True, False]:
    #                 inputs, outputs = make_gol_test_case(i, j, k, alive)
    #                 test_case = make_test(counter, inputs, outputs)
    #                 formula = formula & test_case
    #                 logger.info("Test case formula: {}".format(test_case))

    #                 counter += 1

    # Generate test cases for the 3 bits adder
    counter = 0
    for left in [True, False]:
        for center in [True, False]:
            for right in [True, False]:
                inputs, outputs = make_3_bits_adder_test_case(left, center, right)
                test_case = make_test(counter, inputs, outputs).simplified()
                formula = formula & test_case
                logger.info("Test case formula: {}".format(test_case))

                counter += 1

    model = Solver(use_timer=True, bootstrap_with=formula)

    # Solve the formula
    if model.solve():
        layers = [None for _ in range(len(LAYERS))]
        print("SAT in {} seconds".format(model.time()))

        # Print the model
        result = model.get_model()

        literals = Formula.formulas(result, atoms_only=True)
        for i in range(len(literals)):
            str_literal = str(literals[i])
            if "L_" in str_literal and str_literal[0] != "~" and "active" not in str_literal and "binary" not in str_literal and "unary" not in str_literal:
                layer = str_literal.split("_")[1]

                if layers[int(layer)] is None:
                    layers[int(layer)] = []
                layers[int(layer)].append(str_literal)

        # Print the model
        for i in range(len(layers)):
            if layers[i] is not None:
                layers[i].sort()
                print("Layer {}: {}".format(i, layers[i]))
    else:
        print("UNSAT in {} seconds".format(model.time()))

if __name__ == "__main__":
    main()
