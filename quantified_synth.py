#!/usr/bin/env python3

"""Quantified straight-line program synthesis.

This is a small companion to main.py that avoids building example truth tables.
It synthesizes the same SSA-style logic programs, but constrains them with a
universal equivalence formula over symbolic Boolean inputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from z3 import (
    And,
    Bool,
    BoolRef,
    BoolVal,
    BitVec,
    ForAll,
    If,
    Implies,
    Not,
    Or,
    Solver,
    Sum,
    Xor,
    sat,
)

from dataset_plugins import available_plugins, get_plugin, get_plugin_config, get_quantified_plugin
from main import (
    DEFAULT_PROGRAM_LENGTH,
    OP_BITS,
    EncodingOptions,
    ProgramSpec,
    _build_program,
    _emit_program,
    _extract_program_from_model,
    _operator_expr,
    _select_bv,
)


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _int_literal(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    raise ValueError("Expected integer literal")


def _parse_expr(node: ast.AST, inputs: Sequence[BoolRef]) -> BoolRef:
    if isinstance(node, ast.Expression):
        return _parse_expr(node.body, inputs)

    if isinstance(node, ast.Name):
        name = node.id
        if name.lower() == "true":
            return BoolVal(True)
        if name.lower() == "false":
            return BoolVal(False)
        if name.startswith("I") and name[1:].isdigit():
            idx = int(name[1:])
            if 0 <= idx < len(inputs):
                return inputs[idx]
        raise ValueError(f"Unknown symbol in quantified output expression: {name}")

    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return BoolVal(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
        return Not(_parse_expr(node.operand, inputs))

    if isinstance(node, ast.BoolOp):
        values = [_parse_expr(value, inputs) for value in node.values]
        if isinstance(node.op, ast.And):
            return And(*values)
        if isinstance(node.op, ast.Or):
            return Or(*values)

    if isinstance(node, ast.BinOp):
        left = _parse_expr(node.left, inputs)
        right = _parse_expr(node.right, inputs)
        if isinstance(node.op, ast.BitAnd):
            return And(left, right)
        if isinstance(node.op, ast.BitOr):
            return Or(left, right)
        if isinstance(node.op, ast.BitXor):
            return Xor(left, right)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = node.func.id.lower()
        args = node.args
        if func == "count_bit":
            if len(args) < 2:
                raise ValueError("COUNT_BIT expects inputs followed by a bit index")
            bit = _int_literal(args[-1])
            terms = [_parse_expr(arg, inputs) for arg in args[:-1]]
            count = Sum([If(term, 1, 0) for term in terms])
            modulus = 1 << (bit + 1)
            threshold = 1 << bit
            return (count % modulus) >= threshold

        if func in {"not", "inv"}:
            if len(args) != 1:
                raise ValueError(f"{node.func.id} expects one argument")
            return Not(_parse_expr(args[0], inputs))

        parsed_args = [_parse_expr(arg, inputs) for arg in args]
        if func == "and":
            return And(*parsed_args)
        if func == "or":
            return Or(*parsed_args)
        if func == "xor":
            if not parsed_args:
                raise ValueError("XOR expects at least one argument")
            result = parsed_args[0]
            for arg in parsed_args[1:]:
                result = Xor(result, arg)
            return result
        if func == "maj":
            if len(parsed_args) != 3:
                raise ValueError("MAJ expects three arguments")
            return Or(
                And(parsed_args[0], parsed_args[1]),
                And(parsed_args[0], parsed_args[2]),
                And(parsed_args[1], parsed_args[2]),
            )
        if func == "ite":
            if len(parsed_args) != 3:
                raise ValueError("ITE expects three arguments")
            return If(parsed_args[0], parsed_args[1], parsed_args[2])

    raise ValueError(f"Unsupported quantified output expression: {ast.dump(node)}")


def _parse_output_expr(expr: str, inputs: Sequence[BoolRef]) -> BoolRef:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid quantified output expression {expr!r}: {exc}") from exc
    return _parse_expr(tree, inputs)


def _example_condition(example_inputs: Sequence[Optional[bool]], inputs: Sequence[BoolRef]) -> BoolRef:
    terms: List[BoolRef] = []
    for expected, actual in zip(example_inputs, inputs):
        if expected is None:
            continue
        terms.append(actual if bool(expected) else Not(actual))
    return And(*terms) if terms else BoolVal(True)


def _dnf_constraints_from_examples(
    cfg: Dict[str, Any],
    inputs: Sequence[BoolRef],
    actual_outputs: Sequence[BoolRef],
) -> List[BoolRef]:
    ctype = cfg.get("type")
    if ctype:
        examples, num_inputs, num_outputs = get_plugin(ctype)(cfg)
    elif "examples" in cfg:
        num_inputs = int(cfg["num_inputs"])
        num_outputs = int(cfg["num_outputs"])
        examples = cfg["examples"]
    else:
        raise ValueError("No symbolic spec or examples are available for quantified synthesis")

    if len(inputs) != num_inputs or len(actual_outputs) != num_outputs:
        raise ValueError("Symbolic and example dataset sizes do not match")

    constraints: List[BoolRef] = []
    for out_idx in range(num_outputs):
        care_terms: List[BoolRef] = []
        true_terms: List[BoolRef] = []
        for ex in examples:
            outs = ex["outputs"]
            if outs[out_idx] is None:
                continue
            condition = _example_condition(ex["inputs"], inputs)
            care_terms.append(condition)
            if bool(outs[out_idx]):
                true_terms.append(condition)

        if not care_terms:
            continue
        care = Or(*care_terms)
        expected = Or(*true_terms) if true_terms else BoolVal(False)
        constraints.append(Implies(care, actual_outputs[out_idx] == expected))

    return constraints


def _build_symbolic_constraints(
    cfg: Dict[str, Any],
    inputs: Sequence[BoolRef],
    actual_outputs: Sequence[BoolRef],
) -> List[BoolRef]:
    output_exprs = cfg.get("quantified_outputs", cfg.get("outputs"))
    if output_exprs is None and cfg.get("_quantified_outputs") is not None:
        output_exprs = cfg["_quantified_outputs"]

    if output_exprs is not None:
        if len(output_exprs) != len(actual_outputs):
            raise ValueError("Number of quantified output expressions does not match num_outputs")
        return [
            actual_outputs[idx] == _parse_output_expr(str(expr), inputs)
            for idx, expr in enumerate(output_exprs)
        ]

    return _dnf_constraints_from_examples(cfg, inputs, actual_outputs)


def _load_quantified_config(args: argparse.Namespace) -> Tuple[Dict[str, Any], int, int, int]:
    if args.config:
        cfg = _load_config(Path(args.config))
    elif args.dataset:
        cfg = get_plugin_config(args.dataset)
    else:
        raise SystemExit("Either --dataset or --config must be specified")

    quantified_outputs = cfg.get("quantified_outputs", cfg.get("outputs"))
    if quantified_outputs is None and cfg.get("type"):
        try:
            q_num_inputs, q_num_outputs, quantified_outputs = get_quantified_plugin(cfg["type"])(cfg)
            cfg["_quantified_outputs"] = quantified_outputs
            cfg.setdefault("num_inputs", q_num_inputs)
            cfg.setdefault("num_outputs", q_num_outputs)
        except KeyError:
            pass

    if "num_inputs" in cfg:
        num_inputs = int(cfg["num_inputs"])
    elif cfg.get("type"):
        _examples, num_inputs, _num_outputs = get_plugin(cfg["type"])(cfg)
    else:
        raise ValueError("Quantified synthesis config must specify num_inputs")

    if "num_outputs" in cfg:
        num_outputs = int(cfg["num_outputs"])
    else:
        outputs = cfg.get("quantified_outputs", cfg.get("outputs", cfg.get("_quantified_outputs")))
        if outputs is None:
            if cfg.get("type"):
                _examples, _num_inputs, num_outputs = get_plugin(cfg["type"])(cfg)
            else:
                raise ValueError("Quantified synthesis config must specify num_outputs or symbolic outputs")
        else:
            num_outputs = len(outputs)

    instructions = int(cfg.get("instructions", DEFAULT_PROGRAM_LENGTH))
    if args.instructions is not None:
        instructions = args.instructions
    if instructions < 0:
        raise ValueError("instructions must be non-negative")

    return cfg, num_inputs, num_outputs, instructions


def _build_symbolic_test(
    inputs: Sequence[BoolRef],
    tag: str,
    spec: ProgramSpec,
    options: EncodingOptions,
) -> Tuple[List[BoolRef], List[BoolRef]]:
    constraints: List[BoolRef] = []
    values: List[BoolRef] = [BoolVal(True), *inputs]

    for instr in range(spec.program_length):
        op = BitVec(f"OP_{instr}", OP_BITS)
        src1 = BitVec(f"S1_{instr}", spec.idx_bits)
        src2 = BitVec(f"S2_{instr}", spec.idx_bits)

        left_expr = _select_bv(values, src1, spec.idx_bits)
        right_expr = _select_bv(values, src2, spec.idx_bits)
        values.append(_operator_expr(op, left_expr, right_expr))

    outputs: List[BoolRef] = []
    for out_idx in range(spec.num_outputs):
        selector = BitVec(f"OUT_{out_idx}_idx", spec.idx_bits)
        outputs.append(_select_bv(values, selector, spec.idx_bits))

    return constraints, outputs


def synthesize_quantified(
    cfg: Dict[str, Any],
    spec: ProgramSpec,
    options: EncodingOptions,
) -> Tuple[str, Optional[List[Tuple[Optional[int], int, int]]], Optional[List[int]], float]:
    inputs = [Bool(f"I{idx}") for idx in range(spec.num_inputs)]
    test_constraints, actual_outputs = _build_symbolic_test(inputs, "q", spec, options)
    spec_constraints = _build_symbolic_constraints(cfg, inputs, actual_outputs)

    body = And(
        *test_constraints,
        *spec_constraints,
    )

    solver = Solver()
    solver.add(*_build_program(spec, options))
    solver.add(ForAll(inputs, body))

    start = time.time()
    result = solver.check()
    elapsed = time.time() - start
    if result != sat:
        return str(result), None, None, elapsed

    instrs, outputs = _extract_program_from_model(solver.model(), spec)
    return str(result), instrs, outputs, elapsed


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Synthesize a logic program with quantified Z3 constraints")
    parser.add_argument("--dataset", choices=list(available_plugins().keys()), default="adder", help="Built-in dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to a quantified JSON config")
    parser.add_argument("--instructions", type=int, default=None, help="Override number of SSA instructions")
    parser.add_argument("--encode-boolean", action="store_true", help="Enable boolean selector guards")
    parser.add_argument("--force-ordered", action="store_true", help="Enable constraint on ordered instructions")
    parser.add_argument("--force-useful", action="store_true", help="Enable constraint on useful instructions")
    parser.add_argument("--output-blif", action="store_true", help="Output the found program in BLIF format")
    parser.add_argument("--quiet", action="store_true", help="Suppress informational output")

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    options = EncodingOptions(
        encode_boolean=args.encode_boolean,
        force_ordered=args.force_ordered,
        force_useful=args.force_useful,
    )

    try:
        cfg, num_inputs, num_outputs, instructions = _load_quantified_config(args)
        spec = ProgramSpec(num_inputs=num_inputs, num_outputs=num_outputs, program_length=instructions)
        logger.info("Using quantified configuration: %s", {**cfg, "instructions": instructions})
        result, instrs, outputs, elapsed = synthesize_quantified(cfg, spec, options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if result == "sat":
        logger.info("SAT in %.3f seconds", elapsed)
        assert instrs is not None and outputs is not None
        _emit_program(instrs, outputs, spec.num_inputs, spec.num_outputs, args.output_blif)
        return

    logger.info("%s in %.3f seconds", result.upper(), elapsed)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
