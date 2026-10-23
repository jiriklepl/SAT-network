#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

IOList = List[Optional[bool]]
Example = Dict[str, IOList]

DEFAULT_PROGRAM_LENGTH = 16
DEFAULT_MAX_MATERIALIZED_EXAMPLES = 65536
DEFAULT_BASIS_SIZE = 64


def _datasets_directory() -> Path:
    env_dir = os.environ.get("SAT_SYNTH_DATASETS_DIR")
    if env_dir:
        candidate = Path(env_dir)
        if candidate.is_dir():
            return candidate
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "cpp" / "datasets",
        script_dir / "datasets",
        Path.cwd() / "cpp" / "datasets",
        Path.cwd() / "datasets",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Could not locate datasets directory")


def available_dataset_names() -> List[str]:
    directory = _datasets_directory()
    return sorted(entry.stem for entry in directory.iterdir() if entry.is_file() and entry.suffix == ".json")


def load_dataset_spec(name: str) -> Dict[str, Any]:
    if not name or "/" in name or "\\" in name or ".." in name:
        raise KeyError(f"unknown dataset: {name}")
    path = _datasets_directory() / f"{name}.json"
    if not path.is_file():
        raise KeyError(f"unknown dataset: {name}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def default_dataset_config(name: str) -> Dict[str, Any]:
    spec = load_dataset_spec(name)
    cfg = deepcopy(spec.get("defaults", {}))
    cfg["type"] = name
    cfg["instructions"] = spec.get("instructions", DEFAULT_PROGRAM_LENGTH)
    return cfg


def _required_bits(state_count: int) -> int:
    max_value = max(0, state_count - 1)
    width = 0
    while True:
        width += 1
        max_value >>= 1
        if max_value == 0:
            break
    return width


def _c_div(lhs: int, rhs: int) -> int:
    if rhs == 0:
        raise ValueError("Division by zero in dataset expression")
    quotient = abs(lhs) // abs(rhs)
    return -quotient if (lhs < 0) ^ (rhs < 0) else quotient


def _c_mod(lhs: int, rhs: int) -> int:
    if rhs == 0:
        raise ValueError("Modulo by zero in dataset expression")
    return lhs - _c_div(lhs, rhs) * rhs


@dataclass
class _EvalEnv:
    scalars: Dict[str, int] = field(default_factory=dict)
    tuples: Dict[str, List[int]] = field(default_factory=dict)

    def copy(self) -> _EvalEnv:
        return _EvalEnv(
            scalars=dict(self.scalars),
            tuples={name: list(values) for name, values in self.tuples.items()},
        )


class _ExprNode:
    def eval_scalar(self, env: _EvalEnv) -> int:
        raise TypeError("Bit-list expression cannot be evaluated as a scalar")

    def emit_bits(self, env: _EvalEnv, out: IOList) -> None:
        out.append(self.eval_scalar(env) != 0)


@dataclass(frozen=True)
class _LiteralExpr(_ExprNode):
    value: int

    def eval_scalar(self, env: _EvalEnv) -> int:
        return self.value


class _NullExpr(_ExprNode):
    def eval_scalar(self, env: _EvalEnv) -> int:
        raise ValueError("null cannot be evaluated as a scalar")

    def emit_bits(self, env: _EvalEnv, out: IOList) -> None:
        out.append(None)


@dataclass(frozen=True)
class _ScalarVarExpr(_ExprNode):
    name: str

    def eval_scalar(self, env: _EvalEnv) -> int:
        return env.scalars[self.name]


@dataclass(frozen=True)
class _TupleIndexExpr(_ExprNode):
    tuple_name: str
    index: _ExprNode

    def eval_scalar(self, env: _EvalEnv) -> int:
        idx = self.index.eval_scalar(env)
        tup = env.tuples[self.tuple_name]
        if idx < 0 or idx >= len(tup):
            raise IndexError("Tuple index out of bounds")
        return tup[idx]


@dataclass(frozen=True)
class _UnaryExpr(_ExprNode):
    op: str
    operand: _ExprNode

    def eval_scalar(self, env: _EvalEnv) -> int:
        val = self.operand.eval_scalar(env)
        if self.op == "!":
            return 1 if val == 0 else 0
        if self.op == "-":
            return -val
        if self.op == "~":
            return ~val
        raise ValueError(f"Unsupported unary operator: {self.op}")


@dataclass(frozen=True)
class _BinaryExpr(_ExprNode):
    op: str
    left: _ExprNode
    right: _ExprNode

    def eval_scalar(self, env: _EvalEnv) -> int:
        if self.op == "&&":
            return 1 if (self.left.eval_scalar(env) != 0 and self.right.eval_scalar(env) != 0) else 0
        if self.op == "||":
            return 1 if (self.left.eval_scalar(env) != 0 or self.right.eval_scalar(env) != 0) else 0
        lhs = self.left.eval_scalar(env)
        rhs = self.right.eval_scalar(env)
        if self.op == "+":
            return lhs + rhs
        if self.op == "-":
            return lhs - rhs
        if self.op == "*":
            return lhs * rhs
        if self.op == "/":
            return _c_div(lhs, rhs)
        if self.op == "%":
            return _c_mod(lhs, rhs)
        if self.op == "<<":
            return lhs << rhs
        if self.op == ">>":
            return lhs >> rhs
        if self.op == "<":
            return 1 if lhs < rhs else 0
        if self.op == "<=":
            return 1 if lhs <= rhs else 0
        if self.op == ">":
            return 1 if lhs > rhs else 0
        if self.op == ">=":
            return 1 if lhs >= rhs else 0
        if self.op == "==":
            return 1 if lhs == rhs else 0
        if self.op == "!=":
            return 1 if lhs != rhs else 0
        if self.op == "&":
            return lhs & rhs
        if self.op == "^":
            return lhs ^ rhs
        if self.op == "|":
            return lhs | rhs
        raise ValueError(f"Unsupported binary operator: {self.op}")


@dataclass(frozen=True)
class _TernaryExpr(_ExprNode):
    cond: _ExprNode
    then_expr: _ExprNode
    else_expr: _ExprNode

    def eval_scalar(self, env: _EvalEnv) -> int:
        return self.then_expr.eval_scalar(env) if self.cond.eval_scalar(env) != 0 else self.else_expr.eval_scalar(env)

    def emit_bits(self, env: _EvalEnv, out: IOList) -> None:
        if self.cond.eval_scalar(env) != 0:
            self.then_expr.emit_bits(env, out)
        else:
            self.else_expr.emit_bits(env, out)


@dataclass(frozen=True)
class _RequiredBitsExpr(_ExprNode):
    arg: _ExprNode

    def eval_scalar(self, env: _EvalEnv) -> int:
        return _required_bits(self.arg.eval_scalar(env))


@dataclass(frozen=True)
class _PopcountExpr(_ExprNode):
    arg: _ExprNode

    def eval_scalar(self, env: _EvalEnv) -> int:
        val = self.arg.eval_scalar(env) & 0xFFFFFFFFFFFFFFFF
        return val.bit_count()


@dataclass(frozen=True)
class _CountExpr(_ExprNode):
    tuple_name: str
    target: _ExprNode
    start_index: Optional[_ExprNode] = None

    def eval_scalar(self, env: _EvalEnv) -> int:
        tup = env.tuples[self.tuple_name]
        target_val = self.target.eval_scalar(env)
        start = max(0, self.start_index.eval_scalar(env)) if self.start_index is not None else 0
        return sum(1 for item in tup[start:] if item == target_val)


@dataclass(frozen=True)
class _BitExpr(_ExprNode):
    arg: _ExprNode

    def eval_scalar(self, env: _EvalEnv) -> int:
        return 1 if self.arg.eval_scalar(env) != 0 else 0

    def emit_bits(self, env: _EvalEnv, out: IOList) -> None:
        out.append(self.arg.eval_scalar(env) != 0)


def _push_bits(out: IOList, value: int, width: int) -> None:
    if width < 0:
        raise ValueError("bits() width must be non-negative")
    for bit in range(width):
        out.append(((value >> bit) & 1) != 0)


@dataclass(frozen=True)
class _BitsExpr(_ExprNode):
    value: _ExprNode
    width: _ExprNode

    def emit_bits(self, env: _EvalEnv, out: IOList) -> None:
        _push_bits(out, self.value.eval_scalar(env), self.width.eval_scalar(env))


@dataclass(frozen=True)
class _StateBitsExpr(_ExprNode):
    tuple_name: str
    width: _ExprNode

    def emit_bits(self, env: _EvalEnv, out: IOList) -> None:
        w = self.width.eval_scalar(env)
        for state in env.tuples[self.tuple_name]:
            _push_bits(out, state, w)


class _ExprParser:
    def __init__(
        self,
        source: str,
        resolve_var: Callable[[str], str],
        resolve_tuple: Callable[[str], str],
    ) -> None:
        self._source = source
        self._pos = 0
        self._resolve_var = resolve_var
        self._resolve_tuple = resolve_tuple

    def parse(self) -> _ExprNode:
        result = self._parse_ternary()
        self._skip_ws()
        if self._pos != len(self._source):
            raise ValueError(f"Unexpected trailing characters in expression: {self._source}")
        return result

    def _skip_ws(self) -> None:
        while self._pos < len(self._source) and self._source[self._pos].isspace():
            self._pos += 1

    def _match(self, token: str) -> bool:
        self._skip_ws()
        if self._source.startswith(token, self._pos):
            self._pos += len(token)
            return True
        return False

    def _match_op(self, op: str, disallow_prefix: str = "") -> bool:
        self._skip_ws()
        if disallow_prefix and self._source.startswith(disallow_prefix, self._pos):
            return False
        if self._source.startswith(op, self._pos):
            self._pos += len(op)
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._match(token):
            raise ValueError(f"Expected '{token}' in expression: {self._source}")

    def _parse_identifier(self) -> str:
        self._skip_ws()
        start = self._pos
        while self._pos < len(self._source) and (
            self._source[self._pos].isalnum() or self._source[self._pos] == "_"
        ):
            self._pos += 1
        if self._pos == start:
            raise ValueError(f"Expected identifier in expression: {self._source}")
        return self._source[start : self._pos]

    def _parse_ternary(self) -> _ExprNode:
        expr = self._parse_logical_or()
        if self._match("?"):
            then_expr = self._parse_ternary()
            self._expect(":")
            else_expr = self._parse_ternary()
            return _TernaryExpr(expr, then_expr, else_expr)
        return expr

    def _parse_logical_or(self) -> _ExprNode:
        lhs = self._parse_logical_and()
        while self._match("||"):
            lhs = _BinaryExpr("||", lhs, self._parse_logical_and())
        return lhs

    def _parse_logical_and(self) -> _ExprNode:
        lhs = self._parse_bit_or()
        while self._match("&&"):
            lhs = _BinaryExpr("&&", lhs, self._parse_bit_or())
        return lhs

    def _parse_bit_or(self) -> _ExprNode:
        lhs = self._parse_bit_xor()
        while self._match_op("|", "||"):
            lhs = _BinaryExpr("|", lhs, self._parse_bit_xor())
        return lhs

    def _parse_bit_xor(self) -> _ExprNode:
        lhs = self._parse_bit_and()
        while self._match("^"):
            lhs = _BinaryExpr("^", lhs, self._parse_bit_and())
        return lhs

    def _parse_bit_and(self) -> _ExprNode:
        lhs = self._parse_equality()
        while self._match_op("&", "&&"):
            lhs = _BinaryExpr("&", lhs, self._parse_equality())
        return lhs

    def _parse_equality(self) -> _ExprNode:
        lhs = self._parse_relational()
        while True:
            if self._match("=="):
                lhs = _BinaryExpr("==", lhs, self._parse_relational())
            elif self._match("!="):
                lhs = _BinaryExpr("!=", lhs, self._parse_relational())
            else:
                break
        return lhs

    def _parse_relational(self) -> _ExprNode:
        lhs = self._parse_shift()
        while True:
            if self._match("<="):
                lhs = _BinaryExpr("<=", lhs, self._parse_shift())
            elif self._match(">="):
                lhs = _BinaryExpr(">=", lhs, self._parse_shift())
            elif self._match_op("<", "<<"):
                lhs = _BinaryExpr("<", lhs, self._parse_shift())
            elif self._match_op(">", ">>"):
                lhs = _BinaryExpr(">", lhs, self._parse_shift())
            else:
                break
        return lhs

    def _parse_shift(self) -> _ExprNode:
        lhs = self._parse_additive()
        while True:
            if self._match("<<"):
                lhs = _BinaryExpr("<<", lhs, self._parse_additive())
            elif self._match(">>"):
                lhs = _BinaryExpr(">>", lhs, self._parse_additive())
            else:
                break
        return lhs

    def _parse_additive(self) -> _ExprNode:
        lhs = self._parse_multiplicative()
        while True:
            if self._match("+"):
                lhs = _BinaryExpr("+", lhs, self._parse_multiplicative())
            elif self._match("-"):
                lhs = _BinaryExpr("-", lhs, self._parse_multiplicative())
            else:
                break
        return lhs

    def _parse_multiplicative(self) -> _ExprNode:
        lhs = self._parse_unary()
        while True:
            if self._match("*"):
                lhs = _BinaryExpr("*", lhs, self._parse_unary())
            elif self._match("/"):
                lhs = _BinaryExpr("/", lhs, self._parse_unary())
            elif self._match("%"):
                lhs = _BinaryExpr("%", lhs, self._parse_unary())
            else:
                break
        return lhs

    def _parse_unary(self) -> _ExprNode:
        if self._match_op("!", "!="):
            return _UnaryExpr("!", self._parse_unary())
        if self._match("-"):
            return _UnaryExpr("-", self._parse_unary())
        if self._match("+"):
            return self._parse_unary()
        if self._match("~"):
            return _UnaryExpr("~", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> _ExprNode:
        self._skip_ws()
        if self._match("("):
            inner = self._parse_ternary()
            self._expect(")")
            return inner
        if self._pos < len(self._source) and self._source[self._pos].isdigit():
            start = self._pos
            while self._pos < len(self._source) and self._source[self._pos].isdigit():
                self._pos += 1
            return _LiteralExpr(int(self._source[start : self._pos]))

        ident = self._parse_identifier()
        if ident == "true":
            return _LiteralExpr(1)
        if ident == "false":
            return _LiteralExpr(0)
        if ident == "null":
            return _NullExpr()

        if self._match("("):
            if ident == "required_bits":
                arg = self._parse_ternary()
                self._expect(")")
                return _RequiredBitsExpr(arg)
            if ident == "popcount":
                arg = self._parse_ternary()
                self._expect(")")
                return _PopcountExpr(arg)
            if ident == "bit":
                arg = self._parse_ternary()
                self._expect(")")
                return _BitExpr(arg)
            if ident == "bits":
                val = self._parse_ternary()
                self._expect(",")
                width = self._parse_ternary()
                self._expect(")")
                return _BitsExpr(val, width)
            if ident == "state_bits":
                tuple_name = self._parse_identifier()
                self._expect(",")
                width = self._parse_ternary()
                self._expect(")")
                return _StateBitsExpr(self._resolve_tuple(tuple_name), width)
            if ident == "count":
                tuple_name = self._parse_identifier()
                self._expect(",")
                target = self._parse_ternary()
                start_index: Optional[_ExprNode] = None
                if self._match(","):
                    start_index = self._parse_ternary()
                self._expect(")")
                return _CountExpr(self._resolve_tuple(tuple_name), target, start_index)
            raise ValueError(f"Unknown function '{ident}' in dataset expression")

        if self._match("["):
            idx = self._parse_ternary()
            self._expect("]")
            return _TupleIndexExpr(self._resolve_tuple(ident), idx)

        return _ScalarVarExpr(self._resolve_var(ident))


def _lookup_dotted_path(cfg: Dict[str, Any], dotted_path: str) -> Any:
    curr: Any = cfg
    for key in dotted_path.split("."):
        if not isinstance(curr, dict) or key not in curr or curr[key] is None:
            return None
        curr = curr[key]
    return curr


def _read_param_value(runtime_cfg: Dict[str, Any], param_def: Dict[str, Any]) -> int:
    default_val = param_def["default"]
    found = None
    path_spec = param_def.get("path")
    if isinstance(path_spec, str):
        found = _lookup_dotted_path(runtime_cfg, path_spec)
    elif isinstance(path_spec, list):
        for candidate in path_spec:
            found = _lookup_dotted_path(runtime_cfg, str(candidate))
            if found is not None:
                break
    target = default_val if found is None else found
    if isinstance(default_val, bool):
        return 1 if bool(target) else 0
    return int(target)


@dataclass(frozen=True)
class _RangeQuantifier:
    var: str
    start: int
    end: int


@dataclass(frozen=True)
class _ValuesQuantifier:
    var: str
    values: List[int]


@dataclass(frozen=True)
class _TupleQuantifier:
    var: str
    size: int
    start: int
    end: int


_Quantifier = _RangeQuantifier | _ValuesQuantifier | _TupleQuantifier


@dataclass
class _CompiledDatasetSpec:
    env: _EvalEnv
    num_inputs: int
    num_outputs: int
    instructions: int
    domain_size: int
    quantifiers: List[_Quantifier]
    row_lets: List[Tuple[str, _ExprNode]]
    where: Optional[_ExprNode]
    inputs: List[_ExprNode]
    outputs: List[_ExprNode]

    def evaluate_point(self, scratch: _EvalEnv) -> Optional[Example]:
        for name, ast in self.row_lets:
            scratch.scalars[name] = ast.eval_scalar(scratch)
        if self.where is not None and self.where.eval_scalar(scratch) == 0:
            return None
        in_bits: IOList = []
        for ast in self.inputs:
            ast.emit_bits(scratch, in_bits)
        out_bits: IOList = []
        for ast in self.outputs:
            ast.emit_bits(scratch, out_bits)
        if len(in_bits) != self.num_inputs or len(out_bits) != self.num_outputs:
            raise ValueError("Example length does not match declared input/output sizes")
        if any(bit is None for bit in in_bits):
            raise ValueError("input values cannot be null")
        return {
            "inputs": [bool(bit) for bit in in_bits],
            "outputs": [None if bit is None else bool(bit) for bit in out_bits],
        }


def _compile_dataset_spec(spec: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> _CompiledDatasetSpec:
    env = _EvalEnv()
    for name, param_def in spec.get("params", {}).items():
        env.scalars[name] = _read_param_value(runtime_cfg, param_def)

    raw_static_lets: Dict[str, str] = {
        name: expr if isinstance(expr, str) else json.dumps(expr)
        for name, expr in spec.get("let", {}).items()
    }
    compiling_static: set[str] = set()

    def disallow_tuple(t: str) -> str:
        raise ValueError(f"Tuple variable '{t}' is not available in static context")

    def resolve_static_var(name: str) -> str:
        if name in env.scalars:
            return name
        if name in raw_static_lets:
            if name in compiling_static:
                raise ValueError(f"Cyclic let definition for variable: {name}")
            compiling_static.add(name)
            ast = _ExprParser(raw_static_lets[name], resolve_static_var, disallow_tuple).parse()
            env.scalars[name] = ast.eval_scalar(env)
            compiling_static.remove(name)
            return name
        raise ValueError(f"Unknown variable in static dataset expression: {name}")

    for name in raw_static_lets:
        resolve_static_var(name)

    def eval_static_expr(node: Any) -> int:
        if isinstance(node, bool):
            return 1 if node else 0
        if isinstance(node, int):
            return node
        if isinstance(node, str):
            return _ExprParser(node, resolve_static_var, disallow_tuple).parse().eval_scalar(env)
        raise ValueError("Expected integer, boolean, or expression string in dataset config")

    for constraint in spec.get("constraints", []):
        if eval_static_expr(constraint["check"]) == 0:
            raise ValueError(str(constraint["error"]))

    num_inputs = eval_static_expr(spec["num_inputs"])
    num_outputs = eval_static_expr(spec["num_outputs"])
    default_instructions = (
        eval_static_expr(spec["instructions"]) if "instructions" in spec else DEFAULT_PROGRAM_LENGTH
    )
    instructions = (
        eval_static_expr(runtime_cfg["instructions"])
        if "instructions" in runtime_cfg and runtime_cfg["instructions"] is not None
        else default_instructions
    )

    quantifiers: List[_Quantifier] = []
    domain_size = 1
    for q_json in spec.get("forall", []):
        var_name = str(q_json["var"])
        if "range" in q_json:
            start = eval_static_expr(q_json["range"][0])
            end = eval_static_expr(q_json["range"][1])
            env.scalars[var_name] = start
            quantifiers.append(_RangeQuantifier(var_name, start, end))
            domain_size *= max(0, end - start)
        elif "values" in q_json:
            values = [eval_static_expr(v) for v in q_json["values"]]
            env.scalars[var_name] = values[0] if values else 0
            quantifiers.append(_ValuesQuantifier(var_name, values))
            domain_size *= len(values)
        elif "tuple" in q_json:
            tuple_spec = q_json["tuple"]
            size = eval_static_expr(tuple_spec["size"])
            start = eval_static_expr(tuple_spec["range"][0])
            end = eval_static_expr(tuple_spec["range"][1])
            env.tuples[var_name] = [start] * size
            quantifiers.append(_TupleQuantifier(var_name, size, start, end))
            domain_size *= max(0, end - start) ** size
        else:
            raise ValueError(f"Unsupported quantifier specification for variable: {var_name}")

    if "sample_section" in spec:
        section = str(spec["sample_section"])
        section_cfg = runtime_cfg.get(section)
        if isinstance(section_cfg, dict) and "max_examples" in section_cfg and section_cfg["max_examples"] is not None:
            max_examples = int(section_cfg["max_examples"])
            if max_examples < domain_size:
                if max_examples <= 0:
                    raise ValueError(f"{section}.max_examples must be positive")
                raise ValueError(f"{section}.max_examples sampling is not supported")

    raw_row_lets: Dict[str, str] = {
        name: expr if isinstance(expr, str) else json.dumps(expr)
        for name, expr in spec.get("let_row", {}).items()
    }
    compiling_row: set[str] = set()
    row_lets: List[Tuple[str, _ExprNode]] = []

    def resolve_tuple_var(name: str) -> str:
        if name in env.tuples:
            return name
        raise ValueError(f"Unknown tuple variable in dataset expression: {name}")

    def resolve_row_var(name: str) -> str:
        if name in env.scalars:
            return name
        if name in raw_row_lets:
            if name in compiling_row:
                raise ValueError(f"Cyclic let_row definition for variable: {name}")
            compiling_row.add(name)
            ast = _ExprParser(raw_row_lets[name], resolve_row_var, resolve_tuple_var).parse()
            env.scalars[name] = 0
            row_lets.append((name, ast))
            compiling_row.remove(name)
            return name
        raise ValueError(f"Unknown variable in dataset expression: {name}")

    for name in raw_row_lets:
        resolve_row_var(name)

    def compile_row_expr(node: Any) -> _ExprNode:
        text = node if isinstance(node, str) else json.dumps(node)
        return _ExprParser(text, resolve_row_var, resolve_tuple_var).parse()

    where_expr = compile_row_expr(spec["where"]) if "where" in spec else None
    input_exprs = [compile_row_expr(item) for item in spec["inputs"]]
    output_exprs = [compile_row_expr(item) for item in spec["outputs"]]

    return _CompiledDatasetSpec(
        env=env,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        instructions=instructions,
        domain_size=domain_size,
        quantifiers=quantifiers,
        row_lets=row_lets,
        where=where_expr,
        inputs=input_exprs,
        outputs=output_exprs,
    )


def _for_each_point(spec: _CompiledDatasetSpec, scratch: _EvalEnv, visit: Callable[[_EvalEnv], None]) -> None:
    def run(q_idx: int) -> None:
        if q_idx == len(spec.quantifiers):
            visit(scratch)
            return
        q = spec.quantifiers[q_idx]
        if isinstance(q, _RangeQuantifier):
            for val in range(q.start, q.end):
                scratch.scalars[q.var] = val
                run(q_idx + 1)
        elif isinstance(q, _ValuesQuantifier):
            for val in q.values:
                scratch.scalars[q.var] = val
                run(q_idx + 1)
        else:
            tup = scratch.tuples[q.var]

            def fill(cell: int) -> None:
                if cell == q.size:
                    run(q_idx + 1)
                    return
                for val in range(q.start, q.end):
                    tup[cell] = val
                    fill(cell + 1)

            fill(0)

    run(0)


def _assign_corner(spec: _CompiledDatasetSpec, env: _EvalEnv, high: bool) -> None:
    for q in spec.quantifiers:
        if isinstance(q, _RangeQuantifier):
            env.scalars[q.var] = max(q.start, q.end - 1) if high else q.start
        elif isinstance(q, _ValuesQuantifier):
            if q.values:
                env.scalars[q.var] = q.values[-1] if high else q.values[0]
        else:
            val = max(q.start, q.end - 1) if high else q.start
            env.tuples[q.var] = [val] * q.size


def _seed_basis(spec: _CompiledDatasetSpec, count: int = DEFAULT_BASIS_SIZE, seed: int = 0) -> List[Example]:
    basis: List[Example] = []
    seen: set[Tuple[Optional[bool], ...]] = set()
    env = spec.env.copy()
    target_count = max(1, count)

    def add_current() -> None:
        if len(basis) >= target_count:
            return
        ex = spec.evaluate_point(env)
        if ex is None:
            return
        key = tuple(ex["inputs"])
        if key not in seen:
            seen.add(key)
            basis.append(ex)

    for high in (False, True):
        _assign_corner(spec, env, high)
        add_current()

    for q in spec.quantifiers:
        if isinstance(q, _RangeQuantifier):
            low = q.start
            high_val = max(q.start, q.end - 1)
            for base_high in (False, True):
                _assign_corner(spec, env, base_high)
                env.scalars[q.var] = low if base_high else high_val
                add_current()
        elif isinstance(q, _ValuesQuantifier):
            if q.values:
                for base_high in (False, True):
                    _assign_corner(spec, env, base_high)
                    env.scalars[q.var] = q.values[0] if base_high else q.values[-1]
                    add_current()
        else:
            low = q.start
            high_val = max(q.start, q.end - 1)
            for base_high in (False, True):
                for cell in range(q.size):
                    _assign_corner(spec, env, base_high)
                    env.tuples[q.var][cell] = low if base_high else high_val
                    add_current()

    rng = random.Random(seed)
    max_attempts = target_count * 64 + 256
    for _ in range(max_attempts):
        if len(basis) >= target_count:
            break
        for q in spec.quantifiers:
            if isinstance(q, _RangeQuantifier):
                env.scalars[q.var] = rng.randint(q.start, max(q.start, q.end - 1))
            elif isinstance(q, _ValuesQuantifier):
                if q.values:
                    env.scalars[q.var] = q.values[rng.randrange(len(q.values))]
            else:
                hi = max(q.start, q.end - 1)
                env.tuples[q.var] = [rng.randint(q.start, hi) for _ in range(q.size)]
        add_current()

    return basis


def build_dataset_from_config(
    cfg: Dict[str, Any],
    *,
    max_materialized_examples: int = DEFAULT_MAX_MATERIALIZED_EXAMPLES,
    basis_size: int = DEFAULT_BASIS_SIZE,
    seed: int = 0,
) -> Tuple[List[Example], int, int, int, bool, int]:
    if "examples" in cfg:
        raw_examples = cfg["examples"]
        if not raw_examples:
            raise ValueError("Dataset contains no examples")
        num_inputs = int(cfg["num_inputs"])
        num_outputs = int(cfg["num_outputs"])
        instructions = int(cfg.get("instructions", DEFAULT_PROGRAM_LENGTH))
        examples: List[Example] = []
        for ex in raw_examples:
            inputs = ex["inputs"]
            outputs = ex["outputs"]
            if len(inputs) != num_inputs or len(outputs) != num_outputs:
                raise ValueError("Example length does not match declared input/output sizes")
            if any(v is None for v in inputs):
                raise ValueError("Cannot generate assumptions for examples with input don't-cares")
            examples.append(
                {
                    "inputs": [bool(v) for v in inputs],
                    "outputs": [None if v is None else bool(v) for v in outputs],
                }
            )
        return examples, num_inputs, num_outputs, instructions, True, len(examples)

    if "forall" in cfg:
        spec_json = cfg
    else:
        if "type" not in cfg:
            raise ValueError("dataset config without examples must contain a type")
        spec_json = load_dataset_spec(str(cfg["type"]))

    compiled = _compile_dataset_spec(spec_json, cfg)
    if compiled.domain_size <= max_materialized_examples:
        examples = []
        scratch = compiled.env.copy()

        def collect(point: _EvalEnv) -> None:
            ex = compiled.evaluate_point(point)
            if ex is not None:
                examples.append(ex)

        _for_each_point(compiled, scratch, collect)
        fully_materialized = True
    else:
        examples = _seed_basis(compiled, basis_size, seed)
        fully_materialized = False

    if not examples:
        raise ValueError("Dataset contains no examples")
    return (
        examples,
        compiled.num_inputs,
        compiled.num_outputs,
        compiled.instructions,
        fully_materialized,
        compiled.domain_size,
    )


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
        # so the generated assumptions satisfy structural constraints.
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


def _load_examples(args: argparse.Namespace) -> Tuple[List[Example], int, int, int, bool, int]:
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as file:
            cfg = json.load(file)
    else:
        cfg = default_dataset_config(args.dataset)

    max_materialized = getattr(args, "max_materialized_examples", DEFAULT_MAX_MATERIALIZED_EXAMPLES)
    examples, num_inputs, num_outputs, cfg_instructions, fully_materialized, domain_size = build_dataset_from_config(
        cfg,
        max_materialized_examples=max_materialized,
    )
    instructions = cfg_instructions
    if args.instructions is not None:
        instructions = args.instructions
    return examples, num_inputs, num_outputs, instructions, fully_materialized, domain_size


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
    fully_materialized: bool = True,
) -> Tuple[List[str], int]:
    if strategy not in {"circuit", "dnf"}:
        raise ValueError(f"Unknown assumption strategy: {strategy}")
    if strategy == "circuit" and dataset_name is not None:
        circuit = _maybe_circuit_assumptions(dataset_name, examples, num_inputs, num_outputs)
        if circuit is not None:
            return circuit
    if not fully_materialized:
        raise ValueError(
            "Cannot synthesize exact DNF assumptions for an unmaterialized dataset; "
            "use --strategy circuit or increase --max-materialized-examples"
        )

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
    parser = argparse.ArgumentParser(description="Build an assumption file from a declarative dataset specification")
    parser.add_argument("--dataset", choices=available_dataset_names(), default="life-compressed", help="Built-in dataset specification to encode")
    parser.add_argument("--config", type=str, default=None, help="Path to a custom JSON config")
    parser.add_argument("--instructions", type=int, default=None, help="Instruction budget to report in the header")
    parser.add_argument("--strategy", choices=["circuit", "dnf"], default="circuit", help="Use a cellular automata circuit when available, otherwise fall back to exact DNF")
    parser.add_argument("--output", type=str, default=None, help="Write assumptions to this file instead of stdout")
    parser.add_argument("--max-generated-instructions", type=int, default=20000, help="Abort if the generated assumption program exceeds this size")
    parser.add_argument(
        "--max-materialized-examples",
        type=int,
        default=DEFAULT_MAX_MATERIALIZED_EXAMPLES,
        help="Largest quantified dataset domain to unroll into explicit examples",
    )
    args = parser.parse_args()

    examples, num_inputs, num_outputs, configured_instructions, fully_materialized, domain_size = _load_examples(args)
    dataset_name = None if args.config else args.dataset
    lines, required_instructions = build_assumption_lines(
        examples,
        num_inputs,
        num_outputs,
        dataset_name=dataset_name,
        strategy=args.strategy,
        fully_materialized=fully_materialized,
    )
    if required_instructions > args.max_generated_instructions:
        raise SystemExit(
            f"Generated {required_instructions} instructions, exceeding --max-generated-instructions={args.max_generated_instructions}"
        )

    output_lines = [
        f"# dataset: {args.config if args.config else args.dataset}",
        f"# examples: {len(examples) if fully_materialized else f'{len(examples)} (sampled basis of {domain_size})'}",
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
