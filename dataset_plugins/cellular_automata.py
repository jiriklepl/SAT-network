"""Cellular automata dataset plugins."""

from __future__ import annotations

from itertools import product
import random
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from . import DatasetResult, Example, IOList, register_plugin


MOORE_CELLS = 9
VON_NEUMANN_CELLS = 5


def _bits(value: int, width: int) -> IOList:
    return [(value & (1 << bit)) != 0 for bit in range(width)]


def _state_bits(states: Sequence[int], bits_per_state: int) -> IOList:
    result: IOList = []
    for state in states:
        result.extend(_bits(state, bits_per_state))
    return result


def _iter_state_vectors(
    state_count: int,
    cell_count: int,
    cfg: Dict[str, Any],
    section: str,
) -> Iterable[Tuple[int, ...]]:
    section_cfg = cfg.get(section, {})
    total = state_count ** cell_count
    max_examples = section_cfg.get("max_examples")
    seed = int(section_cfg.get("seed", 0))

    if max_examples is None or int(max_examples) >= total:
        return product(range(state_count), repeat=cell_count)

    sample_size = int(max_examples)
    if sample_size <= 0:
        raise ValueError(f"{section}.max_examples must be positive")

    rng = random.Random(seed)
    samples: set[Tuple[int, ...]] = set()
    while len(samples) < sample_size:
        samples.add(tuple(rng.randrange(state_count) for _ in range(cell_count)))
    return sorted(samples)


def _build_state_dataset(
    cfg: Dict[str, Any],
    section: str,
    state_count: int,
    bits_per_state: int,
    cell_count: int,
    rule: Callable[[Tuple[int, ...]], Sequence[int]],
) -> DatasetResult:
    examples: List[Example] = []
    for states in _iter_state_vectors(state_count, cell_count, cfg, section):
        outputs: IOList = []
        for output_state in rule(states):
            outputs.extend(_bits(output_state, bits_per_state))
        examples.append({"inputs": _state_bits(states, bits_per_state), "outputs": outputs})

    return examples, cell_count * bits_per_state, len(rule(tuple([0] * cell_count))) * bits_per_state


def _build_summary_dataset(
    rows: Iterable[Tuple[IOList, int]],
    num_inputs: int,
    bits_per_output: int,
) -> DatasetResult:
    examples: List[Example] = []
    for inputs, output_state in rows:
        if len(inputs) != num_inputs:
            raise ValueError("Compressed input length does not match declared input count")
        examples.append({"inputs": inputs, "outputs": _bits(output_state, bits_per_output)})
    return examples, num_inputs, bits_per_output


def _moore_column_count_inputs(center: int, left_count: int, center_count: int, right_count: int, center_bits: int) -> IOList:
    return _bits(left_count, 2) + _bits(center_count, 2) + _bits(right_count, 2) + _bits(center, center_bits)


def _build_moore_column_count_dataset(
    state_count: int,
    center_bits: int,
    output_bits: int,
    rule: Callable[[int, int], int],
) -> DatasetResult:
    rows = []
    for center in range(state_count):
        for left_count in range(4):
            for center_count in range(3):
                for right_count in range(4):
                    relevant_neighbors = left_count + center_count + right_count
                    rows.append((
                        _moore_column_count_inputs(center, left_count, center_count, right_count, center_bits),
                        rule(center, relevant_neighbors),
                    ))
    return _build_summary_dataset(rows, center_bits + 6, output_bits)


def _moore_alive_count(states: Sequence[int], alive_state: int = 1) -> int:
    return sum(1 for state in states[1:] if state == alive_state)


def _build_life(cfg: Dict[str, Any]) -> DatasetResult:
    def rule(states: Tuple[int, ...]) -> Sequence[int]:
        center = states[0]
        alive_neighbors = _moore_alive_count(states)
        return [int(alive_neighbors == 3 or (center == 1 and alive_neighbors == 2))]

    return _build_state_dataset(cfg, "life", 2, 1, MOORE_CELLS, rule)


def _build_life_compressed(cfg: Dict[str, Any]) -> DatasetResult:
    return _build_moore_column_count_dataset(
        2,
        1,
        1,
        lambda center, relevant_neighbors: int(relevant_neighbors == 3 or (center == 1 and relevant_neighbors == 2)),
    )


def _build_maze(cfg: Dict[str, Any]) -> DatasetResult:
    def rule(states: Tuple[int, ...]) -> Sequence[int]:
        center = states[0]
        wall_neighbors = _moore_alive_count(states)
        next_state = wall_neighbors == 3 or (center == 1 and wall_neighbors < 6)
        return [int(next_state)]

    return _build_state_dataset(cfg, "maze", 2, 1, MOORE_CELLS, rule)


def _build_maze_compressed(cfg: Dict[str, Any]) -> DatasetResult:
    return _build_moore_column_count_dataset(
        2,
        1,
        1,
        lambda center, relevant_neighbors: int(relevant_neighbors == 3 or (center == 1 and relevant_neighbors < 6)),
    )


def _build_brian(cfg: Dict[str, Any]) -> DatasetResult:
    # States: 0=dead, 1=alive, 2=dying.
    def rule(states: Tuple[int, ...]) -> Sequence[int]:
        center = states[0]
        if center == 1:
            return [2]
        if center == 2:
            return [0]
        return [1 if _moore_alive_count(states, alive_state=1) == 2 else 0]

    return _build_state_dataset(cfg, "brian", 3, 2, MOORE_CELLS, rule)


def _build_brian_compressed(cfg: Dict[str, Any]) -> DatasetResult:
    def rule(center: int, relevant_neighbors: int) -> int:
        if center == 1:
            return 2
        if center == 2:
            return 0
        return 1 if relevant_neighbors == 2 else 0

    return _build_moore_column_count_dataset(3, 2, 2, rule)


def _build_fire(cfg: Dict[str, Any]) -> DatasetResult:
    # States: 0=empty, 1=tree, 2=fire, 3=ash.
    def rule(states: Tuple[int, ...]) -> Sequence[int]:
        center = states[0]
        has_fire_neighbor = any(state == 2 for state in states[1:])
        if center == 1:
            return [2 if has_fire_neighbor else 1]
        if center == 2:
            return [3]
        if center == 3:
            return [3 if has_fire_neighbor else 0]
        return [0]

    return _build_state_dataset(cfg, "fire", 4, 2, VON_NEUMANN_CELLS, rule)


def _build_fire_compressed(cfg: Dict[str, Any]) -> DatasetResult:
    rows = []
    for center in range(4):
        for left_count in range(2):
            for vertical_count in range(3):
                for right_count in range(2):
                    has_fire_neighbor = left_count + vertical_count + right_count > 0
                    if center == 1:
                        output = 2 if has_fire_neighbor else 1
                    elif center == 2:
                        output = 3
                    elif center == 3:
                        output = 3 if has_fire_neighbor else 0
                    else:
                        output = 0
                    rows.append((_bits(left_count, 1) + _bits(vertical_count, 2) + _bits(right_count, 1) + _bits(center, 2), output))
    return _build_summary_dataset(rows, 6, 2)


def _build_wire(cfg: Dict[str, Any]) -> DatasetResult:
    # States: 0=empty, 1=conductor, 2=head, 3=tail.
    def rule(states: Tuple[int, ...]) -> Sequence[int]:
        center = states[0]
        if center == 0:
            return [0]
        if center == 2:
            return [3]
        if center == 3:
            return [1]
        head_neighbors = sum(1 for state in states[1:] if state == 2)
        return [2 if head_neighbors in (1, 2) else 1]

    return _build_state_dataset(cfg, "wire", 4, 2, MOORE_CELLS, rule)


def _build_wire_compressed(cfg: Dict[str, Any]) -> DatasetResult:
    def rule(center: int, relevant_neighbors: int) -> int:
        if center == 0:
            return 0
        if center == 2:
            return 3
        if center == 3:
            return 1
        return 2 if relevant_neighbors in (1, 2) else 1

    return _build_moore_column_count_dataset(4, 2, 2, rule)


def _build_excitable(cfg: Dict[str, Any]) -> DatasetResult:
    state_count = int(cfg.get("excitable", {}).get("states", 8))
    if state_count < 3:
        raise ValueError("excitable.states must be at least 3")
    bits_per_state = max(1, (state_count - 1).bit_length())

    def rule(states: Tuple[int, ...]) -> Sequence[int]:
        center = states[0]
        if center == 0:
            return [1 if any(state == 1 for state in states[1:]) else 0]
        if center == state_count - 1:
            return [0]
        return [center + 1]

    return _build_state_dataset(cfg, "excitable", state_count, bits_per_state, MOORE_CELLS, rule)


def _build_excitable_compressed(cfg: Dict[str, Any]) -> DatasetResult:
    state_count = int(cfg.get("excitable-compressed", cfg.get("excitable", {})).get("states", 8))
    if state_count < 3:
        raise ValueError("excitable-compressed.states must be at least 3")
    bits_per_state = max(1, (state_count - 1).bit_length())

    def rule(center: int, relevant_neighbors: int) -> int:
        if center == 0:
            return 1 if relevant_neighbors > 0 else 0
        if center == state_count - 1:
            return 0
        return center + 1

    return _build_moore_column_count_dataset(state_count, bits_per_state, bits_per_state, rule)


def _build_cyclic(cfg: Dict[str, Any]) -> DatasetResult:
    state_count = int(cfg.get("cyclic", {}).get("states", 32))
    if state_count < 2:
        raise ValueError("cyclic.states must be at least 2")
    bits_per_state = max(1, (state_count - 1).bit_length())

    def rule(states: Tuple[int, ...]) -> Sequence[int]:
        center = states[0]
        successor = (center + 1) % state_count
        return [successor if any(state == successor for state in states[1:]) else center]

    return _build_state_dataset(cfg, "cyclic", state_count, bits_per_state, MOORE_CELLS, rule)


def _build_cyclic_compressed(cfg: Dict[str, Any]) -> DatasetResult:
    state_count = int(cfg.get("cyclic-compressed", cfg.get("cyclic", {})).get("states", 32))
    if state_count < 2:
        raise ValueError("cyclic-compressed.states must be at least 2")
    bits_per_state = max(1, (state_count - 1).bit_length())

    def rule(center: int, relevant_neighbors: int) -> int:
        successor = (center + 1) % state_count
        return successor if relevant_neighbors > 0 else center

    return _build_moore_column_count_dataset(state_count, bits_per_state, bits_per_state, rule)


def _build_fluid(cfg: Dict[str, Any]) -> DatasetResult:
    # Per-cell bit order: up, down, left, right.
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3

    examples: List[Example] = []
    for states in _iter_state_vectors(16, 4, cfg, "fluid"):
        incoming_up = (states[SOUTH] & (1 << UP)) != 0
        incoming_down = (states[NORTH] & (1 << DOWN)) != 0
        incoming_left = (states[EAST] & (1 << LEFT)) != 0
        incoming_right = (states[WEST] & (1 << RIGHT)) != 0

        incoming = [incoming_up, incoming_down, incoming_left, incoming_right]
        if incoming == [True, True, False, False]:
            output = [False, False, True, True]
        elif incoming == [False, False, True, True]:
            output = [True, True, False, False]
        else:
            output = incoming

        examples.append({"inputs": _state_bits(states, 4), "outputs": output})

    return examples, 16, 4


def _build_critters(cfg: Dict[str, Any]) -> DatasetResult:
    examples: List[Example] = []
    for block in product([False, True], repeat=4):
        alive_count = sum(block)
        if alive_count == 2:
            output = list(block)
        elif alive_count == 3:
            output = [not block[3], not block[2], not block[1], not block[0]]
        else:
            output = [not bit for bit in block]
        examples.append({"inputs": list(block), "outputs": output})

    return examples, 4, 4


def _build_traffic(cfg: Dict[str, Any]) -> DatasetResult:
    # States: 0=empty, 1=right-moving car, 2=down-moving car.
    # Inputs: phase bit, previous cell state, center state, next cell state.
    # phase=False evaluates the right-moving sub-step; phase=True evaluates the down-moving sub-step.
    examples: List[Example] = []
    for phase in [False, True]:
        moving_state = 2 if phase else 1
        for prev_state, center_state, next_state in product(range(3), repeat=3):
            if center_state == moving_state and next_state == 0:
                output_state = 0
            elif center_state == 0 and prev_state == moving_state:
                output_state = moving_state
            else:
                output_state = center_state

            inputs: IOList = [phase]
            inputs.extend(_state_bits([prev_state, center_state, next_state], 2))
            examples.append({"inputs": inputs, "outputs": _bits(output_state, 2)})

    return examples, 7, 2


register_plugin("life", _build_life, {
    "instructions": 14,
})

register_plugin("life-compressed", _build_life_compressed, {
    "instructions": 14,
})

register_plugin("maze", _build_maze, {
    "instructions": 10,
})

register_plugin("maze-compressed", _build_maze_compressed, {
    "instructions": 6,
})

register_plugin("brian", _build_brian, {
    "instructions": 24,
    "brian": {
        "max_examples": 4096,
        "seed": 0,
    },
})

register_plugin("brian-compressed", _build_brian_compressed, {
    "instructions": 10,
})

register_plugin("fire", _build_fire, {
    "instructions": 18,
})

register_plugin("fire-compressed", _build_fire_compressed, {
    "instructions": 8,
})

register_plugin("wire", _build_wire, {
    "instructions": 24,
    "wire": {
        "max_examples": 4096,
        "seed": 0,
    },
})

register_plugin("wire-compressed", _build_wire_compressed, {
    "instructions": 10,
})

register_plugin("excitable", _build_excitable, {
    "instructions": 32,
    "excitable": {
        "states": 8,
        "max_examples": 4096,
        "seed": 0,
    },
})

register_plugin("excitable-compressed", _build_excitable_compressed, {
    "instructions": 12,
    "excitable-compressed": {
        "states": 8,
    },
})

register_plugin("cyclic", _build_cyclic, {
    "instructions": 48,
    "cyclic": {
        "states": 32,
        "max_examples": 4096,
        "seed": 0,
    },
})

register_plugin("cyclic-compressed", _build_cyclic_compressed, {
    "instructions": 18,
    "cyclic-compressed": {
        "states": 32,
    },
})

register_plugin("fluid", _build_fluid, {
    "instructions": 18,
    "fluid": {
        "max_examples": 4096,
        "seed": 0,
    },
})

register_plugin("critters", _build_critters, {
    "instructions": 10,
})

register_plugin("traffic", _build_traffic, {
    "instructions": 12,
})
