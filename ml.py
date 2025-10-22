#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "ml.py requires PyTorch. Install it (e.g. `pip install torch`) and rerun."
    ) from exc

from dataset_plugins import available_plugins, get_plugin
import dataset_plugins.gol  # ensure registration
import dataset_plugins.adder3  # ensure registration


Example = Dict[str, List[bool]]
OPS: Tuple[str, ...] = ("OR", "AND", "XOR")


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_examples(
    data: List[Dict[str, Any]], num_inputs: int, num_outputs: int
) -> List[Example]:
    result: List[Example] = []
    for entry in data:
        ins = [bool(v) for v in entry["inputs"]]
        outs = [bool(v) for v in entry["outputs"]]
        if len(ins) != num_inputs or len(outs) != num_outputs:
            raise ValueError("Example length does not match num_inputs/num_outputs")
        result.append({"inputs": ins, "outputs": outs})
    return result


def _build_dataset_from_config(cfg: Dict[str, Any]) -> Tuple[List[Example], int, int, int]:
    ctype = cfg.get("type")

    if "examples" in cfg:
        num_inputs = int(cfg["num_inputs"])
        num_outputs = int(cfg["num_outputs"])
        examples = _collect_examples(cfg["examples"], num_inputs, num_outputs)
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
        raise ValueError("instructions must be provided in the config")
    instructions = int(instructions)
    if instructions <= 0:
        raise ValueError("instructions must be positive for the ML formulation")

    return examples, num_inputs, num_outputs, instructions


def _examples_to_tensors(
    examples: List[Example], num_inputs: int, num_outputs: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.zeros((len(examples), num_inputs), dtype=torch.float32, device=device)
    outputs = torch.zeros(
        (len(examples), num_outputs), dtype=torch.float32, device=device
    )
    for idx, example in enumerate(examples):
        ins = example["inputs"]
        outs = example["outputs"]
        inputs[idx] = torch.tensor(
            [1.0 if bool(v) else 0.0 for v in ins], dtype=torch.float32, device=device
        )
        outputs[idx] = torch.tensor(
            [1.0 if bool(v) else 0.0 for v in outs], dtype=torch.float32, device=device
        )
    return inputs, outputs


class SoftProgramModel(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        program_length: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.program_length = program_length
        self.ops = OPS
        self.num_ops = len(self.ops)
        self.hidden_dim = hidden_dim
        self.feature_dim = program_length + num_inputs + num_outputs

        self._pair_specs: List[List[Tuple[int, int]]] = []
        self.instruction_heads = nn.ModuleList()

        for instr_idx in range(program_length):
            num_sources = num_inputs + 1 + instr_idx  # inputs + const1 + previous nodes
            pairs = list(combinations(range(num_sources), 2))
            if not pairs:
                raise ValueError("Each instruction must have at least one source pair")

            self._pair_specs.append(pairs)
            output_dim = self.num_ops * len(pairs)
            self.instruction_heads.append(
                nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
            )

        if num_outputs > 1:
            self.output_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.feature_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, program_length),
                    )
                    for _ in range(num_outputs)
                ]
            )
        else:
            self.output_heads = nn.ModuleList()

    def _build_node_features(
        self,
        instruction_outputs: List[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if instruction_outputs:
            computed = torch.cat(instruction_outputs, dim=1)
            if len(instruction_outputs) < self.program_length:
                pad = torch.zeros(
                    batch_size,
                    self.program_length - len(instruction_outputs),
                    dtype=dtype,
                    device=device,
                )
                return torch.cat([computed, pad], dim=1)
            return computed
        return torch.zeros(
            batch_size, self.program_length, dtype=dtype, device=device
        )

    def _forward_internal(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor | None,
        collect_details: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]] | None]:
        batch_size = batch_inputs.size(0)
        dtype = batch_inputs.dtype
        device = batch_inputs.device

        if batch_targets is None:
            batch_targets = torch.zeros(
                batch_size, self.num_outputs, dtype=dtype, device=device
            )

        values: List[torch.Tensor] = [batch_inputs]
        const_col = torch.ones((batch_size, 1), dtype=dtype, device=device)
        values.append(const_col)

        instruction_outputs: List[torch.Tensor] = []
        instr_probs: List[torch.Tensor] = [] if collect_details else []

        for instr_idx, head in enumerate(self.instruction_heads):
            sources = torch.cat(values, dim=1)
            candidates: List[torch.Tensor] = []
            for src_a, src_b in self._pair_specs[instr_idx]:
                left = sources[:, src_a : src_a + 1]
                right = sources[:, src_b : src_b + 1]

                or_val = 1.0 - (1.0 - left) * (1.0 - right)
                and_val = left * right
                xor_val = torch.abs(left - right)

                candidates.extend([or_val, and_val, xor_val])

            candidate_tensor = torch.cat(candidates, dim=1)  # [B, num_choices]
            node_features = self._build_node_features(
                instruction_outputs, batch_size, dtype, device
            )
            features = torch.cat([node_features, batch_inputs, batch_targets], dim=1)
            logits = head(features)
            probs = torch.softmax(logits, dim=1)

            node_val = torch.sum(candidate_tensor * probs, dim=1, keepdim=True)

            instruction_outputs.append(node_val)
            values.append(node_val)

            if collect_details:
                instr_probs.append(probs.detach())

        if not instruction_outputs:
            raise RuntimeError("Program length must be positive")

        instr_tensor = torch.cat(instruction_outputs, dim=1)  # [B, program_length]

        if self.num_outputs == 1:
            outputs = instruction_outputs[-1]
            output_probs: List[torch.Tensor] = []
        else:
            node_features = self._build_node_features(
                instruction_outputs, batch_size, dtype, device
            )
            features = torch.cat([node_features, batch_inputs, batch_targets], dim=1)
            collected_outputs: List[torch.Tensor] = []
            output_probs = []
            for head in self.output_heads:
                logits = head(features)
                probs = torch.softmax(logits, dim=1)
                if collect_details:
                    output_probs.append(probs.detach())
                weighted = torch.sum(instr_tensor * probs, dim=1, keepdim=True)
                collected_outputs.append(weighted)
            outputs = torch.cat(collected_outputs, dim=1)

        details = None
        if collect_details:
            details = (instr_probs, output_probs)
        return outputs, details

    def forward(
        self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs, _ = self._forward_internal(
            batch_inputs, batch_targets, collect_details=False
        )
        return outputs

    def collect_distributions(
        self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor | None = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        _, details = self._forward_internal(
            batch_inputs, batch_targets, collect_details=True
        )
        if details is None:
            return [], []
        return details


def train_model(
    model: SoftProgramModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    logger: logging.Logger,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_examples = inputs.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_examples, device=inputs.device)
        epoch_loss = 0.0
        for start in range(0, num_examples, batch_size):
            idx = perm[start : start + batch_size]
            batch_in = inputs[idx]
            batch_out = targets[idx]

            preds = model(batch_in, batch_out).clamp(min=1e-6, max=1.0 - 1e-6)
            loss = F.binary_cross_entropy(preds, batch_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_in.size(0)

        epoch_loss /= num_examples

        with torch.no_grad():
            model.eval()
            preds_full = model(inputs, targets)
            accuracy = ((preds_full >= 0.5) == (targets >= 0.5)).float().mean().item()

        logger.info(
            "Epoch %d/%d - loss %.6f - accuracy %.4f",
            epoch + 1,
            epochs,
            epoch_loss,
            accuracy,
        )

        if accuracy >= 1.0 - 1e-6:
            logger.info("Reached 100%% accuracy at epoch %d. Stopping early.", epoch + 1)
            break


def decode_program(
    model: SoftProgramModel, inputs: torch.Tensor, targets: torch.Tensor
) -> Tuple[List[Tuple[str, int, int]], List[int]]:
    model.eval()
    with torch.no_grad():
        instr_probs, output_probs = model.collect_distributions(inputs, targets)

    instructions: List[Tuple[str, int, int]] = []
    for idx, probs in enumerate(instr_probs):
        if probs.numel() == 0:
            raise RuntimeError("Instruction probability tensor is empty")
        mean_probs = probs.mean(dim=0)
        best = int(torch.argmax(mean_probs).item())
        op_idx = best % model.num_ops
        pair_idx = best // model.num_ops
        src_a, src_b = model._pair_specs[idx][pair_idx]
        instructions.append((model.ops[op_idx], src_a, src_b))

    if model.num_outputs == 1:
        outputs = [model.program_length - 1]
    else:
        outputs = []
        for probs in output_probs:
            mean_probs = probs.mean(dim=0)
            outputs.append(int(torch.argmax(mean_probs).item()))

    return instructions, outputs


def _fmt_source(src_idx: int, num_inputs: int) -> str:
    if src_idx < num_inputs:
        return f"I{src_idx}"
    if src_idx == num_inputs:
        return "1"
    return f"T{src_idx - num_inputs - 1}"


def evaluate_discrete_program(
    instructions: List[Tuple[str, int, int]],
    output_selection: List[int],
    examples: List[Example],
    num_inputs: int,
    num_outputs: int,
) -> Tuple[int, int]:
    mismatches = 0
    for example in examples:
        ins = [1 if bool(v) else 0 for v in example["inputs"]]
        outs = [1 if bool(v) else 0 for v in example["outputs"]]

        values: List[int] = list(ins)
        values.append(1)

        for op_label, src_a, src_b in instructions:
            left = values[src_a]
            right = values[src_b]
            if op_label == "OR":
                val = left | right
            elif op_label == "AND":
                val = left & right
            elif op_label == "XOR":
                val = left ^ right
            else:
                raise ValueError(f"Unsupported operation {op_label}")
            values.append(val)

        actual: List[int] = []
        if num_outputs == 1:
            if instructions:
                sel_idx = output_selection[0] if output_selection else len(instructions) - 1
                sel_idx = max(0, min(sel_idx, len(instructions) - 1))
                actual.append(values[num_inputs + 1 + sel_idx])
            else:
                actual.append(0)
        else:
            for sel_idx in output_selection:
                if 0 <= sel_idx < len(instructions):
                    actual.append(values[num_inputs + 1 + sel_idx])
                else:
                    actual.append(0)

        if actual != outs:
            mismatches += 1

    return mismatches, len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a straight-line program using differentiable learning"
    )
    parser.add_argument(
        "--dataset",
        choices=list(available_plugins().keys()),
        default="gol",
        help="Choose a built-in dataset config",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a custom JSON config"
    )
    parser.add_argument(
        "--instructions",
        type=int,
        default=None,
        help="Override number of SSA instructions",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-2, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger("ml")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    device = torch.device(args.device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    config_path = Path(__file__).parent / "configs"
    if args.config:
        config_path = Path(args.config)
    elif args.dataset:
        config_path = config_path / f"{args.dataset}.json"
    else:
        raise SystemExit("Either --dataset or --config must be specified")

    cfg = _load_config(config_path)
    examples, num_inputs, num_outputs, cfg_instructions = _build_dataset_from_config(cfg)
    if args.instructions is not None:
        instructions = int(args.instructions)
    else:
        instructions = cfg_instructions

    if instructions <= 0:
        raise SystemExit("Program length must be positive")

    logger.info(
        "Loaded %d examples (%d inputs, %d outputs) with %d instructions",
        len(examples),
        num_inputs,
        num_outputs,
        instructions,
    )

    inputs_tensor, outputs_tensor = _examples_to_tensors(
        examples, num_inputs, num_outputs, device
    )

    model = SoftProgramModel(num_inputs, num_outputs, instructions).to(device)

    start_time = time.time()
    train_model(
        model,
        inputs_tensor,
        outputs_tensor,
        batch_size=max(1, min(args.batch_size, len(examples))),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        logger=logger,
    )
    elapsed = time.time() - start_time
    logger.info("Training completed in %.2f seconds", elapsed)

    instr_specs, output_selection = decode_program(
        model, inputs_tensor, outputs_tensor
    )

    logger.info("Decoded program:")
    for idx, (op_label, src_a, src_b) in enumerate(instr_specs):
        logger.info(
            "  T%d: %s(%s, %s)",
            idx,
            op_label,
            _fmt_source(src_a, num_inputs),
            _fmt_source(src_b, num_inputs),
        )

    if num_outputs == 1:
        logger.info("  OUT0: %s", f"T{output_selection[0]}")
    else:
        for out_idx, sel in enumerate(output_selection):
            logger.info("  OUT%d: T%d", out_idx, sel)

    mismatches, total = evaluate_discrete_program(
        instr_specs, output_selection, examples, num_inputs, num_outputs
    )
    if mismatches == 0:
        logger.info("All %d examples matched by the decoded program", total)
    else:
        logger.warning("%d/%d examples mismatched", mismatches, total)


if __name__ == "__main__":
    main()
