#include "postprocess_internal.hpp"

#include "datasets.hpp"
#include "mask.hpp"
#include "program.hpp"

#include <cstddef>

#include <set>
#include <span>
#include <string>
#include <vector>

namespace {

int first_zero_source(std::span<const PackedMask> values) {
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (values[idx].is_zero()) return static_cast<int>(idx);
    }
    return -1;
}

bool source_is_operator(const Program &program, int source, int num_inputs, int op) {
    if (!is_temp_source(source, num_inputs, static_cast<int>(program.instrs.size()))) return false;
    return program.instrs[static_cast<std::size_t>(temp_index_from_source(source, num_inputs))].op == op;
}

bool op_source_contains(const Program &program, int source, int num_inputs, int needle) {
    if (!is_temp_source(source, num_inputs, static_cast<int>(program.instrs.size()))) return false;
    const Instruction &instr = program.instrs[static_cast<std::size_t>(temp_index_from_source(source, num_inputs))];
    return instr.s1 == needle || instr.s2 == needle;
}

int xor_cancellation_replacement(const Program &program, const Instruction &instr, int num_inputs) {
    auto check_nested = [&](int nested_source, int other_source) -> int {
        if (!source_is_operator(program, nested_source, num_inputs, kOpXor)) return -1;
        const Instruction &nested =
            program.instrs[static_cast<std::size_t>(temp_index_from_source(nested_source, num_inputs))];
        if (nested.s1 == other_source) return nested.s2;
        if (nested.s2 == other_source) return nested.s1;
        return -1;
    };
    const int left = check_nested(instr.s1, instr.s2);
    if (left >= 0) return left;
    return check_nested(instr.s2, instr.s1);
}

int algebraic_replacement(const Program &program, std::size_t instr_idx, int num_inputs, int false_source) {
    const Instruction &instr = program.instrs[instr_idx];
    if (instr.op == kOpAnd) {
        if (instr.s1 == instr.s2) return instr.s1;
        if (instr.s1 == kSourceConstantOne) return instr.s2;
        if (instr.s2 == kSourceConstantOne) return instr.s1;
        if (false_source >= 0 && (instr.s1 == false_source || instr.s2 == false_source)) return false_source;
        if (source_is_operator(program, instr.s2, num_inputs, kOpOr) &&
            op_source_contains(program, instr.s2, num_inputs, instr.s1)) {
            return instr.s1;
        }
        if (source_is_operator(program, instr.s1, num_inputs, kOpOr) &&
            op_source_contains(program, instr.s1, num_inputs, instr.s2)) {
            return instr.s2;
        }
    } else if (instr.op == kOpOr) {
        if (instr.s1 == instr.s2) return instr.s1;
        if (false_source >= 0 && instr.s1 == false_source) return instr.s2;
        if (false_source >= 0 && instr.s2 == false_source) return instr.s1;
        if (instr.s1 == kSourceConstantOne || instr.s2 == kSourceConstantOne) return kSourceConstantOne;
        if (source_is_operator(program, instr.s2, num_inputs, kOpAnd) &&
            op_source_contains(program, instr.s2, num_inputs, instr.s1)) {
            return instr.s1;
        }
        if (source_is_operator(program, instr.s1, num_inputs, kOpAnd) &&
            op_source_contains(program, instr.s1, num_inputs, instr.s2)) {
            return instr.s2;
        }
    } else if (instr.op == kOpXor) {
        if (instr.s1 == instr.s2) return false_source;
        if (false_source >= 0 && instr.s1 == false_source) return instr.s2;
        if (false_source >= 0 && instr.s2 == false_source) return instr.s1;
        return xor_cancellation_replacement(program, instr, num_inputs);
    }
    return -1;
}

Program redirect_source(const Program &program, int old_source, int new_source, int num_inputs) {
    Program candidate = program;
    for (auto &instr : candidate.instrs) {
        if (instr.s1 == old_source) instr.s1 = new_source;
        if (instr.s2 == old_source) instr.s2 = new_source;
    }
    for (int &output : candidate.outputs) {
        if (output == old_source) output = new_source;
    }
    return prune_dead_nodes(candidate, num_inputs);
}

bool cares_match(const PackedMask &left, const PackedMask &right, const PackedMask &care_mask) {
    return ((left ^ right) & care_mask).is_zero();
}

}  // namespace

std::vector<Program> generate_mask_simplification_candidates(const Program &base, std::span<const Example> examples,
                                                             int num_inputs, int num_outputs,
                                                             const PackedExamples &packed,
                                                             std::span<const PackedMask> values,
                                                             const std::string &base_key, std::set<std::string> &seen,
                                                             std::size_t max_candidates) {
    const PackedMask all_mask = all_ones(packed.width);
    const int false_source = first_zero_source(values);
    std::vector<Program> candidates;

    auto add = [&](const Program &candidate) {
        if (max_candidates != 0 && candidates.size() >= max_candidates) return;
        push_unique_candidate(candidates, seen, candidate, base_key, examples, num_inputs, num_outputs);
    };

    for (std::size_t source = 1; source < values.size(); ++source) {
        for (std::size_t prior = 0; prior < source; ++prior) {
            if (values[source] == values[prior]) {
                add(redirect_source(base, static_cast<int>(source), static_cast<int>(prior), num_inputs));
            }
        }
    }

    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size(); ++instr_idx) {
        const int replacement = algebraic_replacement(base, instr_idx, num_inputs, false_source);
        if (replacement >= 0) {
            add(redirect_source(base, temp_source(static_cast<int>(instr_idx), num_inputs), replacement, num_inputs));
        }
    }

    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
        const PackedMask care_mask = all_mask ^ packed.output_dont_care_masks[static_cast<std::size_t>(output_idx)];
        int upper = base.outputs[static_cast<std::size_t>(output_idx)];
        if (upper < 0 || static_cast<std::size_t>(upper) > values.size()) upper = static_cast<int>(values.size());
        for (int source = 0; source < upper; ++source) {
            if (!cares_match(values[static_cast<std::size_t>(source)],
                             packed.output_values[static_cast<std::size_t>(output_idx)], care_mask)) {
                continue;
            }
            Program candidate = base;
            candidate.outputs[static_cast<std::size_t>(output_idx)] = source;
            add(prune_dead_nodes(candidate, num_inputs));
        }
    }

    return candidates;
}
