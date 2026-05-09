#include "postprocess.hpp"

#include "mask.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kAnd = 0;
constexpr int kXor = 1;
constexpr int kOr = 2;

struct ProgramScore {
    std::size_t length = 0;
    int max_output_depth = 0;
    int operator_cost = 0;
    std::vector<int> outputs;
    std::vector<int> instr_key;
};

bool operator<(const ProgramScore &left, const ProgramScore &right) {
    return std::tie(left.length, left.max_output_depth, left.operator_cost, left.outputs, left.instr_key) <
           std::tie(right.length, right.max_output_depth, right.operator_cost, right.outputs, right.instr_key);
}

std::string program_key(const Program &program) {
    std::string key;
    for (const auto &instr : program.instrs) {
        key += "T:" + std::to_string(instr.op) + "," + std::to_string(instr.s1) + "," + std::to_string(instr.s2) + ";";
    }
    key += "O:";
    for (int output : program.outputs) {
        key += std::to_string(output) + ",";
    }
    return key;
}

int temp_idx_for_source(int source, int num_inputs) {
    return source - num_inputs - 1;
}

int source_for_temp_idx(std::size_t idx, int num_inputs) {
    return num_inputs + 1 + static_cast<int>(idx);
}

bool is_temp_source(int source, int num_inputs, std::size_t instr_count) {
    const int idx = temp_idx_for_source(source, num_inputs);
    return idx >= 0 && static_cast<std::size_t>(idx) < instr_count;
}

void mark_reachable(int source, const Program &program, int num_inputs, std::vector<bool> &reachable) {
    if (!is_temp_source(source, num_inputs, program.instrs.size())) return;
    const std::size_t idx = static_cast<std::size_t>(temp_idx_for_source(source, num_inputs));
    if (reachable[idx]) return;
    reachable[idx] = true;
    const Instruction &instr = program.instrs[idx];
    mark_reachable(instr.s1, program, num_inputs, reachable);
    mark_reachable(instr.s2, program, num_inputs, reachable);
}

int remap_source(int source, int num_inputs, const std::vector<int> &temp_remap) {
    if (source <= num_inputs) return source;
    const int idx = temp_idx_for_source(source, num_inputs);
    if (idx < 0 || static_cast<std::size_t>(idx) >= temp_remap.size()) return source;
    return temp_remap[static_cast<std::size_t>(idx)];
}

Program prune_dead_nodes(const Program &program, int num_inputs) {
    std::vector<bool> reachable(program.instrs.size(), false);
    for (int output : program.outputs) {
        mark_reachable(output, program, num_inputs, reachable);
    }

    std::vector<int> temp_remap(program.instrs.size(), -1);
    Program result;
    result.outputs = program.outputs;
    for (std::size_t idx = 0; idx < program.instrs.size(); ++idx) {
        if (!reachable[idx]) continue;
        temp_remap[idx] = source_for_temp_idx(result.instrs.size(), num_inputs);
        result.instrs.push_back(program.instrs[idx]);
    }
    for (auto &instr : result.instrs) {
        instr.s1 = remap_source(instr.s1, num_inputs, temp_remap);
        instr.s2 = remap_source(instr.s2, num_inputs, temp_remap);
    }
    for (int &output : result.outputs) {
        output = remap_source(output, num_inputs, temp_remap);
    }
    return result;
}

ProgramScore score_program(const Program &program, int num_inputs) {
    ProgramScore score;
    score.length = program.instrs.size();
    score.outputs = program.outputs;
    std::vector<int> depths(static_cast<std::size_t>(num_inputs + 1), 0);
    score.instr_key.reserve(program.instrs.size() * 3);
    for (const auto &instr : program.instrs) {
        const int depth = std::max(depths[static_cast<std::size_t>(instr.s1)], depths[static_cast<std::size_t>(instr.s2)]) + 1;
        depths.push_back(depth);
        score.operator_cost += instr.op == kXor ? 2 : 1;
        score.instr_key.push_back(instr.op);
        score.instr_key.push_back(instr.s1);
        score.instr_key.push_back(instr.s2);
    }
    for (int output : program.outputs) {
        score.max_output_depth = std::max(score.max_output_depth, depths[static_cast<std::size_t>(output)]);
    }
    return score;
}

std::vector<PackedMask> evaluate_all_sources(
    const Program &program,
    std::span<const PackedMask> input_masks,
    const PackedMask &all_examples_mask
) {
    std::vector<PackedMask> values;
    values.push_back(all_examples_mask);
    values.insert(values.end(), input_masks.begin(), input_masks.end());
    for (const auto &instr : program.instrs) {
        values.push_back(apply_operator_mask(instr.op, values[static_cast<std::size_t>(instr.s1)], values[static_cast<std::size_t>(instr.s2)]) &
                         all_examples_mask);
    }
    return values;
}

int first_zero_source(std::span<const PackedMask> values) {
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (values[idx].is_zero()) return static_cast<int>(idx);
    }
    return -1;
}

bool source_is_operator(const Program &program, int source, int num_inputs, int op) {
    if (!is_temp_source(source, num_inputs, program.instrs.size())) return false;
    return program.instrs[static_cast<std::size_t>(temp_idx_for_source(source, num_inputs))].op == op;
}

bool op_source_contains(const Program &program, int source, int num_inputs, int needle) {
    if (!is_temp_source(source, num_inputs, program.instrs.size())) return false;
    const Instruction &instr = program.instrs[static_cast<std::size_t>(temp_idx_for_source(source, num_inputs))];
    return instr.s1 == needle || instr.s2 == needle;
}

int xor_cancellation_replacement(const Program &program, const Instruction &instr, int num_inputs) {
    auto check_nested = [&](int nested_source, int other_source) -> int {
        if (!source_is_operator(program, nested_source, num_inputs, kXor)) return -1;
        const Instruction &nested = program.instrs[static_cast<std::size_t>(temp_idx_for_source(nested_source, num_inputs))];
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
    const int self_source = source_for_temp_idx(instr_idx, num_inputs);
    (void)self_source;

    if (instr.op == kAnd) {
        if (instr.s1 == instr.s2) return instr.s1;
        if (instr.s1 == 0) return instr.s2;
        if (instr.s2 == 0) return instr.s1;
        if (false_source >= 0 && (instr.s1 == false_source || instr.s2 == false_source)) return false_source;
        if (source_is_operator(program, instr.s2, num_inputs, kOr) && op_source_contains(program, instr.s2, num_inputs, instr.s1)) {
            return instr.s1;
        }
        if (source_is_operator(program, instr.s1, num_inputs, kOr) && op_source_contains(program, instr.s1, num_inputs, instr.s2)) {
            return instr.s2;
        }
    } else if (instr.op == kOr) {
        if (instr.s1 == instr.s2) return instr.s1;
        if (false_source >= 0 && instr.s1 == false_source) return instr.s2;
        if (false_source >= 0 && instr.s2 == false_source) return instr.s1;
        if (instr.s1 == 0 || instr.s2 == 0) return 0;
        if (source_is_operator(program, instr.s2, num_inputs, kAnd) && op_source_contains(program, instr.s2, num_inputs, instr.s1)) {
            return instr.s1;
        }
        if (source_is_operator(program, instr.s1, num_inputs, kAnd) && op_source_contains(program, instr.s1, num_inputs, instr.s2)) {
            return instr.s2;
        }
    } else if (instr.op == kXor) {
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

void push_unique_candidate(
    std::vector<Program> &candidates,
    std::set<std::string> &seen,
    const Program &candidate,
    const std::string &base_key,
    std::span<const Example> examples,
    int num_inputs,
    int num_outputs
) {
    const std::string key = program_key(candidate);
    if (key == base_key || !seen.insert(key).second) return;
    if (!verify_program(candidate, examples, num_inputs, num_outputs).empty()) return;
    candidates.push_back(candidate);
}

std::vector<Program> generate_candidates(
    const Program &program,
    std::span<const Example> examples,
    int num_inputs,
    int num_outputs,
    const PackedExamples &packed,
    std::size_t max_candidates
) {
    const Program base = prune_dead_nodes(program, num_inputs);
    const std::string base_key = program_key(base);
    PackedMask all_mask = all_ones(packed.width);
    const std::vector<PackedMask> values = evaluate_all_sources(base, packed.input_masks, all_mask);
    const int false_source = first_zero_source(values);

    std::vector<Program> candidates;
    std::set<std::string> seen;
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
            add(redirect_source(base, source_for_temp_idx(instr_idx, num_inputs), replacement, num_inputs));
        }
    }

    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
        PackedMask care_mask = all_mask ^ packed.output_dont_care_masks[static_cast<std::size_t>(output_idx)];
        int upper = base.outputs[static_cast<std::size_t>(output_idx)];
        if (upper < 0 || static_cast<std::size_t>(upper) > values.size()) upper = static_cast<int>(values.size());
        for (int source = 0; source < upper; ++source) {
            if (!cares_match(values[static_cast<std::size_t>(source)], packed.output_values[static_cast<std::size_t>(output_idx)], care_mask)) {
                continue;
            }
            Program candidate = base;
            candidate.outputs[static_cast<std::size_t>(output_idx)] = source;
            add(prune_dead_nodes(candidate, num_inputs));
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&](const Program &left, const Program &right) {
        return score_program(left, num_inputs) < score_program(right, num_inputs);
    });
    return candidates;
}

}  // namespace

Program post_process_program(
    const Program &program,
    std::span<const Example> examples,
    int num_inputs,
    int num_outputs,
    const PostProcessOptions &options
) {
    if (!options.enabled || examples.empty()) return program;

    PackedExamples packed = pack_examples(examples, num_inputs, num_outputs);
    Program best = prune_dead_nodes(program, num_inputs);
    if (!verify_program(best, examples, num_inputs, num_outputs).empty()) return program;

    ProgramScore best_score = score_program(best, num_inputs);
    std::vector<Program> beam{best};
    std::set<std::string> globally_seen{program_key(best)};
    const std::size_t max_rounds = options.beam_rounds == 0 ? std::numeric_limits<std::size_t>::max() : options.beam_rounds;

    for (std::size_t round = 0; round < max_rounds; ++round) {
        std::vector<Program> next;
        for (const Program &state : beam) {
            std::vector<Program> candidates =
                generate_candidates(state, examples, num_inputs, num_outputs, packed, options.beam_candidates);
            for (const Program &candidate : candidates) {
                if (globally_seen.insert(program_key(candidate)).second) {
                    next.push_back(candidate);
                }
            }
        }
        if (next.empty()) break;
        std::sort(next.begin(), next.end(), [&](const Program &left, const Program &right) {
            return score_program(left, num_inputs) < score_program(right, num_inputs);
        });
        if (next.size() > options.beam_width) next.resize(options.beam_width);

        bool improved = false;
        for (const Program &candidate : next) {
            const ProgramScore score = score_program(candidate, num_inputs);
            if (score < best_score) {
                best = candidate;
                best_score = score;
                improved = true;
            }
        }
        beam = std::move(next);
        if (options.beam_rounds == 0 && !improved) break;
    }

    return best;
}
