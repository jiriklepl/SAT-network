#include "postprocess_internal.hpp"

#include "datasets.hpp"
#include "mask.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <cstddef>

#include <algorithm>
#include <optional>
#include <set>
#include <span>
#include <stack>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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
    for (const int output : program.outputs) {
        key += std::to_string(output) + ",";
    }
    return key;
}

std::optional<std::string> validate_program_invariants(const Program &program, int num_inputs, int num_outputs) {
    if (num_inputs < 0) return "negative input count";
    if (num_outputs < 0) return "negative output count";
    if (program.outputs.size() != static_cast<std::size_t>(num_outputs)) return "wrong output count";
    for (std::size_t instr_idx = 0; instr_idx < program.instrs.size(); ++instr_idx) {
        const Instruction &instr = program.instrs[instr_idx];
        const int max_source = temp_source(static_cast<int>(instr_idx), num_inputs) - 1;
        if (!known_op(instr.op)) return "unknown operator";
        if (instr.s1 < kSourceConstantOne || instr.s1 > max_source) return "instruction source out of SSA range";
        if (instr.s2 < kSourceConstantOne || instr.s2 > max_source) return "instruction source out of SSA range";
    }
    const int max_output_source = temp_source(static_cast<int>(program.instrs.size()), num_inputs) - 1;
    for (const int output : program.outputs) {
        if (output < kSourceConstantOne || output > max_output_source) return "output source out of range";
    }
    return std::nullopt;
}

void require_valid_program(const Program &program, int num_inputs, int num_outputs, const std::string &context) {
    if (const std::optional<std::string> error = validate_program_invariants(program, num_inputs, num_outputs)) {
        throw std::logic_error(context + ": " + *error);
    }
}

namespace {

void mark_reachable(int source, const Program &program, int num_inputs, std::vector<bool> &reachable) {
    std::stack<int> to_visit;
    to_visit.push(source);
    while (!to_visit.empty()) {
        const int current = to_visit.top();
        to_visit.pop();
        if (!is_temp_source(current, num_inputs, static_cast<int>(program.instrs.size()))) continue;
        const auto idx = static_cast<std::size_t>(temp_index_from_source(current, num_inputs));
        if (reachable[idx]) continue;
        reachable[idx] = true;
        const Instruction &instr = program.instrs[idx];
        to_visit.push(instr.s2);
        to_visit.push(instr.s1);
    }
}

int remap_source(int source, int num_inputs, const std::vector<int> &temp_remap) {
    if (source <= num_inputs) return source;
    const int idx = temp_index_from_source(source, num_inputs);
    if (idx < 0 || static_cast<std::size_t>(idx) >= temp_remap.size()) return source;
    return temp_remap[static_cast<std::size_t>(idx)];
}

}  // namespace

Program prune_dead_nodes(const Program &program, int num_inputs) {
    std::vector<bool> reachable(program.instrs.size(), false);
    for (const int output : program.outputs) {
        mark_reachable(output, program, num_inputs, reachable);
    }

    std::vector<int> temp_remap(program.instrs.size(), -1);
    Program result;
    result.outputs = program.outputs;
    for (std::size_t idx = 0; idx < program.instrs.size(); ++idx) {
        if (!reachable[idx]) continue;
        temp_remap[idx] = temp_source(static_cast<int>(result.instrs.size()), num_inputs);
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
        const int depth =
            std::max(depths[static_cast<std::size_t>(instr.s1)], depths[static_cast<std::size_t>(instr.s2)]) + 1;
        depths.push_back(depth);
        score.operator_cost += instr.op == kOpXor ? kXorOperatorCost : kDefaultOperatorCost;
        score.instr_key.push_back(instr.op);
        score.instr_key.push_back(instr.s1);
        score.instr_key.push_back(instr.s2);
    }
    for (const int output : program.outputs) {
        score.max_output_depth = std::max(score.max_output_depth, depths[static_cast<std::size_t>(output)]);
    }
    return score;
}

std::vector<PackedMask> evaluate_all_sources(const Program &program, std::span<const PackedMask> input_masks,
                                             const PackedMask &all_examples_mask) {
    std::vector<PackedMask> values;
    values.push_back(all_examples_mask);
    values.insert(values.end(), input_masks.begin(), input_masks.end());
    for (const auto &instr : program.instrs) {
        values.push_back(apply_operator_mask(instr.op, values[static_cast<std::size_t>(instr.s1)],
                                             values[static_cast<std::size_t>(instr.s2)]) &
                         all_examples_mask);
    }
    return values;
}

bool push_unique_candidate(std::vector<Program> &candidates, std::set<std::string> &seen, const Program &candidate,
                           const std::string &base_key, std::span<const Example> examples, int num_inputs,
                           int num_outputs, ProfileData *profile) {
    const std::string key = program_key(candidate);
    if (key == base_key || !seen.insert(key).second) return false;
    if (const std::optional<std::string> error = validate_program_invariants(candidate, num_inputs, num_outputs)) {
        if (profile != nullptr) ++profile->post_processing_resynthesis_invalid_candidates;
        throw std::logic_error("post-process generated an invalid candidate program: " + *error);
    }
    if (!verify_program(candidate, examples, num_inputs, num_outputs).empty()) return false;
    candidates.push_back(candidate);
    return true;
}
