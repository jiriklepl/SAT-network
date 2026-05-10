#include "postprocess.hpp"

#include "datasets.hpp"
#include "encoding.hpp"
#include "mask.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <z3++.h>

#include <chrono>
#include <cstddef>

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stack>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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
        if (instr.s1 == 0) return instr.s2;
        if (instr.s2 == 0) return instr.s1;
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

Program materialize_dag(const std::map<int, Instruction> &nodes, std::span<const int> outputs, int num_inputs) {
    Program result;
    std::map<int, int> remap;
    std::set<int> visiting;
    std::set<int> visited;

    auto visit = [&](auto &self, int source) -> int {
        if (source <= num_inputs) return source;
        if (const auto it = remap.find(source); it != remap.end()) return it->second;
        const auto node = nodes.find(source);
        if (node == nodes.end()) throw std::runtime_error("materialized DAG references missing source");
        if (visiting.contains(source)) throw std::runtime_error("cycle in materialized DAG");
        if (visited.contains(source)) return remap.at(source);

        visiting.insert(source);
        const int s1 = self(self, node->second.s1);
        const int s2 = self(self, node->second.s2);
        visiting.erase(source);
        visited.insert(source);

        const int new_source = temp_source(static_cast<int>(result.instrs.size()), num_inputs);
        remap[source] = new_source;
        result.instrs.push_back({node->second.op, s1, s2});
        return new_source;
    };

    result.outputs.reserve(outputs.size());
    for (const int output : outputs) {
        result.outputs.push_back(visit(visit, output));
    }
    return result;
}

bool cares_match(const PackedMask &left, const PackedMask &right, const PackedMask &care_mask) {
    return ((left ^ right) & care_mask).is_zero();
}

std::optional<int> remaining_timeout_ms(Clock::time_point start, double timeout_seconds) {
    if (timeout_seconds <= 0.0) return std::nullopt;
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    const double remaining = timeout_seconds - elapsed;
    if (remaining <= 0.0) return 0;
    return std::max(1, static_cast<int>(remaining * 1000.0));
}

bool generator_timed_out(Clock::time_point start, double timeout_seconds) {
    if (timeout_seconds <= 0.0) return false;
    return std::chrono::duration<double>(Clock::now() - start).count() >= timeout_seconds;
}

bool push_unique_candidate(std::vector<Program> &candidates, std::set<std::string> &seen, const Program &candidate,
                           const std::string &base_key, std::span<const Example> examples, int num_inputs,
                           int num_outputs, ProfileData *profile = nullptr) {
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

std::vector<std::set<int>> build_users(const Program &program, int num_inputs) {
    std::vector<std::set<int>> users(num_inputs + 1 + program.instrs.size());
    for (std::size_t instr_idx = 0; instr_idx < program.instrs.size(); ++instr_idx) {
        const int source = temp_source(static_cast<int>(instr_idx), num_inputs);
        const Instruction &instr = program.instrs[instr_idx];
        if (instr.s1 >= 0 && static_cast<std::size_t>(instr.s1) < users.size()) users[instr.s1].insert(source);
        if (instr.s2 >= 0 && static_cast<std::size_t>(instr.s2) < users.size()) users[instr.s2].insert(source);
    }
    return users;
}

std::map<int, int> build_output_uses(const Program &program) {
    std::map<int, int> output_uses;
    for (const int output : program.outputs) {
        ++output_uses[output];
    }
    return output_uses;
}

bool outputs_limited_to_selected(int source, const std::set<int> &selected, const std::vector<std::set<int>> &users) {
    if (source < 0 || static_cast<std::size_t>(source) >= users.size()) return true;
    return std::all_of(users[static_cast<std::size_t>(source)].begin(), users[static_cast<std::size_t>(source)].end(),
                       [&](int user) { return selected.contains(user); });
}

std::vector<int> component_outputs(const std::set<int> &nodes, const std::vector<std::set<int>> &users,
                                   const std::map<int, int> &output_uses) {
    std::vector<int> outputs;
    for (const int node : nodes) {
        const bool used_by_output = output_uses.contains(node);
        const bool used_outside =
            static_cast<std::size_t>(node) < users.size() &&
            std::any_of(
                users[static_cast<std::size_t>(node)].begin(), users[static_cast<std::size_t>(node)].end(),
                [&](int user) { return !nodes.contains(user); });
        if (used_by_output || used_outside) outputs.push_back(node);
    }
    return outputs;
}

std::set<int> closed_dependency_closure(int root, const Program &program, int num_inputs,
                                        const std::vector<std::set<int>> &users) {
    std::set<int> closure;
    std::set<int> selected{root};
    std::set<int> pending;
    const Instruction &root_instr = program.instrs[static_cast<std::size_t>(temp_index_from_source(root, num_inputs))];
    if (is_temp_source(root_instr.s1, num_inputs, static_cast<int>(program.instrs.size())))
        pending.insert(root_instr.s1);
    if (is_temp_source(root_instr.s2, num_inputs, static_cast<int>(program.instrs.size())))
        pending.insert(root_instr.s2);

    while (!pending.empty()) {
        bool added = false;
        for (auto it = pending.begin(); it != pending.end();) {
            const int source = *it;
            if (!outputs_limited_to_selected(source, selected, users)) {
                ++it;
                continue;
            }
            it = pending.erase(it);
            selected.insert(source);
            closure.insert(source);
            const Instruction &instr =
                program.instrs[static_cast<std::size_t>(temp_index_from_source(source, num_inputs))];
            for (const int dep : {instr.s1, instr.s2}) {
                if (is_temp_source(dep, num_inputs, static_cast<int>(program.instrs.size())) &&
                    !selected.contains(dep)) {
                    pending.insert(dep);
                }
            }
            added = true;
        }
        if (!added) break;
    }
    return closure;
}

std::vector<std::vector<int>> closed_rooted_subcomponents(int root, const std::set<int> &closure,
                                                          const Program &program, int num_inputs,
                                                          const std::vector<std::set<int>> &users, std::size_t size) {
    std::set<std::vector<int>> result;
    std::vector<std::set<int>> stack{{root}};
    std::set<std::set<int>> seen;
    while (!stack.empty()) {
        const std::set<int> selected = std::move(stack.back());
        stack.pop_back();
        if (!seen.insert(selected).second) continue;
        if (selected.size() == size) {
            result.insert(std::vector<int>(selected.begin(), selected.end()));
            continue;
        }
        if (selected.size() > size) continue;

        std::set<int> candidates;
        for (const int selected_node : selected) {
            const Instruction &instr =
                program.instrs[static_cast<std::size_t>(temp_index_from_source(selected_node, num_inputs))];
            for (const int dep : {instr.s1, instr.s2}) {
                if (closure.contains(dep) && !selected.contains(dep)) candidates.insert(dep);
            }
        }
        for (const int candidate : candidates) {
            if (!outputs_limited_to_selected(candidate, selected, users)) continue;
            std::set<int> next = selected;
            next.insert(candidate);
            stack.push_back(std::move(next));
        }
    }
    return {result.begin(), result.end()};
}

Program apply_resynthesized_window(const Program &base, int num_inputs, std::span<const int> window_outputs,
                                   std::span<const int> external_sources, const Program &local_program) {
    std::map<int, Instruction> nodes;
    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size(); ++instr_idx) {
        nodes[temp_source(static_cast<int>(instr_idx), num_inputs)] = base.instrs[instr_idx];
    }

    const std::size_t original_size = base.instrs.size();
    std::vector<int> fresh_sources;
    fresh_sources.reserve(local_program.instrs.size());
    for (std::size_t local_idx = 0; local_idx < local_program.instrs.size(); ++local_idx) {
        fresh_sources.push_back(temp_source(static_cast<int>(original_size + local_idx), num_inputs));
    }

    auto translate_local_source = [&](int source) -> int {
        if (source == kSourceConstantOne) return kSourceConstantOne;
        if (source >= 1 && source <= static_cast<int>(external_sources.size())) {
            return external_sources[static_cast<std::size_t>(source - 1)];
        }
        const int local_instr_idx = source - static_cast<int>(external_sources.size()) - 1;
        if (local_instr_idx < 0 || static_cast<std::size_t>(local_instr_idx) >= fresh_sources.size()) {
            throw std::runtime_error("local resynthesis source out of range");
        }
        return fresh_sources[static_cast<std::size_t>(local_instr_idx)];
    };

    for (std::size_t local_idx = 0; local_idx < local_program.instrs.size(); ++local_idx) {
        const Instruction &local_instr = local_program.instrs[local_idx];
        nodes[fresh_sources[local_idx]] =
            Instruction{local_instr.op, translate_local_source(local_instr.s1), translate_local_source(local_instr.s2)};
    }

    std::map<int, int> replacements;
    for (std::size_t out_idx = 0; out_idx < window_outputs.size(); ++out_idx) {
        replacements[window_outputs[out_idx]] = translate_local_source(local_program.outputs[out_idx]);
    }
    auto replace = [&](int source) {
        const auto it = replacements.find(source);
        return it == replacements.end() ? source : it->second;
    };

    for (std::size_t instr_idx = 0; instr_idx < original_size; ++instr_idx) {
        Instruction &instr = nodes[temp_source(static_cast<int>(instr_idx), num_inputs)];
        instr.s1 = replace(instr.s1);
        instr.s2 = replace(instr.s2);
    }

    std::vector<int> rewritten_outputs = base.outputs;
    for (int &output : rewritten_outputs)
        output = replace(output);
    return materialize_dag(nodes, rewritten_outputs, num_inputs);
}

std::vector<Program> generate_resynthesis_candidates(const Program &base, std::span<const Example> examples,
                                                     int num_inputs, int num_outputs, const PackedExamples &packed,
                                                     const std::vector<PackedMask> &values,
                                                     const PostProcessOptions &options, const std::string &base_key,
                                                     std::set<std::string> &seen, std::size_t max_candidates,
                                                     ProfileData *profile) {
    std::vector<Program> candidates;
    if (base.instrs.size() < 2 || packed.width == 0 || options.resynthesis_maxnodes < 2) return candidates;

    const std::vector<std::set<int>> users = build_users(base, num_inputs);
    const std::map<int, int> output_uses = build_output_uses(base);
    std::set<std::pair<std::vector<int>, std::vector<int>>> seen_windows;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> windows;

    auto add_window = [&](const std::vector<int> &nodes) {
        if (nodes.size() < 2) return;
        const std::set<int> node_set(nodes.begin(), nodes.end());
        std::vector<int> outputs = component_outputs(node_set, users, output_uses);
        if (outputs.empty()) return;
        std::ranges::sort(outputs);
        auto key = std::make_pair(nodes, outputs);
        if (seen_windows.insert(key).second) windows.push_back(std::move(key));
    };

    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size(); ++instr_idx) {
        const int root = temp_source(static_cast<int>(instr_idx), num_inputs);
        std::set<int> full_set = closed_dependency_closure(root, base, num_inputs, users);
        full_set.insert(root);
        if (full_set.size() <= options.resynthesis_maxnodes) {
            add_window(std::vector<int>(full_set.begin(), full_set.end()));
        } else {
            for (const std::vector<int> &nodes :
                 closed_rooted_subcomponents(root, full_set, base, num_inputs, users, options.resynthesis_maxnodes)) {
                add_window(nodes);
            }
        }
    }

    const Clock::time_point generator_start = Clock::now();
    std::size_t sat_results = 0;
    for (std::size_t window_idx = 0; window_idx < windows.size(); ++window_idx) {
        if (max_candidates != 0 && candidates.size() >= max_candidates) break;
        if (options.resynthesis_patience > 0 && sat_results >= options.resynthesis_patience) break;
        if (generator_timed_out(generator_start, options.generator_timeout_seconds)) {
            if (profile != nullptr) ++profile->post_processing_resynthesis_timeout_exits;
            break;
        }
        if (profile != nullptr) ++profile->post_processing_resynthesis_windows_considered;

        const std::vector<int> &window_nodes = windows[window_idx].first;
        const std::vector<int> &window_outputs = windows[window_idx].second;
        const std::set<int> window_set(window_nodes.begin(), window_nodes.end());
        std::set<int> external_set;
        for (const int node : window_nodes) {
            const Instruction &instr = base.instrs[static_cast<std::size_t>(temp_index_from_source(node, num_inputs))];
            for (const int source : {instr.s1, instr.s2}) {
                if (!window_set.contains(source)) external_set.insert(source);
            }
        }
        if (external_set.empty()) continue;

        std::vector<int> external_sources(external_set.begin(), external_set.end());
        const ProgramSpec local_spec{.num_inputs = static_cast<int>(external_sources.size()),
                                     .num_outputs = static_cast<int>(window_outputs.size()),
                                     .program_length = static_cast<int>(window_nodes.size() - 1)};
        z3::context ctx;
        z3::solver solver = make_solver(ctx, "simple-tactic");
        add_exprs(solver, build_program(ctx, local_spec, EncodingOptions{}));
        std::vector<PackedMask> local_inputs;
        local_inputs.reserve(external_sources.size());
        for (const int source : external_sources)
            local_inputs.push_back(values[static_cast<std::size_t>(source)]);
        PackedTestEncoding test =
            build_packed_test(ctx, local_inputs, "resynth" + std::to_string(window_idx), local_spec, EncodingOptions{});
        add_exprs(solver, test.constraints);
        for (std::size_t out_idx = 0; out_idx < window_outputs.size(); ++out_idx) {
            solver.add(test.outputs[out_idx] ==
                       packed_mask_value(ctx, values[static_cast<std::size_t>(window_outputs[out_idx])]));
        }
        const std::optional<int> timeout_ms = remaining_timeout_ms(generator_start, options.generator_timeout_seconds);
        if (timeout_ms.has_value()) {
            if (*timeout_ms <= 0) {
                if (profile != nullptr) ++profile->post_processing_resynthesis_timeout_exits;
                break;
            }
            solver.set("timeout", static_cast<unsigned>(*timeout_ms));
        }
        if (solver.check() != z3::sat) continue;
        if (profile != nullptr) ++profile->post_processing_resynthesis_windows_sat;
        ++sat_results;
        const Program local_program = extract_program(ctx, solver.get_model(), local_spec);
        const Program candidate =
            apply_resynthesized_window(base, num_inputs, window_outputs, external_sources, local_program);
        if (profile != nullptr) ++profile->post_processing_resynthesis_candidates_materialized;
        if (push_unique_candidate(candidates, seen, candidate, base_key, examples, num_inputs, num_outputs, profile) &&
            profile != nullptr) {
            ++profile->post_processing_resynthesis_candidates_accepted;
        }
    }
    return candidates;
}

std::vector<Program> generate_candidates(const Program &program, std::span<const Example> examples, int num_inputs,
                                         int num_outputs, const PackedExamples &packed,
                                         const PostProcessOptions &options, ProfileData *profile) {
    const Program base = prune_dead_nodes(program, num_inputs);
    const std::string base_key = program_key(base);
    const PackedMask all_mask = all_ones(packed.width);
    const std::vector<PackedMask> values = evaluate_all_sources(base, packed.input_masks, all_mask);
    const int false_source = first_zero_source(values);
    const std::size_t max_candidates = options.beam_candidates;

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

    if (max_candidates == 0 || candidates.size() < max_candidates) {
        std::vector<Program> resynth_candidates = generate_resynthesis_candidates(
            base, examples, num_inputs, num_outputs, packed, values, options, base_key, seen,
            max_candidates == 0 ? 0 : max_candidates - candidates.size(), profile);
        candidates.insert(candidates.end(), resynth_candidates.begin(), resynth_candidates.end());
    }

    std::ranges::sort(candidates, [&](const Program &left, const Program &right) {
        return score_program(left, num_inputs) < score_program(right, num_inputs);
    });
    return candidates;
}

}  // namespace

Program post_process_program(const Program &program, std::span<const Example> examples, int num_inputs, int num_outputs,
                             const PostProcessOptions &options, ProfileData *profile) {
    if (!options.enabled || examples.empty()) return program;
    require_valid_program(program, num_inputs, num_outputs, "post-process input program");

    const PackedExamples packed = pack_examples(examples, num_inputs, num_outputs);
    Program best = prune_dead_nodes(program, num_inputs);
    require_valid_program(best, num_inputs, num_outputs, "post-process pruned input program");
    if (!verify_program(best, examples, num_inputs, num_outputs).empty()) return program;

    ProgramScore best_score = score_program(best, num_inputs);
    std::vector<Program> beam{best};
    std::set<std::string> globally_seen{program_key(best)};
    const std::size_t max_rounds =
        options.beam_rounds == 0 ? std::numeric_limits<std::size_t>::max() : options.beam_rounds;

    for (std::size_t round = 0; round < max_rounds; ++round) {
        std::vector<Program> next;
        for (const Program &state : beam) {
            const std::vector<Program> candidates =
                generate_candidates(state, examples, num_inputs, num_outputs, packed, options, profile);
            for (const Program &candidate : candidates) {
                if (globally_seen.insert(program_key(candidate)).second) {
                    next.push_back(candidate);
                }
            }
        }
        if (next.empty()) break;
        std::ranges::sort(next, [&](const Program &left, const Program &right) {
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
