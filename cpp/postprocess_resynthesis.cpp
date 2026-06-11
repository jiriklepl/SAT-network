#include "postprocess_internal.hpp"

#include "datasets.hpp"
#include "encoding.hpp"
#include "mask.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <z3++.h>

#include <cstddef>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

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
    if (is_temp_source(root_instr.s1, num_inputs, static_cast<int>(program.instrs.size()))) {
        pending.insert(root_instr.s1);
    }
    if (is_temp_source(root_instr.s2, num_inputs, static_cast<int>(program.instrs.size()))) {
        pending.insert(root_instr.s2);
    }

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

Program materialize_dag(const std::map<int, Instruction> &nodes, std::span<const int> outputs, int num_inputs) {
    Program result;
    std::map<int, int> remap;
    std::set<int> visiting;
    std::set<int> visited;

    auto visit = [&](this auto &self, int source) -> int {
        if (source <= num_inputs) return source;
        if (const auto it = remap.find(source); it != remap.end()) return it->second;
        const auto node = nodes.find(source);
        if (node == nodes.end()) throw std::runtime_error("materialized DAG references missing source");
        if (visiting.contains(source)) throw std::runtime_error("cycle in materialized DAG");
        if (visited.contains(source)) return remap.at(source);

        visiting.insert(source);
        const int s1 = self(node->second.s1);
        const int s2 = self(node->second.s2);
        visiting.erase(source);
        visited.insert(source);

        const int new_source = temp_source(static_cast<int>(result.instrs.size()), num_inputs);
        remap[source] = new_source;
        result.instrs.push_back({node->second.op, s1, s2});
        return new_source;
    };

    result.outputs.reserve(outputs.size());
    for (const int output : outputs) {
        result.outputs.push_back(visit(output));
    }
    return result;
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
    for (int &output : rewritten_outputs) {
        output = replace(output);
    }
    return materialize_dag(nodes, rewritten_outputs, num_inputs);
}

}  // namespace

std::vector<Program> generate_resynthesis_candidates(const Program &base, std::span<const Example> examples,
                                                     int num_inputs, int num_outputs, const PackedExamples &packed,
                                                     std::span<const PackedMask> values,
                                                     const PostProcessOptions &options, const std::string &base_key,
                                                     std::set<std::string> &seen, std::size_t max_candidates,
                                                     ProfileData *profile) {
    std::vector<Program> candidates;
    PostProcessGeneratorRun generator(profile, PostProcessGeneratorKind::Resynthesis,
                                      options.generator_timeout_seconds);
    if (base.instrs.size() < 2 || packed.width == 0 || options.resynthesis_maxnodes < 2) {
        generator.finish();
        return candidates;
    }

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

    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size() && !generator.timed_out(); ++instr_idx) {
        const int root = temp_source(static_cast<int>(instr_idx), num_inputs);
        std::set<int> full_set = closed_dependency_closure(root, base, num_inputs, users);
        full_set.insert(root);
        if (full_set.size() <= options.resynthesis_maxnodes) {
            add_window(std::vector<int>(full_set.begin(), full_set.end()));
        } else {
            for (const std::vector<int> &nodes :
                 closed_rooted_subcomponents(root, full_set, base, num_inputs, users, options.resynthesis_maxnodes)) {
                if (generator.timed_out()) break;
                add_window(nodes);
            }
        }
    }

    std::size_t sat_results = 0;
    for (std::size_t window_idx = 0; window_idx < windows.size(); ++window_idx) {
        if (max_candidates != 0 && candidates.size() >= max_candidates) break;
        if (options.resynthesis_patience > 0 && sat_results >= options.resynthesis_patience) break;
        if (generator.timed_out()) break;
        generator.candidate_considered();

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
        for (const int source : external_sources) {
            local_inputs.push_back(values[static_cast<std::size_t>(source)]);
        }
        PackedTestEncoding test =
            build_packed_test(ctx, local_inputs, "resynth" + std::to_string(window_idx), local_spec, EncodingOptions{});
        add_exprs(solver, test.constraints);
        for (std::size_t out_idx = 0; out_idx < window_outputs.size(); ++out_idx) {
            solver.add(test.outputs[out_idx] ==
                       packed_mask_value(ctx, values[static_cast<std::size_t>(window_outputs[out_idx])]));
        }
        const std::optional<int> timeout_ms = generator.remaining_timeout_ms();
        if (timeout_ms.has_value()) {
            if (*timeout_ms <= 0) break;
            solver.set("timeout", static_cast<unsigned>(*timeout_ms));
        }
        if (solver.check() != z3::sat) continue;
        if (profile != nullptr) ++profile->post_processing_resynthesis_windows_sat;
        ++sat_results;
        const Program local_program = extract_program(ctx, solver.get_model(), local_spec);
        const Program candidate =
            apply_resynthesized_window(base, num_inputs, window_outputs, external_sources, local_program);
        generator.candidate_materialized();
        push_unique_candidate(candidates, seen, candidate, base_key, examples, num_inputs, num_outputs, generator);
    }
    generator.finish();
    return candidates;
}
