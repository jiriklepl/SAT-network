#include "postprocess_internal.hpp"

#include "datasets.hpp"
#include "mask.hpp"
#include "program.hpp"

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
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

Program materialize_program_dag(const Program &program, int num_inputs) {
    Program result;
    std::map<int, Instruction> nodes;
    for (std::size_t instr_idx = 0; instr_idx < program.instrs.size(); ++instr_idx) {
        nodes[temp_source(static_cast<int>(instr_idx), num_inputs)] = program.instrs[instr_idx];
    }
    std::map<int, int> remap;
    std::set<int> visiting;

    auto visit = [&](auto &self, int source) -> int {
        if (source <= num_inputs) return source;
        if (const auto it = remap.find(source); it != remap.end()) return it->second;
        const auto node = nodes.find(source);
        if (node == nodes.end()) throw std::runtime_error("candidate DAG references missing source");
        if (visiting.contains(source)) throw std::runtime_error("cycle in candidate DAG");
        visiting.insert(source);
        const int s1 = self(self, node->second.s1);
        const int s2 = self(self, node->second.s2);
        visiting.erase(source);
        const int new_source = temp_source(static_cast<int>(result.instrs.size()), num_inputs);
        remap[source] = new_source;
        result.instrs.push_back({node->second.op, s1, s2});
        return new_source;
    };

    result.outputs.reserve(program.outputs.size());
    for (const int output : program.outputs) {
        result.outputs.push_back(visit(visit, output));
    }
    return result;
}

std::vector<std::set<int>> build_instruction_users(const Program &program, int num_inputs) {
    std::vector<std::set<int>> users(num_inputs + 1 + program.instrs.size());
    for (std::size_t instr_idx = 0; instr_idx < program.instrs.size(); ++instr_idx) {
        const int source = temp_source(static_cast<int>(instr_idx), num_inputs);
        const Instruction &instr = program.instrs[instr_idx];
        if (instr.s1 >= 0 && static_cast<std::size_t>(instr.s1) < users.size()) users[instr.s1].insert(source);
        if (instr.s2 >= 0 && static_cast<std::size_t>(instr.s2) < users.size()) users[instr.s2].insert(source);
    }
    return users;
}

std::set<int> output_sources(const Program &program) {
    return {program.outputs.begin(), program.outputs.end()};
}

std::pair<int, int> ordered_pair(int left, int right) {
    return left <= right ? std::make_pair(left, right) : std::make_pair(right, left);
}

bool candidate_source_depends_on_forbidden_source(const Program &program, int num_inputs, int candidate_source,
                                                  int forbidden_source) {
    if (candidate_source == forbidden_source) return true;
    if (!is_temp_source(candidate_source, num_inputs, static_cast<int>(program.instrs.size()))) return false;
    const Instruction &instr =
        program.instrs[static_cast<std::size_t>(temp_index_from_source(candidate_source, num_inputs))];
    return candidate_source_depends_on_forbidden_source(program, num_inputs, instr.s1, forbidden_source) ||
           candidate_source_depends_on_forbidden_source(program, num_inputs, instr.s2, forbidden_source);
}

bool replacement_preserves_outputs(const Program &program, int num_inputs, const PackedExamples &packed,
                                   std::span<const PackedMask> values, const PackedMask &all_mask, int modified_source,
                                   const PackedMask &replacement_mask) {
    std::vector<std::optional<PackedMask>> memo(values.size());
    std::set<int> visiting;
    memo[static_cast<std::size_t>(modified_source)] = replacement_mask;

    auto value = [&](auto &self, int source) -> PackedMask {
        if (source < 0 || static_cast<std::size_t>(source) >= values.size()) {
            throw std::logic_error("replacement source out of range");
        }
        if (const auto &cached = memo[static_cast<std::size_t>(source)]; cached.has_value()) return *cached;
        if (!is_temp_source(source, num_inputs, static_cast<int>(program.instrs.size())) ||
            !candidate_source_depends_on_forbidden_source(program, num_inputs, source, modified_source)) {
            return values[static_cast<std::size_t>(source)];
        }
        if (!visiting.insert(source).second) {
            throw std::logic_error("cycle while checking replacement output preservation");
        }
        const Instruction &instr = program.instrs[static_cast<std::size_t>(temp_index_from_source(source, num_inputs))];
        PackedMask computed = apply_operator_mask(instr.op, self(self, instr.s1), self(self, instr.s2)) & all_mask;
        visiting.erase(source);
        memo[static_cast<std::size_t>(source)] = computed;
        return computed;
    };

    for (std::size_t output_idx = 0; output_idx < program.outputs.size(); ++output_idx) {
        const PackedMask care_mask = all_mask ^ packed.output_dont_care_masks[output_idx];
        if (!cares_match(value(value, program.outputs[output_idx]), packed.output_values[output_idx], care_mask)) {
            return false;
        }
    }
    return true;
}

std::uint64_t stable_seed(unsigned base_seed, const std::string &key, std::size_t salt) {
    std::uint64_t hash = static_cast<std::uint64_t>(base_seed) ^ (salt + 0x9e3779b97f4a7c15ULL);
    for (const char ch : key) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

std::vector<Program> generate_mask_simplification_candidates(
    const Program &base, std::span<const Example> examples, int num_inputs, int num_outputs,
    const PackedExamples &packed, std::span<const PackedMask> values, const std::string &base_key,
    std::set<std::string> &seen, std::size_t max_candidates, const PostProcessOptions &options, ProfileData *profile) {
    const PackedMask all_mask = all_ones(packed.width);
    const int false_source = first_zero_source(values);
    std::vector<Program> candidates;

    PostProcessGeneratorRun mask_generator(profile, PostProcessGeneratorKind::Mask, options.generator_timeout_seconds);
    const std::size_t mask_start = candidates.size();
    auto mask_at_limit = [&] { return max_candidates != 0 && candidates.size() - mask_start >= max_candidates; };
    auto mask_done = [&] { return mask_at_limit() || mask_generator.timed_out(); };
    auto add_mask = [&](const Program &candidate) {
        if (mask_done()) return false;
        mask_generator.candidate_considered();
        mask_generator.candidate_materialized();
        return push_unique_candidate(candidates, seen, candidate, base_key, examples, num_inputs, num_outputs,
                                     mask_generator);
    };

    for (std::size_t source = 1; source < values.size() && !mask_done(); ++source) {
        for (std::size_t prior = 0; prior < source && !mask_done(); ++prior) {
            if (values[source] == values[prior]) {
                add_mask(redirect_source(base, static_cast<int>(source), static_cast<int>(prior), num_inputs));
            }
        }
    }

    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size() && !mask_done(); ++instr_idx) {
        const int replacement = algebraic_replacement(base, instr_idx, num_inputs, false_source);
        if (replacement >= 0) {
            add_mask(
                redirect_source(base, temp_source(static_cast<int>(instr_idx), num_inputs), replacement, num_inputs));
        }
    }

    for (int output_idx = 0; output_idx < num_outputs && !mask_done(); ++output_idx) {
        const PackedMask care_mask = all_mask ^ packed.output_dont_care_masks[static_cast<std::size_t>(output_idx)];
        int upper = base.outputs[static_cast<std::size_t>(output_idx)];
        if (upper < 0 || static_cast<std::size_t>(upper) > values.size()) upper = static_cast<int>(values.size());
        for (int source = 0; source < upper && !mask_done(); ++source) {
            if (!cares_match(values[static_cast<std::size_t>(source)],
                             packed.output_values[static_cast<std::size_t>(output_idx)], care_mask)) {
                continue;
            }
            Program candidate = base;
            candidate.outputs[static_cast<std::size_t>(output_idx)] = source;
            add_mask(prune_dead_nodes(candidate, num_inputs));
        }
    }

    const std::vector<std::set<int>> users = build_instruction_users(base, num_inputs);
    const std::set<int> outputs = output_sources(base);

    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size() && !mask_done(); ++instr_idx) {
        const int current_source = temp_source(static_cast<int>(instr_idx), num_inputs);
        const Instruction &node = base.instrs[instr_idx];

        for (int replacement_source = 0; replacement_source < current_source && !mask_done(); ++replacement_source) {
            if (values[static_cast<std::size_t>(replacement_source)] !=
                values[static_cast<std::size_t>(current_source)]) {
                continue;
            }
            const Program &candidate = base;
            if (candidate.instrs[instr_idx].s1 == current_source || candidate.instrs[instr_idx].s2 == current_source) {
                continue;
            }
            add_mask(redirect_source(candidate, current_source, replacement_source, num_inputs));
        }

        if (users[static_cast<std::size_t>(current_source)].size() == 1 && !outputs.contains(current_source)) {
            const int single_user_source = *users[static_cast<std::size_t>(current_source)].begin();
            for (int replacement_source = 0; replacement_source < single_user_source && !mask_done();
                 ++replacement_source) {
                if (replacement_source == single_user_source) continue;
                if (values[static_cast<std::size_t>(replacement_source)] !=
                    values[static_cast<std::size_t>(single_user_source)]) {
                    continue;
                }
                if (candidate_source_depends_on_forbidden_source(base, num_inputs, replacement_source,
                                                                 single_user_source)) {
                    continue;
                }
                add_mask(redirect_source(base, single_user_source, replacement_source, num_inputs));
            }
        }

        for (int operand_idx = 0; operand_idx < 2 && !mask_done(); ++operand_idx) {
            const int old_operand_source = operand_idx == 0 ? node.s1 : node.s2;
            for (int replacement_source = 0; replacement_source < old_operand_source && !mask_done();
                 ++replacement_source) {
                if (values[static_cast<std::size_t>(replacement_source)] !=
                    values[static_cast<std::size_t>(old_operand_source)]) {
                    continue;
                }
                if (candidate_source_depends_on_forbidden_source(base, num_inputs, replacement_source,
                                                                 current_source)) {
                    continue;
                }
                Program candidate = base;
                Instruction &candidate_node = candidate.instrs[instr_idx];
                if (operand_idx == 0) {
                    candidate_node.s1 = replacement_source;
                } else {
                    candidate_node.s2 = replacement_source;
                }
                add_mask(prune_dead_nodes(candidate, num_inputs));
            }
        }

        if (node.op == kOpXor && source_is_operator(base, node.s1, num_inputs, kOpXor) &&
            source_is_operator(base, node.s2, num_inputs, kOpXor)) {
            const Instruction &left =
                base.instrs[static_cast<std::size_t>(temp_index_from_source(node.s1, num_inputs))];
            const Instruction &right =
                base.instrs[static_cast<std::size_t>(temp_index_from_source(node.s2, num_inputs))];
            for (const int common : {left.s1, left.s2}) {
                if (common != right.s1 && common != right.s2) continue;
                const int left_remaining = left.s1 == common ? left.s2 : left.s1;
                const int right_remaining = right.s1 == common ? right.s2 : right.s1;
                const auto [new_s1, new_s2] = ordered_pair(left_remaining, right_remaining);
                Program candidate = base;
                candidate.instrs[instr_idx].s1 = new_s1;
                candidate.instrs[instr_idx].s2 = new_s2;
                add_mask(prune_dead_nodes(candidate, num_inputs));
            }
        }
    }

    std::map<std::tuple<int, int, int>, int> existing_nodes;
    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size(); ++instr_idx) {
        const Instruction &instr = base.instrs[instr_idx];
        const auto [s1, s2] = ordered_pair(instr.s1, instr.s2);
        existing_nodes.try_emplace({instr.op, s1, s2}, temp_source(static_cast<int>(instr_idx), num_inputs));
    }

    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size() && !mask_done(); ++instr_idx) {
        const int source = temp_source(static_cast<int>(instr_idx), num_inputs);
        const Instruction &node = base.instrs[instr_idx];
        for (const auto [child_source, remaining_source] :
             {std::make_pair(node.s1, node.s2), std::make_pair(node.s2, node.s1)}) {
            if (!source_is_operator(base, child_source, num_inputs, node.op)) continue;
            const Instruction &child =
                base.instrs[static_cast<std::size_t>(temp_index_from_source(child_source, num_inputs))];
            const std::array<int, 3> sources{child.s1, child.s2, remaining_source};
            for (int inner_a_pos = 0; inner_a_pos < 3 && !mask_done(); ++inner_a_pos) {
                for (int inner_b_pos = 0; inner_b_pos < 3 && !mask_done(); ++inner_b_pos) {
                    if (inner_a_pos == inner_b_pos) continue;
                    const int outer_pos = 3 - inner_a_pos - inner_b_pos;
                    const auto [inner_s1, inner_s2] = ordered_pair(sources[static_cast<std::size_t>(inner_a_pos)],
                                                                   sources[static_cast<std::size_t>(inner_b_pos)]);
                    const int outer_source = sources[static_cast<std::size_t>(outer_pos)];
                    if (candidate_source_depends_on_forbidden_source(base, num_inputs, inner_s1, child_source) ||
                        candidate_source_depends_on_forbidden_source(base, num_inputs, inner_s2, child_source) ||
                        candidate_source_depends_on_forbidden_source(base, num_inputs, outer_source, source)) {
                        continue;
                    }
                    Program adjusted = base;
                    adjusted.instrs[static_cast<std::size_t>(temp_index_from_source(child_source, num_inputs))].s1 =
                        inner_s1;
                    adjusted.instrs[static_cast<std::size_t>(temp_index_from_source(child_source, num_inputs))].s2 =
                        inner_s2;
                    const auto [new_s1, new_s2] = ordered_pair(child_source, outer_source);
                    adjusted.instrs[instr_idx].s1 = new_s1;
                    adjusted.instrs[instr_idx].s2 = new_s2;
                    add_mask(materialize_program_dag(adjusted, num_inputs));

                    const auto existing = existing_nodes.find({node.op, inner_s1, inner_s2});
                    if (existing != existing_nodes.end() && existing->second != child_source &&
                        !candidate_source_depends_on_forbidden_source(base, num_inputs, existing->second, source) &&
                        !candidate_source_depends_on_forbidden_source(base, num_inputs, outer_source, source)) {
                        Program regrouped = base;
                        const auto [regroup_s1, regroup_s2] = ordered_pair(existing->second, outer_source);
                        if (ordered_pair(regroup_s1, regroup_s2) == ordered_pair(node.s1, node.s2)) continue;
                        regrouped.instrs[instr_idx].s1 = regroup_s1;
                        regrouped.instrs[instr_idx].s2 = regroup_s2;
                        add_mask(materialize_program_dag(regrouped, num_inputs));
                    }
                }
            }
        }
    }

    mask_generator.finish();

    PostProcessGeneratorRun replacement_generator(profile, PostProcessGeneratorKind::Replacement,
                                                  options.generator_timeout_seconds);
    const std::size_t replacement_start = candidates.size();
    auto replacement_at_limit = [&] {
        return max_candidates != 0 && candidates.size() - replacement_start >= max_candidates;
    };
    auto replacement_done = [&] { return replacement_at_limit() || replacement_generator.timed_out(); };

    for (std::size_t instr_idx = 0; instr_idx < base.instrs.size() && !replacement_done(); ++instr_idx) {
        const int target_source = temp_source(static_cast<int>(instr_idx), num_inputs);
        const int total_sources = temp_source(static_cast<int>(base.instrs.size()), num_inputs);
        std::vector<std::tuple<int, int, int>> replacements;
        for (const int op : {kOpXor, kOpAnd, kOpOr}) {
            if (replacement_done()) break;
            for (int s1 = 0; s1 < total_sources && !replacement_done(); ++s1) {
                if (s1 == target_source ||
                    candidate_source_depends_on_forbidden_source(base, num_inputs, s1, target_source)) {
                    continue;
                }
                for (int s2 = 0; s2 < total_sources && !replacement_done(); ++s2) {
                    if (s2 == target_source ||
                        candidate_source_depends_on_forbidden_source(base, num_inputs, s2, target_source)) {
                        continue;
                    }
                    if (op != kOpXor && s2 < s1) continue;
                    replacements.emplace_back(op, s1, s2);
                }
            }
        }
        std::mt19937_64 rng(stable_seed(options.random_seed, base_key, instr_idx));
        std::shuffle(replacements.begin(), replacements.end(), rng);

        std::size_t valid_replacements_for_node = 0;
        for (const auto &[op, s1, s2] : replacements) {
            if (replacement_done()) break;
            if (options.replace_patience > 0 && valid_replacements_for_node >= options.replace_patience) break;
            replacement_generator.candidate_considered();
            const PackedMask candidate_mask =
                apply_operator_mask(op, values[static_cast<std::size_t>(s1)], values[static_cast<std::size_t>(s2)]) &
                all_mask;
            if (!replacement_preserves_outputs(base, num_inputs, packed, values, all_mask, target_source,
                                               candidate_mask)) {
                continue;
            }
            if (candidate_mask == values[static_cast<std::size_t>(target_source)] && op == base.instrs[instr_idx].op &&
                s1 == base.instrs[instr_idx].s1 && s2 == base.instrs[instr_idx].s2) {
                ++valid_replacements_for_node;
                continue;
            }
            Program candidate = base;
            candidate.instrs[instr_idx] = Instruction{op, s1, s2};
            replacement_generator.candidate_materialized();
            push_unique_candidate(candidates, seen, materialize_program_dag(candidate, num_inputs), base_key, examples,
                                  num_inputs, num_outputs, replacement_generator);
            ++valid_replacements_for_node;
        }
    }

    replacement_generator.finish();
    return candidates;
}
