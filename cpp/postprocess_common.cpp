#include "postprocess_internal.hpp"

#include "datasets.hpp"
#include "mask.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <bit>
#include <chrono>
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

bool operator<(const ProgramScore &left, const ProgramScore &right) {
    return std::tie(left.parts, left.outputs, left.instr_key) < std::tie(right.parts, right.outputs, right.instr_key);
}

namespace {

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void add_generator_seconds(ProfileData &profile, PostProcessGeneratorKind kind, double seconds) {
    switch (kind) {
    case PostProcessGeneratorKind::Mask:
        profile.post_processing_mask_generator_seconds += seconds;
        break;
    case PostProcessGeneratorKind::Replacement:
        profile.post_processing_replacement_generator_seconds += seconds;
        break;
    case PostProcessGeneratorKind::Resynthesis:
        profile.post_processing_resynthesis_generator_seconds += seconds;
        break;
    }
}

void increment_timeout_exits(ProfileData &profile, PostProcessGeneratorKind kind) {
    switch (kind) {
    case PostProcessGeneratorKind::Mask:
        ++profile.post_processing_mask_timeout_exits;
        break;
    case PostProcessGeneratorKind::Replacement:
        ++profile.post_processing_replacement_timeout_exits;
        break;
    case PostProcessGeneratorKind::Resynthesis:
        ++profile.post_processing_resynthesis_timeout_exits;
        break;
    }
}

}  // namespace

PostProcessGeneratorRun::PostProcessGeneratorRun(ProfileData *profile, PostProcessGeneratorKind kind,
                                                 double timeout_seconds)
    : profile_(profile), kind_(kind), timeout_seconds_(timeout_seconds), start_(std::chrono::steady_clock::now()) {}

bool PostProcessGeneratorRun::timed_out() {
    if (timeout_seconds_ <= 0.0) return false;
    const bool expired = elapsed_seconds(start_) >= timeout_seconds_;
    if (expired && profile_ != nullptr && !timeout_recorded_) {
        increment_timeout_exits(*profile_, kind_);
        timeout_recorded_ = true;
    }
    return expired;
}

std::optional<int> PostProcessGeneratorRun::remaining_timeout_ms() {
    if (timeout_seconds_ <= 0.0) return std::nullopt;
    const double remaining = timeout_seconds_ - elapsed_seconds(start_);
    if (remaining <= 0.0) {
        if (profile_ != nullptr && !timeout_recorded_) {
            increment_timeout_exits(*profile_, kind_);
            timeout_recorded_ = true;
        }
        return 0;
    }
    return std::max(1, static_cast<int>(remaining * 1000.0));
}

void PostProcessGeneratorRun::finish() {
    if (profile_ == nullptr || finished_) return;
    add_generator_seconds(*profile_, kind_, elapsed_seconds(start_));
    finished_ = true;
}

void PostProcessGeneratorRun::candidate_considered() {
    if (profile_ == nullptr) return;
    switch (kind_) {
    case PostProcessGeneratorKind::Mask:
        ++profile_->post_processing_mask_candidates_considered;
        break;
    case PostProcessGeneratorKind::Replacement:
        ++profile_->post_processing_replacement_candidates_considered;
        break;
    case PostProcessGeneratorKind::Resynthesis:
        ++profile_->post_processing_resynthesis_candidates_considered;
        ++profile_->post_processing_resynthesis_windows_considered;
        break;
    }
}

void PostProcessGeneratorRun::candidate_materialized() {
    if (profile_ == nullptr) return;
    switch (kind_) {
    case PostProcessGeneratorKind::Mask:
        ++profile_->post_processing_mask_candidates_materialized;
        break;
    case PostProcessGeneratorKind::Replacement:
        ++profile_->post_processing_replacement_candidates_materialized;
        break;
    case PostProcessGeneratorKind::Resynthesis:
        ++profile_->post_processing_resynthesis_candidates_materialized;
        break;
    }
}

void PostProcessGeneratorRun::candidate_accepted() {
    if (profile_ == nullptr) return;
    switch (kind_) {
    case PostProcessGeneratorKind::Mask:
        ++profile_->post_processing_mask_candidates_accepted;
        break;
    case PostProcessGeneratorKind::Replacement:
        ++profile_->post_processing_replacement_candidates_accepted;
        break;
    case PostProcessGeneratorKind::Resynthesis:
        ++profile_->post_processing_resynthesis_candidates_accepted;
        break;
    }
}

void PostProcessGeneratorRun::invalid_candidate() {
    if (profile_ == nullptr) return;
    switch (kind_) {
    case PostProcessGeneratorKind::Mask:
        ++profile_->post_processing_mask_invalid_candidates;
        break;
    case PostProcessGeneratorKind::Replacement:
        ++profile_->post_processing_replacement_invalid_candidates;
        break;
    case PostProcessGeneratorKind::Resynthesis:
        ++profile_->post_processing_resynthesis_invalid_candidates;
        break;
    }
}

PostProcessScoreMetric post_process_score_metric_by_name(const std::string &name) {
    if (name == "program-length") return PostProcessScoreMetric::ProgramLength;
    if (name == "output-depth") return PostProcessScoreMetric::OutputDepth;
    if (name == "max-output-depth") return PostProcessScoreMetric::MaxOutputDepth;
    if (name == "sum-output-depth") return PostProcessScoreMetric::SumOutputDepth;
    if (name == "total-node-depth") return PostProcessScoreMetric::TotalNodeDepth;
    if (name == "total-tree-size") return PostProcessScoreMetric::TotalTreeSize;
    if (name == "operator-cost") return PostProcessScoreMetric::OperatorCost;
    if (name == "xor-count") return PostProcessScoreMetric::XorCount;
    if (name == "output-cone-size") return PostProcessScoreMetric::OutputConeSize;
    if (name == "max-output-cone-size") return PostProcessScoreMetric::MaxOutputConeSize;
    if (name == "sum-output-cone-size") return PostProcessScoreMetric::SumOutputConeSize;
    if (name == "fanout") return PostProcessScoreMetric::Fanout;
    if (name == "max-fanout") return PostProcessScoreMetric::MaxFanout;
    if (name == "sum-fanout") return PostProcessScoreMetric::SumFanout;
    if (name == "one-fanout-count") return PostProcessScoreMetric::OneFanoutCount;
    if (name == "independent-pairs") return PostProcessScoreMetric::IndependentPairs;
    if (name == "entropy") return PostProcessScoreMetric::Entropy;
    if (name == "random") return PostProcessScoreMetric::Random;
    throw std::runtime_error("unsupported post-process score metric");
}

std::string post_process_score_metric_name(PostProcessScoreMetric metric) {
    switch (metric) {
    case PostProcessScoreMetric::ProgramLength:
        return "program-length";
    case PostProcessScoreMetric::OutputDepth:
        return "output-depth";
    case PostProcessScoreMetric::MaxOutputDepth:
        return "max-output-depth";
    case PostProcessScoreMetric::SumOutputDepth:
        return "sum-output-depth";
    case PostProcessScoreMetric::TotalNodeDepth:
        return "total-node-depth";
    case PostProcessScoreMetric::TotalTreeSize:
        return "total-tree-size";
    case PostProcessScoreMetric::OperatorCost:
        return "operator-cost";
    case PostProcessScoreMetric::XorCount:
        return "xor-count";
    case PostProcessScoreMetric::OutputConeSize:
        return "output-cone-size";
    case PostProcessScoreMetric::MaxOutputConeSize:
        return "max-output-cone-size";
    case PostProcessScoreMetric::SumOutputConeSize:
        return "sum-output-cone-size";
    case PostProcessScoreMetric::Fanout:
        return "fanout";
    case PostProcessScoreMetric::MaxFanout:
        return "max-fanout";
    case PostProcessScoreMetric::SumFanout:
        return "sum-fanout";
    case PostProcessScoreMetric::OneFanoutCount:
        return "one-fanout-count";
    case PostProcessScoreMetric::IndependentPairs:
        return "independent-pairs";
    case PostProcessScoreMetric::Entropy:
        return "entropy";
    case PostProcessScoreMetric::Random:
        return "random";
    }
    throw std::runtime_error("unsupported post-process score metric");
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

std::vector<int> instruction_depths(const Program &program, int num_inputs) {
    std::vector<int> depths(static_cast<std::size_t>(num_inputs + 1), 0);
    for (const auto &instr : program.instrs) {
        depths.push_back(
            std::max(depths[static_cast<std::size_t>(instr.s1)], depths[static_cast<std::size_t>(instr.s2)]) + 1);
    }
    return depths;
}

std::vector<int> instruction_tree_sizes(const Program &program, int num_inputs) {
    std::vector<int> sizes(static_cast<std::size_t>(num_inputs + 1), 0);
    for (const auto &instr : program.instrs) {
        sizes.push_back(sizes[static_cast<std::size_t>(instr.s1)] + sizes[static_cast<std::size_t>(instr.s2)] + 1);
    }
    return sizes;
}

std::vector<int> output_depths(const Program &program, int num_inputs) {
    const std::vector<int> depths = instruction_depths(program, num_inputs);
    std::vector<int> result;
    result.reserve(program.outputs.size());
    for (const int output : program.outputs) {
        result.push_back(depths[static_cast<std::size_t>(output)]);
    }
    return result;
}

std::vector<int> output_cone_sizes(const Program &program, int num_inputs) {
    auto collect = [&](this auto &self, int source, std::set<int> &cone) -> void {
        if (!is_temp_source(source, num_inputs, static_cast<int>(program.instrs.size())) || cone.contains(source)) {
            return;
        }
        cone.insert(source);
        const Instruction &instr = program.instrs[static_cast<std::size_t>(temp_index_from_source(source, num_inputs))];
        self(instr.s1, cone);
        self(instr.s2, cone);
    };

    std::vector<int> sizes;
    sizes.reserve(program.outputs.size());
    for (const int output : program.outputs) {
        std::set<int> cone;
        collect(output, cone);
        sizes.push_back(static_cast<int>(cone.size()));
    }
    return sizes;
}

std::vector<int> instruction_fanouts(const Program &program, int num_inputs) {
    std::vector<int> fanouts(program.instrs.size(), 0);
    auto increment = [&](int source) {
        if (!is_temp_source(source, num_inputs, static_cast<int>(program.instrs.size()))) return;
        ++fanouts[static_cast<std::size_t>(temp_index_from_source(source, num_inputs))];
    };
    for (const auto &instr : program.instrs) {
        increment(instr.s1);
        increment(instr.s2);
    }
    for (const int output : program.outputs) {
        increment(output);
    }
    return fanouts;
}

int sum_values(const std::vector<int> &values) {
    int total = 0;
    for (const int value : values) {
        total += value;
    }
    return total;
}

int independent_instruction_pairs(const Program &program, int num_inputs) {
    std::map<std::pair<int, int>, bool> dependency_cache;
    auto depends_on = [&](this auto &self, int source, int target) -> bool {
        const auto key = std::make_pair(source, target);
        if (const auto it = dependency_cache.find(key); it != dependency_cache.end()) return it->second;
        if (source == target) {
            dependency_cache[key] = true;
            return true;
        }
        if (!is_temp_source(source, num_inputs, static_cast<int>(program.instrs.size()))) {
            dependency_cache[key] = false;
            return false;
        }
        const Instruction &instr = program.instrs[static_cast<std::size_t>(temp_index_from_source(source, num_inputs))];
        const bool result = self(instr.s1, target) || self(instr.s2, target);
        dependency_cache[key] = result;
        return result;
    };

    int count = 0;
    for (std::size_t left_idx = 0; left_idx < program.instrs.size(); ++left_idx) {
        const int left_source = temp_source(static_cast<int>(left_idx), num_inputs);
        for (std::size_t right_idx = left_idx + 1; right_idx < program.instrs.size(); ++right_idx) {
            const int right_source = temp_source(static_cast<int>(right_idx), num_inputs);
            if (!depends_on(left_source, right_source) && !depends_on(right_source, left_source)) {
                ++count;
            }
        }
    }
    return count;
}

int operator_cost(const Program &program) {
    int total = 0;
    for (const auto &instr : program.instrs) {
        total += instr.op == kOpXor ? kXorOperatorCost : kDefaultOperatorCost;
    }
    return total;
}

std::size_t popcount(const PackedMask &mask) {
    std::size_t count = 0;
    for (const std::uint64_t word : mask.words()) {
        count += static_cast<std::size_t>(std::popcount(word));
    }
    return count;
}

double node_value_entropy(const Program &program, const PackedExamples &packed) {
    const std::size_t sample_count = program.instrs.size() * static_cast<std::size_t>(packed.width);
    if (sample_count == 0) return 0.0;

    const std::vector<PackedMask> values = evaluate_all_sources(program, packed.input_masks, all_ones(packed.width));
    std::size_t true_count = 0;
    const auto first_temp = static_cast<std::size_t>(packed.input_masks.size() + 1);
    for (std::size_t idx = first_temp; idx < values.size(); ++idx) {
        true_count += popcount(values[idx]);
    }
    const std::size_t false_count = sample_count - true_count;
    double entropy = 0.0;
    for (const std::size_t count : {false_count, true_count}) {
        if (count == 0) continue;
        const double probability = static_cast<double>(count) / static_cast<double>(sample_count);
        entropy -= probability * std::log2(probability);
    }
    return entropy;
}

double deterministic_random_score(const Program &program, unsigned random_seed) {
    std::uint64_t hash = static_cast<std::uint64_t>(random_seed) ^ 0x9e3779b97f4a7c15ULL;
    for (const char ch : program_key(program)) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ULL;
    }
    return static_cast<double>(hash >> 11U) / static_cast<double>(std::uint64_t{1} << 53U);
}

void append_score_value(ProgramScore &score, const std::vector<int> &values, bool descending) {
    for (const int value : values) {
        score.parts.push_back(descending ? -static_cast<double>(value) : static_cast<double>(value));
    }
}

void append_score_value(ProgramScore &score, double value, bool descending) {
    score.parts.push_back(descending ? -value : value);
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

ProgramScore score_program(const Program &program, int num_inputs, const PackedExamples &packed,
                           const PostProcessScorePhase &phase, unsigned random_seed) {
    ProgramScore score;
    score.outputs = program.outputs;
    score.instr_key.reserve(program.instrs.size() * 3);
    std::optional<std::vector<int>> cached_instruction_depths;
    std::optional<std::vector<int>> cached_output_depths;
    std::optional<std::vector<int>> cached_tree_sizes;
    std::optional<std::vector<int>> cached_cone_sizes;
    std::optional<std::vector<int>> cached_fanouts;

    auto get_instruction_depths = [&]() -> const std::vector<int> & {
        if (!cached_instruction_depths.has_value()) cached_instruction_depths = instruction_depths(program, num_inputs);
        return *cached_instruction_depths;
    };
    auto get_output_depths = [&]() -> const std::vector<int> & {
        if (!cached_output_depths.has_value()) cached_output_depths = output_depths(program, num_inputs);
        return *cached_output_depths;
    };
    auto get_tree_sizes = [&]() -> const std::vector<int> & {
        if (!cached_tree_sizes.has_value()) cached_tree_sizes = instruction_tree_sizes(program, num_inputs);
        return *cached_tree_sizes;
    };
    auto get_cone_sizes = [&]() -> const std::vector<int> & {
        if (!cached_cone_sizes.has_value()) cached_cone_sizes = output_cone_sizes(program, num_inputs);
        return *cached_cone_sizes;
    };
    auto get_fanouts = [&]() -> const std::vector<int> & {
        if (!cached_fanouts.has_value()) cached_fanouts = instruction_fanouts(program, num_inputs);
        return *cached_fanouts;
    };

    for (const PostProcessScoreMetricSpec &metric : phase) {
        switch (metric.metric) {
        case PostProcessScoreMetric::ProgramLength:
            append_score_value(score, static_cast<double>(program.instrs.size()), metric.descending);
            break;
        case PostProcessScoreMetric::OutputDepth:
            append_score_value(score, get_output_depths(), metric.descending);
            break;
        case PostProcessScoreMetric::MaxOutputDepth: {
            const std::vector<int> &depths = get_output_depths();
            append_score_value(score, depths.empty() ? 0.0 : static_cast<double>(*std::ranges::max_element(depths)),
                               metric.descending);
            break;
        }
        case PostProcessScoreMetric::SumOutputDepth: {
            const std::vector<int> &depths = get_output_depths();
            append_score_value(score, static_cast<double>(sum_values(depths)), metric.descending);
            break;
        }
        case PostProcessScoreMetric::TotalNodeDepth: {
            const std::vector<int> &depths = get_instruction_depths();
            int total = 0;
            for (auto idx = static_cast<std::size_t>(num_inputs + 1); idx < depths.size(); ++idx) {
                total += depths[idx];
            }
            append_score_value(score, static_cast<double>(total), metric.descending);
            break;
        }
        case PostProcessScoreMetric::TotalTreeSize: {
            const std::vector<int> &sizes = get_tree_sizes();
            int total = 0;
            for (auto idx = static_cast<std::size_t>(num_inputs + 1); idx < sizes.size(); ++idx) {
                total += sizes[idx];
            }
            append_score_value(score, static_cast<double>(total), metric.descending);
            break;
        }
        case PostProcessScoreMetric::OperatorCost:
            append_score_value(score, static_cast<double>(operator_cost(program)), metric.descending);
            break;
        case PostProcessScoreMetric::XorCount:
            append_score_value(score,
                               static_cast<double>(std::ranges::count_if(
                                   program.instrs, [](const Instruction &instr) { return instr.op == kOpXor; })),
                               metric.descending);
            break;
        case PostProcessScoreMetric::OutputConeSize:
            append_score_value(score, get_cone_sizes(), metric.descending);
            break;
        case PostProcessScoreMetric::MaxOutputConeSize: {
            const std::vector<int> &sizes = get_cone_sizes();
            append_score_value(score, sizes.empty() ? 0.0 : static_cast<double>(*std::ranges::max_element(sizes)),
                               metric.descending);
            break;
        }
        case PostProcessScoreMetric::SumOutputConeSize: {
            const std::vector<int> &sizes = get_cone_sizes();
            append_score_value(score, static_cast<double>(sum_values(sizes)), metric.descending);
            break;
        }
        case PostProcessScoreMetric::Fanout:
            append_score_value(score, get_fanouts(), metric.descending);
            break;
        case PostProcessScoreMetric::MaxFanout: {
            const std::vector<int> &fanouts = get_fanouts();
            append_score_value(score, fanouts.empty() ? 0.0 : static_cast<double>(*std::ranges::max_element(fanouts)),
                               metric.descending);
            break;
        }
        case PostProcessScoreMetric::SumFanout: {
            const std::vector<int> &fanouts = get_fanouts();
            append_score_value(score, static_cast<double>(sum_values(fanouts)), metric.descending);
            break;
        }
        case PostProcessScoreMetric::OneFanoutCount: {
            const std::vector<int> &fanouts = get_fanouts();
            append_score_value(score, static_cast<double>(std::ranges::count(fanouts, 1)), metric.descending);
            break;
        }
        case PostProcessScoreMetric::IndependentPairs:
            append_score_value(score, static_cast<double>(independent_instruction_pairs(program, num_inputs)),
                               metric.descending);
            break;
        case PostProcessScoreMetric::Entropy:
            append_score_value(score, node_value_entropy(program, packed), metric.descending);
            break;
        case PostProcessScoreMetric::Random:
            append_score_value(score, deterministic_random_score(program, random_seed), metric.descending);
            break;
        }
    }

    for (const auto &instr : program.instrs) {
        score.instr_key.push_back(instr.op);
        score.instr_key.push_back(instr.s1);
        score.instr_key.push_back(instr.s2);
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
                           int num_outputs, PostProcessGeneratorRun &generator) {
    const std::string key = program_key(candidate);
    if (key == base_key || !seen.insert(key).second) return false;
    if (const std::optional<std::string> error = validate_program_invariants(candidate, num_inputs, num_outputs)) {
        generator.invalid_candidate();
        throw std::logic_error("post-process generated an invalid candidate program: " + *error);
    }
    if (!verify_program(candidate, examples, num_inputs, num_outputs).empty()) return false;
    candidates.push_back(candidate);
    generator.candidate_accepted();
    return true;
}
