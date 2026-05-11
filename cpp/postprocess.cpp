#include "postprocess.hpp"

#include "datasets.hpp"
#include "mask.hpp"
#include "postprocess_internal.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <algorithm>
#include <cstddef>

#include <limits>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string score_phase_summary(const PostProcessScorePhase &phase) {
    std::ostringstream out;
    for (std::size_t idx = 0; idx < phase.size(); ++idx) {
        if (idx != 0) out << ",";
        if (phase[idx].descending) out << "-";
        out << post_process_score_metric_name(phase[idx].metric);
    }
    return out.str();
}

std::vector<Program> generate_candidates(const Program &program, std::span<const Example> examples, int num_inputs,
                                         int num_outputs, const PackedExamples &packed,
                                         const PostProcessOptions &options, const PostProcessScorePhase &phase,
                                         ProfileData *profile) {
    const Program base = prune_dead_nodes(program, num_inputs);
    const std::string base_key = program_key(base);
    const PackedMask all_mask = all_ones(packed.width);
    const std::vector<PackedMask> values = evaluate_all_sources(base, packed.input_masks, all_mask);
    const std::size_t max_candidates = options.beam_candidates;

    std::set<std::string> seen;
    const std::size_t before_mask = profile == nullptr ? 0 : profile->post_processing_mask_candidates_accepted;
    const std::size_t before_replacement =
        profile == nullptr ? 0 : profile->post_processing_replacement_candidates_accepted;
    std::vector<Program> candidates = generate_mask_simplification_candidates(
        base, examples, num_inputs, num_outputs, packed, values, base_key, seen, max_candidates, options, profile);
    const std::size_t after_mask = profile == nullptr ? 0 : profile->post_processing_mask_candidates_accepted;
    const std::size_t after_replacement =
        profile == nullptr ? 0 : profile->post_processing_replacement_candidates_accepted;
    if (options.logger != nullptr) {
        options.logger->log(3, "Post-process mask/replacement generators accepted " +
                                   std::to_string(after_mask - before_mask) + "/" +
                                   std::to_string(after_replacement - before_replacement) + " candidates");
    }
    if (max_candidates == 0 || candidates.size() < max_candidates) {
        const std::size_t before_resynthesis =
            profile == nullptr ? 0 : profile->post_processing_resynthesis_candidates_accepted;
        std::vector<Program> resynth_candidates = generate_resynthesis_candidates(
            base, examples, num_inputs, num_outputs, packed, values, options, base_key, seen,
            max_candidates == 0 ? 0 : max_candidates - candidates.size(), profile);
        candidates.insert(candidates.end(), resynth_candidates.begin(), resynth_candidates.end());
        const std::size_t after_resynthesis =
            profile == nullptr ? 0 : profile->post_processing_resynthesis_candidates_accepted;
        if (options.logger != nullptr) {
            options.logger->log(3, "Post-process resynthesis generator accepted " +
                                       std::to_string(after_resynthesis - before_resynthesis) + " candidates");
        }
    }

    std::ranges::sort(candidates, [&](const Program &left, const Program &right) {
        return score_program(left, num_inputs, packed, phase, options.random_seed) <
               score_program(right, num_inputs, packed, phase, options.random_seed);
    });
    return candidates;
}

Program run_score_phase(const Program &start, std::span<const Example> examples, int num_inputs, int num_outputs,
                        const PackedExamples &packed, const PostProcessOptions &options,
                        const PostProcessScorePhase &phase, ProfileData *profile) {
    Program best = start;
    ProgramScore best_score = score_program(best, num_inputs, packed, phase, options.random_seed);
    const ProgramScore start_score = best_score;
    std::vector<Program> beam{best};
    std::set<std::string> globally_seen{program_key(best)};
    const std::size_t max_rounds =
        options.beam_rounds == 0 ? std::numeric_limits<std::size_t>::max() : options.beam_rounds;

    for (std::size_t round = 0; round < max_rounds; ++round) {
        if (options.logger != nullptr) {
            options.logger->log(3, "Post-process round " + std::to_string(round + 1) + " starting with beam size " +
                                       std::to_string(beam.size()));
        }
        std::vector<Program> next;
        for (const Program &state : beam) {
            const std::vector<Program> candidates =
                generate_candidates(state, examples, num_inputs, num_outputs, packed, options, phase, profile);
            for (const Program &candidate : candidates) {
                if (globally_seen.insert(program_key(candidate)).second) {
                    next.push_back(candidate);
                }
            }
        }
        if (next.empty()) break;
        if (options.logger != nullptr) {
            options.logger->log(3, "Post-process round " + std::to_string(round + 1) + " generated " +
                                       std::to_string(next.size()) + " new candidates");
        }
        std::ranges::sort(next, [&](const Program &left, const Program &right) {
            return score_program(left, num_inputs, packed, phase, options.random_seed) <
                   score_program(right, num_inputs, packed, phase, options.random_seed);
        });
        if (next.size() > options.beam_width) next.resize(options.beam_width);

        bool improved = false;
        for (const Program &candidate : next) {
            const ProgramScore score = score_program(candidate, num_inputs, packed, phase, options.random_seed);
            if (score < best_score) {
                best = candidate;
                best_score = score;
                improved = true;
            }
        }
        if (improved && options.logger != nullptr) {
            options.logger->log(2, "Post-process round " + std::to_string(round + 1) + " improved to " +
                                       std::to_string(best.instrs.size()) + " instructions");
        }
        beam = std::move(next);
        if (options.beam_rounds == 0 && !improved) break;
    }

    return best_score < start_score ? best : start;
}

}  // namespace

Program post_process_program(const Program &program, std::span<const Example> examples, int num_inputs, int num_outputs,
                             const PostProcessOptions &options, ProfileData *profile) {
    if (!options.enabled || examples.empty()) return program;
    require_valid_program(program, num_inputs, num_outputs, "post-process input program");
    if (options.logger != nullptr) {
        options.logger->log(2, "Post-processing program with " + std::to_string(program.instrs.size()) +
                                   " input instructions");
    }

    const PackedExamples packed = pack_examples(examples, num_inputs, num_outputs);
    Program best = prune_dead_nodes(program, num_inputs);
    require_valid_program(best, num_inputs, num_outputs, "post-process pruned input program");
    if (!verify_program(best, examples, num_inputs, num_outputs).empty()) return program;

    for (std::size_t phase_idx = 0; phase_idx < options.score_phases.size(); ++phase_idx) {
        const PostProcessScorePhase &phase = options.score_phases[phase_idx];
        if (phase.empty()) {
            throw std::logic_error("post-process score phase must not be empty");
        }
        if (options.logger != nullptr) {
            options.logger->log(3, "Post-process phase " + std::to_string(phase_idx + 1) + "/" +
                                       std::to_string(options.score_phases.size()) +
                                       " metrics=" + score_phase_summary(phase));
        }
        const std::size_t before = best.instrs.size();
        best = run_score_phase(best, examples, num_inputs, num_outputs, packed, options, phase, profile);
        if (best.instrs.size() < before && options.logger != nullptr) {
            options.logger->log(2, "Post-process phase " + std::to_string(phase_idx + 1) + " reduced program from " +
                                       std::to_string(before) + " to " + std::to_string(best.instrs.size()) +
                                       " instructions");
        }
    }

    if (options.logger != nullptr) {
        options.logger->log(2, "Post-processing finished with " + std::to_string(best.instrs.size()) +
                                   " output instructions");
    }
    return best;
}
