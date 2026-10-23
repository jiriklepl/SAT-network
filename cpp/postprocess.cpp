#include "postprocess.hpp"

#include "datasets.hpp"
#include "mask.hpp"
#include "postprocess_internal.hpp"
#include "profile.hpp"
#include "program.hpp"
#include "symbolic_verifier.hpp"

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

struct ScoredCandidate {
    Program program;
    ProgramScore score;
};

std::vector<ScoredCandidate> score_and_sort_candidates(std::vector<Program> candidates, int num_inputs,
                                                       const PackedExamples &packed, const PostProcessScorePhase &phase,
                                                       unsigned random_seed) {
    std::vector<ScoredCandidate> scored;
    scored.reserve(candidates.size());
    for (auto &cand : candidates) {
        ProgramScore score = score_program(cand, num_inputs, packed, phase, random_seed);
        scored.push_back({std::move(cand), std::move(score)});
    }
    std::ranges::sort(scored, [](const auto &left, const auto &right) { return left.score < right.score; });
    return scored;
}

std::vector<Program> generate_candidates(const Program &program, VerificationOracle &oracle,
                                         const PostProcessOptions &options, const PostProcessScorePhase &phase,
                                         ProfileData *profile) {
    const int num_inputs = oracle.num_inputs();
    const Program base = canonicalize_program(program, num_inputs);
    const std::string base_key = program_key(base);
    const std::size_t max_candidates = options.beam_candidates;

    std::set<std::string> seen;
    const std::size_t before_mask = profile == nullptr ? 0 : profile->post_processing_mask_candidates_accepted;
    const std::size_t before_replacement =
        profile == nullptr ? 0 : profile->post_processing_replacement_candidates_accepted;
    std::vector<Program> candidates =
        generate_mask_simplification_candidates(base, oracle, base_key, seen, max_candidates, options, profile);
    const std::size_t after_mask = profile == nullptr ? 0 : profile->post_processing_mask_candidates_accepted;
    const std::size_t after_replacement =
        profile == nullptr ? 0 : profile->post_processing_replacement_candidates_accepted;
    if (options.logger != nullptr) {
        options.logger->log(3, "Post-process mask/replacement generators accepted " +
                                   std::to_string(after_mask - before_mask) + "/" +
                                   std::to_string(after_replacement - before_replacement) + " candidates");
    }
    const std::size_t before_resynthesis =
        profile == nullptr ? 0 : profile->post_processing_resynthesis_candidates_accepted;
    std::vector<Program> resynth_candidates =
        generate_resynthesis_candidates(base, oracle, options, base_key, seen, max_candidates, profile);
    candidates.insert(candidates.end(), std::make_move_iterator(resynth_candidates.begin()),
                      std::make_move_iterator(resynth_candidates.end()));
    const std::size_t after_resynthesis =
        profile == nullptr ? 0 : profile->post_processing_resynthesis_candidates_accepted;
    if (options.logger != nullptr) {
        options.logger->log(3, "Post-process resynthesis generator accepted " +
                                   std::to_string(after_resynthesis - before_resynthesis) + " candidates");
    }

    std::vector<ScoredCandidate> scored =
        score_and_sort_candidates(std::move(candidates), num_inputs, oracle.packed(), phase, options.random_seed);
    if (max_candidates != 0 && scored.size() > max_candidates) scored.resize(max_candidates);
    std::vector<Program> result;
    result.reserve(scored.size());
    for (auto &item : scored) {
        result.push_back(std::move(item.program));
    }
    return result;
}

Program run_score_phase(const Program &start, VerificationOracle &oracle, const PostProcessOptions &options,
                        const PostProcessScorePhase &phase, ProfileData *profile) {
    const int num_inputs = oracle.num_inputs();
    Program best = start;
    ProgramScore best_score = score_program(best, num_inputs, oracle.packed(), phase, options.random_seed);
    ProgramScore start_score = best_score;
    std::size_t scored_gen = oracle.basis_generation();
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
            const std::vector<Program> candidates = generate_candidates(state, oracle, options, phase, profile);
            for (const Program &candidate : candidates) {
                if (globally_seen.insert(program_key(candidate)).second) {
                    next.push_back(candidate);
                }
            }
        }
        if (next.empty()) break;
        if (oracle.basis_generation() != scored_gen) {
            scored_gen = oracle.basis_generation();
            best_score = score_program(best, num_inputs, oracle.packed(), phase, options.random_seed);
            start_score = score_program(start, num_inputs, oracle.packed(), phase, options.random_seed);
        }
        if (options.logger != nullptr) {
            options.logger->log(3, "Post-process round " + std::to_string(round + 1) + " generated " +
                                       std::to_string(next.size()) + " new candidates");
        }
        std::vector<ScoredCandidate> scored =
            score_and_sort_candidates(std::move(next), num_inputs, oracle.packed(), phase, options.random_seed);
        if (scored.size() > options.beam_width) scored.resize(options.beam_width);

        bool improved = false;
        if (!scored.empty() && scored.front().score < best_score) {
            best = scored.front().program;
            best_score = scored.front().score;
            improved = true;
        }
        if (improved && options.logger != nullptr) {
            options.logger->log(2, "Post-process round " + std::to_string(round + 1) + " improved to " +
                                       std::to_string(best.instrs.size()) + " instructions");
        }
        beam.clear();
        beam.reserve(scored.size());
        for (auto &item : scored) {
            beam.push_back(std::move(item.program));
        }
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

    VerificationOracle oracle(examples, num_inputs, num_outputs, options.verifier, profile);
    Program best = canonicalize_program(program, num_inputs);
    require_valid_program(best, num_inputs, num_outputs, "post-process pruned input program");
    const std::string best_key = program_key(best);
    if (!oracle.verify_candidate(best, best_key)) return program;

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
        best = run_score_phase(best, oracle, options, phase, profile);
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
