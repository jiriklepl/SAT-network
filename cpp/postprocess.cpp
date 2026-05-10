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
#include <vector>

namespace {

std::vector<Program> generate_candidates(const Program &program, std::span<const Example> examples, int num_inputs,
                                         int num_outputs, const PackedExamples &packed,
                                         const PostProcessOptions &options, ProfileData *profile) {
    const Program base = prune_dead_nodes(program, num_inputs);
    const std::string base_key = program_key(base);
    const PackedMask all_mask = all_ones(packed.width);
    const std::vector<PackedMask> values = evaluate_all_sources(base, packed.input_masks, all_mask);
    const std::size_t max_candidates = options.beam_candidates;

    std::set<std::string> seen;
    std::vector<Program> candidates = generate_mask_simplification_candidates(
        base, examples, num_inputs, num_outputs, packed, values, base_key, seen, max_candidates);
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
