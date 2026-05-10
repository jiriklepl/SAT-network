#pragma once

#include "datasets.hpp"
#include "mask.hpp"
#include "postprocess.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <cstddef>

#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

struct ProgramScore {
    std::vector<double> parts;
    std::vector<int> outputs;
    std::vector<int> instr_key;
};

bool operator<(const ProgramScore &left, const ProgramScore &right);

std::string program_key(const Program &program);
std::optional<std::string> validate_program_invariants(const Program &program, int num_inputs, int num_outputs);
void require_valid_program(const Program &program, int num_inputs, int num_outputs, const std::string &context);

Program prune_dead_nodes(const Program &program, int num_inputs);
ProgramScore score_program(const Program &program, int num_inputs, const PackedExamples &packed,
                           const PostProcessScorePhase &phase, unsigned random_seed);
std::vector<PackedMask> evaluate_all_sources(const Program &program, std::span<const PackedMask> input_masks,
                                             const PackedMask &all_examples_mask);

bool push_unique_candidate(std::vector<Program> &candidates, std::set<std::string> &seen, const Program &candidate,
                           const std::string &base_key, std::span<const Example> examples, int num_inputs,
                           int num_outputs, ProfileData *profile = nullptr);

std::vector<Program> generate_mask_simplification_candidates(
    const Program &base, std::span<const Example> examples, int num_inputs, int num_outputs,
    const PackedExamples &packed, std::span<const PackedMask> values, const std::string &base_key,
    std::set<std::string> &seen, std::size_t max_candidates, const PostProcessOptions &options);

std::vector<Program> generate_resynthesis_candidates(const Program &base, std::span<const Example> examples,
                                                     int num_inputs, int num_outputs, const PackedExamples &packed,
                                                     std::span<const PackedMask> values,
                                                     const PostProcessOptions &options, const std::string &base_key,
                                                     std::set<std::string> &seen, std::size_t max_candidates,
                                                     ProfileData *profile);
