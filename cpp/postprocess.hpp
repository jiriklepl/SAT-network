#pragma once

#include "datasets.hpp"
#include "logging.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <cstddef>
#include <cstdint>

#include <span>
#include <string>
#include <vector>

enum class PostProcessScoreMetric : std::uint8_t {
    ProgramLength,
    OutputDepth,
    MaxOutputDepth,
    SumOutputDepth,
    TotalNodeDepth,
    TotalTreeSize,
    OperatorCost,
    XorCount,
    OutputConeSize,
    MaxOutputConeSize,
    SumOutputConeSize,
    Fanout,
    MaxFanout,
    SumFanout,
    OneFanoutCount,
    IndependentPairs,
    Entropy,
    Random,
};

struct PostProcessScoreMetricSpec {
    PostProcessScoreMetric metric = PostProcessScoreMetric::ProgramLength;
    bool descending = false;
};

using PostProcessScorePhase = std::vector<PostProcessScoreMetricSpec>;

struct PostProcessOptions {
    bool enabled = false;
    std::size_t beam_width = 1;
    std::size_t beam_rounds = 0;
    std::size_t beam_candidates = 0;
    std::vector<PostProcessScorePhase> score_phases = {{{PostProcessScoreMetric::ProgramLength, false}}};
    unsigned random_seed = 0;
    std::size_t replace_patience = 50;
    std::size_t resynthesis_maxnodes = 5;
    std::size_t resynthesis_patience = 1;
    double generator_timeout_seconds = 0.0;
    const Logger *logger = nullptr;
};

PostProcessScoreMetric post_process_score_metric_by_name(const std::string &name);
std::string post_process_score_metric_name(PostProcessScoreMetric metric);

Program post_process_program(const Program &program, std::span<const Example> examples, int num_inputs, int num_outputs,
                             const PostProcessOptions &options, ProfileData *profile = nullptr);
