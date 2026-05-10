#pragma once

#include "datasets.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <cstddef>

#include <span>

struct PostProcessOptions {
    bool enabled = false;
    std::size_t beam_width = 1;
    std::size_t beam_rounds = 0;
    std::size_t beam_candidates = 0;
    std::size_t resynthesis_maxnodes = 5;
    std::size_t resynthesis_patience = 1;
    double generator_timeout_seconds = 0.0;
};

Program post_process_program(const Program &program, std::span<const Example> examples, int num_inputs, int num_outputs,
                             const PostProcessOptions &options, ProfileData *profile = nullptr);
