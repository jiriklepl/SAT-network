#pragma once

#include "postprocess.hpp"

#include <cstddef>

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

struct CliOptions {
    std::string config_path;
    std::string dataset_name;
    std::string assume_path;
    std::optional<int> instructions;
    std::string solver = "simple-tactic";
    bool encode_boolean = false;
    bool balanced_select = false;
    bool force_ordered = false;
    bool force_useful = false;
    std::optional<std::size_t> batch_size;
    bool cegis = false;
    std::size_t cegis_initial_size = 64;
    std::size_t cegis_counterexamples = 1;
    bool no_shuffle = false;
    unsigned seed = 0;
    bool quiet = false;
    int verbosity = 1;
    bool list_datasets = false;
    bool dump_dataset = false;
    bool make_smt2 = false;
    bool make_dimacs = false;
    bool make_blif = false;
    bool output_blif = false;
    bool profile = false;
    bool post_process = false;
    std::size_t post_process_beam_width = 1;
    std::size_t post_process_beam_rounds = 0;
    std::size_t post_process_beam_candidates = 0;
    std::vector<PostProcessScorePhase> post_process_score_phases = {
        {{.metric = PostProcessScoreMetric::ProgramLength, .descending = false}}};
    std::size_t post_process_replace_patience = 50;
    std::size_t post_process_resynthesis_maxnodes = 5;
    std::size_t post_process_resynthesis_patience = 1;
    double generator_timeout = 0.0;
};

void print_usage(std::ostream &out);
CliOptions parse_args(int argc, char **argv);
