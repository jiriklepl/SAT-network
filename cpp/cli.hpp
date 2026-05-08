#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>

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
    bool list_datasets = false;
    bool dump_dataset = false;
    bool make_smt2 = false;
    bool make_dimacs = false;
    bool make_blif = false;
    bool output_blif = false;
};

void print_usage(std::ostream &out);
CliOptions parse_args(int argc, char **argv);
void log_info(const CliOptions &options, const std::string &message);
