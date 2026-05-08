#include "assumptions.hpp"
#include "cli.hpp"
#include "config.hpp"
#include "datasets.hpp"
#include "program.hpp"
#include "solver.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

Config load_requested_config(const CliOptions &cli) {
    Config cfg = cli.config_path.empty()
        ? build_dataset_from_config(default_dataset_config(cli.dataset_name))
        : load_config(cli.config_path);
    if (cli.instructions.has_value()) cfg.instructions = *cli.instructions;
    if (cfg.instructions < 0) throw std::runtime_error("--instructions must be non-negative");
    return cfg;
}

SolveOptions make_solve_options(const CliOptions &cli) {
    SolveOptions options;
    options.solver = cli.solver;
    options.encoding = EncodingOptions{cli.encode_boolean, cli.force_ordered, cli.force_useful, cli.balanced_select};
    options.batch_size = cli.batch_size;
    options.cegis = cli.cegis;
    options.cegis_initial_size = cli.cegis_initial_size;
    options.cegis_counterexamples = cli.cegis_counterexamples;
    return options;
}

Assumptions load_assumptions(const CliOptions &cli, const Config &cfg) {
    if (cli.assume_path.empty()) {
        return {};
    }

    ProgramSpec spec{cfg.num_inputs, cfg.num_outputs, cfg.instructions};
    if (cli.assume_path == "-") {
        return parse_assumptions(std::cin, spec);
    }

    std::ifstream file(cli.assume_path);
    if (!file) {
        throw std::runtime_error("Assume file not found: " + cli.assume_path);
    }
    return parse_assumptions(file, spec);
}

}  // namespace

int main(int argc, char **argv) {
    try {
        CliOptions cli = parse_args(argc, argv);
        if (cli.list_datasets) {
            for (const auto &name : available_dataset_names()) {
                std::cout << name << "\n";
            }
            return 0;
        }

        Config cfg = load_requested_config(cli);
        if (cli.dump_dataset) {
            std::cout << config_to_json(cfg).dump(2) << "\n";
            return 0;
        }

        if (!cli.no_shuffle) {
            std::mt19937 rng(cli.seed);
            std::shuffle(cfg.examples.begin(), cfg.examples.end(), rng);
        }

        SolveOptions solve_options = make_solve_options(cli);
        solve_options.assumptions = load_assumptions(cli, cfg);

        log_info(cli, "Built program structure with " + std::to_string(cfg.instructions) + " instructions");
        SolveResult result = solve_config(cfg, solve_options);
        if (result.status == SolveStatus::Sat) {
            emit_program(std::cout, *result.program, cfg.num_inputs);
            log_info(cli, "SAT in " + std::to_string(result.elapsed_seconds) + " seconds");
            return 0;
        }
        if (result.status == SolveStatus::VerificationFailed) {
            if (result.program.has_value()) emit_program(std::cout, *result.program, cfg.num_inputs);
            std::cerr << "Total mismatches: " << result.mismatch_count << "\n";
            return 1;
        }
        if (result.status == SolveStatus::Unsat) {
            log_info(cli, "UNSAT in " + std::to_string(result.elapsed_seconds) + " seconds");
            return 1;
        }
        log_info(cli, "UNKNOWN in " + std::to_string(result.elapsed_seconds) + " seconds");
        return 1;
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 2;
    }
}
