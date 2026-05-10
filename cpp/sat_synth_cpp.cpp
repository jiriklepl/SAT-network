#include "assumptions.hpp"
#include "cli.hpp"
#include "config.hpp"
#include "datasets.hpp"
#include "profile.hpp"
#include "program.hpp"
#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

Config load_requested_config(const CliOptions &cli) {
    Config cfg = cli.config_path.empty() ? build_dataset_from_config(default_dataset_config(cli.dataset_name))
                                         : load_config(cli.config_path);
    if (cli.instructions.has_value()) cfg.instructions = *cli.instructions;
    if (cfg.instructions < 0) throw std::runtime_error("--instructions must be non-negative");
    return cfg;
}

SolveOptions make_solve_options(const CliOptions &cli) {
    return {
        .solver = cli.solver,
        .encoding = {.encode_boolean = cli.encode_boolean,
                     .force_ordered = cli.force_ordered,
                     .force_useful = cli.force_useful,
                     .balanced_select = cli.balanced_select},
        .assumptions = {},
        .batch_size = cli.batch_size,
        .cegis = cli.cegis,
        .cegis_initial_size = cli.cegis_initial_size,
        .cegis_counterexamples = cli.cegis_counterexamples,
        .postprocess =
            {
                .enabled = cli.post_process,
                .beam_width = cli.post_process_beam_width,
                .beam_rounds = cli.post_process_beam_rounds,
                .beam_candidates = cli.post_process_beam_candidates,
                .resynthesis_maxnodes = cli.post_process_resynthesis_maxnodes,
                .resynthesis_patience = cli.post_process_resynthesis_patience,
                .generator_timeout_seconds = cli.generator_timeout,
            },
    };
}

void print_profile(const ProfileData &profile) {
    std::cerr << std::fixed << std::setprecision(6);
    std::cerr << "PROFILE dataset_generation_seconds=" << profile.dataset_generation_seconds << "\n";
    std::cerr << "PROFILE structure_encoding_seconds=" << profile.structure_encoding_seconds << "\n";
    std::cerr << "PROFILE example_packing_seconds=" << profile.example_packing_seconds << "\n";
    std::cerr << "PROFILE example_encoding_seconds=" << profile.example_encoding_seconds << "\n";
    std::cerr << "PROFILE z3_solve_seconds=" << profile.z3_solve_seconds << "\n";
    std::cerr << "PROFILE model_extraction_seconds=" << profile.model_extraction_seconds << "\n";
    std::cerr << "PROFILE post_processing_seconds=" << profile.post_processing_seconds << "\n";
    std::cerr << "PROFILE packed_verification_seconds=" << profile.packed_verification_seconds << "\n";
    std::cerr << "PROFILE structure_constraints=" << profile.structure_constraints << "\n";
    std::cerr << "PROFILE example_constraints=" << profile.example_constraints << "\n";
    std::cerr << "PROFILE example_batches=" << profile.example_batches << "\n";
    std::cerr << "PROFILE packed_examples=" << profile.packed_examples << "\n";
    std::cerr << "PROFILE solver_checks=" << profile.solver_checks << "\n";
    std::cerr << "PROFILE model_extractions=" << profile.model_extractions << "\n";
    std::cerr << "PROFILE post_processing_runs=" << profile.post_processing_runs << "\n";
    std::cerr << "PROFILE post_processing_input_instructions=" << profile.post_processing_input_instructions << "\n";
    std::cerr << "PROFILE post_processing_output_instructions=" << profile.post_processing_output_instructions << "\n";
    std::cerr << "PROFILE post_processing_resynthesis_windows_considered="
              << profile.post_processing_resynthesis_windows_considered << "\n";
    std::cerr << "PROFILE post_processing_resynthesis_windows_sat=" << profile.post_processing_resynthesis_windows_sat
              << "\n";
    std::cerr << "PROFILE post_processing_resynthesis_candidates_materialized="
              << profile.post_processing_resynthesis_candidates_materialized << "\n";
    std::cerr << "PROFILE post_processing_resynthesis_invalid_candidates="
              << profile.post_processing_resynthesis_invalid_candidates << "\n";
    std::cerr << "PROFILE post_processing_resynthesis_candidates_accepted="
              << profile.post_processing_resynthesis_candidates_accepted << "\n";
    std::cerr << "PROFILE post_processing_resynthesis_timeout_exits="
              << profile.post_processing_resynthesis_timeout_exits << "\n";
    std::cerr << "PROFILE verification_examples=" << profile.verification_examples << "\n";
    std::cerr << "PROFILE bv_cache_hits=" << profile.bv_cache_hits << "\n";
    std::cerr << "PROFILE bv_cache_misses=" << profile.bv_cache_misses << "\n";
}

Assumptions load_assumptions(const CliOptions &cli, const Config &cfg) {
    if (cli.assume_path.empty()) {
        return {};
    }

    const ProgramSpec spec{
        .num_inputs = cfg.num_inputs, .num_outputs = cfg.num_outputs, .program_length = cfg.instructions};
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
        const CliOptions cli = parse_args(argc, argv);
        if (cli.list_datasets) {
            for (const auto &name : available_dataset_names()) {
                std::cout << name << "\n";
            }
            return 0;
        }

        ProfileData profile;
        const auto dataset_start = Clock::now();
        Config cfg = load_requested_config(cli);
        if (cli.profile) profile.dataset_generation_seconds = elapsed_seconds(dataset_start);
        if (cli.dump_dataset) {
            std::cout << config_to_json(cfg).dump(2) << "\n";
            if (cli.profile) print_profile(profile);
            return 0;
        }
        if (cli.make_blif) {
            export_spec_blif(std::cout, cfg.examples, cfg.num_inputs, cfg.num_outputs);
            if (cli.profile) print_profile(profile);
            return 0;
        }

        if (!cli.no_shuffle) {
            std::mt19937 rng(cli.seed);
            std::shuffle(cfg.examples.begin(), cfg.examples.end(), rng);
        }

        SolveOptions solve_options = make_solve_options(cli);
        if (cli.profile) solve_options.profile = &profile;
        solve_options.assumptions = load_assumptions(cli, cfg);

        if (cli.make_smt2) {
            std::cout << make_smt2(cfg, solve_options);
            if (cli.profile) print_profile(profile);
            return 0;
        }
        if (cli.make_dimacs) {
            std::cout << make_dimacs(cfg, solve_options);
            if (cli.profile) print_profile(profile);
            return 0;
        }

        log_info(cli, "Built program structure with " + std::to_string(cfg.instructions) + " instructions");
        const SolveResult result = solve_config(cfg, solve_options);
        if (result.status == SolveStatus::Sat) {
            if (cli.output_blif) {
                emit_program_blif(std::cout, *result.program, cfg.num_inputs);
            } else {
                emit_program(std::cout, *result.program, cfg.num_inputs);
            }
            log_info(cli, "SAT in " + std::to_string(result.elapsed_seconds) + " seconds");
            if (cli.profile) print_profile(profile);
            return 0;
        }
        if (result.status == SolveStatus::VerificationFailed) {
            if (result.program.has_value()) {
                if (cli.output_blif) {
                    emit_program_blif(std::cout, *result.program, cfg.num_inputs);
                } else {
                    emit_program(std::cout, *result.program, cfg.num_inputs);
                }
            }
            std::cerr << "Total mismatches: " << result.mismatch_count << "\n";
            if (cli.profile) print_profile(profile);
            return 1;
        }
        if (result.status == SolveStatus::Unsat) {
            log_info(cli, "UNSAT in " + std::to_string(result.elapsed_seconds) + " seconds");
            if (cli.profile) print_profile(profile);
            return 1;
        }
        log_info(cli, "UNKNOWN in " + std::to_string(result.elapsed_seconds) + " seconds");
        if (cli.profile) print_profile(profile);
        return 1;
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 2;
    }
}
