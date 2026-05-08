#include "cli.hpp"
#include "config.hpp"
#include "datasets.hpp"
#include "encoding.hpp"
#include "program.hpp"

#include <z3++.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

Config load_requested_config(const CliOptions &cli) {
    Config cfg = cli.config_path.empty()
        ? build_dataset_from_config(default_dataset_config(cli.dataset_name))
        : load_config(cli.config_path);
    if (cli.instructions.has_value()) cfg.instructions = *cli.instructions;
    if (cfg.instructions < 0) throw std::runtime_error("--instructions must be non-negative");
    return cfg;
}

z3::check_result solve_with_cegis(
    z3::context &ctx,
    z3::solver &solver,
    const Config &cfg,
    const ProgramSpec &spec,
    const EncodingOptions &encoding,
    const CliOptions &cli,
    std::optional<Program> &program
) {
    std::vector<bool> active(cfg.examples.size(), false);
    std::vector<Example> initial;
    std::size_t initial_count = std::min(cli.cegis_initial_size, cfg.examples.size());
    for (std::size_t idx = 0; idx < initial_count; ++idx) {
        active[idx] = true;
        initial.push_back(cfg.examples[idx]);
    }
    add_example_constraints(ctx, solver, initial, "cegis0", spec, encoding);

    std::size_t iteration = 0;
    while (true) {
        z3::check_result result = solver.check();
        if (result != z3::sat) return result;

        program = extract_program(ctx, solver.get_model(), spec);
        std::vector<std::size_t> mismatches = verify_program(*program, cfg.examples, spec.num_inputs, spec.num_outputs);
        if (mismatches.empty()) return result;

        std::vector<Example> counterexamples;
        for (std::size_t idx : mismatches) {
            if (!active[idx]) {
                active[idx] = true;
                counterexamples.push_back(cfg.examples[idx]);
            }
            if (counterexamples.size() >= cli.cegis_counterexamples) break;
        }
        if (counterexamples.empty()) {
            throw std::runtime_error("CEGIS candidate failed on already constrained examples");
        }
        ++iteration;
        add_example_constraints(ctx, solver, counterexamples, "cegis" + std::to_string(iteration), spec, encoding);
    }
}

z3::check_result solve_with_batches(
    z3::context &ctx,
    z3::solver &solver,
    const Config &cfg,
    const ProgramSpec &spec,
    const EncodingOptions &encoding,
    const CliOptions &cli
) {
    z3::check_result result = z3::unknown;
    std::size_t batch_size = cli.batch_size.value_or(cfg.examples.size());
    for (std::size_t offset = 0, batch_idx = 0; offset < cfg.examples.size(); offset += batch_size, ++batch_idx) {
        std::size_t end = std::min(offset + batch_size, cfg.examples.size());
        std::vector<Example> batch(cfg.examples.begin() + static_cast<std::ptrdiff_t>(offset),
                                   cfg.examples.begin() + static_cast<std::ptrdiff_t>(end));
        add_example_constraints(ctx, solver, batch, "b" + std::to_string(batch_idx), spec, encoding);
        result = solver.check();
        if (result != z3::sat) break;
    }
    return result;
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

        ProgramSpec spec{cfg.num_inputs, cfg.num_outputs, cfg.instructions};
        EncodingOptions encoding{cli.encode_boolean, cli.force_ordered, cli.force_useful, cli.balanced_select};
        z3::context ctx;
        z3::solver solver = make_solver(ctx, cli.solver);
        add_exprs(solver, build_program(ctx, spec, encoding));
        log_info(cli, "Built program structure with " + std::to_string(spec.program_length) + " instructions");

        auto start = std::chrono::steady_clock::now();
        std::optional<Program> program;
        z3::check_result result = cli.cegis
            ? solve_with_cegis(ctx, solver, cfg, spec, encoding, cli, program)
            : solve_with_batches(ctx, solver, cfg, spec, encoding, cli);

        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (result == z3::sat) {
            if (!program.has_value()) program = extract_program(ctx, solver.get_model(), spec);
            emit_program(std::cout, *program, spec.num_inputs);
            std::vector<std::size_t> mismatches = verify_program(*program, cfg.examples, spec.num_inputs, spec.num_outputs);
            if (!mismatches.empty()) {
                std::cerr << "Total mismatches: " << mismatches.size() << "\n";
                return 1;
            }
            log_info(cli, "SAT in " + std::to_string(elapsed) + " seconds");
            return 0;
        }
        if (result == z3::unsat) {
            log_info(cli, "UNSAT in " + std::to_string(elapsed) + " seconds");
            return 1;
        }
        log_info(cli, "UNKNOWN in " + std::to_string(elapsed) + " seconds");
        return 1;
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 2;
    }
}
