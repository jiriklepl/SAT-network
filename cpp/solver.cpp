#include "solver.hpp"

#include "encoding.hpp"

#include <z3++.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

z3::check_result solve_with_cegis(
    z3::context &ctx,
    z3::solver &solver,
    const Config &cfg,
    const ProgramSpec &spec,
    const EncodingOptions &encoding,
    const SolveOptions &options,
    std::optional<Program> &program
) {
    std::vector<bool> active(cfg.examples.size(), false);
    std::vector<Example> initial;
    std::size_t initial_count = std::min(options.cegis_initial_size, cfg.examples.size());
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
            if (counterexamples.size() >= options.cegis_counterexamples) break;
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
    const SolveOptions &options
) {
    z3::check_result result = z3::unknown;
    std::size_t batch_size = options.batch_size.value_or(cfg.examples.size());
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

void add_all_example_constraints(
    z3::context &ctx,
    z3::solver &solver,
    const Config &cfg,
    const ProgramSpec &spec,
    const EncodingOptions &encoding,
    const SolveOptions &options
) {
    std::size_t batch_size = options.batch_size.value_or(cfg.examples.size());
    for (std::size_t offset = 0, batch_idx = 0; offset < cfg.examples.size(); offset += batch_size, ++batch_idx) {
        std::size_t end = std::min(offset + batch_size, cfg.examples.size());
        std::vector<Example> batch(cfg.examples.begin() + static_cast<std::ptrdiff_t>(offset),
                                   cfg.examples.begin() + static_cast<std::ptrdiff_t>(end));
        add_example_constraints(ctx, solver, batch, "b" + std::to_string(batch_idx), spec, encoding);
    }
}

SolveStatus map_status(z3::check_result result) {
    if (result == z3::sat) return SolveStatus::Sat;
    if (result == z3::unsat) return SolveStatus::Unsat;
    return SolveStatus::Unknown;
}

}  // namespace

SolveResult solve_config(const Config &cfg, const SolveOptions &options) {
    ProgramSpec spec{cfg.num_inputs, cfg.num_outputs, cfg.instructions};
    z3::context ctx;
    z3::solver solver = make_solver(ctx, options.solver);
    add_exprs(solver, build_program(ctx, spec, options.encoding));
    add_exprs(solver, build_assumption_constraints(ctx, spec, options.assumptions));

    auto start = std::chrono::steady_clock::now();
    std::optional<Program> program;
    z3::check_result z3_result = options.cegis
        ? solve_with_cegis(ctx, solver, cfg, spec, options.encoding, options, program)
        : solve_with_batches(ctx, solver, cfg, spec, options.encoding, options);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    SolveResult result;
    result.status = map_status(z3_result);
    result.elapsed_seconds = elapsed;

    if (z3_result != z3::sat) {
        return result;
    }

    if (!program.has_value()) program = extract_program(ctx, solver.get_model(), spec);
    std::vector<std::size_t> mismatches = verify_program(*program, cfg.examples, spec.num_inputs, spec.num_outputs);
    result.program = std::move(program);
    result.mismatch_count = mismatches.size();
    if (!mismatches.empty()) {
        result.status = SolveStatus::VerificationFailed;
    }
    return result;
}

std::string make_smt2(const Config &cfg, const SolveOptions &options) {
    ProgramSpec spec{cfg.num_inputs, cfg.num_outputs, cfg.instructions};
    z3::context ctx;
    z3::solver solver = make_solver(ctx, options.solver);
    add_exprs(solver, build_program(ctx, spec, options.encoding));
    add_exprs(solver, build_assumption_constraints(ctx, spec, options.assumptions));
    add_all_example_constraints(ctx, solver, cfg, spec, options.encoding, options);
    return solver.to_smt2();
}
