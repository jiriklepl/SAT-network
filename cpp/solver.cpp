#include "solver.hpp"

#include "datasets.hpp"
#include "encoding.hpp"
#include "logging.hpp"
#include "postprocess.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <boost/dynamic_bitset.hpp>

#include <z3++.h>

#include <cstddef>

#include <algorithm>
#include <chrono>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string check_result_name(z3::check_result result) {
    if (result == z3::sat) return "sat";
    if (result == z3::unsat) return "unsat";
    return "unknown";
}

std::string encoding_summary(const EncodingOptions &encoding) {
    std::ostringstream out;
    out << "encoding{boolean=" << (encoding.encode_boolean ? "on" : "off")
        << ", ordered=" << (encoding.force_ordered ? "on" : "off")
        << ", useful=" << (encoding.force_useful ? "on" : "off")
        << ", balanced-select=" << (encoding.balanced_select ? "on" : "off") << "}";
    return out.str();
}

z3::check_result timed_check(z3::solver &solver, ProfileData *profile, const Logger *logger,
                             const std::string &context) {
    const auto start = Clock::now();
    const z3::check_result result = solver.check();
    const double seconds = elapsed_seconds(start);
    if (profile != nullptr) {
        profile->z3_solve_seconds += seconds;
        ++profile->solver_checks;
    }
    if (logger != nullptr) {
        logger->log(3, context + " solver check returned " + check_result_name(result) + " in " +
                           std::to_string(seconds) + " seconds");
    }
    return result;
}

Program timed_extract_program(z3::context &ctx, const z3::model &model, const ProgramSpec &spec, ProfileData *profile) {
    const auto start = Clock::now();
    Program program = extract_program(ctx, model, spec);
    if (profile != nullptr) {
        profile->model_extraction_seconds += elapsed_seconds(start);
        ++profile->model_extractions;
    }
    return program;
}

Program timed_post_process_program(const Program &program, std::span<const Example> examples, int num_inputs,
                                   int num_outputs, const PostProcessOptions &options, ProfileData *profile) {
    const auto start = Clock::now();
    Program processed = post_process_program(program, examples, num_inputs, num_outputs, options, profile);
    if (profile != nullptr) {
        profile->post_processing_seconds += elapsed_seconds(start);
        ++profile->post_processing_runs;
        profile->post_processing_input_instructions += program.instrs.size();
        profile->post_processing_output_instructions += processed.instrs.size();
    }
    return processed;
}

std::vector<std::size_t> timed_verify_program(const Program &program, std::span<const Example> examples, int num_inputs,
                                              int num_outputs, ProfileData *profile) {
    const auto start = Clock::now();
    std::vector<std::size_t> mismatches = verify_program(program, examples, num_inputs, num_outputs);
    if (profile != nullptr) {
        profile->packed_verification_seconds += elapsed_seconds(start);
        profile->verification_examples += examples.size();
    }
    return mismatches;
}

z3::check_result solve_with_cegis(z3::context &ctx, z3::solver &solver, const Config &cfg, const ProgramSpec &spec,
                                  const EncodingOptions &encoding, const SolveOptions &options,
                                  std::optional<Program> &program) {
    boost::dynamic_bitset<> active(cfg.examples.size());
    const std::size_t initial_count = std::min(options.cegis_initial_size, cfg.examples.size());
    for (std::size_t idx = 0; idx < initial_count; ++idx) {
        active[idx] = true;
    }
    if (options.logger != nullptr) {
        options.logger->log(2, "CEGIS starting with " + std::to_string(initial_count) + "/" +
                                   std::to_string(cfg.examples.size()) + " examples");
    }
    add_example_constraints(ctx, solver, std::span<const Example>(cfg.examples.begin(), initial_count), "cegis0", spec,
                            encoding, options.profile);

    std::size_t iteration = 0;
    while (true) {
        const z3::check_result result =
            timed_check(solver, options.profile, options.logger, "CEGIS iteration " + std::to_string(iteration));
        if (result != z3::sat) return result;

        program = timed_extract_program(ctx, solver.get_model(), spec, options.profile);
        const std::vector<std::size_t> mismatches =
            timed_verify_program(*program, cfg.examples, spec.num_inputs, spec.num_outputs, options.profile);
        if (options.logger != nullptr) {
            options.logger->log(3, "CEGIS iteration " + std::to_string(iteration) + " verification found " +
                                       std::to_string(mismatches.size()) + " mismatches");
        }
        if (mismatches.empty()) return result;

        std::vector<Example> counterexamples;
        counterexamples.reserve(options.cegis_counterexamples);
        for (const std::size_t idx : mismatches) {
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
        add_example_constraints(ctx, solver, counterexamples, "cegis" + std::to_string(iteration), spec, encoding,
                                options.profile);
        if (options.logger != nullptr) {
            options.logger->log(2, "CEGIS added " + std::to_string(counterexamples.size()) +
                                       " counterexamples in iteration " + std::to_string(iteration));
        }
    }
}

z3::check_result solve_with_batches(z3::context &ctx, z3::solver &solver, const Config &cfg, const ProgramSpec &spec,
                                    const EncodingOptions &encoding, const SolveOptions &options) {
    z3::check_result result = z3::unknown;
    const std::size_t batch_size = options.batch_size.value_or(cfg.examples.size());
    for (std::size_t offset = 0, batch_idx = 0; offset < cfg.examples.size(); offset += batch_size, ++batch_idx) {
        const std::size_t end = std::min(offset + batch_size, cfg.examples.size());
        if (options.logger != nullptr) {
            options.logger->log(3, "Encoding batch " + std::to_string(batch_idx + 1) + " examples [" +
                                       std::to_string(offset) + ", " + std::to_string(end) + ")");
        }
        add_example_constraints(ctx, solver, std::span<const Example>(cfg.examples.begin() + offset, end - offset),
                                "b" + std::to_string(batch_idx), spec, encoding, options.profile);
        result = timed_check(solver, options.profile, options.logger, "Batch " + std::to_string(batch_idx + 1));
        if (result != z3::sat) break;
        if (options.logger != nullptr) {
            options.logger->log(2, "Processed " + std::to_string(end) + "/" + std::to_string(cfg.examples.size()) +
                                       " examples");
        }
    }
    return result;
}

void add_all_example_constraints(z3::context &ctx, z3::solver &solver, const Config &cfg, const ProgramSpec &spec,
                                 const EncodingOptions &encoding, const SolveOptions &options) {
    const std::size_t batch_size = options.batch_size.value_or(cfg.examples.size());
    for (std::size_t offset = 0, batch_idx = 0; offset < cfg.examples.size(); offset += batch_size, ++batch_idx) {
        const std::size_t end = std::min(offset + batch_size, cfg.examples.size());
        add_example_constraints(ctx, solver, std::span<const Example>(cfg.examples.begin() + offset, end - offset),
                                "b" + std::to_string(batch_idx), spec, encoding, options.profile);
    }
}

SolveStatus map_status(z3::check_result result) {
    if (result == z3::sat) return SolveStatus::Sat;
    if (result == z3::unsat) return SolveStatus::Unsat;
    return SolveStatus::Unknown;
}

}  // namespace

SolveResult solve_config(const Config &cfg, const SolveOptions &options) {
    const ProgramSpec spec{cfg.num_inputs, cfg.num_outputs, cfg.instructions};
    if (options.logger != nullptr) {
        const std::string mode = options.cegis ? "cegis" : "batched";
        const std::string batch =
            options.batch_size.has_value() ? std::to_string(*options.batch_size) : std::string("all");
        options.logger->log(2, "Solving with solver=" + options.solver + ", mode=" + mode + ", batch-size=" + batch +
                                   ", " + encoding_summary(options.encoding));
    }
    z3::context ctx;
    z3::solver solver = make_solver(ctx, options.solver);
    add_exprs(solver, build_program(ctx, spec, options.encoding, options.profile));
    add_exprs(solver, build_assumption_constraints(ctx, spec, options.assumptions));

    const auto start = std::chrono::steady_clock::now();
    std::optional<Program> program;
    const z3::check_result z3_result =
        options.cegis ? solve_with_cegis(ctx, solver, cfg, spec, options.encoding, options, program)
                      : solve_with_batches(ctx, solver, cfg, spec, options.encoding, options);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    SolveResult result;
    result.status = map_status(z3_result);
    result.elapsed_seconds = elapsed;

    if (z3_result != z3::sat) {
        return result;
    }

    if (!program.has_value()) program = timed_extract_program(ctx, solver.get_model(), spec, options.profile);
    program = canonicalize_program(*program, spec.num_inputs);
    if (options.postprocess.enabled) {
        program = timed_post_process_program(*program, cfg.examples, spec.num_inputs, spec.num_outputs,
                                             options.postprocess, options.profile);
    }
    const std::vector<std::size_t> mismatches =
        timed_verify_program(*program, cfg.examples, spec.num_inputs, spec.num_outputs, options.profile);
    if (options.logger != nullptr) {
        options.logger->log(3, "Final verification found " + std::to_string(mismatches.size()) + " mismatches");
    }
    result.program = std::move(program);
    result.mismatch_count = mismatches.size();
    if (!mismatches.empty()) {
        result.status = SolveStatus::VerificationFailed;
    }
    return result;
}

std::string make_smt2(const Config &cfg, const SolveOptions &options) {
    const ProgramSpec spec{
        .num_inputs = cfg.num_inputs, .num_outputs = cfg.num_outputs, .program_length = cfg.instructions};
    z3::context ctx;
    z3::solver solver = make_solver(ctx, options.solver);
    add_exprs(solver, build_program(ctx, spec, options.encoding, options.profile));
    add_exprs(solver, build_assumption_constraints(ctx, spec, options.assumptions));
    add_all_example_constraints(ctx, solver, cfg, spec, options.encoding, options);
    return solver.to_smt2();
}

std::string make_dimacs(const Config &cfg, const SolveOptions &options) {
    const ProgramSpec spec{
        .num_inputs = cfg.num_inputs, .num_outputs = cfg.num_outputs, .program_length = cfg.instructions};
    z3::context ctx;
    z3::solver solver = make_solver(ctx, options.solver);
    add_exprs(solver, build_program(ctx, spec, options.encoding, options.profile));
    add_exprs(solver, build_assumption_constraints(ctx, spec, options.assumptions));
    add_all_example_constraints(ctx, solver, cfg, spec, options.encoding, options);

    z3::goal goal(ctx);
    goal.add(solver.assertions());
    const z3::tactic tactic = z3::tactic(ctx, "simplify") & z3::tactic(ctx, "propagate-values") &
                              z3::tactic(ctx, "bit-blast") & z3::tactic(ctx, "tseitin-cnf");
    const z3::apply_result result = tactic(goal);
    if (result.size() != 1) {
        throw std::runtime_error("DIMACS conversion produced multiple subgoals");
    }
    return result[0].dimacs();
}
