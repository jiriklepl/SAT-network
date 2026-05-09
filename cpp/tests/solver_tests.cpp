#include "datasets.hpp"
#include "solver.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

Config xor_config(int instructions) {
    Config cfg;
    cfg.num_inputs = 2;
    cfg.num_outputs = 1;
    cfg.instructions = instructions;
    cfg.examples = {
        {{false, false}, {false}},
        {{false, true}, {true}},
        {{true, false}, {true}},
        {{true, true}, {false}},
    };
    return cfg;
}

Config wide_identity_config() {
    Config cfg;
    cfg.num_inputs = 1;
    cfg.num_outputs = 1;
    cfg.instructions = 0;
    for (std::size_t idx = 0; idx < 130; ++idx) {
        const bool bit = (idx % 2) != 0;
        cfg.examples.push_back({{bit}, {bit}});
    }
    return cfg;
}

}  // namespace

TEST_CASE("solver reports SAT for CEGIS and batched solving") {
    SolveOptions cegis;
    cegis.cegis = true;
    cegis.cegis_initial_size = 1;
    cegis.cegis_counterexamples = 1;
    SolveResult cegis_result = solve_config(xor_config(1), cegis);
    REQUIRE(cegis_result.status == SolveStatus::Sat);
    REQUIRE(cegis_result.program.has_value());
    REQUIRE(cegis_result.mismatch_count == 0);
    REQUIRE(cegis_result.elapsed_seconds >= 0.0);

    SolveOptions batched;
    batched.batch_size = 2;
    SolveResult batched_result = solve_config(xor_config(1), batched);
    REQUIRE(batched_result.status == SolveStatus::Sat);
    REQUIRE(batched_result.program.has_value());
    REQUIRE(batched_result.mismatch_count == 0);
}

TEST_CASE("solver reports UNSAT without a program") {
    SolveResult unsat = solve_config(xor_config(0), SolveOptions{});
    REQUIRE(unsat.status == SolveStatus::Unsat);
    REQUIRE_FALSE(unsat.program.has_value());
    REQUIRE(unsat.mismatch_count == 0);
}

TEST_CASE("solver propagates unsupported solver errors") {
    SolveOptions options;
    options.solver = "missing-solver";
    REQUIRE_THROWS(solve_config(xor_config(1), options));
}

TEST_CASE("solver supports alternate encoding flags") {
    SolveOptions options;
    options.encoding.encode_boolean = true;
    options.encoding.balanced_select = true;
    options.encoding.force_useful = true;
    SolveResult result = solve_config(xor_config(1), options);
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(result.program.has_value());
    REQUIRE(result.mismatch_count == 0);
}

TEST_CASE("solver encodes batches wider than one mask word") {
    SolveResult result = solve_config(wide_identity_config(), SolveOptions{});
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(result.program.has_value());
    REQUIRE(result.program->outputs == std::vector<int>{1});
    REQUIRE(result.mismatch_count == 0);
}

TEST_CASE("solver records profiling counters and phase timings") {
    ProfileData profile;
    SolveOptions options;
    options.batch_size = 2;
    options.profile = &profile;
    SolveResult result = solve_config(xor_config(1), options);
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(profile.structure_constraints > 0);
    REQUIRE(profile.example_constraints > 0);
    REQUIRE(profile.example_batches == 2);
    REQUIRE(profile.packed_examples == 4);
    REQUIRE(profile.solver_checks >= 1);
    REQUIRE(profile.model_extractions == 1);
    REQUIRE(profile.verification_examples == 4);
    REQUIRE(profile.structure_encoding_seconds >= 0.0);
    REQUIRE(profile.example_packing_seconds >= 0.0);
    REQUIRE(profile.example_encoding_seconds >= 0.0);
    REQUIRE(profile.z3_solve_seconds >= 0.0);
    REQUIRE(profile.model_extraction_seconds >= 0.0);
    REQUIRE(profile.post_processing_seconds >= 0.0);
    REQUIRE(profile.post_processing_runs == 0);
    REQUIRE(profile.packed_verification_seconds >= 0.0);
    REQUIRE(profile.bv_cache_hits > 0);
    REQUIRE(profile.bv_cache_misses > 0);
}

TEST_CASE("solver profiles post-processing when enabled") {
    Config cfg;
    cfg.num_inputs = 2;
    cfg.num_outputs = 1;
    cfg.instructions = 2;
    cfg.examples = {
        {{false, false}, {false}},
        {{false, true}, {true}},
        {{true, false}, {true}},
        {{true, true}, {false}},
    };

    ProfileData profile;
    SolveOptions options;
    options.profile = &profile;
    options.assumptions.instructions = {
        InstructionAssumption{0, 1, 1, 2},
        InstructionAssumption{1, 0, 0, 3},
    };
    options.assumptions.outputs = {OutputAssumption{0, 4}};
    options.postprocess.enabled = true;

    SolveResult result = solve_config(cfg, options);
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(profile.post_processing_runs == 1);
    REQUIRE(profile.post_processing_input_instructions == 2);
    REQUIRE(profile.post_processing_output_instructions < profile.post_processing_input_instructions);
    REQUIRE(profile.post_processing_seconds >= 0.0);
}

TEST_CASE("solver applies assumptions before solving") {
    SolveOptions options;
    options.assumptions.instructions.push_back({0, 1, 1, 2});
    options.assumptions.outputs.push_back({0, 3});
    SolveResult result = solve_config(xor_config(1), options);
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(result.program.has_value());
    REQUIRE(result.program->instrs[0].op == 1);
    REQUIRE(result.program->outputs[0] == 3);

    SolveOptions contradictory;
    contradictory.assumptions.instructions.push_back({0, 0, 1, 1});
    SolveResult unsat = solve_config(xor_config(1), contradictory);
    REQUIRE(unsat.status == SolveStatus::Unsat);
}

TEST_CASE("solver exports SMT2 with structure, assumptions, and examples") {
    SolveOptions options;
    options.batch_size = 2;
    options.assumptions.instructions.push_back({0, 1, 1, 2});
    options.assumptions.outputs.push_back({0, 3});

    std::string smt2 = make_smt2(xor_config(1), options);
    REQUIRE(smt2.find("OP_0") != std::string::npos);
    REQUIRE(smt2.find("S1_0") != std::string::npos);
    REQUIRE(smt2.find("OUT_0_idx") != std::string::npos);
    REQUIRE(smt2.find("b0") != std::string::npos);
    REQUIRE(smt2.find("b1") != std::string::npos);
    REQUIRE(smt2.find("assert") != std::string::npos);
}

TEST_CASE("solver exports DIMACS CNF") {
    SolveOptions options;
    options.batch_size = 2;
    std::string dimacs = make_dimacs(xor_config(1), options);
    REQUIRE(dimacs.find("p cnf ") != std::string::npos);
    REQUIRE(dimacs.find("c ") != std::string::npos);
}
