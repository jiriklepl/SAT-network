#include "datasets.hpp"
#include "program.hpp"
#include "quantified_spec.hpp"
#include "solver.hpp"
#include "symbolic_verifier.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <cstddef>

#include <chrono>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

DatasetLoadOptions force_symbolic() {
    DatasetLoadOptions options;
    options.max_materialized_examples = 0;
    options.basis_size = 2;
    return options;
}

nlohmann::json xor_spec(int instructions) {
    return nlohmann::json{
        {"instructions", instructions},
        {"num_inputs", 2},
        {"num_outputs", 1},
        {"forall", nlohmann::json::array({
                       nlohmann::json{{"var", "a"}, {"range", {0, 2}}},
                       nlohmann::json{{"var", "b"}, {"range", {0, 2}}},
                   })},
        {"inputs", {"bit(a)", "bit(b)"}},
        {"outputs", {"bit(a ^ b)"}},
    };
}

Program random_program(std::mt19937 &rng, int num_inputs, int num_outputs, int length) {
    Program program;
    for (int instr = 0; instr < length; ++instr) {
        const int sources = temp_source(instr, num_inputs);
        std::uniform_int_distribution<int> source(0, sources - 1);
        program.instrs.push_back({std::uniform_int_distribution<int>(0, 2)(rng), source(rng), source(rng)});
    }
    std::uniform_int_distribution<int> output(0, temp_source(length, num_inputs) - 1);
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) program.outputs.push_back(output(rng));
    return program;
}

}  // namespace

TEST_CASE("huge quantified domains are not unrolled") {
    const auto start = std::chrono::steady_clock::now();
    for (const char *name : {"excitable", "cyclic"}) {
        const Config cfg = build_dataset_from_config(default_dataset_config(name));
        REQUIRE_FALSE(cfg.fully_materialized);
        REQUIRE(cfg.quantified != nullptr);
        REQUIRE(cfg.domain_size > kDefaultMaxMaterializedExamples);
        REQUIRE(cfg.examples.size() <= kDefaultBasisSize);
        REQUIRE_FALSE(cfg.examples.empty());
    }
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));

    const Config cyclic = build_dataset_from_config(default_dataset_config("cyclic"));
    REQUIRE(cyclic.domain_size == 35184372088832ULL);  // 32^9
}

TEST_CASE("small quantified domains stay materialized") {
    const Config cfg = build_dataset_from_config(default_dataset_config("life"));
    REQUIRE(cfg.fully_materialized);
    REQUIRE(cfg.examples.size() == 512);
    REQUIRE(cfg.domain_size == 512);
}

TEST_CASE("seeded basis is deterministic and consists of valid rows") {
    DatasetLoadOptions options;
    options.max_materialized_examples = 0;
    options.basis_size = 32;
    options.seed = 7;
    const Config first = build_dataset_from_config(default_dataset_config("excitable"), options);
    const Config second = build_dataset_from_config(default_dataset_config("excitable"), options);
    REQUIRE(first.examples.size() == 32);
    REQUIRE(first.examples.size() == second.examples.size());
    for (std::size_t idx = 0; idx < first.examples.size(); ++idx) {
        REQUIRE(first.examples[idx].inputs == second.examples[idx].inputs);
        REQUIRE(first.examples[idx].output_values == second.examples[idx].output_values);
    }
}

TEST_CASE("symbolic verifier agrees with native verification on random programs") {
    std::mt19937 rng(12345);
    for (const char *name : {"adder", "gol", "gol1", "gol2", "sloppy-adder", "sloppy-adder3", "life", "life-compressed",
                             "maze", "brian", "fire", "wire-compressed", "critters", "traffic"}) {
        INFO(name);
        const Config exact = build_dataset_from_config(default_dataset_config(name));
        REQUIRE(exact.fully_materialized);
        SymbolicVerifier verifier(exact.quantified);
        for (int round = 0; round < 12; ++round) {
            const Program program = random_program(rng, exact.num_inputs, exact.num_outputs, 6);
            const bool wrong =
                !verify_program(program, exact.examples, exact.num_inputs, exact.num_outputs).empty();
            const std::vector<Example> counterexamples = verifier.find_counterexamples(program, 3);
            REQUIRE(wrong == !counterexamples.empty());
            REQUIRE(counterexamples.size() <= 3);
            for (const Example &example : counterexamples) {
                REQUIRE_FALSE(
                    verify_program(program, std::span<const Example>(&example, 1), exact.num_inputs, exact.num_outputs)
                        .empty());
            }
        }
    }
}

TEST_CASE("symbolic verifier handles the full excitable and wire domains") {
    for (const char *name : {"excitable", "wire", "cyclic"}) {
        INFO(name);
        const Config cfg = build_dataset_from_config(default_dataset_config(name), force_symbolic());
        REQUIRE_FALSE(cfg.fully_materialized);
        SymbolicVerifier verifier(cfg.quantified);
        std::mt19937 rng(99);
        const Program program = random_program(rng, cfg.num_inputs, cfg.num_outputs, 4);
        const std::vector<Example> counterexamples = verifier.find_counterexamples(program, 2);
        REQUIRE_FALSE(counterexamples.empty());
        REQUIRE(counterexamples.front().input_count() == static_cast<std::size_t>(cfg.num_inputs));
    }
}

TEST_CASE("symbolic CEGIS solves a quantified spec without unrolling it") {
    const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
    REQUIRE_FALSE(cfg.fully_materialized);
    REQUIRE(cfg.examples.size() <= 2);

    SolveOptions options;
    const SolveResult result = solve_config(cfg, options);
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(result.mismatch_count == 0);
    REQUIRE(result.program.has_value());

    SymbolicVerifier verifier(cfg.quantified);
    REQUIRE(verifier.find_counterexamples(*result.program).empty());
}

TEST_CASE("symbolic CEGIS reports unsatisfiable quantified specs") {
    const Config cfg = build_dataset_from_config(xor_spec(0), force_symbolic());
    const SolveResult result = solve_config(cfg, SolveOptions{});
    REQUIRE(result.status == SolveStatus::Unsat);
}

TEST_CASE("symbolic CEGIS honours null don't-care outputs") {
    nlohmann::json spec = {
        {"instructions", 0},
        {"num_inputs", 2},
        {"num_outputs", 1},
        {"forall", nlohmann::json::array({
                       nlohmann::json{{"var", "a"}, {"range", {0, 2}}},
                       nlohmann::json{{"var", "b"}, {"range", {0, 2}}},
                   })},
        {"inputs", {"bit(a)", "bit(b)"}},
        {"outputs", {"a == 0 ? bit(b) : null"}},
    };
    const Config cfg = build_dataset_from_config(spec, force_symbolic());
    const SolveResult result = solve_config(cfg, SolveOptions{});
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(result.program->outputs.front() == input_source(1));
}

TEST_CASE("post-processing under a symbolic spec only returns proven programs") {
    const Config cfg = build_dataset_from_config(xor_spec(3), force_symbolic());
    SolveOptions options;
    options.postprocess.enabled = true;
    options.postprocess.beam_width = 2;
    const SolveResult result = solve_config(cfg, options);
    REQUIRE(result.status == SolveStatus::Sat);
    REQUIRE(result.program->instrs.size() == 1);

    SymbolicVerifier verifier(cfg.quantified);
    REQUIRE(verifier.find_counterexamples(*result.program).empty());
}

TEST_CASE("PackedMask push_back and PackedExamples append_example match full packing across word boundaries") {
    const Config exact = build_dataset_from_config(default_dataset_config("life"));
    REQUIRE(exact.examples.size() == 512);
    const PackedExamples full = pack_examples(exact.examples, exact.num_inputs, exact.num_outputs);

    PackedExamples incremental =
        pack_examples(std::span<const Example>(exact.examples.begin(), 1), exact.num_inputs, exact.num_outputs);
    for (std::size_t idx = 1; idx < exact.examples.size(); ++idx) {
        incremental.append_example(exact.examples[idx]);
    }
    REQUIRE(incremental.width == full.width);
    REQUIRE(incremental.input_masks == full.input_masks);
    REQUIRE(incremental.output_values == full.output_values);
    REQUIRE(incremental.output_dont_care_masks == full.output_dont_care_masks);

    const PackedExamples aligned_slice = incremental.slice_words(64, 70);
    const PackedExamples expected_aligned =
        pack_examples(std::span<const Example>(exact.examples.begin() + 64, 70), exact.num_inputs, exact.num_outputs);
    REQUIRE(aligned_slice.input_masks == expected_aligned.input_masks);
    REQUIRE(aligned_slice.output_values == expected_aligned.output_values);

    const PackedExamples unaligned_slice = incremental.slice_words(13, 80);
    const PackedExamples expected_unaligned =
        pack_examples(std::span<const Example>(exact.examples.begin() + 13, 80), exact.num_inputs, exact.num_outputs);
    REQUIRE(unaligned_slice.input_masks == expected_unaligned.input_masks);
    REQUIRE(unaligned_slice.output_values == expected_unaligned.output_values);
}

TEST_CASE("end-to-end symbolic synthesis and post-processing match full truth tables on built-in datasets") {
    DatasetLoadOptions sym_opts;
    sym_opts.max_materialized_examples = 2;
    sym_opts.basis_size = 2;
    sym_opts.seed = 42;

    nlohmann::json adder = default_dataset_config("adder");
    adder["instructions"] = 6;  // slightly non-minimal to exercise post-processing

    nlohmann::json sloppy = default_dataset_config("sloppy-adder");
    sloppy["instructions"] = 8;

    nlohmann::json gol1 = default_dataset_config("gol1");

    for (const nlohmann::json &spec_cfg : {adder, sloppy, gol1}) {
        const Config exact = build_dataset_from_config(spec_cfg);
        REQUIRE(exact.fully_materialized);

        const Config symbolic = build_dataset_from_config(spec_cfg, sym_opts);
        REQUIRE_FALSE(symbolic.fully_materialized);
        REQUIRE(symbolic.examples.size() == 2);

        SolveOptions options;
        options.postprocess.enabled = true;
        options.postprocess.beam_width = 2;
        const SolveResult result = solve_config(symbolic, options);
        REQUIRE(result.status == SolveStatus::Sat);
        REQUIRE(result.mismatch_count == 0);
        REQUIRE(result.program.has_value());
        REQUIRE(verify_program(*result.program, exact.examples, exact.num_inputs, exact.num_outputs).empty());
    }
}

TEST_CASE("local window SAT resynthesis refines against symbolic counterexamples") {
    // Spec: OUT0 = I0 ^ I1 ^ I2 (3-input parity)
    nlohmann::json spec = {
        {"instructions", 4},
        {"num_inputs", 3},
        {"num_outputs", 1},
        {"forall", nlohmann::json::array({
                       nlohmann::json{{"var", "a"}, {"range", {0, 2}}},
                       nlohmann::json{{"var", "b"}, {"range", {0, 2}}},
                       nlohmann::json{{"var", "c"}, {"range", {0, 2}}},
                   })},
        {"inputs", {"bit(a)", "bit(b)", "bit(c)"}},
        {"outputs", {"bit(a ^ b ^ c)"}},
    };
    DatasetLoadOptions sym_opts;
    sym_opts.max_materialized_examples = 0;
    sym_opts.basis_size = 1;
    const Config symbolic = build_dataset_from_config(spec, sym_opts);
    const Config exact = build_dataset_from_config(spec);

    // Non-minimal 4-instruction program computing I0 ^ I1 ^ I2 with a redundant window:
    // T0 = XOR(I0, I1), T1 = XOR(T0, 1), T2 = XOR(T1, 1), T3 = XOR(T2, I2), OUT0 = T3
    Program bloated;
    bloated.instrs = {
        {kOpXor, input_source(0), input_source(1)},
        {kOpXor, temp_source(0, 3), kSourceConstantOne},
        {kOpXor, temp_source(1, 3), kSourceConstantOne},
        {kOpXor, temp_source(2, 3), input_source(2)},
    };
    bloated.outputs = {temp_source(3, 3)};

    SymbolicVerifier verifier(symbolic.quantified);
    PostProcessOptions pp_opts;
    pp_opts.enabled = true;
    pp_opts.resynthesis_maxnodes = 4;
    pp_opts.verifier = &verifier;

    const Program simplified =
        post_process_program(bloated, symbolic.examples, symbolic.num_inputs, symbolic.num_outputs, pp_opts);
    REQUIRE(simplified.instrs.size() == 2);
    REQUIRE(verify_program(simplified, exact.examples, exact.num_inputs, exact.num_outputs).empty());
}

TEST_CASE("symbolic datasets reject operations that need every row") {
    const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
    REQUIRE_THROWS(make_smt2(cfg, SolveOptions{}));
    REQUIRE_THROWS(make_dimacs(cfg, SolveOptions{}));
    SolveOptions do_all;
    do_all.do_all = true;
    REQUIRE_THROWS(solve_config(cfg, do_all));
}

