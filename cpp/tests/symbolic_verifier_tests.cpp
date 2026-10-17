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

TEST_CASE("symbolic lowering matches native evaluation across all scalar and boolean operators") {
    // Exercises every BinaryOp, UnaryOp, Ternary, Popcount, Count, TupleIndex (static & dynamic),
    // Quantifier::Range, Quantifier::Values, Quantifier::Tuple, and where filtering in both
    // let_row (Lowering::to_int) and outputs (Lowering::to_bool / Lowering::emit).
    nlohmann::json spec = {
        {"instructions", 6},
        {"let", {{"w", "required_bits(4)"}}},
        {"num_inputs", "2 + w + 6"},
        {"num_outputs", 18},
        {"forall",
         nlohmann::json::array({
             nlohmann::json{{"var", "a"}, {"range", {0, 4}}},
             nlohmann::json{{"var", "b"}, {"values", {0, 1, 2, 3}}},
             nlohmann::json{{"var", "cells"}, {"tuple", {{"size", 3}, {"range", {0, 4}}}}},
         })},
        {"let_row",
         {
             // Relational operators evaluated as integers via Lowering::to_int:
             {"r_eq", "a == b"},
             {"r_ne", "a != b"},
             {"r_lt", "a < b"},
             {"r_le", "a <= b"},
             {"r_gt", "a > b"},
             {"r_ge", "a >= b"},
             // Logical operators evaluated as integers via Lowering::to_int:
             {"r_and", "(a == 1) && (b == 2)"},
             {"r_or", "(a == 0) || (b == 3)"},
             {"r_not", "!(a == b)"},
             // Arithmetic, bitwise, shift, unary operators in Lowering::to_int:
             {"r_arith", "((a + b) * 3 - (a / (b + 1))) % (b + 2)"},
             {"r_neg", "(-a) + (~b)"},
             {"r_shift", "((a + 1) << (b & 1)) + ((a - 3) >> 1)"},
             {"r_bitops", "(a & b) ^ (a | b)"},
             // Functions and tuple operations in Lowering::to_int:
             {"r_pop", "popcount(a ^ b) + popcount(~a)"},
             {"r_cnt_all", "count(cells, a)"},
             {"r_cnt_tail", "count(cells, b, 1)"},
             {"r_dyn_cell", "cells[a % 3]"},
             {"r_tern", "r_eq ? (r_arith + r_dyn_cell) : (r_pop + r_cnt_all + r_cnt_tail)"},
             {"r_bit_fn", "bit(a == 2)"},
         }},
        {"where", "!(a == 3 && b == 3) && cells[0] <= 2"},
        {"inputs", {"bits(a, 2)", "bits(b, w)", "state_bits(cells, 2)"}},
        {"outputs",
         {
             "bit(r_eq)",
             "bit(r_ne)",
             "bit(r_lt)",
             "bit(r_le)",
             "bit(r_gt)",
             "bit(r_ge)",
             "bit(r_and)",
             "bit(r_or)",
             "bit(r_not)",
             "bits(r_arith, 2)",
             "bit((r_neg & 1) != 0)",
             "bit((r_shift & 1) != 0)",
             "bit((r_bitops & 1) != 0)",
             "bit((r_tern & 1) != 0)",
             "bit(r_bit_fn)",
             "a == 0 ? null : bit(cells[0] == cells[2])",
             "true ? bit(false || (a + b >= 3)) : bit(true)",
         }},
    };

    const Config exact = build_dataset_from_config(spec);
    REQUIRE(exact.fully_materialized);
    REQUIRE(exact.examples.size() == (4 * 4 - 1) * 3 * 4 * 4);

    SymbolicVerifier verifier(exact.quantified);
    std::mt19937 rng(20261017);
    for (int round = 0; round < 20; ++round) {
        const Program program = random_program(rng, exact.num_inputs, exact.num_outputs, 6);
        const bool has_mismatch =
            !verify_program(program, exact.examples, exact.num_inputs, exact.num_outputs).empty();
        const std::vector<Example> counterexamples = verifier.find_counterexamples(program, 4);
        REQUIRE(has_mismatch == !counterexamples.empty());
        REQUIRE(counterexamples.size() <= 4);
        for (const Example &cex : counterexamples) {
            REQUIRE_FALSE(
                verify_program(program, std::span<const Example>(&cex, 1), exact.num_inputs, exact.num_outputs)
                    .empty());
        }
    }
}

TEST_CASE("symbolic verifier handles empty quantifier domains") {
    {
        const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
        auto empty_range = std::const_pointer_cast<QuantifiedSpec>(cfg.quantified);
        empty_range->quantifiers[0].spec = Quantifier::Range{.slot = 0, .start = 3, .end = 3};
        SymbolicVerifier range_verifier(empty_range);
        Program p{.instrs = {}, .outputs = {kSourceConstantOne}};
        REQUIRE(range_verifier.find_counterexamples(p).empty());
    }
    {
        const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
        auto empty_values = std::const_pointer_cast<QuantifiedSpec>(cfg.quantified);
        empty_values->quantifiers[0].spec = Quantifier::Values{.slot = 0, .values = {}};
        SymbolicVerifier values_verifier(empty_values);
        Program p{.instrs = {}, .outputs = {kSourceConstantOne}};
        REQUIRE(values_verifier.find_counterexamples(p).empty());
    }
}

TEST_CASE("symbolic verifier rejects out-of-range and invalid specifications") {
    REQUIRE_THROWS_AS(SymbolicVerifier(nullptr), std::invalid_argument);

    {
        const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
        auto neg_range = std::const_pointer_cast<QuantifiedSpec>(cfg.quantified);
        neg_range->quantifiers[0].spec = Quantifier::Range{.slot = 0, .start = -1, .end = 2};
        REQUIRE_THROWS(SymbolicVerifier(neg_range));
    }
    {
        const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
        auto huge_range = std::const_pointer_cast<QuantifiedSpec>(cfg.quantified);
        huge_range->quantifiers[0].spec = Quantifier::Range{.slot = 0, .start = 0, .end = (std::int64_t{1} << 30) + 1};
        REQUIRE_THROWS(SymbolicVerifier(huge_range));
    }
    {
        const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
        auto neg_values = std::const_pointer_cast<QuantifiedSpec>(cfg.quantified);
        neg_values->quantifiers[0].spec = Quantifier::Values{.slot = 0, .values = {0, -1}};
        REQUIRE_THROWS(SymbolicVerifier(neg_values));
    }
    const Config base = build_dataset_from_config(xor_spec(1), force_symbolic());

    // Dynamic required_bits() must be rejected by SymbolicVerifier
    {
        nlohmann::json bad_req_bits = {
            {"instructions", 1},
            {"num_inputs", 1},
            {"num_outputs", 1},
            {"forall", nlohmann::json::array({nlohmann::json{{"var", "a"}, {"range", {1, 3}}}})},
            {"inputs", {"bit(a)"}},
            {"outputs", {"bit(required_bits(a))"}},
        };
        const Config cfg = build_dataset_from_config(bad_req_bits, force_symbolic());
        REQUIRE_THROWS(SymbolicVerifier(cfg.quantified));
    }

    // Dynamic count() start index must be rejected by SymbolicVerifier
    {
        nlohmann::json bad_count_start = {
            {"instructions", 1},
            {"num_inputs", 2},
            {"num_outputs", 1},
            {"forall",
             nlohmann::json::array({
                 nlohmann::json{{"var", "a"}, {"range", {0, 2}}},
                 nlohmann::json{{"var", "cells"}, {"tuple", {{"size", 2}, {"range", {0, 2}}}}},
             })},
            {"inputs", {"state_bits(cells, 1)"}},
            {"outputs", {"bit(count(cells, 1, a))"}},
        };
        const Config cfg = build_dataset_from_config(bad_count_start, force_symbolic());
        REQUIRE_THROWS(SymbolicVerifier(cfg.quantified));
    }

    // Dynamic bits() width must be rejected by SymbolicVerifier
    {
        nlohmann::json bad_bits_width = {
            {"instructions", 1},
            {"num_inputs", 1},
            {"num_outputs", 1},
            {"forall", nlohmann::json::array({nlohmann::json{{"var", "a"}, {"range", {1, 2}}}})},
            {"inputs", {"bit(a)"}},
            {"outputs", {"bits(a, a)"}},
        };
        const Config cfg = build_dataset_from_config(bad_bits_width, force_symbolic());
        REQUIRE_THROWS(SymbolicVerifier(cfg.quantified));
    }

    // Mismatched ternary emitter branch widths must be rejected by SymbolicVerifier
    {
        nlohmann::json bad_ternary_emit = {
            {"instructions", 1},
            {"num_inputs", 1},
            {"num_outputs", 1},
            {"forall", nlohmann::json::array({nlohmann::json{{"var", "a"}, {"range", {0, 2}}}})},
            {"inputs", {"bit(a)"}},
            {"outputs", {"a == 0 ? bits(a, 1) : bits(a, 2)"}},
        };
        // force_symbolic with basis_size=1 where corner a=0 has width 1
        DatasetLoadOptions opts;
        opts.max_materialized_examples = 0;
        opts.basis_size = 1;
        const Config cfg = build_dataset_from_config(bad_ternary_emit, opts);
        REQUIRE_THROWS(SymbolicVerifier(cfg.quantified));
    }

    SymbolicVerifier verifier(base.quantified);
    // Wrong output count
    REQUIRE_THROWS_AS(verifier.find_counterexamples(Program{.instrs = {}, .outputs = {}}), std::invalid_argument);
    // Out-of-bounds source wire
    REQUIRE_THROWS_AS(verifier.find_counterexamples(Program{.instrs = {}, .outputs = {99}}), std::invalid_argument);
    // Unknown opcode
    REQUIRE_THROWS_AS(
        verifier.find_counterexamples(Program{.instrs = {{99, 1, 2}}, .outputs = {3}}), std::invalid_argument);
}

TEST_CASE("VerificationOracle caches verified keys, tracks basis generation, and handles timeouts") {
    const Config cfg = build_dataset_from_config(xor_spec(1), force_symbolic());
    SymbolicVerifier verifier(cfg.quantified);
    ProfileData profile;

    std::vector<Example> init_basis = {cfg.examples.front()};
    VerificationOracle oracle(init_basis, cfg.num_inputs, cfg.num_outputs, &verifier, &profile);
    REQUIRE(oracle.basis_generation() == 0);
    REQUIRE(oracle.examples().size() == 1);

    // Wrong program: OUT0 = I0 (matches (0,0)->0 but fails on (0,1)->1)
    const Program wrong_prog{.instrs = {}, .outputs = {input_source(0)}};
    REQUIRE_FALSE(oracle.verify_candidate(wrong_prog, "wrong_key", std::nullopt, 2));
    REQUIRE(oracle.basis_generation() == 1);
    REQUIRE(oracle.examples().size() > 1);
    REQUIRE(profile.symbolic_verifier_checks == 1);

    // Correct program: T0 = XOR(I0, I1), OUT0 = T0
    const Program right_prog{.instrs = {{kOpXor, input_source(0), input_source(1)}}, .outputs = {temp_source(0, 2)}};
    // Timeout <= 0 before verification causes conservative rejection without a Z3 check
    bool timed_out = false;
    REQUIRE(verifier.find_counterexamples(right_prog, 1, 0, &profile, &timed_out).empty());
    REQUIRE(timed_out);
    REQUIRE_FALSE(oracle.verify_candidate(right_prog, "right_key", 0, 1));
    REQUIRE(profile.symbolic_verifier_checks == 1);

    // Normal verification succeeds and caches "right_key"
    REQUIRE(oracle.verify_candidate(right_prog, "right_key", std::nullopt, 1));
    REQUIRE(profile.symbolic_verifier_checks == 2);

    // Second call with the same canonical key hits the cache without running Z3
    REQUIRE(oracle.verify_candidate(right_prog, "right_key", std::nullopt, 1));
    REQUIRE(profile.symbolic_verifier_checks == 2);

    // Pre-marking a key also skips Z3 verification
    oracle.mark_verified("pre_marked_key");
    REQUIRE(oracle.verify_candidate(right_prog, "pre_marked_key", std::nullopt, 1));
    REQUIRE(profile.symbolic_verifier_checks == 2);
}
