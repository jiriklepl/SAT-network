#include "postprocess.hpp"

#include "datasets.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

PostProcessOptions enabled_options() {
    return {.enabled = true};
}

std::vector<Example> identity_examples() {
    return {
        {{false}, {false}},
        {{true}, {true}},
    };
}

std::vector<Example> xor_examples() {
    return {
        {{false, false}, {false}},
        {{false, true}, {true}},
        {{true, false}, {true}},
        {{true, true}, {false}},
    };
}

void require_equivalent(const Program &program, std::span<const Example> examples, int num_inputs, int num_outputs) {
    REQUIRE(verify_program(program, examples, num_inputs, num_outputs).empty());
}

std::vector<Example> all_examples(int num_inputs, bool (*output_fn)(const std::vector<int> &)) {
    std::vector<Example> examples;
    const std::size_t count = std::size_t{1} << static_cast<unsigned>(num_inputs);
    examples.reserve(count);
    for (std::size_t bits = 0; bits < count; ++bits) {
        std::vector<int> inputs(static_cast<std::size_t>(num_inputs), 0);
        for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
            inputs[static_cast<std::size_t>(input_idx)] =
                static_cast<int>((bits >> static_cast<unsigned>(input_idx)) & 1U);
        }
        Example example;
        for (const int input : inputs) {
            example.push_input(input != 0);
        }
        example.push_output(output_fn(inputs));
        examples.push_back(std::move(example));
    }
    return examples;
}

}  // namespace

TEST_CASE("post-process leaves an already minimal correct program equivalent") {
    Program program{{Instruction{kOpXor, 1, 2}}, {3}};
    Program simplified = post_process_program(program, xor_examples(), 2, 1, enabled_options());
    REQUIRE(simplified.instrs.size() == 1);
    REQUIRE(simplified.outputs == std::vector<int>{3});
    require_equivalent(simplified, xor_examples(), 2, 1);
}

TEST_CASE("post-process removes unused instructions") {
    Program program{{Instruction{kOpXor, 1, 2}, Instruction{kOpAnd, 1, 2}}, {3}};
    Program simplified = post_process_program(program, xor_examples(), 2, 1, enabled_options());
    REQUIRE(simplified.instrs.size() == 1);
    REQUIRE(simplified.outputs == std::vector<int>{3});
    require_equivalent(simplified, xor_examples(), 2, 1);
}

TEST_CASE("post-process simplifies output selectors under don't-cares") {
    std::vector<Example> examples = {
        {{false}, {std::nullopt}},
        {{true}, {true}},
    };
    Program program{{Instruction{kOpOr, kSourceConstantOne, 1}}, {2}};
    Program simplified = post_process_program(program, examples, 1, 1, enabled_options());
    REQUIRE(simplified.instrs.empty());
    REQUIRE(simplified.outputs == std::vector<int>{0});
    require_equivalent(simplified, examples, 1, 1);
}

TEST_CASE("post-process applies algebraic identities and annihilators") {
    Program and_identity{{Instruction{kOpAnd, kSourceConstantOne, 1}}, {2}};
    Program simplified_and = post_process_program(and_identity, identity_examples(), 1, 1, enabled_options());
    REQUIRE(simplified_and.instrs.empty());
    REQUIRE(simplified_and.outputs == std::vector<int>{1});

    Program false_source{{Instruction{kOpXor, 1, 1}, Instruction{kOpOr, 1, 2}}, {3}};
    Program simplified_or = post_process_program(false_source, identity_examples(), 1, 1, enabled_options());
    REQUIRE(simplified_or.instrs.empty());
    REQUIRE(simplified_or.outputs == std::vector<int>{1});

    Program xor_self{{Instruction{kOpXor, 1, 1}, Instruction{kOpXor, 2, 2}}, {3}};
    std::vector<Example> false_examples = {
        {{false}, {false}},
        {{true}, {false}},
    };
    Program simplified_xor = post_process_program(xor_self, false_examples, 1, 1, enabled_options());
    REQUIRE(simplified_xor.instrs.size() <= 1);
    require_equivalent(simplified_xor, false_examples, 1, 1);
}

TEST_CASE("post-process applies absorption and XOR cancellation") {
    Program absorption{{Instruction{kOpOr, 1, 2}, Instruction{kOpAnd, 1, 3}}, {4}};
    std::vector<Example> first_input_examples = {
        {{false, false}, {false}},
        {{false, true}, {false}},
        {{true, false}, {true}},
        {{true, true}, {true}},
    };
    Program simplified_absorption = post_process_program(absorption, first_input_examples, 2, 1, enabled_options());
    REQUIRE(simplified_absorption.instrs.empty());
    REQUIRE(simplified_absorption.outputs == std::vector<int>{1});

    Program cancellation{{Instruction{kOpXor, 1, 2}, Instruction{kOpXor, 3, 2}}, {4}};
    Program simplified_cancellation = post_process_program(cancellation, first_input_examples, 2, 1, enabled_options());
    REQUIRE(simplified_cancellation.instrs.empty());
    REQUIRE(simplified_cancellation.outputs == std::vector<int>{1});
}

TEST_CASE("post-process tie-breaking picks the earliest matching source") {
    std::vector<Example> examples = {
        {{false, false}, {false}},
        {{true, true}, {true}},
    };
    Program program{{Instruction{kOpAnd, 1, 2}}, {3}};
    Program simplified = post_process_program(program, examples, 2, 1, enabled_options());
    REQUIRE(simplified.instrs.empty());
    REQUIRE(simplified.outputs == std::vector<int>{1});
    require_equivalent(simplified, examples, 2, 1);
}

TEST_CASE("post-process local SAT resynthesis reduces a two-node window") {
    std::vector<Example> examples = {
        {{false, false, false}, {false}}, {{false, false, true}, {true}}, {{false, true, false}, {false}},
        {{false, true, true}, {true}},    {{true, false, true}, {true}},  {{true, true, false}, {true}},
        {{true, true, true}, {true}},
    };
    Program program{{Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 4, 3}}, {5}};
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 2;
    options.resynthesis_patience = 1;
    Program simplified = post_process_program(program, examples, 3, 1, options);
    REQUIRE(simplified.instrs.size() == 1);
    require_equivalent(simplified, examples, 3, 1);
}

TEST_CASE("post-process rejects invalid input programs with a logic error") {
    Program invalid{{Instruction{kOpXor, 1, 4}}, {3}};
    REQUIRE_THROWS_AS(post_process_program(invalid, xor_examples(), 2, 1, enabled_options()), std::logic_error);
}

TEST_CASE("post-process local SAT resynthesis preserves multiple outputs") {
    std::vector<Example> examples = {
        {{false, false, false}, {false, false}}, {{false, false, true}, {false, true}},
        {{false, true, false}, {false, false}},  {{false, true, true}, {false, true}},
        {{true, false, true}, {false, true}},    {{true, true, false}, {true, true}},
        {{true, true, true}, {true, true}},
    };
    Program program{{Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 4, 3}}, {4, 5}};
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 2;
    options.resynthesis_patience = 0;
    Program simplified = post_process_program(program, examples, 3, 2, options);
    require_equivalent(simplified, examples, 3, 2);
}

TEST_CASE("post-process local SAT resynthesis remains correct with don't-care outputs and timeout") {
    std::vector<Example> examples = {
        {{false, false, false}, {false}}, {{false, false, true}, {true}},         {{false, true, false}, {false}},
        {{false, true, true}, {true}},    {{true, false, false}, {std::nullopt}}, {{true, false, true}, {true}},
        {{true, true, false}, {true}},    {{true, true, true}, {true}},
    };
    Program program{{Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 4, 3}}, {5}};
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 2;
    options.resynthesis_patience = 1;
    options.generator_timeout_seconds = 0.001;
    Program simplified = post_process_program(program, examples, 3, 1, options);
    require_equivalent(simplified, examples, 3, 1);

    options.generator_timeout_seconds = 0.0;
    simplified = post_process_program(program, examples, 3, 1, options);
    require_equivalent(simplified, examples, 3, 1);
}

TEST_CASE("post-process local SAT resynthesis keeps later users source-valid") {
    std::vector<Example> examples = {
        {{false, false, false, false}, {false}}, {{false, false, true, false}, {true}},
        {{false, true, true, true}, {false}},    {{true, false, true, true}, {false}},
        {{true, true, false, false}, {true}},    {{true, true, true, false}, {true}},
    };
    Program program{
        {Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 3, 5}, Instruction{kOpXor, 6, 4}},
        {7},
    };
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 2;
    options.resynthesis_patience = 0;
    Program simplified = post_process_program(program, examples, 4, 1, options);
    require_equivalent(simplified, examples, 4, 1);
}

TEST_CASE("post-process local SAT resynthesis materializes fresh temps before surviving users") {
    std::vector<Example> examples = {
        {{false, false, false, false}, {false}}, {{false, false, true, false}, {true}},
        {{false, true, false, true}, {true}},    {{true, false, true, true}, {false}},
        {{true, true, false, false}, {true}},    {{true, true, true, false}, {true}},
        {{true, true, true, true}, {false}},
    };
    Program program{
        {Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 5, 3}, Instruction{kOpOr, 6, 3}, Instruction{kOpXor, 7, 4}},
        {8},
    };
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 3;
    options.resynthesis_patience = 0;
    Program simplified = post_process_program(program, examples, 4, 1, options);
    REQUIRE(simplified.instrs.size() <= 3);
    require_equivalent(simplified, examples, 4, 1);
}

TEST_CASE("post-process local SAT resynthesis preserves mixed window outputs") {
    std::vector<Example> examples = {
        {{false, false, false, false}, {false, false}}, {{false, false, true, false}, {true, true}},
        {{false, true, false, true}, {false, true}},    {{true, false, true, true}, {true, false}},
        {{true, true, false, false}, {true, true}},     {{true, true, true, true}, {true, false}},
    };
    Program program{
        {Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 5, 3}, Instruction{kOpOr, 6, 3}, Instruction{kOpXor, 7, 4}},
        {6, 8},
    };
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 2;
    options.resynthesis_patience = 0;
    Program simplified = post_process_program(program, examples, 4, 2, options);
    REQUIRE(simplified.instrs.size() <= 3);
    require_equivalent(simplified, examples, 4, 2);
}

TEST_CASE("post-process local SAT resynthesis handles more than one mask word") {
    std::vector<Example> examples = all_examples(7, [](const std::vector<int> &inputs) {
        return (((inputs[0] != 0) && (inputs[1] != 0)) || (inputs[2] != 0)) != (inputs[3] != 0);
    });
    Program program{
        {Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 8, 3}, Instruction{kOpOr, 9, 3}, Instruction{kOpXor, 10, 4}},
        {11},
    };
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 3;
    options.resynthesis_patience = 0;
    Program simplified = post_process_program(program, examples, 7, 1, options);
    REQUIRE(simplified.instrs.size() <= 3);
    require_equivalent(simplified, examples, 7, 1);
}

TEST_CASE("post-process local SAT resynthesis updates profile counters") {
    std::vector<Example> examples = {
        {{false, false, false}, {false}}, {{false, false, true}, {true}}, {{false, true, false}, {false}},
        {{false, true, true}, {true}},    {{true, false, true}, {true}},  {{true, true, false}, {true}},
        {{true, true, true}, {true}},
    };
    Program program{{Instruction{kOpAnd, 1, 2}, Instruction{kOpOr, 4, 3}}, {5}};
    PostProcessOptions options = enabled_options();
    options.resynthesis_maxnodes = 2;
    options.resynthesis_patience = 1;
    ProfileData profile;
    Program simplified = post_process_program(program, examples, 3, 1, options, &profile);
    require_equivalent(simplified, examples, 3, 1);
    REQUIRE(profile.post_processing_resynthesis_windows_considered > 0);
    REQUIRE(profile.post_processing_resynthesis_windows_sat > 0);
    REQUIRE(profile.post_processing_resynthesis_candidates_materialized > 0);
    REQUIRE(profile.post_processing_resynthesis_candidates_accepted > 0);
}
