#include "postprocess.hpp"

#include "datasets.hpp"
#include "program.hpp"

#include <catch2/catch_test_macros.hpp>

#include <span>
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
