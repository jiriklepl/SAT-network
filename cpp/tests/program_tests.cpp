#include "datasets.hpp"
#include "program.hpp"

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <vector>

TEST_CASE("program source formatting and text emission match Python format") {
    REQUIRE(format_source(0, 2) == "1");
    REQUIRE(format_source(1, 2) == "I0");
    REQUIRE(format_source(2, 2) == "I1");
    REQUIRE(format_source(3, 2) == "T0");

    Program xor_program{{Instruction{kOpXor, 1, 2}}, {3}};
    std::ostringstream emitted;
    emit_program(emitted, xor_program, 2);
    REQUIRE(emitted.str() == "T0: XOR(I0, I1)\nOUT0: T0\n");
}

TEST_CASE("program BLIF emission matches supported logic gates") {
    Program program{{Instruction{kOpAnd, 1, 2}, Instruction{kOpXor, 1, 2}, Instruction{kOpOr, 3, 4}}, {5}};
    std::ostringstream emitted;
    emit_program_blif(emitted, program, 2);
    REQUIRE(emitted.str().find(".model spec\n") == 0);
    REQUIRE(emitted.str().find(".names I0 I1 T0\n11 1\n") != std::string::npos);
    REQUIRE(emitted.str().find(".names I0 I1 T1\n10 1\n01 1\n") != std::string::npos);
    REQUIRE(emitted.str().find(".names T0 T1 T2\n10 1\n01 1\n11 1\n") != std::string::npos);
    REQUIRE(emitted.str().find(".names T2 OUT0\n1 1\n.end\n") != std::string::npos);

    Program duplicate_and{{Instruction{kOpAnd, 1, 1}}, {2}};
    REQUIRE_THROWS(emit_program_blif(emitted, duplicate_and, 1));
}

TEST_CASE("spec BLIF export emits truth table and rejects don't-cares") {
    std::vector<Example> examples = {
        {{false, false}, {false}},
        {{false, true}, {true}},
        {{true, false}, {true}},
        {{true, true}, {false}},
    };
    std::ostringstream emitted;
    export_spec_blif(emitted, examples, 2, 1);
    REQUIRE(emitted.str() ==
            ".model synth_program\n.inputs I0 I1\n.outputs OUT0\n.names I0 I1 OUT0\n01 1\n10 1\n.end\n");

    std::vector<Example> dont_care = {{{false}, {std::nullopt}}};
    REQUIRE_THROWS(export_spec_blif(emitted, dont_care, 1, 1));
}

TEST_CASE("packed verification accepts valid XOR and reports mismatches") {
    std::vector<Example> examples = {
        {{false, false}, {false}},
        {{false, true}, {true}},
        {{true, false}, {true}},
        {{true, true}, {false}},
    };

    Program xor_program{{Instruction{kOpXor, 1, 2}}, {3}};
    REQUIRE(verify_program(xor_program, examples, 2, 1).empty());

    Program or_program{{Instruction{kOpOr, 1, 2}}, {3}};
    std::vector<std::size_t> mismatches = verify_program(or_program, examples, 2, 1);
    REQUIRE(mismatches == std::vector<std::size_t>{3});
}

TEST_CASE("packed verification respects don't-care output masks") {
    std::vector<Example> examples = {
        {{false}, {std::nullopt}},
        {{true}, {true}},
    };
    Program constant_one{{}, {0}};
    REQUIRE(verify_program(constant_one, examples, 1, 1).empty());

    PackedExamples packed = pack_examples(examples, 1, 1);
    REQUIRE(packed.width == 2);
    REQUIRE(packed.input_masks[0].set_bit_indices() == std::vector<std::size_t>{1});
    REQUIRE(packed.output_values[0].set_bit_indices() == std::vector<std::size_t>{1});
    REQUIRE(packed.output_dont_care_masks[0].set_bit_indices() == std::vector<std::size_t>{0});
}

TEST_CASE("packed verification handles more than one mask word") {
    std::vector<Example> examples;
    for (std::size_t idx = 0; idx < 130; ++idx) {
        const bool bit = (idx % 2) != 0;
        examples.push_back({{bit}, {bit}});
    }

    Program identity{{}, {1}};
    REQUIRE(verify_program(identity, examples, 1, 1).empty());

    Program constant_one{{}, {0}};
    std::vector<std::size_t> mismatches = verify_program(constant_one, examples, 1, 1);
    std::vector<std::size_t> expected;
    for (std::size_t idx = 0; idx < 130; idx += 2) {
        expected.push_back(idx);
    }
    REQUIRE(mismatches == expected);
}

TEST_CASE("program spec computes source counts and bit widths") {
    REQUIRE(ProgramSpec{2, 1, 0}.total_sources() == 3);
    REQUIRE(ProgramSpec{2, 1, 0}.idx_bits() == 2);
    REQUIRE(ProgramSpec{3, 2, 5}.total_sources() == 9);
    REQUIRE(ProgramSpec{3, 2, 5}.idx_bits() == 4);
}
