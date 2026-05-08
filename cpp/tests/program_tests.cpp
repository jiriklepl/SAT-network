#include "datasets.hpp"
#include "program.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <vector>

using boost::multiprecision::cpp_int;

TEST_CASE("program source formatting and text emission match Python format") {
    REQUIRE(format_source(0, 2) == "1");
    REQUIRE(format_source(1, 2) == "I0");
    REQUIRE(format_source(2, 2) == "I1");
    REQUIRE(format_source(3, 2) == "T0");

    Program xor_program{{Instruction{1, 1, 2}}, {3}};
    std::ostringstream emitted;
    emit_program(emitted, xor_program, 2);
    REQUIRE(emitted.str() == "T0: XOR(I0, I1)\nOUT0: T0\n");
}

TEST_CASE("packed verification accepts valid XOR and reports mismatches") {
    std::vector<Example> examples = {
        {{false, false}, {false}},
        {{false, true}, {true}},
        {{true, false}, {true}},
        {{true, true}, {false}},
    };

    Program xor_program{{Instruction{1, 1, 2}}, {3}};
    REQUIRE(verify_program(xor_program, examples, 2, 1).empty());

    Program or_program{{Instruction{2, 1, 2}}, {3}};
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
    REQUIRE(packed.input_masks[0] == cpp_int(2));
    REQUIRE(packed.output_values[0] == cpp_int(2));
    REQUIRE(packed.output_dont_care_masks[0] == cpp_int(1));
}

TEST_CASE("program spec computes source counts and bit widths") {
    REQUIRE(ProgramSpec{2, 1, 0}.total_sources() == 3);
    REQUIRE(ProgramSpec{2, 1, 0}.idx_bits() == 2);
    REQUIRE(ProgramSpec{3, 2, 5}.total_sources() == 9);
    REQUIRE(ProgramSpec{3, 2, 5}.idx_bits() == 4);
}
