#include "assumptions.hpp"
#include "encoding.hpp"

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>

namespace {

Assumptions parse_text(const std::string &text, const ProgramSpec &spec) {
    std::istringstream input(text);
    return parse_assumptions(input, spec);
}

}  // namespace

TEST_CASE("assumption parser accepts instructions outputs comments and blanks") {
    ProgramSpec spec{2, 1, 1};
    Assumptions assumptions = parse_text(
        "# fixed xor\n"
        "\n"
        "T0: XOR(I0, I1)\n"
        "OUT0: T0\n",
        spec);

    REQUIRE(assumptions.instructions.size() == 1);
    REQUIRE(assumptions.instructions[0].instr_idx == 0);
    REQUIRE(assumptions.instructions[0].op == kOpXor);
    REQUIRE(assumptions.instructions[0].s1 == 1);
    REQUIRE(assumptions.instructions[0].s2 == 2);
    REQUIRE(assumptions.outputs.size() == 1);
    REQUIRE(assumptions.outputs[0].out_idx == 0);
    REQUIRE(assumptions.outputs[0].source == 3);
}

TEST_CASE("assumption parser rejects malformed assumptions") {
    ProgramSpec spec{2, 1, 1};
    REQUIRE_THROWS(parse_text("T0 XOR(I0, I1)\n", spec));
    REQUIRE_THROWS(parse_text("X0: I0\n", spec));
    REQUIRE_THROWS(parse_text("T0: NAND(I0, I1)\n", spec));
    REQUIRE_THROWS(parse_text("T0: XOR(I9, I1)\n", spec));
    REQUIRE_THROWS(parse_text("T9: XOR(I0, I1)\n", spec));
    REQUIRE_THROWS(parse_text("OUT9: T0\n", spec));
    REQUIRE_THROWS(parse_text("OUT0: BAD\n", spec));
    REQUIRE_THROWS(parse_text("T0: XOR(I0)\n", spec));
}

TEST_CASE("assumption constraints can make a candidate SAT or UNSAT") {
    ProgramSpec spec{2, 1, 1};
    z3::context ctx;

    {
        z3::solver solver(ctx);
        add_exprs(solver, build_program(ctx, spec, EncodingOptions{}));
        add_exprs(solver, build_assumption_constraints(ctx, spec, parse_text("T0: XOR(I0, I1)\nOUT0: T0\n", spec)));
        REQUIRE(solver.check() == z3::sat);
    }

    {
        z3::solver solver(ctx);
        add_exprs(solver, build_program(ctx, spec, EncodingOptions{}));
        add_exprs(solver, build_assumption_constraints(ctx, spec, parse_text("T0: AND(I0, I0)\n", spec)));
        REQUIRE(solver.check() == z3::unsat);
    }
}
