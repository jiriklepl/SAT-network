#pragma once

#include "program.hpp"

#include <iosfwd>
#include <vector>

struct InstructionAssumption {
    int instr_idx = 0;
    int op = 0;
    int s1 = 0;
    int s2 = 0;
};

struct OutputAssumption {
    int out_idx = 0;
    int source = 0;
};

struct Assumptions {
    std::vector<InstructionAssumption> instructions;
    std::vector<OutputAssumption> outputs;
};

Assumptions parse_assumptions(std::istream &input, const ProgramSpec &spec);
