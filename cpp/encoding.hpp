#pragma once

#include "assumptions.hpp"
#include "datasets.hpp"
#include "program.hpp"

#include <z3++.h>

#include <string>
#include <vector>

std::vector<z3::expr> build_program(z3::context &ctx, const ProgramSpec &spec, const EncodingOptions &options);
std::vector<z3::expr> build_assumption_constraints(z3::context &ctx, const ProgramSpec &spec, const Assumptions &assumptions);
void add_exprs(z3::solver &solver, const std::vector<z3::expr> &exprs);
void add_example_constraints(
    z3::context &ctx,
    z3::solver &solver,
    const std::vector<Example> &examples,
    const std::string &tag,
    const ProgramSpec &spec,
    const EncodingOptions &options);
Program extract_program(z3::context &ctx, const z3::model &model, const ProgramSpec &spec);
z3::solver make_solver(z3::context &ctx, const std::string &solver_choice);
