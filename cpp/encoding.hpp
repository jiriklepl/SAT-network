#pragma once

#include "assumptions.hpp"
#include "datasets.hpp"
#include "profile.hpp"
#include "program.hpp"

#include <z3++.h>

#include <span>
#include <string>
#include <vector>

struct PackedTestEncoding {
    std::vector<z3::expr> constraints;
    std::vector<z3::expr> outputs;
};

std::vector<z3::expr> build_program(z3::context &ctx, const ProgramSpec &spec, const EncodingOptions &options,
                                    ProfileData *profile = nullptr);
PackedTestEncoding build_packed_test(z3::context &ctx, std::span<const PackedMask> input_masks, const std::string &tag,
                                     const ProgramSpec &spec, const EncodingOptions &options,
                                     ProfileData *profile = nullptr);
z3::expr packed_mask_value(z3::context &ctx, const PackedMask &mask);
std::vector<z3::expr> build_assumption_constraints(z3::context &ctx, const ProgramSpec &spec,
                                                   const Assumptions &assumptions);
void add_exprs(z3::solver &solver, std::span<const z3::expr> exprs);
void add_example_constraints(z3::context &ctx, z3::solver &solver, std::span<const Example> examples,
                             const std::string &tag, const ProgramSpec &spec, const EncodingOptions &options,
                             ProfileData *profile = nullptr);
Program extract_program(z3::context &ctx, const z3::model &model, const ProgramSpec &spec);
z3::solver make_solver(z3::context &ctx, const std::string &solver_choice);
