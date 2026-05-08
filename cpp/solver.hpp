#pragma once

#include "assumptions.hpp"
#include "datasets.hpp"
#include "program.hpp"

#include <cstddef>
#include <optional>
#include <string>

enum class SolveStatus {
    Sat,
    Unsat,
    Unknown,
    VerificationFailed,
};

struct SolveOptions {
    std::string solver = "simple-tactic";
    EncodingOptions encoding;
    Assumptions assumptions;
    std::optional<std::size_t> batch_size;
    bool cegis = false;
    std::size_t cegis_initial_size = 64;
    std::size_t cegis_counterexamples = 1;
};

struct SolveResult {
    SolveStatus status = SolveStatus::Unknown;
    std::optional<Program> program;
    double elapsed_seconds = 0.0;
    std::size_t mismatch_count = 0;
};

SolveResult solve_config(const Config &cfg, const SolveOptions &options);
std::string make_smt2(const Config &cfg, const SolveOptions &options);
