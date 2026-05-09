#pragma once

#include <cstddef>

struct ProfileData {
    double dataset_generation_seconds = 0.0;
    double structure_encoding_seconds = 0.0;
    double example_packing_seconds = 0.0;
    double example_encoding_seconds = 0.0;
    double z3_solve_seconds = 0.0;
    double model_extraction_seconds = 0.0;
    double packed_verification_seconds = 0.0;

    std::size_t structure_constraints = 0;
    std::size_t example_constraints = 0;
    std::size_t example_batches = 0;
    std::size_t packed_examples = 0;
    std::size_t solver_checks = 0;
    std::size_t model_extractions = 0;
    std::size_t verification_examples = 0;
    std::size_t bv_cache_hits = 0;
    std::size_t bv_cache_misses = 0;
};
