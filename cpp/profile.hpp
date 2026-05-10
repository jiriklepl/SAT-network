#pragma once

#include <cstddef>

struct ProfileData {
    double dataset_generation_seconds = 0.0;
    double structure_encoding_seconds = 0.0;
    double example_packing_seconds = 0.0;
    double example_encoding_seconds = 0.0;
    double z3_solve_seconds = 0.0;
    double model_extraction_seconds = 0.0;
    double post_processing_seconds = 0.0;
    double post_processing_mask_generator_seconds = 0.0;
    double post_processing_replacement_generator_seconds = 0.0;
    double post_processing_resynthesis_generator_seconds = 0.0;
    double packed_verification_seconds = 0.0;

    std::size_t structure_constraints = 0;
    std::size_t example_constraints = 0;
    std::size_t example_batches = 0;
    std::size_t packed_examples = 0;
    std::size_t solver_checks = 0;
    std::size_t model_extractions = 0;
    std::size_t post_processing_runs = 0;
    std::size_t post_processing_input_instructions = 0;
    std::size_t post_processing_output_instructions = 0;
    std::size_t post_processing_mask_candidates_considered = 0;
    std::size_t post_processing_mask_candidates_materialized = 0;
    std::size_t post_processing_mask_candidates_accepted = 0;
    std::size_t post_processing_mask_invalid_candidates = 0;
    std::size_t post_processing_mask_timeout_exits = 0;
    std::size_t post_processing_replacement_candidates_considered = 0;
    std::size_t post_processing_replacement_candidates_materialized = 0;
    std::size_t post_processing_replacement_candidates_accepted = 0;
    std::size_t post_processing_replacement_invalid_candidates = 0;
    std::size_t post_processing_replacement_timeout_exits = 0;
    std::size_t post_processing_resynthesis_windows_considered = 0;
    std::size_t post_processing_resynthesis_windows_sat = 0;
    std::size_t post_processing_resynthesis_candidates_considered = 0;
    std::size_t post_processing_resynthesis_candidates_materialized = 0;
    std::size_t post_processing_resynthesis_invalid_candidates = 0;
    std::size_t post_processing_resynthesis_candidates_accepted = 0;
    std::size_t post_processing_resynthesis_timeout_exits = 0;
    std::size_t verification_examples = 0;
    std::size_t bv_cache_hits = 0;
    std::size_t bv_cache_misses = 0;
};
