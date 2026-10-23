#pragma once

#include "datasets.hpp"
#include "mask.hpp"
#include "postprocess.hpp"
#include "profile.hpp"
#include "program.hpp"
#include "symbolic_verifier.hpp"

#include <cstddef>
#include <cstdint>

#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

struct ProgramScore {
    std::vector<double> parts;
    std::vector<int> outputs;
    std::vector<int> instr_key;
};

enum class PostProcessGeneratorKind : std::uint8_t {
    Mask,
    Replacement,
    Resynthesis,
};

class PostProcessGeneratorRun {
public:
    PostProcessGeneratorRun(ProfileData *profile, PostProcessGeneratorKind kind, double timeout_seconds);

    [[nodiscard]] bool timed_out();
    [[nodiscard]] std::optional<int> remaining_timeout_ms();
    void finish();
    void candidate_considered();
    void candidate_materialized();
    void candidate_accepted();
    void invalid_candidate();

private:
    struct Metrics {
        double *seconds = nullptr;
        std::size_t *considered = nullptr;
        std::size_t *materialized = nullptr;
        std::size_t *accepted = nullptr;
        std::size_t *invalid = nullptr;
        std::size_t *timeout_exits = nullptr;
    };

    ProfileData *profile_ = nullptr;
    PostProcessGeneratorKind kind_;
    double timeout_seconds_ = 0.0;
    std::chrono::steady_clock::time_point start_;
    Metrics metrics_;
    bool timeout_recorded_ = false;
    bool finished_ = false;
};

bool operator<(const ProgramScore &left, const ProgramScore &right);

std::string program_key(const Program &program);
std::optional<std::string> validate_program_invariants(const Program &program, int num_inputs, int num_outputs);
void require_valid_program(const Program &program, int num_inputs, int num_outputs, const std::string &context);

std::vector<std::set<int>> build_users(const Program &program, int num_inputs);
Program materialize_dag(const std::map<int, Instruction> &nodes, std::span<const int> outputs, int num_inputs);
Program prune_dead_nodes(const Program &program, int num_inputs);
ProgramScore score_program(const Program &program, int num_inputs, const PackedExamples &packed,
                           const PostProcessScorePhase &phase, unsigned random_seed);

bool push_unique_candidate(std::vector<Program> &candidates, std::set<std::string> &seen, const Program &candidate,
                           const std::string &base_key, VerificationOracle &oracle,
                           PostProcessGeneratorRun &generator);

std::vector<Program> generate_mask_simplification_candidates(const Program &base, VerificationOracle &oracle,
                                                             const std::string &base_key, std::set<std::string> &seen,
                                                             std::size_t max_candidates,
                                                             const PostProcessOptions &options, ProfileData *profile);

std::vector<Program> generate_resynthesis_candidates(const Program &base, VerificationOracle &oracle,
                                                     const PostProcessOptions &options, const std::string &base_key,
                                                     std::set<std::string> &seen, std::size_t max_candidates,
                                                     ProfileData *profile);
