#pragma once

#include "datasets.hpp"
#include "profile.hpp"
#include "program.hpp"
#include "quantified_spec.hpp"

#include <z3++.h>

#include <cstddef>

#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

// Proves (or refutes) that a concrete straight-line `Program` realises a quantified dataset spec for *every*
// assignment of the `forall` variables, using only quantifier-free bit-vector Z3 queries.
//
// The spec is lowered once at construction (quantified variables become fixed-width bit-vector constants, the
// `inputs` / `outputs` emitters become Boolean expressions plus "care" bits for `null` outputs). Each query then
// instantiates the candidate program as plain Boolean gates and asks Z3 for an assignment on which some cared-for
// output differs: a combinational miter.
class SymbolicVerifier {
public:
    explicit SymbolicVerifier(std::shared_ptr<const QuantifiedSpec> spec);

    // Returns up to `max_counterexamples` distinct spec rows on which `program` is wrong; empty means proven.
    // Throws std::runtime_error if Z3 gives up (unless `timeout_ms` was set and `timed_out != nullptr`).
    std::vector<Example> find_counterexamples(const Program &program, std::size_t max_counterexamples = 1,
                                              std::optional<int> timeout_ms = std::nullopt,
                                              ProfileData *profile = nullptr, bool *timed_out = nullptr);

private:
    struct VarBinding {
        enum class Kind { Scalar, TupleCell } kind;
        std::size_t slot;
        std::size_t cell;
        z3::expr raw;  // narrow bit-vector variable actually seen by the solver
    };

    std::shared_ptr<const QuantifiedSpec> spec_;
    z3::context ctx_;
    z3::solver solver_;
    std::vector<VarBinding> bindings_;
    std::vector<z3::expr> domain_constraints_;
    std::vector<z3::expr> input_values_;
    std::vector<z3::expr> output_values_;
    std::vector<z3::expr> output_cares_;
};

// Unified candidate verification oracle combining fast 64-bit word-parallel simulation on the active
// `PackedExamples` basis with exact symbolic miter verification for unmaterialized quantified specs.
class VerificationOracle {
public:
    VerificationOracle(std::span<const Example> initial_examples, int num_inputs, int num_outputs,
                       SymbolicVerifier *verifier = nullptr, ProfileData *profile = nullptr);

    int num_inputs() const noexcept { return num_inputs_; }
    int num_outputs() const noexcept { return num_outputs_; }
    std::span<const Example> examples() const noexcept { return examples_; }
    const PackedExamples &packed() const noexcept { return packed_; }
    bool has_symbolic_verifier() const noexcept { return verifier_ != nullptr; }
    std::size_t basis_generation() const noexcept { return basis_generation_; }

    void mark_verified(const std::string &key);
    void add_examples(std::span<const Example> learned);

    bool verify_candidate(const Program &candidate, const std::string &canonical_key = {},
                          std::optional<int> timeout_ms = std::nullopt, std::size_t max_counterexamples = 4);

private:
    std::vector<Example> examples_;
    PackedExamples packed_;
    int num_inputs_ = 0;
    int num_outputs_ = 0;
    SymbolicVerifier *verifier_ = nullptr;
    ProfileData *profile_ = nullptr;
    std::set<std::string> verified_keys_;
    std::size_t basis_generation_ = 0;
};

