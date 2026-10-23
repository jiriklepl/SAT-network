#pragma once

#include "datasets.hpp"
#include "mask.hpp"

#include <cstddef>

#include <algorithm>
#include <bit>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct LogicOperator {
    int code;
    const char *label;
    int canonical_rank;
};

constexpr int kSourceConstantOne = 0;
constexpr int kOpAnd = 0;
constexpr int kOpXor = 1;
constexpr int kOpOr = 2;
constexpr int kDefaultOperatorCost = 1;
constexpr int kXorOperatorCost = 2;

constexpr bool is_commutative_op(int op) noexcept {
    return op == kOpAnd || op == kOpXor || op == kOpOr;
}

constexpr int input_source(int input_idx) {
    return input_idx + 1;
}

constexpr int temp_source(int temp_idx, int num_inputs) {
    return num_inputs + 1 + temp_idx;
}

constexpr int temp_index_from_source(int source, int num_inputs) {
    return source - num_inputs - 1;
}

constexpr bool is_temp_source(int source, int num_inputs, int program_length) {
    const int idx = temp_index_from_source(source, num_inputs);
    return idx >= 0 && idx < program_length;
}

struct ProgramSpec {
    int num_inputs = 0;
    int num_outputs = 0;
    int program_length = 0;

    constexpr int total_sources() const {
        return temp_source(program_length, num_inputs);
    }

    constexpr unsigned idx_bits() const {
        const int max_idx = std::max(0, total_sources() - 1);
        const unsigned bits = std::bit_width(static_cast<unsigned>(max_idx));
        return std::max(1U, bits);
    }
};

constexpr ProgramSpec make_program_spec(const Config &cfg) {
    return {.num_inputs = cfg.num_inputs, .num_outputs = cfg.num_outputs, .program_length = cfg.instructions};
}

struct EncodingOptions {
    bool encode_boolean = false;
    bool force_ordered = false;
    bool force_useful = false;
    bool balanced_select = false;
};

struct Instruction {
    int op = -1;
    int s1 = 0;
    int s2 = 0;
};

struct Program {
    std::vector<Instruction> instrs;
    std::vector<int> outputs;
};

struct PackedExamples {
    unsigned width = 0;
    std::vector<PackedMask> input_masks;
    std::vector<PackedMask> output_values;
    std::vector<PackedMask> output_dont_care_masks;

    void append_example(const Example &ex);
    PackedExamples slice_words(std::size_t start_example, std::size_t count) const;
};

const std::vector<LogicOperator> &logic_operators();
std::string bv_name(const std::string &prefix, int idx);
std::string output_selector_name(int out_idx);
int op_rank(int code);
int op_code_by_label(const std::string &label);
const char *op_label(int code);
bool known_op(int code);
PackedMask apply_operator_mask(int code, const PackedMask &left, const PackedMask &right);
std::string format_source(int idx, int num_inputs);
int parse_source(std::string_view text, int num_inputs, std::optional<int> max_program_length = std::nullopt);

PackedExamples pack_examples(std::span<const Example> examples, int num_inputs, int num_outputs);
std::vector<PackedMask> evaluate_all_sources(const Program &program, std::span<const PackedMask> input_masks,
                                             const PackedMask &all_examples_mask);
std::vector<PackedMask> evaluate_program_masks(const Program &program, std::span<const PackedMask> input_masks,
                                               const PackedMask &all_examples_mask);
bool verify_program_packed_ok(const Program &program, const PackedExamples &packed, int num_outputs);
std::vector<std::size_t> verify_program_packed(const Program &program, const PackedExamples &packed, int num_outputs);
std::vector<std::size_t> verify_program(const Program &program, std::span<const Example> examples, int num_inputs,
                                        int num_outputs);
void emit_program(std::ostream &out, const Program &program, int num_inputs);
Program parse_program(std::istream &in, int num_inputs, int num_outputs);
Program parse_program(std::string_view text, int num_inputs, int num_outputs);
void emit_program_blif(std::ostream &out, const Program &program, int num_inputs);
void export_spec_blif(std::ostream &out, std::span<const Example> examples, int num_inputs, int num_outputs);
