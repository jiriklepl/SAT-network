#pragma once

#include "datasets.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <cstddef>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

struct LogicOperator {
    int code;
    const char *label;
    int canonical_rank;
};

struct ProgramSpec {
    int num_inputs = 0;
    int num_outputs = 0;
    int program_length = 0;

    int total_sources() const;
    unsigned idx_bits() const;
};

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
    std::vector<boost::multiprecision::cpp_int> input_masks;
    std::vector<boost::multiprecision::cpp_int> output_values;
    std::vector<boost::multiprecision::cpp_int> output_dont_care_masks;
};

const std::vector<LogicOperator> &logic_operators();
std::string cpp_int_to_string(const boost::multiprecision::cpp_int &value);
boost::multiprecision::cpp_int all_ones(unsigned width);
std::string bv_name(const std::string &prefix, int idx);
int op_rank(int code);
int op_code_by_label(const std::string &label);
const char *op_label(int code);
bool known_op(int code);
boost::multiprecision::cpp_int apply_operator_mask(
    int code,
    const boost::multiprecision::cpp_int &left,
    const boost::multiprecision::cpp_int &right);
std::string format_source(int idx, int num_inputs);
PackedExamples pack_examples(std::span<const Example> examples, int num_inputs, int num_outputs);
std::vector<boost::multiprecision::cpp_int> evaluate_program_masks(
    const Program &program,
    std::span<const boost::multiprecision::cpp_int> input_masks,
    const boost::multiprecision::cpp_int &all_examples_mask);
std::vector<std::size_t> verify_program(
    const Program &program,
    std::span<const Example> examples,
    int num_inputs,
    int num_outputs);
void emit_program(std::ostream &out, const Program &program, int num_inputs);
void emit_program_blif(std::ostream &out, const Program &program, int num_inputs);
void export_spec_blif(std::ostream &out, std::span<const Example> examples, int num_inputs, int num_outputs);
