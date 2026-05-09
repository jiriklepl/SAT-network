#include "program.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <bit>

using boost::multiprecision::cpp_int;

int ProgramSpec::total_sources() const {
    return num_inputs + 1 + program_length;
}

unsigned ProgramSpec::idx_bits() const {
    const int max_idx = std::max(0, total_sources() - 1);
    const unsigned bits = std::bit_width(static_cast<unsigned>(max_idx));
    return std::max(1u, bits);
}

const std::vector<LogicOperator> &logic_operators() {
    static const std::vector<LogicOperator> operators{
        {0, "AND", 1},
        {1, "XOR", 0},
        {2, "OR", 2},
    };
    return operators;
}

std::string cpp_int_to_string(const cpp_int &value) {
    return value.convert_to<std::string>();
}

cpp_int all_ones(unsigned width) {
    return (cpp_int{1} << width) - 1;
}

std::string bv_name(const std::string &prefix, int idx) {
    return prefix + "_" + std::to_string(idx);
}

int op_rank(int code) {
    for (const auto &op : logic_operators()) {
        if (op.code == code) {
            return op.canonical_rank;
        }
    }
    throw std::runtime_error("unknown operator code");
}

int op_code_by_label(const std::string &label) {
    for (const auto &op : logic_operators()) {
        if (label == op.label) {
            return op.code;
        }
    }
    throw std::runtime_error("unknown operator label");
}

const char *op_label(int code) {
    for (const auto &op : logic_operators()) {
        if (op.code == code) {
            return op.label;
        }
    }
    return "?";
}

const std::vector<const char *> &op_blif_rows(int code) {
    static const std::vector<const char *> and_rows = {"11 1"};
    static const std::vector<const char *> xor_rows = {"10 1", "01 1"};
    static const std::vector<const char *> or_rows = {"10 1", "01 1", "11 1"};
    if (code == 0) return and_rows;
    if (code == 1) return xor_rows;
    if (code == 2) return or_rows;
    throw std::runtime_error("Unsupported operation in BLIF output");
}

bool known_op(int code) {
    const auto &operators = logic_operators();
    return std::any_of(operators.begin(), operators.end(), [&](const LogicOperator &op) {
        return op.code == code;
    });
}

cpp_int apply_operator_mask(int code, const cpp_int &left, const cpp_int &right) {
    if (code == 0) return left & right;
    if (code == 1) return left ^ right;
    if (code == 2) return left | right;
    return 0;
}

std::string format_source(int idx, int num_inputs) {
    if (idx == 0) return "1";
    if (idx <= num_inputs) return "I" + std::to_string(idx - 1);
    return "T" + std::to_string(idx - num_inputs - 1);
}

PackedExamples pack_examples(std::span<const Example> examples, int num_inputs, int num_outputs) {
    PackedExamples packed;
    packed.width = static_cast<unsigned>(examples.size());
    packed.input_masks.assign(num_inputs, 0);
    packed.output_values.assign(num_outputs, 0);
    packed.output_dont_care_masks.assign(num_outputs, 0);

    for (std::size_t ex_idx = 0; ex_idx < examples.size(); ++ex_idx) {
        cpp_int bit = cpp_int(1) << ex_idx;
        const auto &ex = examples[ex_idx];
        for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
            if (ex.input(static_cast<std::size_t>(input_idx))) {
                packed.input_masks[input_idx] |= bit;
            }
        }
        for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
            const std::size_t idx = static_cast<std::size_t>(output_idx);
            if (ex.output_dont_care[idx]) {
                packed.output_dont_care_masks[output_idx] |= bit;
            } else if (ex.output_values[idx]) {
                packed.output_values[output_idx] |= bit;
            }
        }
    }
    return packed;
}

std::vector<cpp_int> evaluate_program_masks(
    const Program &program,
    std::span<const cpp_int> input_masks,
    const cpp_int &all_examples_mask
) {
    std::vector<cpp_int> values;
    values.push_back(all_examples_mask);
    values.insert(values.end(), input_masks.begin(), input_masks.end());
    for (const auto &instr : program.instrs) {
        cpp_int value = instr.op < 0 ? 0 : (apply_operator_mask(instr.op, values[instr.s1], values[instr.s2]) & all_examples_mask);
        values.push_back(value);
    }
    std::vector<cpp_int> outputs;
    for (int selector : program.outputs) {
        outputs.push_back(values[selector]);
    }
    return outputs;
}

std::vector<std::size_t> verify_program(
    const Program &program,
    std::span<const Example> examples,
    int num_inputs,
    int num_outputs
) {
    if (examples.empty()) return {};
    PackedExamples packed = pack_examples(examples, num_inputs, num_outputs);
    cpp_int full_mask = all_ones(packed.width);
    std::vector<cpp_int> actual = evaluate_program_masks(program, packed.input_masks, full_mask);
    cpp_int mismatch_mask = 0;
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
        cpp_int care_mask = full_mask ^ packed.output_dont_care_masks[out_idx];
        mismatch_mask |= (actual[out_idx] ^ packed.output_values[out_idx]) & care_mask;
    }
    std::vector<std::size_t> mismatches;
    for (std::size_t idx = 0; idx < examples.size(); ++idx) {
        if (((mismatch_mask >> idx) & 1) != 0) {
            mismatches.push_back(idx);
        }
    }
    return mismatches;
}

void emit_program(std::ostream &out, const Program &program, int num_inputs) {
    for (std::size_t instr_idx = 0; instr_idx < program.instrs.size(); ++instr_idx) {
        const auto &instr = program.instrs[instr_idx];
        out << "T" << instr_idx << ": " << op_label(instr.op) << "("
            << format_source(instr.s1, num_inputs) << ", "
            << format_source(instr.s2, num_inputs) << ")\n";
    }
    for (std::size_t out_idx = 0; out_idx < program.outputs.size(); ++out_idx) {
        out << "OUT" << out_idx << ": " << format_source(program.outputs[out_idx], num_inputs) << "\n";
    }
}

void emit_program_blif(std::ostream &out, const Program &program, int num_inputs) {
    out << ".model spec\n";
    out << ".inputs";
    for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
        out << " I" << input_idx;
    }
    out << "\n.outputs";
    for (std::size_t out_idx = 0; out_idx < program.outputs.size(); ++out_idx) {
        out << " OUT" << out_idx;
    }
    out << "\n";

    for (std::size_t instr_idx = 0; instr_idx < program.instrs.size(); ++instr_idx) {
        const auto &instr = program.instrs[instr_idx];
        if (!known_op(instr.op)) {
            throw std::runtime_error("Unsupported operation in BLIF output");
        }
        if (instr.s1 == instr.s2) {
            if (instr.op != 1) {
                throw std::runtime_error("Only XOR may use duplicate sources in BLIF output");
            }
            out << ".names T" << instr_idx << "\n";
        } else {
            out << ".names " << format_source(instr.s1, num_inputs) << " "
                << format_source(instr.s2, num_inputs) << " T" << instr_idx << "\n";
            for (const char *row : op_blif_rows(instr.op)) {
                out << row << "\n";
            }
        }
    }

    for (std::size_t out_idx = 0; out_idx < program.outputs.size(); ++out_idx) {
        out << ".names " << format_source(program.outputs[out_idx], num_inputs) << " OUT" << out_idx << "\n";
        out << "1 1\n";
    }
    out << ".end\n";
}

void export_spec_blif(std::ostream &out, std::span<const Example> examples, int num_inputs, int num_outputs) {
    out << ".model synth_program\n";
    out << ".inputs";
    for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
        out << " I" << input_idx;
    }
    out << "\n.outputs";
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
        out << " OUT" << out_idx;
    }
    out << "\n";

    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
        out << ".names";
        for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
            out << " I" << input_idx;
        }
        out << " OUT" << out_idx << "\n";
        for (const auto &example : examples) {
            const auto output = example.output(static_cast<std::size_t>(out_idx));
            if (!output.has_value()) {
                throw std::runtime_error("Cannot export BLIF with don't-care outputs");
            }
            if (*output) {
                for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
                    out << (example.input(static_cast<std::size_t>(input_idx)) ? '1' : '0');
                }
                out << " 1\n";
            }
        }
    }

    out << ".end\n";
}
