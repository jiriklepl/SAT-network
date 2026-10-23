#include "program.hpp"

#include "datasets.hpp"
#include "mask.hpp"
#include "string_utils.hpp"

#include <cstddef>

#include <algorithm>
#include <istream>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

const std::vector<LogicOperator> &logic_operators() {
    static const std::vector<LogicOperator> operators{
        {.code = kOpAnd, .label = "AND", .canonical_rank = 1},
        {.code = kOpXor, .label = "XOR", .canonical_rank = 0},
        {.code = kOpOr, .label = "OR", .canonical_rank = 2},
    };
    return operators;
}

std::string bv_name(const std::string &prefix, int idx) {
    return prefix + "_" + std::to_string(idx);
}

std::string output_selector_name(int out_idx) {
    return "OUT_" + std::to_string(out_idx) + "_idx";
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
    if (code == kOpAnd) return and_rows;
    if (code == kOpXor) return xor_rows;
    if (code == kOpOr) return or_rows;
    throw std::runtime_error("Unsupported operation in BLIF output");
}

bool known_op(int code) {
    const auto &operators = logic_operators();
    return std::ranges::any_of(operators, [&](const LogicOperator &op) { return op.code == code; });
}

PackedMask apply_operator_mask(int code, const PackedMask &left, const PackedMask &right) {
    if (code == kOpAnd) return left & right;
    if (code == kOpXor) return left ^ right;
    if (code == kOpOr) return left | right;
    return PackedMask(left.width());
}

std::string format_source(int idx, int num_inputs) {
    if (idx == kSourceConstantOne) return "1";
    if (idx <= num_inputs) return "I" + std::to_string(idx - 1);
    return "T" + std::to_string(temp_index_from_source(idx, num_inputs));
}

int parse_source(std::string_view text, int num_inputs, std::optional<int> max_program_length) {
    const std::string_view tok = trim_view(text);
    if (tok == "1") return kSourceConstantOne;
    if (tok.starts_with("I")) {
        std::size_t parsed = 0;
        const std::string sub(tok.substr(1));
        int idx = 0;
        try {
            idx = std::stoi(sub, &parsed);
        } catch (const std::exception &) {
            throw std::runtime_error("Invalid input index in source: " + std::string(tok));
        }
        if (parsed != sub.size() || idx < 0 || idx >= num_inputs) {
            throw std::runtime_error("Input index out of range: " + std::string(tok));
        }
        return input_source(idx);
    }
    if (tok.starts_with("T")) {
        std::size_t parsed = 0;
        const std::string sub(tok.substr(1));
        int idx = 0;
        try {
            idx = std::stoi(sub, &parsed);
        } catch (const std::exception &) {
            throw std::runtime_error("Invalid temporary index in source: " + std::string(tok));
        }
        if (parsed != sub.size() || idx < 0 || (max_program_length.has_value() && idx >= *max_program_length)) {
            throw std::runtime_error("Temporary index out of range: " + std::string(tok));
        }
        return temp_source(idx, num_inputs);
    }
    throw std::runtime_error("Unknown source argument: " + std::string(tok));
}

void PackedExamples::append_example(const Example &ex) {
    if (ex.input_count() != input_masks.size() || ex.output_count() != output_values.size()) {
        throw std::runtime_error("Example length does not match PackedExamples dimensions");
    }
    for (std::size_t i = 0; i < input_masks.size(); ++i) {
        input_masks[i].push_back(ex.input(i));
    }
    for (std::size_t j = 0; j < output_values.size(); ++j) {
        const bool dc = ex.output_dont_care[j];
        output_dont_care_masks[j].push_back(dc);
        output_values[j].push_back(!dc && ex.output_values[j]);
    }
    ++width;
}

PackedExamples PackedExamples::slice_words(std::size_t start_example, std::size_t count) const {
    if (start_example > width) {
        throw std::out_of_range("PackedExamples slice start out of range");
    }
    const std::size_t actual_count = std::min<std::size_t>(count, width - start_example);
    PackedExamples result;
    result.width = static_cast<unsigned>(actual_count);
    result.input_masks.reserve(input_masks.size());
    for (const auto &m : input_masks) result.input_masks.push_back(m.slice(start_example, actual_count));
    result.output_values.reserve(output_values.size());
    for (const auto &m : output_values) result.output_values.push_back(m.slice(start_example, actual_count));
    result.output_dont_care_masks.reserve(output_dont_care_masks.size());
    for (const auto &m : output_dont_care_masks) {
        result.output_dont_care_masks.push_back(m.slice(start_example, actual_count));
    }
    return result;
}

PackedExamples pack_examples(std::span<const Example> examples, int num_inputs, int num_outputs) {
    PackedExamples packed;
    packed.width = static_cast<unsigned>(examples.size());
    packed.input_masks.assign(num_inputs, PackedMask(packed.width));
    packed.output_values.assign(num_outputs, PackedMask(packed.width));
    packed.output_dont_care_masks.assign(num_outputs, PackedMask(packed.width));

    for (std::size_t ex_idx = 0; ex_idx < examples.size(); ++ex_idx) {
        const auto &ex = examples[ex_idx];
        for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
            if (ex.input(static_cast<std::size_t>(input_idx))) {
                packed.input_masks[input_idx].set(ex_idx);
            }
        }
        for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
            const auto idx = static_cast<std::size_t>(output_idx);
            if (ex.output_dont_care[idx]) {
                packed.output_dont_care_masks[output_idx].set(ex_idx);
            } else if (ex.output_values[idx]) {
                packed.output_values[output_idx].set(ex_idx);
            }
        }
    }
    return packed;
}

std::vector<PackedMask> evaluate_all_sources(const Program &program, std::span<const PackedMask> input_masks,
                                             const PackedMask &all_examples_mask) {
    std::vector<PackedMask> values;
    values.reserve(1 + input_masks.size() + program.instrs.size());
    values.push_back(all_examples_mask);
    values.insert(values.end(), input_masks.begin(), input_masks.end());
    for (const auto &instr : program.instrs) {
        const PackedMask value =
            instr.op < 0 ? PackedMask(all_examples_mask.width())
                         : (apply_operator_mask(instr.op, values[static_cast<std::size_t>(instr.s1)],
                                                values[static_cast<std::size_t>(instr.s2)]) &
                            all_examples_mask);
        values.push_back(value);
    }
    return values;
}

std::vector<PackedMask> evaluate_program_masks(const Program &program, std::span<const PackedMask> input_masks,
                                               const PackedMask &all_examples_mask) {
    const std::vector<PackedMask> values = evaluate_all_sources(program, input_masks, all_examples_mask);
    std::vector<PackedMask> outputs;
    outputs.reserve(program.outputs.size());
    for (const int selector : program.outputs) {
        outputs.push_back(values[static_cast<std::size_t>(selector)]);
    }
    return outputs;
}

bool verify_program_packed_ok(const Program &program, const PackedExamples &packed, int num_outputs) {
    if (packed.width == 0) return true;
    const PackedMask full_mask = all_ones(packed.width);
    const std::vector<PackedMask> actual = evaluate_program_masks(program, packed.input_masks, full_mask);
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
        const PackedMask care_mask = full_mask ^ packed.output_dont_care_masks[out_idx];
        if (!((actual[out_idx] ^ packed.output_values[out_idx]) & care_mask).is_zero()) {
            return false;
        }
    }
    return true;
}

std::vector<std::size_t> verify_program_packed(const Program &program, const PackedExamples &packed, int num_outputs) {
    if (packed.width == 0) return {};
    const PackedMask full_mask = all_ones(packed.width);
    const std::vector<PackedMask> actual = evaluate_program_masks(program, packed.input_masks, full_mask);
    PackedMask mismatch_mask(packed.width);
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
        const PackedMask care_mask = full_mask ^ packed.output_dont_care_masks[out_idx];
        mismatch_mask |= ((actual[out_idx] ^ packed.output_values[out_idx]) & care_mask);
    }
    return mismatch_mask.set_bit_indices();
}

std::vector<std::size_t> verify_program(const Program &program, std::span<const Example> examples, int num_inputs,
                                        int num_outputs) {
    if (examples.empty()) return {};
    const PackedExamples packed = pack_examples(examples, num_inputs, num_outputs);
    return verify_program_packed(program, packed, num_outputs);
}

void emit_program(std::ostream &out, const Program &program, int num_inputs) {
    for (std::size_t instr_idx = 0; instr_idx < program.instrs.size(); ++instr_idx) {
        const auto &instr = program.instrs[instr_idx];
        out << "T" << instr_idx << ": " << op_label(instr.op) << "(" << format_source(instr.s1, num_inputs) << ", "
            << format_source(instr.s2, num_inputs) << ")\n";
    }
    for (std::size_t out_idx = 0; out_idx < program.outputs.size(); ++out_idx) {
        out << "OUT" << out_idx << ": " << format_source(program.outputs[out_idx], num_inputs) << "\n";
    }
}

Program parse_program(std::istream &in, int num_inputs, int num_outputs) {
    Program program;
    program.outputs.assign(static_cast<std::size_t>(num_outputs), -1);
    std::string raw_line;
    while (std::getline(in, raw_line)) {
        const std::string_view line = trim_view(raw_line);
        if (line.empty() || line.front() == '#') continue;
        const auto colon = line.find(':');
        if (colon == std::string_view::npos) {
            throw std::runtime_error("Invalid program line (missing colon): " + std::string(line));
        }
        const std::string_view lhs = trim_view(line.substr(0, colon));
        const std::string_view rhs = trim_view(line.substr(colon + 1));
        if (lhs.starts_with("T")) {
            const auto lparen = rhs.find('(');
            const auto comma = rhs.find(',', lparen);
            const auto rparen = rhs.rfind(')');
            if (lparen == std::string_view::npos || comma == std::string_view::npos ||
                rparen == std::string_view::npos || rparen <= comma) {
                throw std::runtime_error("Invalid instruction syntax: " + std::string(line));
            }
            const std::string_view op_name = trim_view(rhs.substr(0, lparen));
            const std::string_view left_tok = trim_view(rhs.substr(lparen + 1, comma - lparen - 1));
            const std::string_view right_tok = trim_view(rhs.substr(comma + 1, rparen - comma - 1));
            program.instrs.push_back({
                .op = op_code_by_label(std::string(op_name)),
                .s1 = parse_source(left_tok, num_inputs),
                .s2 = parse_source(right_tok, num_inputs),
            });
        } else if (lhs.starts_with("OUT")) {
            const std::string sub(lhs.substr(3));
            std::size_t parsed = 0;
            int out_idx = 0;
            try {
                out_idx = std::stoi(sub, &parsed);
            } catch (const std::exception &) {
                throw std::runtime_error("Invalid output index: " + std::string(lhs));
            }
            if (parsed != sub.size() || out_idx < 0 || out_idx >= num_outputs) {
                throw std::runtime_error("Output index out of range: " + std::string(lhs));
            }
            program.outputs[static_cast<std::size_t>(out_idx)] = parse_source(rhs, num_inputs);
        } else {
            throw std::runtime_error("Unexpected program line prefix: " + std::string(line));
        }
    }
    for (std::size_t idx = 0; idx < program.outputs.size(); ++idx) {
        if (program.outputs[idx] < 0) {
            throw std::runtime_error("Output OUT" + std::to_string(idx) + " was not defined in program");
        }
    }
    return program;
}

Program parse_program(std::string_view text, int num_inputs, int num_outputs) {
    std::istringstream stream{std::string(text)};
    return parse_program(stream, num_inputs, num_outputs);
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
            if (instr.op != kOpXor) {
                throw std::runtime_error("Only XOR may use duplicate sources in BLIF output");
            }
            out << ".names T" << instr_idx << "\n";
        } else {
            out << ".names " << format_source(instr.s1, num_inputs) << " " << format_source(instr.s2, num_inputs)
                << " T" << instr_idx << "\n";
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
