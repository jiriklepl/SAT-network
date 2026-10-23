#include "assumptions.hpp"

#include "program.hpp"
#include "string_utils.hpp"

#include <cstddef>

#include <exception>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

int parse_index(std::string_view kind, std::string_view raw) {
    if (raw.empty()) {
        throw std::runtime_error("Invalid " + std::string(kind) + " index in assumption: " + std::string(kind) +
                                 std::string(raw));
    }
    std::size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(std::string(raw), &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("Invalid " + std::string(kind) + " index in assumption: " + std::string(kind) +
                                 std::string(raw));
    }
    if (parsed != raw.size()) {
        throw std::runtime_error("Invalid " + std::string(kind) + " index in assumption: " + std::string(kind) +
                                 std::string(raw));
    }
    return value;
}

std::pair<std::string_view, std::string_view> split_once(std::string_view line, char delimiter,
                                                         std::string_view error) {
    const std::size_t pos = line.find(delimiter);
    if (pos == std::string_view::npos) {
        throw std::runtime_error(std::string(error) + std::string(line));
    }
    return {trim_view(line.substr(0, pos)), trim_view(line.substr(pos + 1))};
}

InstructionAssumption parse_instruction(std::string_view lhs, std::string_view rhs, const ProgramSpec &spec) {
    const int instr_idx = parse_index("T", lhs.substr(1));
    if (instr_idx < 0 || instr_idx >= spec.program_length) {
        throw std::runtime_error("Temporary index out of range in assumption: " + std::string(lhs));
    }

    const std::size_t open = rhs.find('(');
    const std::size_t close = rhs.rfind(')');
    if (open == std::string_view::npos || close == std::string_view::npos || close < open || close != rhs.size() - 1) {
        throw std::runtime_error("Invalid instruction assumption: " + std::string(lhs) + ": " + std::string(rhs));
    }

    const std::string op_label_raw = trim_copy(rhs.substr(0, open));
    int op = 0;
    try {
        op = op_code_by_label(op_label_raw);
    } catch (const std::exception &) {
        throw std::runtime_error("Unknown operation in assumption: " + op_label_raw);
    }

    const std::string_view args = rhs.substr(open + 1, close - open - 1);
    const std::size_t comma = args.find(',');
    if (comma == std::string_view::npos || args.find(',', comma + 1) != std::string_view::npos) {
        throw std::runtime_error("Invalid instruction assumption: " + std::string(lhs) + ": " + std::string(rhs));
    }

    return {
        .instr_idx = instr_idx,
        .op = op,
        .s1 = parse_source(args.substr(0, comma), spec.num_inputs, spec.program_length),
        .s2 = parse_source(args.substr(comma + 1), spec.num_inputs, spec.program_length),
    };
}

OutputAssumption parse_output(std::string_view lhs, std::string_view rhs, const ProgramSpec &spec) {
    const int out_idx = parse_index("OUT", lhs.substr(3));
    if (out_idx < 0 || out_idx >= spec.num_outputs) {
        throw std::runtime_error("Output index out of range in assumption: " + std::string(lhs));
    }
    return {.out_idx = out_idx, .source = parse_source(rhs, spec.num_inputs, spec.program_length)};
}

}  // namespace

Assumptions parse_assumptions(std::istream &input, const ProgramSpec &spec) {
    Assumptions assumptions;
    std::string raw_line;
    while (std::getline(input, raw_line)) {
        const std::string_view line = trim_view(raw_line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        const auto [lhs, rhs] = split_once(line, ':', "Invalid assumption line: ");
        if (lhs.starts_with("T")) {
            assumptions.instructions.push_back(parse_instruction(lhs, rhs, spec));
        } else if (lhs.starts_with("OUT")) {
            assumptions.outputs.push_back(parse_output(lhs, rhs, spec));
        } else {
            throw std::runtime_error("Unknown LHS in assumption: " + std::string(lhs));
        }
    }
    return assumptions;
}
