#include "assumptions.hpp"

#include "program.hpp"

#include <cctype>
#include <cstddef>

#include <algorithm>
#include <exception>
#include <istream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::string trim(const std::string &value) {
    const auto begin = std::ranges::find_if_not(value, [](unsigned char ch) { return std::isspace(ch) != 0; });
    const auto end = std::ranges::find_if_not(std::ranges::reverse_view(value), [](unsigned char ch) {
                         return std::isspace(ch) != 0;
                     }).base();
    if (begin >= end) return "";
    return {begin, end};
}

int parse_index(const std::string &kind, const std::string &raw) {
    if (raw.empty()) {
        throw std::runtime_error("Invalid " + kind + " index in assumption: " + kind + raw);
    }
    std::size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(raw, &parsed);
    } catch (const std::exception &exc) {
        throw std::runtime_error("Invalid " + kind + " index in assumption: " + kind + raw);
    }
    if (parsed != raw.size()) {
        throw std::runtime_error("Invalid " + kind + " index in assumption: " + kind + raw);
    }
    return value;
}

int translate_source(const std::string &raw_arg, const ProgramSpec &spec) {
    const std::string arg = trim(raw_arg);
    if (arg == "1") {
        return kSourceConstantOne;
    }
    if (!arg.empty() && arg[0] == 'I') {
        const int input_idx = parse_index("I", arg.substr(1));
        if (input_idx < 0 || input_idx >= spec.num_inputs) {
            throw std::runtime_error("Input index out of range in assumption: " + arg);
        }
        return input_source(input_idx);
    }
    if (!arg.empty() && arg[0] == 'T') {
        const int temp_idx = parse_index("T", arg.substr(1));
        if (temp_idx < 0 || temp_idx >= spec.program_length) {
            throw std::runtime_error("Temporary index out of range in assumption: " + arg);
        }
        return temp_source(temp_idx, spec.num_inputs);
    }
    throw std::runtime_error("Unknown argument in assumption: " + arg);
}

std::pair<std::string, std::string> split_once(const std::string &line, char delimiter, const std::string &error) {
    const std::size_t pos = line.find(delimiter);
    if (pos == std::string::npos) {
        throw std::runtime_error(error + line);
    }
    return {trim(line.substr(0, pos)), trim(line.substr(pos + 1))};
}

InstructionAssumption parse_instruction(const std::string &lhs, const std::string &rhs, const ProgramSpec &spec) {
    const int instr_idx = parse_index("T", lhs.substr(1));
    if (instr_idx < 0 || instr_idx >= spec.program_length) {
        throw std::runtime_error("Temporary index out of range in assumption: " + lhs);
    }

    const std::size_t open = rhs.find('(');
    const std::size_t close = rhs.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close < open || close != rhs.size() - 1) {
        throw std::runtime_error("Invalid instruction assumption: " + lhs + ": " + rhs);
    }

    const std::string op_label_raw = trim(rhs.substr(0, open));
    int op = 0;
    try {
        op = op_code_by_label(op_label_raw);
    } catch (const std::exception &) {
        throw std::runtime_error("Unknown operation in assumption: " + op_label_raw);
    }

    const std::string args = rhs.substr(open + 1, close - open - 1);
    const std::size_t comma = args.find(',');
    if (comma == std::string::npos || args.find(',', comma + 1) != std::string::npos) {
        throw std::runtime_error("Invalid instruction assumption: " + lhs + ": " + rhs);
    }

    return {
        .instr_idx = instr_idx,
        .op = op,
        .s1 = translate_source(args.substr(0, comma), spec),
        .s2 = translate_source(args.substr(comma + 1), spec),
    };
}

OutputAssumption parse_output(const std::string &lhs, const std::string &rhs, const ProgramSpec &spec) {
    const int out_idx = parse_index("OUT", lhs.substr(3));
    if (out_idx < 0 || out_idx >= spec.num_outputs) {
        throw std::runtime_error("Output index out of range in assumption: " + lhs);
    }
    return {.out_idx = out_idx, .source = translate_source(rhs, spec)};
}

}  // namespace

Assumptions parse_assumptions(std::istream &input, const ProgramSpec &spec) {
    Assumptions assumptions;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const auto [lhs, rhs] = split_once(line, ':', "Invalid assumption line: ");
        if (lhs.starts_with("T")) {
            assumptions.instructions.push_back(parse_instruction(lhs, rhs, spec));
        } else if (lhs.starts_with("OUT")) {
            assumptions.outputs.push_back(parse_output(lhs, rhs, spec));
        } else {
            throw std::runtime_error("Unknown LHS in assumption: " + lhs);
        }
    }
    return assumptions;
}
