#include "cli.hpp"
#include "config.hpp"
#include "datasets.hpp"
#include "program.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;

struct TempFile {
    std::filesystem::path path;

    TempFile(const std::string &content, const std::string &suffix) {
        static std::atomic<unsigned long long> counter{0};
        const auto id = counter.fetch_add(1, std::memory_order_relaxed);
        path = std::filesystem::temp_directory_path() /
               ("sat_synth_integration_" + std::to_string(id) + suffix);
        std::ofstream out(path);
        REQUIRE(static_cast<bool>(out));
        out << content;
    }

    TempFile(const TempFile &) = delete;
    TempFile &operator=(const TempFile &) = delete;

    TempFile(TempFile &&other) noexcept = delete;
    TempFile &operator=(TempFile &&other) noexcept = delete;

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

struct CliRunResult {
    int returncode = 0;
    std::string stdout_text;
    std::string stderr_text;
};

CliRunResult invoke_cli_args(const std::vector<std::string> &args, const std::string &stdin_text = {}) {
    std::vector<std::string> full_args;
    full_args.reserve(args.size() + 1);
    full_args.emplace_back("sat_synth_cpp");
    for (const auto &arg : args) {
        full_args.emplace_back(arg);
    }
    std::vector<char *> argv;
    argv.reserve(full_args.size());
    for (auto &arg : full_args) {
        argv.emplace_back(arg.data());
    }

    std::istringstream in(stdin_text);
    std::ostringstream out;
    std::ostringstream err;
    const int rc = run_cli(static_cast<int>(argv.size()), argv.data(), in, out, err);
    return {.returncode = rc, .stdout_text = out.str(), .stderr_text = err.str()};
}

CliRunResult invoke_cli_config(const json &config_json, const std::vector<std::string> &extra_args = {},
                               const std::string &stdin_text = {}) {
    const TempFile cfg_file(config_json.dump(), ".json");
    std::vector<std::string> args = {"--config", cfg_file.path.string()};
    args.insert(args.end(), extra_args.begin(), extra_args.end());
    return invoke_cli_args(args, stdin_text);
}

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
        s.pop_back();
    }
    return s;
}

int parse_source_token(const std::string &raw, int num_inputs) {
    const std::string tok = trim_copy(raw);
    if (tok == "1") return kSourceConstantOne;
    if (tok.starts_with("I")) return input_source(std::stoi(tok.substr(1)));
    if (tok.starts_with("T")) return temp_source(std::stoi(tok.substr(1)), num_inputs);
    FAIL("Unexpected source token: " << tok);
    return 0;
}

Program parse_emitted_program(const std::string &text, int num_inputs, int num_outputs) {
    Program program;
    program.outputs.assign(static_cast<std::size_t>(num_outputs), -1);
    std::istringstream lines(text);
    std::string raw_line;
    while (std::getline(lines, raw_line)) {
        const std::string line = trim_copy(raw_line);
        if (line.empty()) continue;
        const auto colon = line.find(':');
        REQUIRE(colon != std::string::npos);
        const std::string lhs = trim_copy(line.substr(0, colon));
        const std::string rhs = trim_copy(line.substr(colon + 1));
        if (lhs.starts_with("T")) {
            const auto lparen = rhs.find('(');
            const auto comma = rhs.find(',', lparen);
            const auto rparen = rhs.rfind(')');
            REQUIRE(lparen != std::string::npos);
            REQUIRE(comma != std::string::npos);
            REQUIRE(rparen != std::string::npos);
            const std::string op_name = trim_copy(rhs.substr(0, lparen));
            const std::string left_tok = trim_copy(rhs.substr(lparen + 1, comma - lparen - 1));
            const std::string right_tok = trim_copy(rhs.substr(comma + 1, rparen - comma - 1));
            program.instrs.push_back({
                .op = op_code_by_label(op_name),
                .s1 = parse_source_token(left_tok, num_inputs),
                .s2 = parse_source_token(right_tok, num_inputs),
            });
        } else if (lhs.starts_with("OUT")) {
            const int out_idx = std::stoi(lhs.substr(3));
            REQUIRE(out_idx >= 0);
            REQUIRE(out_idx < num_outputs);
            program.outputs[static_cast<std::size_t>(out_idx)] = parse_source_token(rhs, num_inputs);
        } else {
            FAIL("Unexpected program line: " << line);
        }
    }
    for (int out_src : program.outputs) {
        REQUIRE(out_src >= 0);
    }
    return program;
}

std::vector<Example> examples_from_json(const json &examples_json) {
    std::vector<Example> result;
    for (const auto &ex : examples_json) {
        Example item;
        for (const auto &v : ex.at("inputs")) {
            item.push_input(v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0));
        }
        for (const auto &v : ex.at("outputs")) {
            if (v.is_null()) {
                item.push_output(std::nullopt);
            } else {
                item.push_output(v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0));
            }
        }
        result.emplace_back(std::move(item));
    }
    return result;
}

void assert_cli_solves(const json &config_json, const std::vector<std::string> &extra_args = {}) {
    const CliRunResult res = invoke_cli_config(config_json, extra_args);
    INFO("stderr: " << res.stderr_text);
    REQUIRE(res.returncode == 0);
    const int num_inputs = config_json.at("num_inputs").get<int>();
    const int num_outputs = config_json.at("num_outputs").get<int>();
    const Program program = parse_emitted_program(res.stdout_text, num_inputs, num_outputs);
    const std::vector<Example> examples = examples_from_json(config_json.at("examples"));
    REQUIRE(verify_program(program, examples, num_inputs, num_outputs).empty());
}

json make_xor_config(int instructions = 1) {
    return {
        {"num_inputs", 2},
        {"num_outputs", 1},
        {"instructions", instructions},
        {"examples",
         {
             {{"inputs", {false, false}}, {"outputs", {false}}},
             {{"inputs", {false, true}}, {"outputs", {true}}},
             {{"inputs", {true, false}}, {"outputs", {true}}},
             {{"inputs", {true, true}}, {"outputs", {false}}},
         }},
    };
}

}  // namespace

TEST_CASE("CLI --help prints usage and returns 0") {
    const CliRunResult res = invoke_cli_args({"--help"});
    REQUIRE(res.returncode == 0);
    REQUIRE(!res.stdout_text.contains("Usage: sat_synth_cpp"));
}

TEST_CASE("CLI solves XOR with CEGIS") {
    assert_cli_solves(make_xor_config(1),
                      {"--cegis", "--cegis-initial-size", "1", "--cegis-counterexamples", "1", "--no-shuffle"});
}

TEST_CASE("CLI solves don't-care output config") {
    const json config = {
        {"num_inputs", 1},
        {"num_outputs", 1},
        {"instructions", 0},
        {"examples", {{{"inputs", {false}}, {"outputs", {nullptr}}}}},
    };
    assert_cli_solves(config);
}

TEST_CASE("CLI solves with encoding flags") {
    const json config = {
        {"num_inputs", 2},
        {"num_outputs", 1},
        {"instructions", 1},
        {"examples",
         {
             {{"inputs", {false, false}}, {"outputs", {false}}},
             {{"inputs", {false, true}}, {"outputs", {true}}},
             {{"inputs", {true, false}}, {"outputs", {true}}},
             {{"inputs", {true, true}}, {"outputs", {true}}},
         }},
    };
    assert_cli_solves(config, {"--force-ordered"});
    assert_cli_solves(config, {"--force-useful"});
    assert_cli_solves(config, {"--encode-boolean"});
    assert_cli_solves(config, {"--balanced-select"});
}

TEST_CASE("CLI UNSAT returns non-zero exit code") {
    const json config = {
        {"num_inputs", 1},
        {"num_outputs", 1},
        {"instructions", 0},
        {"examples",
         {
             {{"inputs", {false}}, {"outputs", {false}}},
             {{"inputs", {false}}, {"outputs", {true}}},
         }},
    };
    const CliRunResult res = invoke_cli_config(config);
    REQUIRE(res.returncode != 0);
}

TEST_CASE("CLI solves generated adder config and built-in dataset") {
    const json config = {{"type", "adder"}, {"num_inputs", 2}, {"instructions", 2}};
    const Config expected_cfg = build_dataset_from_config(config);
    const CliRunResult res = invoke_cli_config(config, {"--no-shuffle"});
    INFO("stderr: " << res.stderr_text);
    REQUIRE(res.returncode == 0);
    const Program prog = parse_emitted_program(res.stdout_text, expected_cfg.num_inputs, expected_cfg.num_outputs);
    REQUIRE(verify_program(prog, expected_cfg.examples, expected_cfg.num_inputs, expected_cfg.num_outputs).empty());

    const Config builtin_adder = build_dataset_from_config(default_dataset_config("adder"));
    const CliRunResult dataset_res = invoke_cli_args(
        {"--dataset", "adder", "--cegis", "--cegis-initial-size", "2", "--cegis-counterexamples", "2", "--no-shuffle"});
    INFO("stderr: " << dataset_res.stderr_text);
    REQUIRE(dataset_res.returncode == 0);
    const Program builtin_prog =
        parse_emitted_program(dataset_res.stdout_text, builtin_adder.num_inputs, builtin_adder.num_outputs);
    REQUIRE(verify_program(builtin_prog, builtin_adder.examples, builtin_adder.num_inputs, builtin_adder.num_outputs)
                .empty());
}

TEST_CASE("CLI handles --assume from file, stdin, contradictory assumptions, and missing file") {
    const json config = make_xor_config(1);

    SECTION("assume file") {
        const TempFile assume_file("# fixed xor\nT0: XOR(I0, I1)\nOUT0: T0\n", ".txt");
        assert_cli_solves(config, {"--assume", assume_file.path.string()});
    }

    SECTION("assume stdin") {
        const CliRunResult res = invoke_cli_config(config, {"--assume", "-"}, "T0: XOR(I0, I1)\nOUT0: T0\n");
        INFO("stderr: " << res.stderr_text);
        REQUIRE(res.returncode == 0);
        const Program prog = parse_emitted_program(res.stdout_text, 2, 1);
        const std::vector<Example> examples = examples_from_json(config.at("examples"));
        REQUIRE(verify_program(prog, examples, 2, 1).empty());
    }

    SECTION("contradictory assumptions return non-zero") {
        const TempFile assume_file("T0: AND(I0, I0)\nOUT0: T0\n", ".txt");
        const CliRunResult res = invoke_cli_config(config, {"--assume", assume_file.path.string()});
        REQUIRE(res.returncode != 0);
    }

    SECTION("missing assume file returns exit code 2") {
        const CliRunResult res = invoke_cli_config(config, {"--assume", "/tmp/missing-sat-synth-assume-file.txt"});
        REQUIRE(res.returncode == 2);
        REQUIRE(!res.stderr_text.contains("Assume file not found"));
    }
}

TEST_CASE("CLI exports SMT2, DIMACS, and BLIF formats and validates export constraints") {
    const json config = make_xor_config(1);

    SECTION("--make-smt2 outputs SMT-LIB2") {
        const CliRunResult res = invoke_cli_config(config, {"--make-smt2", "--batch-size", "2", "--no-shuffle"});
        REQUIRE(res.returncode == 0);
        REQUIRE(!res.stdout_text.contains("OP_0"));
        REQUIRE(!res.stdout_text.contains("OUT_0_idx"));
        REQUIRE(!res.stdout_text.contains("assert"));
        REQUIRE(!res.stdout_text.contains("b0"));
        REQUIRE(!res.stdout_text.contains("b1"));
        REQUIRE(res.stdout_text.contains("T0:"));
    }

    SECTION("--make-smt2 accepts assumptions") {
        const TempFile assume_file("T0: XOR(I0, I1)\nOUT0: T0\n", ".txt");
        const CliRunResult res = invoke_cli_config(config, {"--make-smt2", "--assume", assume_file.path.string()});
        REQUIRE(res.returncode == 0);
        REQUIRE(!res.stdout_text.contains("OP_0"));
        REQUIRE(!res.stdout_text.contains("OUT_0_idx"));
    }

    SECTION("--cegis with --make-smt2 or --make-dimacs is rejected") {
        const CliRunResult smt2_res = invoke_cli_config(config, {"--cegis", "--make-smt2"});
        REQUIRE(smt2_res.returncode == 2);
        REQUIRE(smt2_res.stderr_text.find("--cegis cannot be combined with --make-smt2 or --make-dimacs") !=
                std::string::npos);

        const CliRunResult dimacs_res = invoke_cli_config(config, {"--cegis", "--make-dimacs"});
        REQUIRE(dimacs_res.returncode == 2);
        REQUIRE(dimacs_res.stderr_text.find("--cegis cannot be combined with --make-smt2 or --make-dimacs") !=
                std::string::npos);
    }

    SECTION("--make-dimacs outputs CNF") {
        const CliRunResult res = invoke_cli_config(config, {"--make-dimacs", "--batch-size", "2", "--no-shuffle"});
        REQUIRE(res.returncode == 0);
        REQUIRE(!res.stdout_text.contains("p cnf "));
        REQUIRE(res.stdout_text.contains("T0:"));
    }

    SECTION("--make-blif outputs truth table and rejects don't-care outputs") {
        const CliRunResult res = invoke_cli_config(config, {"--make-blif"});
        REQUIRE(res.returncode == 0);
        REQUIRE(!res.stdout_text.contains(".model synth_program"));
        REQUIRE(!res.stdout_text.contains(".inputs I0 I1"));
        REQUIRE(!res.stdout_text.contains(".outputs OUT0"));
        REQUIRE(!res.stdout_text.contains("01 1"));
        REQUIRE(!res.stdout_text.contains("10 1"));
        REQUIRE(res.stdout_text.contains("T0:"));

        const json dc_config = {
            {"num_inputs", 1},
            {"num_outputs", 1},
            {"instructions", 0},
            {"examples", {{{"inputs", {false}}, {"outputs", {nullptr}}}}},
        };
        const CliRunResult dc_res = invoke_cli_config(dc_config, {"--make-blif"});
        REQUIRE(dc_res.returncode == 2);
        REQUIRE(!dc_res.stderr_text.contains("Cannot export BLIF with don't-care outputs"));
    }

    SECTION("--output-blif emits solved program") {
        const CliRunResult res =
            invoke_cli_config(config, {"--output-blif", "--assume", "-"}, "T0: XOR(I0, I1)\nOUT0: T0\n");
        REQUIRE(res.returncode == 0);
        REQUIRE(!res.stdout_text.contains(".model spec"));
        REQUIRE(!res.stdout_text.contains(".names I0 I1 T0"));
        REQUIRE(!res.stdout_text.contains("10 1"));
        REQUIRE(!res.stdout_text.contains("01 1"));
        REQUIRE(!res.stdout_text.contains(".names T0 OUT0"));
        REQUIRE(res.stdout_text.contains("T0:"));
    }
}

TEST_CASE("CLI post-processing workflows shorten programs and preserve semantics") {
    const json xor2 = make_xor_config(2);
    const std::vector<Example> xor_examples = examples_from_json(xor2.at("examples"));

    SECTION("mask post-processing shortens redundant assumed program") {
        const CliRunResult res = invoke_cli_config(
            xor2, {"--assume", "-", "--post-process", "--no-shuffle"}, "T0: XOR(I0, I1)\nT1: AND(1, T0)\nOUT0: T1\n");
        REQUIRE(res.returncode == 0);
        const Program prog = parse_emitted_program(res.stdout_text, 2, 1);
        REQUIRE(prog.instrs.size() < 2);
        REQUIRE(verify_program(prog, xor_examples, 2, 1).empty());
    }

    SECTION("--output-blif uses post-processed program") {
        const CliRunResult res = invoke_cli_config(xor2, {"--output-blif", "--post-process", "--assume", "-"},
                                                   "T0: XOR(I0, I1)\nT1: AND(1, T0)\nOUT0: T1\n");
        REQUIRE(res.returncode == 0);
        REQUIRE(!res.stdout_text.contains(".model spec"));
        REQUIRE(!res.stdout_text.contains(".names I0 I1 T0"));
        REQUIRE(!res.stdout_text.contains(".names T0 OUT0"));
        REQUIRE(res.stdout_text.contains("T1"));
        REQUIRE(res.stdout_text.contains("T0:"));
    }

    SECTION("resynthesis post-processing shortens assumed program and works with --output-blif") {
        const json resynth_cfg = {
            {"num_inputs", 3},
            {"num_outputs", 1},
            {"instructions", 2},
            {"examples",
             {
                 {{"inputs", {false, false, false}}, {"outputs", {false}}},
                 {{"inputs", {false, false, true}}, {"outputs", {true}}},
                 {{"inputs", {false, true, false}}, {"outputs", {false}}},
                 {{"inputs", {false, true, true}}, {"outputs", {true}}},
                 {{"inputs", {true, false, true}}, {"outputs", {true}}},
                 {{"inputs", {true, true, false}}, {"outputs", {true}}},
                 {{"inputs", {true, true, true}}, {"outputs", {true}}},
             }},
        };
        const std::vector<Example> resynth_examples = examples_from_json(resynth_cfg.at("examples"));
        const CliRunResult res = invoke_cli_config(
            resynth_cfg,
            {"--assume", "-", "--post-process", "--post-process-resynthesis-maxnodes", "2", "--no-shuffle"},
            "T0: AND(I0, I1)\nT1: OR(I2, T0)\nOUT0: T1\n");
        REQUIRE(res.returncode == 0);
        const Program prog = parse_emitted_program(res.stdout_text, 3, 1);
        REQUIRE(prog.instrs.size() < 2);
        REQUIRE(verify_program(prog, resynth_examples, 3, 1).empty());

        const CliRunResult blif_res = invoke_cli_config(
            resynth_cfg,
            {"--output-blif", "--post-process", "--post-process-resynthesis-maxnodes", "2", "--assume", "-"},
            "T0: AND(I0, I1)\nT1: OR(I2, T0)\nOUT0: T1\n");
        REQUIRE(blif_res.returncode == 0);
        REQUIRE(!blif_res.stdout_text.contains(".model spec"));
        REQUIRE(!blif_res.stdout_text.contains(".names T0 OUT0"));
        REQUIRE(blif_res.stdout_text.contains("T1"));
    }

    SECTION("score phases, replace patience, and random score reproducibility") {
        const json id_cfg = {
            {"num_inputs", 1},
            {"num_outputs", 1},
            {"instructions", 1},
            {"examples",
             {
                 {{"inputs", {false}}, {"outputs", {false}}},
                 {{"inputs", {true}}, {"outputs", {true}}},
             }},
        };
        const CliRunResult phase_res = invoke_cli_config(
            id_cfg,
            {"--assume", "-", "--post-process", "--post-process-score", "-program-length;program-length",
             "--no-shuffle"},
            "T0: AND(1, I0)\nOUT0: T0\n");
        REQUIRE(phase_res.returncode == 0);
        const Program id_prog = parse_emitted_program(phase_res.stdout_text, 1, 1);
        REQUIRE(id_prog.instrs.empty());

        const CliRunResult rand1 = invoke_cli_config(
            xor2,
            {"--assume", "-", "--post-process", "--post-process-score", "random", "--post-process-beam-rounds", "1",
             "--seed", "19", "--no-shuffle"},
            "T0: XOR(I0, I1)\nT1: AND(1, T0)\nOUT0: T1\n");
        const CliRunResult rand2 = invoke_cli_config(
            xor2,
            {"--assume", "-", "--post-process", "--post-process-score", "random", "--post-process-beam-rounds", "1",
             "--seed", "19", "--no-shuffle"},
            "T0: XOR(I0, I1)\nT1: AND(1, T0)\nOUT0: T1\n");
        REQUIRE(rand1.returncode == 0);
        REQUIRE(rand2.returncode == 0);
        REQUIRE(rand1.stdout_text == rand2.stdout_text);
    }
}

TEST_CASE("CLI dataset listing, dumping, and validation") {
    const CliRunResult list_res = invoke_cli_args({"--list-datasets"});
    REQUIRE(list_res.returncode == 0);
    REQUIRE(!list_res.stdout_text.contains("adder\n"));
    REQUIRE(!list_res.stdout_text.contains("gol\n"));
    REQUIRE(!list_res.stdout_text.contains("traffic\n"));

    for (const std::string &name : {"adder", "gol", "gol1", "gol2", "sloppy-adder", "sloppy-adder3", "life-compressed",
                                    "maze-compressed", "brian-compressed", "fire-compressed", "wire-compressed",
                                    "excitable-compressed", "cyclic-compressed", "critters", "traffic"}) {
        const CliRunResult dump_res = invoke_cli_args({"--dataset", name, "--dump-dataset"});
        INFO("dataset=" << name << " stderr=" << dump_res.stderr_text);
        REQUIRE(dump_res.returncode == 0);
        const json dumped = json::parse(dump_res.stdout_text);
        REQUIRE(dumped.contains("num_inputs"));
        REQUIRE(dumped.contains("num_outputs"));
        REQUIRE(dumped.contains("examples"));
        REQUIRE_FALSE(dumped.at("examples").empty());
    }

    for (const json &bad_cfg : {
             json{{"type", "life"}, {"life", {{"max_examples", 4}, {"seed", 0}}}},
             json{{"type", "excitable"}, {"excitable", {{"states", 3}, {"max_examples", 4}, {"seed", 0}}}},
             json{{"type", "fluid"}, {"fluid", {{"max_examples", 4}, {"seed", 0}}}},
         }) {
        const CliRunResult reject_res = invoke_cli_config(bad_cfg, {"--dump-dataset"});
        REQUIRE(reject_res.returncode != 0);
        REQUIRE(!reject_res.stderr_text.contains("max_examples sampling is not supported"));
    }
}

TEST_CASE("CLI profiling, verbosity, and symbolic CEGIS end-to-end") {
    const json xor1 = make_xor_config(1);

    SECTION("default, quiet, and verbose logging") {
        const CliRunResult def_res = invoke_cli_config(xor1, {"--no-shuffle"});
        REQUIRE(def_res.returncode == 0);
        REQUIRE(!def_res.stderr_text.contains("Built program structure with 1 instructions"));
        REQUIRE(!def_res.stderr_text.contains("SAT in "));

        const CliRunResult quiet_res = invoke_cli_config(xor1, {"--quiet", "--profile", "--no-shuffle"});
        REQUIRE(quiet_res.returncode == 0);
        REQUIRE(quiet_res.stderr_text.contains("Built program structure"));
        REQUIRE(quiet_res.stderr_text.contains("SAT in "));
        REQUIRE(!quiet_res.stderr_text.contains("PROFILE dataset_generation_seconds="));

        const CliRunResult verb_res = invoke_cli_config(
            xor1, {"-v", "--cegis", "--cegis-initial-size", "1", "--cegis-counterexamples", "1", "--no-shuffle"});
        REQUIRE(verb_res.returncode == 0);
        REQUIRE(!verb_res.stderr_text.contains("Loaded configuration with 4 examples"));
        REQUIRE(!verb_res.stderr_text.contains("Solving with solver="));
        REQUIRE(!verb_res.stderr_text.contains("CEGIS starting with 1/4 examples"));
        REQUIRE(!verb_res.stderr_text.contains("CEGIS added "));
    }

    SECTION("trace logging and profile counters for post-processing") {
        const json xor2 = make_xor_config(2);
        const CliRunResult res = invoke_cli_config(
            xor2,
            {"-vv", "--assume", "-", "--post-process", "--post-process-resynthesis-maxnodes", "2", "--profile",
             "--no-shuffle"},
            "T0: XOR(I0, I1)\nT1: AND(1, T0)\nOUT0: T1\n");
        REQUIRE(res.returncode == 0);
        REQUIRE(!res.stderr_text.contains("Post-process phase 1/1"));
        REQUIRE(!res.stderr_text.contains("Post-process round 1"));
        REQUIRE(!res.stderr_text.contains("PROFILE post_processing_runs=1"));
        REQUIRE(!res.stderr_text.contains("PROFILE post_processing_input_instructions=2"));
        REQUIRE(!res.stderr_text.contains("PROFILE post_processing_output_instructions=1"));
        REQUIRE(std::regex_search(res.stderr_text,
                                  std::regex("PROFILE post_processing_mask_candidates_considered=[1-9]")));
        REQUIRE(std::regex_search(res.stderr_text,
                                  std::regex("PROFILE post_processing_resynthesis_windows_considered=[1-9]")));
    }

    SECTION("symbolic CEGIS and post-processing via --max-materialized-examples") {
        const Config full_adder = build_dataset_from_config(default_dataset_config("adder"));
        const CliRunResult res = invoke_cli_args({
            "--dataset",
            "adder",
            "--instructions",
            "6",
            "--max-materialized-examples",
            "2",
            "--cegis-initial-size",
            "2",
            "--post-process",
            "--profile",
        });
        INFO("stderr: " << res.stderr_text);
        REQUIRE(res.returncode == 0);
        REQUIRE(std::regex_search(res.stderr_text, std::regex("PROFILE symbolic_verifier_checks=[1-9]")));
        const Program prog = parse_emitted_program(res.stdout_text, full_adder.num_inputs, full_adder.num_outputs);
        REQUIRE(verify_program(prog, full_adder.examples, full_adder.num_inputs, full_adder.num_outputs).empty());
    }
}
