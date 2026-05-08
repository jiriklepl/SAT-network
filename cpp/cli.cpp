#include "cli.hpp"

#include <cxxopts.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message);
}

cxxopts::Options make_options() {
    cxxopts::Options options("sat_synth_cpp", "Synthesize a logic program with z3");
    options.add_options()
        ("config", "Path to a JSON config", cxxopts::value<std::string>())
        ("dataset", "Choose a built-in dataset config", cxxopts::value<std::string>())
        ("assume", "Path to a text file with assumed program bits, or - for stdin", cxxopts::value<std::string>())
        ("list-datasets", "List built-in datasets")
        ("dump-dataset", "Print the generated dataset JSON and exit")
        ("make-smt2", "Output the problem in SMT-LIB2 format and exit")
        ("make-dimacs", "Output the problem in DIMACS CNF format and exit")
        ("make-blif", "Output the problem specification in BLIF format and exit")
        ("output-blif", "Output the found program in BLIF format")
        ("instructions", "Override number of SSA instructions", cxxopts::value<int>())
        ("solver", "Solver to use: z3, simple-tactic, ctx-simplify-tactic", cxxopts::value<std::string>()->default_value("simple-tactic"))
        ("encode-boolean", "Enable boolean source/output selection encoding")
        ("balanced-select", "Use balanced ITE trees for source/output selectors")
        ("force-ordered", "Enable ordered-instruction constraints")
        ("force-useful", "Require every instruction to feed a later instruction or output")
        ("batch-size", "Number of examples to add per encoded batch", cxxopts::value<std::size_t>())
        ("cegis", "Use counterexample-guided solving")
        ("cegis-initial-size", "Initial number of examples for --cegis", cxxopts::value<std::size_t>()->default_value("64"))
        ("cegis-counterexamples", "Maximum counterexamples to add per CEGIS iteration", cxxopts::value<std::size_t>()->default_value("1"))
        ("no-shuffle", "Disable shuffling of examples")
        ("seed", "Seed for shuffling examples", cxxopts::value<unsigned>()->default_value("0"))
        ("quiet", "Suppress informational output")
        ("h,help", "Print usage");
    return options;
}

}  // namespace

void print_usage(std::ostream &out) {
    cxxopts::Options options = make_options();
    out << "Usage: sat_synth_cpp (--config PATH | --dataset NAME) [options]\n\n"
        << "C++ v1 supports explicit examples and built-in generated datasets.\n\n"
        << options.help();
}

CliOptions parse_args(int argc, char **argv) {
    cxxopts::Options parser = make_options();
    cxxopts::ParseResult parsed = parser.parse(argc, argv);
    if (parsed.count("help")) {
        print_usage(std::cout);
        std::exit(0);
    }

    CliOptions options;
    if (parsed.count("config")) options.config_path = parsed["config"].as<std::string>();
    if (parsed.count("dataset")) options.dataset_name = parsed["dataset"].as<std::string>();
    if (parsed.count("assume")) options.assume_path = parsed["assume"].as<std::string>();
    if (parsed.count("instructions")) options.instructions = parsed["instructions"].as<int>();
    options.solver = parsed["solver"].as<std::string>();
    options.encode_boolean = parsed.count("encode-boolean") > 0;
    options.balanced_select = parsed.count("balanced-select") > 0;
    options.force_ordered = parsed.count("force-ordered") > 0;
    options.force_useful = parsed.count("force-useful") > 0;
    if (parsed.count("batch-size")) options.batch_size = parsed["batch-size"].as<std::size_t>();
    options.cegis = parsed.count("cegis") > 0;
    options.cegis_initial_size = parsed["cegis-initial-size"].as<std::size_t>();
    options.cegis_counterexamples = parsed["cegis-counterexamples"].as<std::size_t>();
    options.no_shuffle = parsed.count("no-shuffle") > 0;
    options.seed = parsed["seed"].as<unsigned>();
    options.quiet = parsed.count("quiet") > 0;
    options.list_datasets = parsed.count("list-datasets") > 0;
    options.dump_dataset = parsed.count("dump-dataset") > 0;
    options.make_smt2 = parsed.count("make-smt2") > 0;
    options.make_dimacs = parsed.count("make-dimacs") > 0;
    options.make_blif = parsed.count("make-blif") > 0;
    options.output_blif = parsed.count("output-blif") > 0;

    if (options.list_datasets) return options;
    if (options.config_path.empty() == options.dataset_name.empty()) {
        usage_error("exactly one of --config or --dataset is required");
    }
    if (options.cegis && (options.make_smt2 || options.make_dimacs)) {
        usage_error("--cegis cannot be combined with --make-smt2 or --make-dimacs");
    }
    if (options.batch_size.has_value() && *options.batch_size == 0) usage_error("--batch-size must be positive");
    if (options.cegis_initial_size == 0) usage_error("--cegis-initial-size must be positive");
    if (options.cegis_counterexamples == 0) usage_error("--cegis-counterexamples must be positive");
    return options;
}

void log_info(const CliOptions &options, const std::string &message) {
    if (!options.quiet) std::cerr << message << "\n";
}
