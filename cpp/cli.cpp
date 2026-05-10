#include "cli.hpp"

#include "postprocess.hpp"

#include <cxxopts.hpp>

#include <cctype>
#include <cstdlib>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message);
}

std::string trim(std::string value) {
    auto first = value.begin();
    while (first != value.end() && std::isspace(static_cast<unsigned char>(*first)) != 0) {
        ++first;
    }
    auto last = value.end();
    while (last != first && std::isspace(static_cast<unsigned char>(*(last - 1))) != 0) {
        --last;
    }
    return {first, last};
}

std::vector<std::string> split_preserving_empty(const std::string &value, char delimiter) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t pos = value.find(delimiter, start);
        if (pos == std::string::npos) {
            parts.push_back(value.substr(start));
            break;
        }
        parts.push_back(value.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}

std::vector<PostProcessScorePhase> parse_post_process_score(const std::string &value) {
    std::vector<PostProcessScorePhase> phases;
    for (const std::string &raw_phase : split_preserving_empty(value, ';')) {
        const std::string phase_text = trim(raw_phase);
        if (phase_text.empty()) usage_error("--post-process-score contains an empty phase");
        PostProcessScorePhase phase;
        for (const std::string &raw_metric : split_preserving_empty(phase_text, ',')) {
            std::string metric_text = trim(raw_metric);
            if (metric_text.empty()) usage_error("--post-process-score contains an empty metric");
            bool descending = false;
            if (metric_text.front() == '-') {
                descending = true;
                metric_text.erase(metric_text.begin());
                metric_text = trim(metric_text);
            }
            if (metric_text.empty()) usage_error("--post-process-score contains an empty metric");
            try {
                phase.push_back({post_process_score_metric_by_name(metric_text), descending});
            } catch (const std::exception &) {
                usage_error("Unsupported --post-process-score metric: " + metric_text);
            }
        }
        phases.push_back(std::move(phase));
    }
    if (phases.empty()) usage_error("--post-process-score must specify at least one metric");
    return phases;
}

cxxopts::Options make_options() {
    cxxopts::Options options("sat_synth_cpp", "Synthesize a logic program with z3");
    // clang-format off
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
        ("post-process", "Run mask-only post-processing on the solved program")
        ("post-process-beam-width", "Maximum post-processing beam width", cxxopts::value<int>()->default_value("1"))
        ("post-process-beam-rounds", "Post-processing beam rounds; 0 means until no improvement", cxxopts::value<int>()->default_value("0"))
        ("post-process-beam-candidates", "Maximum neighbor candidates per beam state; 0 means unlimited", cxxopts::value<int>()->default_value("0"))
        ("post-process-score", "Comma-separated post-processing score metrics; use ; for phases and -metric for descending", cxxopts::value<std::string>()->default_value("program-length"))
        ("post-process-resynthesis-maxnodes", "Maximum local SAT resynthesis window size", cxxopts::value<int>()->default_value("5"))
        ("post-process-resynthesis-patience", "Maximum SAT resynthesis results per beam state; 0 means unlimited", cxxopts::value<int>()->default_value("1"))
        ("generator-timeout", "Maximum seconds spent in each post-process candidate generator; 0 means unlimited", cxxopts::value<double>()->default_value("0"))
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
        ("profile", "Print C++ phase timings and counters to stderr")
        ("h,help", "Print usage");
    // clang-format on
    return options;
}

}  // namespace

void print_usage(std::ostream &out) {
    const cxxopts::Options options = make_options();
    out << "Usage: sat_synth_cpp (--config PATH | --dataset NAME) [options]\n\n"
        << "C++ v1 supports explicit examples and built-in generated datasets.\n\n"
        << options.help();
}

CliOptions parse_args(int argc, char **argv) {
    cxxopts::Options parser = make_options();
    const cxxopts::ParseResult parsed = parser.parse(argc, argv);
    if (parsed.contains("help")) {
        print_usage(std::cout);
        std::exit(0);
    }

    CliOptions options;
    if (parsed.contains("config")) options.config_path = parsed["config"].as<std::string>();
    if (parsed.contains("dataset")) options.dataset_name = parsed["dataset"].as<std::string>();
    if (parsed.contains("assume")) options.assume_path = parsed["assume"].as<std::string>();
    if (parsed.contains("instructions")) options.instructions = parsed["instructions"].as<int>();
    options.solver = parsed["solver"].as<std::string>();
    options.encode_boolean = parsed.contains("encode-boolean");
    options.balanced_select = parsed.contains("balanced-select");
    options.force_ordered = parsed.contains("force-ordered");
    options.force_useful = parsed.contains("force-useful");
    if (parsed.contains("batch-size")) options.batch_size = parsed["batch-size"].as<std::size_t>();
    options.cegis = parsed.contains("cegis");
    options.cegis_initial_size = parsed["cegis-initial-size"].as<std::size_t>();
    options.cegis_counterexamples = parsed["cegis-counterexamples"].as<std::size_t>();
    options.no_shuffle = parsed.contains("no-shuffle");
    options.seed = parsed["seed"].as<unsigned>();
    options.quiet = parsed.contains("quiet");
    options.list_datasets = parsed.contains("list-datasets");
    options.dump_dataset = parsed.contains("dump-dataset");
    options.make_smt2 = parsed.contains("make-smt2");
    options.make_dimacs = parsed.contains("make-dimacs");
    options.make_blif = parsed.contains("make-blif");
    options.output_blif = parsed.contains("output-blif");
    options.profile = parsed.contains("profile");
    options.post_process = parsed.contains("post-process");
    const int post_process_beam_width = parsed["post-process-beam-width"].as<int>();
    const int post_process_beam_rounds = parsed["post-process-beam-rounds"].as<int>();
    const int post_process_beam_candidates = parsed["post-process-beam-candidates"].as<int>();
    const std::string post_process_score = parsed["post-process-score"].as<std::string>();
    const int post_process_resynthesis_maxnodes = parsed["post-process-resynthesis-maxnodes"].as<int>();
    const int post_process_resynthesis_patience = parsed["post-process-resynthesis-patience"].as<int>();
    const double generator_timeout = parsed["generator-timeout"].as<double>();

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
    if (post_process_beam_width < 1) usage_error("--post-process-beam-width must be at least 1");
    if (post_process_beam_rounds < 0) usage_error("--post-process-beam-rounds must be non-negative");
    if (post_process_beam_candidates < 0) usage_error("--post-process-beam-candidates must be non-negative");
    options.post_process_score_phases = parse_post_process_score(post_process_score);
    if (post_process_resynthesis_maxnodes < 2) usage_error("--post-process-resynthesis-maxnodes must be at least 2");
    if (post_process_resynthesis_patience < 0) usage_error("--post-process-resynthesis-patience must be non-negative");
    if (generator_timeout < 0.0) usage_error("--generator-timeout must be non-negative");
    options.post_process_beam_width = static_cast<std::size_t>(post_process_beam_width);
    options.post_process_beam_rounds = static_cast<std::size_t>(post_process_beam_rounds);
    options.post_process_beam_candidates = static_cast<std::size_t>(post_process_beam_candidates);
    options.post_process_resynthesis_maxnodes = static_cast<std::size_t>(post_process_resynthesis_maxnodes);
    options.post_process_resynthesis_patience = static_cast<std::size_t>(post_process_resynthesis_patience);
    options.generator_timeout = generator_timeout;
    return options;
}

void log_info(const CliOptions &options, const std::string &message) {
    if (!options.quiet) std::cerr << message << "\n";
}
