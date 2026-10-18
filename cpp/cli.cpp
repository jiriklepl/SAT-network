#include "cli.hpp"

#include "assumptions.hpp"
#include "config.hpp"
#include "datasets.hpp"
#include "logging.hpp"
#include "postprocess.hpp"
#include "profile.hpp"
#include "program.hpp"
#include "solver.hpp"

#include <cxxopts.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message);
}

std::string trim(std::string_view value) {
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
                phase.push_back({.metric=post_process_score_metric_by_name(metric_text), .descending=descending});
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
        ("post-process-replace-patience", "Accepted replacement-search candidates per instruction; 0 means unlimited", cxxopts::value<int>()->default_value("50"))
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
        ("do-all", "Synthesize for all example batches at once")
        ("cegis", "Use counterexample-guided solving")
        ("cegis-initial-size", "Initial number of examples for --cegis", cxxopts::value<std::size_t>()->default_value("64"))
        ("cegis-counterexamples", "Maximum counterexamples to add per CEGIS iteration", cxxopts::value<std::size_t>()->default_value("1"))
        ("max-materialized-examples", "Largest quantified dataset to unroll into explicit examples; bigger domains are solved symbolically with CEGIS", cxxopts::value<std::size_t>())
        ("no-shuffle", "Disable shuffling of examples")
        ("seed", "Seed for shuffling examples", cxxopts::value<unsigned>()->default_value("0"))
        ("quiet", "Suppress informational output")
        ("v,verbose", "Increase human-readable progress logging; repeat for trace output")
        ("profile", "Print C++ phase timings and counters to stderr")
        ("h,help", "Print usage");
    // clang-format on
    return options;
}

bool exports_full_dataset(const CliOptions &cli) {
    return cli.dump_dataset || cli.make_blif || cli.make_smt2 || cli.make_dimacs;
}

Config load_requested_config(const CliOptions &cli) {
    DatasetLoadOptions load_options;
    load_options.max_materialized_examples =
        cli.max_materialized_examples.value_or(exports_full_dataset(cli) ? kExportMaxMaterializedExamples
                                                                         : kDefaultMaxMaterializedExamples);
    load_options.basis_size = std::max(cli.cegis_initial_size, static_cast<std::size_t>(kDefaultBasisSize));
    load_options.seed = cli.seed;
    Config cfg = cli.config_path.empty()
                     ? build_dataset_from_config(default_dataset_config(cli.dataset_name), load_options)
                     : load_config(cli.config_path, load_options);
    if (exports_full_dataset(cli) && !cfg.fully_materialized) {
        throw std::runtime_error("the quantified dataset has " + std::to_string(cfg.domain_size) +
                                 " rows, which exceeds --max-materialized-examples (" +
                                 std::to_string(load_options.max_materialized_examples) +
                                 "); it cannot be exported without unrolling");
    }
    if (cli.instructions.has_value()) cfg.instructions = *cli.instructions;
    if (cfg.instructions < 0) throw std::runtime_error("--instructions must be non-negative");
    return cfg;
}

SolveOptions make_solve_options(const CliOptions &cli, const Logger &logger) {
    return {
        .solver = cli.solver,
        .encoding = {.encode_boolean = cli.encode_boolean,
                     .force_ordered = cli.force_ordered,
                     .force_useful = cli.force_useful,
                     .balanced_select = cli.balanced_select},
        .assumptions = {},
        .batch_size = cli.batch_size,
        .do_all = cli.do_all,
        .cegis = cli.cegis,
        .cegis_initial_size = cli.cegis_initial_size,
        .cegis_counterexamples = cli.cegis_counterexamples,
        .postprocess =
            {
                .enabled = cli.post_process,
                .beam_width = cli.post_process_beam_width,
                .beam_rounds = cli.post_process_beam_rounds,
                .beam_candidates = cli.post_process_beam_candidates,
                .score_phases = cli.post_process_score_phases,
                .random_seed = cli.seed,
                .replace_patience = cli.post_process_replace_patience,
                .resynthesis_maxnodes = cli.post_process_resynthesis_maxnodes,
                .resynthesis_patience = cli.post_process_resynthesis_patience,
                .generator_timeout_seconds = cli.generator_timeout,
                .logger = &logger,
            },
        .logger = &logger,
    };
}

void print_profile(std::ostream &err, const ProfileData &profile) {
    err << std::fixed << std::setprecision(6);
    err << "PROFILE dataset_generation_seconds=" << profile.dataset_generation_seconds << "\n";
    err << "PROFILE structure_encoding_seconds=" << profile.structure_encoding_seconds << "\n";
    err << "PROFILE example_packing_seconds=" << profile.example_packing_seconds << "\n";
    err << "PROFILE example_encoding_seconds=" << profile.example_encoding_seconds << "\n";
    err << "PROFILE z3_solve_seconds=" << profile.z3_solve_seconds << "\n";
    err << "PROFILE model_extraction_seconds=" << profile.model_extraction_seconds << "\n";
    err << "PROFILE post_processing_seconds=" << profile.post_processing_seconds << "\n";
    err << "PROFILE post_processing_mask_generator_seconds=" << profile.post_processing_mask_generator_seconds << "\n";
    err << "PROFILE post_processing_replacement_generator_seconds="
        << profile.post_processing_replacement_generator_seconds << "\n";
    err << "PROFILE post_processing_resynthesis_generator_seconds="
        << profile.post_processing_resynthesis_generator_seconds << "\n";
    err << "PROFILE packed_verification_seconds=" << profile.packed_verification_seconds << "\n";
    err << "PROFILE symbolic_verifier_seconds=" << profile.symbolic_verifier_seconds << "\n";
    err << "PROFILE structure_constraints=" << profile.structure_constraints << "\n";
    err << "PROFILE example_constraints=" << profile.example_constraints << "\n";
    err << "PROFILE example_batches=" << profile.example_batches << "\n";
    err << "PROFILE packed_examples=" << profile.packed_examples << "\n";
    err << "PROFILE solver_checks=" << profile.solver_checks << "\n";
    err << "PROFILE model_extractions=" << profile.model_extractions << "\n";
    err << "PROFILE post_processing_runs=" << profile.post_processing_runs << "\n";
    err << "PROFILE post_processing_input_instructions=" << profile.post_processing_input_instructions << "\n";
    err << "PROFILE post_processing_output_instructions=" << profile.post_processing_output_instructions << "\n";
    err << "PROFILE post_processing_mask_candidates_considered="
        << profile.post_processing_mask_candidates_considered << "\n";
    err << "PROFILE post_processing_mask_candidates_materialized="
        << profile.post_processing_mask_candidates_materialized << "\n";
    err << "PROFILE post_processing_mask_candidates_accepted=" << profile.post_processing_mask_candidates_accepted
        << "\n";
    err << "PROFILE post_processing_mask_invalid_candidates=" << profile.post_processing_mask_invalid_candidates
        << "\n";
    err << "PROFILE post_processing_mask_timeout_exits=" << profile.post_processing_mask_timeout_exits << "\n";
    err << "PROFILE post_processing_replacement_candidates_considered="
        << profile.post_processing_replacement_candidates_considered << "\n";
    err << "PROFILE post_processing_replacement_candidates_materialized="
        << profile.post_processing_replacement_candidates_materialized << "\n";
    err << "PROFILE post_processing_replacement_candidates_accepted="
        << profile.post_processing_replacement_candidates_accepted << "\n";
    err << "PROFILE post_processing_replacement_invalid_candidates="
        << profile.post_processing_replacement_invalid_candidates << "\n";
    err << "PROFILE post_processing_replacement_timeout_exits="
        << profile.post_processing_replacement_timeout_exits << "\n";
    err << "PROFILE post_processing_resynthesis_windows_considered="
        << profile.post_processing_resynthesis_windows_considered << "\n";
    err << "PROFILE post_processing_resynthesis_windows_sat=" << profile.post_processing_resynthesis_windows_sat
        << "\n";
    err << "PROFILE post_processing_resynthesis_candidates_considered="
        << profile.post_processing_resynthesis_candidates_considered << "\n";
    err << "PROFILE post_processing_resynthesis_candidates_materialized="
        << profile.post_processing_resynthesis_candidates_materialized << "\n";
    err << "PROFILE post_processing_resynthesis_invalid_candidates="
        << profile.post_processing_resynthesis_invalid_candidates << "\n";
    err << "PROFILE post_processing_resynthesis_candidates_accepted="
        << profile.post_processing_resynthesis_candidates_accepted << "\n";
    err << "PROFILE post_processing_resynthesis_timeout_exits="
        << profile.post_processing_resynthesis_timeout_exits << "\n";
    err << "PROFILE verification_examples=" << profile.verification_examples << "\n";
    err << "PROFILE symbolic_verifier_checks=" << profile.symbolic_verifier_checks << "\n";
    err << "PROFILE symbolic_counterexamples=" << profile.symbolic_counterexamples << "\n";
    err << "PROFILE bv_cache_hits=" << profile.bv_cache_hits << "\n";
    err << "PROFILE bv_cache_misses=" << profile.bv_cache_misses << "\n";
}

Assumptions load_assumptions(const CliOptions &cli, const Config &cfg, std::istream &in) {
    if (cli.assume_path.empty()) {
        return {};
    }

    const ProgramSpec spec{
        .num_inputs = cfg.num_inputs, .num_outputs = cfg.num_outputs, .program_length = cfg.instructions};
    if (cli.assume_path == "-") {
        return parse_assumptions(in, spec);
    }

    std::ifstream file(cli.assume_path);
    if (!file) {
        throw std::runtime_error("Assume file not found: " + cli.assume_path);
    }
    return parse_assumptions(file, spec);
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

    CliOptions options;
    if (parsed.contains("help")) {
        options.help = true;
        return options;
    }

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
    options.do_all = parsed.contains("do-all");
    options.cegis = parsed.contains("cegis");
    options.cegis_initial_size = parsed["cegis-initial-size"].as<std::size_t>();
    options.cegis_counterexamples = parsed["cegis-counterexamples"].as<std::size_t>();
    if (parsed.contains("max-materialized-examples")) {
        options.max_materialized_examples = parsed["max-materialized-examples"].as<std::size_t>();
    }
    options.no_shuffle = parsed.contains("no-shuffle");
    options.seed = parsed["seed"].as<unsigned>();
    options.quiet = parsed.contains("quiet");
    options.verbosity = 1 + static_cast<int>(parsed.count("verbose"));
    if (options.quiet) options.verbosity = 0;
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
    const int post_process_replace_patience = parsed["post-process-replace-patience"].as<int>();
    const int post_process_resynthesis_maxnodes = parsed["post-process-resynthesis-maxnodes"].as<int>();
    const int post_process_resynthesis_patience = parsed["post-process-resynthesis-patience"].as<int>();
    const double generator_timeout = parsed["generator-timeout"].as<double>();

    if (options.list_datasets) return options;
    if (options.config_path.empty() == options.dataset_name.empty()) {
        usage_error("exactly one of --config or --dataset is required");
    }
    if (options.cegis && options.do_all) {
        usage_error("--cegis and --do-all are mutually exclusive");
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
    if (post_process_replace_patience < 0) usage_error("--post-process-replace-patience must be non-negative");
    if (post_process_resynthesis_maxnodes < 2) usage_error("--post-process-resynthesis-maxnodes must be at least 2");
    if (post_process_resynthesis_patience < 0) usage_error("--post-process-resynthesis-patience must be non-negative");
    if (generator_timeout < 0.0) usage_error("--generator-timeout must be non-negative");
    options.post_process_beam_width = static_cast<std::size_t>(post_process_beam_width);
    options.post_process_beam_rounds = static_cast<std::size_t>(post_process_beam_rounds);
    options.post_process_beam_candidates = static_cast<std::size_t>(post_process_beam_candidates);
    options.post_process_replace_patience = static_cast<std::size_t>(post_process_replace_patience);
    options.post_process_resynthesis_maxnodes = static_cast<std::size_t>(post_process_resynthesis_maxnodes);
    options.post_process_resynthesis_patience = static_cast<std::size_t>(post_process_resynthesis_patience);
    options.generator_timeout = generator_timeout;
    return options;
}

int run_cli(int argc, char **argv, std::istream &in, std::ostream &out, std::ostream &err) {
    try {
        const CliOptions cli = parse_args(argc, argv);
        if (cli.help) {
            print_usage(out);
            return 0;
        }
        const Logger logger(cli.verbosity, err);
        if (cli.list_datasets) {
            for (const auto &name : available_dataset_names()) {
                out << name << "\n";
            }
            return 0;
        }

        ProfileData profile;
        const auto dataset_start = Clock::now();
        Config cfg = load_requested_config(cli);
        if (cli.profile) profile.dataset_generation_seconds = elapsed_seconds(dataset_start);
        logger.log(2, "Loaded configuration with " + std::to_string(cfg.examples.size()) + " examples, " +
                          std::to_string(cfg.num_inputs) + " inputs, " + std::to_string(cfg.num_outputs) +
                          " outputs, and " + std::to_string(cfg.instructions) + " instructions");
        if (cli.dump_dataset) {
            out << config_to_json(cfg).dump(2) << "\n";
            if (cli.profile) print_profile(err, profile);
            return 0;
        }
        if (cli.make_blif) {
            export_spec_blif(out, cfg.examples, cfg.num_inputs, cfg.num_outputs);
            if (cli.profile) print_profile(err, profile);
            return 0;
        }

        if (!cli.no_shuffle) {
            std::mt19937 rng(cli.seed);
            std::ranges::shuffle(cfg.examples, rng);
        }

        SolveOptions solve_options = make_solve_options(cli, logger);
        if (cli.profile) solve_options.profile = &profile;
        solve_options.assumptions = load_assumptions(cli, cfg, in);

        if (cli.make_smt2) {
            out << make_smt2(cfg, solve_options);
            if (cli.profile) print_profile(err, profile);
            return 0;
        }
        if (cli.make_dimacs) {
            out << make_dimacs(cfg, solve_options);
            if (cli.profile) print_profile(err, profile);
            return 0;
        }

        logger.log(1, "Built program structure with " + std::to_string(cfg.instructions) + " instructions");
        const SolveResult result = solve_config(cfg, solve_options);
        if (result.status == SolveStatus::Sat) {
            if (cli.output_blif) {
                emit_program_blif(out, *result.program, cfg.num_inputs);
            } else {
                emit_program(out, *result.program, cfg.num_inputs);
            }
            logger.log(1, "SAT in " + std::to_string(result.elapsed_seconds) + " seconds");
            if (cli.profile) print_profile(err, profile);
            return 0;
        }
        if (result.status == SolveStatus::VerificationFailed) {
            if (result.program.has_value()) {
                if (cli.output_blif) {
                    emit_program_blif(out, *result.program, cfg.num_inputs);
                } else {
                    emit_program(out, *result.program, cfg.num_inputs);
                }
            }
            err << "Total mismatches: " << result.mismatch_count << "\n";
            if (cli.profile) print_profile(err, profile);
            return 1;
        }
        if (result.status == SolveStatus::Unsat) {
            logger.log(1, "UNSAT in " + std::to_string(result.elapsed_seconds) + " seconds");
            if (cli.profile) print_profile(err, profile);
            return 1;
        }
        logger.log(1, "UNKNOWN in " + std::to_string(result.elapsed_seconds) + " seconds");
        if (cli.profile) print_profile(err, profile);
        return 1;
    } catch (const std::exception &exc) {
        err << exc.what() << "\n";
        return 2;
    }
}
