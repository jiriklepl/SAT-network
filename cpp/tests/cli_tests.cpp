#include "cli.hpp"

#include <catch2/catch_test_macros.hpp>

#include <iterator>
#include <sstream>
#include <string>

TEST_CASE("CLI parses supported solver options") {
    const char *argv[] = {
        "sat_synth_cpp",
        "--dataset",
        "adder",
        "--assume",
        "program.txt",
        "--instructions",
        "2",
        "--solver",
        "z3",
        "--encode-boolean",
        "--balanced-select",
        "--force-ordered",
        "--force-useful",
        "--cegis",
        "--cegis-initial-size",
        "2",
        "--cegis-counterexamples",
        "2",
        "--no-shuffle",
        "--quiet",
        "--profile",
        "--post-process",
        "--post-process-beam-width",
        "3",
        "--post-process-beam-rounds",
        "4",
        "--post-process-beam-candidates",
        "5",
        "--post-process-resynthesis-maxnodes",
        "6",
        "--post-process-resynthesis-patience",
        "7",
        "--generator-timeout",
        "0.25",
    };
    CliOptions options = parse_args(static_cast<int>(std::size(argv)), const_cast<char **>(argv));
    REQUIRE(options.dataset_name == "adder");
    REQUIRE(options.assume_path == "program.txt");
    REQUIRE(options.instructions.has_value());
    REQUIRE(*options.instructions == 2);
    REQUIRE(options.solver == "z3");
    REQUIRE(options.encode_boolean);
    REQUIRE(options.balanced_select);
    REQUIRE(options.force_ordered);
    REQUIRE(options.force_useful);
    REQUIRE(options.cegis);
    REQUIRE(options.cegis_initial_size == 2);
    REQUIRE(options.cegis_counterexamples == 2);
    REQUIRE(options.no_shuffle);
    REQUIRE(options.quiet);
    REQUIRE(options.profile);
    REQUIRE(options.post_process);
    REQUIRE(options.post_process_beam_width == 3);
    REQUIRE(options.post_process_beam_rounds == 4);
    REQUIRE(options.post_process_beam_candidates == 5);
    REQUIRE(options.post_process_resynthesis_maxnodes == 6);
    REQUIRE(options.post_process_resynthesis_patience == 7);
    REQUIRE(options.generator_timeout == 0.25);
}

TEST_CASE("CLI parses SMT2 export option") {
    const char *argv[] = {"sat_synth_cpp", "--dataset",   "adder",        "--make-smt2",
                          "--make-dimacs", "--make-blif", "--output-blif"};
    CliOptions options = parse_args(static_cast<int>(std::size(argv)), const_cast<char **>(argv));
    REQUIRE(options.dataset_name == "adder");
    REQUIRE(options.make_smt2);
    REQUIRE(options.make_dimacs);
    REQUIRE(options.make_blif);
    REQUIRE(options.output_blif);
}

TEST_CASE("CLI validation rejects invalid combinations and counts") {
    const char *both_inputs[] = {"sat_synth_cpp", "--dataset", "adder", "--config", "x.json"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(both_inputs)), const_cast<char **>(both_inputs)));

    const char *missing_input[] = {"sat_synth_cpp"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(missing_input)), const_cast<char **>(missing_input)));

    const char *zero_batch[] = {"sat_synth_cpp", "--dataset", "adder", "--batch-size", "0"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(zero_batch)), const_cast<char **>(zero_batch)));

    const char *zero_initial[] = {"sat_synth_cpp", "--dataset", "adder", "--cegis-initial-size", "0"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(zero_initial)), const_cast<char **>(zero_initial)));

    const char *zero_counterexamples[] = {"sat_synth_cpp", "--dataset", "adder", "--cegis-counterexamples", "0"};
    REQUIRE_THROWS(
        parse_args(static_cast<int>(std::size(zero_counterexamples)), const_cast<char **>(zero_counterexamples)));

    const char *cegis_smt2[] = {"sat_synth_cpp", "--dataset", "adder", "--cegis", "--make-smt2"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(cegis_smt2)), const_cast<char **>(cegis_smt2)));

    const char *cegis_dimacs[] = {"sat_synth_cpp", "--dataset", "adder", "--cegis", "--make-dimacs"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(cegis_dimacs)), const_cast<char **>(cegis_dimacs)));

    const char *zero_beam_width[] = {"sat_synth_cpp", "--dataset", "adder", "--post-process-beam-width", "0"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(zero_beam_width)), const_cast<char **>(zero_beam_width)));

    const char *negative_beam_rounds[] = {"sat_synth_cpp", "--dataset", "adder", "--post-process-beam-rounds", "-1"};
    REQUIRE_THROWS(
        parse_args(static_cast<int>(std::size(negative_beam_rounds)), const_cast<char **>(negative_beam_rounds)));

    const char *small_resynthesis_window[] = {"sat_synth_cpp", "--dataset", "adder",
                                              "--post-process-resynthesis-maxnodes", "1"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(small_resynthesis_window)),
                              const_cast<char **>(small_resynthesis_window)));

    const char *negative_resynthesis_patience[] = {"sat_synth_cpp", "--dataset", "adder",
                                                   "--post-process-resynthesis-patience", "-1"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(negative_resynthesis_patience)),
                              const_cast<char **>(negative_resynthesis_patience)));

    const char *negative_generator_timeout[] = {"sat_synth_cpp", "--dataset", "adder", "--generator-timeout", "-1"};
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(negative_generator_timeout)),
                              const_cast<char **>(negative_generator_timeout)));
}

TEST_CASE("CLI list-datasets works without config or dataset") {
    const char *argv[] = {"sat_synth_cpp", "--list-datasets"};
    CliOptions options = parse_args(static_cast<int>(std::size(argv)), const_cast<char **>(argv));
    REQUIRE(options.list_datasets);
    REQUIRE(options.config_path.empty());
    REQUIRE(options.dataset_name.empty());
}

TEST_CASE("usage text mentions main input modes") {
    std::ostringstream out;
    print_usage(out);
    REQUIRE(out.str().find("--config") != std::string::npos);
    REQUIRE(out.str().find("--dataset") != std::string::npos);
    REQUIRE(out.str().find("--assume") != std::string::npos);
    REQUIRE(out.str().find("--make-smt2") != std::string::npos);
    REQUIRE(out.str().find("--make-dimacs") != std::string::npos);
    REQUIRE(out.str().find("--make-blif") != std::string::npos);
    REQUIRE(out.str().find("--output-blif") != std::string::npos);
    REQUIRE(out.str().find("--profile") != std::string::npos);
    REQUIRE(out.str().find("--post-process") != std::string::npos);
    REQUIRE(out.str().find("--post-process-resynthesis-maxnodes") != std::string::npos);
    REQUIRE(out.str().find("--generator-timeout") != std::string::npos);
    REQUIRE(out.str().find("--list-datasets") != std::string::npos);
}
