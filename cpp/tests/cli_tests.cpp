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
    REQUIRE_THROWS(parse_args(static_cast<int>(std::size(zero_counterexamples)), const_cast<char **>(zero_counterexamples)));
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
    REQUIRE(out.str().find("--list-datasets") != std::string::npos);
}
