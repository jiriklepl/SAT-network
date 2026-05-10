#include "config.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

TEST_CASE("explicit configs parse and dump don't-care outputs") {
    nlohmann::json raw = {
        {"num_inputs", 1},
        {"num_outputs", 1},
        {"instructions", 0},
        {"examples", nlohmann::json::array({
                         {{"inputs", nlohmann::json::array({false})}, {"outputs", nlohmann::json::array({nullptr})}},
                     })},
    };

    Config cfg = load_explicit_examples(raw);
    REQUIRE(cfg.num_inputs == 1);
    REQUIRE(cfg.num_outputs == 1);
    REQUIRE(cfg.instructions == 0);
    REQUIRE(cfg.examples.size() == 1);
    REQUIRE_FALSE(cfg.examples[0].output(0).has_value());

    nlohmann::json dumped = config_to_json(cfg);
    REQUIRE(dumped["num_inputs"].get<int>() == 1);
    REQUIRE(dumped["examples"][0]["outputs"][0].is_null());
}

TEST_CASE("explicit config parser rejects invalid shapes and values") {
    nlohmann::json valid = {
        {"num_inputs", 1},
        {"num_outputs", 1},
        {"examples", nlohmann::json::array({
                         {{"inputs", nlohmann::json::array({false})}, {"outputs", nlohmann::json::array({true})}},
                     })},
    };

    nlohmann::json bad_dims = valid;
    bad_dims["num_inputs"] = 0;
    REQUIRE_THROWS(load_explicit_examples(bad_dims));

    nlohmann::json bad_input = valid;
    bad_input["examples"][0]["inputs"][0] = 1;
    REQUIRE_THROWS(load_explicit_examples(bad_input));

    nlohmann::json bad_output = valid;
    bad_output["examples"][0]["outputs"][0] = 1;
    REQUIRE_THROWS(load_explicit_examples(bad_output));

    nlohmann::json bad_lengths = valid;
    bad_lengths["examples"][0]["inputs"] = nlohmann::json::array({false, true});
    REQUIRE_THROWS(load_explicit_examples(bad_lengths));

    nlohmann::json empty = valid;
    empty["examples"] = nlohmann::json::array();
    REQUIRE_THROWS(load_explicit_examples(empty));
}

TEST_CASE("typed configs delegate to C++ dataset generation") {
    Config cfg = build_dataset_from_config(nlohmann::json{{"type", "adder"}, {"num_inputs", 2}, {"instructions", 2}});
    REQUIRE(cfg.num_inputs == 2);
    REQUIRE(cfg.num_outputs == 2);
    REQUIRE(cfg.instructions == 2);
    REQUIRE(cfg.examples.size() == 4);
}
