#include "datasets.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <vector>

TEST_CASE("available datasets include the expected built-ins") {
    std::vector<std::string> names = available_dataset_names();
    REQUIRE(std::find(names.begin(), names.end(), "adder") != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "gol") != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "traffic") != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "life-compressed") != names.end());
}

TEST_CASE("built-in datasets generate expected basic shapes") {
    Config adder = build_dataset_from_config(nlohmann::json{{"type", "adder"}, {"num_inputs", 2}, {"instructions", 2}});
    REQUIRE(adder.num_inputs == 2);
    REQUIRE(adder.num_outputs == 2);
    REQUIRE(adder.instructions == 2);
    REQUIRE(adder.examples.size() == 4);

    Config traffic = build_dataset_from_config(default_dataset_config("traffic"));
    REQUIRE(traffic.num_inputs == 7);
    REQUIRE(traffic.num_outputs == 2);
    REQUIRE(traffic.examples.size() == 54);

    Config life_compressed = build_dataset_from_config(default_dataset_config("life-compressed"));
    REQUIRE(life_compressed.num_inputs == 7);
    REQUIRE(life_compressed.num_outputs == 1);
    REQUIRE(life_compressed.examples.size() == 96);
}

TEST_CASE("dataset generator rejects unknown and unsupported sampled configs") {
    REQUIRE_THROWS(default_dataset_config("missing"));
    REQUIRE_THROWS(build_dataset_from_config(nlohmann::json{{"type", "missing"}}));

    nlohmann::json sampled = default_dataset_config("life");
    sampled["life"] = {{"max_examples", 4}, {"seed", 0}};
    REQUIRE_THROWS(build_dataset_from_config(sampled));
}

TEST_CASE("specific dataset rows match documented rules") {
    Config adder = build_dataset_from_config(nlohmann::json{{"type", "adder"}, {"num_inputs", 2}});
    REQUIRE(adder.examples[0].inputs == std::vector<bool>{false, false});
    REQUIRE(adder.examples[0].outputs.size() == 2);
    REQUIRE(adder.examples[0].outputs[0] == false);
    REQUIRE(adder.examples[0].outputs[1] == false);
    REQUIRE(adder.examples[3].inputs == std::vector<bool>{true, true});
    REQUIRE(adder.examples[3].outputs[0] == false);
    REQUIRE(adder.examples[3].outputs[1] == true);

    Config critters = build_dataset_from_config(default_dataset_config("critters"));
    REQUIRE(critters.examples.front().inputs == std::vector<bool>{false, false, false, false});
    REQUIRE(critters.examples.front().outputs == std::vector<std::optional<bool>>{true, true, true, true});
}
