#include "datasets.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

void require_throws_message(const nlohmann::json &config, const std::string &message) {
    try {
        (void)build_dataset_from_config(config);
    } catch (const std::exception &exc) {
        REQUIRE(std::string(exc.what()) == message);
        return;
    }
    FAIL("expected dataset generation to throw");
}

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

    nlohmann::json sampled_life = default_dataset_config("life");
    sampled_life["life"] = {{"max_examples", 4}, {"seed", 0}};
    require_throws_message(sampled_life, "life.max_examples sampling is not supported by the C++ dataset generator yet");

    nlohmann::json sampled_excitable = default_dataset_config("excitable");
    sampled_excitable["excitable"] = {{"states", 3}, {"max_examples", 4}, {"seed", 0}};
    require_throws_message(sampled_excitable, "excitable.max_examples sampling is not supported by the C++ dataset generator yet");
}

TEST_CASE("specific dataset rows match documented rules") {
    Config adder = build_dataset_from_config(nlohmann::json{{"type", "adder"}, {"num_inputs", 2}});
    REQUIRE_FALSE(adder.examples[0].input(0));
    REQUIRE_FALSE(adder.examples[0].input(1));
    REQUIRE(adder.examples[0].output_count() == 2);
    REQUIRE(adder.examples[0].output(0) == false);
    REQUIRE(adder.examples[0].output(1) == false);
    REQUIRE(adder.examples[3].input(0));
    REQUIRE(adder.examples[3].input(1));
    REQUIRE(adder.examples[3].output(0) == false);
    REQUIRE(adder.examples[3].output(1) == true);

    Config critters = build_dataset_from_config(default_dataset_config("critters"));
    REQUIRE_FALSE(critters.examples.front().input(0));
    REQUIRE_FALSE(critters.examples.front().input(1));
    REQUIRE_FALSE(critters.examples.front().input(2));
    REQUIRE_FALSE(critters.examples.front().input(3));
    REQUIRE(critters.examples.front().output(0) == true);
    REQUIRE(critters.examples.front().output(1) == true);
    REQUIRE(critters.examples.front().output(2) == true);
    REQUIRE(critters.examples.front().output(3) == true);
}

TEST_CASE("compact deterministic parametric datasets are supported") {
    Config excitable = build_dataset_from_config(nlohmann::json{
        {"type", "excitable"},
        {"instructions", 32},
        {"excitable", {{"states", 3}}},
    });
    REQUIRE(excitable.num_inputs == 18);
    REQUIRE(excitable.num_outputs == 2);
    REQUIRE(excitable.examples.size() == 19683);

    Config cyclic = build_dataset_from_config(nlohmann::json{
        {"type", "cyclic"},
        {"instructions", 48},
        {"cyclic", {{"states", 3}}},
    });
    REQUIRE(cyclic.num_inputs == 18);
    REQUIRE(cyclic.num_outputs == 2);
    REQUIRE(cyclic.examples.size() == 19683);

    Config excitable_compressed = build_dataset_from_config(nlohmann::json{
        {"type", "excitable-compressed"},
        {"instructions", 12},
        {"excitable-compressed", {{"states", 3}}},
    });
    REQUIRE(excitable_compressed.num_inputs == 8);
    REQUIRE(excitable_compressed.num_outputs == 2);
    REQUIRE(excitable_compressed.examples.size() == 144);

    Config cyclic_compressed = build_dataset_from_config(nlohmann::json{
        {"type", "cyclic-compressed"},
        {"instructions", 18},
        {"cyclic-compressed", {{"states", 3}}},
    });
    REQUIRE(cyclic_compressed.num_inputs == 8);
    REQUIRE(cyclic_compressed.num_outputs == 2);
    REQUIRE(cyclic_compressed.examples.size() == 144);
}
