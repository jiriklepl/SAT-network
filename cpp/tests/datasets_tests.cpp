#include "datasets.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <exception>
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
    using namespace std::string_literals;

    std::vector<std::string> names = available_dataset_names();
    REQUIRE(std::ranges::find(names, "adder"s) != names.end());
    REQUIRE(std::ranges::find(names, "gol"s) != names.end());
    REQUIRE(std::ranges::find(names, "traffic"s) != names.end());
    REQUIRE(std::ranges::find(names, "life-compressed"s) != names.end());
}

TEST_CASE("built-in datasets generate expected basic shapes") {
    const Config adder = build_dataset_from_config(nlohmann::json{{"type", "adder"}, {"num_inputs", 2}, {"instructions", 2}});
    REQUIRE(adder.num_inputs == 2);
    REQUIRE(adder.num_outputs == 2);
    REQUIRE(adder.instructions == 2);
    REQUIRE(adder.examples.size() == 4);

    const Config traffic = build_dataset_from_config(default_dataset_config("traffic"));
    REQUIRE(traffic.num_inputs == 7);
    REQUIRE(traffic.num_outputs == 2);
    REQUIRE(traffic.examples.size() == 54);

    const Config life_compressed = build_dataset_from_config(default_dataset_config("life-compressed"));
    REQUIRE(life_compressed.num_inputs == 7);
    REQUIRE(life_compressed.num_outputs == 1);
    REQUIRE(life_compressed.examples.size() == 96);
}

TEST_CASE("dataset generator rejects unknown and unsupported sampled configs") {
    REQUIRE_THROWS(default_dataset_config("missing"));
    REQUIRE_THROWS(build_dataset_from_config(nlohmann::json{{"type", "missing"}}));

    nlohmann::json sampled_life = default_dataset_config("life");
    sampled_life["life"] = {{"max_examples", 4}, {"seed", 0}};
    require_throws_message(sampled_life,
                           "life.max_examples sampling is not supported by the C++ dataset generator yet");

    nlohmann::json sampled_excitable = default_dataset_config("excitable");
    sampled_excitable["excitable"] = {{"states", 3}, {"max_examples", 4}, {"seed", 0}};
    require_throws_message(sampled_excitable,
                           "excitable.max_examples sampling is not supported by the C++ dataset generator yet");
}

TEST_CASE("specific dataset rows match documented rules") {
    const Config adder = build_dataset_from_config(nlohmann::json{{"type", "adder"}, {"num_inputs", 2}});
    REQUIRE_FALSE(adder.examples[0].input(0));
    REQUIRE_FALSE(adder.examples[0].input(1));
    REQUIRE(adder.examples[0].output_count() == 2);
    REQUIRE(adder.examples[0].output(0) == false);
    REQUIRE(adder.examples[0].output(1) == false);
    REQUIRE(adder.examples[3].input(0));
    REQUIRE(adder.examples[3].input(1));
    REQUIRE(adder.examples[3].output(0) == false);
    REQUIRE(adder.examples[3].output(1) == true);

    const Config critters = build_dataset_from_config(default_dataset_config("critters"));
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
    const Config excitable = build_dataset_from_config(nlohmann::json{
        {"type", "excitable"},
        {"instructions", 32},
        {"excitable", {{"states", 3}}},
    });
    REQUIRE(excitable.num_inputs == 18);
    REQUIRE(excitable.num_outputs == 2);
    REQUIRE(excitable.examples.size() == 19683);

    const Config cyclic = build_dataset_from_config(nlohmann::json{
        {"type", "cyclic"},
        {"instructions", 48},
        {"cyclic", {{"states", 3}}},
    });
    REQUIRE(cyclic.num_inputs == 18);
    REQUIRE(cyclic.num_outputs == 2);
    REQUIRE(cyclic.examples.size() == 19683);

    const Config excitable_compressed = build_dataset_from_config(nlohmann::json{
        {"type", "excitable-compressed"},
        {"instructions", 12},
        {"excitable-compressed", {{"states", 3}}},
    });
    REQUIRE(excitable_compressed.num_inputs == 8);
    REQUIRE(excitable_compressed.num_outputs == 2);
    REQUIRE(excitable_compressed.examples.size() == 144);

    const Config cyclic_compressed = build_dataset_from_config(nlohmann::json{
        {"type", "cyclic-compressed"},
        {"instructions", 18},
        {"cyclic-compressed", {{"states", 3}}},
    });
    REQUIRE(cyclic_compressed.num_inputs == 8);
    REQUIRE(cyclic_compressed.num_outputs == 2);
    REQUIRE(cyclic_compressed.examples.size() == 144);
}

TEST_CASE("declarative quantified dataset specs and parameter constraints are evaluated") {
    require_throws_message(
        nlohmann::json{
            {"type", "excitable"},
            {"excitable", {{"states", 2}}},
        },
        "excitable.states must be at least 3");
    require_throws_message(
        nlohmann::json{
            {"type", "cyclic"},
            {"cyclic", {{"states", 1}}},
        },
        "cyclic.states must be at least 2");

    const Config inline_quantified = build_dataset_from_config(nlohmann::json{
        {"instructions", 3},
        {"num_inputs", 2},
        {"num_outputs", 1},
        {"forall", nlohmann::json::array({
                       nlohmann::json{{"var", "a"}, {"range", {0, 2}}},
                       nlohmann::json{{"var", "b"}, {"range", {0, 2}}},
                   })},
        {"let_row", {{"xor_val", "a ^ b"}}},
        {"inputs", {"bit(a)", "bit(b)"}},
        {"outputs", {"bit(xor_val)"}},
    });
    REQUIRE(inline_quantified.num_inputs == 2);
    REQUIRE(inline_quantified.num_outputs == 1);
    REQUIRE(inline_quantified.instructions == 3);
    REQUIRE(inline_quantified.examples.size() == 4);
    REQUIRE(inline_quantified.examples[0].output(0) == false);
    REQUIRE(inline_quantified.examples[1].output(0) == true);
    REQUIRE(inline_quantified.examples[2].output(0) == true);
    REQUIRE(inline_quantified.examples[3].output(0) == false);
}

TEST_CASE("expression parser and native evaluator cover all operators and built-in functions") {
    nlohmann::json spec = {
        {"params",
         {
             {"flag", {{"default", true}, {"path", {"custom.flag", "fallback.flag"}}}},
             {"offset", {{"default", 3}, {"path", "custom.offset"}}},
         }},
        {"let",
         {
             {"w", "required_bits(4)"},
             {"derived", "+offset + (flag ? 1 : 0)"},
         }},
        {"instructions", "derived"},
        {"num_inputs", "w + 2"},
        {"num_outputs", 12},
        {"forall",
         nlohmann::json::array({
             nlohmann::json{{"var", "a"}, {"values", {1, 3}}},
             nlohmann::json{{"var", "cells"}, {"tuple", {{"size", 2}, {"range", {0, 2}}}}},
         })},
        {"let_row",
         {
             {"sum_cells", "cells[0] + cells[1]"},
             {"dyn_cell", "cells[a == 3 ? 1 : 0]"},
             {"cnt_ones", "count(cells, 1)"},
             {"cnt_tail", "count(cells, 1, 1)"},
             {"pop", "popcount(a)"},
         }},
        {"where", "!(a == 1 && cells[0] == 1 && cells[1] == 1)"},
        {"inputs", {"bits(a, w)", "state_bits(cells, 1)"}},
        {"outputs",
         {
             "bit(a == 3 && sum_cells >= 1)",
             "bit(a == 1 || sum_cells == 0)",
             "bit((a + 5) / 2 == 3 && (a + 5) % 2 == 0)",
             "bit((a << 1) >> 1 == a)",
             "bit(((a & 1) ^ (a | 2)) != 0)",
             "bit((-a < 0) && (~0 < 0))",
             "bit(dyn_cell <= 1 && dyn_cell >= 0)",
             "bit(cnt_ones > cnt_tail)",
             "bit(pop == (a == 3 ? 2 : 1))",
             "a == 1 ? null : bit(true)",
             "bit(false)",
             "sum_cells == 1",
         }},
    };

    const Config cfg = build_dataset_from_config(spec);
    REQUIRE(cfg.instructions == 4);
    REQUIRE(cfg.num_inputs == 4);
    REQUIRE(cfg.num_outputs == 12);
    // Total domain is 2 * 2 * 2 = 8, minus 1 filtered by `where` = 7
    REQUIRE(cfg.examples.size() == 7);
    // First row: a=1, cells=(0, 0) -> a==1 so output 9 is null (don't care), output 10 is false
    REQUIRE_FALSE(cfg.examples.front().output(9).has_value());
    REQUIRE(cfg.examples.front().output(10) == false);
}

TEST_CASE("expression parser and dataset compiler reject invalid expressions and configs") {
    // Unknown dataset or path traversal
    REQUIRE_THROWS(build_dataset_from_config(nlohmann::json{{"type", "../adder"}}));
    REQUIRE_THROWS(build_dataset_from_config(nlohmann::json{{"type", "a/b"}}));
    REQUIRE_THROWS(build_dataset_from_config(nlohmann::json::object()));

    auto base_spec = []() -> nlohmann::json {
        return {
            {"instructions", 1},
            {"num_inputs", 1},
            {"num_outputs", 1},
            {"forall", nlohmann::json::array({nlohmann::json{{"var", "a"}, {"range", {0, 2}}}})},
            {"inputs", {"bit(a)"}},
            {"outputs", {"bit(a)"}},
        };
    };

    // Cyclic static let
    {
        nlohmann::json s = base_spec();
        s["let"] = {{"x", "y + 1"}, {"y", "x + 1"}};
        REQUIRE_THROWS(build_dataset_from_config(s));
    }

    // Cyclic let_row
    {
        nlohmann::json s = base_spec();
        s["let_row"] = {{"x", "y + 1"}, {"y", "x + 1"}};
        s["outputs"] = {"bit(x)"};
        REQUIRE_THROWS(build_dataset_from_config(s));
    }

    // Division by zero and modulo by zero
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(1 / 0)"};
        require_throws_message(s, "Division by zero in dataset expression");
    }
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(1 % 0)"};
        require_throws_message(s, "Modulo by zero in dataset expression");
    }

    // Unknown variable, unknown tuple, unknown function, trailing characters
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(unknown_var)"};
        REQUIRE_THROWS(build_dataset_from_config(s));
    }
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(unknown_tuple[0])"};
        REQUIRE_THROWS(build_dataset_from_config(s));
    }
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(unknown_fn(a))"};
        REQUIRE_THROWS(build_dataset_from_config(s));
    }
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(a) trailing"};
        REQUIRE_THROWS(build_dataset_from_config(s));
    }

    // Tuple index out of bounds
    {
        nlohmann::json s = base_spec();
        s["forall"] = nlohmann::json::array({nlohmann::json{{"var", "cells"}, {"tuple", {{"size", 2}, {"range", {0, 2}}}}}});
        s["inputs"] = {"state_bits(cells, 1)"};
        s["num_inputs"] = 2;
        s["outputs"] = {"bit(cells[5])"};
        require_throws_message(s, "Tuple index out of bounds");
    }

    // Null in inputs or scalar context, bit-list in scalar context, negative bits width
    {
        nlohmann::json s = base_spec();
        s["inputs"] = {"null"};
        require_throws_message(s, "input values cannot be null");
    }
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(1 + null)"};
        require_throws_message(s, "null cannot be evaluated as a scalar");
    }
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bit(1 + bits(a, 1))"};
        require_throws_message(s, "Bit-list expression cannot be evaluated as a scalar");
    }
    {
        nlohmann::json s = base_spec();
        s["outputs"] = {"bits(a, -1)"};
        require_throws_message(s, "bits() width must be non-negative");
    }

    // Empty dataset after where filtering
    {
        nlohmann::json s = base_spec();
        s["where"] = "0";
        require_throws_message(s, "Dataset contains no examples");
    }

    // Zero max_examples in sample_section
    {
        nlohmann::json sampled_life = default_dataset_config("life");
        sampled_life["life"] = {{"max_examples", 0}};
        require_throws_message(sampled_life, "life.max_examples must be positive");
    }
}
