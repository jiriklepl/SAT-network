#pragma once

#include <boost/dynamic_bitset.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>

#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

constexpr int kDefaultProgramLength = 16;

struct Example {
    boost::dynamic_bitset<> inputs;
    boost::dynamic_bitset<> output_values;
    boost::dynamic_bitset<> output_dont_care;

    Example() = default;
    Example(std::initializer_list<bool> input_values, std::initializer_list<std::optional<bool>> output_values);

    std::size_t input_count() const;
    std::size_t output_count() const;
    bool input(std::size_t idx) const;
    std::optional<bool> output(std::size_t idx) const;
    void push_input(bool value);
    void push_output(std::optional<bool> value);
};

struct Config {
    std::vector<Example> examples;
    int num_inputs = 0;
    int num_outputs = 0;
    int instructions = kDefaultProgramLength;
};

std::vector<std::string> available_dataset_names();
nlohmann::json default_dataset_config(const std::string &name);
Config build_dataset_from_config(const nlohmann::json &cfg);
