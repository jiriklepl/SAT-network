#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

constexpr int kDefaultProgramLength = 16;

struct Example {
    std::vector<bool> inputs;
    std::vector<std::optional<bool>> outputs;
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
