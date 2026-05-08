#include "config.hpp"

#include <fstream>
#include <stdexcept>

nlohmann::json config_to_json(const Config &cfg) {
    nlohmann::json result;
    result["num_inputs"] = cfg.num_inputs;
    result["num_outputs"] = cfg.num_outputs;
    result["instructions"] = cfg.instructions;
    result["examples"] = nlohmann::json::array();
    for (const auto &example : cfg.examples) {
        nlohmann::json raw_example;
        raw_example["inputs"] = nlohmann::json::array();
        raw_example["outputs"] = nlohmann::json::array();
        for (bool input : example.inputs) {
            raw_example["inputs"].push_back(input);
        }
        for (const auto &output : example.outputs) {
            if (output.has_value()) {
                raw_example["outputs"].push_back(*output);
            } else {
                raw_example["outputs"].push_back(nullptr);
            }
        }
        result["examples"].push_back(std::move(raw_example));
    }
    return result;
}

Config load_explicit_examples(const nlohmann::json &cfg) {
    Config result;
    result.num_inputs = cfg.at("num_inputs").get<int>();
    result.num_outputs = cfg.at("num_outputs").get<int>();
    result.instructions = cfg.value("instructions", kDefaultProgramLength);
    if (result.num_inputs <= 0 || result.num_outputs <= 0 || result.instructions < 0) {
        throw std::runtime_error("Invalid num_inputs, num_outputs, or instructions");
    }

    for (const auto &raw_ex : cfg.at("examples")) {
        Example ex;
        for (const auto &raw_input : raw_ex.at("inputs")) {
            if (!raw_input.is_boolean()) {
                throw std::runtime_error("Input values must be booleans");
            }
            ex.inputs.push_back(raw_input.get<bool>());
        }
        for (const auto &raw_output : raw_ex.at("outputs")) {
            if (raw_output.is_null()) {
                ex.outputs.push_back(std::nullopt);
            } else if (raw_output.is_boolean()) {
                ex.outputs.push_back(raw_output.get<bool>());
            } else {
                throw std::runtime_error("Output values must be booleans or null");
            }
        }
        if (static_cast<int>(ex.inputs.size()) != result.num_inputs ||
            static_cast<int>(ex.outputs.size()) != result.num_outputs) {
            throw std::runtime_error("Example length does not match declared input/output sizes");
        }
        result.examples.push_back(std::move(ex));
    }
    if (result.examples.empty()) {
        throw std::runtime_error("Dataset contains no examples");
    }
    return result;
}

Config load_config(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Config file not found: " + path);
    }
    nlohmann::json cfg;
    in >> cfg;

    if (cfg.contains("examples")) return load_explicit_examples(cfg);
    return build_dataset_from_config(cfg);
}
