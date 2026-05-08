#pragma once

#include "datasets.hpp"

#include <nlohmann/json.hpp>

#include <string>

nlohmann::json config_to_json(const Config &cfg);
Config load_explicit_examples(const nlohmann::json &cfg);
Config load_config(const std::string &path);
