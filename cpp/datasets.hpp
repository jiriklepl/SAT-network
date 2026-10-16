#pragma once

#include <boost/dynamic_bitset.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>

#include <initializer_list>
#include <memory>
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

struct QuantifiedSpec;

constexpr std::size_t kDefaultMaxMaterializedExamples = 65536;
constexpr std::size_t kExportMaxMaterializedExamples = std::size_t{1} << 22;
constexpr std::size_t kDefaultBasisSize = 64;

struct Config {
    // Complete truth table when `fully_materialized`; otherwise a seed basis of valid examples.
    std::vector<Example> examples;
    int num_inputs = 0;
    int num_outputs = 0;
    int instructions = kDefaultProgramLength;
    // Present for datasets generated from a declarative `forall` spec.
    std::shared_ptr<const QuantifiedSpec> quantified;
    bool fully_materialized = true;
    std::size_t domain_size = 0;
};

struct DatasetLoadOptions {
    // Quantified domains larger than this are not unrolled; they are solved symbolically.
    std::size_t max_materialized_examples = kDefaultMaxMaterializedExamples;
    std::size_t basis_size = kDefaultBasisSize;
    unsigned seed = 0;
};

std::vector<std::string> available_dataset_names();
nlohmann::json default_dataset_config(const std::string &name);
Config build_dataset_from_config(const nlohmann::json &cfg, const DatasetLoadOptions &options = {});

