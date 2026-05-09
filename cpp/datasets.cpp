#include "datasets.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <stdexcept>
#include <tuple>

namespace {

using IOList = std::vector<std::optional<bool>>;
using StateRule = std::function<std::vector<int>(const std::vector<int> &)>;
using SummaryRule = std::function<int(int, int)>;

constexpr int kMooreCells = 9;
constexpr int kVonNeumannCells = 5;

IOList bits(int value, int width) {
    IOList result;
    result.reserve(width);
    for (int bit = 0; bit < width; ++bit) {
        result.push_back((value & (1 << bit)) != 0);
    }
    return result;
}

IOList state_bits(const std::vector<int> &states, int bits_per_state) {
    IOList result;
    result.reserve(states.size() * static_cast<std::size_t>(bits_per_state));
    for (int state : states) {
        IOList state_value = bits(state, bits_per_state);
        result.insert(result.end(), state_value.begin(), state_value.end());
    }
    return result;
}

int required_bits(int state_count) {
    int max_value = std::max(0, state_count - 1);
    int width = 0;
    do {
        ++width;
        max_value >>= 1;
    } while (max_value != 0);
    return width;
}

int json_int(const nlohmann::json &cfg, const std::string &key, int default_value) {
    return cfg.contains(key) ? cfg.at(key).get<int>() : default_value;
}

bool json_bool(const nlohmann::json &cfg, const std::string &key, bool default_value) {
    return cfg.contains(key) ? cfg.at(key).get<bool>() : default_value;
}

const nlohmann::json &section_or_empty(const nlohmann::json &cfg, const std::string &section) {
    static const nlohmann::json empty = nlohmann::json::object();
    if (!cfg.contains(section) || cfg.at(section).is_null()) {
        return empty;
    }
    return cfg.at(section);
}

void add_example(Config &dataset, IOList inputs, IOList outputs) {
    Example example;
    example.inputs.reserve(inputs.size());
    for (const auto &input : inputs) {
        if (!input.has_value()) {
            throw std::runtime_error("input values cannot be null");
        }
        example.inputs.push_back(*input);
    }
    example.outputs = std::move(outputs);
    dataset.examples.push_back(std::move(example));
}

std::vector<std::vector<int>> state_vectors(int state_count, int cell_count, const nlohmann::json &cfg, const std::string &section) {
    const nlohmann::json &section_cfg = section_or_empty(cfg, section);
    std::size_t total = 1;
    for (int i = 0; i < cell_count; ++i) {
        total *= static_cast<std::size_t>(state_count);
    }

    if (!section_cfg.contains("max_examples") || section_cfg.at("max_examples").get<std::size_t>() >= total) {
        std::vector<std::vector<int>> rows;
        rows.reserve(total);
        for (std::size_t encoded = 0; encoded < total; ++encoded) {
            std::size_t value = encoded;
            std::vector<int> states(cell_count);
            for (int cell = cell_count - 1; cell >= 0; --cell) {
                states[cell] = static_cast<int>(value % static_cast<std::size_t>(state_count));
                value /= static_cast<std::size_t>(state_count);
            }
            rows.push_back(std::move(states));
        }
        return rows;
    }

    if (section_cfg.at("max_examples").get<std::size_t>() == 0) {
        throw std::runtime_error(section + ".max_examples must be positive");
    }
    throw std::runtime_error(section + ".max_examples sampling is not supported by the C++ dataset generator yet");
}

Config build_state_dataset(
    const nlohmann::json &cfg,
    const std::string &section,
    int state_count,
    int bits_per_state,
    int cell_count,
    const StateRule &rule
) {
    Config dataset;
    dataset.num_inputs = cell_count * bits_per_state;
    dataset.num_outputs = static_cast<int>(rule(std::vector<int>(cell_count, 0)).size()) * bits_per_state;
    for (const auto &states : state_vectors(state_count, cell_count, cfg, section)) {
        IOList outputs;
        for (int output_state : rule(states)) {
            IOList output_bits = bits(output_state, bits_per_state);
            outputs.insert(outputs.end(), output_bits.begin(), output_bits.end());
        }
        add_example(dataset, state_bits(states, bits_per_state), std::move(outputs));
    }
    return dataset;
}

Config build_summary_dataset(const std::vector<std::pair<IOList, int>> &rows, int num_inputs, int bits_per_output) {
    Config dataset;
    dataset.num_inputs = num_inputs;
    dataset.num_outputs = bits_per_output;
    for (const auto &[inputs, output_state] : rows) {
        if (static_cast<int>(inputs.size()) != num_inputs) {
            throw std::runtime_error("compressed input length does not match declared input count");
        }
        add_example(dataset, inputs, bits(output_state, bits_per_output));
    }
    return dataset;
}

IOList moore_column_count_inputs(int center, int left_count, int center_count, int right_count, int center_bits) {
    IOList result = bits(left_count, 2);
    IOList center_count_bits = bits(center_count, 2);
    IOList right_count_bits = bits(right_count, 2);
    IOList center_value = bits(center, center_bits);
    result.insert(result.end(), center_count_bits.begin(), center_count_bits.end());
    result.insert(result.end(), right_count_bits.begin(), right_count_bits.end());
    result.insert(result.end(), center_value.begin(), center_value.end());
    return result;
}

Config build_moore_column_count_dataset(int state_count, int center_bits, int output_bits, const SummaryRule &rule) {
    std::vector<std::pair<IOList, int>> rows;
    for (int center = 0; center < state_count; ++center) {
        for (int left_count = 0; left_count < 4; ++left_count) {
            for (int center_count = 0; center_count < 3; ++center_count) {
                for (int right_count = 0; right_count < 4; ++right_count) {
                    int relevant_neighbors = left_count + center_count + right_count;
                    rows.emplace_back(
                        moore_column_count_inputs(center, left_count, center_count, right_count, center_bits),
                        rule(center, relevant_neighbors));
                }
            }
        }
    }
    return build_summary_dataset(rows, center_bits + 6, output_bits);
}

int moore_alive_count(const std::vector<int> &states, int alive_state = 1) {
    return static_cast<int>(std::count(states.begin() + 1, states.end(), alive_state));
}

Config with_instructions(Config dataset, int instructions) {
    dataset.instructions = instructions;
    return dataset;
}

Config build_adder(const nlohmann::json &cfg) {
    int num_inputs = json_int(cfg, "num_inputs", 3);
    int num_outputs = 0;
    while ((1 << num_outputs) < num_inputs + 1) {
        ++num_outputs;
    }

    Config dataset;
    dataset.num_inputs = num_inputs;
    dataset.num_outputs = num_outputs;
    for (int value = 0; value < (1 << num_inputs); ++value) {
        int total = 0;
        IOList inputs;
        for (int bit = 0; bit < num_inputs; ++bit) {
            bool input = (value & (1 << bit)) != 0;
            inputs.push_back(input);
            total += input ? 1 : 0;
        }
        add_example(dataset, inputs, bits(total, num_outputs));
    }
    return with_instructions(std::move(dataset), 5);
}

Config build_gol(const nlohmann::json &cfg) {
    const nlohmann::json &gol = section_or_empty(cfg, "gol");
    int left_range = json_int(gol, "left_range", 4);
    int center_range = json_int(gol, "center_range", 3);
    int right_range = json_int(gol, "right_range", 4);
    bool include_alive = json_bool(gol, "include_alive", true);

    Config dataset;
    dataset.num_inputs = 7;
    dataset.num_outputs = 1;
    for (int left = 0; left < left_range; ++left) {
        for (int center = 0; center < center_range; ++center) {
            for (int right = 0; right < right_range; ++right) {
                for (bool alive : include_alive ? std::vector<bool>{true, false} : std::vector<bool>{false}) {
                    IOList inputs = {
                        (left & 1) != 0,
                        ((left >> 1) & 1) != 0,
                        (center & 1) != 0,
                        ((center >> 1) & 1) != 0,
                        (right & 1) != 0,
                        ((right >> 1) & 1) != 0,
                        alive,
                    };
                    int sum = left + center + right;
                    add_example(dataset, inputs, IOList{sum == 3 || (sum == 2 && alive)});
                }
            }
        }
    }
    return with_instructions(std::move(dataset), 14);
}

Config build_gol1(const nlohmann::json &) {
    Config dataset;
    dataset.num_inputs = 7;
    dataset.num_outputs = 2;
    for (int left = 0; left < 4; ++left) {
        for (int center = 0; center < 3; ++center) {
            for (int right = 0; right < 4; ++right) {
                for (bool alive : {true, false}) {
                    int sum = left + center + right;
                    int carry = ((left & 1) + (center & 1) + (right & 1)) / 2;
                    IOList inputs = {
                        (left & 1) != 0,
                        ((left >> 1) & 1) != 0,
                        (center & 1) != 0,
                        ((center >> 1) & 1) != 0,
                        (right & 1) != 0,
                        ((right >> 1) & 1) != 0,
                        alive,
                    };
                    add_example(dataset, inputs, IOList{((sum | static_cast<int>(alive)) % 2) == 1, carry == 1});
                }
            }
        }
    }
    return with_instructions(std::move(dataset), 6);
}

Config build_gol2(const nlohmann::json &) {
    Config dataset;
    dataset.num_inputs = 5;
    dataset.num_outputs = 1;
    for (int left = 0; left < 4; ++left) {
        for (int center = 0; center < 3; ++center) {
            for (int right = 0; right < 4; ++right) {
                for (bool alive : {true, false}) {
                    int sum = left + center + right;
                    int carry = ((left & 1) + (center & 1) + (right & 1)) / 2;
                    IOList inputs = {
                        ((left >> 1) & 1) != 0,
                        ((center >> 1) & 1) != 0,
                        ((right >> 1) & 1) != 0,
                        carry == 1,
                        ((sum | static_cast<int>(alive)) % 2) == 1,
                    };
                    add_example(dataset, inputs, IOList{(sum | static_cast<int>(alive)) == 3});
                }
            }
        }
    }
    return with_instructions(std::move(dataset), 8);
}

Config build_sloppy_adder(const nlohmann::json &cfg) {
    const nlohmann::json &section = section_or_empty(cfg, "sloppy_adder");
    int left_range = json_int(section, "left_range", 4);
    int right_range = json_int(section, "right_range", 4);
    Config dataset;
    dataset.num_inputs = json_int(cfg, "num_inputs", 4);
    dataset.num_outputs = json_int(cfg, "num_outputs", 3);
    for (int left = 0; left < left_range; ++left) {
        for (int right = 0; right < right_range; ++right) {
            int result = left + right;
            bool carry = result >= 4;
            IOList inputs = {((left >> 1) & 1) != 0, (left & 1) != 0, ((right >> 1) & 1) != 0, (right & 1) != 0};
            IOList outputs = {carry, carry ? std::nullopt : std::optional<bool>((result & 2) != 0), carry ? std::nullopt : std::optional<bool>((result & 1) != 0)};
            add_example(dataset, inputs, outputs);
        }
    }
    return with_instructions(std::move(dataset), 7);
}

Config build_sloppy_adder3(const nlohmann::json &cfg) {
    const nlohmann::json &section = section_or_empty(cfg, "sloppy_adder");
    int left_range = json_int(section, "left_range", 4);
    int center_range = json_int(section, "center_range", 3);
    int right_range = json_int(section, "right_range", 4);
    Config dataset;
    dataset.num_inputs = json_int(cfg, "num_inputs", 6);
    dataset.num_outputs = json_int(cfg, "num_outputs", 3);
    for (int left = 0; left < left_range; ++left) {
        for (int center = 0; center < center_range; ++center) {
            for (int right = 0; right < right_range; ++right) {
                int result = left + center + right;
                bool carry = result >= 4;
                IOList inputs = {
                    ((left >> 1) & 1) != 0, (left & 1) != 0,
                    ((center >> 1) & 1) != 0, (center & 1) != 0,
                    ((right >> 1) & 1) != 0, (right & 1) != 0,
                };
                IOList outputs = {carry, carry ? std::nullopt : std::optional<bool>((result & 2) != 0), carry ? std::nullopt : std::optional<bool>((result & 1) != 0)};
                add_example(dataset, inputs, outputs);
            }
        }
    }
    return with_instructions(std::move(dataset), 12);
}

Config build_life(const nlohmann::json &cfg) {
    return with_instructions(build_state_dataset(cfg, "life", 2, 1, kMooreCells, [](const std::vector<int> &states) {
        int center = states[0];
        int alive_neighbors = moore_alive_count(states);
        return std::vector<int>{(alive_neighbors == 3 || (center == 1 && alive_neighbors == 2)) ? 1 : 0};
    }), 14);
}

Config build_life_compressed(const nlohmann::json &) {
    return with_instructions(build_moore_column_count_dataset(2, 1, 1, [](int center, int n) {
        return (n == 3 || (center == 1 && n == 2)) ? 1 : 0;
    }), 14);
}

Config build_maze(const nlohmann::json &cfg) {
    return with_instructions(build_state_dataset(cfg, "maze", 2, 1, kMooreCells, [](const std::vector<int> &states) {
        int center = states[0];
        int wall_neighbors = moore_alive_count(states);
        return std::vector<int>{(wall_neighbors == 3 || (center == 1 && wall_neighbors < 6)) ? 1 : 0};
    }), 10);
}

Config build_maze_compressed(const nlohmann::json &) {
    return with_instructions(build_moore_column_count_dataset(2, 1, 1, [](int center, int n) {
        return (n == 3 || (center == 1 && n < 6)) ? 1 : 0;
    }), 6);
}

Config build_brian(const nlohmann::json &cfg) {
    return with_instructions(build_state_dataset(cfg, "brian", 3, 2, kMooreCells, [](const std::vector<int> &states) {
        int center = states[0];
        if (center == 1) return std::vector<int>{2};
        if (center == 2) return std::vector<int>{0};
        return std::vector<int>{moore_alive_count(states, 1) == 2 ? 1 : 0};
    }), 24);
}

Config build_brian_compressed(const nlohmann::json &) {
    return with_instructions(build_moore_column_count_dataset(3, 2, 2, [](int center, int n) {
        if (center == 1) return 2;
        if (center == 2) return 0;
        return n == 2 ? 1 : 0;
    }), 10);
}

Config build_fire(const nlohmann::json &cfg) {
    return with_instructions(build_state_dataset(cfg, "fire", 4, 2, kVonNeumannCells, [](const std::vector<int> &states) {
        int center = states[0];
        bool has_fire_neighbor = std::find(states.begin() + 1, states.end(), 2) != states.end();
        if (center == 1) return std::vector<int>{has_fire_neighbor ? 2 : 1};
        if (center == 2) return std::vector<int>{3};
        if (center == 3) return std::vector<int>{has_fire_neighbor ? 3 : 0};
        return std::vector<int>{0};
    }), 18);
}

Config build_fire_compressed(const nlohmann::json &) {
    std::vector<std::pair<IOList, int>> rows;
    for (int center = 0; center < 4; ++center) {
        for (int left_count = 0; left_count < 2; ++left_count) {
            for (int vertical_count = 0; vertical_count < 3; ++vertical_count) {
                for (int right_count = 0; right_count < 2; ++right_count) {
                    bool has_fire_neighbor = left_count + vertical_count + right_count > 0;
                    int output = 0;
                    if (center == 1) output = has_fire_neighbor ? 2 : 1;
                    else if (center == 2) output = 3;
                    else if (center == 3) output = has_fire_neighbor ? 3 : 0;
                    IOList input = bits(left_count, 1);
                    IOList vertical = bits(vertical_count, 2);
                    IOList right = bits(right_count, 1);
                    IOList center_bits = bits(center, 2);
                    input.insert(input.end(), vertical.begin(), vertical.end());
                    input.insert(input.end(), right.begin(), right.end());
                    input.insert(input.end(), center_bits.begin(), center_bits.end());
                    rows.emplace_back(std::move(input), output);
                }
            }
        }
    }
    return with_instructions(build_summary_dataset(rows, 6, 2), 8);
}

Config build_wire(const nlohmann::json &cfg) {
    return with_instructions(build_state_dataset(cfg, "wire", 4, 2, kMooreCells, [](const std::vector<int> &states) {
        int center = states[0];
        if (center == 0) return std::vector<int>{0};
        if (center == 2) return std::vector<int>{3};
        if (center == 3) return std::vector<int>{1};
        int heads = static_cast<int>(std::count(states.begin() + 1, states.end(), 2));
        return std::vector<int>{(heads == 1 || heads == 2) ? 2 : 1};
    }), 24);
}

Config build_wire_compressed(const nlohmann::json &) {
    return with_instructions(build_moore_column_count_dataset(4, 2, 2, [](int center, int n) {
        if (center == 0) return 0;
        if (center == 2) return 3;
        if (center == 3) return 1;
        return (n == 1 || n == 2) ? 2 : 1;
    }), 10);
}

Config build_excitable(const nlohmann::json &cfg) {
    const nlohmann::json &section = section_or_empty(cfg, "excitable");
    int state_count = json_int(section, "states", 8);
    if (state_count < 3) throw std::runtime_error("excitable.states must be at least 3");
    int bits_per_state = required_bits(state_count);
    return with_instructions(build_state_dataset(cfg, "excitable", state_count, bits_per_state, kMooreCells, [state_count](const std::vector<int> &states) {
        int center = states[0];
        if (center == 0) return std::vector<int>{std::find(states.begin() + 1, states.end(), 1) != states.end() ? 1 : 0};
        if (center == state_count - 1) return std::vector<int>{0};
        return std::vector<int>{center + 1};
    }), 32);
}

Config build_excitable_compressed(const nlohmann::json &cfg) {
    const nlohmann::json &section = cfg.contains("excitable-compressed") ? cfg.at("excitable-compressed") : section_or_empty(cfg, "excitable");
    int state_count = json_int(section, "states", 8);
    if (state_count < 3) throw std::runtime_error("excitable-compressed.states must be at least 3");
    int bits_per_state = required_bits(state_count);
    return with_instructions(build_moore_column_count_dataset(state_count, bits_per_state, bits_per_state, [state_count](int center, int n) {
        if (center == 0) return n > 0 ? 1 : 0;
        if (center == state_count - 1) return 0;
        return center + 1;
    }), 12);
}

Config build_cyclic(const nlohmann::json &cfg) {
    const nlohmann::json &section = section_or_empty(cfg, "cyclic");
    int state_count = json_int(section, "states", 32);
    if (state_count < 2) throw std::runtime_error("cyclic.states must be at least 2");
    int bits_per_state = required_bits(state_count);
    return with_instructions(build_state_dataset(cfg, "cyclic", state_count, bits_per_state, kMooreCells, [state_count](const std::vector<int> &states) {
        int center = states[0];
        int successor = (center + 1) % state_count;
        return std::vector<int>{std::find(states.begin() + 1, states.end(), successor) != states.end() ? successor : center};
    }), 48);
}

Config build_cyclic_compressed(const nlohmann::json &cfg) {
    const nlohmann::json &section = cfg.contains("cyclic-compressed") ? cfg.at("cyclic-compressed") : section_or_empty(cfg, "cyclic");
    int state_count = json_int(section, "states", 32);
    if (state_count < 2) throw std::runtime_error("cyclic-compressed.states must be at least 2");
    int bits_per_state = required_bits(state_count);
    return with_instructions(build_moore_column_count_dataset(state_count, bits_per_state, bits_per_state, [state_count](int center, int n) {
        int successor = (center + 1) % state_count;
        return n > 0 ? successor : center;
    }), 18);
}

Config build_fluid(const nlohmann::json &cfg) {
    constexpr int up = 0;
    constexpr int down = 1;
    constexpr int left = 2;
    constexpr int right = 3;
    constexpr int north = 0;
    constexpr int south = 1;
    constexpr int west = 2;
    constexpr int east = 3;

    Config dataset;
    dataset.num_inputs = 16;
    dataset.num_outputs = 4;
    for (const auto &states : state_vectors(16, 4, cfg, "fluid")) {
        bool incoming_up = (states[south] & (1 << up)) != 0;
        bool incoming_down = (states[north] & (1 << down)) != 0;
        bool incoming_left = (states[east] & (1 << left)) != 0;
        bool incoming_right = (states[west] & (1 << right)) != 0;
        IOList output = {incoming_up, incoming_down, incoming_left, incoming_right};
        if (output == IOList{true, true, false, false}) {
            output = {false, false, true, true};
        } else if (output == IOList{false, false, true, true}) {
            output = {true, true, false, false};
        }
        add_example(dataset, state_bits(states, 4), output);
    }
    return with_instructions(std::move(dataset), 18);
}

Config build_critters(const nlohmann::json &) {
    Config dataset;
    dataset.num_inputs = 4;
    dataset.num_outputs = 4;
    for (bool b0 : {false, true}) {
        for (bool b1 : {false, true}) {
            for (bool b2 : {false, true}) {
                for (bool b3 : {false, true}) {
                    IOList block = {b0, b1, b2, b3};
                    int alive_count = 0;
                    for (const auto &bit : block) alive_count += *bit ? 1 : 0;
                    IOList output;
                    if (alive_count == 2) {
                        output = block;
                    } else if (alive_count == 3) {
                        output = {!*block[3], !*block[2], !*block[1], !*block[0]};
                    } else {
                        for (const auto &bit : block) output.push_back(!*bit);
                    }
                    add_example(dataset, block, output);
                }
            }
        }
    }
    return with_instructions(std::move(dataset), 10);
}

Config build_traffic(const nlohmann::json &) {
    Config dataset;
    dataset.num_inputs = 7;
    dataset.num_outputs = 2;
    for (bool phase : {false, true}) {
        int moving_state = phase ? 2 : 1;
        for (int prev_state = 0; prev_state < 3; ++prev_state) {
            for (int center_state = 0; center_state < 3; ++center_state) {
                for (int next_state = 0; next_state < 3; ++next_state) {
                    int output_state = center_state;
                    if (center_state == moving_state && next_state == 0) output_state = 0;
                    else if (center_state == 0 && prev_state == moving_state) output_state = moving_state;
                    IOList inputs = {phase};
                    IOList states = state_bits({prev_state, center_state, next_state}, 2);
                    inputs.insert(inputs.end(), states.begin(), states.end());
                    add_example(dataset, inputs, bits(output_state, 2));
                }
            }
        }
    }
    return with_instructions(std::move(dataset), 12);
}

using Builder = std::function<Config(const nlohmann::json &)>;

const std::map<std::string, std::pair<int, Builder>> &registry() {
    static const std::map<std::string, std::pair<int, Builder>> builders = {
        {"adder", {5, build_adder}},
        {"gol", {14, build_gol}},
        {"gol1", {6, build_gol1}},
        {"gol2", {8, build_gol2}},
        {"sloppy-adder", {7, build_sloppy_adder}},
        {"sloppy-adder3", {12, build_sloppy_adder3}},
        {"life", {14, build_life}},
        {"life-compressed", {14, build_life_compressed}},
        {"maze", {10, build_maze}},
        {"maze-compressed", {6, build_maze_compressed}},
        {"brian", {24, build_brian}},
        {"brian-compressed", {10, build_brian_compressed}},
        {"fire", {18, build_fire}},
        {"fire-compressed", {8, build_fire_compressed}},
        {"wire", {24, build_wire}},
        {"wire-compressed", {10, build_wire_compressed}},
        {"excitable", {32, build_excitable}},
        {"excitable-compressed", {12, build_excitable_compressed}},
        {"cyclic", {48, build_cyclic}},
        {"cyclic-compressed", {18, build_cyclic_compressed}},
        {"fluid", {18, build_fluid}},
        {"critters", {10, build_critters}},
        {"traffic", {12, build_traffic}},
    };
    return builders;
}

}  // namespace

std::vector<std::string> available_dataset_names() {
    std::vector<std::string> names;
    for (const auto &[name, _] : registry()) {
        names.push_back(name);
    }
    return names;
}

nlohmann::json default_dataset_config(const std::string &name) {
    const auto it = registry().find(name);
    if (it == registry().end()) {
        throw std::runtime_error("unknown dataset: " + name);
    }
    nlohmann::json cfg = nlohmann::json::object();
    cfg["type"] = name;
    cfg["instructions"] = it->second.first;
    if (name == "adder") {
        cfg["num_inputs"] = 3;
    } else if (name == "gol") {
        cfg["gol"] = {{"left_range", 4}, {"center_range", 3}, {"right_range", 4}, {"include_alive", true}};
    } else if (name == "sloppy-adder") {
        cfg["num_inputs"] = 4;
        cfg["num_outputs"] = 3;
        cfg["sloppy_adder"] = {{"left_range", 4}, {"right_range", 4}};
    } else if (name == "sloppy-adder3") {
        cfg["num_inputs"] = 6;
        cfg["num_outputs"] = 3;
        cfg["sloppy_adder"] = {{"left_range", 4}, {"center_range", 3}, {"right_range", 4}};
    } else if (name == "excitable") {
        cfg["excitable"] = {{"states", 8}};
    } else if (name == "excitable-compressed") {
        cfg["excitable-compressed"] = {{"states", 8}};
    } else if (name == "cyclic") {
        cfg["cyclic"] = {{"states", 32}};
    } else if (name == "cyclic-compressed") {
        cfg["cyclic-compressed"] = {{"states", 32}};
    }
    return cfg;
}

Config build_dataset_from_config(const nlohmann::json &cfg) {
    if (!cfg.contains("type")) {
        throw std::runtime_error("dataset config without examples must contain a type");
    }
    const std::string name = cfg.at("type").get<std::string>();
    const auto it = registry().find(name);
    if (it == registry().end()) {
        throw std::runtime_error("unknown dataset: " + name);
    }
    Config dataset = it->second.second(cfg);
    dataset.instructions = cfg.value("instructions", dataset.instructions);
    if (dataset.examples.empty()) {
        throw std::runtime_error("Dataset contains no examples");
    }
    return dataset;
}
