#include "datasets.hpp"

#include "quantified_spec.hpp"

#include <bit>
#include <cctype>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

struct BitList {
    boost::dynamic_bitset<> values;
    boost::dynamic_bitset<> dont_care;

    std::size_t size() const {
        return values.size();
    }

    void push_back(std::optional<bool> value) {
        values.push_back(value.value_or(false));
        dont_care.push_back(!value.has_value());
    }

    std::optional<bool> operator[](std::size_t idx) const {
        if (dont_care[idx]) return std::nullopt;
        return values[idx];
    }
};

void push_bits(BitList &out, std::int64_t value, std::int64_t width) {
    if (width < 0) {
        throw std::runtime_error("bits() width must be non-negative");
    }
    for (std::int64_t bit = 0; bit < width; ++bit) {
        out.push_back(((value >> bit) & 1) != 0);
    }
}

std::int64_t required_bits(std::int64_t state_count) {
    std::int64_t max_value = std::max<std::int64_t>(0, state_count - 1);
    std::int64_t width = 0;
    do {
        ++width;
        max_value >>= 1;
    } while (max_value != 0);
    return width;
}

Example make_example(const BitList &inputs, const BitList &outputs, int num_inputs, int num_outputs) {
    if (static_cast<int>(inputs.size()) != num_inputs || static_cast<int>(outputs.size()) != num_outputs) {
        throw std::runtime_error("Example length does not match declared input/output sizes");
    }
    Example example;
    for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
        const auto input = inputs[idx];
        if (!input.has_value()) {
            throw std::runtime_error("input values cannot be null");
        }
        example.push_input(*input);
    }
    for (std::size_t idx = 0; idx < outputs.size(); ++idx) {
        example.push_output(outputs[idx]);
    }
    return example;
}

std::filesystem::path datasets_directory() {
    if (const char *env_dir = std::getenv("SAT_SYNTH_DATASETS_DIR"); env_dir != nullptr && *env_dir != '\0') {
        std::filesystem::path candidate(env_dir);
        if (std::filesystem::is_directory(candidate)) {
            return candidate;
        }
    }
    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::path(__FILE__).parent_path() / "datasets",
        std::filesystem::current_path() / "cpp" / "datasets",
        std::filesystem::current_path() / "datasets",
    };
    for (const auto &candidate : candidates) {
        if (std::filesystem::is_directory(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("Could not locate datasets directory");
}

nlohmann::json load_dataset_spec(const std::string &name) {
    if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
        name.find("..") != std::string::npos) {
        throw std::runtime_error("unknown dataset: " + name);
    }
    const std::filesystem::path path = datasets_directory() / (name + ".json");
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("unknown dataset: " + name);
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("unknown dataset: " + name);
    }
    nlohmann::json spec;
    in >> spec;
    return spec;
}

std::int64_t eval_scalar(const Expr &expr, const EvalEnv &env) {
    return std::visit(
        [&](const auto &n) -> std::int64_t {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, Expr::Literal>) {
                return n.value;
            } else if constexpr (std::is_same_v<T, Expr::NullLiteral>) {
                throw std::runtime_error("null cannot be evaluated as a scalar");
            } else if constexpr (std::is_same_v<T, Expr::ScalarVar>) {
                return env.scalars[n.slot];
            } else if constexpr (std::is_same_v<T, Expr::TupleIndex>) {
                const std::int64_t idx = eval_scalar(*n.index, env);
                const auto &tuple = env.tuples[n.tuple_slot];
                if (idx < 0 || static_cast<std::size_t>(idx) >= tuple.size()) {
                    throw std::runtime_error("Tuple index out of bounds");
                }
                return tuple[static_cast<std::size_t>(idx)];
            } else if constexpr (std::is_same_v<T, Expr::Unary>) {
                const std::int64_t val = eval_scalar(*n.operand, env);
                switch (n.op) {
                    case UnaryOp::LogicalNot:
                        return val == 0 ? 1 : 0;
                    case UnaryOp::Negate:
                        return -val;
                    case UnaryOp::BitNot:
                        return ~val;
                }
                return 0;
            } else if constexpr (std::is_same_v<T, Expr::Binary>) {
                if (n.op == BinaryOp::LogicalAnd) {
                    return (eval_scalar(*n.left, env) != 0 && eval_scalar(*n.right, env) != 0) ? 1 : 0;
                }
                if (n.op == BinaryOp::LogicalOr) {
                    return (eval_scalar(*n.left, env) != 0 || eval_scalar(*n.right, env) != 0) ? 1 : 0;
                }
                const std::int64_t lhs = eval_scalar(*n.left, env);
                const std::int64_t rhs = eval_scalar(*n.right, env);
                switch (n.op) {
                    case BinaryOp::Add:
                        return lhs + rhs;
                    case BinaryOp::Sub:
                        return lhs - rhs;
                    case BinaryOp::Mul:
                        return lhs * rhs;
                    case BinaryOp::Div:
                        if (rhs == 0) throw std::runtime_error("Division by zero in dataset expression");
                        return lhs / rhs;
                    case BinaryOp::Mod:
                        if (rhs == 0) throw std::runtime_error("Modulo by zero in dataset expression");
                        return lhs % rhs;
                    case BinaryOp::Shl:
                        return lhs << rhs;
                    case BinaryOp::Shr:
                        return lhs >> rhs;
                    case BinaryOp::Lt:
                        return lhs < rhs ? 1 : 0;
                    case BinaryOp::Le:
                        return lhs <= rhs ? 1 : 0;
                    case BinaryOp::Gt:
                        return lhs > rhs ? 1 : 0;
                    case BinaryOp::Ge:
                        return lhs >= rhs ? 1 : 0;
                    case BinaryOp::Eq:
                        return lhs == rhs ? 1 : 0;
                    case BinaryOp::Ne:
                        return lhs != rhs ? 1 : 0;
                    case BinaryOp::BitAnd:
                        return lhs & rhs;
                    case BinaryOp::BitXor:
                        return lhs ^ rhs;
                    case BinaryOp::BitOr:
                        return lhs | rhs;
                    case BinaryOp::LogicalAnd:
                    case BinaryOp::LogicalOr:
                        break;
                }
                return 0;
            } else if constexpr (std::is_same_v<T, Expr::Ternary>) {
                return eval_scalar(*n.cond, env) != 0 ? eval_scalar(*n.then_expr, env) : eval_scalar(*n.else_expr, env);
            } else if constexpr (std::is_same_v<T, Expr::RequiredBits>) {
                return required_bits(eval_scalar(*n.arg, env));
            } else if constexpr (std::is_same_v<T, Expr::Popcount>) {
                const auto val = static_cast<std::uint64_t>(eval_scalar(*n.arg, env));
                return static_cast<std::int64_t>(std::popcount(val));
            } else if constexpr (std::is_same_v<T, Expr::Count>) {
                const auto &tuple = env.tuples[n.tuple_slot];
                const std::int64_t target = eval_scalar(*n.target, env);
                const std::int64_t start = n.start_index ? eval_scalar(*n.start_index, env) : 0;
                std::int64_t matches = 0;
                for (std::size_t idx = static_cast<std::size_t>(std::max<std::int64_t>(0, start)); idx < tuple.size();
                     ++idx) {
                    if (tuple[idx] == target) {
                        ++matches;
                    }
                }
                return matches;
            } else if constexpr (std::is_same_v<T, Expr::Bit>) {
                return eval_scalar(*n.arg, env) != 0 ? 1 : 0;
            } else {
                throw std::runtime_error("Bit-list expression cannot be evaluated as a scalar");
            }
        },
        expr.node);
}

void emit_bits(const Expr &expr, const EvalEnv &env, BitList &out) {
    std::visit(
        [&](const auto &n) {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, Expr::NullLiteral>) {
                out.push_back(std::nullopt);
            } else if constexpr (std::is_same_v<T, Expr::Bit>) {
                out.push_back(eval_scalar(*n.arg, env) != 0);
            } else if constexpr (std::is_same_v<T, Expr::Bits>) {
                push_bits(out, eval_scalar(*n.value, env), eval_scalar(*n.width, env));
            } else if constexpr (std::is_same_v<T, Expr::StateBits>) {
                const std::int64_t width = eval_scalar(*n.width, env);
                for (const std::int64_t state : env.tuples[n.tuple_slot]) {
                    push_bits(out, state, width);
                }
            } else if constexpr (std::is_same_v<T, Expr::Ternary>) {
                if (eval_scalar(*n.cond, env) != 0) {
                    emit_bits(*n.then_expr, env, out);
                } else {
                    emit_bits(*n.else_expr, env, out);
                }
            } else {
                out.push_back(eval_scalar(expr, env) != 0);
            }
        },
        expr.node);
}

template <typename VarResolver, typename TupleResolver>
class ExprParser {
public:
    ExprParser(std::string source, VarResolver resolve_var, TupleResolver resolve_tuple)
        : source_(std::move(source)), resolve_var_(std::move(resolve_var)), resolve_tuple_(std::move(resolve_tuple)) {}

    std::unique_ptr<Expr> parse() {
        auto result = parse_ternary();
        skip_ws();
        if (pos_ != source_.size()) {
            throw std::runtime_error("Unexpected trailing characters in expression: " + source_);
        }
        return result;
    }

private:
    std::string source_;
    std::size_t pos_ = 0;
    VarResolver resolve_var_;
    TupleResolver resolve_tuple_;

    void skip_ws() {
        while (pos_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[pos_])) != 0) {
            ++pos_;
        }
    }

    bool match(std::string_view token) {
        skip_ws();
        if (std::string_view(source_).substr(pos_).starts_with(token)) {
            pos_ += token.size();
            return true;
        }
        return false;
    }

    bool match_op(std::string_view op, std::string_view disallow_prefix = {}) {
        skip_ws();
        const std::string_view rem = std::string_view(source_).substr(pos_);
        if (!disallow_prefix.empty() && rem.starts_with(disallow_prefix)) {
            return false;
        }
        if (rem.starts_with(op)) {
            pos_ += op.size();
            return true;
        }
        return false;
    }

    void expect(std::string_view token) {
        if (!match(token)) {
            throw std::runtime_error("Expected '" + std::string(token) + "' in expression: " + std::string(source_));
        }
    }

    std::string parse_identifier() {
        skip_ws();
        const std::size_t start = pos_;
        while (pos_ < source_.size() &&
               (std::isalnum(static_cast<unsigned char>(source_[pos_])) != 0 || source_[pos_] == '_')) {
            ++pos_;
        }
        if (pos_ == start) {
            throw std::runtime_error("Expected identifier in expression: " + std::string(source_));
        }
        return std::string(source_.substr(start, pos_ - start));
    }

    std::unique_ptr<Expr> parse_ternary() {
        auto expr = parse_logical_or();
        if (match("?")) {
            auto then_expr = parse_ternary();
            expect(":");
            auto else_expr = parse_ternary();
            return std::make_unique<Expr>(Expr::Ternary{
                .cond = std::move(expr),
                .then_expr = std::move(then_expr),
                .else_expr = std::move(else_expr),
            });
        }
        return expr;
    }

    std::unique_ptr<Expr> parse_logical_or() {
        auto lhs = parse_logical_and();
        while (match("||")) {
            auto rhs = parse_logical_and();
            lhs = std::make_unique<Expr>(
                Expr::Binary{.op = BinaryOp::LogicalOr, .left = std::move(lhs), .right = std::move(rhs)});
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_logical_and() {
        auto lhs = parse_bit_or();
        while (match("&&")) {
            auto rhs = parse_bit_or();
            lhs = std::make_unique<Expr>(
                Expr::Binary{.op = BinaryOp::LogicalAnd, .left = std::move(lhs), .right = std::move(rhs)});
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_bit_or() {
        auto lhs = parse_bit_xor();
        while (match_op("|", "||")) {
            auto rhs = parse_bit_xor();
            lhs = std::make_unique<Expr>(
                Expr::Binary{.op = BinaryOp::BitOr, .left = std::move(lhs), .right = std::move(rhs)});
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_bit_xor() {
        auto lhs = parse_bit_and();
        while (match("^")) {
            auto rhs = parse_bit_and();
            lhs = std::make_unique<Expr>(
                Expr::Binary{.op = BinaryOp::BitXor, .left = std::move(lhs), .right = std::move(rhs)});
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_bit_and() {
        auto lhs = parse_equality();
        while (match_op("&", "&&")) {
            auto rhs = parse_equality();
            lhs = std::make_unique<Expr>(
                Expr::Binary{.op = BinaryOp::BitAnd, .left = std::move(lhs), .right = std::move(rhs)});
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_equality() {
        auto lhs = parse_relational();
        while (true) {
            if (match("==")) {
                auto rhs = parse_relational();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Eq, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match("!=")) {
                auto rhs = parse_relational();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Ne, .left = std::move(lhs), .right = std::move(rhs)});
            } else {
                break;
            }
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_relational() {
        auto lhs = parse_shift();
        while (true) {
            if (match("<=")) {
                auto rhs = parse_shift();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Le, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match(">=")) {
                auto rhs = parse_shift();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Ge, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match_op("<", "<<")) {
                auto rhs = parse_shift();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Lt, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match_op(">", ">>")) {
                auto rhs = parse_shift();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Gt, .left = std::move(lhs), .right = std::move(rhs)});
            } else {
                break;
            }
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_shift() {
        auto lhs = parse_additive();
        while (true) {
            if (match("<<")) {
                auto rhs = parse_additive();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Shl, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match(">>")) {
                auto rhs = parse_additive();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Shr, .left = std::move(lhs), .right = std::move(rhs)});
            } else {
                break;
            }
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_additive() {
        auto lhs = parse_multiplicative();
        while (true) {
            if (match("+")) {
                auto rhs = parse_multiplicative();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Add, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match("-")) {
                auto rhs = parse_multiplicative();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Sub, .left = std::move(lhs), .right = std::move(rhs)});
            } else {
                break;
            }
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_multiplicative() {
        auto lhs = parse_unary();
        while (true) {
            if (match("*")) {
                auto rhs = parse_unary();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Mul, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match("/")) {
                auto rhs = parse_unary();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Div, .left = std::move(lhs), .right = std::move(rhs)});
            } else if (match("%")) {
                auto rhs = parse_unary();
                lhs = std::make_unique<Expr>(
                    Expr::Binary{.op = BinaryOp::Mod, .left = std::move(lhs), .right = std::move(rhs)});
            } else {
                break;
            }
        }
        return lhs;
    }

    std::unique_ptr<Expr> parse_unary() {
        if (match_op("!", "!=")) {
            return std::make_unique<Expr>(Expr::Unary{.op = UnaryOp::LogicalNot, .operand = parse_unary()});
        }
        if (match("-")) {
            return std::make_unique<Expr>(Expr::Unary{.op = UnaryOp::Negate, .operand = parse_unary()});
        }
        if (match("+")) {
            return parse_unary();
        }
        if (match("~")) {
            return std::make_unique<Expr>(Expr::Unary{.op = UnaryOp::BitNot, .operand = parse_unary()});
        }
        return parse_primary();
    }

    std::unique_ptr<Expr> parse_primary() {
        skip_ws();
        if (match("(")) {
            auto inner = parse_ternary();
            expect(")");
            return inner;
        }
        if (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_])) != 0) {
            std::int64_t val = 0;
            while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[pos_])) != 0) {
                val = val * 10 + (source_[pos_] - '0');
                ++pos_;
            }
            return std::make_unique<Expr>(Expr::Literal{.value = val});
        }

        const std::string ident = parse_identifier();
        if (ident == "true") {
            return std::make_unique<Expr>(Expr::Literal{.value = 1});
        }
        if (ident == "false") {
            return std::make_unique<Expr>(Expr::Literal{.value = 0});
        }
        if (ident == "null") {
            return std::make_unique<Expr>(Expr::NullLiteral{});
        }

        if (match("(")) {
            if (ident == "required_bits") {
                auto arg = parse_ternary();
                expect(")");
                return std::make_unique<Expr>(Expr::RequiredBits{.arg = std::move(arg)});
            }
            if (ident == "popcount") {
                auto arg = parse_ternary();
                expect(")");
                return std::make_unique<Expr>(Expr::Popcount{.arg = std::move(arg)});
            }
            if (ident == "bit") {
                auto arg = parse_ternary();
                expect(")");
                return std::make_unique<Expr>(Expr::Bit{.arg = std::move(arg)});
            }
            if (ident == "bits") {
                auto val = parse_ternary();
                expect(",");
                auto width = parse_ternary();
                expect(")");
                return std::make_unique<Expr>(Expr::Bits{.value = std::move(val), .width = std::move(width)});
            }
            if (ident == "state_bits") {
                const std::string tuple_name = parse_identifier();
                expect(",");
                auto width = parse_ternary();
                expect(")");
                return std::make_unique<Expr>(
                    Expr::StateBits{.tuple_slot = resolve_tuple_(tuple_name), .width = std::move(width)});
            }
            if (ident == "count") {
                const std::string tuple_name = parse_identifier();
                expect(",");
                auto target = parse_ternary();
                std::unique_ptr<Expr> start_index;
                if (match(",")) {
                    start_index = parse_ternary();
                }
                expect(")");
                return std::make_unique<Expr>(Expr::Count{
                    .tuple_slot = resolve_tuple_(tuple_name),
                    .target = std::move(target),
                    .start_index = std::move(start_index),
                });
            }
            throw std::runtime_error("Unknown function '" + ident + "' in dataset expression");
        }

        if (match("[")) {
            auto idx = parse_ternary();
            expect("]");
            return std::make_unique<Expr>(
                Expr::TupleIndex{.tuple_slot = resolve_tuple_(ident), .index = std::move(idx)});
        }

        return std::make_unique<Expr>(Expr::ScalarVar{.slot = resolve_var_(ident)});
    }
};

const nlohmann::json *lookup_dotted_path(const nlohmann::json &cfg, std::string_view dotted_path) {
    const nlohmann::json *curr = &cfg;
    std::size_t start = 0;
    while (start <= dotted_path.size()) {
        const std::size_t dot = dotted_path.find('.', start);
        const std::string key(
            dotted_path.substr(start, dot == std::string_view::npos ? std::string_view::npos : dot - start));
        if (!curr->is_object() || !curr->contains(key) || curr->at(key).is_null()) {
            return nullptr;
        }
        curr = &curr->at(key);
        if (dot == std::string_view::npos) {
            return curr;
        }
        start = dot + 1;
    }
    return nullptr;
}

std::int64_t read_param_value(const nlohmann::json &runtime_cfg, const nlohmann::json &param_def) {
    const nlohmann::json &default_val = param_def.at("default");
    const nlohmann::json *found = nullptr;
    if (param_def.contains("path")) {
        const nlohmann::json &path_spec = param_def.at("path");
        if (path_spec.is_string()) {
            found = lookup_dotted_path(runtime_cfg, path_spec.get<std::string>());
        } else if (path_spec.is_array()) {
            for (const auto &candidate : path_spec) {
                found = lookup_dotted_path(runtime_cfg, candidate.get<std::string>());
                if (found != nullptr) {
                    break;
                }
            }
        }
    }
    const nlohmann::json &target = found != nullptr ? *found : default_val;
    if (default_val.is_boolean()) {
        return target.get<bool>() ? 1 : 0;
    }
    return target.get<std::int64_t>();
}

std::size_t saturating_mul(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return std::numeric_limits<std::size_t>::max();
    }
    return lhs * rhs;
}

std::shared_ptr<QuantifiedSpec> compile_dataset_spec(const nlohmann::json &spec, const nlohmann::json &runtime_cfg) {
    auto compiled = std::make_shared<QuantifiedSpec>();
    EvalEnv &env = compiled->env;

    if (spec.contains("params")) {
        for (const auto &[name, param_def] : spec.at("params").items()) {
            const std::size_t slot = env.ensure_scalar_slot(name);
            env.scalars[slot] = read_param_value(runtime_cfg, param_def);
        }
    }

    std::unordered_map<std::string, std::string> raw_static_lets;
    if (spec.contains("let")) {
        for (const auto &[name, expr_json] : spec.at("let").items()) {
            raw_static_lets.emplace(name, expr_json.is_string() ? expr_json.get<std::string>() : expr_json.dump());
        }
    }

    std::set<std::string> compiling_static;
    const auto resolve_static_var = [&](this const auto &self, const std::string &name) -> std::size_t {
        if (auto it = env.scalar_slots.find(name); it != env.scalar_slots.end()) {
            return it->second;
        }
        if (auto it = raw_static_lets.find(name); it != raw_static_lets.end()) {
            if (!compiling_static.insert(name).second) {
                throw std::runtime_error("Cyclic let definition for variable: " + name);
            }
            ExprParser parser(it->second, self, [](const std::string &t) -> std::size_t {
                throw std::runtime_error("Tuple variable '" + t + "' is not available in static context");
            });
            auto ast = parser.parse();
            const std::int64_t value = eval_scalar(*ast, env);
            const std::size_t slot = env.ensure_scalar_slot(name);
            env.scalars[slot] = value;
            compiling_static.erase(name);
            return slot;
        }
        throw std::runtime_error("Unknown variable in static dataset expression: " + name);
    };

    for (const auto &[name, _] : raw_static_lets) {
        (void)resolve_static_var(name);
    }
    compiled->static_slot_count = env.scalars.size();

    auto eval_static_expr = [&](const nlohmann::json &node) -> std::int64_t {
        if (node.is_number_integer()) {
            return node.get<std::int64_t>();
        }
        if (node.is_boolean()) {
            return node.get<bool>() ? 1 : 0;
        }
        if (node.is_string()) {
            ExprParser parser(node.get<std::string>(), resolve_static_var, [](const std::string &t) -> std::size_t {
                throw std::runtime_error("Tuple variable '" + t + "' is not available in static context");
            });
            return eval_scalar(*parser.parse(), env);
        }
        throw std::runtime_error("Expected integer, boolean, or expression string in dataset config");
    };

    if (spec.contains("constraints")) {
        for (const auto &constraint : spec.at("constraints")) {
            if (eval_static_expr(constraint.at("check")) == 0) {
                throw std::runtime_error(constraint.at("error").get<std::string>());
            }
        }
    }

    compiled->num_inputs = static_cast<int>(eval_static_expr(spec.at("num_inputs")));
    compiled->num_outputs = static_cast<int>(eval_static_expr(spec.at("num_outputs")));
    const int default_instructions =
        spec.contains("instructions") ? static_cast<int>(eval_static_expr(spec.at("instructions"))) : kDefaultProgramLength;
    compiled->instructions = runtime_cfg.value("instructions", default_instructions);

    if (spec.contains("forall")) {
        for (const auto &q_json : spec.at("forall")) {
            const std::string var_name = q_json.at("var").get<std::string>();
            if (q_json.contains("range")) {
                const auto &range = q_json.at("range");
                const std::int64_t start = eval_static_expr(range.at(0));
                const std::int64_t end = eval_static_expr(range.at(1));
                const std::size_t slot = env.ensure_scalar_slot(var_name);
                compiled->quantifiers.push_back(Quantifier{Quantifier::Range{.slot = slot, .start = start, .end = end}});
                compiled->domain_size = saturating_mul(
                    compiled->domain_size, static_cast<std::size_t>(std::max<std::int64_t>(0, end - start)));
            } else if (q_json.contains("values")) {
                std::vector<std::int64_t> values;
                for (const auto &val_json : q_json.at("values")) {
                    values.push_back(eval_static_expr(val_json));
                }
                const std::size_t slot = env.ensure_scalar_slot(var_name);
                compiled->domain_size = saturating_mul(compiled->domain_size, values.size());
                compiled->quantifiers.push_back(Quantifier{Quantifier::Values{.slot = slot, .values = std::move(values)}});
            } else if (q_json.contains("tuple")) {
                const auto &tuple_spec = q_json.at("tuple");
                const auto size = static_cast<std::size_t>(eval_static_expr(tuple_spec.at("size")));
                const auto &range = tuple_spec.at("range");
                const std::int64_t start = eval_static_expr(range.at(0));
                const std::int64_t end = eval_static_expr(range.at(1));
                const std::size_t tuple_slot = env.ensure_tuple_slot(var_name);
                env.tuples[tuple_slot].assign(size, start);
                const auto span = static_cast<std::size_t>(std::max<std::int64_t>(0, end - start));
                for (std::size_t idx = 0; idx < size; ++idx) {
                    compiled->domain_size = saturating_mul(compiled->domain_size, span);
                }
                compiled->quantifiers.push_back(Quantifier{
                    Quantifier::Tuple{.tuple_slot = tuple_slot, .size = size, .start = start, .end = end}});
            } else {
                throw std::runtime_error("Unsupported quantifier specification for variable: " + var_name);
            }
        }
    }

    if (spec.contains("sample_section")) {
        const std::string section = spec.at("sample_section").get<std::string>();
        if (runtime_cfg.contains(section) && runtime_cfg.at(section).is_object() &&
            runtime_cfg.at(section).contains("max_examples")) {
            const std::size_t max_examples = runtime_cfg.at(section).at("max_examples").get<std::size_t>();
            if (max_examples < compiled->domain_size) {
                if (max_examples == 0) {
                    throw std::runtime_error(section + ".max_examples must be positive");
                }
                throw std::runtime_error(section +
                                         ".max_examples sampling is not supported by the C++ dataset generator yet");
            }
        }
    }

    std::unordered_map<std::string, std::string> raw_row_lets;
    if (spec.contains("let_row")) {
        for (const auto &[name, expr_json] : spec.at("let_row").items()) {
            raw_row_lets.emplace(name, expr_json.is_string() ? expr_json.get<std::string>() : expr_json.dump());
        }
    }

    std::set<std::string> compiling_row;

    auto resolve_tuple_var = [&](const std::string &name) -> std::size_t {
        if (auto it = env.tuple_slots.find(name); it != env.tuple_slots.end()) {
            return it->second;
        }
        throw std::runtime_error("Unknown tuple variable in dataset expression: " + name);
    };

    const auto resolve_row_var = [&](this const auto &self, const std::string &name) -> std::size_t {
        if (auto it = env.scalar_slots.find(name); it != env.scalar_slots.end()) {
            return it->second;
        }
        if (auto it = raw_row_lets.find(name); it != raw_row_lets.end()) {
            if (!compiling_row.insert(name).second) {
                throw std::runtime_error("Cyclic let_row definition for variable: " + name);
            }
            ExprParser parser(it->second, self, resolve_tuple_var);
            auto ast = parser.parse();
            const std::size_t slot = env.ensure_scalar_slot(name);
            compiled->row_lets.emplace_back(slot, std::move(ast));
            compiling_row.erase(name);
            return slot;
        }
        throw std::runtime_error("Unknown variable in dataset expression: " + name);
    };

    for (const auto &[name, _] : raw_row_lets) {
        (void)resolve_row_var(name);
    }

    auto compile_row_expr = [&](const nlohmann::json &node) -> std::unique_ptr<Expr> {
        const std::string text = node.is_string() ? node.get<std::string>() : node.dump();
        ExprParser parser(text, resolve_row_var, resolve_tuple_var);
        return parser.parse();
    };

    if (spec.contains("where")) {
        compiled->where = compile_row_expr(spec.at("where"));
    }
    for (const auto &in_expr : spec.at("inputs")) {
        compiled->inputs.push_back(compile_row_expr(in_expr));
    }
    for (const auto &out_expr : spec.at("outputs")) {
        compiled->outputs.push_back(compile_row_expr(out_expr));
    }
    return compiled;
}

template <typename Visit>
// Calls `visit` with `scratch` set to every quantifier assignment, in the order used by the Python plugins.
void for_each_point(const QuantifiedSpec &spec, EvalEnv &scratch, const Visit &visit) {
    const auto run = [&spec, &visit, &scratch](this const auto &self, std::size_t q_idx) -> void {
        if (q_idx == spec.quantifiers.size()) {
            visit(scratch);
            return;
        }

        std::visit(
            [&scratch, &self, q_idx](const auto &q) {
                using T = std::decay_t<decltype(q)>;
                if constexpr (std::is_same_v<T, Quantifier::Range>) {
                    for (std::int64_t val = q.start; val < q.end; ++val) {
                        scratch.scalars[q.slot] = val;
                        self(q_idx + 1);
                    }
                } else if constexpr (std::is_same_v<T, Quantifier::Values>) {
                    for (const std::int64_t val : q.values) {
                        scratch.scalars[q.slot] = val;
                        self(q_idx + 1);
                    }
                } else {
                    const auto fill = [&](this auto &fill_self, std::size_t cell) -> void {
                        if (cell == q.size) {
                            self(q_idx + 1);
                            return;
                        }
                        for (std::int64_t val = q.start; val < q.end; ++val) {
                            scratch.tuples[q.tuple_slot][cell] = val;
                            fill_self(cell + 1);
                        }
                    };
                    fill(0);
                }
            },
            spec.quantifiers[q_idx].spec);
    };
    run(0);
}

}  // namespace

Example::Example(std::initializer_list<bool> input_values, std::initializer_list<std::optional<bool>> output_values) {
    for (const bool value : input_values) {
        push_input(value);
    }
    for (const auto &value : output_values) {
        push_output(value);
    }
}

std::size_t Example::input_count() const {
    return inputs.size();
}

std::size_t Example::output_count() const {
    return output_values.size();
}

bool Example::input(std::size_t idx) const {
    return inputs[idx];
}

std::optional<bool> Example::output(std::size_t idx) const {
    if (output_dont_care[idx]) {
        return std::nullopt;
    }
    return output_values[idx];
}

void Example::push_input(bool value) {
    inputs.push_back(value);
}

void Example::push_output(std::optional<bool> value) {
    output_values.push_back(value.value_or(false));
    output_dont_care.push_back(!value.has_value());
}

std::vector<std::string> available_dataset_names() {
    std::vector<std::string> names;
    for (const auto &entry : std::filesystem::directory_iterator(datasets_directory())) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            names.push_back(entry.path().stem().string());
        }
    }
    std::ranges::sort(names);
    return names;
}

nlohmann::json default_dataset_config(const std::string &name) {
    const nlohmann::json spec = load_dataset_spec(name);
    nlohmann::json cfg = spec.value("defaults", nlohmann::json::object());
    cfg["type"] = name;
    cfg["instructions"] = spec.value("instructions", kDefaultProgramLength);
    return cfg;
}

std::int64_t evaluate_scalar(const Expr &expr, const EvalEnv &env) {
    return eval_scalar(expr, env);
}

std::optional<Example> QuantifiedSpec::evaluate_point(EvalEnv &scratch) const {
    for (const auto &[slot, ast] : row_lets) {
        scratch.scalars[slot] = eval_scalar(*ast, scratch);
    }
    if (where && eval_scalar(*where, scratch) == 0) {
        return std::nullopt;
    }
    BitList in_bits;
    for (const auto &ast : inputs) {
        emit_bits(*ast, scratch, in_bits);
    }
    BitList out_bits;
    for (const auto &ast : outputs) {
        emit_bits(*ast, scratch, out_bits);
    }
    return make_example(in_bits, out_bits, num_inputs, num_outputs);
}

void QuantifiedSpec::assign_random(EvalEnv &scratch, std::mt19937_64 &rng) const {
    auto pick = [&](std::int64_t start, std::int64_t end) {
        return std::uniform_int_distribution<std::int64_t>(start, std::max(start, end - 1))(rng);
    };
    for (const Quantifier &quantifier : quantifiers) {
        std::visit(
            [&](const auto &q) {
                using T = std::decay_t<decltype(q)>;
                if constexpr (std::is_same_v<T, Quantifier::Range>) {
                    scratch.scalars[q.slot] = pick(q.start, q.end);
                } else if constexpr (std::is_same_v<T, Quantifier::Values>) {
                    if (!q.values.empty()) {
                        scratch.scalars[q.slot] = q.values[pick(0, static_cast<std::int64_t>(q.values.size()))];
                    }
                } else {
                    for (std::size_t cell = 0; cell < q.size; ++cell) {
                        scratch.tuples[q.tuple_slot][cell] = pick(q.start, q.end);
                    }
                }
            },
            quantifier.spec);
    }
}

namespace {

// Moves every quantified variable to the low or high end of its domain.
void assign_corner(const QuantifiedSpec &spec, EvalEnv &env, bool high) {
    for (const Quantifier &quantifier : spec.quantifiers) {
        std::visit(
            [&](const auto &q) {
                using T = std::decay_t<decltype(q)>;
                if constexpr (std::is_same_v<T, Quantifier::Range>) {
                    env.scalars[q.slot] = high ? std::max(q.start, q.end - 1) : q.start;
                } else if constexpr (std::is_same_v<T, Quantifier::Values>) {
                    if (!q.values.empty()) env.scalars[q.slot] = high ? q.values.back() : q.values.front();
                } else {
                    env.tuples[q.tuple_slot].assign(q.size, high ? std::max(q.start, q.end - 1) : q.start);
                }
            },
            quantifier.spec);
    }
}

}  // namespace

std::vector<Example> seed_basis(const QuantifiedSpec &spec, std::size_t count, unsigned seed) {
    std::vector<Example> basis;
    std::set<boost::dynamic_bitset<>> seen;
    EvalEnv env = spec.env;
    count = std::max<std::size_t>(count, 1);

    auto add_current = [&]() {
        if (basis.size() >= count) return;
        std::optional<Example> example = spec.evaluate_point(env);
        if (!example.has_value()) return;
        if (seen.insert(example->inputs).second) basis.push_back(std::move(*example));
    };

    for (const bool high : {false, true}) {
        assign_corner(spec, env, high);
        add_current();
    }
    // Single-variable and single-cell deviations from each corner exercise every quantifier position.
    for (const Quantifier &quantifier : spec.quantifiers) {
        std::visit(
            [&](const auto &q) {
                using T = std::decay_t<decltype(q)>;
                if constexpr (std::is_same_v<T, Quantifier::Range>) {
                    const std::int64_t low = q.start;
                    const std::int64_t high = std::max(q.start, q.end - 1);
                    for (const bool base_high : {false, true}) {
                        assign_corner(spec, env, base_high);
                        env.scalars[q.slot] = base_high ? low : high;
                        add_current();
                    }
                } else if constexpr (std::is_same_v<T, Quantifier::Values>) {
                    if (!q.values.empty()) {
                        for (const bool base_high : {false, true}) {
                            assign_corner(spec, env, base_high);
                            env.scalars[q.slot] = base_high ? q.values.front() : q.values.back();
                            add_current();
                        }
                    }
                } else {
                    const std::int64_t low = q.start;
                    const std::int64_t high = std::max(q.start, q.end - 1);
                    for (const bool base_high : {false, true}) {
                        for (std::size_t cell = 0; cell < q.size; ++cell) {
                            assign_corner(spec, env, base_high);
                            env.tuples[q.tuple_slot][cell] = base_high ? low : high;
                            add_current();
                        }
                    }
                }
            },
            quantifier.spec);
    }

    std::mt19937_64 rng(seed);
    const std::size_t max_attempts = count * 64 + 256;
    for (std::size_t attempt = 0; attempt < max_attempts && basis.size() < count; ++attempt) {
        spec.assign_random(env, rng);
        add_current();
    }
    return basis;
}

Config build_dataset_from_config(const nlohmann::json &cfg, const DatasetLoadOptions &options) {
    nlohmann::json spec;
    if (cfg.contains("forall")) {
        spec = cfg;
    } else {
        if (!cfg.contains("type")) {
            throw std::runtime_error("dataset config without examples must contain a type");
        }
        const std::string name = cfg.at("type").get<std::string>();
        spec = load_dataset_spec(name);
    }

    const std::shared_ptr<QuantifiedSpec> compiled = compile_dataset_spec(spec, cfg);
    Config dataset {
        .examples = {},
        .num_inputs = compiled->num_inputs,
        .num_outputs = compiled->num_outputs,
        .instructions = compiled->instructions,
        .quantified = compiled,
        .fully_materialized = true,
        .domain_size = compiled->domain_size,
    };

    if (compiled->domain_size <= options.max_materialized_examples) {
        EvalEnv env = compiled->env;
        for_each_point(*compiled, env, [&dataset, compiled](EvalEnv &point) {
            if (auto example = compiled->evaluate_point(point)) {
                if (example) {
                    dataset.examples.push_back(std::move(*example));
                }
            }
        });
    } else {
        dataset.fully_materialized = false;
        dataset.examples = seed_basis(*compiled, options.basis_size, options.seed);
    }

    if (dataset.examples.empty()) {
        throw std::runtime_error("Dataset contains no examples");
    }
    return dataset;
}
