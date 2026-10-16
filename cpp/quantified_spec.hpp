#pragma once

#include "datasets.hpp"

#include <cstddef>
#include <cstdint>

#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// Scalar and tuple variable storage used while evaluating a quantified dataset spec.
// Slots are allocated in this order: parameters, static `let` bindings, `forall` variables, `let_row` bindings.
struct EvalEnv {
    std::vector<std::int64_t> scalars;
    std::vector<std::vector<std::int64_t>> tuples;
    std::unordered_map<std::string, std::size_t> scalar_slots;
    std::unordered_map<std::string, std::size_t> tuple_slots;

    std::size_t ensure_scalar_slot(const std::string &name) {
        auto [it, inserted] = scalar_slots.try_emplace(name, scalars.size());
        if (inserted) {
            scalars.push_back(0);
        }
        return it->second;
    }

    std::size_t ensure_tuple_slot(const std::string &name) {
        auto [it, inserted] = tuple_slots.try_emplace(name, tuples.size());
        if (inserted) {
            tuples.emplace_back();
        }
        return it->second;
    }
};

enum class UnaryOp {
    LogicalNot,
    Negate,
    BitNot,
};

enum class BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    LogicalAnd,
    LogicalOr,
};

struct Expr {
    struct Literal {
        std::int64_t value;
    };
    struct NullLiteral {};
    struct ScalarVar {
        std::size_t slot;
    };
    struct TupleIndex {
        std::size_t tuple_slot;
        std::unique_ptr<Expr> index;
    };
    struct Unary {
        UnaryOp op;
        std::unique_ptr<Expr> operand;
    };
    struct Binary {
        BinaryOp op;
        std::unique_ptr<Expr> left;
        std::unique_ptr<Expr> right;
    };
    struct Ternary {
        std::unique_ptr<Expr> cond;
        std::unique_ptr<Expr> then_expr;
        std::unique_ptr<Expr> else_expr;
    };
    struct RequiredBits {
        std::unique_ptr<Expr> arg;
    };
    struct Popcount {
        std::unique_ptr<Expr> arg;
    };
    struct Count {
        std::size_t tuple_slot;
        std::unique_ptr<Expr> target;
        std::unique_ptr<Expr> start_index;
    };
    struct Bit {
        std::unique_ptr<Expr> arg;
    };
    struct Bits {
        std::unique_ptr<Expr> value;
        std::unique_ptr<Expr> width;
    };
    struct StateBits {
        std::size_t tuple_slot;
        std::unique_ptr<Expr> width;
    };

    using Variant = std::variant<Literal, NullLiteral, ScalarVar, TupleIndex, Unary, Binary, Ternary, RequiredBits,
                                 Popcount, Count, Bit, Bits, StateBits>;
    Variant node;

    explicit Expr(Variant v) : node(std::move(v)) {}
};

// One `forall` binder. Domains are half-open: [start, end).
struct Quantifier {
    struct Range {
        std::size_t slot;
        std::int64_t start;
        std::int64_t end;
    };
    struct Values {
        std::size_t slot;
        std::vector<std::int64_t> values;
    };
    struct Tuple {
        std::size_t tuple_slot;
        std::size_t size;
        std::int64_t start;
        std::int64_t end;
    };
    std::variant<Range, Values, Tuple> spec;
};

// Compiled, immutable form of a declarative dataset config. It can be enumerated natively (exact regime),
// sampled, evaluated at a single point, or lowered to quantifier-free Z3 by `SymbolicVerifier`.
struct QuantifiedSpec {
    EvalEnv env;                         // static slots hold resolved values; other slots are scratch
    std::size_t static_slot_count = 0;   // scalar slots below this index are compile-time constants
    std::vector<Quantifier> quantifiers;
    std::vector<std::pair<std::size_t, std::unique_ptr<Expr>>> row_lets;  // in dependency order
    std::unique_ptr<Expr> where;
    std::vector<std::unique_ptr<Expr>> inputs;
    std::vector<std::unique_ptr<Expr>> outputs;
    int num_inputs = 0;
    int num_outputs = 0;
    int instructions = kDefaultProgramLength;
    std::size_t domain_size = 1;         // saturated upper bound on the number of rows (ignores `where`)

    // Evaluates the example at the assignment currently stored in `scratch` (nullopt if filtered by `where`).
    std::optional<Example> evaluate_point(EvalEnv &scratch) const;
    // Overwrites the quantified variables in `scratch` with a uniformly random in-domain assignment.
    void assign_random(EvalEnv &scratch, std::mt19937_64 &rng) const;
};

// Evaluates a scalar expression natively (used for static widths/indices and by the symbolic lowering).
std::int64_t evaluate_scalar(const Expr &expr, const EvalEnv &env);

// Deterministic initial counterexample basis: domain corners followed by random in-domain points.
std::vector<Example> seed_basis(const QuantifiedSpec &spec, std::size_t count, unsigned seed);
