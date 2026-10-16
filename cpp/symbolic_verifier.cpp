#include "symbolic_verifier.hpp"

#include "encoding.hpp"

#include <boost/dynamic_bitset.hpp>
#include <z3++.h>

#include <bit>
#include <chrono>
#include <cstdint>

#include <algorithm>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr unsigned kWidth = 32;  // every integer expression is lowered to a 32-bit signed bit-vector

using Bit = std::pair<z3::expr, z3::expr>;  // (value, care)

z3::expr balanced_sum(z3::context &ctx, std::vector<z3::expr> terms) {
    if (terms.empty()) return ctx.bv_val(0, kWidth);
    while (terms.size() > 1) {
        std::vector<z3::expr> next;
        next.reserve((terms.size() + 1) / 2);
        for (std::size_t i = 0; i + 1 < terms.size(); i += 2) {
            next.push_back(terms[i] + terms[i + 1]);
        }
        if (terms.size() % 2 == 1) {
            next.push_back(std::move(terms.back()));
        }
        terms = std::move(next);
    }
    return std::move(terms.front());
}

class Lowering {
public:
    Lowering(z3::context &ctx, const QuantifiedSpec &spec) : ctx_(ctx), static_env_(spec.env) {
        scalars_.resize(spec.env.scalars.size());
        is_static_slot_.assign(spec.env.scalars.size(), false);
        for (std::size_t slot = 0; slot < spec.static_slot_count; ++slot) {
            scalars_[slot] = bv(spec.env.scalars[slot]);
            is_static_slot_[slot] = true;
        }
        tuples_.resize(spec.env.tuples.size());
    }

    void bind_scalar(std::size_t slot, const z3::expr &value) {
        scalars_[slot] = value;
        is_static_slot_[slot] = false;
    }

    void bind_let_row(std::size_t slot, const Expr &ast) {
        if (is_static(ast)) {
            const std::int64_t val = static_value(ast);
            static_env_.scalars[slot] = val;
            scalars_[slot] = bv(val);
            is_static_slot_[slot] = true;
        } else {
            scalars_[slot] = to_int(ast);
            is_static_slot_[slot] = false;
        }
    }

    void bind_tuple_cell(std::size_t tuple_slot, const z3::expr &value) {
        tuples_[tuple_slot].push_back(value);
    }

    z3::expr bv(std::int64_t value) const {
        return ctx_.bv_val(value, kWidth);
    }

    bool is_static(const Expr &expr) const {
        return std::visit(
            [&](const auto &n) -> bool {
                using T = std::decay_t<decltype(n)>;
                if constexpr (std::is_same_v<T, Expr::Literal>) {
                    return true;
                } else if constexpr (std::is_same_v<T, Expr::NullLiteral>) {
                    return false;
                } else if constexpr (std::is_same_v<T, Expr::ScalarVar>) {
                    return n.slot < is_static_slot_.size() && is_static_slot_[n.slot];
                } else if constexpr (std::is_same_v<T, Expr::Unary>) {
                    return is_static(*n.operand);
                } else if constexpr (std::is_same_v<T, Expr::Binary>) {
                    return is_static(*n.left) && is_static(*n.right);
                } else if constexpr (std::is_same_v<T, Expr::Ternary>) {
                    return is_static(*n.cond) && is_static(*n.then_expr) && is_static(*n.else_expr);
                } else if constexpr (std::is_same_v<T, Expr::RequiredBits> || std::is_same_v<T, Expr::Popcount> ||
                                     std::is_same_v<T, Expr::Bit>) {
                    return is_static(*n.arg);
                } else {
                    return false;  // tuple access, count, bits and state_bits depend on quantified variables
                }
            },
            expr.node);
    }

    std::int64_t static_value(const Expr &expr) const {
        return evaluate_scalar(expr, static_env_);
    }

    z3::expr to_int(const Expr &expr) {
        if (is_static(expr)) return bv(static_value(expr));
        return std::visit(
            [&](const auto &n) -> z3::expr {
                using T = std::decay_t<decltype(n)>;
                if constexpr (std::is_same_v<T, Expr::ScalarVar>) {
                    if (!scalars_[n.slot].has_value()) {
                        throw std::runtime_error("dataset expression uses a variable before it is defined");
                    }
                    return *scalars_[n.slot];
                } else if constexpr (std::is_same_v<T, Expr::TupleIndex>) {
                    const auto &cells = tuples_.at(n.tuple_slot);
                    if (cells.empty()) throw std::runtime_error("empty tuple in dataset expression");
                    if (is_static(*n.index)) {
                        const std::int64_t idx = static_value(*n.index);
                        if (idx < 0 || static_cast<std::size_t>(idx) >= cells.size()) {
                            throw std::runtime_error("Tuple index out of bounds");
                        }
                        return cells[static_cast<std::size_t>(idx)];
                    }
                    const z3::expr idx = to_int(*n.index);
                    auto select_range = [&](this auto &self, std::size_t lo, std::size_t hi) -> z3::expr {
                        if (lo + 1 == hi) return cells[lo];
                        const std::size_t mid = lo + (hi - lo) / 2;
                        return z3::ite(idx < bv(static_cast<std::int64_t>(mid)), self(lo, mid), self(mid, hi));
                    };
                    return select_range(0, cells.size());
                } else if constexpr (std::is_same_v<T, Expr::Unary>) {
                    switch (n.op) {
                        case UnaryOp::LogicalNot:
                            return from_bool(!to_bool(*n.operand));
                        case UnaryOp::Negate:
                            return -to_int(*n.operand);
                        case UnaryOp::BitNot:
                            return ~to_int(*n.operand);
                    }
                    throw std::logic_error("unreachable unary operator");
                } else if constexpr (std::is_same_v<T, Expr::Binary>) {
                    switch (n.op) {
                        case BinaryOp::LogicalAnd:
                        case BinaryOp::LogicalOr:
                        case BinaryOp::Lt:
                        case BinaryOp::Le:
                        case BinaryOp::Gt:
                        case BinaryOp::Ge:
                        case BinaryOp::Eq:
                        case BinaryOp::Ne:
                            return from_bool(to_bool(expr));
                        default:
                            break;
                    }
                    const z3::expr lhs = to_int(*n.left);
                    const z3::expr rhs = to_int(*n.right);
                    switch (n.op) {
                        case BinaryOp::Add:
                            return lhs + rhs;
                        case BinaryOp::Sub:
                            return lhs - rhs;
                        case BinaryOp::Mul:
                            return lhs * rhs;
                        case BinaryOp::Div:
                            return z3::to_expr(ctx_, Z3_mk_bvsdiv(ctx_, lhs, rhs));
                        case BinaryOp::Mod:
                            return z3::srem(lhs, rhs);
                        case BinaryOp::Shl:
                            return z3::shl(lhs, rhs);
                        case BinaryOp::Shr:
                            return z3::ashr(lhs, rhs);
                        case BinaryOp::BitAnd:
                            return lhs & rhs;
                        case BinaryOp::BitXor:
                            return lhs ^ rhs;
                        case BinaryOp::BitOr:
                            return lhs | rhs;
                        default:
                            break;
                    }
                    throw std::logic_error("unreachable binary operator");
                } else if constexpr (std::is_same_v<T, Expr::Ternary>) {
                    return z3::ite(to_bool(*n.cond), to_int(*n.then_expr), to_int(*n.else_expr));
                } else if constexpr (std::is_same_v<T, Expr::RequiredBits>) {
                    throw std::runtime_error("required_bits() argument must not depend on quantified variables");
                } else if constexpr (std::is_same_v<T, Expr::Popcount>) {
                    const z3::expr arg = to_int(*n.arg);
                    std::vector<z3::expr> terms;
                    terms.reserve(kWidth);
                    for (unsigned bit = 0; bit < kWidth; ++bit) {
                        terms.push_back(z3::zext(arg.extract(bit, bit), kWidth - 1));
                    }
                    return balanced_sum(ctx_, std::move(terms));
                } else if constexpr (std::is_same_v<T, Expr::Count>) {
                    const auto &cells = tuples_.at(n.tuple_slot);
                    std::int64_t first = 0;
                    if (n.start_index) {
                        if (!is_static(*n.start_index)) {
                            throw std::runtime_error("count() start index must not depend on quantified variables");
                        }
                        first = std::max<std::int64_t>(0, static_value(*n.start_index));
                    }
                    const z3::expr target = to_int(*n.target);
                    std::vector<z3::expr> terms;
                    for (std::size_t cell = static_cast<std::size_t>(first); cell < cells.size(); ++cell) {
                        terms.push_back(z3::ite(cells[cell] == target, bv(1), bv(0)));
                    }
                    return balanced_sum(ctx_, std::move(terms));
                } else if constexpr (std::is_same_v<T, Expr::Bit>) {
                    return from_bool(to_bool(*n.arg));
                } else {
                    throw std::runtime_error("bit-list expression used where a scalar is required");
                }
            },
            expr.node);
    }

    z3::expr to_bool(const Expr &expr) {
        if (is_static(expr)) return ctx_.bool_val(static_value(expr) != 0);
        if (const auto *unary = std::get_if<Expr::Unary>(&expr.node); unary != nullptr && unary->op == UnaryOp::LogicalNot) {
            return !to_bool(*unary->operand);
        }
        if (const auto *binary = std::get_if<Expr::Binary>(&expr.node)) {
            switch (binary->op) {
                case BinaryOp::LogicalAnd:
                    return to_bool(*binary->left) && to_bool(*binary->right);
                case BinaryOp::LogicalOr:
                    return to_bool(*binary->left) || to_bool(*binary->right);
                case BinaryOp::Lt:
                    return to_int(*binary->left) < to_int(*binary->right);
                case BinaryOp::Le:
                    return to_int(*binary->left) <= to_int(*binary->right);
                case BinaryOp::Gt:
                    return to_int(*binary->left) > to_int(*binary->right);
                case BinaryOp::Ge:
                    return to_int(*binary->left) >= to_int(*binary->right);
                case BinaryOp::Eq:
                    return to_int(*binary->left) == to_int(*binary->right);
                case BinaryOp::Ne:
                    return to_int(*binary->left) != to_int(*binary->right);
                default:
                    break;
            }
        }
        if (const auto *ternary = std::get_if<Expr::Ternary>(&expr.node)) {
            return z3::ite(to_bool(*ternary->cond), to_bool(*ternary->then_expr), to_bool(*ternary->else_expr));
        }
        if (const auto *bit = std::get_if<Expr::Bit>(&expr.node)) {
            return to_bool(*bit->arg);
        }
        return to_int(expr) != bv(0);
    }

    // Appends the (value, care) pairs produced by an input/output emitter.
    void emit(const Expr &expr, std::vector<Bit> &out) {
        if (std::holds_alternative<Expr::NullLiteral>(expr.node)) {
            out.emplace_back(ctx_.bool_val(false), ctx_.bool_val(false));
        } else if (const auto *bit = std::get_if<Expr::Bit>(&expr.node)) {
            out.emplace_back(to_bool(*bit->arg), ctx_.bool_val(true));
        } else if (const auto *bits = std::get_if<Expr::Bits>(&expr.node)) {
            emit_value_bits(to_int(*bits->value), static_width(*bits->width), out);
        } else if (const auto *state_bits = std::get_if<Expr::StateBits>(&expr.node)) {
            const unsigned width = static_width(*state_bits->width);
            for (const z3::expr &cell : tuples_.at(state_bits->tuple_slot)) {
                emit_value_bits(cell, width, out);
            }
        } else if (const auto *ternary = std::get_if<Expr::Ternary>(&expr.node)) {
            if (is_static(*ternary->cond)) {
                emit(static_value(*ternary->cond) != 0 ? *ternary->then_expr : *ternary->else_expr, out);
                return;
            }
            std::vector<Bit> then_bits;
            std::vector<Bit> else_bits;
            emit(*ternary->then_expr, then_bits);
            emit(*ternary->else_expr, else_bits);
            if (then_bits.size() != else_bits.size()) {
                throw std::runtime_error("ternary branches of a bit emitter must produce the same number of bits");
            }
            const z3::expr cond = to_bool(*ternary->cond);
            for (std::size_t idx = 0; idx < then_bits.size(); ++idx) {
                out.emplace_back(z3::ite(cond, then_bits[idx].first, else_bits[idx].first),
                                 z3::ite(cond, then_bits[idx].second, else_bits[idx].second));
            }
        } else {
            out.emplace_back(to_bool(expr), ctx_.bool_val(true));
        }
    }

private:
    z3::context &ctx_;
    EvalEnv static_env_;
    std::vector<std::optional<z3::expr>> scalars_;
    std::vector<bool> is_static_slot_;
    std::vector<std::vector<z3::expr>> tuples_;

    z3::expr from_bool(const z3::expr &cond) const {
        return z3::ite(cond, bv(1), bv(0));
    }

    unsigned static_width(const Expr &expr) const {
        if (!is_static(expr)) throw std::runtime_error("bit widths must not depend on quantified variables");
        const std::int64_t width = static_value(expr);
        if (width < 0) throw std::runtime_error("bits() width must be non-negative");
        return static_cast<unsigned>(width);
    }

    void emit_value_bits(const z3::expr &value, unsigned width, std::vector<Bit> &out) const {
        for (unsigned bit = 0; bit < width; ++bit) {
            const unsigned source = std::min(bit, kWidth - 1);  // bits above 31 replicate the sign bit
            out.emplace_back(value.extract(source, source) == ctx_.bv_val(1, 1), ctx_.bool_val(true));
        }
    }
};

unsigned bits_for_max(std::int64_t max_value) {
    return std::max(
        1U, static_cast<unsigned>(std::bit_width(static_cast<std::uint64_t>(std::max<std::int64_t>(0, max_value)))));
}

struct SolverScope {
    z3::solver &solver;
    explicit SolverScope(z3::solver &s) : solver(s) { solver.push(); }
    ~SolverScope() { solver.pop(); }

    SolverScope(const SolverScope &) = delete;
    SolverScope &operator=(const SolverScope &) = delete;
};

}  // namespace

SymbolicVerifier::SymbolicVerifier(std::shared_ptr<const QuantifiedSpec> spec)
    : spec_(std::move(spec)), solver_(ctx_, "QF_BV") {
    if (!spec_) throw std::invalid_argument("SymbolicVerifier requires a quantified dataset spec");
    solver_ = make_solver(ctx_, "simple-tactic");
    Lowering lowering(ctx_, *spec_);

    // Declares one narrow variable per quantified scalar/cell and constrains it to [start, end).
    auto declare_range = [&](const std::string &name, std::int64_t start, std::int64_t end) -> z3::expr {
        if (start < 0 || end > (std::int64_t{1} << 30)) {
            throw std::runtime_error("quantifier domain is outside the supported symbolic range [0, 2^30]");
        }
        if (end <= start) {
            domain_constraints_.push_back(ctx_.bool_val(false));
            return ctx_.bv_const(name.c_str(), 1);
        }
        const unsigned width = bits_for_max(end - 1);
        const z3::expr raw = ctx_.bv_const(name.c_str(), width);
        if (start > 0) domain_constraints_.push_back(z3::uge(raw, ctx_.bv_val(start, width)));
        if (end < (std::int64_t{1} << width)) domain_constraints_.push_back(z3::ult(raw, ctx_.bv_val(end, width)));
        return raw;
    };
    auto widen = [&](const z3::expr &raw) { return z3::zext(raw, kWidth - raw.get_sort().bv_size()); };

    for (const Quantifier &quantifier : spec_->quantifiers) {
        std::visit(
            [&](const auto &q) {
                using T = std::decay_t<decltype(q)>;
                if constexpr (std::is_same_v<T, Quantifier::Range>) {
                    const z3::expr raw = declare_range("q_s" + std::to_string(q.slot), q.start, q.end);
                    lowering.bind_scalar(q.slot, widen(raw));
                    bindings_.push_back({VarBinding::Kind::Scalar, q.slot, 0, raw});
                } else if constexpr (std::is_same_v<T, Quantifier::Values>) {
                    if (q.values.empty()) {
                        domain_constraints_.push_back(ctx_.bool_val(false));
                    }
                    std::int64_t max_value = 0;
                    for (const std::int64_t value : q.values) {
                        if (value < 0 || value > (std::int64_t{1} << 30)) {
                            throw std::runtime_error("quantifier value is outside the supported symbolic range");
                        }
                        max_value = std::max(max_value, value);
                    }
                    const unsigned width = bits_for_max(max_value);
                    const z3::expr raw = ctx_.bv_const(("q_s" + std::to_string(q.slot)).c_str(), width);
                    z3::expr any = ctx_.bool_val(false);
                    for (const std::int64_t value : q.values) any = any || (raw == ctx_.bv_val(value, width));
                    domain_constraints_.push_back(any);
                    lowering.bind_scalar(q.slot, widen(raw));
                    bindings_.push_back({VarBinding::Kind::Scalar, q.slot, 0, raw});
                } else {
                    for (std::size_t cell = 0; cell < q.size; ++cell) {
                        const z3::expr raw = declare_range(
                            "q_t" + std::to_string(q.tuple_slot) + "_" + std::to_string(cell), q.start, q.end);
                        lowering.bind_tuple_cell(q.tuple_slot, widen(raw));
                        bindings_.push_back({VarBinding::Kind::TupleCell, q.tuple_slot, cell, raw});
                    }
                }
            },
            quantifier.spec);
    }

    for (const auto &[slot, ast] : spec_->row_lets) {
        lowering.bind_let_row(slot, *ast);
    }
    if (spec_->where) domain_constraints_.push_back(lowering.to_bool(*spec_->where));

    std::vector<Bit> input_bits;
    for (const auto &ast : spec_->inputs) lowering.emit(*ast, input_bits);
    std::vector<Bit> output_bits;
    for (const auto &ast : spec_->outputs) lowering.emit(*ast, output_bits);
    if (static_cast<int>(input_bits.size()) != spec_->num_inputs ||
        static_cast<int>(output_bits.size()) != spec_->num_outputs) {
        throw std::runtime_error("Example length does not match declared input/output sizes");
    }
    for (const auto &[value, care] : input_bits) input_values_.push_back(value);
    for (const auto &[value, care] : output_bits) {
        output_values_.push_back(value);
        output_cares_.push_back(care);
    }

    for (const z3::expr &constraint : domain_constraints_) {
        solver_.add(constraint);
    }
}

std::vector<Example> SymbolicVerifier::find_counterexamples(const Program &program, std::size_t max_counterexamples,
                                                            std::optional<int> timeout_ms, ProfileData *profile,
                                                            bool *timed_out) {
    if (timed_out != nullptr) *timed_out = false;
    const auto start = Clock::now();
    const int num_inputs = spec_->num_inputs;
    const int num_outputs = spec_->num_outputs;
    if (static_cast<int>(program.outputs.size()) != num_outputs) {
        throw std::invalid_argument("program output count does not match the dataset");
    }

    // Wire 0 is constant one, wires 1..n are the inputs, the rest are the program's instructions.
    std::vector<z3::expr> wires;
    wires.reserve(1 + input_values_.size() + program.instrs.size());
    wires.push_back(ctx_.bool_val(true));
    for (const z3::expr &input : input_values_) wires.push_back(input);
    auto wire = [&](int source) -> const z3::expr & {
        if (source < 0 || static_cast<std::size_t>(source) >= wires.size()) {
            throw std::invalid_argument("program references an undefined source");
        }
        return wires[static_cast<std::size_t>(source)];
    };
    for (const Instruction &instr : program.instrs) {
        const z3::expr left = wire(instr.s1);
        const z3::expr right = wire(instr.s2);
        switch (instr.op) {
            case kOpAnd:
                wires.push_back(left && right);
                break;
            case kOpXor:
                wires.push_back(left != right);
                break;
            case kOpOr:
                wires.push_back(left || right);
                break;
            default:
                throw std::invalid_argument("program uses an unknown operator");
        }
    }

    z3::expr miter = ctx_.bool_val(false);
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
        const auto idx = static_cast<std::size_t>(out_idx);
        miter = miter || (output_cares_[idx] && (wire(program.outputs[idx]) != output_values_[idx]));
    }

    std::vector<Example> counterexamples;
    {
        SolverScope scope(solver_);
        if (timeout_ms.has_value()) {
            if (*timeout_ms <= 0) {
                if (timed_out != nullptr) *timed_out = true;
                return counterexamples;
            }
            solver_.set("timeout", static_cast<unsigned>(*timeout_ms));
        } else {
            solver_.set("timeout", 0U);
        }
        solver_.add(miter);

        const z3::check_result result = solver_.check();
        if (profile != nullptr) ++profile->symbolic_verifier_checks;
        if (result == z3::unsat) {
            if (profile != nullptr) {
                profile->symbolic_verifier_seconds += std::chrono::duration<double>(Clock::now() - start).count();
            }
            return counterexamples;
        }
        if (result != z3::sat) {
            if (profile != nullptr) {
                profile->symbolic_verifier_seconds += std::chrono::duration<double>(Clock::now() - start).count();
            }
            if (timeout_ms.has_value() && timed_out != nullptr) {
                *timed_out = true;
                return counterexamples;
            }
            throw std::runtime_error("symbolic verification returned unknown");
        }

        const z3::model model = solver_.get_model();
        EvalEnv env = spec_->env;
        for (const VarBinding &binding : bindings_) {
            const z3::expr value = model.eval(binding.raw, true);
            const std::uint64_t concrete = value.get_numeral_uint64();
            if (binding.kind == VarBinding::Kind::Scalar) {
                env.scalars[binding.slot] = static_cast<std::int64_t>(concrete);
            } else {
                env.tuples[binding.slot][binding.cell] = static_cast<std::int64_t>(concrete);
            }
        }

        std::optional<Example> primary = spec_->evaluate_point(env);
        if (!primary.has_value()) {
            throw std::runtime_error("symbolic counterexample violates the dataset's where clause");
        }
        if (verify_program(program, std::span<const Example>(&*primary, 1), num_inputs, num_outputs).empty()) {
            throw std::runtime_error("symbolic counterexample is not reproducible by the native evaluator");
        }

        std::set<boost::dynamic_bitset<>> seen_inputs;
        seen_inputs.insert(primary->inputs);
        counterexamples.push_back(std::move(*primary));

        const std::size_t target_count = std::max<std::size_t>(max_counterexamples, 1);
        if (counterexamples.size() < target_count) {
            auto try_neighbor = [&](EvalEnv &candidate_env) {
                if (counterexamples.size() >= target_count) return;
                std::optional<Example> neighbor = spec_->evaluate_point(candidate_env);
                if (!neighbor.has_value()) return;
                if (seen_inputs.contains(neighbor->inputs)) return;
                if (!verify_program(program, std::span<const Example>(&*neighbor, 1), num_inputs, num_outputs).empty()) {
                    seen_inputs.insert(neighbor->inputs);
                    counterexamples.push_back(std::move(*neighbor));
                }
            };

            for (const Quantifier &quantifier : spec_->quantifiers) {
                if (counterexamples.size() >= target_count) break;
                std::visit(
                    [&](const auto &q) {
                        using T = std::decay_t<decltype(q)>;
                        if constexpr (std::is_same_v<T, Quantifier::Range>) {
                            const std::int64_t orig = env.scalars[q.slot];
                            for (const std::int64_t cand : {orig - 1, orig + 1, orig ^ 1, q.start, q.end - 1}) {
                                if (cand >= q.start && cand < q.end && cand != orig) {
                                    env.scalars[q.slot] = cand;
                                    try_neighbor(env);
                                }
                            }
                            env.scalars[q.slot] = orig;
                        } else if constexpr (std::is_same_v<T, Quantifier::Values>) {
                            const std::int64_t orig = env.scalars[q.slot];
                            for (const std::int64_t cand : q.values) {
                                if (cand != orig) {
                                    env.scalars[q.slot] = cand;
                                    try_neighbor(env);
                                }
                            }
                            env.scalars[q.slot] = orig;
                        } else {
                            for (std::size_t cell = 0; cell < q.size && counterexamples.size() < target_count; ++cell) {
                                const std::int64_t orig = env.tuples[q.tuple_slot][cell];
                                for (const std::int64_t cand : {orig - 1, orig + 1, q.start, q.end - 1}) {
                                    if (cand >= q.start && cand < q.end && cand != orig) {
                                        env.tuples[q.tuple_slot][cell] = cand;
                                        try_neighbor(env);
                                    }
                                }
                                env.tuples[q.tuple_slot][cell] = orig;
                            }
                        }
                    },
                    quantifier.spec);
            }
        }
    }

    if (profile != nullptr) {
        profile->symbolic_verifier_seconds += std::chrono::duration<double>(Clock::now() - start).count();
        profile->symbolic_counterexamples += counterexamples.size();
    }
    return counterexamples;
}

VerificationOracle::VerificationOracle(std::span<const Example> initial_examples, int num_inputs, int num_outputs,
                                       SymbolicVerifier *verifier, ProfileData *profile)
    : examples_(initial_examples.begin(), initial_examples.end()),
      packed_(pack_examples(examples_, num_inputs, num_outputs)),
      num_inputs_(num_inputs),
      num_outputs_(num_outputs),
      verifier_(verifier),
      profile_(profile) {}

void VerificationOracle::mark_verified(const std::string &key) {
    if (!key.empty()) verified_keys_.insert(key);
}

void VerificationOracle::add_examples(std::span<const Example> learned) {
    if (learned.empty()) return;
    examples_.reserve(examples_.size() + learned.size());
    for (const Example &ex : learned) {
        examples_.push_back(ex);
        packed_.append_example(ex);
    }
    ++basis_generation_;
}

bool VerificationOracle::verify_candidate(const Program &candidate, const std::string &canonical_key,
                                          std::optional<int> timeout_ms, std::size_t max_counterexamples) {
    if (!verify_program_packed_ok(candidate, packed_, num_outputs_)) {
        return false;
    }
    if (verifier_ == nullptr) {
        return true;
    }
    if (!canonical_key.empty() && verified_keys_.contains(canonical_key)) {
        return true;
    }
    bool timed_out = false;
    const std::vector<Example> counterexamples =
        verifier_->find_counterexamples(candidate, max_counterexamples, timeout_ms, profile_, &timed_out);
    if (timed_out) {
        return false;
    }
    if (counterexamples.empty()) {
        if (!canonical_key.empty()) verified_keys_.insert(canonical_key);
        return true;
    }
    add_examples(counterexamples);
    return false;
}
