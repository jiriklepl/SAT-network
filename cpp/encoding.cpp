#include "encoding.hpp"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>

using boost::multiprecision::cpp_int;

namespace {

z3::expr bv_const(z3::context &ctx, const std::string &name, unsigned bits) {
    return ctx.bv_const(name.c_str(), bits);
}

z3::expr bv_val(z3::context &ctx, const cpp_int &value, unsigned bits) {
    return ctx.bv_val(cpp_int_to_string(value).c_str(), bits);
}

z3::expr bv_val(z3::context &ctx, uint64_t value, unsigned bits) {
    return ctx.bv_val(value, bits);
}

z3::expr op_code_expr(z3::context &ctx, int code) {
    return bv_val(ctx, static_cast<uint64_t>(code), 2);
}

z3::expr apply_operator(int code, const z3::expr &left, const z3::expr &right) {
    if (code == 0) return left & right;
    if (code == 1) return left ^ right;
    if (code == 2) return left | right;
    throw std::runtime_error("unknown operator code");
}

z3::expr operator_constraint(z3::context &ctx, const z3::expr &op) {
    z3::expr_vector disj(ctx);
    for (const auto &logic_op : logic_operators()) {
        disj.push_back(op == op_code_expr(ctx, logic_op.code));
    }
    return z3::mk_or(disj);
}

z3::expr operator_expr(z3::context &ctx, const z3::expr &op, const z3::expr &left, const z3::expr &right) {
    const auto &operators = logic_operators();
    z3::expr expr = apply_operator(operators.back().code, left, right);
    for (auto it = operators.rbegin() + 1; it != operators.rend(); ++it) {
        expr = z3::ite(op == op_code_expr(ctx, it->code), apply_operator(it->code, left, right), expr);
    }
    return expr;
}

z3::expr operator_rank_expr(z3::context &ctx, const z3::expr &op) {
    const auto &operators = logic_operators();
    z3::expr expr = bv_val(ctx, static_cast<uint64_t>(operators.back().canonical_rank), 2);
    for (auto it = operators.rbegin() + 1; it != operators.rend(); ++it) {
        expr = z3::ite(
            op == op_code_expr(ctx, it->code),
            bv_val(ctx, static_cast<uint64_t>(it->canonical_rank), 2),
            expr);
    }
    return expr;
}

z3::expr select_bv(
    z3::context &ctx,
    const std::vector<z3::expr> &values,
    const z3::expr &idx_var,
    unsigned bits,
    bool balanced
) {
    if (values.empty()) {
        throw std::runtime_error("select values must be non-empty");
    }
    if (!balanced) {
        z3::expr result = values[0];
        for (std::size_t idx = 1; idx < values.size(); ++idx) {
            result = z3::ite(idx_var == bv_val(ctx, static_cast<uint64_t>(idx), bits), values[idx], result);
        }
        return result;
    }

    std::function<z3::expr(std::size_t, std::size_t, const z3::expr &)> build =
        [&](std::size_t lo, std::size_t hi, const z3::expr &default_expr) -> z3::expr {
            if (lo >= hi) return default_expr;
            if (hi - lo == 1) {
                return z3::ite(idx_var == bv_val(ctx, static_cast<uint64_t>(lo), bits), values[lo], default_expr);
            }
            std::size_t mid = (lo + hi) / 2;
            z3::expr right = build(mid, hi, default_expr);
            return build(lo, mid, right);
        };
    return build(1, values.size(), values[0]);
}

std::pair<std::vector<z3::expr>, std::vector<z3::expr>> build_test(
    z3::context &ctx,
    const PackedExamples &packed,
    const std::string &tag,
    const ProgramSpec &spec,
    const EncodingOptions &options
) {
    std::vector<z3::expr> constraints;
    std::vector<z3::expr> values;
    values.push_back(bv_val(ctx, all_ones(packed.width), packed.width));
    for (const auto &input_mask : packed.input_masks) {
        values.push_back(bv_val(ctx, input_mask, packed.width));
    }

    for (int instr = 0; instr < spec.program_length; ++instr) {
        const z3::expr op = bv_const(ctx, bv_name("OP", instr), 2);
        const z3::expr src1 = bv_const(ctx, bv_name("S1", instr), spec.idx_bits());
        const z3::expr src2 = bv_const(ctx, bv_name("S2", instr), spec.idx_bits());
        const z3::expr val = bv_const(ctx, "VAL_" + tag + "_" + std::to_string(instr), packed.width);

        z3::expr left = values[0];
        z3::expr right = values[0];
        if (options.encode_boolean) {
            left = bv_const(ctx, "LEFT_" + tag + "_" + std::to_string(instr), packed.width);
            right = bv_const(ctx, "RIGHT_" + tag + "_" + std::to_string(instr), packed.width);
            for (std::size_t source_idx = 0; source_idx < values.size(); ++source_idx) {
                z3::expr s1_bool = ctx.bool_const(("S1_" + std::to_string(instr) + "_eq_" + std::to_string(source_idx)).c_str());
                z3::expr s2_bool = ctx.bool_const(("S2_" + std::to_string(instr) + "_eq_" + std::to_string(source_idx)).c_str());
                constraints.push_back(z3::implies(s1_bool, left == values[source_idx]));
                constraints.push_back(z3::implies(s2_bool, right == values[source_idx]));
            }
        } else {
            left = select_bv(ctx, values, src1, spec.idx_bits(), options.balanced_select);
            right = select_bv(ctx, values, src2, spec.idx_bits(), options.balanced_select);
        }

        constraints.push_back(val == operator_expr(ctx, op, left, right));
        values.push_back(val);
    }

    std::vector<z3::expr> outputs;
    for (int out_idx = 0; out_idx < spec.num_outputs; ++out_idx) {
        const z3::expr selector = bv_const(ctx, "OUT_" + std::to_string(out_idx) + "_idx", spec.idx_bits());
        if (options.encode_boolean) {
            const z3::expr out_expr = bv_const(ctx, "OUTVAL_" + tag + "_" + std::to_string(out_idx), packed.width);
            for (std::size_t source_idx = 0; source_idx < values.size(); ++source_idx) {
                const z3::expr selector_bool = ctx.bool_const(("OUT_" + std::to_string(out_idx) + "_eq_" + std::to_string(source_idx)).c_str());
                constraints.push_back(z3::implies(selector_bool, out_expr == values[source_idx]));
            }
            outputs.push_back(out_expr);
        } else {
            outputs.push_back(select_bv(ctx, values, selector, spec.idx_bits(), options.balanced_select));
        }
    }
    return {constraints, outputs};
}

uint64_t model_bv_uint64(const z3::model &model, const z3::expr &expr, const std::string &name) {
    z3::expr value = model.eval(expr, true);
    uint64_t result = 0;
    if (!Z3_get_numeral_uint64(value.ctx(), value, &result)) {
        throw std::runtime_error("Model did not provide a concrete value for " + name + ": " + value.to_string());
    }
    return result;
}

}  // namespace

std::vector<z3::expr> build_program(z3::context &ctx, const ProgramSpec &spec, const EncodingOptions &options) {
    if (spec.num_inputs <= 0 || spec.num_outputs <= 0 || spec.program_length < 0) {
        throw std::runtime_error("invalid program spec");
    }
    std::vector<z3::expr> constraints;
    std::vector<z3::expr> ops;
    std::vector<z3::expr> src1s;
    std::vector<z3::expr> src2s;
    std::vector<z3::expr> output_selectors;
    ops.reserve(spec.program_length);
    src1s.reserve(spec.program_length);
    src2s.reserve(spec.program_length);
    for (int instr = 0; instr < spec.program_length; ++instr) {
        ops.push_back(bv_const(ctx, bv_name("OP", instr), 2));
        src1s.push_back(bv_const(ctx, bv_name("S1", instr), spec.idx_bits()));
        src2s.push_back(bv_const(ctx, bv_name("S2", instr), spec.idx_bits()));
    }
    for (int out_idx = 0; out_idx < spec.num_outputs; ++out_idx) {
        output_selectors.push_back(bv_const(ctx, "OUT_" + std::to_string(out_idx) + "_idx", spec.idx_bits()));
    }

    for (int instr = 0; instr < spec.program_length; ++instr) {
        const int idx = spec.num_inputs + 1 + instr;
        const int max_idx = idx - 1;
        const z3::expr op = ops[instr];
        const z3::expr src1 = src1s[instr];
        const z3::expr src2 = src2s[instr];

        if (options.force_ordered && instr > 0) {
            const z3::expr pre_src1 = src1s[instr - 1];
            const z3::expr pre_src2 = src2s[instr - 1];
            const z3::expr pre_rank = operator_rank_expr(ctx, ops[instr - 1]);
            const z3::expr rank = operator_rank_expr(ctx, op);
            constraints.push_back(z3::ule(pre_src2, src2));
            constraints.push_back(z3::implies(pre_src2 == src2, z3::ule(pre_src1, src1)));
            constraints.push_back(z3::implies((pre_src2 == src2) && (pre_src1 == src1), z3::ult(pre_rank, rank)));
        }

        if (options.force_useful) {
            z3::expr_vector useful(ctx);
            z3::expr idx_bv = bv_val(ctx, static_cast<uint64_t>(idx), spec.idx_bits());
            for (const auto &selector : output_selectors) useful.push_back(selector == idx_bv);
            for (int next = instr + 1; next < spec.program_length; ++next) {
                useful.push_back(src1s[next] == idx_bv);
                useful.push_back(src2s[next] == idx_bv);
            }
            constraints.push_back(z3::mk_or(useful));
        }

        if (options.encode_boolean) {
            for (int source_idx = 0; source_idx <= max_idx; ++source_idx) {
                const z3::expr s1_bool = ctx.bool_const(("S1_" + std::to_string(instr) + "_eq_" + std::to_string(source_idx)).c_str());
                const z3::expr s2_bool = ctx.bool_const(("S2_" + std::to_string(instr) + "_eq_" + std::to_string(source_idx)).c_str());
                const z3::expr source_bv = bv_val(ctx, static_cast<uint64_t>(source_idx), spec.idx_bits());
                constraints.push_back(s1_bool == (src1 == source_bv));
                constraints.push_back(s2_bool == (src2 == source_bv));
            }
        }

        constraints.push_back(operator_constraint(ctx, op));
        constraints.push_back(z3::ule(src1, bv_val(ctx, static_cast<uint64_t>(max_idx), spec.idx_bits())));
        constraints.push_back(z3::ule(src2, bv_val(ctx, static_cast<uint64_t>(max_idx), spec.idx_bits())));
        constraints.push_back(z3::ult(src1, src2) || ((op == op_code_expr(ctx, 1)) && (src1 == src2)));
    }

    z3::expr max_total_idx = bv_val(ctx, static_cast<uint64_t>(spec.total_sources() - 1), spec.idx_bits());
    for (int out_idx = 0; out_idx < spec.num_outputs; ++out_idx) {
        z3::expr selector = output_selectors[out_idx];
        constraints.push_back(z3::ule(selector, max_total_idx));
        if (options.encode_boolean) {
            for (int source_idx = 0; source_idx < spec.total_sources(); ++source_idx) {
                z3::expr selector_bool = ctx.bool_const(("OUT_" + std::to_string(out_idx) + "_eq_" + std::to_string(source_idx)).c_str());
                constraints.push_back(selector_bool == (selector == bv_val(ctx, static_cast<uint64_t>(source_idx), spec.idx_bits())));
            }
        }
    }
    return constraints;
}

std::vector<z3::expr> build_assumption_constraints(z3::context &ctx, const ProgramSpec &spec, const Assumptions &assumptions) {
    std::vector<z3::expr> constraints;
    for (const auto &instr : assumptions.instructions) {
        constraints.push_back(bv_const(ctx, bv_name("OP", instr.instr_idx), 2) == bv_val(ctx, static_cast<uint64_t>(instr.op), 2));
        constraints.push_back(
            bv_const(ctx, bv_name("S1", instr.instr_idx), spec.idx_bits()) ==
            bv_val(ctx, static_cast<uint64_t>(instr.s1), spec.idx_bits()));
        constraints.push_back(
            bv_const(ctx, bv_name("S2", instr.instr_idx), spec.idx_bits()) ==
            bv_val(ctx, static_cast<uint64_t>(instr.s2), spec.idx_bits()));
    }
    for (const auto &output : assumptions.outputs) {
        constraints.push_back(
            bv_const(ctx, "OUT_" + std::to_string(output.out_idx) + "_idx", spec.idx_bits()) ==
            bv_val(ctx, static_cast<uint64_t>(output.source), spec.idx_bits()));
    }
    return constraints;
}

void add_exprs(z3::solver &solver, const std::vector<z3::expr> &exprs) {
    for (const auto &expr : exprs) solver.add(expr);
}

void add_example_constraints(
    z3::context &ctx,
    z3::solver &solver,
    const std::vector<Example> &examples,
    const std::string &tag,
    const ProgramSpec &spec,
    const EncodingOptions &options
) {
    PackedExamples packed = pack_examples(examples, spec.num_inputs, spec.num_outputs);
    auto [constraints, outputs] = build_test(ctx, packed, tag, spec, options);
    add_exprs(solver, constraints);
    for (int out_idx = 0; out_idx < spec.num_outputs; ++out_idx) {
        z3::expr expected = bv_val(ctx, packed.output_values[out_idx], packed.width);
        if (packed.output_dont_care_masks[out_idx] != 0) {
            z3::expr dont_care = bv_val(ctx, packed.output_dont_care_masks[out_idx], packed.width);
            solver.add((outputs[out_idx] | dont_care) == (expected | dont_care));
        } else {
            solver.add(outputs[out_idx] == expected);
        }
    }
}

Program extract_program(z3::context &ctx, const z3::model &model, const ProgramSpec &spec) {
    Program program;
    for (int instr = 0; instr < spec.program_length; ++instr) {
        int op = static_cast<int>(model_bv_uint64(model, bv_const(ctx, bv_name("OP", instr), 2), "OP_" + std::to_string(instr)));
        int s1 = static_cast<int>(model_bv_uint64(model, bv_const(ctx, bv_name("S1", instr), spec.idx_bits()), "S1_" + std::to_string(instr)));
        int s2 = static_cast<int>(model_bv_uint64(model, bv_const(ctx, bv_name("S2", instr), spec.idx_bits()), "S2_" + std::to_string(instr)));
        program.instrs.push_back({known_op(op) ? op : -1, s1, s2});
    }
    for (int out_idx = 0; out_idx < spec.num_outputs; ++out_idx) {
        program.outputs.push_back(static_cast<int>(model_bv_uint64(
            model,
            bv_const(ctx, "OUT_" + std::to_string(out_idx) + "_idx", spec.idx_bits()),
            "OUT_" + std::to_string(out_idx) + "_idx")));
    }
    return program;
}

z3::solver make_solver(z3::context &ctx, const std::string &solver_choice) {
    if (solver_choice == "z3") {
        return z3::solver(ctx, "QF_BV");
    }
    if (solver_choice == "simple-tactic") {
        z3::tactic tactic = z3::tactic(ctx, "simplify") & z3::tactic(ctx, "propagate-values") &
                            z3::tactic(ctx, "bit-blast") & z3::tactic(ctx, "sat");
        return tactic.mk_solver();
    }
    if (solver_choice == "ctx-simplify-tactic") {
        z3::tactic tactic = z3::tactic(ctx, "ctx-simplify") & z3::tactic(ctx, "propagate-values") &
                            z3::tactic(ctx, "bit-blast") & z3::tactic(ctx, "sat");
        return tactic.mk_solver();
    }
    throw std::runtime_error("Unsupported solver: " + solver_choice);
}
