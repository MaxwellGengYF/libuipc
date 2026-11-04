#pragma once
/********************************************************************
 * @file contact_models/codim_ipc_contact_function.h
 * @brief T8.1b — the IPC contact coefficient / C1-clamped friction functions shared by
 * every codim IPC contact model (port of
 * src/backends/cuda/contact_system/contact_models/codim_ipc_contact_function.h, 128
 * reference lines).
 *
 * It is the "device math" prologue of the four IPC models: it pulls in the generated
 * barrier law (`sym/codim_ipc_contact.inl`, *inside* the namespace, exactly like the
 * reference) and adds the reference's own hand-written bodies
 *  * the C1 clamping of the friction velocity norm (`f0` / `f1_div_rel_dx_norm` /
 *    `f2_term`),
 *  * `friction_energy` / `friction_gradient` / `friction_hessian` (the 2x2 tangential
 *    friction law, `mu * lambda * f0(...)`),
 *  * the local `make_spd(LMat<2,2>&)` (eigen-clamp-reconstruct) and
 *  * `normal_force` (`-dB/dD * 2 * sqrt(D)`, the IPC normal-force magnitude).
 * The four `PT_/EE_/PE_/PP_` kappa/barrier helpers live in
 * `codim_ipc_simplex_{normal,frictional}_contact_function.h` and build on this file.
 *
 * THE ONE STRUCTURAL DEVIATION (cookbook §7.3 / §7.36; the established port convention —
 * see `luisa_device/math/distance/flagged.h`'s header): five reference bodies branch on
 * *device* values — the C1 clamp's `if(x2 >= epsvh * epsvh)` and the friction Hessian's
 * three-way `if(sq_norm >= epsvh2) / if(sq_norm == 0.0) / else`.  LC cannot take host
 * control flow on a device predicate (RULE M2), so the port evaluates every branch into a
 * local and *selects* (`distance::detail::select_scalar(f, t, pred)` == `pred ? t : f`,
 * plus the local `select_mat22` for the 2x2 Hessian).  A select passes its chosen operand
 * through unchanged, so each branch's value is bit-identical to the reference's taken
 * branch and the discarded candidates are only extra work — including any `inf`/`NaN`
 * they may produce (the reference's not-taken branch could not have written anything
 * either).  The `2.0`/`3.0` literals are the reference's (`int` literals are spelled as
 * the `double` literals the numerics layer's DSL instantiations need).
 *********************************************************************/
#include <type_define.h>
#include <luisa_device/math/distance/distance.h>
#include <luisa_device/math/matrix_ops.h>
#include <utils/friction_utils.h>

namespace uipc::backend::luisa
{
namespace sym::codim_ipc_contact
{
#include "sym/codim_ipc_contact.inl"

    namespace ldist = luisa_device::math::distance;
    using ldist::detail::select_scalar;
    using lps::LMat;
    using lps::LVec;

    /// The reference's 2x2 element-wise select (`distance::detail::select_vec` is
    /// vector-only, and `LMat` is not `vec_like`).
    template <typename T, typename P>
    [[nodiscard]] inline LMat<T, 2, 2> select_mat22(const LMat<T, 2, 2>& f,
                                                   const LMat<T, 2, 2>& t,
                                                   const P&             pred) noexcept
    {
        return lps::make_mat<T, 2, 2>(
            [&](int r, int c) noexcept { return select_scalar(f(r, c), t(r, c), pred); });
    }

    // C1 clamping
    template <typename T>
    inline void f0(const T& x2, const T& epsvh, T& f0)
    {
        namespace d = lps::detail;
        // reference: if(x2 >= epsvh * epsvh) f0 = sqrt(x2);
        //            else f0 = x2 * (-sqrt(x2) / 3 + epsvh) / (epsvh * epsvh) + epsvh / 3;
        T ge = lps::sqrt(x2);
        T lt = d::add(d::div(d::mul(x2,
                                    d::add(d::neg(d::div(lps::sqrt(x2), 3.0)), epsvh)),
                             d::mul(epsvh, epsvh)),
                      d::div(epsvh, 3.0));
        f0 = select_scalar(lt, ge, x2 >= d::mul(epsvh, epsvh));
    }

    template <typename T>
    inline void f1_div_rel_dx_norm(const T& x2, const T& epsvh, T& result)
    {
        namespace d = lps::detail;
        // reference: if(x2 >= epsvh * epsvh) result = 1 / sqrt(x2);
        //            else result = (-sqrt(x2) + 2 * epsvh) / (epsvh * epsvh);
        T ge = d::div(1.0, lps::sqrt(x2));
        T lt = d::div(d::add(d::neg(lps::sqrt(x2)), d::mul(2.0, epsvh)),
                      d::mul(epsvh, epsvh));
        result = select_scalar(lt, ge, x2 >= d::mul(epsvh, epsvh));
    }

    template <typename T>
    inline void f2_term(const T& x2, const T& epsvh, T& term)
    {
        // reference: term = -1 / (epsvh * epsvh); (same for x2 >= epsvh * epsvh)
        (void)x2;
        term = lps::detail::neg(lps::detail::div(1.0, lps::detail::mul(epsvh, epsvh)));
    }

    template <typename T>
    [[nodiscard]] inline T friction_energy(const T&          mu,
                                           const T&          lambda,
                                           const T&          eps_vh,
                                           const LVec<T, 2>& tan_rel_x)
    {
        namespace d = lps::detail;
        T f0_val;
        f0(tan_rel_x.squaredNorm(), eps_vh, f0_val);
        return d::mul(d::mul(mu, lambda), f0_val);
    }

    template <typename T>
    inline void friction_gradient(LVec<T, 2>&       G2,
                                  const T&          mu,
                                  const T&          lambda,
                                  const T&          eps_vh,
                                  const LVec<T, 2>& tan_rel_x)
    {
        namespace d = lps::detail;
        T f1_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_val);
        G2 = d::mul(d::mul(mu, lambda), f1_val) * tan_rel_x;
    }

    template <typename T>
    inline void make_spd(LMat<T, 2, 2>& mat)
    {
        namespace d = lps::detail;
        LVec<T, 2>    eigen_values;
        LMat<T, 2, 2> eigen_vectors;
        lps::evd(mat, eigen_values, eigen_vectors);
        for(int i = 0; i < 2; ++i)
        {
            auto& v = eigen_values(i);
            v       = select_scalar(v, lps::zero_of<T>(), v < 0.0);
        }
        mat = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
    }

    template <typename T>
    inline void friction_hessian(LMat<T, 2, 2>&   H2x2,
                                 const T&         mu,
                                 const T&         lambda,
                                 const T&         eps_vh,
                                 const LVec<T, 2>& tan_rel_x)
    {
        namespace d = lps::detail;
        auto  sq_norm = tan_rel_x.squaredNorm();
        auto  epsvh2  = d::mul(eps_vh, eps_vh);
        T     f1_div_rel_dx_norm_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_div_rel_dx_norm_val);

        // reference: if(sq_norm >= epsvh2) {
        //                Vector2 ubar(-tan_rel_x[1], tan_rel_x[0]);
        //                H2x2 = ubar * (mu * lambda * f1 / sq_norm) * ubar.transpose(); }
        LVec<T, 2> ubar{d::neg(tan_rel_x(1)), tan_rel_x(0)};
        auto       H_ge = ubar * d::div(d::mul(d::mul(mu, lambda), f1_div_rel_dx_norm_val), sq_norm)
                    * ubar.transpose();

        // reference: else if(sq_norm == 0.0) {
        //                H2x2 = (mu * lambda * f1) * Matrix2x2::Identity(); }
        auto H_eq0 = d::mul(d::mul(mu, lambda), f1_div_rel_dx_norm_val)
                     * LMat<T, 2, 2>::Identity();

        // reference: else { ... H2x2 = tan_rel_x * (f2 / relDXNorm) * tan_rel_x.transpose();
        //                   H2x2.diagonal().array() += f1; make_spd(H2x2); H2x2 *= mu * lambda; }
        T f2_term_val;
        f2_term(sq_norm, eps_vh, f2_term_val);
        auto relDXNorm = lps::sqrt(sq_norm);
        auto H_lt      = tan_rel_x * d::div(f2_term_val, relDXNorm) * tan_rel_x.transpose();
        for(int i = 0; i < 2; ++i)
            H_lt(i, i) = d::add(H_lt(i, i), f1_div_rel_dx_norm_val);
        make_spd(H_lt);
        H_lt = H_lt * d::mul(mu, lambda);
        // The reference's chain is
        //   if (sq_norm >= epsvh2)  H2x2 = H_ge;
        //   else if (sq_norm == 0)  H2x2 = H_eq0;
        //   else                    H2x2 = H_lt;
        // and `select_scalar(f, t, p)` is `p ? t : f`, so the selects nest f-outer /
        // t-inner-from-the-innermost-branch.  The earlier spelling had them swapped: it returned
        // `H_lt` for `sq_norm == 0`, and `H_lt` divides by `sqrt(sq_norm)`, so the *whole*
        // friction Hessian was NaN for a pair with no tangential motion (measured: `nan = 144` on
        // the assembled 12x12 PT block, and the host instantiation of the same body was NaN as
        // well -- which is how the test pinned it).
        H2x2 = select_mat22(select_mat22(H_lt, H_eq0, sq_norm == 0.0),
                            H_ge,
                            sq_norm >= epsvh2);
    }

    template <typename T>
    [[nodiscard]] inline T normal_force(const T& kappa,
                                        const T& d_hat,
                                        const T& thickness,
                                        const T& D)
    {
        namespace d = lps::detail;
        auto  d_      = lps::sqrt(D);
        T     dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        auto dBdd = d::mul(d::mul(dBdD, 2.0), d_);
        return d::neg(dBdd);  // > 0
    }
}  // namespace sym::codim_ipc_contact
}  // namespace uipc::backend::luisa
